"""Generation, training, and recovery for rectangular sampler modules."""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from gammaboard_process import Domain, RectangularDomain, Feedback, SampleBatch, GenerationStatus
from glnis.core.nn.execution import SamplerExecution
from glnis.core.nn.losses import Loss
from glnis.core.nn.sampler_module import RectangularSamplerModule
from glnis.core.nn.schedulers import Scheduler
from glnis.utils.types import BuildFromKind, Config, RectangularTrainingStatus


@dataclass
class OrchestratorConfig(Config):
    save_path: str | None = None
    module_kwargs: dict = field(default_factory=dict)


@dataclass
class OrchestratorSnapshot:
    kind: str
    save_path: str
    cfg_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.cfg_kwargs = deepcopy(self.cfg_kwargs)
        if "sampler_kwargs" in self.cfg_kwargs:
            legacy = self.cfg_kwargs.pop("sampler_kwargs")
            if "module_kwargs" in self.cfg_kwargs and self.cfg_kwargs["module_kwargs"] != legacy:
                raise ValueError("conflicting module_kwargs in checkpoint")
            self.cfg_kwargs["module_kwargs"] = legacy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RectangularOrchestratorConfig(OrchestratorConfig):
    seed: int | None = 42
    training_steps: int = 100
    training_batch_size: int = 1000
    max_batch_size: int = 100_000
    use_gpu: bool = True
    use_f64: bool = True
    cuda_id: int = 1
    learning_rate: float = 1e-3
    loss: dict = field(default_factory=lambda: {"kind": "kl_divergence"})
    scheduler: dict | None = field(default_factory=lambda: {"kind": "cosineannealing"})
    execution: dict = field(default_factory=lambda: {"kind": "eager"})

    @classmethod
    def from_kwargs(cls, **kwargs):
        return cls(**cls.normalize_kwargs(**kwargs))

    @staticmethod
    def normalize_kwargs(**kwargs):
        """Normalize a partial table without filling in unrelated defaults."""
        values = deepcopy(kwargs)
        # Accept existing run files while writing only the table-based format.
        if "loss_type" in values or "loss_kwargs" in values:
            legacy = {"kind": values.pop("loss_type", "kl_divergence"), **values.pop("loss_kwargs", {})}
            if "loss" in values and values["loss"] != legacy:
                raise ValueError("configure either loss or loss_type/loss_kwargs")
            values["loss"] = legacy
        if "use_scheduler" in values or "scheduler_type" in values:
            enabled = values.pop("use_scheduler", True)
            kind = values.pop("scheduler_type", "cosineannealing")
            legacy = {"kind": kind} if enabled and kind is not None else None
            if "scheduler" in values and values["scheduler"] != legacy:
                raise ValueError("configure either scheduler or use_scheduler/scheduler_type")
            values["scheduler"] = legacy
        return values

    def __post_init__(self):
        for name in ("training_steps", "training_batch_size", "max_batch_size", "cuda_id"):
            value = getattr(self, name)
            minimum = 0 if name in ("training_steps", "cuda_id") else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"orchestrator.{name} must be an integer >= {minimum}")
        for name in ("use_gpu", "use_f64"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"orchestrator.{name} must be bool")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("orchestrator.seed must be a nonnegative integer or None")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("orchestrator.learning_rate must be positive and finite")
        if not isinstance(self.module_kwargs, dict) or not isinstance(self.loss, dict) or not isinstance(self.execution, dict):
            raise ValueError("module_kwargs, loss and execution must be configuration tables")
        if self.scheduler is not None and not isinstance(self.scheduler, dict):
            raise ValueError("scheduler must be a configuration table or None")
        self.module_kwargs = deepcopy(self.module_kwargs)
        self.loss = deepcopy(self.loss)
        self.scheduler = deepcopy(self.scheduler)
        self.execution = deepcopy(self.execution)
        SamplerExecution.subclass_for_kind(self.execution.get("kind", SamplerExecution.DEFAULT_KIND))


@dataclass
class RectangularOrchestratorSnapshotData:
    training_status: RectangularTrainingStatus
    module_snapshot: bytes
    optimizer_state: dict | None
    scheduler_state: dict | None = None
    torch_cpu_rng_state: Tensor | None = None
    torch_gpu_rng_state: Tensor | None = None


class Orchestrator(ABC, BuildFromKind):
    KIND = "ABCOrchestrator"
    DEFAULT_KIND = "rectangular"

    @abstractmethod
    def generate(self, max_samples: int) -> SampleBatch | GenerationStatus:
        pass

    @abstractmethod
    def feedback(self, feedback: Feedback) -> None:
        pass

    @abstractmethod
    def snapshot(self) -> OrchestratorSnapshot:
        pass

    @classmethod
    def from_snapshot(cls, *, snapshot: Any, domain: Domain, **kwargs):
        snapshot = OrchestratorSnapshot(**snapshot) if isinstance(snapshot, dict) else snapshot
        return cls.subclass_for_kind(snapshot.kind)._from_snapshot(domain, snapshot, **kwargs)

    @classmethod
    @abstractmethod
    def _from_snapshot(cls, domain: Domain, snapshot: Any, **kwargs):
        pass

    @classmethod
    def from_stage(cls, *, snapshot: Any, domain: Domain, **kwargs):
        snapshot = OrchestratorSnapshot(**snapshot) if isinstance(snapshot, dict) else snapshot
        return cls.subclass_for_kind(snapshot.kind)._from_stage(domain, snapshot, **kwargs)

    @classmethod
    def _from_stage(cls, domain: Domain, snapshot: Any, **kwargs):
        raise NotImplementedError(f"{cls.__name__} does not support stage handoff")

    def pdf(self, xs_discrete: NDArray, xs_continuous: NDArray) -> NDArray | None:
        return None

    def discrete_mass(self, subspace: NDArray) -> float | None:
        return None

    def diagnostics(self):
        return {}

    def training_samples_remaining(self):
        return None


class RectangularOrchestrator(Orchestrator):
    KIND = "rectangular"
    DEFAULT_KIND = "rectangular"
    RESTORE_OVERRIDES = {"save_path", "use_gpu", "use_f64", "cuda_id", "max_batch_size", "execution"}

    def __init__(self, domain: RectangularDomain, dims_c: int = 0,
                 cfg: RectangularOrchestratorConfig | None = None,
                 snapshot: OrchestratorSnapshot | None = None,
                 fresh_training: bool = False, **kwargs):
        domain.validate()
        self.domain = domain
        self.num_discrete_dims = len(domain.discrete_cardinalities)
        self.dims_c = dims_c
        self.cfg = cfg if cfg is not None else RectangularOrchestratorConfig.from_kwargs(**kwargs)
        self.device = self._get_device()
        self.dtype = torch.float64 if self.cfg.use_f64 else torch.float32
        self.loss_function = Loss.from_kwargs(**self.cfg.loss)
        snapshot_data = self._read_snapshot(snapshot) if snapshot is not None else None

        if snapshot_data is None:
            if self.cfg.seed is not None:
                torch.manual_seed(self.cfg.seed)
            self.module = RectangularSamplerModule.from_kwargs(
                domain=domain, dims_c=dims_c, **self.cfg.module_kwargs)
        else:
            self.module = RectangularSamplerModule.from_snapshot(snapshot_data.module_snapshot)
            if self.module.domain.require_rectangular_shape() != domain.require_rectangular_shape():
                raise ValueError("checkpoint module domain does not match requested domain")
            if self.module.dims_c != dims_c:
                raise ValueError("checkpoint conditioning dimensions do not match requested dimensions")
        self.module.to(device=self.device, dtype=self.dtype)

        if snapshot_data is not None and not fresh_training:
            self.training_status = snapshot_data.training_status
            if self.training_status.training_batch_size != self.cfg.training_batch_size:
                raise ValueError("checkpoint training window does not match training_batch_size")
        else:
            self.training_status = RectangularTrainingStatus(
                self.domain.continuous_dims + self.num_discrete_dims, self.cfg.training_batch_size)
            if fresh_training and self.cfg.seed is not None:
                torch.manual_seed(self.cfg.seed)
        self.training_status.to(self.device, self.dtype)

        parameters = [p for p in self.module.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(parameters, lr=self.cfg.learning_rate) if parameters else None
        if self.cfg.training_steps and self.optimizer is None:
            raise ValueError("training requires a sampler module with trainable parameters")
        self.scheduler = (Scheduler.from_kwargs(optimizer=self.optimizer, training_steps=self.cfg.training_steps,
                                               **self.cfg.scheduler)
                          if self.cfg.scheduler is not None and self.optimizer is not None else None)
        self.module.train(self.training_status.step < self.cfg.training_steps)
        self.execution = SamplerExecution.from_kwargs(module=self.module, **self.cfg.execution)
        if snapshot_data is not None and not fresh_training:
            if snapshot_data.optimizer_state is not None and self.optimizer is not None:
                self.optimizer.load_state_dict(snapshot_data.optimizer_state)
            if snapshot_data.scheduler_state is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(snapshot_data.scheduler_state)
            # Restore last, after all reconstruction that might consume random numbers.
            if snapshot_data.torch_cpu_rng_state is not None:
                torch.set_rng_state(snapshot_data.torch_cpu_rng_state.cpu())
            if snapshot_data.torch_gpu_rng_state is not None and self.device.type == "cuda":
                torch.cuda.set_rng_state(snapshot_data.torch_gpu_rng_state.cpu(), self.device)

    def _read_snapshot(self, snapshot):
        source_path = Path(snapshot.save_path)
        with source_path.open("rb") as stream:
            # Let module/buffer placement and optimizer loading move their own
            # tensors, preserving Adam's CPU step counters.
            data = torch.load(stream, weights_only=False, map_location="cpu")
        if not isinstance(data, RectangularOrchestratorSnapshotData):
            raise ValueError("snapshot data must be RectangularOrchestratorSnapshotData")
        return data

    def snapshot(self) -> OrchestratorSnapshot:
        if self.cfg.save_path is None:
            raise ValueError("save_path must be specified to save a snapshot")
        destination = Path(self.cfg.save_path)
        temporary = destination.with_name(destination.name + ".tmp")
        data = RectangularOrchestratorSnapshotData(
            training_status=self.training_status, module_snapshot=self.module.snapshot(),
            optimizer_state=self.optimizer.state_dict() if self.optimizer is not None else None,
            scheduler_state=self.scheduler.state_dict() if self.scheduler is not None else None,
            torch_cpu_rng_state=torch.get_rng_state(),
            torch_gpu_rng_state=torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("wb") as stream:
                torch.save(data, stream)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return OrchestratorSnapshot(kind=self.KIND, save_path=str(destination), cfg_kwargs=self.cfg.to_dict())

    @classmethod
    def _from_snapshot(cls, domain, snapshot, cfg=None, **kwargs):
        if snapshot.kind != cls.KIND:
            raise ValueError("checkpoint orchestrator kind does not match requested kind")
        saved = RectangularOrchestratorConfig.from_kwargs(**snapshot.cfg_kwargs)
        effective = cfg or RectangularOrchestratorConfig.from_kwargs(**{**saved.to_dict(), **kwargs})
        for name, value in saved.to_dict().items():
            if name not in cls.RESTORE_OVERRIDES and effective.to_dict()[name] != value:
                raise ValueError(f"cannot override {name} during restore; use from_stage for a fresh training run")
        return cls(domain=domain, cfg=effective, snapshot=snapshot)

    @classmethod
    def _from_stage(cls, domain, snapshot, cfg=None, **kwargs):
        saved = RectangularOrchestratorConfig.from_kwargs(**snapshot.cfg_kwargs)
        effective = cfg or RectangularOrchestratorConfig.from_kwargs(**{**saved.to_dict(), **kwargs})
        if effective.module_kwargs != saved.module_kwargs:
            raise ValueError("stage handoff cannot change the sampler module architecture")
        return cls(domain=domain, cfg=effective, snapshot=snapshot, fresh_training=True)

    def generate(self, max_samples: int) -> SampleBatch | GenerationStatus:
        if type(max_samples) is not int or max_samples <= 0:
            raise ValueError("max_samples must be a positive integer")
        remaining = self._training_samples_remaining()
        if remaining == 0:
            return GenerationStatus.WAITING
        count = min(max_samples, remaining if remaining is not None else self.cfg.max_batch_size)
        continuous = np.empty((count, self.domain.continuous_dims), dtype=np.float64)
        discrete = np.empty((count, self.num_discrete_dims), dtype=np.int64)
        weights = np.empty(count, dtype=np.float64)
        for start in range(0, count, self.cfg.max_batch_size):
            size = min(self.cfg.max_batch_size, count - start)
            if self.execution.cuda_graphs and self.device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad() if remaining is not None else torch.inference_mode():
                samples, prob = self.execution.sample(size, return_prob=True)
            if not torch.isfinite(prob).all() or torch.any(prob <= 0):
                raise ValueError("sampler returned nonpositive or nonfinite probabilities")
            if remaining is not None:
                self.training_status.add_generation_data(samples, prob)
            discrete[start:start + size], continuous[start:start + size] = self.module.to_disc_cont(samples)
            weights[start:start + size] = 1 / prob.numpy(force=True)
        return SampleBatch(xs_continuous=continuous, xs_discrete=discrete, weights=weights,
                           training_remaining=remaining)

    def optimization_step(self) -> None:
        if self.optimizer is None:
            raise ValueError("optimization requires a sampler module with trainable parameters")
        self.optimizer.zero_grad()
        last_loss = None
        if not torch.any(self.training_status.ingested_weights):
            last_loss = 0.0
        else:
            if self.execution.cuda_graphs and self.device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            q_test = self.execution.prob(self.training_status.generated_samples)
            loss = self.loss_function(self.training_status, q_test)
            if not torch.isfinite(loss).item():
                warnings.warn("nonfinite batch loss: skipping optimization", RuntimeWarning)
            else:
                loss.backward()
                if all(p.grad is None or torch.isfinite(p.grad).all() for p in self.module.parameters()):
                    self.optimizer.step()
                    last_loss = loss.item()
                    if self.scheduler is not None:
                        self.scheduler.step()
                else:
                    warnings.warn("nonfinite batch gradients: skipping optimization", RuntimeWarning)
        self.training_status.finish_step(last_loss)
        if self.training_status.step >= self.cfg.training_steps:
            self.module.eval()

    def feedback(self, feedback: Feedback) -> None:
        if self.training_status.step >= self.cfg.training_steps:
            return
        self.training_status.add_feedback_data(np.asarray(feedback.values))
        if self.training_status.test_ready_for_training():
            self.optimization_step()

    def pdf(self, xs_discrete: NDArray, xs_continuous: NDArray, c: NDArray | None = None):
        result = self.module.pdf(xs_discrete, xs_continuous, c)
        return None if result is None else np.asarray(result).reshape(-1)

    def discrete_mass(self, subspace: NDArray, c: NDArray | None = None):
        return self.module.discrete_mass(subspace, c)

    def _training_samples_remaining(self) -> int | None:
        if self.training_status.step < self.cfg.training_steps:
            return self.cfg.training_batch_size - self.training_status.num_generated
        return None

    def training_samples_remaining(self):
        return self._training_samples_remaining()

    def diagnostics(self):
        status = self.training_status
        return {"step": status.step, "last_loss": status.last_loss,
                "trained_samples": status.total_trained_samples}

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available() and self.cfg.use_gpu:
            return torch.device(f"cuda:{min(self.cfg.cuda_id, torch.cuda.device_count() - 1)}")
        return torch.device("cpu")
