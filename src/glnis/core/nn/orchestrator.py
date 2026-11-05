# type:ignore

import torch
import numpy as np

from typing import Literal, Any
from numpy.typing import NDArray
from torch._tensor import Tensor
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from pathlib import Path

from gammaboard_process import Domain, RectangularDomain, Feedback, SampleBatch, GenerationStatus

from glnis.utils.types import RectangularTrainingStatus
from glnis.core.nn.sampler_module import RectangularSamplerModule
from glnis.core.nn.losses import get_loss
from glnis.core.config import BuildFromKind, Config


@dataclass
class OrchestratorConfig(Config):
    save_path: str | None = None
    sampler_kwargs: dict = field(default_factory=dict)


@dataclass
class OrchestratorSnapshot:
    kind: str
    save_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RectangularOrchestratorConfig(OrchestratorConfig):
    seed: int = 42
    training_steps: int = 100
    training_batch_size: int = 1000
    max_batch_size: int = 100_000
    use_gpu: bool = True
    use_f64: bool = True
    cuda_id: int = 1
    learning_rate: float = 1e-3
    use_scheduler: bool = True
    scheduler_type: Literal["cosineannealing"] = "cosineannealing"
    loss_type: Literal["variance", "kl_divergence", "rkl_divergence",] = "kl_divergence"
    loss_kwargs: dict = field(default_factory=dict)
    sampler_kwargs: dict = field(default_factory=dict)


@dataclass
class RectangularOrchestratorSnapshotData:
    cfg: RectangularOrchestratorConfig
    training_status: RectangularTrainingStatus
    module_snapshot: bytes
    optimizer_state: dict
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
    def from_snapshot(
        cls,
        *,
        snapshot: Any,
        domain: Domain,
    ):
        snapshot = OrchestratorSnapshot(**snapshot) if not isinstance(snapshot, OrchestratorSnapshot) else snapshot
        # subclass = cls.subclass_for_kind(snapshot.kind)
        subclass = RectangularOrchestrator if snapshot.kind == "rectangular" else cls.subclass_for_kind(snapshot.kind)
        return subclass._from_snapshot(domain, snapshot)

    @classmethod
    @abstractmethod
    def _from_snapshot(cls, domain: Domain, snapshot: Any) -> None:
        pass


class RectangularOrchestrator(Orchestrator):
    KIND = "rectangular"
    DEFAULT_KIND = "rectangular"

    def __init__(
        self,
        domain: RectangularDomain,
        dims_c: int = 0,
        cfg: RectangularOrchestratorConfig | None = None,
        snapshot: RectangularOrchestratorSnapshotData | None = None,
        **kwargs,
    ):
        self.domain = domain
        self.num_discrete_dims = len(domain.discrete_cardinalities)
        self.dims_c = dims_c

        if snapshot is not None:
            self.cfg = snapshot.cfg
        else:
            self.cfg = cfg if cfg is not None else RectangularOrchestratorConfig.from_kwargs(**kwargs)

        self.device = self._get_device()
        torch.set_default_dtype(torch.float64 if self.cfg.use_f64 else torch.float32)
        torch.set_default_device(self.device)

        if snapshot is not None:
            self.device = self._get_device()
            self.module = RectangularSamplerModule.from_snapshot(snapshot.module_snapshot)
            if snapshot.torch_cpu_rng_state is not None:
                torch.set_rng_state(snapshot.torch_cpu_rng_state)
            if snapshot.torch_gpu_rng_state is not None and self.device.type == "cuda":
                torch.cuda.set_rng_state_all(snapshot.torch_gpu_rng_state)
            self.training_status = snapshot.training_status or RectangularTrainingStatus(
                self.domain.continuous_dims + self.num_discrete_dims, self.cfg.training_batch_size
            )
            self.optimizer = torch.optim.Adam(self.module.parameters(), lr=self.cfg.learning_rate)
            if snapshot.optimizer_state is not None:
                self.optimizer.load_state_dict(snapshot.optimizer_state)
            self.scheduler = self._get_scheduler(
                self.cfg.training_steps, self.cfg.scheduler_type, self.optimizer,) if self.cfg.use_scheduler else None
            if snapshot.scheduler_state is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(snapshot.scheduler_state)
        else:
            self.module: RectangularSamplerModule = RectangularSamplerModule.from_kwargs(
                domain=self.domain,
                dims_c=dims_c,
                **self.cfg.sampler_kwargs,
            )
            self.training_status = RectangularTrainingStatus(
                self.domain.continuous_dims + self.num_discrete_dims, self.cfg.training_batch_size
            )
            if self.cfg.seed is not None:
                torch.manual_seed(self.cfg.seed)
            self.optimizer = torch.optim.Adam(self.module.parameters(), lr=self.cfg.learning_rate)
            self.scheduler = self._get_scheduler(
                self.cfg.training_steps, self.cfg.scheduler_type, self.optimizer,) if self.cfg.use_scheduler else None
        
        self.training_status.to_device(self.device)
        self.loss_function = get_loss(self.cfg.loss_type, self.cfg.loss_kwargs)
        self.module.to(self.device, dtype=torch.float64 if self.cfg.use_f64 else torch.float32)

    def snapshot(self) -> OrchestratorSnapshot:
        if self.cfg.save_path is None:
            raise ValueError("save_path must be specified in the configuration to save a snapshot.") from None

        snapshot = OrchestratorSnapshot(
            kind=self.KIND,
            save_path=self.cfg.save_path,
        )
        destination = Path(self.cfg.save_path)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            snapshot_data = RectangularOrchestratorSnapshotData(
                cfg=self.cfg,
                training_status=self.training_status,
                module_snapshot=self.module.snapshot(),
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict() if self.scheduler is not None else None,
                torch_cpu_rng_state=torch.get_rng_state(),
                torch_gpu_rng_state=torch.cuda.get_rng_state_all() if self._get_device().type == "cuda" else None,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb") as stream:
                torch.save(snapshot_data, stream)
            temporary.replace(destination)
        except Exception as error:
            raise RuntimeError(f"Failed to save MadNIS Integrator state to file '{destination}': {error}") from error
        finally:
            temporary.unlink(missing_ok=True)
        return snapshot

    @classmethod
    def _from_snapshot(cls, domain: RectangularDomain, snapshot: OrchestratorSnapshot | dict):
        if not isinstance(snapshot, OrchestratorSnapshot):
            snapshot = OrchestratorSnapshot(**snapshot)
        if snapshot.kind != cls.KIND:
            raise ValueError(f"snapshot kind '{snapshot.kind}' does not match orchestrator kind '{cls.KIND}'")
        if snapshot.save_path is None:
            raise ValueError("snapshot save_path must be set to load a snapshot")
        source_path = Path(snapshot.save_path)
        if not source_path.exists():
            raise FileNotFoundError(f"snapshot file '{source_path}' does not exist")
        try:
            with source_path.open("rb") as stream:
                snapshot_data = torch.load(stream, weights_only=False)
            if not isinstance(snapshot_data, RectangularOrchestratorSnapshotData):
                raise ValueError("snapshot data must be an instance of RectangularOrchestratorSnapshotData")
            return cls(domain=domain, snapshot=snapshot_data)
        except Exception as error:
            raise RuntimeError(f"Failed to load MadNIS Integrator state from file '{source_path}': {error}") from error

    def generate(self, max_samples: int) -> SampleBatch | GenerationStatus:
        training_remaining = self._training_samples_remaining()
        if training_remaining == 0:
            return GenerationStatus.WAITING
        if training_remaining is not None:
            nr_samples = min(max_samples, training_remaining)
        else:
            nr_samples = min(max_samples, self.cfg.max_batch_size)

        continuous = np.empty((nr_samples, self.domain.continuous_dims), dtype=np.float64)
        discrete = np.empty((nr_samples, self.num_discrete_dims), dtype=np.int64)
        wgt = np.empty((nr_samples), dtype=np.float64)

        n_eval = 0
        while n_eval < nr_samples:
            n = min(self.cfg.max_batch_size, nr_samples - n_eval)
            with torch.no_grad():
                x_all, prob = self.module.sample(
                    n,
                    return_prob=True,
                )

            if training_remaining is not None:
                self.training_status.add_generation_data(x_all, prob)

            discrete[n_eval: n_eval + n, :], continuous[n_eval: n_eval + n, :] = self.module.to_disc_cont(x_all)

            wgt[n_eval: n_eval + n] = 1 / prob.numpy(force=True)
            n_eval += n

        return SampleBatch(
            xs_continuous=continuous,
            xs_discrete=discrete,
            weights=wgt,
            training_remaining=training_remaining
        )

    def optimization_step(self) -> None:
        self.optimizer.zero_grad()
        q_test = self.module.prob(self.training_status.generated_samples)
        loss = self.loss_function(self.training_status, q_test)
        if loss.isnan().item():
            warnings.warn("nan batch: skipping optimization")
            last_loss = None
        else:
            loss.backward()
            self.optimizer.step()
            last_loss = loss.item()
            if self.scheduler is not None:
                self.scheduler.step()

        self.training_status.finish_step(last_loss)

    def feedback(self, feedback: Feedback) -> None:
        if self.training_status.step >= self.cfg.training_steps:
            gb.log(
                f"Training target of {self.cfg.training_steps * self.cfg.training_batch_size} samples already reached. Ignoring additional training values.", level="warn")
            return
        values = np.asarray(feedback.values)
        self.training_status.add_feedback_data(values)

        if self.training_status.test_ready_for_training():
            self.optimization_step()

    def pdf(self, xs_discrete: NDArray, xs_continuous: NDArray, c: NDArray | None = None) -> NDArray:
        return self.module.pdf(xs_discrete, xs_continuous, c)

    def discrete_mass(self, subspace: NDArray, c: NDArray | None = None) -> float | None:
        return self.module.discrete_mass(subspace, c)

    def _training_samples_remaining(self) -> int | None:
        if self.training_status.step < self.cfg.training_steps:
            return max(self.cfg.training_batch_size - self.training_status.num_generated, 0)
        return None

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available() and self.cfg.use_gpu:
            cuda_id = min(self.cfg.cuda_id, torch.cuda.device_count() - 1)
            major, minor = torch.cuda.get_device_capability(cuda_id)
            if (7, 0) <= (major, minor) < (12, 0):
                torch.cuda.set_device(cuda_id)
                return torch.device(f"cuda:{cuda_id}")
        return torch.device("cpu")

    def _get_scheduler(
        self, T_max: int, scheduler_type: str | None, optimizer
    ) -> torch.optim.lr_scheduler.CosineAnnealingLR | None:
        if scheduler_type is None:
            return None
        match scheduler_type.lower():
            case "cosineannealing":
                def scheduler(optimizer):
                    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
            case _:
                return None

        return scheduler(optimizer)
