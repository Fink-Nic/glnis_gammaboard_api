# type: ignore
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from copy import deepcopy
import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple, get_args, get_origin, get_type_hints
from types import UnionType

import numpy as np
import torch
import gammaboard_process as gb
from madnis.integrator import Integrand, Integrator, losses
from madnis.integrator import SampleBatch as MadnisSampleBatch
from numpy.typing import NDArray
from torch._tensor import Tensor


def _validate_config(config: Any, path: str = "sampler") -> None:
    """Validate dataclass settings without coercing TOML strings or booleans."""
    for name, annotation in get_type_hints(type(config)).items():
        value = getattr(config, name)
        key = f"{path}.{name}"
        origin = get_origin(annotation)
        if origin is UnionType:
            if value is None and type(None) in get_args(annotation):
                continue
            annotation = next(item for item in get_args(annotation) if item is not type(None))
            origin = get_origin(annotation)
        if origin is Literal:
            if value not in get_args(annotation):
                raise ValueError(f"{key} must be one of {get_args(annotation)!r}")
        elif annotation is int:
            minimum = 0 if name in {"seed", "training_steps", "cuda_id"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        elif annotation is float:
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a finite number > 0")
        elif annotation in (bool, str):
            if type(value) is not annotation:
                raise ValueError(f"{key} must be {annotation.__name__}")
        elif is_dataclass(annotation):
            if not isinstance(value, annotation):
                raise ValueError(f"{key} must be a {annotation.__name__}")
            _validate_config(value, key)


@dataclass
class FlowConfig:
    """
    Config for the Normalizing Flow module which generates the continuous samples.
    """

    uniform_latent: bool = True
    permutations: Literal["log"] = "log"
    layers: int = 3
    units: int = 128
    bins: int = 8
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_bin_derivative: float = 1e-3


@dataclass
class TransformerConfig:
    """
    Config for the Transformer module which generates the discrete samples.
    """

    embedding_dim: int = 64
    feedforward_dim: int = 64
    heads: int = 4
    mlp_units: int = 64
    transformer_layers: int = 2


@dataclass
class MadeConfig:
    """
    Config for the MADE module which generates the discrete samples. Alternative to Transformer (but worse).
    """

    layers: int = 3
    nodes_per_feature: int = 64


@dataclass
class MadnisConfig:
    """
    General config.
    Args:
        seed:
            Random seed for reproducibility. No Stream_ID for now.
        batch_size:
            Number of samples per training step.
        max_batch_size:
            Maximum number of samples to generate in one forward pass when calling get_samples(). Avoids out-of-memory errors on GPU.
        learning_rate:
            Learning rate for the optimizer.
        use_scheduler:
            If true, a learning rate scheduler will be used during training.
        scheduler_type:
            Currently only supports "cosineannealing". Ignored if use_scheduler is False.
        loss_type:
            Loss function to optimize during training. Options are "variance", "variance_softclip", "kl_divergence", and "kl_divergence_softclip".
        discrete_dims_position:
            Whether the sampler generates the discrete points before or after the continuous.
        discrete_model:
            Whether to use a Transformer or MADE for the discrete flow.
        flow_config:
            See ``FlowConfig`` dataclass for details
        transformer_config:
            See ``TransformerConfig`` dataclass for details
        made_config:
            See ``MadeConfig`` dataclass for details
    """

    seed: int = 42
    kind: Literal["default", "sgn_strat_naive", "sgn_strat_full"] = "default"
    training_steps: int = 100
    training_batch_size: int = 1000
    max_batch_size: int = 100_000
    save_path: str | None = None
    use_gpu: bool = True
    cuda_id: int = 1
    learning_rate: float = 1e-3
    use_scheduler: bool = True
    scheduler_type: Literal["cosineannealing"] = "cosineannealing"
    loss_type: Literal[
        "variance", "variance_softclip", "kl_divergence", "kl_divergence_softclip", "rkl_divergence",
    ] = "kl_divergence"
    discrete_dims_position: Literal["first", "last"] = "first"
    condition_integrand_first: bool = True
    freeze_discrete_model: bool = False
    discrete_model: Literal["transformer", "made"] = "transformer"
    flow_config: FlowConfig = field(default_factory=FlowConfig)
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    made_config: MadeConfig = field(default_factory=MadeConfig)
    gb_loglevel: Literal["trace", "debug", "info", "warn", "error"] = "trace"

    def __post_init__(self) -> None:
        _validate_config(self)
        if self.transformer_config.embedding_dim % self.transformer_config.heads:
            raise ValueError("sampler.transformer_config.embedding_dim must be divisible by heads")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> MadnisConfig:
        if not isinstance(config_dict, dict):
            raise ValueError("sampler must be a configuration table")
        unknown = set(config_dict) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown sampler settings: {', '.join(f'sampler.{key}' for key in sorted(unknown))}")
        values = dict(config_dict)
        for name, config_type in (
            ("flow_config", FlowConfig),
            ("transformer_config", TransformerConfig),
            ("made_config", MadeConfig),
        ):
            nested = values.get(name, {})
            if isinstance(nested, config_type):
                continue
            if not isinstance(nested, dict):
                raise ValueError(f"sampler.{name} must be a configuration table")
            unknown = set(nested) - {item.name for item in fields(config_type)}
            if unknown:
                raise ValueError(f"Unknown sampler settings: {', '.join(f'sampler.{name}.{key}' for key in sorted(unknown))}")
            values[name] = config_type(**nested)
        return cls(**values)


@dataclass
class TrainingStatus:
    sample_dim: int
    training_batch_size: int
    num_generated: int = 0
    num_ingested: int = 0
    total_trained_samples: int = 0
    produced_batches: int = 0
    produced_samples: int = 0
    step: int = 0
    last_loss: float | None = None
    loss_history: List[float | None] = field(default_factory=list)
    generated_samples: Tensor = None
    generated_probs: Tensor = None
    ingested_weights: Tensor = None

    def __post_init__(self):
        if self.generated_samples is None:
            self.generated_samples = torch.empty(
                (self.training_batch_size, self.sample_dim), dtype=torch.float64
            )
        if self.generated_probs is None:
            self.generated_probs = torch.empty(
                (self.training_batch_size,), dtype=torch.float64
            )
        if self.ingested_weights is None:
            self.ingested_weights = torch.empty(
                (self.training_batch_size,), dtype=torch.float64
            )

    def to_device(self, device: torch.device) -> None:
        self.generated_samples = self.generated_samples.to(device)
        self.generated_probs = self.generated_probs.to(device)
        self.ingested_weights = self.ingested_weights.to(device)

    def add_generation_data(self, samples: Tensor, probs: Tensor) -> None:
        assert samples.shape[0] == probs.shape[0], "Number of samples and sampling weights must match."
        n_samples = samples.shape[0]
        if n_samples == self.training_batch_size:
            self.generated_samples = samples
            self.generated_probs = probs
        else:
            self.generated_probs[self.num_generated:self.num_generated + n_samples] = probs
            self.generated_samples[self.num_generated:self.num_generated + n_samples, :] = samples
        self.num_generated += n_samples
        self.produced_batches += 1

    def add_feedback_data(self, weights: NDArray) -> None:
        weights = torch.as_tensor(weights, dtype=torch.float64, device=self.ingested_weights.device)
        n_weights = weights.shape[0]
        if n_weights == self.training_batch_size:
            self.ingested_weights = weights
        else:
            self.ingested_weights[self.num_ingested:self.num_ingested + n_weights] = weights
        self.num_ingested += n_weights

    def test_ready_for_training(self) -> bool:
        return self.num_generated == self.training_batch_size and self.num_ingested == self.training_batch_size

    def add_loss(self, loss: float) -> None:
        self.loss_history.append(loss)
        self.last_loss = loss
        self.step += 1
        self.num_generated = 0
        self.num_ingested = 0
        self.total_trained_samples += self.training_batch_size


@dataclass
class GammaLoopEvaluatorMetadata:
    state_folder: str
    process_id: int
    integrand_name: str
    momentum_space: bool = False
    coordinate_space: str = "x_space"
    domain_axes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, metadata: dict[str, Any]) -> GammaLoopEvaluatorMetadata:
        if metadata.get("kind") != "gammaloop":
            raise ValueError("GammaLoop metadata must have kind='gammaloop'")
        return cls(
            state_folder=str(metadata["state_folder"]),
            process_id=int(metadata["process_id"]),
            integrand_name=str(metadata["integrand_name"]),
            momentum_space=bool(metadata.get("momentum_space", False)),
            coordinate_space=str(metadata.get("coordinate_space", "x_space")),
            domain_axes=[str(axis) for axis in metadata.get("domain_axes", [])],
        )


def parse_gammaloop_metadata(
    evaluator_metadata: dict[str, Any] | None,
) -> GammaLoopEvaluatorMetadata | None:
    if not evaluator_metadata:
        return None
    if evaluator_metadata.get("kind") != "gammaloop":
        return None
    return GammaLoopEvaluatorMetadata.from_dict(evaluator_metadata)


class MadnisSampler(gb.Sampler):
    CHECKPOINT_KIND = "default"

    def __init__(
        self,
        *,
        discrete_cardinalities: List[int],
        continuous_dims: int,
        cfg: MadnisConfig | None = None,
        seed: int = 42,
        training_steps: int = 100,
        training_batch_size: int = 1000,
        max_batch_size: int = 100_000,
        use_gpu: bool = True,
        cuda_id: int = 1,
        learning_rate: float = 1e-3,
        use_scheduler: bool = True,
        save_path: str | None = None,
        scheduler_type: Literal["cosineannealing"] = "cosineannealing",
        loss_type: Literal[
            "variance", "variance_softclip", "kl_divergence", "kl_divergence_softclip"
        ] = "kl_divergence",
        discrete_dims_position: Literal["first", "last"] = "first",
        condition_integrand_first: bool = True,
        freeze_discrete_model: bool = False, 
        discrete_model: Literal["transformer", "made"] = "transformer",
        flow_config: dict[str, Any] | FlowConfig | None = None,
        transformer_config: dict[str, Any] | TransformerConfig | None = None,
        made_config: dict[str, Any] | MadeConfig | None = None,
        training_status: TrainingStatus | None = None,
        torch_cpu_rng_state: Tensor | None = None,
        torch_gpu_rng_state: Tensor | None = None,
        madnis_blob: bytes | None = None,
        evaluator_metadata: dict[str, Any] | None = None,
        parser: dict[str, Any] | None = None,
        mapping: dict[str, Any] | None = None,
        graph_properties: dict[str, Any] | None = None,
        gb_loglevel: Literal["trace", "debug", "info", "warn", "error"] = "trace",
        _build_integrator: bool = True,
    ):
        torch.set_default_dtype(torch.float64)

        self.cfg: MadnisConfig = cfg or MadnisConfig(
            seed=seed,
            training_steps=training_steps,
            training_batch_size=training_batch_size,
            max_batch_size=max_batch_size,
            use_gpu=use_gpu,
            cuda_id=cuda_id,
            learning_rate=learning_rate,
            use_scheduler=use_scheduler,
            save_path=save_path,
            scheduler_type=scheduler_type,
            loss_type=loss_type,
            discrete_dims_position=discrete_dims_position,
            condition_integrand_first=condition_integrand_first,
            freeze_discrete_model=freeze_discrete_model,
            discrete_model=discrete_model,
            flow_config=(
                flow_config
                if isinstance(flow_config, FlowConfig)
                else FlowConfig(**(flow_config or {}))
            ),
            transformer_config=(
                transformer_config
                if isinstance(transformer_config, TransformerConfig)
                else TransformerConfig(**(transformer_config or {}))
            ),
            made_config=(
                made_config
                if isinstance(made_config, MadeConfig)
                else MadeConfig(**(made_config or {}))
            ),
            gb_loglevel=gb_loglevel,
        )

        self.discrete_cardinalities: List[int] = discrete_cardinalities
        self.continuous_dims: int = continuous_dims
        self._input_discrete_cardinalities = list(discrete_cardinalities)
        self._input_continuous_dims = continuous_dims
        self._checkpoint_initialization = deepcopy(dict(
            parser=parser, mapping=mapping, graph_properties=graph_properties,
            evaluator_metadata=evaluator_metadata,
        ))
        self.evaluator_metadata: dict[str, Any] = evaluator_metadata or {}
        self.gammaloop_metadata = parse_gammaloop_metadata(self.evaluator_metadata)
        if mapping is not None:
            try:
                from glnis.core.parser import MetaDataParser
            except ImportError:
                raise ImportError(
                    "Momentum space evaluation is only supported in the 'glnis_gammaboard_api' package."
                )
            Parser = MetaDataParser(
                config=parser,
                metadata=self.evaluator_metadata,
                graph_properties=graph_properties,
            )
            self.mapping = Parser.get_layered_mapping_instance(mapping)
        else:
            self.mapping = None

        if self.mapping is not None:
            self.num_discrete_dims = len(self.mapping.discrete_cardinalities) + len(
                self.discrete_cardinalities
            )
            self.continuous_dims = (
                self.mapping.continuous_dims_in or self.continuous_dims
            )
        else:
            self.num_discrete_dims = len(self.discrete_cardinalities)

        if not isinstance(self.discrete_cardinalities, list) or any(
            cardinality <= 0 for cardinality in self.discrete_cardinalities
        ):
            raise TypeError(
                "discrete_cardinalities must be a list of positive integers"
            )
        self.discrete_cardinalities = [
            int(cardinality) for cardinality in self.discrete_cardinalities
        ]
        if self.mapping is not None:
            if self.cfg.condition_integrand_first:
                all_discrete_cardinalities = (
                    self.discrete_cardinalities + self.mapping.discrete_cardinalities
                )
            else:
                all_discrete_cardinalities = (
                    self.mapping.discrete_cardinalities + self.discrete_cardinalities
                )
        else:
            all_discrete_cardinalities = self.discrete_cardinalities
        self._all_discrete_cardinalities = [int(c) for c in all_discrete_cardinalities]
        self._sampled_discrete_indices = np.flatnonzero(
            np.asarray(self._all_discrete_cardinalities) > 1
        )
        self._sampled_discrete_cardinalities = [
            self._all_discrete_cardinalities[i] for i in self._sampled_discrete_indices
        ]
        self.num_discrete_dims = len(self._all_discrete_cardinalities)
        if self.continuous_dims <= 0:
            raise ValueError("continuous_dims must be > 0")

        self.device = self._get_device()
        self.training_status = training_status or TrainingStatus(self.continuous_dims + self.num_discrete_dims, self.cfg.training_batch_size)
        self.training_status.to_device(self.device)

        if torch_cpu_rng_state is not None:
            if torch_gpu_rng_state is not None and self.device.type == "cuda":
                torch.cuda.set_rng_state(torch_gpu_rng_state, device=self.device)
            torch.set_rng_state(torch_cpu_rng_state)
        else:
            torch.manual_seed(self.cfg.seed)
            torch.cuda.manual_seed(self.cfg.seed)
        if not _build_integrator:
            self.madnis = None
        elif madnis_blob is not None:
            buffer = io.BytesIO(madnis_blob)
            self.madnis: Integrator = torch.load(
                buffer, map_location="cpu", weights_only=False
            )
            if self.device.type != "cpu":
                # Model tensors and optimizer moments belong on the training
                # device, but Adam's non-capturable step counters stay on CPU.
                # Optimizer.load_state_dict applies that per-value policy.
                optimizer = self.madnis.optimizer
                optimizer_state = optimizer.state_dict()
                parameter_names = {id(parameter): name for name, parameter in self.madnis.named_parameters()}
                group_names = [[parameter_names[id(parameter)] for parameter in group["params"]]
                               for group in optimizer.param_groups]
                self.madnis.to(self.device)
                parameters = dict(self.madnis.named_parameters())
                for group, names in zip(optimizer.param_groups, group_names):
                    group["params"] = [parameters[name] for name in names]
                optimizer.load_state_dict(optimizer_state)
        else:
            self.madnis: Integrator = self._get_madnis_integrator()

        if self.cfg.freeze_discrete_model and self.madnis is not None:
            if hasattr(self.madnis.flow, "discrete_flow") and self.madnis.flow.discrete_flow is not None:
                for param in self.madnis.flow.discrete_flow.parameters():
                    param.requires_grad = False
            else:
                gb.log("Warning: freeze_discrete_model is True but no discrete flow found.", level="warn")

        gb.log(f"MadNIS Sampler initialized with {self.continuous_dims} continuous dims and {self.num_discrete_dims} discrete dims (cardinalities: {self.discrete_cardinalities})", level=self.cfg.gb_loglevel)


    @classmethod
    def from_config(
            cls,
            continuous_dims: int,
            discrete_cardinalities: list[int],
            cfg: MadnisConfig,
            **values,
        ) -> "MadnisSampler":

        match cfg.kind:
            case "default":
                cls = MadnisSampler
            case "sgn_strat_naive":
                cls = MadnisSamplerNaiveSgnStrat
            case "sgn_strat_full":
                cls = MadnisSamplerFullSgnStrat

        return cls(
            continuous_dims=continuous_dims,
            discrete_cardinalities=discrete_cardinalities,
            cfg=cfg,
            **values,
        )


    @classmethod
    def from_snapshot(
        cls,
        *,
        snapshot: dict[str, Any],
        discrete_cardinalities: List[int],
        continuous_dims: int,
        init_args: dict[str, Any] | None = None,
        evaluator_metadata: dict[str, Any] | None = None,
    ) -> MadnisSampler:
        """Restore models, unfinished feedback, and RNG state from one checkpoint.

        New checkpoints retain their configuration; old checkpoints still use
        init_args. Device/output settings can be overridden, while model and
        optimizer state are restored from the checkpoint.
        Dimensions retain the public, pre-stratification/pre-mapping convention.
        """
        save_path = snapshot.get("save_path")
        if save_path is None:
            raise ValueError("Snapshot is missing 'save_path' key required for loading the Integrator state.")
        try:
            with open(save_path, "rb") as stream:
                state = torch.load(stream, map_location="cpu", weights_only=False)
            version = state.get("checkpoint_version", 1)
            if version not in (1, 2):
                raise ValueError(f"Unsupported sampler checkpoint version {version}")

            overrides = dict(init_args or {})
            initialization = deepcopy(state.get("initialization", {}))
            for name in ("parser", "mapping", "graph_properties", "evaluator_metadata"):
                if name in overrides:
                    initialization[name] = overrides.pop(name)
            if snapshot.get("graph_properties") is not None:
                initialization["graph_properties"] = snapshot["graph_properties"]
            if evaluator_metadata is not None:
                initialization["evaluator_metadata"] = evaluator_metadata

            settings = deepcopy(state.get("config", {}))
            settings.setdefault("kind", state.get("kind", snapshot.get("kind", cls.CHECKPOINT_KIND)))
            settings["save_path"] = str(save_path)
            for name, value in overrides.items():
                if isinstance(value, dict) and isinstance(settings.get(name), dict):
                    settings[name] = {**settings[name], **value}
                else:
                    settings[name] = value
            cfg = MadnisConfig.from_dict(settings)
            if state.get("kind", cfg.kind) != cfg.kind:
                raise ValueError("Cannot restore a checkpoint using a different sampler kind")
            # Legacy payloads already contain training_status; it was never part
            # of the JSON snapshot returned to GammaBoard.
            instance = cls._from_checkpoint_state(
                state, continuous_dims=continuous_dims,
                discrete_cardinalities=discrete_cardinalities,
                cfg=cfg, initialization=initialization,
            )
            # Constructors seed the shared generators. Restore them only after
            # every model (including both sign children) has been reconstructed.
            cpu_state = state.get("torch_cpu_rng_state")
            if cpu_state is not None:
                torch.set_rng_state(cpu_state)
            gpu_states = state.get("torch_gpu_rng_states", {})
            for device in instance._checkpoint_devices():
                rng_state = gpu_states.get(device.index)
                if rng_state is None and not gpu_states:
                    rng_state = state.get("torch_gpu_rng_state")
                if rng_state is not None:
                    torch.cuda.set_rng_state(rng_state, device=device)
            return instance
        except Exception as error:
            raise RuntimeError(f"Failed to load MadNIS Integrator state from file '{save_path}': {error}") from error

    @classmethod
    def _from_checkpoint_state(
        cls, state: dict[str, Any], *, continuous_dims: int | None = None,
        discrete_cardinalities: list[int] | None = None,
        cfg: MadnisConfig | None = None, initialization: dict[str, Any] | None = None,
    ) -> MadnisSampler:
        cfg = cfg or MadnisConfig.from_dict(state["config"])
        arguments = deepcopy(state.get("initialization", {})) if initialization is None else initialization
        arguments = dict(arguments)
        arguments["training_status"] = state.get("training_status")
        if cfg.kind == "sgn_strat_full":
            arguments["_checkpoint_children"] = state["children"]
            arguments["_pending_strata"] = state["pending_strata"]
        else:
            arguments["madnis_blob"] = state["madnis_blob"]
        instance = cls.from_config(
            continuous_dims=state["continuous_dims"] if continuous_dims is None else continuous_dims,
            discrete_cardinalities=state["discrete_cardinalities"] if discrete_cardinalities is None else discrete_cardinalities,
            cfg=cfg,
            **arguments,
        )
        if instance.training_status.training_batch_size != cfg.training_batch_size:
            raise ValueError("Checkpoint training batch size does not match the restored configuration")
        if cfg.kind != "sgn_strat_full":
            instance._bind_integrator_callbacks()
        return instance

    def _bind_integrator_callbacks(self) -> None:
        """Bind deserialized model callbacks to this sampler, never its old owner."""
        self.madnis.integrand = self._get_madnis_integrand()
        self.madnis.loss = self._get_loss()
        channel_remap = (
            None if self.madnis.group_channels and not self.madnis.group_channels_uniform
            else self.madnis.integrand.remap_channels
        )
        flow = self.madnis.flow
        for component in (flow, getattr(flow, "continuous_flow", None), getattr(flow, "discrete_flow", None)):
            if component is None:
                continue
            if hasattr(component, "channel_remap_function"):
                component.channel_remap_function = channel_remap
            if hasattr(component, "prior_prob_function"):
                component.prior_prob_function = self._madnis_discrete_prior_prob_function

    def _integrator_blob(self) -> bytes:
        """Serialize the model and optimizer while excluding bound callbacks."""
        if self.madnis is None:
            raise RuntimeError("MadnisSampler not properly initialized with an Integrator instance")
        attributes = [(self.madnis, "integrand"), (self.madnis, "loss")]
        flow = self.madnis.flow
        for component in (flow, getattr(flow, "continuous_flow", None), getattr(flow, "discrete_flow", None)):
            if component is not None:
                attributes.extend((component, name) for name in ("prior_prob_function", "channel_remap_function")
                                  if hasattr(component, name))
        saved = []
        try:
            for owner, name in attributes:
                saved.append((owner, name, getattr(owner, name)))
                setattr(owner, name, None)
            buffer = io.BytesIO()
            torch.save(self.madnis, buffer)
            return buffer.getvalue()
        finally:
            for owner, name, value in reversed(saved):
                setattr(owner, name, value)

    def _checkpoint_common_state(self) -> dict[str, Any]:
        configuration = asdict(self.cfg)
        configuration["kind"] = self.CHECKPOINT_KIND
        initialization = deepcopy(self._checkpoint_initialization)
        if self.mapping is not None:
            initialization["graph_properties"] = self.mapping.graph_properties.as_init_dict()
        return dict(
            kind=self.CHECKPOINT_KIND,
            config=configuration,
            continuous_dims=self._input_continuous_dims,
            discrete_cardinalities=list(self._input_discrete_cardinalities),
            initialization=initialization,
            training_status=self.training_status,
        )

    def _checkpoint_state(self) -> dict[str, Any]:
        return {**self._checkpoint_common_state(), "madnis_blob": self._integrator_blob()}

    def _checkpoint_devices(self) -> set[torch.device]:
        return {self.device} if self.device.type == "cuda" else set()

    def snapshot(self) -> Dict[str, Any]:
        """Atomically save a self-contained checkpoint and return JSON metadata."""
        if self.cfg.save_path is None:
            return dict(Warning="No save path provided. No snapshot was created.")
        destination = Path(self.cfg.save_path)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            rng = dict(
                torch_cpu_rng_state=torch.get_rng_state(),
                torch_gpu_rng_states={device.index: torch.cuda.get_rng_state(device)
                                      for device in self._checkpoint_devices()},
            )
            state = dict(self._checkpoint_state(), checkpoint_version=2, **rng)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb") as stream:
                torch.save(state, stream)
            temporary.replace(destination)
        except Exception as error:
            raise RuntimeError(f"Failed to save MadNIS Integrator state to file '{destination}': {error}") from error
        finally:
            temporary.unlink(missing_ok=True)
        return dict(
            save_path=str(destination), kind=self.CHECKPOINT_KIND, checkpoint_version=2,
            graph_properties=state["initialization"].get("graph_properties"),
        )

    def generate(self, remaining_sample_budget: int | None) -> Any:
        training_remaining = self.training_samples_remaining()
        if training_remaining == 0:
            return gb.GenerationStatus.WAITING
        if training_remaining is not None:
            if remaining_sample_budget is not None:
                remaining_sample_budget = min(remaining_sample_budget, training_remaining)
            else:
                remaining_sample_budget = training_remaining
        nr_samples = remaining_sample_budget if remaining_sample_budget is not None else self.cfg.max_batch_size
        gb.log(f"Producing {nr_samples} samples with MadNIS sampler ({self.training_status.step=}, {self.training_status.produced_samples=}, {self.training_status.total_trained_samples})", level=self.cfg.gb_loglevel)
        continuous = np.empty((nr_samples, self.continuous_dims), dtype=np.float64)
        discrete = np.empty((nr_samples, self.num_discrete_dims), dtype=np.int64)
        wgt = np.empty((nr_samples), dtype=np.float64)

        n_eval = 0
        while n_eval < nr_samples:
            n = min(self.cfg.max_batch_size, nr_samples - n_eval)
            with torch.no_grad():
                x_all, prob = self.madnis.flow.sample(
                    n,
                    return_prob=True,
                    device=self.device,
                    dtype=torch.float64,
                )

            if training_remaining is not None:
                self.training_status.add_generation_data(x_all, prob)

            discrete[n_eval: n_eval + n, :], continuous[n_eval: n_eval + n, :] = (
                self._madnis_output_to_disc_cont(x_all)
            )
            wgt[n_eval: n_eval + n] = 1 / prob.numpy(force=True)
            n_eval += n

        discrete, continuous, wgt = self._apply_mapping(discrete, continuous, wgt)

        return gb.SampleBatch(xs_discrete=discrete, xs_continuous=continuous, weights=wgt, training_remaining=training_remaining)

    def feedback(self, values: NDArray) -> None:
        if self.training_status.step >= self.cfg.training_steps:
            gb.log(f"Training target of {self.cfg.training_steps * self.cfg.training_batch_size} samples already reached. Ignoring additional training values.", level="warn")
            return
        values = np.asarray(values)
        self.training_status.add_feedback_data(values)

        if self.training_status.test_ready_for_training():
            self._train_step()

    def training_samples_remaining(self) -> int | None:
        if self.training_status.step < self.cfg.training_steps:
            return max(self.cfg.training_batch_size - self.training_status.num_generated, 0)
        return None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Optional runtime diagnostics. Empty dict means no diagnostics available."""
        diagnostics: Dict[str, Any] = dict(
            produced_batches=self.training_status.produced_batches,
            produced_samples=self.training_status.produced_samples,
            total_trained_samples=self.training_status.total_trained_samples,
            last_loss=0.0 if self.training_status.last_loss is None else self.training_status.last_loss,
        )
        if self.gammaloop_metadata is not None:
            diagnostics["gammaloop_metadata"] = asdict(self.gammaloop_metadata)
        return diagnostics

    def pdf(self, xs_discrete: NDArray, xs_continuous: NDArray) -> NDArray | None:
        """Return per-sample PDF values if supported.

        Return a float64 array with shape (nr_samples,) or None to signal that
        the sampler does not support/doesn't provide a PDF for the given batch.
        """
        n_samples = len(xs_discrete)
        if xs_continuous is None:
            xs_continuous = np.zeros((n_samples, 0), dtype=xs_discrete.dtype)

        _, continuous, prob = self._apply_mapping(xs_discrete, xs_continuous, np.ones(n_samples, dtype=np.float64), inverse=True)
        sampled_discrete = xs_discrete[:, self._sampled_discrete_indices]
        prob = prob.flatten()
        # The flow lives in the coordinates before the physical-space mapping.
        # The inverse mapping supplies both those coordinates and its Jacobian.
        if self.madnis.integrand.discrete_dims_position == "first":
            x_all = np.hstack([sampled_discrete, continuous])
        elif self.madnis.integrand.discrete_dims_position == "last":
            x_all = np.hstack([continuous, sampled_discrete])
        else:
            raise ValueError(
                f"Invalid discrete_dims_position: {self.madnis.integrand.discrete_dims_position}"
            )
        x_all = torch.as_tensor(
            x_all.astype(np.float64),
            device=self.device,
            dtype=torch.float64 if xs_continuous.shape[1] > 0 else torch.int64,
        )

        n_eval = 0
        while n_eval < n_samples:
            n = min(self.cfg.max_batch_size, n_samples - n_eval)
            with torch.no_grad():
                if xs_continuous.shape[1] > 0:
                    prob[n_eval: n_eval + n] *= (
                        self.madnis.flow.prob(x_all[n_eval: n_eval + n, :])
                        .numpy(force=True)
                        .reshape(-1)
                    )
                else:
                    prob[n_eval: n_eval + n] *= (
                        self.madnis.flow.discrete_flow.prob(
                            x_all[n_eval: n_eval + n, :]
                        )
                        .numpy(force=True)
                        .reshape(-1)
                    )
            n_eval += n
        
        return prob

    def get_strat_cardinalities(self) -> list[int] | None:
        """Return the shape of stratified channels. Assumed to be at the front of the discrete dimensions. Return None if stratified sampling is not supported."""
        return None

    def strat_channel_transform(self, values: NDArray, strata: NDArray) -> NDArray:
        """
        Transform the integrand values to account for stratified sampling.
        To be called on the aggregator side if stratified sampling is used. 
        """
        raise NotImplementedError(
            "Stratified sampling is not implemented for the default MadNIS sampler."
        )


    def _get_device(self) -> torch.device:
        if torch.cuda.is_available() and self.cfg.use_gpu:
            cuda_id = min(self.cfg.cuda_id, torch.cuda.device_count() - 1)
            major, minor = torch.cuda.get_device_capability(cuda_id)
            if (7, 0) <= (major, minor) < (12, 0):
                torch.cuda.set_device(cuda_id)
                return torch.device(f"cuda:{cuda_id}")
        return torch.device("cpu")

    def _get_scheduler(
        self, T_max: int, scheduler_type: str | None
    ) -> torch.optim.lr_scheduler.CosineAnnealingLR | None:
        if scheduler_type is None:
            return None
        match scheduler_type.lower():
            case "cosineannealing":
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.madnis.optimizer, T_max=T_max
                )
            case _:
                return None

    def _get_loss(self) -> Callable | None:
        try:
            from glnis.core.nn.losses import get_loss
        except ImportError:
            raise ImportError(
                "Loss functions are only supported in the 'glnis_gammaboard_api' package."
            )
        return get_loss(self.cfg.loss_type.lower())

    def _train_step(self) -> None:
        # print(f"Ingested zeros: {np.sum(np.isclose(self.training_status.ingested_weights, 0.0))} / {len(self.training_status.ingested_weights)}")

        madnis_samples = MadnisSampleBatch(
            x=self.training_status.generated_samples,
            y=None,
            q_sample=self.training_status.generated_probs,
            func_vals=self.training_status.ingested_weights * self.training_status.generated_probs,
            channels=None,
        )
        last_loss = self.madnis._optimization_step(madnis_samples)[0]
        if not np.isfinite(last_loss):
            last_loss = None
        self.training_status.add_loss(last_loss)

    def _madnis_discrete_prior_prob_function(
        self, indices: Tensor, dim: int = 0
    ) -> Tensor:
        """
        Implements a default flat prior.
        """
        num_disc_input = indices.shape[1]
        if num_disc_input == len(self._sampled_discrete_cardinalities):
            return torch.zeros(indices.shape, dtype=torch.float64, device=self.device)

        full_dim = int(self._sampled_discrete_indices[num_disc_input])
        if self.mapping is not None:
            mapping_start = (
                len(self.discrete_cardinalities)
                if self.cfg.condition_integrand_first else 0
            )
            mapping_end = mapping_start + len(self.mapping.discrete_cardinalities)
            if mapping_start <= full_dim < mapping_end:
                # Mapping priors still use the original dimension numbering and
                # expect a prefix with constant dimensions restored as zeros.
                full_indices = np.zeros((len(indices), full_dim), dtype=np.uint64)
                full_indices[:, self._sampled_discrete_indices[:num_disc_input]] = (
                    indices.numpy(force=True)
                )
                prior = self.mapping.discrete_prior_prob_function(
                    full_indices[:, mapping_start:], full_dim - mapping_start
                )
                return torch.from_numpy(prior.astype(np.float64)).to(device=self.device)

        disc_dim = self._sampled_discrete_cardinalities[num_disc_input]
        return torch.full(
            (len(indices), disc_dim),
            1.0 / disc_dim,
            dtype=torch.float64,
            device=self.device,
        )

    def _flat_discrete_prior_prob_function(
        self, indices: NDArray, dim: int = 0
    ) -> NDArray:
        """
        Implements a default flat prior for the integrand.
        """
        num_disc_input = indices.shape[1]
        if num_disc_input == len(self.discrete_cardinalities):
            return np.zeros(indices.shape, dtype=np.float64)

        disc_dim = self.discrete_cardinalities[num_disc_input]
        return np.full((len(indices), disc_dim), 1.0 / disc_dim, dtype=np.float64)

    def _madnis_eval(self, x_all: Tensor) -> Tensor:
        raise NotImplementedError(
            "This should not get called, since we are sidestepping the usual training process."
        )

    def _madnis_output_to_disc_cont(self, x_all: Tensor) -> Tuple[NDArray, NDArray]:
        n_discrete = len(self._sampled_discrete_cardinalities)
        if self.madnis.integrand.discrete_dims_position == "first":
            sampled_discrete = x_all[:, :n_discrete].numpy(force=True)
            continuous = x_all[:, n_discrete:].numpy(force=True)
        else:
            sampled_discrete = (
                x_all[:, -n_discrete:].numpy(force=True)
                if n_discrete
                else np.zeros((len(x_all), 0), dtype=np.float64)
            )
            continuous = (
                x_all[:, :-n_discrete].numpy(force=True)
                if n_discrete
                else x_all.numpy(force=True)
            )
        discrete = np.zeros(
            (len(x_all), self.num_discrete_dims), dtype=sampled_discrete.dtype
        )
        discrete[:, self._sampled_discrete_indices] = sampled_discrete
        return discrete, continuous

    def _get_madnis_integrand(self) -> Integrand:
        discrete_cardinalities = self._sampled_discrete_cardinalities
        return Integrand(
            function=self._madnis_eval,
            input_dim=self.continuous_dims + len(discrete_cardinalities),
            discrete_dims=discrete_cardinalities,
            discrete_dims_position=self.cfg.discrete_dims_position,
            discrete_prior_prob_function=self._madnis_discrete_prior_prob_function,
        )

    def _apply_mapping(self, discrete: NDArray, continuous: NDArray, weights: NDArray, inverse: bool = False) -> Tuple[NDArray, NDArray]:
        if self.mapping is not None:
            if self.cfg.condition_integrand_first:
                discrete_integrand, discrete = np.split(discrete, [len(self.discrete_cardinalities)], axis=1)
            else:
                discrete, discrete_integrand = np.split(discrete, [self.mapping.num_discrete_dims], axis=1)
            if inverse:
                return (discrete_integrand, *self.mapping.backward(discrete, continuous, weights))
            return (discrete_integrand, *self.mapping.forward(discrete, continuous, weights))
        return discrete, continuous, weights

    def _get_madnis_integrator(self) -> Integrator:
        return Integrator(
            self._get_madnis_integrand(),
            device=self.device,
            discrete_flow_kwargs=asdict(
                self.cfg.transformer_config
                if self.cfg.discrete_model == "transformer"
                else self.cfg.made_config
            ),
            loss=self._get_loss(),
            batch_size=self.cfg.training_batch_size,
            discrete_model=self.cfg.discrete_model,
            learning_rate=self.cfg.learning_rate,
            flow_kwargs=asdict(self.cfg.flow_config),
        )


class MadnisSamplerNaiveSgnStrat(MadnisSampler):
    CHECKPOINT_KIND = "sgn_strat_naive"

    def __init__(
        self,
        discrete_cardinalities: list[int],
        **kwargs
    ):
        discrete_cardinalities = [2] + discrete_cardinalities
        super().__init__(
            discrete_cardinalities=discrete_cardinalities,
            **kwargs,
        )
        if not self.cfg.condition_integrand_first:
            raise ValueError(
                "MadnisSamplerNaiveSgnStrat requires condition_integrand_first=True"
            )

    def generate(self, remaining_sample_budget: int | None) -> Any:
        batch = super().generate(remaining_sample_budget)
        if isinstance(batch, gb.GenerationStatus):
            return batch
        strata = batch.xs_discrete[:, 0]
        batch.xs_discrete = batch.xs_discrete[:, 1:]
        return batch, strata

    def _checkpoint_state(self) -> dict[str, Any]:
        state = super()._checkpoint_state()
        # The public constructor prepends the sign coordinate itself.
        state["discrete_cardinalities"] = state["discrete_cardinalities"][1:]
        return state

    def get_strat_cardinalities(self) -> list[int] | None:
        """
        Return the cardinalities of the stratified channels.
        """
        return [2]

    def strat_channel_transform(self, values: NDArray, strata: NDArray) -> NDArray:
        """
        Transform the values based on the stratification.
        To be called on the aggregator side if stratified sampling is used. 
        """
        values[strata == 0] = np.clip(values[strata == 0], a_min=0, a_max=None)
        values[strata == 1] = np.clip(values[strata == 1], a_min=None, a_max=0)
        return values

    # def pdf(self, xs_discrete: NDArray, xs_continuous: NDArray) -> NDArray | None:
    #     result = super().pdf(xs_discrete, xs_continuous)
    #     if result is not None:
    #         # An additional factor of 2 enters through the discrete prior, which we need to cancel
    #         result *= 2.0
    #     return result

class MadnisSamplerFullSgnStrat(MadnisSampler):
    CHECKPOINT_KIND = "sgn_strat_full"

    def __init__(
        self,
        *args,
        _checkpoint_children: list[dict[str, Any]] | None = None,
        _pending_strata: NDArray | None = None,
        **kwargs
    ):
        super().__init__(
            *args,
            _build_integrator=False,
            **kwargs
        )
        if not self.cfg.condition_integrand_first:
            raise ValueError(
                "MadnisSamplerFullSgnStrat requires condition_integrand_first=True"
            )
        if self.cfg.training_batch_size % 2 != 0:
            raise ValueError(
                "MadnisSamplerFullSgnStrat requires training_batch_size to be even"
            )
        if self.cfg.max_batch_size % 2 != 0:
            raise ValueError(
                "MadnisSamplerFullSgnStrat requires max_batch_size to be even"
            )
        del self.madnis
        if _checkpoint_children is None:
            children = []
            for index in range(2):
                child_config = asdict(self.cfg)
                child_config.update(
                    kind="default", training_batch_size=self.cfg.training_batch_size // 2,
                    max_batch_size=self.cfg.max_batch_size // 2,
                    seed=self.cfg.seed + index, cuda_id=self.cfg.cuda_id + index,
                )
                children.append(MadnisSampler(
                    discrete_cardinalities=self.discrete_cardinalities,
                    continuous_dims=self.continuous_dims,
                    cfg=MadnisConfig.from_dict(child_config),
                ))
        else:
            if len(_checkpoint_children) != 2:
                raise ValueError("Full sign-stratified checkpoints must contain two child samplers")
            children = []
            for index, child_state in enumerate(_checkpoint_children):
                child_config = dict(child_state["config"])
                child_config.update(
                    use_gpu=self.cfg.use_gpu, cuda_id=self.cfg.cuda_id + index,
                    training_steps=self.cfg.training_steps,
                    training_batch_size=self.cfg.training_batch_size // 2,
                    max_batch_size=self.cfg.max_batch_size // 2,
                    save_path=self.cfg.save_path,
                )
                children.append(MadnisSampler._from_checkpoint_state(
                    child_state, cfg=MadnisConfig.from_dict(child_config),
                ))
        self.madnis_1, self.madnis_2 = children
        self.pending_strata = np.array([] if _pending_strata is None else _pending_strata, dtype=np.int64, copy=True)
        if self.pending_strata.ndim != 1 or np.any((self.pending_strata < 0) | (self.pending_strata > 1)):
            raise ValueError("Checkpoint pending_strata must contain one-dimensional sign channel indices")
        if len(self.pending_strata) != self.training_status.num_generated - self.training_status.num_ingested:
            raise ValueError("Checkpoint pending strata do not match outstanding training feedback")
        self.discrete_cardinalities = [2] + self.discrete_cardinalities

    def _checkpoint_state(self) -> dict[str, Any]:
        return dict(
            self._checkpoint_common_state(),
            children=[self.madnis_1._checkpoint_state(), self.madnis_2._checkpoint_state()],
            pending_strata=self.pending_strata.copy(),
        )

    def _checkpoint_devices(self) -> set[torch.device]:
        return super()._checkpoint_devices() | self.madnis_1._checkpoint_devices() | self.madnis_2._checkpoint_devices()

    def generate(self, remaining_sample_budget: int | None) -> Any:
        training_remaining_1 = self.madnis_1.training_samples_remaining()
        training_remaining_2 = self.madnis_2.training_samples_remaining()
        training_remaining = self.training_samples_remaining()
        if training_remaining == 0:
            return gb.GenerationStatus.WAITING
        if training_remaining is not None:
            # A child may finish its window before its sibling. Until the
            # parent completes this step, don't draw from that child's next
            # training window (or from its inference distribution).
            if self.madnis_1.training_status.step > self.training_status.step:
                training_remaining_1 = 0
            if self.madnis_2.training_status.step > self.training_status.step:
                training_remaining_2 = 0
        nr_samples = remaining_sample_budget if remaining_sample_budget is not None else self.cfg.max_batch_size
        if training_remaining is not None:
            nr_samples = min(nr_samples, training_remaining)
        nr_samples_1 = min(nr_samples, training_remaining_1 if training_remaining_1 is not None else self.madnis_1.cfg.max_batch_size)
        nr_samples_2 = min(nr_samples, training_remaining_2 if training_remaining_2 is not None else self.madnis_2.cfg.max_batch_size)
        nr_samples_2 = min(nr_samples_2, nr_samples - nr_samples_1)
        batches = []
        if nr_samples_1 > 0:
            batches.append(self.madnis_1.generate(nr_samples_1))
        if nr_samples_2 > 0:
            batches.append(self.madnis_2.generate(nr_samples_2))
        discrete = np.vstack([batch.xs_discrete for batch in batches])
        continuous = np.vstack([batch.xs_continuous for batch in batches])
        weights = np.hstack([batch.weights for batch in batches]) * 2.0
        strata = np.hstack([np.zeros(nr_samples_1, dtype=np.int64), np.ones(nr_samples_2, dtype=np.int64)])

        if training_remaining is not None:
            x_all = np.hstack([discrete[:, self._sampled_discrete_indices], continuous])
            prob = 1 / weights
            self.training_status.add_generation_data(
                torch.as_tensor(x_all, dtype=self.training_status.generated_samples.dtype, device=self.training_status.generated_samples.device), 
                torch.as_tensor(prob, dtype=self.training_status.generated_probs.dtype, device=self.training_status.generated_probs.device)
                )
            self._add_generation_strata(strata)
        return gb.SampleBatch(xs_discrete=discrete, xs_continuous=continuous, weights=weights, training_remaining=training_remaining), strata
    
    def feedback(self, values: NDArray) -> None:
        if self.training_status.step >= self.cfg.training_steps:
            gb.log(f"Training target of {self.cfg.training_steps * self.cfg.training_batch_size} samples already reached. Ignoring additional training values.", level="warn")
            return
        values = np.asarray(values)
        self.training_status.add_feedback_data(values)
        feedback_strata = self._get_feedback_strata(len(values))
        values_1 = values[feedback_strata == 0]
        values_2 = values[feedback_strata == 1]
        if len(values_1) > 0:
            self.madnis_1.feedback(values_1)
        if len(values_2) > 0:
            self.madnis_2.feedback(values_2)

        if self.training_status.test_ready_for_training():
            self._train_step()
        

    def training_samples_remaining(self) -> int | None:
        # Parent counters describe the current composite window even when one
        # child has already advanced to its next window after partial feedback.
        return super().training_samples_remaining()

    def get_strat_cardinalities(self) -> list[int] | None:
        """
        Return the cardinalities of the stratified channels.
        """
        return [2]

    def strat_channel_transform(self, values: NDArray, strata: NDArray) -> NDArray:
        """
        Transform the values based on the stratification.
        To be called on the aggregator side if stratified sampling is used. 
        """
        values[strata == 0] = np.clip(values[strata == 0], a_min=0, a_max=None)
        values[strata == 1] = np.clip(values[strata == 1], a_min=None, a_max=0)
        return values

    def pdf(self, xs_discrete: NDArray, xs_continuous: NDArray) -> NDArray | None:
        """
        Return per-sample PDF values if supported.

        Return a float64 array with shape (nr_samples,) or None to signal that
        the sampler does not support/doesn't provide a PDF for the given batch.
        """
        strata, xs_discrete = np.split(xs_discrete, [1], axis=1)
        strata = strata.flatten()
        result = np.empty((len(xs_discrete),), dtype=np.float64)
        n_samples_1 = np.sum(strata == 0)
        n_samples_2 = np.sum(strata == 1)
        if n_samples_1 > 0:
            pdf_1 = self.madnis_1.pdf(xs_discrete[strata == 0], None if xs_continuous is None else xs_continuous[strata == 0])
        else:
            pdf_1 = np.empty((0,), dtype=np.float64)
        if n_samples_2 > 0:
            pdf_2 = self.madnis_2.pdf(xs_discrete[strata == 1], None if xs_continuous is None else xs_continuous[strata == 1])
        else:
            pdf_2 = np.empty((0,), dtype=np.float64)
        result[strata == 0] = pdf_1
        result[strata == 1] = pdf_2

        # The parent sampler's PDF is the average of the two children, since each child is responsible for half of the samples.
        return result / 2.0

    def _train_step(self) -> None:
        last_loss_1 = self.madnis_1.training_status.last_loss
        last_loss_2 = self.madnis_2.training_status.last_loss
        if last_loss_1 is None or last_loss_2 is None:
            last_loss = None
        else:
            last_loss = (last_loss_1 + last_loss_2) / 2
        self.training_status.add_loss(last_loss)

    def _add_generation_strata(self, strata: NDArray) -> None:
        self.pending_strata = np.hstack([self.pending_strata, strata])

    def _get_feedback_strata(self, n_feedback: int) -> NDArray:
        if len(self.pending_strata) < n_feedback:
            raise ValueError(
                f"Not enough pending strata to provide feedback for {n_feedback} samples."
            )
        feedback_strata = self.pending_strata[:n_feedback]
        self.pending_strata = self.pending_strata[n_feedback:]
        return feedback_strata
    
