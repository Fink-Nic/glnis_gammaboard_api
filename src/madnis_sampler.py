# type: ignore
from __future__ import annotations

import numpy as np
import torch
from typing import Any, Callable, Literal, List, Tuple, Dict
from numpy.typing import NDArray
from torch._tensor import Tensor
from dataclasses import dataclass, asdict, field

from gammaboard_process import SampleBatch, Sampler
from madnis.integrator import Integrator, Integrand, losses
from madnis.integrator import SampleBatch as MadnisSampleBatch


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
    training_steps: int = 100
    training_batch_size: int = 1000
    max_batch_size: int = 100_000
    use_gpu: bool = True
    cuda_id: int = 1
    learning_rate: float = 1e-3
    use_scheduler: bool = True
    save_path: str | None = None
    scheduler_type: Literal["cosineannealing"] = "cosineannealing"
    loss_type: Literal["variance", "variance_softclip",
                       "kl_divergence", "kl_divergence_softclip"] = "kl_divergence"
    discrete_dims_position: Literal["first", "last"] = "first"
    discrete_model: Literal["transformer", "made"] = "transformer"
    flow_config: FlowConfig = field(default_factory=FlowConfig)
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    made_config: MadeConfig = field(default_factory=MadeConfig)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> MadnisConfig:
        flow_config = FlowConfig(**config_dict.get("flow_config", {}))
        transformer_config = TransformerConfig(**config_dict.get("transformer_config", {}))
        made_config = MadeConfig(**config_dict.get("made_config", {}))
        return cls(
            seed=config_dict.get("seed", 42),
            training_steps=config_dict.get("training_steps", 100),
            training_batch_size=config_dict.get("training_batch_size", 1000),
            max_batch_size=config_dict.get("max_batch_size", 100_000),
            use_gpu=config_dict.get("use_gpu", True),
            cuda_id=config_dict.get("cuda_id", 1),
            learning_rate=config_dict.get("learning_rate", 1e-3),
            use_scheduler=config_dict.get("use_scheduler", True),
            save_path=config_dict.get("save_path", None),
            scheduler_type=config_dict.get("scheduler_type", "cosineannealing"),
            loss_type=config_dict.get("loss_type", "kl_divergence"),
            discrete_dims_position=config_dict.get("discrete_dims_position", "first"),
            discrete_model=config_dict.get("discrete_model", "transformer"),
            flow_config=flow_config,
            transformer_config=transformer_config,
            made_config=made_config,
        )


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


class MadnisSampler(Sampler):
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
        loss_type: Literal["variance", "variance_softclip",
                           "kl_divergence", "kl_divergence_softclip"] = "kl_divergence",
        discrete_dims_position: Literal["first", "last"] = "first",
        discrete_model: Literal["transformer", "made"] = "transformer",
        flow_config: dict[str, Any] | FlowConfig | None = None,
        transformer_config: dict[str, Any] | TransformerConfig | None = None,
        made_config: dict[str, Any] | MadeConfig | None = None,
        trained_samples: int | None = None,
        total_trained_samples: int | None = None,
        produced_batches: int | None = None,
        produced_samples: int | None = None,
        step: int | None = None,
        last_loss: float | None = None,
        pending_weights: List[NDArray] | None = None,
        pending_training_samples: List[Tensor] | None = None,
        pending_training_probs: List[Tensor] | None = None,
        torch_cpu_rng_state: Tensor | None = None,
        torch_gpu_rng_state: Tensor | None = None,
        madnis_blob: bytes | None = None,
        evaluator_metadata: dict[str, Any] | None = None,
        parameterisation: dict[str, Any] | None = None,
        graph_properties: dict[str, Any] | None = None,
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
        )

        self.discrete_cardinalities: List[int] = discrete_cardinalities
        self.continuous_dims: int = continuous_dims
        self.evaluator_metadata: dict[str, Any] = evaluator_metadata or {}
        if parameterisation is not None:
            if isinstance(parameterisation, dict):
                parameterisation = [parameterisation]
            try:
                from glnis.core.parser import MetaDataParser
            except ImportError:
                raise ImportError(
                    "Momentum space evaluation is only supported in the 'glnis_gammaboard_api' package.")
            Parser = MetaDataParser(madnis_config=self.cfg, metadata=self.evaluator_metadata,
                                    graph_properties=graph_properties)
            self.transform = Parser.get_layered_parameterisation_instance()
        else:
            self.transform = None

        if self.transform is not None:
            self.num_discrete_dims = len(self.transform.discrete_dims) + len(self.discrete_cardinalities)
        else:
            self.num_discrete_dims = len(self.discrete_cardinalities)

        if not isinstance(self.discrete_cardinalities, list) or any(
                cardinality <= 0 for cardinality in self.discrete_cardinalities):
            raise TypeError("discrete_cardinalities must be a list of positive integers")
        self.discrete_cardinalities = [int(cardinality) for cardinality in self.discrete_cardinalities]
        if self.continuous_dims <= 0:
            raise ValueError("continuous_dims must be > 0")

        self.device = self._get_device()
        self.step: int = step or 0
        self.last_loss: float | None = last_loss or None

        self.training_target_samples = self.cfg.training_steps * self.cfg.training_batch_size
        self.trained_samples: int = trained_samples or 0
        self.total_trained_samples: int = total_trained_samples or 0
        self.produced_batches: int = produced_batches or 0
        self.produced_samples: int = produced_samples or 0

        self.pending_weights: List[NDArray] = pending_weights or []
        self.pending_training_samples: List[Tensor] = pending_training_samples or []
        self.pending_training_probs: List[Tensor] = pending_training_probs or []
        if torch_cpu_rng_state is not None:
            if torch_gpu_rng_state is not None and self.device.type == 'cuda':
                torch.cuda.set_rng_state(torch_gpu_rng_state)
            torch.set_rng_state(torch_cpu_rng_state)
        else:
            torch.manual_seed(self.cfg.seed)
            torch.cuda.manual_seed(self.cfg.seed)
        if madnis_blob is not None:
            import io
            buffer = io.BytesIO(madnis_blob)
            self.madnis: Integrator = torch.load(buffer, map_location=self.device, weights_only=False)
        else:
            self.madnis: Integrator = self._get_madnis_integrator()

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

        save_path = snapshot.get("save_path")
        if save_path is None:
            raise ValueError("Snapshot is missing 'save_path' key required for loading the Integrator state.")
        try:
            import pickle
            with open(save_path, 'rb') as f:
                state: Dict[str, Any] = pickle.load(f)

            parameterisation = init_args.pop("parameterisation", None) if init_args else None
            graph_properties = init_args.pop("graph_properties", None) if init_args else None

            instance = cls(
                discrete_cardinalities=discrete_cardinalities,
                continuous_dims=int(continuous_dims),
                cfg=MadnisConfig.from_dict(init_args or {}),
                trained_samples=snapshot.get("trained_samples"),
                total_trained_samples=snapshot.get("total_trained_samples"),
                produced_batches=snapshot.get("produced_batches"),
                produced_samples=snapshot.get("produced_samples"),
                step=snapshot.get("step"),
                last_loss=snapshot.get("last_loss"),
                pending_weights=state.get("pending_weights"),
                pending_training_samples=state.get("pending_training_samples"),
                pending_training_probs=state.get("pending_training_probs"),
                torch_cpu_rng_state=state.get("torch_cpu_rng_state"),
                torch_gpu_rng_state=state.get("torch_gpu_rng_state"),
                madnis_blob=state.get("madnis_blob"),
                evaluator_metadata=evaluator_metadata,
                parameterisation=parameterisation,
                graph_properties=graph_properties
            )
            instance.madnis.integrand = instance._get_madnis_integrand()
            ch_remap = (
                None
                if instance.madnis.group_channels and not instance.madnis.group_channels_uniform
                else instance.madnis.integrand.remap_channels
            )
            instance.madnis.loss = instance._get_loss()
            if hasattr(instance.madnis.flow, 'channel_remap_function'):
                instance.madnis.flow.channel_remap_function = ch_remap
            if hasattr(instance.madnis.flow, 'continuous_flow') and instance.madnis.flow.continuous_flow is not None:
                instance.madnis.flow.continuous_flow.channel_remap_function = ch_remap
            if hasattr(instance.madnis.flow, 'discrete_flow') and instance.madnis.flow.discrete_flow is not None:
                instance.madnis.flow.discrete_flow.prior_prob_function = instance._madnis_discrete_prior_prob_function
                if hasattr(instance.madnis.flow.discrete_flow, 'channel_remap_function'):
                    instance.madnis.flow.discrete_flow.channel_remap_function = ch_remap
        except Exception as e:
            raise RuntimeError(f"Failed to load MadNIS Integrator state from file '{save_path}': {e}")

        return instance

    def snapshot(self) -> Dict[str, Any]:
        if self.madnis is None:
            raise RuntimeError("MadnisSampler not properly initialized with an Integrator instance.")

        if self.cfg.save_path is None:
            return dict(Warning="No save path provided. No snapshot was created.")

        tmp_integrand = self.madnis.integrand
        tmp_loss = self.madnis.loss
        tmp_ch_remap = (
            None
            if self.madnis.group_channels and not self.madnis.group_channels_uniform
            else tmp_integrand.remap_channels
        )
        try:
            self.madnis.integrand = None
            self.madnis.loss = None
            self.madnis.flow.prior_prob_function = None
            if hasattr(self.madnis.flow, 'channel_remap_function'):
                self.madnis.flow.channel_remap_function = None
            if hasattr(self.madnis.flow, 'continuous_flow') and self.madnis.flow.continuous_flow is not None:
                self.madnis.flow.continuous_flow.channel_remap_function = None
            if hasattr(self.madnis.flow, 'discrete_flow') and self.madnis.flow.discrete_flow is not None:
                self.madnis.flow.discrete_flow.prior_prob_function = None
                if hasattr(self.madnis.flow.discrete_flow, 'channel_remap_function'):
                    self.madnis.flow.discrete_flow.channel_remap_function = None

            from pathlib import Path
            import pickle
            import io
            Path(self.cfg.save_path).parent.mkdir(parents=True, exist_ok=True)
            buffer = io.BytesIO()
            torch.save(self.madnis, buffer)
            with open(self.cfg.save_path, 'wb') as f:
                pickle.dump(dict(
                    pending_weights=self.pending_weights,
                    pending_training_samples=self.pending_training_samples,
                    pending_training_probs=self.pending_training_probs,
                    torch_cpu_rng_state=torch.get_rng_state(),
                    torch_gpu_rng_state=torch.cuda.get_rng_state() if self.device.type == 'cuda' else None,
                    madnis_blob=buffer.getvalue(),
                ), f)

        except Exception as e:
            raise RuntimeError(f"Failed to save MadNIS Integrator state to file '{self.cfg.save_path}': {e}")
        finally:
            self.madnis.integrand = tmp_integrand
            self.madnis.loss = tmp_loss
            self.madnis.flow.prior_prob_function = self._madnis_discrete_prior_prob_function
            if hasattr(self.madnis.flow, 'channel_remap_function'):
                self.madnis.flow.channel_remap_function = tmp_ch_remap
            if hasattr(self.madnis.flow, 'continuous_flow') and self.madnis.flow.continuous_flow is not None:
                self.madnis.flow.continuous_flow.channel_remap_function = tmp_ch_remap
            if hasattr(self.madnis.flow, 'discrete_flow') and self.madnis.flow.discrete_flow is not None:
                self.madnis.flow.discrete_flow.prior_prob_function = self._madnis_discrete_prior_prob_function
                if hasattr(self.madnis.flow.discrete_flow, 'channel_remap_function'):
                    self.madnis.flow.discrete_flow.channel_remap_function = tmp_ch_remap
        snapshot: Dict[str, int | str | float] = dict(
            trained_samples=self.trained_samples,
            total_trained_samples=self.total_trained_samples,
            produced_batches=self.produced_batches,
            produced_samples=self.produced_samples,
            step=self.step,
            save_path=self.cfg.save_path,
        )
        if self.last_loss is not None:
            snapshot["last_loss"] = self.last_loss
        return snapshot

    def sample_plan(self) -> Dict[str, Any]:
        n_batch_remaining = self.training_samples_remaining()
        return dict(
            kind="produce",
            nr_samples=n_batch_remaining if n_batch_remaining is not None else self.cfg.max_batch_size,
        )

    def training_samples_remaining(self) -> int | None:
        if self.total_trained_samples >= self.training_target_samples:
            return None
        if self.step < self.cfg.training_steps:
            return max(self.cfg.training_batch_size - self.trained_samples, 0)
        return None

    def produce_latent_batch(self, nr_samples: int) -> SampleBatch:
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
            discrete[n_eval:n_eval+n, :], continuous[n_eval:n_eval+n, :] = self._madnis_output_to_disc_cont(x_all)
            wgt[n_eval:n_eval+n] = 1 / prob.numpy(force=True)
            n_eval += n
            if self.training_samples_remaining() is not None:
                self.pending_training_samples.append(x_all)
                self.pending_training_probs.append(prob)
                self.trained_samples += n
                self.total_trained_samples += n

        if self.transform is not None:
            discrete, continuous, wgt = self.transform.parameterise(discrete, continuous, wgt)
        self.produced_batches += 1
        self.produced_samples += nr_samples

        return SampleBatch(xs_discrete=discrete, xs_continuous=continuous, weights=wgt)

    def ingest_training_values(self, training_values: NDArray) -> None:
        training_values = np.asarray(training_values)
        n_samples = training_values.shape[0]
        self.pending_weights.append(training_values)

        if self.trained_samples >= self.cfg.training_batch_size:
            self._train_step()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Optional runtime diagnostics. Empty dict means no diagnostics available."""
        diagnostics: Dict[str, Any] = dict(
            produced_batches=self.produced_batches,
            produced_samples=self.produced_samples,
            total_trained_samples=self.total_trained_samples,
        )
        if self.last_loss is not None:
            diagnostics["loss"] = self.last_loss
        if self.gammaloop_metadata is not None:
            diagnostics["gammaloop_metadata"] = asdict(self.gammaloop_metadata)
        return diagnostics

    def pdf(
        self, xs_discrete: NDArray, xs_continuous: NDArray
    ) -> NDArray | None:
        """Return per-sample PDF values if supported.

        Return a float64 array with shape (nr_samples,) or None to signal that
        the sampler does not support/doesn't provide a PDF for the given batch.
        """
        if self.transform is not None:
            raise NotImplementedError("PDF evaluation is not supported when using a parameterisation.")

        n_samples = len(xs_discrete)
        if xs_continuous is None:
            xs_continuous = np.zeros((n_samples, 0), dtype=xs_discrete.dtype)
        if self.madnis.integrand.discrete_dims_position == "first":
            x_all = np.hstack([xs_discrete, xs_continuous])
        elif self.madnis.integrand.discrete_dims_position == "last":
            x_all = np.hstack([xs_continuous, xs_discrete])
        else:
            raise ValueError(f"Invalid discrete_dims_position: {self.madnis.integrand.discrete_dims_position}")
        prob = np.empty((n_samples,), dtype=np.float64)
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
                    prob[n_eval:n_eval+n] = self.madnis.flow.prob(
                        x_all[n_eval:n_eval+n, :]).numpy(force=True).reshape(-1)
                else:
                    prob[n_eval:n_eval+n] = self.madnis.flow.discrete_flow.prob(
                        x_all[n_eval:n_eval+n, :]).numpy(force=True).reshape(-1)
            n_eval += n
        return prob

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available() and self.cfg.use_gpu:
            cuda_id = min(self.cfg.cuda_id, torch.cuda.device_count() - 1)
            major, minor = torch.cuda.get_device_capability(cuda_id)
            if (7, 0) <= (major, minor) < (12, 0):
                torch.cuda.set_device(cuda_id)
                return torch.device(f'cuda:{cuda_id}')
        return torch.device('cpu')

    def _get_scheduler(self, T_max: int, scheduler_type: str | None
                       ) -> torch.optim.lr_scheduler.CosineAnnealingLR | None:
        if scheduler_type is None:
            return None
        match scheduler_type.lower():
            case 'cosineannealing':
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.madnis.optimizer, T_max=T_max)
            case _:
                return None

    def _get_loss(self) -> Callable | None:
        try:
            from glnis.core.nn.losses import get_loss
        except ImportError:
            raise ImportError("Loss functions are only supported in the 'glnis_gammaboard_api' package.")
        return get_loss(self.cfg.loss_type.lower())

    def _train_step(self) -> None:
        f_lens = [len(w) for w in self.pending_weights]
        x_lens = [len(s) for s in self.pending_training_samples]
        p_lens = [len(p) for p in self.pending_training_probs]
        if not (f_lens == x_lens == p_lens):
            raise RuntimeError(
                f"Mismatch in pending training data lengths: integrand weights {f_lens}, samples {x_lens}, probs {p_lens}.")

        func_vals = torch.cat([torch.from_numpy(w).to(device=self.device, dtype=torch.float64)
                               for w in self.pending_weights])
        x_all = torch.cat(self.pending_training_samples, dim=0)
        probs = torch.cat(self.pending_training_probs, dim=0)
        madnis_samples = MadnisSampleBatch(
            x=x_all,
            y=None,
            q_sample=probs,
            func_vals=func_vals,
            channels=None,
        )
        self.last_loss = self.madnis._optimization_step(madnis_samples)[0]
        if self.madnis.scheduler is not None:
            self.madnis.scheduler.step()
        self.madnis.step += 1
        self.step += 1
        self.trained_samples = 0
        self.pending_training_samples.clear()
        self.pending_training_probs.clear()
        self.pending_weights.clear()

    def _madnis_discrete_prior_prob_function(self, indices: Tensor, dim: int = 0) -> Tensor:
        """
        Implements a default flat prior.
        """
        num_disc_input = indices.shape[1]
        if num_disc_input == self.num_discrete_dims:
            return torch.zeros(indices.shape, dtype=torch.float64, device=self.device)

        if self.transform is not None:
            if self.transform.condition_integrand_first:
                n_dim = len(self.discrete_cardinalities)
                prior1: Callable = self._flat_discrete_prior_prob_function
                prior2: Callable = self.transform.discrete_prior_prob_function
            else:
                n_dim = len(self.param.discrete_dims)
                prior1: Callable = self.transform.discrete_prior_prob_function
                prior2: Callable = self._flat_discrete_prior_prob_function

            if dim < n_dim:
                return torch.from_numpy(prior1(indices, dim).astype(np.float64)).to(device=self.device)

            return torch.from_numpy(prior2(indices[:, n_dim:], dim - n_dim).astype(np.float64)).to(device=self.device)

    def _flat_discrete_prior_prob_function(self, indices: NDArray, dim: int = 0) -> NDArray:
        """
        Implements a default flat prior for the integrand.
        """
        num_disc_input = indices.shape[1]
        if num_disc_input == len(self.discrete_cardinalities):
            return np.zeros(indices.shape, dtype=np.float64)

        disc_dim = self.discrete_cardinalities[num_disc_input]
        return np.full((len(indices), disc_dim), 1.0 / disc_dim, dtype=np.float64)

    def _madnis_eval(self, x_all: Tensor) -> Tensor:
        raise NotImplementedError("This should not get called, since we are sidestepping the usual training process.")

    def _madnis_output_to_disc_cont(self, x_all: Tensor) -> Tuple[NDArray, NDArray]:
        if self.madnis.integrand.discrete_dims_position == "first":
            discrete = x_all[:, :self.num_discrete_dims].numpy(force=True)
            continuous = x_all[:, self.num_discrete_dims:].numpy(force=True)
        else:
            discrete = x_all[:, -self.num_discrete_dims:].numpy(force=True)
            continuous = x_all[:, :-self.num_discrete_dims].numpy(force=True)
        return discrete, continuous

    def _get_madnis_integrand(self) -> Integrand:
        if self.transform is not None:
            if self.transform.condition_integrand_first:
                discrete_dims = self.discrete_cardinalities + self.transform.discrete_dims
            else:
                discrete_dims = self.transform.discrete_dims + self.discrete_cardinalities
        else:
            discrete_dims = self.discrete_cardinalities
        return Integrand(
            function=self._madnis_eval,
            input_dim=self.continuous_dims + len(discrete_dims),
            discrete_dims=discrete_dims,
            discrete_dims_position=self.cfg.discrete_dims_position,
            discrete_prior_prob_function=self._madnis_discrete_prior_prob_function,
        )

    def _get_madnis_integrator(self) -> Integrator:
        return Integrator(
            self._get_madnis_integrand(),
            device=self.device,
            discrete_flow_kwargs=asdict(
                self.cfg.transformer_config if self.cfg.discrete_model == "transformer"
                else self.cfg.made_config),
            loss=self._get_loss(),
            batch_size=self.cfg.training_batch_size,
            discrete_model=self.cfg.discrete_model,
            learning_rate=self.cfg.learning_rate,
            flow_kwargs=asdict(self.cfg.flow_config),
        )
