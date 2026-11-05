# type: ignore
from __future__ import annotations

from typing import Any, Literal, get_args, get_origin, get_type_hints
from types import UnionType
from numpy.typing import NDArray
from torch._tensor import Tensor
from abc import ABC, abstractmethod

from dataclasses import dataclass, field, fields, is_dataclass, asdict
import io

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from gammaboard_process import Domain
from madnis.nn import Flow, MixedFlow, DiscreteMADE, DiscreteTransformer
from glnis.core.config import BuildFromKind, Config


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



class SamplerModule(ABC, nn.Module, BuildFromKind):

    """
    A base class for sampler modules.
    """
    KIND = "SamplerModuleBase"
    DEFAULT_KIND = "madnis rectangular"
    

    def __init__(
            self,
            domain: Domain,
            ):
        super().__init__()
        self.domain = domain

    @classmethod
    def from_snapshot(cls, snapshot: bytes):
        buffer = io.BytesIO(snapshot)
        return torch.load(buffer, map_location="cpu", weights_only=False)

    @abstractmethod
    def sample(self, n_samples: int, c: Tensor | None = None, return_prob: bool = False) -> Tensor:
        """Sample from the model.

        Args:
            n_samples: Number of samples to draw.
            c: Optional conditioning tensor.
            return_prob: Whether to return the probability of the samples.

        """
        pass

    @abstractmethod
    def pdf(self, xs_discrete: NDArray, xs_continuous: NDArray, c: NDArray | None = None) -> NDArray | None:
        """Evaluate the probability density function (PDF) of the model at given discrete and continuous points.

        Args:
            xs_discrete: Discrete input points.
            xs_continuous: Continuous input points.
            c: Optional conditioning tensor.

        Returns:
            The probability density function values at the given points, or None if not applicable.

        """
        pass

    @abstractmethod
    def prob(self, xs: Tensor, c: Tensor | None = None) -> Tensor:
        """
        Evaluate the model in reverse and return the probability density of each point in output space.
        Use during training with gradients enabled.

        Args:
            xs: Previously generated output of the model.
            c: Optional conditioning tensor.

        Returns:
            The probability dnesity function values at the given points, including gradients.
        
        """
        pass

    @abstractmethod
    def discrete_mass(self, subspace: NDArray, c: NDArray | None = None) -> NDArray | None:
        """Evaluate the discrete mass of the model over specified subspaces.

        Args:
            subspaces: Discrete subspace to evaluate.
            conditionals: Optional conditioning tensor.

        Returns:
            The discrete mass values for the specified subspace, or None if not applicable.

        """
        pass

    @abstractmethod
    def to_disc_cont(self, xs: Tensor) -> tuple[NDArray, NDArray]:
        """
        Transforms the sampled output Tensor to (xs_discrete, xs_continuous) NDArrays of expected shape.
        """
        pass

    def snapshot(self) -> bytes:
        buffer = io.BytesIO()
        torch.save(self, buffer)
        return buffer.getvalue()



class RectangularSamplerModule(SamplerModule):
    """
    A base class for rectangular sampler modules.
    """
    KIND = "RectangularModuleBase"
    DEFAULT_KIND = "madnis rectangular"

    def __init__(
            self,
            domain: Domain,
            dims_c: int = 0,
            ):
        super().__init__(domain=domain)
        self.dims_c = dims_c
        try:
            self.discrete_cardinalities, self.continuous_dims = self.domain.require_rectangular_shape()
        except ValueError as e:
            raise ValueError(f"Rectangular Sampler {self.__class__} requires a rectangular domain.") from None


@dataclass
class FlowConfig(Config):
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
class TransformerConfig(Config):
    """
    Config for the Transformer module which generates the discrete samples.
    """

    embedding_dim: int = 64
    feedforward_dim: int = 64
    heads: int = 4
    mlp_units: int = 64
    transformer_layers: int = 2


@dataclass
class MadeConfig(Config):
    """
    Config for the MADE module which generates the discrete samples. Alternative to Transformer (but worse).
    """

    layers: int = 3
    nodes_per_feature: int = 64

@dataclass
class MadnisConfig(Config):
    """
    General config.
    Args:
        discrete_dims_position:
            Whether the sampler generates the discrete points before or after the continuous. Conditions the second on the first.
        discrete_model:
            Whether to use a Transformer or MADE for the discrete flow.
        flow_config:
            See ``FlowConfig`` dataclass for details
        transformer_config:
            See ``TransformerConfig`` dataclass for details
        made_config:
            See ``MadeConfig`` dataclass for details
    """

    discrete_dims_position: Literal["first", "last"] = "first"
    freeze_discrete_model: bool = False
    discrete_model: Literal["transformer", "made"] = "transformer"
    flow_config: FlowConfig = field(default_factory=FlowConfig)
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    made_config: MadeConfig = field(default_factory=MadeConfig)

    def __post_init__(self) -> None:
        _validate_config(self)
        if self.transformer_config.embedding_dim % self.transformer_config.heads:
            raise ValueError("sampler.transformer_config.embedding_dim must be divisible by heads")

    @classmethod
    def from_kwargs(cls, **kwargs: dict[str, Any]) -> MadnisConfig:
        if not isinstance(kwargs, dict):
            raise ValueError("sampler must be a configuration table")
        unknown = set(kwargs) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown sampler settings: {', '.join(f'sampler.{key}' for key in sorted(unknown))}")
        values = dict(kwargs)
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

    @property
    def discrete_kwargs(self) -> dict[str, Any]:
        return self.transformer_config.to_dict() if self.discrete_model == "transformer" else self.made_config.to_dict()

    @property
    def flow_kwargs(self) -> dict[str, Any]:
        return self.flow_config.to_dict()


class MadnisSamplerModule(RectangularSamplerModule):
    KIND = "madnis rectangular"
    DEFAULT_KIND = "madnis rectangular"

    def __init__(
        self,
        domain: Domain,
        dims_c: int = 0,
        cfg : MadnisConfig | None = None,
        **kwargs
    ):
        super().__init__(domain, dims_c)
        self.cfg = cfg if cfg is not None else MadnisConfig.from_kwargs(**kwargs)
        
        if len(self.discrete_cardinalities) == 0:
            flow = Flow(
                dims_in=self.continuous_dims,
                dims_c=self.dims_c,
                **self.cfg.flow_kwargs,
            )
        elif self.continuous_dims == 0:
            if self.cfg.discrete_model == "made":
                flow = DiscreteMADE(
                    dims_in=self.discrete_cardinalities,
                    dims_c=self.dims_c,
                    prior_prob_function=None,
                    **self.cfg.discrete_kwargs,
                )
            elif self.cfg.discrete_model == "transformer":
                flow = DiscreteTransformer(
                    dims_in=self.discrete_cardinalities,
                    dims_c=self.dims_c,
                    prior_prob_function=None,
                    **self.cfg.discrete_kwargs,
                )
            else:
                raise ValueError("discrete_model must be 'made' or 'transformer'")
        else:
            flow = MixedFlow(
                dims_in_continuous=self.continuous_dims,
                dims_in_discrete=self.discrete_cardinalities,
                dims_c=self.dims_c,
                discrete_dims_position=self.cfg.discrete_dims_position,
                discrete_model=self.cfg.discrete_model,
                continuous_kwargs=self.cfg.flow_kwargs,
                discrete_kwargs=self.cfg.discrete_kwargs,
            )
        self.flow = flow

        # Dummy to determine device and dtype
        self.register_buffer("dummy", torch.zeros((1,)))

        if self.cfg.freeze_discrete_model and hasattr(self.flow, "discrete_flow"):
            for param in self.flow.discrete_flow.parameters():
                param.requires_grad = False

    def sample(self, n_samples: int, c: Tensor | None = None, return_prob: bool = False) -> Tensor:
        """Sample from the flow model.

        Args:
            n_samples: Number of samples to draw.
            c: Optional conditioning tensor.
            return_prob: Whether to return the probability of the samples.

        Returns:
            A tensor of samples drawn from the model, optionally with their probabilities.

        """
        return self.flow.sample(n=n_samples, c=c, return_prob=return_prob, device=self.dummy.device, dtype=self.dummy.dtype)

    def to_disc_cont(self, xs: Tensor) -> tuple[NDArray, NDArray]:
        """Split a sampled tensor into discrete and continuous parts based on the flow's configuration."""
        if isinstance(self.flow, MixedFlow):
            if self.flow.discrete_dims_first:
                xs_discrete = xs[:, :self.flow.dims_in_discrete]
                xs_continuous = xs[:, self.flow.dims_in_discrete:]
            else:
                xs_discrete = xs[:, -self.flow.dims_in_discrete:]
                xs_continuous = xs[:, :-self.flow.dims_in_discrete]
            return xs_discrete.numpy(force=True), xs_continuous.numpy(force=True)
        elif isinstance(self.flow, Flow):
            return np.zeros((xs.shape[0], 0), dtype=np.uint64), xs.numpy(force=True)
        elif isinstance(self.flow, (DiscreteMADE, DiscreteTransformer)):
            return xs.numpy(force=True), np.zeros((xs.shape[0], 0), dtype=np.float64)
        else:
            raise ValueError("Unsupported flow type for splitting into discrete and continuous parts.")

    def pdf(self, xs_discrete: NDArray, xs_continuous: NDArray, c: NDArray | None = None) -> NDArray | None:
        try:
            with torch.no_grad():
                xs_tensor = self._to_xs_tensor(xs_discrete, xs_continuous)
                probs = self.flow.prob(xs_tensor, c=c)
                return probs.numpy(force=True).ravel()
        except:
            return None

    def prob(self, xs: Tensor, c: Tensor | None = None) -> Tensor:
        return self.flow.prob(xs, c)

    def discrete_mass(self, subspace: NDArray, c: NDArray | None = None) -> float | None:
        subspace = torch.as_tensor(subspace, device=self.dummy.device, dtype=torch.int64)
        if c is not None:
            c = torch.as_tensor(c, device=self.dummy.device, dtype=self.dummy.dtype)
        try:
            with torch.no_grad():
                if not isinstance(self.flow, MixedFlow):
                    res = self.flow.prob(subspace, c=c) # In the case of a discrete-only flow
                res = self.flow.discrete_flow.prob(subspace, c=c)
        except Exception as e:
            raise Exception(f"Error in discrete_mass: {e}")
        return res.item() if res is not None else None

    def _to_xs_tensor(self, xs_discrete: NDArray, xs_continuous: NDArray) -> Tensor:
        """Combine discrete and continuous tensors into a single tensor based on the flow's configuration."""
        if isinstance(self.flow, MixedFlow):
            if self.flow.discrete_dims_first:
                xs = np.hstack([xs_discrete, xs_continuous])
            else:
                xs = np.hstack([xs_continuous, xs_discrete])
        elif isinstance(self.flow, Flow):
            xs = xs_continuous
        elif isinstance(self.flow, (DiscreteMADE, DiscreteTransformer)):
            xs = xs_discrete
        else:
            raise ValueError("Unsupported flow type for combining discrete and continuous parts.")

        return torch.as_tensor(xs, device=self.dummy.device, dtype=self.dummy.dtype)
