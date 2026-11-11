# type: ignore
from __future__ import annotations

from numpy.typing import NDArray
from torch._tensor import Tensor
from abc import ABC, abstractmethod

from dataclasses import dataclass, field
import io

import numpy as np
import torch
import torch.nn as nn

from gammaboard_process import Domain
from glnis.utils.types import Config
from glnis.core.nn.flow.continuous_flow import Flow
from glnis.core.nn.flow.discrete_flow import DiscreteFlow
from glnis.utils.types import BuildFromKind


class SamplerModule(ABC, nn.Module, BuildFromKind):

    """
    A base class for sampler modules.
    """
    KIND = "SamplerModuleBase"
    DEFAULT_KIND = "mixed flow"
    

    def __init__(
            self,
            domain: Domain,
            ):
        super().__init__()
        self.domain = domain

    @classmethod
    def from_snapshot(cls, snapshot: bytes):
        buffer = io.BytesIO(snapshot)
        module = torch.load(buffer, map_location="cpu", weights_only=False)
        if not isinstance(module, cls):
            raise ValueError("checkpoint does not contain a compatible sampler module")
        return module

    @abstractmethod
    def sample(self, n_samples: int, c: Tensor | None = None, return_prob: bool = False) -> Tensor | tuple[Tensor, Tensor]:
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
            xs: Complete internal points, shape (batch, discrete_dims + continuous_dims).
            c: Optional conditioning tensor.

        Returns:
            The probability dnesity function values at the given points, including gradients.
        
        """
        pass

    @abstractmethod
    def discrete_mass(self, subspace: NDArray, c: NDArray | None = None) -> float | None:
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
    DEFAULT_KIND = "mixed flow"

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
class MixedFlowConfig(Config):
    """
    Config for the MixedFlow module which generates both discrete and continuous samples.
    """
    discrete_dims_first: bool = True
    flow: dict = field(default_factory=dict)
    discrete: dict = field(default_factory=dict)


class MixedFlow(RectangularSamplerModule):
    KIND = "mixed flow"
    DEFAULT_KIND = "mixed flow"

    def __init__(
        self,
        domain: Domain,
        dims_c: int = 0,
        cfg : MixedFlowConfig | None = None,
        **kwargs
    ):
        super().__init__(domain, dims_c)
        self.cfg = cfg if cfg is not None else MixedFlowConfig.from_kwargs(**kwargs)
        self.discrete_cardinalities
        self.continuous_dims
        if self.cfg.discrete_dims_first:
            dims_c_discrete = dims_c
            dims_c_continuous = sum(self.discrete_cardinalities) + dims_c
        else:
            dims_c_discrete = dims_c + self.continuous_dims
            dims_c_continuous = dims_c

        self.discrete_flow = DiscreteFlow.from_kwargs(
            dims_in=self.discrete_cardinalities,
            dims_c=dims_c_discrete,
            **self.cfg.discrete,
        ) if self.discrete_cardinalities else None

        self.continuous_flow = Flow.from_kwargs(
            dims_in=self.continuous_dims,
            dims_c=dims_c_continuous,
            **self.cfg.flow,
        )

        self.max_dim_discrete = max(self.discrete_cardinalities) if self.discrete_cardinalities else 0
        # Retain this buffer for compatibility with existing state dictionaries.
        one_hot_mask = []
        for dim in self.discrete_cardinalities:
            one_hot_mask.extend([True] * dim + [False] * (self.max_dim_discrete - dim))
        self.register_buffer("one_hot_mask", torch.tensor(one_hot_mask, dtype=torch.bool))
        self.register_buffer("dummy", torch.tensor([0.0]))


    def prob(self, xs: Tensor, c: Tensor | None = None) -> Tensor:
        if xs.ndim != 2 or xs.shape[1] != len(self.discrete_cardinalities) + self.continuous_dims:
            raise ValueError("prob requires complete internal coordinates")
        if self.discrete_flow is None:
            return self.continuous_flow.prob(xs, c=c)

        xs_discrete, xs_continuous = self._to_disc_cont_tensor(xs)
        
        if self.cfg.discrete_dims_first:
            prob_discrete = self.discrete_flow.prob(xs_discrete, c=c)
            condition = self._encode_discrete(xs_discrete)
            if c is not None:
                condition = torch.cat((c, condition), dim=1)
            prob_continuous = self.continuous_flow.prob(xs_continuous, c=condition)
        else:
            prob_continuous = self.continuous_flow.prob(xs_continuous, c=c)
            condition = (
                xs_continuous if c is None else torch.cat((c, xs_continuous), dim=1)
            )
            prob_discrete = self.discrete_flow.prob(xs_discrete, c=condition)
        return prob_discrete * prob_continuous

    def sample(self, n_samples: int, c: Tensor | None = None, return_prob: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Sample from the flow model.

        Args:
            n_samples: Number of samples to draw.
            c: Optional conditioning tensor.
            return_prob: Whether to return the probability of the samples.

        Returns:
            A tensor of samples drawn from the model, optionally with their probabilities.

        """
        if self.discrete_flow is None:
            x_continuous, prob_continuous = self.continuous_flow.sample(
                n=n_samples,
                c=c,
                return_prob=True,
                device=self.dummy.device,
                dtype=self.dummy.dtype,
            )
            if return_prob:
                return x_continuous, prob_continuous
            return x_continuous
        if self.cfg.discrete_dims_first:
            x_discrete, prob_discrete = self.discrete_flow.sample(
                n=n_samples,
                c=c,
                return_prob=True,
                device=self.dummy.device,
                dtype=self.dummy.dtype,
            )
            condition = self._encode_discrete(x_discrete)
            if c is not None:
                condition = torch.cat((c, condition), dim=1)
            x_continuous, prob_continuous = self.continuous_flow.sample(
                c=condition,
                return_prob=True,
                device=self.dummy.device,
                dtype=self.dummy.dtype,
            )
            x = torch.cat((x_discrete.to(x_continuous.dtype), x_continuous), dim=1)
        else:
            x_continuous, prob_continuous = self.continuous_flow.sample(
                n=n_samples,
                c=c,
                return_prob=True,
                device=self.dummy.device,
                dtype=self.dummy.dtype,
            )
            condition = (
                x_continuous if c is None else torch.cat((c, x_continuous), dim=1)
            )
            x_discrete, prob_discrete = self.discrete_flow.sample(
                c=condition,
                return_prob=True,
                device=self.dummy.device,
                dtype=self.dummy.dtype,
            )
            x = torch.cat((x_continuous, x_discrete.to(x_continuous.dtype)), dim=1)

        if return_prob:
            return x, prob_discrete * prob_continuous
        return x

    def to_disc_cont(self, xs: Tensor) -> tuple[NDArray, NDArray]:
        """Split a sampled tensor into discrete and continuous parts based on the flow's configuration."""
        xs_discrete, xs_continuous = self._to_disc_cont_tensor(xs)
        return xs_discrete.numpy(force=True), xs_continuous.numpy(force=True)

    def pdf(self, xs_discrete: NDArray, xs_continuous: NDArray, c: NDArray | None = None) -> NDArray | None:
        with torch.no_grad():
            xs_tensor = self._to_xs_tensor(xs_discrete, xs_continuous)
            if c is not None:
                c = torch.as_tensor(c, device=self.dummy.device, dtype=self.dummy.dtype)
            probs = self.prob(xs_tensor, c=c)
            return probs.numpy(force=True).ravel()

    def discrete_mass(self, subspace: NDArray, c: NDArray | None = None) -> float | None:
        if self.discrete_flow is None:
            return 1.0 if np.asarray(subspace).shape[-1] == 0 else 0.0
        if not self.cfg.discrete_dims_first and self.continuous_dims:
            # Marginalizing a continuously conditioned discrete model is not analytic.
            return None
        subspace = torch.as_tensor(subspace, device=self.dummy.device, dtype=torch.int64)
        if c is not None:
            c = torch.as_tensor(c, device=self.dummy.device, dtype=self.dummy.dtype)
        with torch.no_grad():
            res = self.discrete_flow.discrete_mass(subspace, c=c)
        return res[0].item() if res is not None else None

    def _to_disc_cont_tensor(self, xs: Tensor) -> tuple[Tensor, Tensor]:
        """Split a sampled tensor into discrete and continuous parts based on the flow's configuration."""
        if self.discrete_flow is None:
            return torch.empty((xs.shape[0], 0), device=xs.device, dtype=torch.int64), xs
        if self.cfg.discrete_dims_first:
            xs_discrete, xs_continuous = xs.tensor_split([self.discrete_flow.num_discrete_dims], dim=1)
        else:
            xs_continuous, xs_discrete = xs.tensor_split([self.continuous_flow.dims_in], dim=1)
        return xs_discrete.long(), xs_continuous

    def _to_xs_tensor(self, xs_discrete: NDArray, xs_continuous: NDArray) -> Tensor:
        """Combine discrete and continuous tensors into a single tensor based on the flow's configuration."""
        if self.discrete_flow is None:
            return torch.as_tensor(xs_continuous, device=self.dummy.device, dtype=self.dummy.dtype)
        if self.cfg.discrete_dims_first:
            xs = np.hstack([xs_discrete, xs_continuous])
        else:
            xs = np.hstack([xs_continuous, xs_discrete])
        return torch.as_tensor(xs, device=self.dummy.device, dtype=self.dummy.dtype)

    def _encode_discrete(self, indices: torch.Tensor) -> torch.Tensor:
        # Each cardinality is fixed by the domain. Separate one-hot blocks avoid
        # boolean indexing with a data-dependent output shape during compilation.
        return torch.cat([
            torch.functional.F.one_hot(indices[:, i], cardinality)
            for i, cardinality in enumerate(self.discrete_cardinalities)
        ], dim=1).to(self.dummy.dtype)
