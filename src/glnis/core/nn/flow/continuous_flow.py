import math
from collections.abc import Callable
from typing import Literal, Protocol
from dataclasses import field

import numpy as np
import torch
import torch.nn as nn

from glnis.utils.types import BuildFromKind, Config

from .subnet_constructor import SubnetConstructor
from .spline_transform import SplineTransform
from glnis.core.config import dataclass


L2PI = -0.5 * math.log(2 * math.pi)

Mapping = Callable[[torch.Tensor, bool], tuple[torch.Tensor, torch.Tensor]]


class Distribution(Protocol):
    """
    Protocol for a (potentially learnable) distribution that can be used for sampling and
    density estimation, like a normalizing flow.
    """

    def sample(
        self,
        n: int,
        c: torch.Tensor | None = None,
        return_prob: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Draws samples following the distribution

        Args:
            n: number of samples
            c: condition, shape (n, dims_c) or None for an unconditional flow
            return_prob: if True, also return the probabilities
            device: device of the returned tensor. Only required if no condition is given.
            dtype: dtype of the returned tensor. Only required if no condition is given.
        Returns:
            samples with shape (n, dims_in). Depending on the arguments ``return_log_prob``,
            ``return_prob``, this function should also return the log-probabilities with shape (n, ),
            the probabilities with shape (n, ).
        """
        ...

    def log_prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the log-probabilities of the input data.

        Args:
            x: input data, shape (n, dims_in)
            c: condition, shape (n, dims_c) or None for an unconditional flow
        Returns:
            log-probabilities with shape (n, )
        """
        return self.prob(x, c).log()

    def prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the probabilities of the input data.

        Args:
            x: input data, shape (n, dims_in)
            c: condition, shape (n, dims_c) or None for an unconditional flow
        Returns:
            probabilities with shape (n, )
        """
        return self.log_prob(x, c).exp()


@dataclass
class FlowConfig(Config):
    permutations: Literal["log", "random", "exchange"] = "log"
    condition_masks: list[list[bool]] | None = None
    blocks: int | None = None
    spline: dict = field(default_factory=dict)
    subnet: dict = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, **kwargs):
        instance = cls(**kwargs)
        if instance.permutations in ("random", "exchange") and instance.blocks is None:
            raise ValueError(
                f"Number of coupling blocks must be specified if permutations is {instance.permutations}"
            )
        return instance

class Flow(nn.Module, Distribution, BuildFromKind):
    """
    Coupling-block based normalizing flow (1605.08803) using rational quadratic spline
    transformations (1906.04032). Both conditional and non-conditional flows are supported. The
    class also allows to build multi-channel flows, i.e. an efficient implementation of multiple
    independent flows with the same hyperparameters.
    """

    KIND = "coupling"
    DEFAULT_KIND = "coupling"

    def __init__(
        self,
        dims_in: int,
        dims_c: int = 0,
        **kwargs,
    ):
        """
        Args:
            dims_in: input dimension
            dims_c: condition dimension
            uniform_latent: If True, encode mapping from [0,1]^d to [0,1]^d and use a uniform
                latent space distribution. If False, encode mapping from R^d to R^d and use
                Gaussian latent space distribution.
            permutations: Defines the strategy to permute the input dimensions between coupling
                blocks. "log": logarithmic decomposition, so that every dimension is conditioned on
                every other dimension at least once. "random": randomly permute dimensions.
                "exchange": condition the first half of the input on the second half, then the
                other way around, repeatedly.
            condition_masks: Overwrites the permutation strategy with a custom conditioning mask
                with shape (blocks, dims_in). Components where the mask is True are used as
                condition, and components where it is False are transformed.
            blocks: number of coupling blocks. Only needed if permutations is "random" or
                "exchange"
            subnet_constructor: function used to construct the flow sub-networks, with the
                number of input features and output features of the subnet as arguments. If None,
                the MLP (single channel) or StackedMLP (multi-channel) classes are used.
        """
        super().__init__()
        self.cfg = FlowConfig.from_kwargs(**kwargs)

        self.dims_in = dims_in
        self.dims_c = dims_c
        if type(dims_in) is not int or dims_in < 0:
            raise ValueError("dims_in must be a nonnegative integer")

        subnet_constructor = lambda features_in, features_out: SubnetConstructor.from_kwargs(
            features_in=features_in,
            features_c=self.dims_c,
            features_out=features_out,
            **self.cfg.subnet
        )

        if dims_in == 0:
            condition_masks = torch.empty((0, 0), dtype=torch.bool)
        elif dims_in == 1 and self.cfg.condition_masks is None:
            condition_masks = torch.zeros((self.cfg.blocks or 2, 1), dtype=torch.bool)
        elif self.cfg.condition_masks is None:
            if self.cfg.permutations == "log":
                n_perms = int(np.ceil(np.log2(dims_in)))
                condition_masks = (
                    torch.tensor(
                        [
                            [int(i) for i in np.binary_repr(i, n_perms)]
                            for i in range(dims_in)
                        ]
                    )
                    .flip(dims=(1,))
                    .bool()
                    .t()
                    .repeat_interleave(2, dim=0)
                )
                condition_masks[1::2, :] ^= True
            elif self.cfg.permutations == "exchange":
                condition_masks = torch.cat(
                    (
                        torch.ones(dims_in // 2, dtype=torch.bool),
                        torch.zeros(dims_in - dims_in // 2, dtype=torch.bool),
                    )
                )[None, :].repeat((self.cfg.blocks, 1))
                condition_masks[1::2, :] ^= True
            elif self.cfg.permutations == "random":
                condition_masks = torch.cat(
                    (
                        torch.ones(dims_in // 2, dtype=torch.bool),
                        torch.zeros(dims_in - dims_in // 2, dtype=torch.bool),
                    )
                )[None, :].repeat((self.cfg.blocks, 1))
                for i in range(self.cfg.blocks):
                    condition_masks[i] = condition_masks[i][torch.randperm(dims_in)]
            else:
                raise ValueError(f"Unknown permutation type {self.cfg.permutations}")
        else:
            condition_masks = torch.tensor(self.cfg.condition_masks)

        if condition_masks.ndim != 2 or condition_masks.shape[1] != dims_in:
            raise ValueError("condition_masks must have shape (blocks, dims_in)")
        condition_masks = condition_masks.bool()
        # A block with no transformed coordinates is an identity.
        condition_masks = condition_masks[~condition_masks.all(dim=1)]
        self.register_buffer("condition_masks", condition_masks)
        self._set_coordinate_indices()

        self.subnets = nn.ModuleList()
        self.coupling_transforms: list[SplineTransform] = []
        # needed to get one fake dim when 1D only
        dim1 = 1 if self.dims_in == 1 else 0
        for mask in self.condition_masks: # type: ignore
            dims_cond = int(torch.count_nonzero(mask))
            coupling_transform = SplineTransform.from_kwargs(dims_in=dims_in - dims_cond, **self.cfg.spline)
            self.subnets.append(
                subnet_constructor(
                    features_in=dims_cond + dim1,
                    features_out=coupling_transform.get_cond_dims(),
                )
            )
            self.coupling_transforms.append(coupling_transform)

    def _set_coordinate_indices(self):
        # Masks describe architecture, not batch data. Integer indices keep the
        # number of features static for the compiler, including empty conditions.
        masks = self.condition_masks.cpu()
        self._condition_indices = tuple(mask.nonzero().flatten().tolist() for mask in masks)
        self._transform_indices = tuple((~mask).nonzero().flatten().tolist() for mask in masks)

    def __setstate__(self, state):
        super().__setstate__(state)
        # Modules are also stored as complete objects in existing checkpoints.
        if not hasattr(self, "_condition_indices"):
            self._set_coordinate_indices()

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._set_coordinate_indices()

    def transform(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms the input data into the latent space or back.

        Args:
            x: input data, shape (n, dims_in)
            c: condition, shape (n, dims_c) or None for an unconditional flow
            inverse: if True, use inverted transformation (i.e. the sampling direction)
        Returns:
            tuple containing the transformed values with shape (n, dims_in), and log Jacobian
            determinants with shape (n, ) of the transformation
        """
        x = x.clone()
        jac = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        if x.shape[0] == 0:
            return x, jac

        batch_size = x.shape[0]
        blocks = list(zip(self._condition_indices, self._transform_indices,
                          self.subnets, self.coupling_transforms))
        if inverse:
            blocks.reverse()
        for condition_indices, transform_indices, subnet, coupling_transform in blocks:
            x_trafo = x[:, transform_indices]
            x_cond = x[:, condition_indices] if self.dims_in > 1 else torch.ones_like(x_trafo)
            if c is not None:
                x_cond = torch.cat((x_cond, c), dim=1)
            subnet_out = subnet(x_cond).reshape(
                (batch_size, x_trafo.shape[1], -1)
            )
            x_out, block_jac = coupling_transform(
                x_trafo, subnet_out, inverse
            )
            x[:, transform_indices] = x_out
            jac += block_jac.sum(dim=1)

        return x, jac


    def log_prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the log-probabilities of the input data.

        Args:
            x: input data, shape (n, dims_in)
            c: condition, shape (n, dims_c) or None for an unconditional flow
            return_latent: if True, also return the latent space vector
        Returns:
            log-probabilities with shape (n, ). If ``return_latent`` is True, it also returns the
            latent space vector with shape (n, dims_in).
        """
        z, jac = self.transform(x, c, False)
        inside = ((z >= 0) & (z <= 1)).all(dim=1)
        return torch.where(inside, jac, -torch.inf)

    def prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the probabilities of the input data.

        Args:
            x: input data, shape (n, dims_in)
            c: condition, shape (n, dims_c) or None for an unconditional flow
            return_latent: if True, also return the latent space vector
        Returns:
            probabilities with shape (n, ). If ``return_latent`` is True, it also returns the
            latent space vector with shape (n, dims_in).
        """
        return self.log_prob(x, c).exp()

    def sample(
        self,
        n: int | None = None,
        c: torch.Tensor | None = None,
        return_prob: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Draws samples from the probability distribution encoded by the flow.

        Args:
            n: number of samples. Only required if no condition is given.
            c: condition, shape (n, dims_c) or None for an unconditional flow
            return_prob: if True, also return the probabilities
            device: device of the returned tensor. Only required if no condition is given.
            dtype: dtype of the returned tensor. Only required if no condition is given.
        Returns:
            samples with shape (n, dims_in). Optionally returns the probabilities with shape (n, ).
        """
        if n is None:
            if c is None:
                raise ValueError("Either n or c must be specified")
            n = len(c)
        if c is None:
            options = {}
            if device is not None:
                options["device"] = device
            if dtype is not None:
                options["dtype"] = dtype
        else:
            options = {"device": c.device, "dtype": c.dtype}

        z = torch.rand((n, self.dims_in), **options)
        x, jac = self.transform(z, c, True)

        if return_prob:
            log_prob = -jac
            return x, log_prob.exp()
        return x
