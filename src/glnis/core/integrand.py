"""Graph-independent integrands in the coordinates returned by a sampler."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from inspect import signature
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from glnis.utils.types import Result


def _positive_dimension(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError("continuous_dims must be a positive integer")
    return int(value)


def _real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return array


def _scalar(value: float, name: str, *, positive: bool = False) -> float:
    array = _real_array(value, name)
    if array.ndim != 0 or (positive and array <= 0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}scalar")
    return float(array)


class Integrand(ABC):
    """Evaluate unweighted values; sampling and mapping weights live elsewhere.

    Inputs have shapes ``(N, continuous_dims)`` and
    ``(N, len(discrete_cardinalities))``. Discrete dimensions use a counting
    measure: the integral sums over their possible values.
    """

    IDENTIFIER = "integrand"

    def __init__(
        self,
        continuous_dims: int,
        discrete_cardinalities: list[int] | None = None,
        *,
        target: Result | None = None,
    ) -> None:
        self.continuous_dims = _positive_dimension(continuous_dims)
        self.discrete_cardinalities = [
            _positive_dimension(cardinality)
            for cardinality in ([] if discrete_cardinalities is None else discrete_cardinalities)
        ]
        self.dtype = np.dtype(np.float64)
        if target is not None and not isinstance(target, Result):
            raise TypeError("target must be a Result or None")
        self.target = deepcopy(target)

    def evaluate_batch(
        self, continuous: ArrayLike, discrete: ArrayLike | None = None
    ) -> NDArray:
        """Return one real or complex value per point, without changing inputs."""
        continuous = _real_array(continuous, "continuous")
        if continuous.ndim != 2 or continuous.shape[1] != self.continuous_dims:
            raise ValueError(f"continuous must have shape (N, {self.continuous_dims})")
        n_points = len(continuous)
        n_discrete = len(self.discrete_cardinalities)
        if discrete is None:
            if n_discrete:
                raise ValueError(f"discrete must have shape (N, {n_discrete})")
            discrete = np.empty((n_points, 0), dtype=np.int64)
        else:
            discrete = np.asarray(discrete)
            if discrete.shape != (n_points, n_discrete):
                raise ValueError(f"discrete must have shape ({n_points}, {n_discrete})")
            if discrete.size and discrete.dtype.kind not in "iu":
                raise ValueError("discrete coordinates must be integers")
            for index, cardinality in enumerate(self.discrete_cardinalities):
                if np.any(discrete[:, index] < 0) or np.any(discrete[:, index] >= cardinality):
                    raise ValueError(f"discrete column {index} must be in [0, {cardinality})")
            discrete = discrete.astype(np.int64, copy=False)
        values = np.asarray(self._evaluate_batch(continuous, discrete))
        if values.shape != (n_points,) or values.dtype.kind not in "iufc":
            raise ValueError("integrand output must be a numeric array with shape (N,)")
        return values

    def __call__(
        self, continuous: ArrayLike, discrete: ArrayLike | None = None
    ) -> NDArray:
        return self.evaluate_batch(continuous, discrete)

    @abstractmethod
    def _evaluate_batch(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        """Evaluate a batch whose dimensions and discrete indices were checked."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Integrand:
        """Construct an integrand from a strict ``[integrand]`` configuration."""
        if not isinstance(config, Mapping):
            raise TypeError("integrand configuration must be a mapping")
        kwargs = dict(config)
        kind = kwargs.pop("kind", None)
        registry = {
            "gaussian": GaussianIntegrand,
            "sine": SineIntegrand,
            "ghostbumps": GhostBumpsIntegrand,
        }
        if not isinstance(kind, str) or kind not in registry:
            raise ValueError(f"integrand.kind must be one of {', '.join(registry)}")
        if "target" in kwargs and isinstance(kwargs["target"], Mapping):
            target = dict(kwargs["target"])
            allowed = {"n_points", "total_time"} | {
                f"{phase}_{stat}"
                for phase in ("real", "imag", "abs_real", "abs_imag")
                for stat in ("mean", "error")
            }
            unknown = set(target) - allowed
            if unknown:
                raise ValueError(f"Unknown integrand.target options: {', '.join(sorted(unknown))}")
            for name, value in target.items():
                if name == "n_points":
                    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                        raise ValueError("integrand.target.n_points must be a nonnegative integer")
                else:
                    value = _scalar(value, f"integrand.target.{name}")
                    if (name.endswith("_error") or name == "total_time") and value < 0:
                        raise ValueError(f"integrand.target.{name} must be nonnegative")
            if target.get("n_points", 0) < 2 and any(
                name.endswith("_error") and value > 0 for name, value in target.items()
            ):
                raise ValueError("integrand.target.n_points must be >= 2 when a target error is positive")
            kwargs["target"] = Result.from_kwargs(**target)
        factory = registry[kind]
        try:
            signature(factory).bind(**kwargs)
        except TypeError as error:
            raise ValueError(f"Invalid {kind} integrand configuration: {error}") from error
        return factory(**kwargs)


class GaussianIntegrand(Integrand):
    """Gaussian components whose total integral is one over all real coordinates.

    Multiple widths introduce one discrete component index. Components share
    one normalization, the sum of their unnormalized integrals, preserving the
    original convention. ``offset`` is shared or one vector per component.

    No reference target is assumed: integration on the unit cube covers only
    part of the Gaussian unless the sampler maps to all real space.
    """

    IDENTIFIER = "gaussian integrand"

    def __init__(
        self,
        continuous_dims: int,
        sigma: float | list[float] = 1.0,
        offset: ArrayLike | None = None,
        const_f: bool = False,
        *,
        target: Result | None = None,
    ) -> None:
        continuous_dims = _positive_dimension(continuous_dims)
        sigma_array = _real_array(sigma, "sigma")
        if sigma_array.ndim > 1 or not sigma_array.size or np.any(sigma_array <= 0):
            raise ValueError("sigma must be a positive scalar or a nonempty list of positive widths")
        self.sigma = sigma_array.reshape(-1).copy()
        n_components = len(self.sigma)
        if offset is None:
            self.offset = np.zeros((n_components, continuous_dims))
        else:
            offsets = _real_array(offset, "offset")
            if offsets.shape == (continuous_dims,):
                offsets = np.broadcast_to(offsets, (n_components, continuous_dims))
            elif offsets.shape != (n_components, continuous_dims):
                raise ValueError(
                    f"offset must have shape ({continuous_dims},) or ({n_components}, {continuous_dims})"
                )
            self.offset = offsets.copy()
        if not isinstance(const_f, (bool, np.bool_)):
            raise ValueError("const_f must be a boolean")
        self.const_f = bool(const_f)
        log_norms = continuous_dims * (0.5 * np.log(2 * np.pi) + np.log(self.sigma))
        maximum = log_norms.max()
        self._log_normalization = maximum + np.log(np.exp(log_norms - maximum).sum())
        super().__init__(
            continuous_dims,
            [n_components] if n_components > 1 else [],
            target=target,
        )

    def _evaluate_batch(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        if self.const_f:
            return np.ones(len(continuous), dtype=self.dtype)
        component = discrete[:, 0] if self.discrete_cardinalities else np.zeros(len(continuous), dtype=np.int64)
        centered = (continuous - self.offset[component]) / self.sigma[component, None]
        return np.exp(-0.5 * np.square(centered).sum(axis=1) - self._log_normalization)


class SineIntegrand(Integrand):
    """Mean of integer-frequency sines on the unit cube, plus an offset."""

    IDENTIFIER = "sine integrand"

    def __init__(
        self,
        continuous_dims: int = 1,
        omega: int = 1,
        offset: float = 0.0,
        *,
        target: Result | None = None,
    ) -> None:
        if isinstance(omega, (bool, np.bool_)) or not isinstance(omega, (int, np.integer)):
            raise ValueError("omega must be an integer")
        self.omega = int(omega)
        self.offset = _scalar(offset, "offset")
        super().__init__(
            continuous_dims,
            target=Result.from_kwargs(real_mean=self.offset) if target is None else target,
        )

    def _evaluate_batch(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        return np.sin(2 * np.pi * self.omega * continuous).mean(axis=1) + self.offset


class GhostBumpsIntegrand(Integrand):
    """Sum of unnormalized Gaussian bumps; defaults reproduce two 2D bumps."""

    IDENTIFIER = "ghostbumps integrand"

    def __init__(
        self,
        centers: ArrayLike = ((0.75, 0.75), (0.25, 0.25)),
        sigma: float = 0.15,
        *,
        target: Result | None = None,
    ) -> None:
        self.centers = _real_array(centers, "centers").copy()
        if self.centers.ndim != 2 or not all(self.centers.shape):
            raise ValueError("centers must be a nonempty array with shape (n_centers, continuous_dims)")
        self.sigma = _scalar(sigma, "sigma", positive=True)
        super().__init__(self.centers.shape[1], target=target)

    def _evaluate_batch(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        result = np.zeros(len(continuous), dtype=self.dtype)
        for center in self.centers:
            displacement = (continuous - center) / self.sigma
            result += np.exp(-0.5 * np.square(displacement).sum(axis=1))
        return result
