# type: ignore[override]

from __future__ import annotations

from copy import deepcopy
from math import erf, sqrt

import numpy as np
import pytest

from glnis.core.integrand import GaussianIntegrand, GhostBumpsIntegrand, Integrand, SineIntegrand


def test_gaussian_component_offsets_and_inputs_are_unchanged() -> None:
    offsets = np.array([[1.0, 2.0], [-3.0, 4.0]])
    continuous = offsets.copy()
    discrete = np.array([[0], [1]], dtype=np.uint64)
    integrand = GaussianIntegrand(2, sigma=[0.5, 2.0], offset=offsets)
    normalization = 2 * np.pi * (0.5**2 + 2.0**2)
    np.testing.assert_allclose(integrand(continuous, discrete), np.full(2, 1 / normalization))
    np.testing.assert_array_equal(continuous, offsets)
    np.testing.assert_array_equal(discrete, [[0], [1]])
    offsets[:] = 99
    np.testing.assert_allclose(integrand(continuous, discrete), np.full(2, 1 / normalization))
    assert integrand.discrete_cardinalities == [2]
    assert integrand.target is None


def test_gaussian_shared_offset_and_full_space_normalization() -> None:
    integrand = GaussianIntegrand(1, sigma=[0.5, 1.5], offset=[0.3])
    nodes, weights = np.polynomial.legendre.leggauss(180)
    coordinates = (nodes * 15 + 0.3).reshape(-1, 1)
    total = sum(
        np.dot(weights * 15, integrand(coordinates, np.full((len(nodes), 1), component)))
        for component in range(2)
    )
    assert total == pytest.approx(1.0, abs=1e-12)


def test_gaussian_cube_integral_has_no_incorrect_unit_target() -> None:
    integrand = GaussianIntegrand(1)
    nodes, weights = np.polynomial.legendre.leggauss(20)
    integral = np.dot(weights / 2, integrand(((nodes + 1) / 2).reshape(-1, 1)))
    assert integral == pytest.approx(erf(1 / sqrt(2)) / 2)
    assert integrand.target is None


def test_constant_gaussian_returns_one_value_per_point() -> None:
    integrand = GaussianIntegrand(3, const_f=True)
    np.testing.assert_array_equal(integrand(np.zeros((4, 3))), np.ones(4))


def test_sine_uses_direct_dimensions_and_unit_cube_reference() -> None:
    integrand = SineIntegrand(continuous_dims=2, omega=3, offset=0.7)
    nodes, weights = np.polynomial.legendre.leggauss(40)
    coordinates = np.column_stack(((nodes + 1) / 2, (nodes + 1) / 2))
    assert np.dot(weights / 2, integrand(coordinates)) == pytest.approx(0.7)
    assert integrand.continuous_dims == 2
    assert integrand.target.real.mean == 0.7
    np.testing.assert_array_equal(SineIntegrand(omega=0, offset=2)([[0], [1]]), [2, 2])


def test_ghostbumps_defaults_and_dimension_inference() -> None:
    default = GhostBumpsIntegrand()
    expected = 1 + np.exp(-0.5 / (2 * 0.15**2))
    np.testing.assert_allclose(default([[0.75, 0.75], [0.25, 0.25]]), [expected, expected])
    one_dimensional = GhostBumpsIntegrand(centers=[[0], [1]], sigma=0.5)
    assert one_dimensional.continuous_dims == 1
    assert one_dimensional.discrete_cardinalities == []
    assert one_dimensional.target is None
    np.testing.assert_allclose(one_dimensional([[0.5]]), [2 * np.exp(-0.5)])


@pytest.mark.parametrize("integrand", [GaussianIntegrand(2), SineIntegrand(2), GhostBumpsIntegrand()])
def test_empty_batches_have_consistent_shape(integrand) -> None:
    assert integrand(np.empty((0, 2))).shape == (0,)


@pytest.mark.parametrize(
    "config, error",
    [
        ({"kind": "missing"}, "integrand.kind"),
        ({"kind": "gaussian"}, "continuous_dims"),
        ({"kind": "sine", "n_sines": 2}, "n_sines"),
        ({"kind": "sine", "continuous_dims": 1.5}, "positive integer"),
        ({"kind": "sine", "continuous_dims": True}, "positive integer"),
        ({"kind": "sine", "omega": 1.5}, "integer"),
        ({"kind": "sine", "offset": np.inf}, "finite"),
        ({"kind": "gaussian", "continuous_dims": 2, "sigma": 0}, "positive"),
        ({"kind": "gaussian", "continuous_dims": 2, "sigma": []}, "nonempty"),
        ({"kind": "gaussian", "continuous_dims": 2, "sigma": [[1]]}, "positive"),
        ({"kind": "gaussian", "continuous_dims": 2, "offset": [1]}, "offset"),
        ({"kind": "gaussian", "continuous_dims": 2, "const_f": "yes"}, "boolean"),
        ({"kind": "ghostbumps", "centers": []}, "centers"),
        ({"kind": "ghostbumps", "sigma": -1}, "positive"),
        ({"kind": "sine", "target": {"typo": 1}}, "target"),
    ],
)
def test_invalid_configuration_is_rejected(config, error) -> None:
    with pytest.raises(ValueError, match=error):
        Integrand.from_config(config)


def test_configuration_factory_preserves_input_and_builds_reference_result() -> None:
    config = {
        "kind": "gaussian", "continuous_dims": 1, "sigma": [1, 2],
        "offset": [[0], [1]], "target": {"real_mean": 1.0},
    }
    original = deepcopy(config)
    integrand = Integrand.from_config(config)
    assert isinstance(integrand, GaussianIntegrand)
    assert integrand.target.real.mean == 1.0
    assert config == original


@pytest.mark.parametrize("phase", ["real", "imag", "abs_real", "abs_imag"])
def test_reference_errors_require_a_sample_count_and_are_preserved(phase) -> None:
    target = {f"{phase}_mean": 1.0, f"{phase}_error": 0.1}
    for count in (None, 0, 1):
        settings = target if count is None else {**target, "n_points": count}
        with pytest.raises(ValueError, match="n_points must be >= 2"):
            Integrand.from_config({"kind": "sine", "target": settings})
    integrand = Integrand.from_config({"kind": "sine", "target": {**target, "n_points": 2}})
    assert getattr(integrand.target, phase).error == pytest.approx(0.1)


@pytest.mark.parametrize(
    "continuous, discrete, error",
    [
        ([0, 1], [[0], [0]], "continuous"),
        ([[0, 1]], [[0]], "continuous"),
        ([[0]], None, "discrete"),
        ([[0]], [[0], [1]], "discrete"),
        ([[0]], [[0.5]], "integers"),
        ([[0]], [[-1]], "column"),
        ([[0]], [[2]], "column"),
        ([[np.nan]], [[0]], "finite"),
    ],
)
def test_input_shapes_and_component_indices_are_checked(continuous, discrete, error) -> None:
    integrand = GaussianIntegrand(1, sigma=[1, 2])
    with pytest.raises(ValueError, match=error):
        integrand(continuous, discrete)
