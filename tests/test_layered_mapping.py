"""Configuration-driven tests for :class:`LayeredMapping`.

By default every ``*.toml`` file in ``tests/cfgs/layered_mappings`` is tested.
Adding another configuration to that directory automatically extends the suite.
The public helpers below can also be imported by future mapping-specific tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from glnis.core.mappings import LayeredMapping, Mapping, Space
from glnis.utils.types import GraphProperties


CONFIG_DIR = Path(__file__).parent / "cfgs" / "layered_mappings"


@dataclass(frozen=True)
class LayeredMappingCase:
    """A parsed TOML configuration and its instantiated mapping."""

    path: Path
    config: dict[str, Any]
    mapping: LayeredMapping


def layered_mapping_config_paths(config_dir: Path = CONFIG_DIR) -> list[Path]:
    """Return all LayeredMapping TOML cases in deterministic order."""

    paths = sorted(path for path in config_dir.glob("*.toml") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No LayeredMapping TOML files found in {config_dir}")
    return paths


def load_layered_mapping_case(path: str | Path) -> LayeredMappingCase:
    """Instantiate a LayeredMapping and GraphProperties from a TOML test case."""

    config_path = Path(path)
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    try:
        input_space = Space[str(config.get("input_space", "HCUBE")).upper()]
    except KeyError as error:
        raise ValueError(
            f"Unknown input_space in {config_path}: {config.get('input_space')!r}"
        ) from error

    graph_properties = GraphProperties(**config["graph_properties"])
    mapping = LayeredMapping(
        graph_properties=graph_properties,
        layer_cfgs=config["layers"],
        input_space=input_space,
        use_f128=bool(config.get("use_f128", False)),
    )
    return LayeredMappingCase(config_path, config, mapping)


def generate_discrete_inputs(
    mapping: LayeredMapping,
    n_points: int,
    rng: np.random.Generator,
) -> NDArray[np.uint64]:
    """Generate discrete inputs with MadNIS's autoregressive CDF logic.

    MadNIS starts with an empty integer tensor. For each discrete dimension it
    requests the prior conditioned on the generated prefix, constructs a CDF,
    samples that dimension, and appends the result to the prefix.
    """

    discrete = np.zeros((n_points, 0), dtype=np.uint64)
    for dim, cardinality in enumerate(mapping.discrete_cardinalities):
        prior = np.asarray(
            mapping.discrete_prior_prob_function(discrete, dim),
            dtype=np.float64,
        )

        assert prior.shape == (n_points, cardinality)
        assert np.all(np.isfinite(prior))
        assert np.all(prior >= 0.0)
        np.testing.assert_allclose(prior.sum(axis=1), 1.0)

        cdf = np.cumsum(prior, axis=1)
        samples = np.sum(rng.random((n_points, 1)) > cdf, axis=1)
        assert np.all(samples < cardinality)
        discrete = np.column_stack((discrete, samples.astype(np.uint64)))

    return discrete


def _supports_transform(mapping: LayeredMapping, direction: str) -> bool:
    """Return whether every layer implements the requested transform."""

    for layer in mapping.mappings:
        source_space = layer.input_space if direction == "forward" else layer.output_space
        method_name = (
            "_map_from_hcube"
            if source_space is Space.HCUBE
            else "_map_from_momentum"
        )
        if getattr(type(layer), method_name) is getattr(Mapping, method_name):
            return False
    return True


def _continuous_samples(
    space: Space,
    n_points: int,
    dimension: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    if space is Space.HCUBE:
        return rng.uniform(0.1, 0.9, size=(n_points, dimension))
    return rng.normal(size=(n_points, dimension))


def _assert_transform_output(
    output: tuple[NDArray, NDArray],
    n_points: int,
    continuous_dimension: int,
) -> None:
    continuous, weights = output
    assert continuous.shape == (n_points, continuous_dimension)
    assert weights.shape == (n_points,)
    assert np.all(np.isfinite(continuous))
    assert np.all(np.isfinite(weights))


@pytest.fixture(
    params=layered_mapping_config_paths(),
    ids=lambda config_path: config_path.stem,
)
def layered_mapping_case(request: pytest.FixtureRequest) -> LayeredMappingCase:
    """Instantiate each default TOML case independently for every test."""

    return load_layered_mapping_case(request.param)


def test_instantiates_layered_mapping(layered_mapping_case: LayeredMappingCase) -> None:
    case = layered_mapping_case

    assert isinstance(case.mapping, LayeredMapping)
    assert len(case.mapping.mappings) == len(case.config["layers"])
    assert all(isinstance(layer, Mapping) for layer in case.mapping.mappings)


def test_continuous_dimensions(layered_mapping_case: LayeredMappingCase) -> None:
    case = layered_mapping_case
    expected = case.config["expected"]

    assert case.mapping.continuous_dims_in == expected["continuous_dims_in"]
    assert case.mapping.continuous_dims_out == expected["continuous_dims_out"]


def test_discrete_cardinalities(layered_mapping_case: LayeredMappingCase) -> None:
    case = layered_mapping_case
    expected = case.config["expected"]

    assert case.mapping.discrete_cardinalities == expected["discrete_cardinalities"]
    assert len(case.mapping.discrete_cardinalities) == sum(
        layer.num_discrete_dims for layer in case.mapping.mappings
    )


def test_forward_and_backward_transforms(
    layered_mapping_case: LayeredMappingCase,
) -> None:
    case = layered_mapping_case
    mapping = case.mapping
    expected = case.config["expected"]
    n_points = int(case.config.get("n_points", 64))
    rng = np.random.default_rng(int(case.config.get("seed", 0)))
    discrete = generate_discrete_inputs(mapping, n_points, rng)
    weights = rng.uniform(0.5, 1.5, size=n_points)

    input_space = mapping.mappings[0].input_space
    output_space = mapping.mappings[-1].output_space
    forward_input = _continuous_samples(
        input_space, n_points, mapping.continuous_dims_in, rng
    )
    supports_forward = _supports_transform(mapping, "forward")
    supports_backward = _supports_transform(mapping, "backward")

    if supports_forward:
        forward_output = mapping.forward(discrete, forward_input, weights)
        _assert_transform_output(
            forward_output, n_points, mapping.continuous_dims_out
        )
    else:
        with pytest.raises(NotImplementedError):
            mapping.forward(discrete, forward_input, weights)
        forward_output = None

    if supports_backward:
        if forward_output is None:
            backward_input = _continuous_samples(
                output_space, n_points, mapping.continuous_dims_out, rng
            )
            backward_weights = weights
        else:
            backward_input, backward_weights = forward_output

        backward_output = mapping.backward(
            discrete, backward_input, backward_weights
        )
        _assert_transform_output(
            backward_output, n_points, mapping.continuous_dims_in
        )
    else:
        backward_input = (
            forward_output[0]
            if forward_output is not None
            else _continuous_samples(
                output_space, n_points, mapping.continuous_dims_out, rng
            )
        )
        with pytest.raises(NotImplementedError):
            mapping.backward(discrete, backward_input, weights)
        backward_output = None

    if supports_forward and supports_backward:
        backward_continuous, backward_weights = backward_output # type: ignore[assignment]
        np.testing.assert_allclose(
            backward_continuous,
            forward_input,
            rtol=float(expected.get("round_trip_rtol", 1.0e-10)),
            atol=float(expected.get("round_trip_atol", 1.0e-10)),
        )
        np.testing.assert_allclose(
            backward_weights,
            weights,
            rtol=float(expected.get("round_trip_rtol", 1.0e-10)),
            atol=float(expected.get("round_trip_atol", 1.0e-10)),
        )


def test_discrete_prior_autoregressive_sampling(
    layered_mapping_case: LayeredMappingCase,
) -> None:
    case = layered_mapping_case
    n_points = int(case.config.get("n_points", 64))
    rng = np.random.default_rng(int(case.config.get("seed", 0)))

    discrete = generate_discrete_inputs(case.mapping, n_points, rng)

    assert discrete.shape == (
        n_points,
        len(case.mapping.discrete_cardinalities),
    )
    for values, cardinality in zip(
        discrete.T, case.mapping.discrete_cardinalities
    ):
        assert np.all(values < cardinality)

    terminal_prior = case.mapping.discrete_prior_prob_function(
        discrete, len(case.mapping.discrete_cardinalities)
    )
    assert terminal_prior.shape == discrete.shape
    assert np.count_nonzero(terminal_prior) == 0
