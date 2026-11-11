"""Mappings own Jacobian correction, including raw training feedback."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import torch

from gammaboard_process import Feedback, RectangularDomain, StratumAssignments
from gammaboard_sampler import GammaBoardSampler
from glnis.core.mappings import LayeredMapping, Mapping, Space
from glnis.utils.types import GraphProperties


@pytest.mark.parametrize("kind", ["spherical", "ose", "fermi", "jac"])
def test_mappings_generate_and_expose_public_density(sampler_settings, graph_settings, kind):
    sampler = GammaBoardSampler(domain=RectangularDomain((1, 2), 3), mapping={"layer_cfgs": [{"kind": kind}]},
                                graph_properties=graph_settings, **sampler_settings)
    batch = sampler.generate(8)
    assert batch.xs_discrete.shape == (8, 2)
    assert batch.xs_continuous.shape == (8, 3)
    assert np.isfinite(batch.weights).all() and (batch.weights > 0).all()
    original = batch.xs_continuous.copy()
    density = sampler.pdf(batch.xs_discrete, batch.xs_continuous)
    assert density.shape == (8,)
    assert np.isfinite(density).all() and (density > 0).all()
    np.testing.assert_array_equal(batch.xs_continuous, original)
    if kind == "spherical":
        np.testing.assert_allclose(density * batch.weights, 1, rtol=1e-10)


def test_mapped_feedback_applies_only_mapping_factor_and_preserves_mask(sampler_settings, graph_settings):
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 3, strata_dims=1),
                                mapping={"layer_cfgs": [{"kind": "ose"}]}, graph_properties=graph_settings,
                                **sampler_settings)
    batch = sampler.generate(8)
    q = sampler.orchestrator.training_status.generated_probs.numpy().copy()
    discrete, continuous = sampler.orchestrator.module.to_disc_cont(sampler.orchestrator.training_status.generated_samples)
    mapped = sampler.mapping.forward(discrete[:, 1:], continuous)
    factors = mapped.jacobian * mapped.integration_weight
    np.testing.assert_allclose(batch.weights, factors / q)
    raw = np.arange(1., 9.)
    requested = np.array([0, 1] * 4)
    actual = np.array([0, 0] * 4)
    feedback = Feedback(raw, StratumAssignments(((0,), (1,)), requested, actual))
    sampler.feedback(feedback)
    expected = np.where(requested == actual, raw * factors, 0.)
    np.testing.assert_allclose(sampler.orchestrator.training_status.ingested_weights.numpy(), expected)
    np.testing.assert_array_equal(raw, np.arange(1., 9.))
    assert sampler.pending_mapping_weights == []


def test_hidden_channel_density_uses_geometric_jacobians(sampler_settings, graph_settings):
    graph_settings["edge_masses"] = [0., 2.]
    graph_settings["edge_momentum_shifts"] = [[0., 0., 0.], [.3, -.4, .1]]
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 3),
                                mapping={"layer_cfgs": [{"kind": "ose"}]}, graph_properties=graph_settings,
                                **sampler_settings)
    discrete = np.array([[0], [1]], dtype=np.int64)
    momentum = np.array([[.2, -.3, .6], [1., 2., -1.]])
    expected = np.zeros(2)
    mapping = sampler.mapping.mappings[0]
    for channel in range(2):
        channels = np.full((2, 1), channel, dtype=np.int64)
        jac, coordinates = mapping.mapping.backward(momentum.copy(), channels)
        q = sampler.orchestrator.pdf(np.hstack((discrete, channels)), coordinates)
        expected += q * jac.ravel()
    np.testing.assert_allclose(sampler.pdf(discrete, momentum), expected, rtol=1e-12)


def test_noninvertible_mapping_reports_no_density(sampler_settings, graph_settings):
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 3),
                                mapping={"layer_cfgs": [{"kind": "kaapo"}]}, graph_properties=graph_settings,
                                **sampler_settings)
    batch = sampler.generate(8)
    assert sampler.pdf(batch.xs_discrete, batch.xs_continuous) is None


def test_mapping_can_expand_continuous_dimensions(graph_settings):
    mapping = LayeredMapping(GraphProperties(**graph_settings), [{"kind": "reduced kaapo"}])
    result, jac, weight = mapping.forward(np.empty((4, 0), dtype=np.int64), np.array([[.2], [.3], [.6], [.7]]))
    assert result.shape == (4, 3)
    assert jac.shape == (4,)
    assert np.isfinite(result).all() and np.isfinite(jac).all()


def test_smoothing_mapping_inverse(graph_settings):
    mapping = LayeredMapping(GraphProperties(**graph_settings), [{"kind": "s"}])
    coordinates = np.array([[.1, .3, .8], [.7, .6, .4]])
    mapped, forward, weight = mapping.forward(None, coordinates)
    inverse, reverse, reverse_weight = mapping.backward(None, mapped)
    np.testing.assert_allclose(inverse, coordinates)
    np.testing.assert_allclose(forward * reverse, 1.)


def test_stacked_multichannel_geometric_jacobians_roundtrip(graph_settings):
    mapping = LayeredMapping(GraphProperties(**graph_settings), [
        {"kind": "ose", "mapping_kwargs": {"kind": "spherical", "conformal_scale": 1.25}},
        {"kind": "ose", "mapping_kwargs": {"kind": "spherical", "conformal_scale": .8}},
    ])
    discrete = np.array([[0, 1], [1, 0]], dtype=np.int64)
    coordinates = np.array([[.2, .3, .6], [.7, .6, .4]])
    mapped, forward, weight = mapping.forward(discrete, coordinates)
    restored, reverse, reverse_weight = mapping.backward(discrete, mapped)
    np.testing.assert_allclose(restored, coordinates, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(forward * reverse, 1, rtol=1e-12)
    np.testing.assert_allclose(weight * reverse_weight, 1, rtol=1e-12)


@pytest.mark.parametrize("space", [Space.HCUBE, Space.MOMENTUM])
def test_identity_has_unit_coordinate_jacobians_in_both_directions(graph_settings, space):
    mapping = LayeredMapping(GraphProperties(**graph_settings), [{"kind": "identity"}], input_space=space)
    coordinates = np.array([[.2, .3, .6], [.7, .6, .4]])
    forward = mapping.forward(None, coordinates)
    backward = mapping.backward(None, forward.continuous)
    for result in (forward, backward):
        np.testing.assert_array_equal(result.continuous, coordinates)
        np.testing.assert_array_equal(result.jacobian, np.ones(2))
        np.testing.assert_array_equal(result.integration_weight, np.ones(2))


@pytest.mark.parametrize("space", [Space.HCUBE, Space.MOMENTUM])
def test_multichannel_jacobians_equal_the_underlying_coordinate_transform(graph_settings, space):
    graph_settings["edge_masses"] = [0., 2.]
    mapping = Mapping.from_kind("ose", graph_properties=GraphProperties(**graph_settings), input_space=space)
    discrete = np.array([[0], [1]], dtype=np.int64)
    coordinates = np.array([[.2, .3, .6], [.7, .6, .4]])
    _, momentum = mapping.mapping.forward(coordinates, discrete)
    inputs = coordinates if space is Space.HCUBE else momentum
    expected = mapping.mapping.forward if space is Space.HCUBE else mapping.mapping.backward
    expected_jacobian, expected_output = expected(inputs.copy(), discrete)
    jacobian, output = mapping.forward(inputs.copy(), discrete)
    np.testing.assert_allclose(output, expected_output)
    np.testing.assert_allclose(jacobian, expected_jacobian)
    reverse_jacobian, restored = mapping.backward(output, discrete)
    np.testing.assert_allclose(restored, inputs)
    np.testing.assert_allclose(jacobian * reverse_jacobian, 1.)

    r_squared = (momentum ** 2).sum(axis=1)
    channel_weights = (r_squared[:, None] + np.array([0., 4.])) ** -.5
    partition = channel_weights[np.arange(2), discrete[:, 0]] / channel_weights.sum(axis=1)
    expected_weight = partition if space is Space.HCUBE else 1 / partition
    np.testing.assert_allclose(mapping.integration_weight(inputs, output, discrete), expected_weight)


def test_partition_changes_sample_weights_but_not_the_public_density(sampler_settings, graph_settings):
    graph_settings["edge_masses"] = [0., 2.]
    sampler_settings["orchestrator"]["training_steps"] = 0
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 3),
                                mapping={"layer_cfgs": [{"kind": "ose"}]}, graph_properties=graph_settings,
                                **sampler_settings)
    rng = torch.get_rng_state()
    before = sampler.generate(4)
    density = sampler.pdf(before.xs_discrete, before.xs_continuous)
    sampler.mapping.mappings[0].ose_exponent = 0.
    torch.set_rng_state(rng)
    after = sampler.generate(4)
    np.testing.assert_array_equal(after.xs_discrete, before.xs_discrete)
    np.testing.assert_array_equal(after.xs_continuous, before.xs_continuous)
    assert not np.allclose(after.weights, before.weights)
    np.testing.assert_allclose(sampler.pdf(before.xs_discrete, before.xs_continuous), density)


def test_identity_preserves_a_nonuniform_proposal_density(sampler_settings, graph_settings):
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 3),
                                mapping={"layer_cfgs": [{"kind": "identity"}]}, graph_properties=graph_settings,
                                **sampler_settings)
    for _ in range(2):
        batch = sampler.generate(8)
        sampler.feedback(Feedback(1 + batch.xs_continuous.sum(axis=1)))
    batch = sampler.generate(4)
    density = sampler.pdf(batch.xs_discrete, batch.xs_continuous)
    np.testing.assert_allclose(density, sampler.orchestrator.pdf(batch.xs_discrete, batch.xs_continuous))
    assert not np.allclose(density, .5)
