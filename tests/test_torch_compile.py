"""Graph capture is a cheap default check; benchmark actual Inductor separately."""
from copy import deepcopy

import pytest
import torch

from gammaboard_process import RectangularDomain
from glnis.core.nn.flow.continuous_flow import Flow
from glnis.core.nn.flow.spline_transform import unconstrained_rational_quadratic_spline
from glnis.core.nn.sampler_module import MixedFlow, SamplerModule


@pytest.mark.parametrize("discrete_first", [True, False])
@pytest.mark.parametrize("cardinalities,continuous", [((), 3), ((2, 3), 3), ((2, 3), 0)])
def test_compiled_methods_preserve_conditioned_probabilities_and_gradients(
        sampler_settings, discrete_first, cardinalities, continuous):
    settings = deepcopy(sampler_settings["module"])
    settings.pop("kind")
    model = MixedFlow(RectangularDomain(cardinalities, continuous), dims_c=2,
                      discrete_dims_first=discrete_first, **settings)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
    condition = torch.randn(7, 2)
    sample = torch.compile(model.sample, backend="eager", fullgraph=True)
    prob = torch.compile(model.prob, backend="eager", fullgraph=True)
    for _ in range(2):
        with torch.no_grad():
            xs, sampled_prob = sample(7, c=condition, return_prob=True)
            torch.testing.assert_close(sampled_prob, model.prob(xs, c=condition))
        eager_q = model.prob(xs, c=condition)
        compiled_q = prob(xs, c=condition)
        torch.testing.assert_close(compiled_q, eager_q)
        params = tuple(model.parameters())
        eager_grad = torch.autograd.grad(eager_q.log().mean(), params, allow_unused=True)
        compiled_grad = torch.autograd.grad(compiled_q.log().mean(), params, allow_unused=True)
        for parameter, expected, actual in zip(params, eager_grad, compiled_grad):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected)
                with torch.no_grad():
                    parameter.add_(-0.001 * expected)


def test_support_changes_do_not_change_graph_shape(sampler_settings):
    settings = deepcopy(sampler_settings["module"])
    settings.pop("kind")
    model = MixedFlow(RectangularDomain((), 3), **settings)
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    prob = torch.compile(model.prob, backend=backend, fullgraph=True)
    for outside_rows in (0, 1, 7):
        xs = torch.rand(7, 3)
        xs[:outside_rows] = 2.0
        torch.testing.assert_close(prob(xs), model.prob(xs))
    assert len(graphs) == 1


@pytest.mark.parametrize("inverse", [False, True])
def test_spline_outside_rows_have_identity_gradients(inverse):
    xs = torch.tensor([[0.3, 0.6], [-2., 0.4], [0.4, 2.]], requires_grad=True)
    widths = torch.randn(3, 2, 4, requires_grad=True)
    heights = torch.randn(3, 2, 4, requires_grad=True)
    derivatives = torch.randn(3, 2, 5, requires_grad=True)
    output, jac = unconstrained_rational_quadratic_spline(
        xs, widths, heights, derivatives, inverse=inverse)
    torch.testing.assert_close(output[1:], xs[1:])
    torch.testing.assert_close(jac[1:], torch.zeros_like(jac[1:]))
    (output.sum() + jac.sum()).backward()
    torch.testing.assert_close(xs.grad[1:], torch.ones_like(xs[1:]))
    for parameters in (widths, heights, derivatives):
        assert torch.isfinite(parameters.grad).all()
        torch.testing.assert_close(parameters.grad[1:], torch.zeros_like(parameters[1:]))


def test_compiled_callables_stay_outside_snapshot(sampler_settings):
    settings = deepcopy(sampler_settings["module"])
    settings.pop("kind")
    model = MixedFlow(RectangularDomain((2, 3), 3), **settings)
    # Simulate a checkpoint written before integer coordinate indices existed.
    del model.continuous_flow._condition_indices
    del model.continuous_flow._transform_indices
    restored = SamplerModule.from_snapshot(model.snapshot())
    sample = torch.compile(restored.sample, backend="eager", fullgraph=True)
    with torch.no_grad():
        sample(7, return_prob=True)
    snapshot = restored.snapshot()
    rng = torch.get_rng_state()
    with torch.no_grad():
        expected_xs, expected_q = sample(7, return_prob=True)
    resumed = SamplerModule.from_snapshot(snapshot)
    resumed_sample = torch.compile(resumed.sample, backend="eager", fullgraph=True)
    torch.set_rng_state(rng)
    with torch.no_grad():
        actual_xs, actual_q = resumed_sample(7, return_prob=True)
    torch.testing.assert_close(actual_xs, expected_xs, rtol=0, atol=0)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    assert type(resumed) is MixedFlow


def test_loading_condition_masks_rebuilds_coordinate_indices():
    source = Flow(2, condition_masks=[[True, False], [False, True]],
                  subnet={"layers": 2, "units": 8}, spline={"bins": 4})
    target = Flow(2, condition_masks=[[False, True], [True, False]],
                  subnet={"layers": 2, "units": 8}, spline={"bins": 4})
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
    target.load_state_dict(source.state_dict())
    xs = torch.rand(7, 2)
    torch.testing.assert_close(target.transform(xs), source.transform(xs))
