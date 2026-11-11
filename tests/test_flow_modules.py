"""Probability identities must hold beyond zero-initialized networks."""
import numpy as np
import pytest
import torch

from gammaboard_process import RectangularDomain
from glnis.core.nn.flow.discrete_flow import DiscreteTransformer
from glnis.core.nn.flow.continuous_flow import Flow
from glnis.core.nn.flow.subnet_constructor import MLP, MLP_fully_conditioned
from glnis.core.nn.sampler_module import MixedFlow


@pytest.mark.parametrize("dimensions", [(2,), (2, 3), (2, 3, 2)])
@pytest.mark.parametrize("conditional_dims", [0, 2])
@pytest.mark.parametrize("subnet", ["mlp", "mlp fully conditioned"])
def test_transformer_generation_equals_probability_and_prefix_sums(dimensions, conditional_dims, subnet):
    torch.manual_seed(42)
    model = DiscreteTransformer(dims_in=dimensions, dims_c=conditional_dims, embedding_dim=8,
                               feedforward_dim=8, heads=2, layers=1,
                               subnet={"kind": subnet, "layers": 2, "units": 8})
    with torch.no_grad():
        for parameter in model.subnet.parameters():
            parameter.normal_(0, .2)
    condition = torch.randn(7, conditional_dims) if conditional_dims else None
    x, density = model.sample(n=7, c=condition, return_prob=True)
    torch.testing.assert_close(density, model.prob(x, c=condition), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model.discrete_mass(x[:, :0], condition), torch.ones(7))
    model.prob(x, c=condition).log().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.x_embedding.parameters()) if len(dimensions) > 1 else True


@pytest.mark.parametrize("subnet", ["mlp", "mlp fully conditioned"])
@pytest.mark.parametrize("discrete_first", [True, False])
def test_mixed_flow_external_conditioning(sampler_settings, subnet, discrete_first):
    settings = sampler_settings["sampler"]
    settings.pop("kind")
    settings["flow"]["subnet"]["kind"] = subnet
    settings["discrete"]["subnet"]["kind"] = subnet
    model = MixedFlow(domain=RectangularDomain((2, 3), 2), dims_c=2,
                      discrete_dims_first=discrete_first, **settings)
    condition = torch.randn(5, 2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.01 * torch.randn_like(parameter))
    x, q = model.sample(5, c=condition, return_prob=True)
    torch.testing.assert_close(q, model.prob(x, c=condition), rtol=1e-10, atol=1e-12)
    discrete, continuous = model.to_disc_cont(x)
    np.testing.assert_allclose(model.pdf(discrete, continuous, condition.numpy()), q.detach().numpy(), rtol=1e-10)


@pytest.mark.parametrize("dims", [0, 1, 3])
def test_continuous_flow_roundtrip_and_support(dims):
    flow = Flow(dims_in=dims, subnet={"layers": 2, "units": 8}, spline={"bins": 4})
    with torch.no_grad():
        for subnet in flow.subnets:
            subnet.layers[-1].weight.normal_(0, .05)
    x, q = flow.sample(11, return_prob=True)
    torch.testing.assert_close(q, flow.prob(x), rtol=1e-10, atol=1e-12)
    z, forward = flow.transform(x)
    reconstructed, reverse = flow.transform(z, inverse=True)
    torch.testing.assert_close(reconstructed, x)
    torch.testing.assert_close(forward, -reverse)
    assert flow.prob(torch.empty((0, dims))).shape == (0,)
    if dims:
        outside = torch.ones((1, dims)) * 2
        assert flow.prob(outside).item() == 0.
        assert torch.isfinite(flow.prob(torch.ones((1, dims)))).all()
