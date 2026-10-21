# type:ignore
from __future__ import annotations

import numpy as np
import pytest
import torch

from gammaboard_process import GenerationStatus

from madnis_sampler import MadnisSampler


def test_partial_training_draws_request_feedback_until_the_window_is_complete() -> None:
    sampler = MadnisSampler(
        discrete_cardinalities=[],
        continuous_dims=2,
        use_gpu=False,
        training_steps=1,
        training_batch_size=4,
        max_batch_size=2,
        flow_config={"layers": 2, "units": 16, "bins": 4},
    )

    first = sampler.generate(2)
    second = sampler.generate(2)
    assert first.training_remaining == 4
    assert second.training_remaining == 2
    assert sampler.generate(1) == GenerationStatus.WAITING

    # Weighted feedback for the constant integrand f=1 is 1/q, i.e. weights.
    sampler.feedback(first.weights)
    assert sampler.training_status.step == 0
    sampler.feedback(second.weights)
    assert sampler.training_status.step == 1
    assert sampler.training_status.total_trained_samples == 4

    inference = sampler.generate(3)
    assert inference.training_remaining is None
    assert inference.xs_continuous.shape == (3, 2)


def test_pdf_masks_cardinality_one_discrete_dimensions(monkeypatch) -> None:
    sampler = MadnisSampler(
        discrete_cardinalities=[1, 3],
        continuous_dims=2,
        use_gpu=False,
        training_steps=1,
        training_batch_size=4,
        max_batch_size=16,
    )

    seen: list[torch.Tensor] = []

    def fake_prob(x: torch.Tensor) -> torch.Tensor:
        seen.append(x.detach().clone())
        return torch.ones(len(x), device=x.device, dtype=x.dtype)

    monkeypatch.setattr(sampler.madnis.flow, "prob", fake_prob)

    xs_discrete = np.array([[0, 1], [99, 2]], dtype=np.uint64)
    xs_continuous = np.full((2, 2), 0.5, dtype=np.float64)
    pdf = sampler.pdf(xs_discrete, xs_continuous)

    assert sampler.madnis.integrand.discrete_dims == [3]
    assert len(seen) == 1
    np.testing.assert_allclose(
        seen[0].numpy(),
        [[1.0, 0.5, 0.5], [2.0, 0.5, 0.5]],
    )
    np.testing.assert_allclose(pdf, np.ones(2))


@pytest.mark.parametrize("position", ["first", "last"])
def test_pdf_evaluates_flow_in_inverse_mapped_coordinates(monkeypatch, position) -> None:
    sampler = MadnisSampler(
        discrete_cardinalities=[1, 3], continuous_dims=2,
        use_gpu=False, training_steps=1, training_batch_size=4,
        max_batch_size=1, discrete_dims_position=position,
    )
    # A nontrivial map, with different Jacobians, detects evaluating the density
    # in physical coordinates or dropping/double-applying the inverse Jacobian.
    physical = np.array([[12.0, -4.0], [24.0, 8.0]])
    mapped = np.array([[0.2, 0.4], [0.6, 0.8]])
    inverse_jacobian = np.array([0.25, 0.5])
    discrete = np.array([[0, 1], [0, 2]], dtype=np.uint64)

    def inverse_mapping(d, x, weight, inverse=False):
        assert inverse
        np.testing.assert_array_equal(d, discrete)
        np.testing.assert_array_equal(x, physical)
        np.testing.assert_array_equal(weight, np.ones(2))
        return d, mapped, inverse_jacobian.copy()

    seen = []

    def density(x):
        seen.append(x.detach().numpy().copy())
        coordinates = x[:, 1:] if position == "first" else x[:, :-1]
        return coordinates[:, 0] + 2 * coordinates[:, 1]

    monkeypatch.setattr(sampler, "_apply_mapping", inverse_mapping)
    monkeypatch.setattr(sampler.madnis.flow, "prob", density)
    pdf = sampler.pdf(discrete, physical)
    expected = np.hstack([discrete[:, 1:], mapped] if position == "first"
                         else [mapped, discrete[:, 1:]])
    np.testing.assert_allclose(np.concatenate(seen), expected)
    np.testing.assert_allclose(pdf, [0.25, 1.1])
