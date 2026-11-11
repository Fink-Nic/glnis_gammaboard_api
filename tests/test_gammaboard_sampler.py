"""Public generation, feedback, densities, and discrete constraints."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import torch

from gammaboard_process import Feedback, GenerationStatus, RectangularDomain, StratumAssignments
from gammaboard_sampler import GammaBoardSampler


@pytest.mark.parametrize("cardinalities,continuous_dims", [
    ((), 3), ((1,), 3), ((2,), 1), ((2, 3), 0), ((1, 2, 1, 3), 3), ((), 0),
])
@pytest.mark.parametrize("discrete_first", [True, False])
def test_sample_density_agreement_before_and_after_training(sampler_settings, cardinalities, continuous_dims, discrete_first):
    settings = deepcopy(sampler_settings)
    settings["module"]["discrete_dims_first"] = discrete_first
    if not cardinalities and not continuous_dims:
        settings["orchestrator"]["training_steps"] = 0
    sampler = GammaBoardSampler(domain=RectangularDomain(cardinalities, continuous_dims), **settings)
    for _ in range(settings["orchestrator"]["training_steps"] + 1):
        batch = sampler.generate(8)
        assert batch.xs_discrete.shape == (len(batch.weights), len(cardinalities))
        assert batch.xs_continuous.shape == (len(batch.weights), continuous_dims)
        for index, size in enumerate(cardinalities):
            assert ((0 <= batch.xs_discrete[:, index]) & (batch.xs_discrete[:, index] < size)).all()
        density = sampler.pdf(batch.xs_discrete, batch.xs_continuous)
        assert density.shape == batch.weights.shape
        np.testing.assert_allclose(density * batch.weights, 1, rtol=1e-10)
        if batch.training_remaining is not None:
            values = 1 + batch.xs_continuous.sum(axis=1) + batch.xs_discrete.sum(axis=1)
            sampler.feedback(Feedback(values))


def test_defaults_and_generation_limits():
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 1), orchestrator={"training_batch_size": 5, "use_gpu": False})
    assert len(sampler.generate(2).weights) == 2
    assert len(sampler.generate(20).weights) == 3
    assert sampler.generate(1) is GenerationStatus.WAITING
    sampler.feedback(Feedback(np.ones(5)))
    assert sampler.orchestrator.training_status.step == 1
    for count in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            sampler.generate(count)


@pytest.mark.parametrize("f64", [False, True])
def test_fragmented_signed_and_zero_training(sampler_settings, f64):
    sampler_settings["orchestrator"].update(use_f64=f64)
    sampler_settings["orchestrator"]["loss"]["normalize_across_training_steps"] = True
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 2), **sampler_settings)
    for step in range(2):
        for size in (3, 2, 3):
            batch = sampler.generate(size)
            values = -1 - batch.xs_continuous[:, 0] if step == 0 else np.zeros(size)
            sampler.feedback(Feedback(values))
    assert sampler.orchestrator.training_status.abs_result.mean > 0
    assert sampler.orchestrator.training_status.loss_history[-1] == 0.
    assert all(np.isfinite(loss) for loss in sampler.orchestrator.training_status.loss_history)
    assert sampler.orchestrator.training_status.generated_samples.dtype == (torch.float64 if f64 else torch.float32)


def test_constraints_marginalize_singletons_and_holes(sampler_settings):
    sampler = GammaBoardSampler(domain=RectangularDomain((1, 2, 1, 3), 2), **sampler_settings)
    assert sampler.discrete_mass([{}, {0: 0}, {0: 1}, {1: 0}, {3: 2}]) == pytest.approx([1., 1., 0., .5, 1/3])
    full = [{0: 0, 1: i, 2: 0, 3: j} for i, j in product(range(2), range(3))]
    assert sum(sampler.discrete_mass(full)) == pytest.approx(1.)
    batch = sampler.generate(8)
    sampler.feedback(Feedback(1 + batch.xs_discrete[:, 1] + batch.xs_discrete[:, 3]))
    mass = sampler.discrete_mass(full)
    assert sum(mass) == pytest.approx(1.)
    assert sampler.discrete_mass([{3: 2}])[0] == pytest.approx(mass[2] + mass[5])
    with pytest.raises(ValueError, match="out of bounds"):
        sampler.discrete_mass([{4: 0}])


def test_continuous_first_discrete_marginal_is_explicitly_unavailable(sampler_settings):
    sampler_settings["module"]["discrete_dims_first"] = False
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 2), **sampler_settings)
    assert sampler.discrete_mass([{}]) == [1.]
    assert sampler.discrete_mass([{0: 0}]) == [None]


def test_masked_feedback_preserves_caller_data(sampler_settings):
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 2, strata_dims=1), **sampler_settings)
    sampler.generate(8)
    values = np.arange(1., 9.)
    requested = np.array([0, 1] * 4)
    actual = np.array([0, 0] * 4)
    feedback = Feedback(values, StratumAssignments(((0,), (1,)), requested, actual))
    sampler.feedback(feedback)
    np.testing.assert_array_equal(values, np.arange(1., 9.))
    np.testing.assert_array_equal(sampler.orchestrator.training_status.ingested_weights.numpy(), np.where(requested == actual, values, 0))


def test_invalid_feedback_does_not_corrupt_training(sampler_settings):
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 2), **sampler_settings)
    sampler.generate(2)
    with pytest.raises(ValueError, match="exceeds"):
        sampler.feedback(Feedback(np.ones(3)))
    assert sampler.orchestrator.training_status.num_ingested == 0
    sampler.feedback(Feedback(np.ones(2)))


def test_seed_covers_initialization_without_changing_global_defaults(sampler_settings):
    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    a = GammaBoardSampler(domain=RectangularDomain((2,), 2), **sampler_settings)
    b = GammaBoardSampler(domain=a.domain, **sampler_settings)
    for actual, expected in zip(a.orchestrator.module.parameters(), b.orchestrator.module.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.get_default_dtype() == dtype
    assert torch.get_default_device() == device
