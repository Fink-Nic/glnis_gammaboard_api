from types import SimpleNamespace

import numpy as np
import pytest
import torch

from glnis.core.nn.losses import Loss, KLDivergence
from glnis.core.nn.schedulers import Scheduler
from glnis.core.nn.orchestrator import RectangularOrchestrator
from glnis.utils.types import RectangularTrainingStatus
from gammaboard_process import RectangularDomain
from gammaboard_sampler import GammaBoardSampler


@pytest.mark.parametrize("kind", ["kl_divergence", "rkl_divergence", "variance"])
def test_losses_and_gradients_match_exact_categorical_objective(kind):
    indices = torch.tensor([0, 1, 1, 1])
    target = torch.tensor([.4, .6])
    q_sample = torch.tensor([.25, .75])[indices]
    logits = torch.tensor([.1, -.2], requires_grad=True)
    probability = logits.softmax(0)
    status = SimpleNamespace(ingested_weights=target[indices], generated_probs=q_sample,
                             abs_result=SimpleNamespace(mean=1.), result=SimpleNamespace(mean=1.))
    actual = Loss.from_kwargs(kind=kind, normalize_across_training_steps=True)(status, probability[indices])
    if kind == "kl_divergence":
        expected = (target * (target.log() - probability.log())).sum()
    elif kind == "rkl_divergence":
        expected = (probability * (probability.log() - target.log())).sum()
    else:
        expected = (target.square() / probability).sum() - 1
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("kind", ["kl_divergence", "rkl_divergence", "variance"])
def test_zero_losses_are_finite_and_differentiable(kind):
    status = RectangularTrainingStatus(1, 4)
    status.generated_probs.fill_(.5)
    status.ingested_weights.zero_()
    q = torch.full((4,), .5, requires_grad=True)
    loss = Loss.from_kind(kind)(status, q)
    assert loss.item() == 0.
    loss.backward()
    torch.testing.assert_close(q.grad, torch.zeros(4))


class ExperimentLoss(KLDivergence):
    KIND = "test experiment loss"


class ExperimentOrchestrator(RectangularOrchestrator):
    KIND = "test experiment orchestrator"


def test_registered_components_are_used_by_the_public_constructor(sampler_settings):
    sampler_settings["orchestrator"].update(kind=ExperimentOrchestrator.KIND, loss={"kind": ExperimentLoss.KIND})
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 1), **sampler_settings)
    assert isinstance(sampler.orchestrator, ExperimentOrchestrator)
    assert isinstance(sampler.orchestrator.loss_function, ExperimentLoss)


@pytest.mark.parametrize("settings,expected", [
    ({"kind": "cosineannealing"}, [.1, .05, 0.]),
    ({"kind": "step", "step_size": 1, "gamma": .5}, [.1, .05, .025]),
    ({"kind": "exponential", "gamma": .5}, [.1, .05, .025]),
])
def test_scheduler_factories_and_state_roundtrip(settings, expected):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=.1)
    scheduler = Scheduler.from_kwargs(optimizer=optimizer, training_steps=2, **settings)
    values = [optimizer.param_groups[0]["lr"]]
    for _ in range(2):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    assert values == pytest.approx(expected)
    copy = Scheduler.from_kwargs(optimizer=optimizer, training_steps=2, **settings)
    copy.load_state_dict(scheduler.state_dict())
    assert copy.state_dict() == scheduler.state_dict()


def test_invalid_component_kinds_fail_at_construction(sampler_settings):
    for field in ("loss", "scheduler"):
        settings = {**sampler_settings, "orchestrator": {**sampler_settings["orchestrator"], field: {"kind": "missing"}}}
        with pytest.raises(ValueError, match="Unknown kind"):
            GammaBoardSampler(domain=RectangularDomain((2,), 1), **settings)


class ExperimentScheduler(torch.optim.lr_scheduler.StepLR, Scheduler):
    KIND = "test experiment scheduler"

    def __init__(self, optimizer, training_steps):
        super().__init__(optimizer, step_size=1, gamma=.5)


def test_custom_factories_survive_restore_and_stage_transfer(tmp_path, sampler_settings):
    sampler_settings["orchestrator"].update(kind=ExperimentOrchestrator.KIND, loss={"kind": ExperimentLoss.KIND},
                                            scheduler={"kind": ExperimentScheduler.KIND})
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 1), save_path=str(tmp_path / "custom.pt"), **sampler_settings)
    checkpoint = sampler.snapshot()
    restored = GammaBoardSampler.from_snapshot(snapshot=checkpoint, domain=sampler.domain)
    stage = GammaBoardSampler.from_stage(source_state=checkpoint, source_args={}, init_args={}, domain=sampler.domain)
    for model in (restored, stage):
        assert isinstance(model.orchestrator, ExperimentOrchestrator)
        assert isinstance(model.orchestrator.loss_function, ExperimentLoss)
        assert isinstance(model.orchestrator.scheduler, ExperimentScheduler)
