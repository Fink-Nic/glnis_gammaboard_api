"""Exact interrupted-run recovery and deliberate fresh-stage handoff."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from gammaboard_process import Feedback, RectangularDomain
from gammaboard_sampler import GammaBoardSampler


def values(batch):
    return 1 + batch.xs_continuous.sum(axis=1) + batch.xs_discrete.sum(axis=1)


def parameters(sampler):
    return deepcopy(sampler.orchestrator.module.state_dict())


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("legacy_pending", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_interrupted_checkpoint_resumes_exactly(tmp_path, sampler_settings, graph_settings, mapped, legacy_pending, compiled):
    if compiled:
        sampler_settings["orchestrator"]["execution"] = {"kind": "torch_compile", "backend": "eager"}
    if mapped:
        sampler_settings.update(mapping={"layer_cfgs": [{"kind": "ose"}]}, graph_properties=graph_settings)
    sampler = GammaBoardSampler(domain=RectangularDomain((1, 2), 3), save_path=str(tmp_path / "sampler.pt"), **sampler_settings)
    first = sampler.generate(8)
    sampler.feedback(Feedback(values(first)))
    consumed = sampler.generate(2)
    sampler.feedback(Feedback(values(consumed)))
    outstanding = sampler.generate(4)
    checkpoint = json.loads(json.dumps(sampler.snapshot()))
    if legacy_pending:
        checkpoint["pending_mapping_jac"] = checkpoint.pop("pending_mapping_weights")
    expected = sampler.generate(2)
    sampler.feedback(Feedback(values(outstanding)))
    sampler.feedback(Feedback(values(expected)))
    expected_parameters = parameters(sampler)
    expected_optimizer = deepcopy(sampler.orchestrator.optimizer.state_dict())
    expected_scheduler = deepcopy(sampler.orchestrator.scheduler.state_dict())

    restored = GammaBoardSampler.from_snapshot(snapshot=checkpoint, domain=sampler.domain)
    assert restored.orchestrator.training_status.num_generated == 6
    assert restored.orchestrator.training_status.num_ingested == 2
    assert len(restored.pending_mapping_weights) == (4 if mapped else 0)
    actual = restored.generate(2)
    np.testing.assert_array_equal(actual.xs_discrete, expected.xs_discrete)
    np.testing.assert_array_equal(actual.xs_continuous, expected.xs_continuous)
    np.testing.assert_array_equal(actual.weights, expected.weights)
    restored.feedback(Feedback(values(outstanding)))
    restored.feedback(Feedback(values(actual)))
    for name, tensor in expected_parameters.items():
        torch.testing.assert_close(restored.orchestrator.module.state_dict()[name], tensor, rtol=0, atol=0)
    assert restored.orchestrator.scheduler.state_dict() == expected_scheduler
    for index, state in expected_optimizer["state"].items():
        for name, expected_value in state.items():
            torch.testing.assert_close(restored.orchestrator.optimizer.state_dict()["state"][index][name], expected_value, rtol=0, atol=0)
    assert restored.pending_mapping_weights == []
    assert restored.orchestrator.training_status.step == 2


def test_restore_runtime_overrides_read_original_checkpoint(tmp_path, sampler_settings):
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 2), save_path=str(tmp_path / "source.pt"), **sampler_settings)
    first = sampler.generate(8)
    sampler.feedback(Feedback(values(first)))
    batch = sampler.generate(3)
    snapshot = sampler.snapshot()
    original = deepcopy(snapshot)
    restored = GammaBoardSampler.from_snapshot(snapshot=snapshot, domain=sampler.domain, init_args={
        "save_path": str(tmp_path / "destination.pt"), "orchestrator": {"use_f64": False, "use_gpu": False, "max_batch_size": 2},
    })
    assert snapshot == original
    assert restored.orchestrator.training_status.num_generated == 3
    assert restored.orchestrator.module.dummy.dtype == torch.float32
    restored.feedback(Feedback(values(batch)))
    restored.snapshot()
    assert (tmp_path / "source.pt").is_file()
    assert (tmp_path / "destination.pt").is_file()
    assert restored.orchestrator.training_status.step == 1
    last = restored.generate(8)
    restored.feedback(Feedback(values(last)))
    assert restored.orchestrator.training_status.step == 2
    assert np.isfinite(restored.orchestrator.training_status.last_loss)
    with pytest.raises(ValueError, match="from_stage"):
        GammaBoardSampler.from_snapshot(snapshot=snapshot, domain=sampler.domain,
                                        init_args={"orchestrator": {"training_batch_size": 7}})
    with pytest.raises(ValueError, match="domain"):
        GammaBoardSampler.from_snapshot(snapshot=snapshot, domain=RectangularDomain((3,), 2))


@pytest.mark.parametrize("trained", [False, True])
@pytest.mark.parametrize("new_training_steps", [0, 1])
def test_stage_keeps_proposal_and_resets_training(tmp_path, sampler_settings, graph_settings, trained, new_training_steps):
    sampler_settings.update(mapping={"layer_cfgs": [{"kind": "spherical"}]}, graph_properties=graph_settings)
    source = GammaBoardSampler(domain=RectangularDomain((2,), 3), save_path=str(tmp_path / "source.pt"), **sampler_settings)
    if trained:
        batch = source.generate(8)
        source.feedback(Feedback(values(batch)))
    source.generate(3)
    snapshot = source.snapshot()
    expected = parameters(source)
    stage = GammaBoardSampler.from_stage(source_state=snapshot, source_args=sampler_settings,
                                         init_args={"save_path": str(tmp_path / "stage.pt"), "orchestrator": {
                                             "training_steps": new_training_steps, "training_batch_size": 6,
                                             "learning_rate": .02, "loss": {"kind": "variance"},
                                             "scheduler": {"kind": "step", "step_size": 1, "gamma": .5},
                                         }}, domain=source.domain)
    for name, tensor in expected.items():
        torch.testing.assert_close(stage.orchestrator.module.state_dict()[name], tensor, rtol=0, atol=0)
    status = stage.orchestrator.training_status
    assert status.step == status.num_generated == status.num_ingested == status.total_trained_samples == 0
    assert status.loss_history == []
    assert status.result._n == status.abs_result._n == 0
    assert stage.pending_mapping_weights == []
    assert stage.orchestrator.optimizer.state == {}
    assert stage.orchestrator.optimizer.param_groups[0]["lr"] == .02
    assert stage.orchestrator.scheduler.last_epoch == 0
    batch = stage.generate(6)
    assert batch.training_remaining == (6 if new_training_steps else None)
    if new_training_steps:
        stage.feedback(Feedback(values(batch)))
        assert status.step == 1
        assert any(not torch.equal(stage.orchestrator.module.state_dict()[name], tensor) for name, tensor in expected.items())
    stage.snapshot()
    assert (tmp_path / "stage.pt").exists()
    assert source.orchestrator.training_status.num_generated == 3


def test_failed_snapshot_preserves_last_checkpoint(tmp_path, sampler_settings, monkeypatch):
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 2), save_path=str(tmp_path / "sampler.pt"), **sampler_settings)
    sampler.snapshot()
    saved = (tmp_path / "sampler.pt").read_bytes()
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        sampler.snapshot()
    assert (tmp_path / "sampler.pt").read_bytes() == saved
    assert not (tmp_path / "sampler.pt.tmp").exists()


def test_restore_metadata_without_config_uses_init_args(tmp_path, sampler_settings):
    settings = {**sampler_settings, "save_path": str(tmp_path / "sampler.pt")}
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 2), **settings)
    checkpoint = sampler.snapshot()
    checkpoint.pop("config")
    with pytest.raises(ValueError, match="init_args"):
        GammaBoardSampler.from_snapshot(snapshot=checkpoint, domain=sampler.domain)
    restored = GammaBoardSampler.from_snapshot(snapshot=checkpoint, domain=sampler.domain, init_args=settings)
    for name, expected in parameters(sampler).items():
        torch.testing.assert_close(restored.orchestrator.module.state_dict()[name], expected, rtol=0, atol=0)


def test_stage_normalizes_legacy_loss_and_scheduler_overrides(tmp_path, sampler_settings):
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 2), save_path=str(tmp_path / "sampler.pt"), **sampler_settings)
    stage = GammaBoardSampler.from_stage(source_state=sampler.snapshot(), source_args={}, domain=sampler.domain,
                                         init_args={"orchestrator": {"loss_type": "variance", "use_scheduler": False}})
    assert stage.orchestrator.cfg.loss == {"kind": "variance"}
    assert stage.orchestrator.scheduler is None


def test_legacy_module_names_migrate_only_when_reading_checkpoints(tmp_path, sampler_settings):
    source = GammaBoardSampler(domain=RectangularDomain((2,), 2), save_path=str(tmp_path / "old.pt"), **sampler_settings)
    batch = source.generate(3)
    snapshot = source.snapshot()
    snapshot["config"]["sampler"] = snapshot["config"].pop("module")
    saved = snapshot["orchestrator_snapshot"]["cfg_kwargs"]
    saved["sampler_kwargs"] = saved.pop("module_kwargs")
    original = deepcopy(snapshot)
    restored = GammaBoardSampler.from_snapshot(snapshot=snapshot, domain=source.domain)
    assert snapshot == original
    assert restored.cfg.module == source.cfg.module
    assert restored.orchestrator.cfg.module_kwargs == source.cfg.module
    restored.feedback(Feedback(values(batch)))
    rewritten = restored.snapshot()
    assert "sampler" not in rewritten["config"]
    assert "sampler_kwargs" not in rewritten["orchestrator_snapshot"]["cfg_kwargs"]
    stage = GammaBoardSampler.from_stage(source_state=snapshot, source_args={}, init_args={}, domain=source.domain)
    assert stage.cfg.module == source.cfg.module
    assert stage.orchestrator.training_status.num_generated == 0
    with pytest.raises(ValueError, match="module"):
        GammaBoardSampler.from_snapshot(snapshot=snapshot, domain=source.domain,
            init_args={"module": {"flow": {"subnet": {"units": 16}}}})
