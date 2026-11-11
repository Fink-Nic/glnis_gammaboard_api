"""Execution policy, owned buffers, and recovery across process boundaries."""
from copy import deepcopy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from gammaboard_process import Feedback, RectangularDomain
from gammaboard_sampler import GammaBoardSampler
from glnis.core.nn.execution import SamplerExecution, TorchCompileConfig
from glnis.utils.types import RectangularTrainingStatus


def test_lazy_execution_keeps_compiler_state_outside_snapshots(tmp_path, sampler_settings, monkeypatch):
    construction, calls = [], []

    def compile_method(method, **options):
        construction.append((method.__name__, options))

        def run(*args, **kwargs):
            calls.append((method.__name__, torch.is_inference_mode_enabled(), method.__self__.training))
            return method(*args, **kwargs)
        return run

    monkeypatch.setattr(torch, "compile", compile_method)
    sampler_settings["orchestrator"].update(training_steps=1, max_batch_size=8,
                                            execution={"kind": "torch_compile"})
    original = deepcopy(sampler_settings)
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 2), save_path=str(tmp_path / "sampler.pt"), **sampler_settings)
    assert sampler_settings == original
    assert [name for name, _ in construction] == ["sample", "prob"]
    assert construction[0][1]["dynamic"] is True
    assert construction[1][1]["dynamic"] is False
    assert calls == []
    batch = sampler.generate(8)
    sampler.feedback(Feedback(1 + batch.xs_continuous.sum(1)))
    assert not sampler.orchestrator.module.training
    sampler.generate(8)
    assert calls == [("sample", False, True), ("prob", False, True), ("sample", True, False)]
    # Marginal queries remain eager, including their prefix scoring.
    sampler.discrete_mass([{0: 0}])
    assert len(calls) == 3
    rng = torch.get_rng_state()
    snapshot = sampler.snapshot()
    assert len(construction) == 2
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    loaded = torch.load(tmp_path / "sampler.pt", weights_only=False)
    assert b"TorchCompileExecution" not in loaded.module_snapshot
    assert snapshot["config"]["orchestrator"]["execution"]["kind"] == "torch_compile"
    stage = GammaBoardSampler.from_stage(source_state=snapshot, source_args={},
                                         init_args={"orchestrator": {"training_steps": 1}}, domain=sampler.domain)
    assert stage.orchestrator.module.training
    assert stage.orchestrator.training_status.step == 0
    assert stage.orchestrator.optimizer.state == {}
    assert len(construction) == 4
    restored = GammaBoardSampler.from_snapshot(snapshot=snapshot, domain=sampler.domain,
                                               init_args={"orchestrator": {"execution": {"kind": "eager"}}})
    assert type(restored.orchestrator.execution) is SamplerExecution
    assert not restored.orchestrator.module.training


@pytest.mark.parametrize("disabled", ["sample", "prob"])
def test_execution_can_compile_selected_method(sampler_settings, monkeypatch, disabled):
    compiled = []
    monkeypatch.setattr(torch, "compile", lambda method, **kwargs: compiled.append(method.__name__) or method)
    sampler_settings["orchestrator"]["execution"] = {"kind": "torch_compile", disabled: False}
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 2), **sampler_settings)
    assert compiled == [name for name in ("sample", "prob") if name != disabled]
    assert getattr(sampler.orchestrator.execution, disabled) == getattr(sampler.orchestrator.module, disabled)


@pytest.mark.parametrize("sizes", [[4], [1, 3]])
def test_training_buffers_own_generated_tensors(sizes):
    status = RectangularTrainingStatus(2, 4)
    sample_storage, prob_storage = status.generated_samples.data_ptr(), status.generated_probs.data_ptr()
    for size in sizes:
        samples, probs = torch.ones(size, 2), torch.full((size,), 0.5)
        status.add_generation_data(samples, probs)
        samples.zero_()
        probs.zero_()
    torch.testing.assert_close(status.generated_samples, torch.ones(4, 2))
    torch.testing.assert_close(status.generated_probs, torch.full((4,), 0.5))
    assert status.generated_samples.data_ptr() == sample_storage
    assert status.generated_probs.data_ptr() == prob_storage


@pytest.mark.parametrize("options", [{"backend": ""}, {"sample": None}, {"prob": []}])
def test_invalid_compile_configuration(options):
    with pytest.raises(ValueError, match="execution"):
        TorchCompileConfig.from_kwargs(**options)


def test_compiled_checkpoint_in_fresh_process(tmp_path, sampler_settings):
    # The default suite exercises capture. Run this same test with BACKEND=inductor
    # and DEVICE=cuda:0 / MODE=reduce-overhead to exercise actual code generation.
    backend = os.environ.get("GLNIS_TEST_COMPILE_BACKEND", "eager")
    device = os.environ.get("GLNIS_TEST_COMPILE_DEVICE", "cpu")
    mode = os.environ.get("GLNIS_TEST_COMPILE_MODE", "default")
    options = {"dynamic": False}
    if backend == "inductor":
        options["mode"] = mode
    sampler_settings["orchestrator"].update(use_gpu=device.startswith("cuda"), gpu_id=0,
                                            max_batch_size=8, execution={"kind": "torch_compile",
                                                "backend": backend, "sample": options, "prob": options})
    sampler = GammaBoardSampler(domain=RectangularDomain((2,), 2), save_path=str(tmp_path / "checkpoint.pt"), **sampler_settings)
    assert str(sampler.orchestrator.device) == device
    batch = sampler.generate(8)
    sampler.feedback(Feedback(1 + batch.xs_continuous.sum(1) + batch.xs_discrete.sum(1)))
    first = sampler.generate(2)
    sampler.feedback(Feedback(1 + first.xs_continuous.sum(1) + first.xs_discrete.sum(1)))
    pending = sampler.generate(3)
    metadata = sampler.snapshot()
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    pending_values = 1 + pending.xs_continuous.sum(1) + pending.xs_discrete.sum(1)
    (tmp_path / "pending.json").write_text(json.dumps(pending_values.tolist()))
    expected = sampler.generate(3)
    sampler.feedback(Feedback(pending_values))
    sampler.feedback(Feedback(1 + expected.xs_continuous.sum(1) + expected.xs_discrete.sum(1)))
    inference = sampler.generate(5)
    completed = subprocess.run([sys.executable, "-c", '''
import json, sys, torch
import numpy as np
from pathlib import Path
from gammaboard_process import Feedback, RectangularDomain
from gammaboard_sampler import GammaBoardSampler
torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)
root = Path(sys.argv[1])
sampler = GammaBoardSampler.from_snapshot(snapshot=json.loads((root / "metadata.json").read_text()),
                                          domain=RectangularDomain((2,), 2))
assert sampler.orchestrator.training_status.num_generated == 5
assert sampler.orchestrator.training_status.num_ingested == 2
batch = sampler.generate(3)
sampler.feedback(Feedback(np.array(json.loads((root / "pending.json").read_text()))))
sampler.feedback(Feedback(1 + batch.xs_continuous.sum(1) + batch.xs_discrete.sum(1)))
inference = sampler.generate(5)
torch.save({"batch": batch, "inference": inference, "module": sampler.orchestrator.module.state_dict(),
            "optimizer": sampler.orchestrator.optimizer.state_dict(),
            "step_devices": [str(state["step"].device) for state in sampler.orchestrator.optimizer.state.values()],
            "scheduler": sampler.orchestrator.scheduler.state_dict(),
            "training": sampler.orchestrator.module.training}, root / "actual.pt")
''', str(tmp_path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = torch.load(tmp_path / "actual.pt", weights_only=False, map_location="cpu")
    for name, expected_batch in (("batch", expected), ("inference", inference)):
        for field in ("xs_discrete", "xs_continuous", "weights"):
            np.testing.assert_array_equal(getattr(actual[name], field), getattr(expected_batch, field))
    for name, tensor in sampler.orchestrator.module.state_dict().items():
        torch.testing.assert_close(actual["module"][name], tensor.cpu(), rtol=0, atol=0)
    for index, state in sampler.orchestrator.optimizer.state_dict()["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(actual["optimizer"]["state"][index][name], tensor.cpu(), rtol=0, atol=0)
    assert actual["scheduler"] == sampler.orchestrator.scheduler.state_dict()
    assert actual["step_devices"] == [str(state["step"].device) for state in sampler.orchestrator.optimizer.state.values()]
    assert actual["training"] is False
