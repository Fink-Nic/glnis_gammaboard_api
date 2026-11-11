"""Automatic selection uses visible CUDA order, process occupancy, then free VRAM."""
import subprocess
from types import SimpleNamespace

import pytest
import torch

from glnis.core.nn import orchestrator
from glnis.core.nn.orchestrator import RectangularOrchestrator, RectangularOrchestratorConfig


@pytest.fixture
def cuda_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    # Physical devices C, A, B are visible in that order.
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: SimpleNamespace(uuid=("c", "a", "b")[index]))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: ((6, 9, 2)[index], 10))
    sampler = object.__new__(RectangularOrchestrator)
    sampler.cfg = RectangularOrchestratorConfig()
    return sampler


@pytest.mark.parametrize("busy,expected", [({"a"}, 0), ({"a", "c"}, 2), ({"a", "b", "c"}, 1)])
def test_auto_prefers_first_idle_visible_gpu_then_most_free(cuda_devices, monkeypatch, busy, expected):
    report = "<nvidia_smi_log>" + "".join(
        f"<gpu><uuid>GPU-{uuid}</uuid><processes>"
        + ("<process_info><pid>123</pid></process_info>" if uuid in busy else "")
        + "</processes></gpu>" for uuid in ("a", "b", "c")) + "</nvidia_smi_log>"
    monkeypatch.setattr(orchestrator.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=report))
    assert cuda_devices._get_device() == torch.device(f"cuda:{expected}")


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 5),
                                   subprocess.CalledProcessError(1, "nvidia-smi"), "invalid XML", "<nvidia_smi_log/>",
                                   "<nvidia_smi_log><gpu><uuid>GPU-c</uuid><processes>N/A</processes></gpu></nvidia_smi_log>"])
def test_missing_process_information_falls_back_to_free_vram(cuda_devices, monkeypatch, failure):
    def report(*args, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return SimpleNamespace(stdout=failure)
    monkeypatch.setattr(orchestrator.subprocess, "run", report)
    assert cuda_devices._get_device() == torch.device("cuda:1")


def test_free_memory_ties_choose_first_visible_gpu(cuda_devices, monkeypatch):
    monkeypatch.setattr(orchestrator.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="<nvidia_smi_log/>"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (10, 20))
    assert cuda_devices._get_device() == torch.device("cuda:0")


def test_explicit_gpu_and_cpu_do_not_probe_processes(cuda_devices, monkeypatch):
    monkeypatch.setattr(orchestrator.subprocess, "run", lambda *args, **kwargs: pytest.fail("unnecessary process query"))
    cuda_devices.cfg.gpu_id = 2
    assert cuda_devices._get_device() == torch.device("cuda:2")
    cuda_devices.cfg.gpu_id = 3
    with pytest.raises(ValueError, match="visible CUDA"):
        cuda_devices._get_device()
    cuda_devices.cfg.use_gpu = False
    assert cuda_devices._get_device() == torch.device("cpu")
    cuda_devices.cfg.use_gpu = True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cuda_devices._get_device() == torch.device("cpu")


@pytest.mark.parametrize("gpu_id", [-1, True, False, 1.0, "0", "available", None])
def test_gpu_id_validation(gpu_id):
    with pytest.raises(ValueError, match="gpu_id"):
        RectangularOrchestratorConfig(gpu_id=gpu_id)


@pytest.mark.parametrize("settings", [{"cuda_id": 0}, {"loss_type": "variance"}, {"loss_kwargs": {}},
                                     {"use_scheduler": False}, {"scheduler_type": "step"}, {"sampler_kwargs": {}}])
def test_retired_configuration_keys_are_rejected(settings):
    with pytest.raises(TypeError, match="unexpected keyword"):
        RectangularOrchestratorConfig.from_kwargs(**settings)
