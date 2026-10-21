"""Comparison execution uses fresh sequential runs and existing history only."""

from copy import deepcopy
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from glnis.core.accumulator import DefaultAccumulator, TrainingAccumulator
from glnis.core.config import RunConfig
from glnis.scripts.sampler_comparison import (
    SamplerCompData, TrainingProgress, load_comparison, run_sampler_comp,
)
from glnis.utils.types import LayerData, Result


def sample_data(values):
    data = LayerData(n_points=len(values), n_cont=1)
    data.continuous = np.arange(len(values))[:, None]
    data.func_val = np.asarray(values)
    data.update()
    return data


def history_record(values, step, phase="real", loss=1.0):
    accumulator = TrainingAccumulator(sample_data(values), training_phase=phase)
    result = accumulator.training_result
    return dict(step=step, n_samples=len(values), loss=loss, training_phase=phase,
                training_result={name: float(getattr(result, name)) for name in ("mean", "error", "rsd", "tvar")},
                observables=accumulator.get_observables())


def settings(tmp_path):
    return {
        "comparison": {"output_dir": str(tmp_path / "results"), "run_name": "test", "plot": False},
        "integrand": {"kind": "sine", "continuous_dims": 2, "offset": 2.0},
        "aggregator": {"inference_samples": 7, "log_every": 0},
        "sampler": {"use_gpu": False, "training_steps": 2, "training_batch_size": 4,
                    "max_batch_size": 3, "flow_config": {"layers": 2, "units": 8, "bins": 4}},
        "samplers": [{"name": "narrow", "seed": 1}, {"name": "wide", "seed": 2, "training_batch_size": 6}],
    }


@pytest.mark.parametrize("phase", ["real", "imag", "abs"])
def test_progress_uses_recorded_phase_and_absolute_statistics(phase):
    first = np.array([1 + 5j, -4 - 2j, 2 - 4j])
    second = np.array([-1 - 2j, 2 + 1j])
    history = [history_record(first, 1, phase), history_record(second, 2, phase, loss=None)]
    original = deepcopy(history)
    progress = TrainingProgress.from_history(history, phase)
    assert progress.nspl_losses == progress.nspl_snapshot == [3, 5]
    assert progress.steps_losses == progress.steps_snapshot == [1, 2]
    assert progress.losses == [1.0, None]
    for index, values in enumerate((first, second)):
        selected = np.abs(values) if phase == "abs" else getattr(values, phase)
        absolute = np.abs(selected)
        assert progress.means[index] == pytest.approx(selected.mean())
        assert progress.errors[index] == pytest.approx(selected.std(ddof=1) / np.sqrt(len(values)))
        assert progress.rsds[index] == pytest.approx(selected.std(ddof=1) / abs(selected.mean()))
        assert progress.abs_rsds[index] == pytest.approx(absolute.std(ddof=1) / absolute.mean())
    if phase != "abs":
        assert progress.rsds != progress.abs_rsds
    assert not progress.steps_discrete and not progress.discrete_probs
    assert progress.all_channels.size == 0
    assert history[0]["training_result"] == original[0]["training_result"]


def install_fake_runs(monkeypatch, events, *, fail_seed=None):
    references = []

    class FakeAggregator:
        def __init__(self, run_config):
            self.config = run_config
            self.seed = run_config.sampler["seed"]
            self.integrand = SimpleNamespace(target=Result.from_kwargs(real_mean=0.0))
            self.sampler = SimpleNamespace(get_diagnostics=lambda: {"seed": self.seed})
            self.training_history = []
            self.inference_history = []

        def run(self):
            events.append(("run", self.seed))
            if self.seed == fail_seed:
                raise RuntimeError("simulated sampler failure")
            for step in range(1, self.config.sampler["training_steps"] + 1):
                n = self.config.sampler["training_batch_size"]
                self.training_history.append(history_record(np.arange(n) + self.seed, step))
            n_inference = self.config.aggregator.inference_samples
            result = DefaultAccumulator(sample_data(np.full(n_inference, self.seed)))
            self.accumulator = result
            self.inference_history = [{"n_samples": n_inference, "observables": result.get_observables()}]
            return result

    def build(run_config):
        gc.collect()
        assert all(reference() is None for reference in references), "prior sampler must be released first"
        events.append(("build", run_config.sampler["seed"]))
        aggregator = FakeAggregator(run_config)
        references.append(weakref.ref(aggregator))
        return aggregator

    monkeypatch.setattr(RunConfig, "build", build)
    return references


def test_sequential_runs_preserve_dataclasses_and_persist_results(monkeypatch, tmp_path):
    from glnis.scripts import sampler_comparison

    events = []
    references = install_fake_runs(monkeypatch, events)
    writes = []
    save_comparison = sampler_comparison._save_comparison

    def record_save(path, data):
        writes.append(list(data.result))
        save_comparison(path, data)

    monkeypatch.setattr(sampler_comparison, "_save_comparison", record_save)
    config = settings(tmp_path)
    original = deepcopy(config)
    data = run_sampler_comp(config)
    assert events == [("build", 1), ("run", 1), ("build", 2), ("run", 2)]
    assert writes == [["narrow", "wide"]]
    assert all(reference() is None for reference in references)
    assert list(data.result) == ["narrow", "wide"]
    assert data.training_progress["narrow"].nspl_snapshot == [4, 8]
    assert data.training_progress["wide"].nspl_snapshot == [6, 12]
    assert data.result["narrow"].real.mean == 1
    assert data.result["wide"].real.mean == 2
    assert all(result.n_points == 7 for result in data.result.values())
    assert data.run_configs["wide"]["sampler"]["training_batch_size"] == 6
    assert all(state is None for state in data.integrator_states.values())
    assert config == original
    assert data.output_file.is_file()
    assert [path.suffix for path in data.artifact_paths] == [".pkl", ".txt"]
    loaded = load_comparison(data.output_file)
    assert isinstance(loaded, SamplerCompData)
    assert isinstance(loaded.training_progress["wide"], TrainingProgress)
    assert loaded.result["wide"] == data.result["wide"]
    assert loaded.training_progress["wide"].means == data.training_progress["wide"].means
    assert len(loaded.inference_histories["wide"]) == 1


def test_no_output_and_no_training_have_no_extra_sampling(monkeypatch, tmp_path):
    events = []
    install_fake_runs(monkeypatch, events)
    config = settings(tmp_path)
    config["sampler"]["training_steps"] = 0
    config["aggregator"]["mode"] = "infer"
    data = run_sampler_comp(config, no_output=True)
    assert len(events) == 4
    assert not (tmp_path / "results").exists()
    assert data.output_file is None
    assert data.artifact_paths == []
    assert all(not progress.steps_snapshot for progress in data.training_progress.values())
    assert all(result.n_points == 7 for result in data.result.values())


def test_failed_later_run_does_not_save_partial_comparison(monkeypatch, tmp_path):
    events = []
    references = install_fake_runs(monkeypatch, events, fail_seed=2)
    with pytest.raises(RuntimeError, match="simulated sampler failure"):
        run_sampler_comp(settings(tmp_path))
    gc.collect()
    assert all(reference() is None for reference in references)
    files = list((tmp_path / "results").glob("*.pkl"))
    assert files == []
    assert not list((tmp_path / "results").glob("*.tmp"))


def test_cli_from_toml_saves_portable_pickle_and_can_replot(tmp_path):
    config = tmp_path / "compare.toml"
    config.write_text('''
[comparison]
run_name = "smoke"
output_dir = "results"
[comparison.plotting]
dpi = 50
figsize = [8, 6]
[integrand]
kind = "sine"
continuous_dims = 2
offset = 2.0
[aggregator]
inference_samples = 7
evaluation_batch_size = 2
log_every = 0
[sampler]
training_steps = 1
training_batch_size = 5
max_batch_size = 3
use_gpu = false
[sampler.flow_config]
layers = 2
units = 8
bins = 4
[[samplers]]
name = "first"
[[samplers]]
name = "second"
training_steps = 2
''')
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "glnis.scripts.sampler_comparison"]
    completed = subprocess.run(command + [str(config)], cwd=tmp_path.parent,
                               env=environment, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.index("Running first") < completed.stdout.index("Running second")
    files = list((tmp_path / "results").glob("*.pkl"))
    assert len(files) == 1
    data = load_comparison(files[0])
    assert data.__class__.__module__ == "glnis.scripts.sampler_comparison"
    assert list(data.result) == ["first", "second"]
    assert [len(data.training_progress[name].means) for name in data.result] == [1, 2]
    assert [data.result[name].n_points for name in data.result] == [7, 7]
    assert sum(record["n_samples"] for record in data.training_histories["second"]) == 10
    for artifact in data.artifact_paths:
        assert artifact.is_file() and artifact.stat().st_size > 0
    assert len(list(files[0].parent.glob("*.png"))) == 2
    replotted = subprocess.run(command + [str(files[0]), "--replot"], cwd=tmp_path,
                               env=environment, capture_output=True, text=True, timeout=60)
    assert replotted.returncode == 0, replotted.stdout + replotted.stderr
    assert "Running " not in replotted.stdout


def test_checkpoint_setting_respects_no_output(monkeypatch, tmp_path):
    events = []
    install_fake_runs(monkeypatch, events)
    config = settings(tmp_path)
    config["comparison"]["save_checkpoints"] = True
    data = run_sampler_comp(config, no_output=True)
    assert not (tmp_path / "results").exists()
    assert data.artifact_paths == []
    assert all(state is None for state in data.integrator_states.values())


def test_restore_requires_available_checkpoint_and_configuration(tmp_path):
    data = SamplerCompData(["example"])
    with pytest.raises(KeyError, match="Unknown comparison sampler"):
        data.restore_aggregator("missing")
    with pytest.raises(ValueError, match="enable comparison.save_checkpoints"):
        data.restore_aggregator("example")
    with pytest.raises(ValueError, match="use_gpu must be bool"):
        data.restore_aggregator("example", use_gpu="yes")
    data.integrator_states["example"] = {"save_path": "checkpoint.pt"}
    with pytest.raises(ValueError, match="No run configuration"):
        data.restore_aggregator("example")
    data.run_configs["example"] = {}
    with pytest.raises(ValueError, match="comparison file location"):
        data.restore_aggregator("example")


@pytest.mark.parametrize("save_data", [False, True])
def test_completed_checkpoints_restore_all_strategies_and_survive_relocation(tmp_path, monkeypatch, save_data):
    from glnis.core.sampler_aggregator import SamplerAggregator
    from madnis_sampler import MadnisSampler, MadnisSamplerNaiveSgnStrat, MadnisSamplerFullSgnStrat
    import torch

    config = settings(tmp_path)
    config["comparison"].update(save_checkpoints=True, save_data=save_data)
    config["integrand"]["offset"] = 0.1
    config["sampler"].update(
        training_steps=1, training_batch_size=64, max_batch_size=32,
        save_path=str(tmp_path / "shared-must-not-be-written.pt"),
        transformer_config={"embedding_dim": 8, "feedforward_dim": 8, "heads": 2,
                            "mlp_units": 8, "transformer_layers": 1},
    )
    config["samplers"] = [
        {"name": "../default label", "kind": "default"},
        {"name": "naive", "kind": "sgn_strat_naive"},
        {"name": "full", "kind": "sgn_strat_full"},
    ]
    expected = []
    snapshot = SamplerAggregator.snapshot
    continuous = np.array([[0.1, 0.2], [0.8, 0.3], [0.4, 0.9], [0.7, 0.6]])

    def record_snapshot(aggregator, path=None):
        cardinalities = aggregator.pdf_discrete_cardinalities
        discrete = np.zeros((len(continuous), len(cardinalities)), dtype=np.int64)
        if aggregator.strat_cardinalities:
            discrete[:, 0] = [0, 1, 0, 1]
        expected.append((discrete, aggregator.pdf(discrete, continuous)))
        return snapshot(aggregator, path)

    monkeypatch.setattr(SamplerAggregator, "snapshot", record_snapshot)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = run_sampler_comp(config)
        assert not (tmp_path / "shared-must-not-be-written.pt").exists()
        checkpoints = [Path(state["save_path"]) for state in data.integrator_states.values()]
        assert all(not path.is_absolute() for path in checkpoints)
        assert [path.name for path in checkpoints] == ["000.pt", "001.pt", "002.pt"]
        assert len(list((tmp_path / "results").glob("*.pkl"))) == int(save_data)
        if save_data:
            destination = tmp_path / "relocated"
            shutil.move(str(tmp_path / "results"), destination)
            data = load_comparison(destination / data.output_file.name)
            assert data.output_file.parent == destination
        else:
            assert data.output_file is None
        kinds = [MadnisSampler, MadnisSamplerNaiveSgnStrat, MadnisSamplerFullSgnStrat]
        for index, (name, sampler_type) in enumerate(zip(data.result, kinds, strict=True)):
            aggregator = data.restore_aggregator(name, use_gpu=False)
            assert type(aggregator.sampler) is sampler_type
            assert aggregator.sampler.training_samples_remaining() is None
            assert aggregator.sampler.training_status.step == 1
            assert not aggregator.training_history and not aggregator.inference_history
            discrete, expected_pdf = expected[index]
            np.testing.assert_array_equal(aggregator.pdf(discrete, continuous), expected_pdf)
            assert Path(aggregator.sampler.cfg.save_path).name == f"{index:03d}.pt"
            del aggregator
            gc.collect()
    finally:
        torch.set_num_threads(previous_threads)
