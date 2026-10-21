from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from glnis.core.integrand import SineIntegrand
from glnis.core.sampler_aggregator import SamplerAggregator
from glnis.scripts.sampler_comparison import SamplerCompData, run_sampler_comp
from glnis.scripts.slice_plots import (
    SlicePlotConfig, SlicePlotData, _geometry, load_slice_data, run_slice_plots,
)


def slice_settings():
    return {"slice_plot": {
        "normalize": False,
        "batch_size": 3,
        "plotting": {"dpi": 40, "plot_singles": False},
        "slices": [
            {"name": "plain-line", "origin": [0.5, 0.3], "discrete": [], "dirs": [0],
             "grid": [{"start": 0, "stop": 1, "num": 9}]},
            {"name": "positive-plane", "origin": [0.5, 0.5], "discrete": [0], "dirs": [0, 1],
             "grid": [{"start": 0.1, "stop": 0.9, "num": 5}, {"start": 0.2, "stop": 0.8, "num": 4}]},
            {"name": "negative-line", "origin": [0.5, 0.6], "discrete": [1], "dirs": [0],
             "grid": [{"start": 0.1, "stop": 0.9, "num": 7}]},
            {"name": "wrong-dimension", "origin": [0.1, 0.2, 0.3], "dirs": [0]},
            {"name": "wrong-channel", "origin": [0.1, 0.2], "discrete": [2], "dirs": [0]},
            {"name": "wrong-direction", "origin": [0.1, 0.2], "dirs": [2]},
        ],
    }}


@pytest.fixture(scope="module")
def comparison(tmp_path_factory):
    directory = tmp_path_factory.mktemp("slice-comparison")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return run_sampler_comp({
            "comparison": {"output_dir": str(directory), "save_checkpoints": True, "plot": False},
            "integrand": {"kind": "sine", "continuous_dims": 2, "omega": 2, "offset": 0.1},
            "aggregator": {"inference_samples": 16, "evaluation_batch_size": 4, "log_every": 0},
            "sampler": {"training_steps": 1, "training_batch_size": 16, "max_batch_size": 8,
                        "use_gpu": False, "flow_config": {"layers": 2, "units": 8, "bins": 4},
                        "transformer_config": {"embedding_dim": 8, "feedforward_dim": 8,
                                               "heads": 2, "mlp_units": 8, "transformer_layers": 1}},
            "samplers": [{"name": "default"}, {"name": "naive", "kind": "sgn_strat_naive"},
                         {"name": "full", "kind": "sgn_strat_full"}],
            **slice_settings(),
        })
    finally:
        torch.set_num_threads(threads)


def test_recovered_slices_skip_incompatible_samplers_and_never_sample(comparison, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Slice plots must not train or draw samples")

    monkeypatch.setattr(SamplerAggregator, "run", forbidden)
    data = run_slice_plots(comparison, no_output=True)
    assert isinstance(data, SlicePlotData)
    assert data.output_file is None and data.artifact_paths == []
    slices = {item.name: item for item in data.slices1d + data.slices2d}
    assert set(slices["plain-line"].prob) == {"default"}
    assert set(slices["positive-plane"].prob) == {"naive", "full"}
    assert set(slices["negative-line"].prob) == {"naive", "full"}
    assert not slices["wrong-dimension"].prob
    assert not slices["wrong-channel"].prob
    assert set(data.skipped["wrong-direction"]) == {"default", "naive", "full"}
    assert set(data.skipped["plain-line"]) == {"naive", "full"}
    assert set(data.skipped["positive-plane"]) == {"default"}

    line = slices["plain-line"]
    np.testing.assert_array_equal(line.origin, [0, 0.3])
    assert np.isnan(line.prob["default"][[0, -1]]).all()
    points = np.column_stack([np.linspace(0, 1, 9)[1:-1], np.full(7, 0.3)])
    np.testing.assert_allclose(line.func_val[1:-1], SineIntegrand(2, omega=2, offset=0.1)(points))
    for label in ("naive", "full"):
        assert slices["positive-plane"].prob[label].shape == (4, 5)
        assert np.all(slices["positive-plane"].func_vals[label] >= 0)
        assert np.all(slices["negative-line"].func_vals[label] <= 0)
    assert np.any(slices["positive-plane"].func_val == 0)
    assert np.any(slices["positive-plane"].func_val > 0)

    plane = deepcopy(slices["positive-plane"])
    points, shape = _geometry(plane)
    for label in ("naive", "full"):
        restored = comparison.restore_aggregator(label, use_gpu=False)
        expected = restored.pdf(np.zeros((len(points), 1), dtype=np.int64), points)
        np.testing.assert_array_equal(slices["positive-plane"].prob[label], expected.reshape(shape))


def test_vector_directions_preserve_origin_and_grid_orientation():
    config = SlicePlotConfig.from_dict({"slice_plot": {"slices": [{
        "origin": [0.2, 0.4], "dirs": [[0.2, 0.1]],
        "grid": [{"start": 0, "stop": 1, "num": 3}],
    }]}})
    points, shape = _geometry(config.slices[0])
    np.testing.assert_allclose(points, [[0.2, 0.4], [0.3, 0.45], [0.4, 0.5]])
    assert shape == (3,)


@pytest.mark.parametrize("setting, value", [
    ("EPS", -1), ("EPS", 0.5), ("domain", "bad"), ("normalize", "true"),
    ("normalization", 0), ("normalization", float("nan")), ("batch_size", 0),
    ("use_gpu", 1), ("save_data", 1), ("run_name", "../bad"),
    ("plotting", {"dpi": 0}), ("plotting", {"cmap_name": "not-a-map"}),
    ("plotting", {"plot_singles": "false"}), ("unknown", True),
])
def test_invalid_slice_settings_are_rejected(setting, value):
    settings = slice_settings()
    settings["slice_plot"][setting] = value
    with pytest.raises(ValueError):
        SlicePlotConfig.from_dict(settings)


def test_slice_config_paths_and_array_shapes(tmp_path):
    path = tmp_path / "slices.toml"
    path.write_text('''
[slice_plot]
comparison_file = "runs/data.pkl"
output_dir = "plots"
[[slice_plot.slices]]
name = "plane"
origin = [0.5, 0.5]
discrete = []
dirs = [0, 1]
grid = [{num = 5}]
''')
    config = SlicePlotConfig.from_toml(path)
    assert config.comparison_file == tmp_path / "runs/data.pkl"
    assert config.output_dir == tmp_path / "plots"
    assert len(config.slices[0].grid) == 2
    assert config.slices[0].grid[0].num == 5


def test_no_checkpoint_reports_actionable_error(comparison):
    data = deepcopy(comparison)
    data.integrator_states = dict.fromkeys(data.integrator_states)
    with pytest.raises(ValueError, match="save_checkpoints"):
        run_slice_plots(data, no_output=True)


def test_normalizer_is_applied_once(comparison, tmp_path):
    settings = tmp_path / "normalized.toml"
    settings.write_text('''
[slice_plot]
normalize = true
normalization = 2.0
[[slice_plot.slices]]
origin = [0.5, 0.3]
discrete = []
dirs = [0]
grid = [{start = 0.1, stop = 0.9, num = 5}]
''')
    data = run_slice_plots(comparison, settings_file=settings, no_output=True)
    assert data.itg == 2
    points = np.column_stack([np.linspace(0.1, 0.9, 5), np.full(5, 0.3)])
    expected = SineIntegrand(2, omega=2, offset=0.1)(points) / 2
    np.testing.assert_allclose(data.slices1d[0].func_val, expected)


def test_mapped_pdf_slices_use_physical_coordinates_outside_unit_cube(tmp_path):
    run_config = {
        "sampler": {
            "use_gpu": False, "training_steps": 0, "max_batch_size": 4,
            "flow_config": {"layers": 2, "units": 8, "bins": 4},
            "mapping": {"layer_cfgs": [{"kind": "spherical"}]},
            "graph_properties": {"edge_masses": [0.0], "edge_momentum_shifts": [[0.0, 0.0, 0.0]],
                                 "graph_signature": [[1]], "lmb_edges": [[0]], "generation_lmb_edges": [0]},
        },
        "integrand": {"kind": "gaussian", "continuous_dims": 3},
        "aggregator": {"mode": "infer"},
    }
    aggregator = SamplerAggregator.from_config(run_config)
    settings = {"slice_plot": {"normalize": False, "slices": [{
        "origin": [0.0, 0.3, 0.4], "discrete": [], "dirs": [0],
        "grid": [{"start": -2, "stop": 2, "num": 8}],
    }]}}
    comparison = SamplerCompData(["mapped"], settings=settings)
    comparison.run_configs["mapped"] = run_config
    comparison.integrator_states["mapped"] = aggregator.snapshot(tmp_path / "mapped.pt")
    data = run_slice_plots(comparison, no_output=True)
    result = data.slices1d[0]
    points = np.column_stack([np.linspace(-2, 2, 8), np.full(8, 0.3), np.full(8, 0.4)])
    np.testing.assert_allclose(result.func_val, aggregator.integrand(points))
    np.testing.assert_allclose(result.prob["mapped"], aggregator.pdf(np.empty((8, 0), dtype=int), points))
    assert np.isfinite(result.prob["mapped"]).all()


def test_toml_cli_and_replot_without_checkpoint_access(comparison, tmp_path):
    config = tmp_path / "slices.toml"
    checkpoint_source = str(comparison.output_file)
    config.write_text(f'''
[slice_plot]
comparison_file = {checkpoint_source!r}
output_dir = "plots"
normalize = false
[slice_plot.plotting]
dpi = 40
plot_singles = false
[[slice_plot.slices]]
name = "plain"
origin = [0.5, 0.3]
discrete = []
dirs = [0]
grid = [{{start = 0.1, stop = 0.9, num = 9}}]
[[slice_plot.slices]]
name = "sign"
origin = [0.5, 0.5]
discrete = [0]
dirs = [0, 1]
grid = [{{start = 0.1, stop = 0.9, num = 6}}]
''')
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "glnis.scripts.slice_plot"]
    completed = subprocess.run(command + [str(config)], cwd=tmp_path.parent, env=environment,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    files = list((tmp_path / "plots").glob("*.pkl"))
    assert len(files) == 1
    data = load_slice_data(files[0])
    assert data.__class__.__module__ == "glnis.scripts.slice_plots"
    assert len(list((tmp_path / "plots").glob("*.png"))) == 2
    assert set(data.slices2d[0].prob) == {"naive", "full"}

    directory = next(iter(comparison.integrator_states.values()))["save_path"]
    directory = comparison.output_file.parent / Path(directory).parent
    moved = directory.with_name(directory.name + "_temporarily_unavailable")
    directory.rename(moved)
    try:
        completed = subprocess.run(command + [str(files[0]), "--replot"], cwd=tmp_path,
                                   env=environment, capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    finally:
        moved.rename(directory)
