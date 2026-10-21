from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from glnis.core.config import RunConfig
from glnis.scripts.comparison_config import ComparisonConfig, ComparisonPlotConfig
from madnis_sampler import MadnisSampler


def settings() -> dict:
    return {
        "integrand": {"kind": "gaussian", "continuous_dims": 2},
        "aggregator": {"mode": "train_then_infer", "inference_samples": 0},
        "sampler": {
            "kind": "madnis", "use_gpu": False, "training_steps": 0,
            "flow_config": {"layers": 2, "units": 16, "bins": 4},
        },
        "samplers": [{"name": "baseline"}, {"name": "wider", "flow_config": {"units": 32}}],
    }


def graph_settings() -> dict:
    return {
        "edge_masses": [0.0], "edge_momentum_shifts": [[0.0, 0.0, 0.0]],
        "graph_signature": [[1]], "lmb_edges": [[0]], "generation_lmb_edges": [0],
    }


def test_defaults_and_nested_sampler_overrides(tmp_path: Path):
    raw = settings()
    original = deepcopy(raw)
    config = ComparisonConfig.from_dict(raw, base_dir=tmp_path)
    assert [name for name, _ in config.named_runs] == ["baseline", "wider"]
    baseline, wider = [run for _, run in config.named_runs]
    assert baseline.sampler["flow_config"]["units"] == 16
    assert wider.sampler["flow_config"]["units"] == 32
    assert wider.sampler["flow_config"]["bins"] == 4
    assert wider.sampler["flow_config"]["layers"] == 2
    assert config.run_name == "sampler_comparison"
    assert config.output_dir == tmp_path / "outputs/sampler_comparison"
    assert config.save_data is True and config.plot is True
    assert config.save_checkpoints is False
    assert config.slice_plot is None
    assert config.comment == ""
    assert config.plotting.dpi == 150
    assert config.plotting.figsize == (10.0, 8.0)
    assert raw == original
    baseline.sampler["flow_config"]["units"] = 999
    assert wider.sampler["flow_config"]["units"] == 32


def test_mapping_lists_replace_instead_of_concatenating():
    raw = settings()
    raw["sampler"].update(
        graph_properties=graph_settings(),
        mapping={"input_space": "HCUBE", "layer_cfgs": [{"kind": "spherical"}, {"kind": "identity"}]},
    )
    raw["samplers"][1]["mapping"] = {"layer_cfgs": [{"kind": "identity", "force_dim": 2}]}
    config = ComparisonConfig.from_dict(raw)
    first, second = [run for _, run in config.named_runs]
    assert len(first.sampler["mapping"]["layer_cfgs"]) == 2
    assert second.sampler["mapping"]["layer_cfgs"] == [{"kind": "identity", "force_dim": 2}]
    assert second.sampler["mapping"]["input_space"] is first.sampler["mapping"]["input_space"]


@pytest.mark.parametrize("entries, match", [
    ([], "nonempty list"),
    ({"name": "one"}, "nonempty list"),
    ([{}], r"samplers\[0\].name"),
    ([{"name": " "}], r"samplers\[0\].name"),
    ([{"name": 1}], r"samplers\[0\].name"),
    ([{"name": "same"}, {"name": "same"}], "duplicates"),
])
def test_named_runs_require_unique_nonempty_labels(entries, match):
    with pytest.raises(ValueError, match=match):
        ComparisonConfig.from_dict({**settings(), "samplers": entries})


def test_labels_are_display_names_and_do_not_determine_output_paths(tmp_path: Path):
    raw = settings()
    raw["samplers"] = [{"name": "../display/label with spaces"}]
    config = ComparisonConfig.from_dict(raw, base_dir=tmp_path)
    assert config.named_runs[0][0] == "../display/label with spaces"
    assert config.output_dir == tmp_path / "outputs/sampler_comparison"


@pytest.mark.parametrize("name", ["", " ", ".", "..", "../escape", "sub/name", "sub\\name", "bad\0name"])
def test_run_name_is_one_safe_filename_component(name):
    with pytest.raises(ValueError, match="comparison.run_name"):
        ComparisonConfig.from_dict({**settings(), "comparison": {"run_name": name}})


@pytest.mark.parametrize("changes, match", [
    ({"extra": {}}, "extra"),
    ({"comparison": {"save": True}}, "comparison.save"),
    ({"comparison": {"save_data": "yes"}}, "comparison.save_data"),
    ({"comparison": {"save_checkpoints": 1}}, "comparison.save_checkpoints"),
    ({"comparison": {"plot": 1}}, "comparison.plot"),
    ({"comparison": {"comment": 5}}, "comparison.comment"),
    ({"comparison": {"output_dir": ""}}, "comparison.output_dir"),
    ({"comparison": {"plotting": {"unknown": 1}}}, "comparison.plotting.unknown"),
    ({"sampler": {"train_steps": 3}}, "sampler.train_steps"),
    ({"samplers": [{"name": "one", "integrand": {"kind": "sine"}}]}, "sampler.integrand"),
    ({"integrand": {"kind": "gaussian", "continuous_dims": 2, "sigam": 1.0}}, "integrand.*sigam"),
    ({"slice_plot": []}, "slice_plot"),
])
def test_rejects_unknown_and_invalid_settings(changes, match):
    with pytest.raises(ValueError, match=match):
        ComparisonConfig.from_dict({**settings(), **changes})


@pytest.mark.parametrize("options, field", [
    ({"dpi": 0}, "dpi"),
    ({"dpi": True}, "dpi"),
    ({"dpi": 2.5}, "dpi"),
    ({"figsize": [10]}, "figsize"),
    ({"figsize": [10, 0]}, "figsize"),
    ({"figsize": [10, True]}, "figsize"),
    ({"figsize": [10, float("inf")]}, "figsize"),
])
def test_plotting_settings_validate_values(options, field):
    with pytest.raises(ValueError, match=f"comparison.plotting.{field}"):
        ComparisonPlotConfig.from_dict(options)
    with pytest.raises(ValueError, match=f"comparison.plotting.{field}"):
        ComparisonPlotConfig(**options)


def test_all_entries_are_validated_before_any_sampler_construction(monkeypatch):
    constructed = []

    def unexpected_constructor(*args, **kwargs):
        constructed.append(True)
        raise AssertionError("Parsing must not instantiate samplers")

    monkeypatch.setattr(MadnisSampler, "__init__", unexpected_constructor)
    monkeypatch.setattr(RunConfig, "build", unexpected_constructor)
    raw = settings()
    raw["samplers"][1]["flow_config"]["units"] = 0
    with pytest.raises(ValueError, match=r"samplers\[1\].*wider.*sampler.flow_config.units"):
        ComparisonConfig.from_dict(raw)
    assert constructed == []
    assert len(ComparisonConfig.from_dict(settings()).named_runs) == 2
    assert constructed == []


@pytest.mark.parametrize("mode", ["train", "infer", "train_then_infer"])
def test_zero_sample_modes_are_valid(mode):
    raw = settings()
    raw["aggregator"]["mode"] = mode
    config = ComparisonConfig.from_dict(raw)
    assert config.aggregator.mode == mode
    assert config.aggregator.inference_samples == 0
    assert all(run.sampler["training_steps"] == 0 for _, run in config.named_runs)


def test_inference_mode_rejects_fresh_sampler_that_needs_training():
    raw = settings()
    raw["aggregator"]["mode"] = "infer"
    raw["samplers"][1]["training_steps"] = 2
    with pytest.raises(ValueError, match=r"samplers\[1\].*training_steps.*infer"):
        ComparisonConfig.from_dict(raw)


def test_toml_resolves_shared_and_per_run_paths_relative_to_file(tmp_path: Path, monkeypatch):
    directory = tmp_path / "configuration"
    directory.mkdir()
    source = directory / "comparison.toml"
    source.write_text('''
[comparison]
run_name = "example"
output_dir = "../results"
comment = "a comparison"
save_data = false
[slice_plot]
comparison_file = "../previous/comparison.pkl"
output_dir = "../slices"
[comparison.plotting]
dpi = 200
figsize = [12, 9]
[integrand]
kind = "sine"
[aggregator]
mode = "infer"
[sampler]
training_steps = 0
save_path = "checkpoints/shared.pt"
[sampler.parser]
gammaloop_directory = "../gammaloop"
[sampler.evaluator_metadata]
kind = "gammaloop"
state_folder = "states/shared"
process_id = 0
integrand_name = "example"
[[samplers]]
name = "first"
[[samplers]]
name = "second"
save_path = "../other.pt"
[samplers.evaluator_metadata]
state_folder = "states/second"
''')
    monkeypatch.chdir(tmp_path)
    config = ComparisonConfig.from_toml(source)
    first, second = [run.sampler for _, run in config.named_runs]
    assert config.output_dir == tmp_path / "results"
    assert config.comment == "a comparison"
    assert config.save_data is False
    assert config.slice_plot["comparison_file"] == str(tmp_path / "previous/comparison.pkl")
    assert config.slice_plot["output_dir"] == str(tmp_path / "slices")
    assert config.plotting.dpi == 200
    assert config.plotting.figsize == (12.0, 9.0)
    assert first["save_path"] == str(directory / "checkpoints/shared.pt")
    assert second["save_path"] == str(tmp_path / "other.pt")
    assert first["evaluator_metadata"]["state_folder"] == str(directory / "states/shared")
    assert second["evaluator_metadata"]["state_folder"] == str(directory / "states/second")
    assert second["parser"]["gammaloop_directory"] == str(tmp_path / "gammaloop")
    assert not config.output_dir.exists()


def test_dict_paths_resolve_against_cwd_and_saved_settings_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = settings()
    raw["sampler"]["save_path"] = "checkpoints/run.pt"
    config = ComparisonConfig.from_dict(raw)
    assert config.named_runs[0][1].sampler["save_path"] == str(tmp_path / "checkpoints/run.pt")
    snapshot = config.settings
    restored = ComparisonConfig.from_dict(snapshot)
    assert restored.settings == snapshot
    snapshot["integrand"]["continuous_dims"] = 9
    snapshot["samplers"][0]["flow_config"]["units"] = 999
    assert config.integrand["continuous_dims"] == 2
    assert config.named_runs[0][1].sampler["flow_config"]["units"] == 16


def test_shared_sampler_and_aggregator_defaults_are_optional():
    config = ComparisonConfig.from_dict({
        "integrand": {"kind": "sine"}, "samplers": [{"name": "default"}],
    })
    assert config.named_runs[0][1].sampler["kind"] == "default"
    assert config.aggregator.mode == "train_then_infer"


def test_checkpoint_and_slice_settings_roundtrip_without_mutating_input():
    raw = settings()
    raw["comparison"] = {"save_checkpoints": True}
    raw["slice_plot"] = {"slices": [{"continuous": [0, 1], "fixed_continuous": [0.5, 0.5]}]}
    original = deepcopy(raw)
    config = ComparisonConfig.from_dict(raw)
    assert config.save_checkpoints is True
    assert config.slice_plot == raw["slice_plot"]
    assert ComparisonConfig.from_dict(config.settings).settings == config.settings
    config.settings["slice_plot"]["slices"].append({})
    assert raw == original
    assert config.slice_plot == original["slice_plot"]
