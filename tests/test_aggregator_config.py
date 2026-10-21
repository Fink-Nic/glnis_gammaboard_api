from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import madnis_sampler

from glnis.core.config import AggregatorConfig, RunConfig
from glnis.core.mappings import Space
from madnis_sampler import FlowConfig, MadnisConfig, MadnisSampler


def run_settings() -> dict:
    return {
        "sampler": {
            "kind": "madnis", "use_gpu": False, "training_steps": 0,
            "max_batch_size": 8, "gb_loglevel": "error",
            "flow_config": {"layers": 2, "units": 8, "bins": 4},
        },
        "integrand": {"kind": "gaussian", "continuous_dims": 2},
        "aggregator": {"mode": "infer", "inference_samples": 7, "log_every": 0},
    }


def graph_settings() -> dict:
    return {
        "edge_masses": [0.0], "edge_momentum_shifts": [[0.0, 0.0, 0.0]],
        "graph_signature": [[1]], "lmb_edges": [[0]], "generation_lmb_edges": [0],
    }


@pytest.mark.parametrize("settings, field", [
    ({"evaluation_workers": 0}, "evaluation_workers"),
    ({"generation_batch_size": True}, "generation_batch_size"),
    ({"evaluation_batch_size": 1.5}, "evaluation_batch_size"),
    ({"max_in_flight_batches": -1}, "max_in_flight_batches"),
    ({"inference_samples": -1}, "inference_samples"),
    ({"log_every": -1}, "log_every"),
    ({"mode": "training"}, "mode"),
    ({"training_phase": "complex"}, "training_phase"),
])
def test_aggregator_config_validates_direct_and_dict_construction(settings, field):
    with pytest.raises(ValueError, match=f"aggregator.{field}"):
        AggregatorConfig(**settings)
    with pytest.raises(ValueError, match=f"aggregator.{field}"):
        AggregatorConfig.from_dict(settings)


def test_defaults_and_unknown_settings():
    defaults = AggregatorConfig()
    assert defaults.generation_batch_size is None
    assert defaults.log_every == 10
    with pytest.raises(ValueError, match="aggregator.worker"):
        AggregatorConfig.from_dict({"worker": 1})
    with pytest.raises(ValueError, match="extra"):
        RunConfig.from_dict({**run_settings(), "extra": {}})
    with pytest.raises(ValueError, match=r"Missing \[integrand\]"):
        RunConfig.from_dict({"sampler": {}})
    with pytest.raises(ValueError, match="sampler.kind"):
        RunConfig.from_dict({**run_settings(), "sampler": {"kind": "unknown"}})


@pytest.mark.parametrize("kind, expected", [
    (None, "default"),
    ("madnis", "default"),
    ("default", "default"),
    ("sgn_strat_naive", "sgn_strat_naive"),
    ("sgn_strat_full", "sgn_strat_full"),
])
def test_sampler_strategy_kind_from_dict_and_toml(tmp_path, kind, expected):
    settings = run_settings()
    settings["sampler"].pop("kind")
    if kind is not None:
        settings["sampler"]["kind"] = kind
    original = deepcopy(settings)
    config = RunConfig.from_dict(settings)
    assert config.sampler["kind"] == expected
    assert settings == original
    # Already-normalized configurations must retain their selected strategy.
    assert RunConfig(config.sampler, config.integrand, config.aggregator).sampler == config.sampler

    source = tmp_path / "strategy.toml"
    kind_line = "" if kind is None else f'kind = "{kind}"\n'
    source.write_text('[sampler]\n' + kind_line + '[integrand]\nkind = "gaussian"\n')
    assert RunConfig.from_toml(source).sampler["kind"] == expected


@pytest.mark.parametrize("kind, class_name", [
    ("default", "MadnisSampler"),
    ("sgn_strat_naive", "MadnisSamplerNaiveSgnStrat"),
    ("sgn_strat_full", "MadnisSamplerFullSgnStrat"),
])
def test_build_dispatches_selected_sampler_strategy(monkeypatch, kind, class_name):
    selected_class = getattr(madnis_sampler, class_name)
    constructor_calls = []

    def initialize(self, *, continuous_dims, discrete_cardinalities, cfg, **values):
        # Exercise the real strategy factory without initializing unfinished
        # stratified models or relying on their generation implementations.
        constructor_calls.append(values)
        self.continuous_dims = continuous_dims
        self.discrete_cardinalities = discrete_cardinalities
        self.cfg = cfg
        self.mapping = None

    monkeypatch.setattr(selected_class, "__init__", initialize)
    settings = run_settings()
    settings["sampler"]["kind"] = kind
    aggregator = RunConfig.from_dict(settings).build()
    assert type(aggregator.sampler) is selected_class
    assert aggregator.sampler.cfg.kind == kind
    assert aggregator.sampler.continuous_dims == 2
    assert aggregator.sampler.discrete_cardinalities == []
    assert constructor_calls == [{}]


@pytest.mark.parametrize("settings, field", [
    ({"batch_size": 5}, "sampler.batch_size"),
    ({"training_steps": True}, "sampler.training_steps"),
    ({"training_batch_size": 0}, "sampler.training_batch_size"),
    ({"max_batch_size": -1}, "sampler.max_batch_size"),
    ({"use_gpu": "false"}, "sampler.use_gpu"),
    ({"learning_rate": float("nan")}, "sampler.learning_rate"),
    ({"flow_config": {"unit": 16}}, "sampler.flow_config.unit"),
    ({"flow_config": {"units": False}}, "sampler.flow_config.units"),
])
def test_madnis_config_rejects_invalid_and_unknown_settings(settings, field):
    with pytest.raises(ValueError, match=field):
        MadnisConfig.from_dict(settings)


def test_madnis_config_preserves_nested_configs_and_loglevel():
    config = MadnisConfig.from_dict({"gb_loglevel": "error", "flow_config": {"units": 16}})
    assert config.gb_loglevel == "error"
    assert isinstance(config.flow_config, FlowConfig)
    assert config.flow_config.units == 16
    with pytest.raises(ValueError, match="sampler.training_steps"):
        MadnisConfig(training_steps=-1)


def test_toml_paths_resolve_against_file_directory(tmp_path: Path, monkeypatch):
    directory = tmp_path / "configuration"
    directory.mkdir()
    source = directory / "run.toml"
    source.write_text('''
[sampler]
kind = "madnis"
save_path = "checkpoints/model.pt"
[sampler.parser]
gammaloop_directory = "../gammaloop"
[sampler.evaluator_metadata]
kind = "gammaloop"
state_folder = "states/example"
process_id = 0
integrand_name = "example"
[integrand]
kind = "gaussian"
continuous_dims = 2
''')
    monkeypatch.chdir(tmp_path)
    config = RunConfig.from_toml(source)
    assert config.sampler["save_path"] == str(directory / "checkpoints/model.pt")
    assert config.sampler["parser"]["gammaloop_directory"] == str(tmp_path / "gammaloop")
    assert config.sampler["evaluator_metadata"]["state_folder"] == str(directory / "states/example")


def test_mapping_normalization_and_unknown_options():
    settings = run_settings()
    settings["sampler"].update(
        graph_properties=graph_settings(),
        mapping={"input_space": "hcube", "layer_cfgs": [{"kind": "spherical"}]},
    )
    original = deepcopy(settings)
    config = RunConfig.from_dict(settings)
    assert config.sampler["mapping"]["input_space"] is Space.HCUBE
    assert settings == original
    settings["sampler"]["mapping"]["layer_cfgs"][0]["conformal_sclae"] = 2.0
    with pytest.raises(ValueError, match="conformal_sclae"):
        RunConfig.from_dict(settings)


def test_nested_mapping_options_are_validated_recursively():
    settings = run_settings()
    nested = {
        "kind": "spherical", "input_space": "momentum",
        "cfg": {"input_space": "leave_mapping_specific_data_unchanged"},
    }
    settings["sampler"].update(
        graph_properties=graph_settings(),
        mapping={"layer_cfgs": [{"kind": "ose", "mapping_kwargs": nested}]},
    )
    config = RunConfig.from_dict(settings)
    normalized = config.sampler["mapping"]["layer_cfgs"][0]["mapping_kwargs"]
    assert normalized["input_space"] is Space.MOMENTUM
    assert normalized["cfg"] == nested["cfg"]
    nested["conformal_sclae"] = 2.0
    with pytest.raises(ValueError, match=r"layer_cfgs\[0\].mapping_kwargs.conformal_sclae"):
        RunConfig.from_dict(settings)
    del nested["conformal_sclae"]
    nested["input_space"] = "invalid"
    with pytest.raises(ValueError, match=r"layer_cfgs\[0\].mapping_kwargs.input_space"):
        RunConfig.from_dict(settings)


def test_build_derives_dimensions_and_preserves_settings():
    aggregator = RunConfig.from_dict(run_settings()).build()
    assert aggregator.sampler.continuous_dims == 2
    assert aggregator.sampler.discrete_cardinalities == []
    assert aggregator.sampler.cfg.gb_loglevel == "error"
    assert aggregator.sampler.cfg.flow_config.units == 8
    assert aggregator.sampler.get_diagnostics()["total_trained_samples"] == 0


def test_build_rejects_mismatched_mapping_dimensions():
    settings = run_settings()
    settings["sampler"].update(
        graph_properties=graph_settings(), mapping={"layer_cfgs": [{"kind": "spherical"}]},
    )
    with pytest.raises(ValueError, match=r"sampler.mapping output dimension 3.*integrand.continuous_dims=2"):
        RunConfig.from_dict(settings).build()


def test_build_rejects_invalid_integrand_config():
    settings = run_settings()
    settings["integrand"]["sigam"] = 0.2
    with pytest.raises(ValueError, match="integrand.*sigam"):
        RunConfig.from_dict(settings).build()


def test_diagnostics_with_gammaloop_metadata():
    metadata = {
        "kind": "gammaloop", "state_folder": "/example/state",
        "process_id": 0, "integrand_name": "test",
    }
    sampler = MadnisSampler(
        continuous_dims=2, discrete_cardinalities=[], use_gpu=False, training_steps=0,
        flow_config={"layers": 2, "units": 8, "bins": 4}, evaluator_metadata=metadata,
    )
    assert sampler.get_diagnostics()["gammaloop_metadata"]["state_folder"] == "/example/state"
