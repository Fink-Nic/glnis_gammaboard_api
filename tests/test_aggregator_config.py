"""Configuration tables use the same component factories as direct construction."""
from copy import deepcopy

import pytest

from gammaboard_sampler import GammaBoardSampler
from glnis.core.config import AggregatorConfig, RunConfig


def test_build_derives_public_domain_and_preserves_configuration(sampler_settings):
    settings = {"sampler": sampler_settings, "integrand": {"kind": "gaussian", "continuous_dims": 2, "sigma": [.4, .8]},
                "aggregator": {"inference_samples": 4}}
    original = deepcopy(settings)
    config = RunConfig.from_dict(settings)
    run = config.build()
    assert isinstance(run.sampler, GammaBoardSampler)
    assert run.sampler.domain.discrete_cardinalities == (2,)
    assert run.sampler.domain.continuous_dims == 2
    assert run.generation_batch_size == 4
    assert settings == original


@pytest.mark.parametrize("field,value", [("training_steps", -1), ("training_batch_size", 0), ("max_batch_size", True), ("use_gpu", 1)])
def test_invalid_orchestrator_settings_fail_before_construction(sampler_settings, field, value):
    sampler_settings["orchestrator"][field] = value
    with pytest.raises(ValueError, match=field):
        RunConfig.from_dict({"sampler": sampler_settings, "integrand": {"kind": "sine"}})


@pytest.mark.parametrize("component", ["loss", "scheduler", "execution"])
def test_unknown_kinds_are_rejected(sampler_settings, component):
    sampler_settings["orchestrator"][component] = {"kind": "missing"}
    with pytest.raises(ValueError, match="Unknown kind"):
        RunConfig.from_dict({"sampler": sampler_settings, "integrand": {"kind": "sine"}})


def test_old_flat_sampler_settings_have_an_actionable_error():
    with pytest.raises(ValueError, match="sampler.training_steps"):
        RunConfig.from_dict({"sampler": {"training_steps": 1}, "integrand": {"kind": "sine"}})


def test_module_configuration_has_one_adapter_route(sampler_settings):
    config = RunConfig.from_dict({"sampler": sampler_settings, "integrand": {"kind": "sine"}})
    sampler = config.build().sampler
    assert sampler.cfg.module == sampler_settings["module"]
    assert sampler.orchestrator.cfg.module_kwargs == sampler_settings["module"]
    assert "module_kwargs" not in config.sampler["orchestrator"]
    assert "module_kwargs" not in sampler.cfg.orchestrator
    for settings in ({"sampler": sampler_settings["module"]},
                     {"orchestrator": {"module_kwargs": sampler_settings["module"]}}):
        with pytest.raises(ValueError, match="module|sampler.sampler"):
            RunConfig.from_dict({"sampler": settings, "integrand": {"kind": "sine"}})


def test_toml_paths_are_relative_to_configuration(tmp_path, monkeypatch):
    source = tmp_path / "config.toml"
    source.write_text('''[sampler]
save_path = "checkpoints/sampler.pt"
[sampler.orchestrator]
training_steps = 0
use_gpu = false
[sampler.evaluator_metadata]
kind = "gammaloop"
state_folder = "state"
process_id = 0
integrand_name = "test"
[integrand]
kind = "sine"
''')
    monkeypatch.chdir(tmp_path.parent)
    config = RunConfig.from_toml(source)
    assert config.sampler["save_path"] == str(tmp_path / "checkpoints/sampler.pt")
    assert config.sampler["evaluator_metadata"]["state_folder"] == str(tmp_path / "state")


def test_mapping_schema_and_output_dimension_validation(sampler_settings, graph_settings):
    sampler_settings.update(mapping={"input_space": "HCUBE", "layer_cfgs": [{"kind": "spherical"}]}, graph_properties=graph_settings)
    config = RunConfig.from_dict({"sampler": sampler_settings, "integrand": {"kind": "gaussian", "continuous_dims": 3}})
    assert config.sampler["mapping"]["input_space"] == "HCUBE"
    assert config.build().sampler.mapping is not None
    with pytest.raises(ValueError, match="output dimension"):
        RunConfig.from_dict({"sampler": sampler_settings, "integrand": {"kind": "gaussian", "continuous_dims": 2}}).build()


@pytest.mark.parametrize("settings", [{"evaluation_workers": 0}, {"inference_samples": -1}, {"training_phase": "missing"}, {"mode": "missing"}])
def test_aggregator_validation(settings):
    with pytest.raises(ValueError):
        AggregatorConfig.from_dict(settings)
