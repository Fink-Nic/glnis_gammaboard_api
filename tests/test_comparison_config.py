from copy import deepcopy
from pathlib import Path

import pytest

from glnis.scripts.comparison_config import ComparisonConfig


def settings(sampler_settings):
    return {"integrand": {"kind": "sine", "continuous_dims": 2}, "sampler": sampler_settings,
            "samplers": [{"name": "small"}, {"name": "large", "module": {"flow": {"subnet": {"units": 16}}}}]}


def test_nested_component_overrides_preserve_sibling_settings(sampler_settings):
    config = settings(sampler_settings)
    original = deepcopy(config)
    result = ComparisonConfig.from_dict(config)
    first, second = [run for _, run in result.named_runs]
    assert first.sampler["module"]["flow"]["subnet"]["units"] == 8
    assert second.sampler["module"]["flow"]["subnet"]["units"] == 16
    assert second.sampler["module"]["flow"]["spline"]["bins"] == 4
    assert config == original


@pytest.mark.parametrize("names", [["same", "same"], [""], [1]])
def test_names_are_nonempty_unique_strings(sampler_settings, names):
    config = settings(sampler_settings)
    config["samplers"] = [{"name": name} for name in names]
    with pytest.raises(ValueError):
        ComparisonConfig.from_dict(config)


def test_inference_rejects_a_fresh_training_proposal(sampler_settings):
    config = settings(sampler_settings)
    config["aggregator"] = {"mode": "infer"}
    with pytest.raises(ValueError, match="training_steps"):
        ComparisonConfig.from_dict(config)
    config["sampler"]["orchestrator"]["training_steps"] = 0
    assert len(ComparisonConfig.from_dict(config).named_runs) == 2


def test_unknown_settings_are_rejected_for_every_run(sampler_settings):
    config = settings(sampler_settings)
    config["samplers"][1]["orchestrator"] = {"training_batch_size": 0}
    with pytest.raises(ValueError, match="training_batch_size"):
        ComparisonConfig.from_dict(config)
