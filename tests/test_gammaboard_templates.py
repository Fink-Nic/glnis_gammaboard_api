"""Bundled GammaBoard cards route typed replacements into the current API."""
from copy import deepcopy
from pathlib import Path
import re

# GammaBoard accepts TOML 1.1, including multiline inline tables. Python 3.13's
# standard-library parser only implements TOML 1.0.
import tomli

import pytest

from glnis.core.config import RunConfig


TEMPLATES = Path(__file__).resolve().parents[1] / "examples/templates"


def expand(value, replacements):
    """The subset of GammaBoard replacements used by these cards, for API checks."""
    if isinstance(value, dict):
        return {key: expand(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, replacements) for item in value]
    if not isinstance(value, str):
        return value
    pattern = r"\$\((\w+):([^)]*)\)"
    matches = list(re.finditer(pattern, value))
    for match in reversed(matches):
        name, default = match.groups()
        selected = deepcopy(replacements[name]) if name in replacements else tomli.loads("value = " + default)["value"]
        if match.span() == (0, len(value)):
            return selected
        value = value[:match.start()] + str(selected) + value[match.end():]
    return value


@pytest.mark.parametrize("path", sorted((TEMPLATES / "runs").glob("*.toml")), ids=lambda path: path.stem)
def test_run_cards_use_current_sampler_configuration(path):
    raw = tomli.loads(path.read_text())
    card = expand(raw, raw.get("replacements", {}))
    if card.get("kind") == "integration_campaign":
        assert len(card["children"]) == 11
        assert len({child["replacements"]["graph_group"] for child in card["children"]}) == 11
        for child in card["children"]:
            assert (path.parent / child["run"]["file"]).is_file()
        return
    training, inference = card["task_queue"]
    process = training["sampler"]["config"]
    assert process["kind"] == "process_sampler"
    assert "glnis_gammaboard_api/.venv/bin/glnis-gammaboard-sampler" in process["command"][0]
    assert training["publish_result"] is False
    assert inference["sampler"]["from_name"] == training["name"]
    config = RunConfig.from_dict({"sampler": process["args"], "integrand": {"kind": "sine", "continuous_dims": 6}})
    settings = config.sampler["orchestrator"]
    assert settings["gpu_id"] == "auto"
    assert training["stop_condition"]["max_samples"] == settings["training_steps"] * settings["training_batch_size"]
    assert "module" in config.sampler and "module_kwargs" not in settings


@pytest.mark.parametrize("execution", [
    {"kind": "eager"},
    {"kind": "torch_compile", "sample": {"dynamic": True}, "prob": {"dynamic": False}},
    {"kind": "torch_compile", "sample": False},
])
def test_baseline_accepts_complete_policy_and_scheduler_replacements(execution):
    raw = tomli.loads((TEMPLATES / "runs/gammaloop_train_sample.toml").read_text())
    replacements = {**raw["replacements"], "execution": execution, "scheduler": False, "gpu_id": 2,
                    "flow_layers": 3, "flow_units": 256}
    card = expand(raw, replacements)
    config = RunConfig.from_dict({"sampler": card["task_queue"][0]["sampler"]["config"]["args"],
                                  "integrand": {"kind": "sine", "continuous_dims": 6}})
    settings = config.sampler["orchestrator"]
    assert settings["execution"] == execution
    assert settings["scheduler"] is None
    assert settings["gpu_id"] == 2
    assert config.sampler["module"]["flow"]["subnet"] == {"kind": "mlp", "layers": 3, "units": 256}


@pytest.mark.parametrize("path", sorted((TEMPLATES / "tasks").glob("*.toml")), ids=lambda path: path.stem)
def test_append_cards_have_named_current_task_definitions(path):
    card = tomli.loads(path.read_text())
    names = [task["name"] for task in card["task_queue"]]
    assert len(names) == len(set(names))
    assert all(task["kind"] in {"sample", "pdf_adaptation_image", "pdf_adaptation_plot_line"}
               for task in card["task_queue"])
