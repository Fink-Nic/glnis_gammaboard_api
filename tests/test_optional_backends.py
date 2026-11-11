"""Core imports and mappings must work without any GammaLoop dependencies."""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from glnis import _optional


def test_core_parser_and_explicit_mapping_do_not_import_optional_backends():
    # A fresh interpreter also catches accidental eager imports when other tests
    # have already cached parser or installed optional dependencies in sys.modules.
    result = subprocess.run(
        [sys.executable, "-c", r'''
import importlib.abc
import json
import sys
from dataclasses import asdict

class DenyOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"symbolica", "pydot", "momtrop", "glnis_gammaloop_native"} or fullname == "glnis._rust":
            raise AssertionError(f"Core attempted optional import: {fullname}")

# Torch itself probes pydot in recent versions. Import it before the
# assertion finder so this test isolates imports owned by this project.
import torch
sys.modules.pop("pydot", None)
sys.meta_path.insert(0, DenyOptionalImports())
import numpy as np
from glnis.core.parser import MetaDataParser, ModelParser, DotParser

properties = dict(
    edge_masses=[0.0], edge_momentum_shifts=[[0.0, 0.0, 0.0]],
    graph_signature=[[1]], lmb_edges=[[0]], generation_lmb_edges=[0],
)
metadata = dict(kind="gammaloop", state_folder="unused-state", process_id=0, integrand_name="unused")
parser = MetaDataParser(metadata=metadata, graph_properties=properties)
gp = parser.get_graph_properties()
assert gp.n_loops == 1
assert gp.n_channels == 1
# Resolved checkpoint properties remain usable even when metadata names GammaLoop.
restored = MetaDataParser(metadata=metadata, graph_properties=asdict(gp))
mapping = restored.get_layered_mapping_instance(dict(layer_cfgs=[dict(kind="spherical", conformal_scale=2.0)]))
continuous = np.full((3, 3), 0.25)
discrete = np.empty((3, 0), dtype=np.uint64)
mapped, jacobian, integration_weight = mapping.forward(discrete, continuous)
assert mapped.shape == (3, 3)
assert np.isfinite(mapped).all() and np.isfinite(jacobian).all()
model = ModelParser(json.dumps(dict(particles=[dict(name="x", mass="ZERO")], parameters=[])))
assert model.get_particle_mass_from_name("x") == [0.0, 0.0]
'''],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module", ["symbolica", "pydot", "glnis_gammaloop_native._rust"])
def test_missing_dependency_has_actionable_install_guidance(monkeypatch, module):
    missing = ModuleNotFoundError("not installed", name=module.split(".")[0])

    def missing_import(name):
        assert name == module
        raise missing

    monkeypatch.setattr(_optional, "import_module", missing_import)
    with pytest.raises(_optional.MissingGammaLoopDependency, match=r"glnis_gammaboard_api\[gammaloop\]") as caught:
        _optional.require_gammaloop_dependency(module, "Test operation")
    assert caught.value.name == missing.name
    assert caught.value.__cause__ is missing
    assert "Test operation" in str(caught.value)
    assert "./native/gammaloop" in str(caught.value)


@pytest.mark.parametrize("error", [
    ModuleNotFoundError("transitive dependency missing", name="other_dependency"),
    ModuleNotFoundError("native extension missing", name="glnis_gammaloop_native._rust"),
    ImportError("shared library load failed"),
    RuntimeError("license failed"),
])
def test_broken_installed_dependency_preserves_original_exception(monkeypatch, error):
    def broken_import(name):
        raise error

    monkeypatch.setattr(_optional, "import_module", broken_import)
    with pytest.raises(type(error)) as caught:
        _optional.gammaloop_native()
    assert caught.value is error


def test_dot_loading_only_requests_pydot_and_symbolic_extraction_requests_symbolica(monkeypatch):
    from glnis.core import parser

    requested = []
    graph = object()

    def optional_import(module):
        requested.append(module)
        if module == "pydot":
            return SimpleNamespace(graph_from_dot_data=lambda data: [graph])
        raise ModuleNotFoundError("not installed", name=module)

    monkeypatch.setattr(_optional, "import_module", optional_import)
    dot = parser.DotParser("digraph {}", "{}", dot_from_string=True)
    assert dot.get_dot_graph(0) is graph
    assert requested == ["pydot"]
    with pytest.raises(_optional.MissingGammaLoopDependency, match="symbolica"):
        dot.get_graph_properties(0, [])
    assert requested == ["pydot", "symbolica"]


def test_reading_state_requires_companion_not_legacy_in_tree_extension(monkeypatch):
    from glnis.core.parser import MetaDataParser

    # Keep this test independent of unrelated GammaBoard protocol changes.
    sampler = ModuleType("gammaboard_sampler")
    sampler.parse_gammaloop_metadata = lambda data: SimpleNamespace(**data)
    monkeypatch.setitem(sys.modules, "gammaboard_sampler", sampler)
    legacy = ModuleType("glnis._rust")
    legacy.parse_state_simplified = lambda *args: pytest.fail("Loaded legacy extension")
    monkeypatch.setitem(sys.modules, "glnis._rust", legacy)
    requested = []

    def missing_import(module):
        requested.append(module)
        raise ModuleNotFoundError("not installed", name="glnis_gammaloop_native")

    monkeypatch.setattr(_optional, "import_module", missing_import)
    parser = MetaDataParser(metadata=dict(
        kind="gammaloop", state_folder="unused", process_id=0, integrand_name="unused",
    ))
    assert requested == []
    with pytest.raises(_optional.MissingGammaLoopDependency, match="Reading GammaLoop states"):
        parser.get_graph_properties()
    assert requested == ["glnis_gammaloop_native._rust"]
