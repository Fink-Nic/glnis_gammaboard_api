"""The core distribution must build independently of its native companion."""

from __future__ import annotations

from email.parser import Parser
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib", ".a", ".rlib")


def _source_snapshot(destination: Path) -> None:
    """Copy build inputs without caches or large local compiled artifacts."""
    destination.mkdir()
    for name in ("pyproject.toml", "MANIFEST.in", "README.md"):
        shutil.copy2(ROOT / name, destination / name)
    for source in (ROOT / "src").rglob("*.py"):
        if "__pycache__" in source.parts or any(
            part.endswith(".egg-info") for part in source.parts
        ):
            continue
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    # Simulate artifacts left by an older editable native build. They must not
    # leak into either archive, even when they are inside a discovered package.
    for name in ("_rust.cpython-313-x86_64-linux-gnu.so", "unwanted.dll"):
        (destination / "src/glnis" / name).write_bytes(b"native artifact sentinel")
    (destination / "src/sitecustomize.py").write_text(
        "raise RuntimeError('global sitecustomize must not be installed')\n"
    )
    companion = destination / "native/gammaloop/src"
    companion.mkdir(parents=True)
    (companion / "lib.rs").write_text("// companion source must not ship in core\n")


def _build(source: Path, output: Path, methods: tuple[str, ...], env: dict) -> None:
    output.mkdir()
    script = "import setuptools.build_meta as backend; " + "; ".join(
        f"backend.{method}({str(output)!r})" for method in methods
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=source,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_no_native_files(names: list[str]) -> None:
    assert not [name for name in names if name.endswith(NATIVE_SUFFIXES)]
    assert not [name for name in names if "sitecustomize" in name]
    assert not [name for name in names if "/native/" in f"/{name}"]
    assert not [name for name in names if "/src/rust/" in f"/{name}"]


def _inspect_wheel(wheel: Path) -> None:
    assert wheel.name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        _assert_no_native_files(names)
        for module in (
            "madnis_sampler.py",
            "run_sampler.py",
            "glnis_runtime_bootstrap.py",
            "glnis/core/parser.py",
            "glnis/core/integrand.py",
            "glnis/core/mappings.py",
        ):
            assert module in names
        # This subpackage has no __init__.py; ordinary package discovery would
        # silently drop the neural-network implementations from installed wheels.
        assert any(name.startswith("glnis/core/nn/") and name.endswith(".py") for name in names)

        metadata = Parser().parsestr(archive.read(next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )).decode())
        dependencies = metadata.get_all("Requires-Dist", [])
        by_name = {}
        for requirement in dependencies:
            name = re.match(r"[A-Za-z0-9_.-]+", requirement).group()
            by_name[name.lower().replace("_", "-")] = requirement
        for name in ("symbolica", "pydot", "glnis-gammaloop-native"):
            assert 'extra == "gammaloop"' in by_name[name]
        assert 'extra == "test"' in by_name["pytest"]
        assert 'extra == "test"' in by_name["setuptools"]
        assert "momtrop" not in by_name
        assert "maturin" not in by_name
        assert metadata["Name"].replace("_", "-") == "glnis-gammaboard-api"

        entrypoints = archive.read(next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )).decode()
        configured = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
        for command, target in configured.items():
            assert f"{command} = {target}" in entrypoints


def test_core_wheel_and_sdist_build_without_compiler_tools(tmp_path):
    if importlib.util.find_spec("setuptools") is None:
        pytest.skip("setuptools is required to exercise the core packaging backend")

    source = tmp_path / "source"
    _source_snapshot(source)
    empty_path = tmp_path / "no-executables"
    empty_path.mkdir()
    env = dict(os.environ, PATH=str(empty_path))
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        env.pop(name, None)

    dist = tmp_path / "dist"
    _build(source, dist, ("build_sdist", "build_wheel"), env)
    _inspect_wheel(next(dist.glob("*.whl")))

    with tarfile.open(next(dist.glob("*.tar.gz"))) as archive:
        _assert_no_native_files(archive.getnames())
        archive.extractall(tmp_path / "unpacked", filter="data")
    unpacked = next((tmp_path / "unpacked").iterdir())
    assert not (unpacked / "Cargo.toml").exists()
    assert not (unpacked / "src/glnis/pyproject.toml").exists()
    rebuilt = tmp_path / "rebuilt"
    _build(unpacked, rebuilt, ("build_wheel",), env)
    _inspect_wheel(next(rebuilt.glob("*.whl")))
