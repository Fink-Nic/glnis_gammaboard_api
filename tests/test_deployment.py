"""Cluster-free checks for frozen submissions and atomic image publication."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("glnis_ubelix", ROOT / "ubelix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    module.ROOT = checkout
    for name in module.INPUTS:
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in ("src", "native/gammaloop/src", "native/gammaloop/python"):
            target.mkdir()
            (target / "code.py").write_text("original\n")
        else:
            target.write_text((ROOT / name).read_text())
    calls = []
    with monkeypatch.context() as patch:
        patch.setattr(module.shutil, "which", lambda name: "/fake/sbatch")
        patch.setattr(module.subprocess, "run", lambda command, **kwargs:
                      calls.append(command) or SimpleNamespace(stdout="123;cluster\n"))
        patch.setattr(sys, "argv", ["ubelix.py", "--with-gammaloop"])
        module.main()
    record = next((checkout / "artifacts/ubelix").glob("build-*/build.json"))
    return module, record, calls


def test_submission_freezes_source_and_never_builds_on_login_node(deployment):
    module, record, calls = deployment
    assert len(calls) == 1 and calls[0][0] == "sbatch"
    assert not any("--gres" in arg or "--gpus" in arg for arg in calls[0])
    assert "--wrap" in calls[0] and "--run-build" in calls[0][-1]
    assert (record.parent / "build.py").read_text() == (ROOT / "ubelix.py").read_text()
    (module.ROOT / "src/code.py").write_text("edited after submission\n")
    with tarfile.open(record.parent / "source.tar") as archive:
        assert archive.extractfile("src/code.py").read() == b"original\n"
        assert not {"flake.nix", "flake.lock"}.intersection(archive.getnames())
    saved = json.loads(record.read_text())
    assert saved["state"] == "submitted" and saved["with_gammaloop"] is True
    assert saved["source_sha256"] == hashlib.sha256((record.parent / "source.tar").read_bytes()).hexdigest()


def test_dry_run_has_no_filesystem_or_submission_side_effects(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("glnis_ubelix", ROOT / "ubelix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    monkeypatch.setattr(sys, "argv", ["ubelix.py", "--dry-run"])
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected submission"))
    module.main()
    assert "sbatch" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", [None, "build", "test"])
def test_build_uses_scratch_and_publishes_only_after_success(deployment, tmp_path, monkeypatch, failure):
    module, record, _ = deployment
    image = module.ROOT / "glnis.sif"
    image.write_bytes(b"previous image")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    calls = tmp_path / "calls.jsonl"
    apptainer = tools / "apptainer"
    apptainer.write_text(f"#!{sys.executable}\n" + '''
import json, os
from pathlib import Path
import sys
with open(os.environ["CHECK_CALLS"], "a") as stream:
    stream.write(json.dumps({"args": sys.argv[1:], "cwd": os.getcwd(),
                             "tmp": os.environ["APPTAINER_TMPDIR"],
                             "override": os.environ.get("APPTAINERENV_LD_LIBRARY_PATH")}) + "\\n")
if os.environ.get("CHECK_FAIL") == sys.argv[1]:
    raise SystemExit(2)
if sys.argv[1] == "build":
    assert Path("src/code.py").read_text() == "original\\n"
    Path(sys.argv[-2]).write_bytes(b"validated image")
''')
    apptainer.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setenv("CHECK_CALLS", str(calls))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("APPTAINERENV_LD_LIBRARY_PATH", "/incompatible/host/libs")
    if failure:
        monkeypatch.setenv("CHECK_FAIL", failure)
        with pytest.raises(subprocess.CalledProcessError):
            module.run_build(record)
        assert image.read_bytes() == b"previous image"
    else:
        module.run_build(record)
        assert image.read_bytes() == b"validated image"
    saved = json.loads(record.read_text())
    assert saved["state"] == ("failed" if failure else "completed")
    assert list(scratch.iterdir()) == []
    assert not list(image.parent.glob("*.partial-*"))
    invocations = [json.loads(line) for line in calls.read_text().splitlines()]
    build = invocations[0]
    assert Path(build["cwd"]).is_relative_to(scratch)
    assert Path(build["tmp"]).is_relative_to(scratch)
    assert "WITH_GAMMALOOP=1" in build["args"]
    assert "--nv" not in build["args"]
    assert all(call["override"] is None for call in invocations)
    if not failure:
        assert invocations[1]["args"][:2] == ["test", "--cleanenv"]
        assert Path(invocations[1]["cwd"]).is_relative_to(scratch)
        assert saved["image_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()


def test_build_phase_refuses_to_run_without_slurm(deployment, monkeypatch):
    module, record, _ = deployment
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm allocation"):
        module.run_build(record)


def test_importing_entrypoints_does_not_patch_the_process_environment():
    completed = subprocess.run([sys.executable, "-c", '''
import ctypes.util
import os
import subprocess
keys = ("PATH", "LD_LIBRARY_PATH", "CC", "CXX", "OMP_NUM_THREADS", "TRITON_LIBCUDA_PATH")
environment = {key: os.environ.get(key) for key in keys}
check_output = subprocess.check_output
find_library = ctypes.util.find_library
soname = getattr(ctypes.util, "_findSoname_ldconfig", None)
import run_sampler
import glnis.scripts.run
import glnis.scripts.sampler_comparison
import glnis.scripts.slice_plots
assert {key: os.environ.get(key) for key in keys} == environment
assert subprocess.check_output is check_output
assert ctypes.util.find_library is find_library
assert getattr(ctypes.util, "_findSoname_ldconfig", None) is soname
'''], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
