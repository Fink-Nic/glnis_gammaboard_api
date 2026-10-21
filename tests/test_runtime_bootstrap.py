"""Launch behavior shared by the editable venv and packaged runtime."""

import json
import os
import sys
from unittest.mock import Mock

import pytest

import glnis_runtime_bootstrap as bootstrap


def test_native_paths_come_from_the_selected_environment(tmp_path, monkeypatch):
    configured = tmp_path / "configured"
    explicit = tmp_path / "explicit"
    configured.mkdir()
    explicit.mkdir()
    (tmp_path / "glnis-runtime.json").write_text(
        json.dumps({"library_dirs": [str(configured)], "tool_dirs": []})
    )
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(bootstrap, "_DRIVER_DIRS", ())
    monkeypatch.setenv("GLNIS_LIBRARY_PATH", str(explicit))
    monkeypatch.delenv("GLNIS_TOOL_PATH", raising=False)

    assert bootstrap._native_library_dirs() == [str(explicit), str(configured)]
    assert bootstrap._native_tool_dirs() == []


@pytest.mark.parametrize(
    "arguments",
    [
        ["-u", "-m", "run_sampler", "argument"],
        ["-c", "print('hello')", "argument"],
        ["/venv/bin/glnis-gammaboard-sampler", "argument"],
    ],
)
def test_deferred_reexec_preserves_python_invocation(tmp_path, monkeypatch, arguments):
    monkeypatch.setattr(bootstrap, "_REEXEC_NEEDED", False)
    monkeypatch.setattr(bootstrap, "_DRIVER_DIRS", ())
    monkeypatch.setattr(bootstrap, "_native_library_dirs", lambda: [str(tmp_path)])
    monkeypatch.setattr(bootstrap, "_native_tool_dirs", lambda: [])
    monkeypatch.setattr(bootstrap, "_first_existing_tool", lambda name: None)
    monkeypatch.setattr(sys, "orig_argv", ["python", *arguments])
    monkeypatch.setattr(sys, "executable", "/venv/bin/python")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.delenv("GLNIS_RUNTIME_BOOTSTRAPPED", raising=False)
    execute = Mock()
    monkeypatch.setattr(os, "execve", execute)

    bootstrap._bootstrap_native_environment(allow_reexec=False)
    execute.assert_not_called()
    bootstrap._bootstrap_native_environment(allow_reexec=True)
    executable, argv, environment = execute.call_args.args
    assert executable == "/venv/bin/python"
    assert argv == [executable, *arguments]
    assert environment["LD_LIBRARY_PATH"] == str(tmp_path)
    assert environment["GLNIS_RUNTIME_BOOTSTRAPPED"] == "1"

    execute.reset_mock()
    monkeypatch.setenv("GLNIS_RUNTIME_BOOTSTRAPPED", "1")
    bootstrap._bootstrap_native_environment(allow_reexec=True)
    execute.assert_not_called()


def test_real_ldconfig_is_preserved(monkeypatch):
    monkeypatch.setattr(bootstrap, "_find_in_library_dirs", lambda name: None)
    monkeypatch.setattr(os, "access", lambda *args: True)
    original = bootstrap._subprocess.check_output
    bootstrap._patch_triton_ldconfig_probe()
    assert bootstrap._subprocess.check_output is original


def test_system_driver_search_does_not_override_runtime_libraries(tmp_path, monkeypatch):
    native = str(tmp_path / "native")
    inherited = str(tmp_path / "inherited")
    driver = str(tmp_path / "driver")
    system_dirs = ["/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"]
    monkeypatch.setattr(bootstrap, "_configured_dirs",
                        lambda key, env: [native] if key == "library_dirs" else [])
    monkeypatch.setattr(bootstrap, "_DRIVER_DIRS", (driver, *system_dirs))
    monkeypatch.setattr(bootstrap._os.path, "isdir", lambda path: True)
    monkeypatch.setattr(bootstrap, "_first_existing_tool", lambda name: None)
    monkeypatch.setattr(bootstrap, "_REEXEC_NEEDED", False)
    monkeypatch.setenv("LD_LIBRARY_PATH", inherited)

    bootstrap._bootstrap_native_environment(allow_reexec=False)

    assert bootstrap._os.environ["LD_LIBRARY_PATH"].split(":") == [native, inherited, driver]
    assert set(system_dirs).issubset(bootstrap._library_dirs())


def test_missing_ldconfig_shim_only_handles_its_exact_probe(monkeypatch):
    monkeypatch.setattr(bootstrap, "_find_in_library_dirs", lambda name: None)
    monkeypatch.setattr(bootstrap, "_ldconfig_p_text", lambda: "library listing\n")
    monkeypatch.setattr(os, "access", lambda *args: False)
    original = Mock(return_value=b"other command")
    original._glnis_patched = False
    monkeypatch.setattr(bootstrap._subprocess, "check_output", original)
    bootstrap._patch_triton_ldconfig_probe()
    patched = bootstrap._subprocess.check_output
    bootstrap._patch_triton_ldconfig_probe()
    assert bootstrap._subprocess.check_output is patched
    assert patched(["/sbin/ldconfig", "-p"]) == b"library listing\n"
    assert patched(["/sbin/ldconfig", "-p"], text=True) == "library listing\n"
    assert patched(["another-command"]) == b"other command"
    original.assert_called_once_with(["another-command"])
