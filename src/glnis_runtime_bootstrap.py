"""Bootstrap native-library and Triton compatibility for direct venv launches.

GammaBoard often launches the sampler entry point outside `nix develop`.  In
that environment Python wheels can miss Nix native library paths, and Triton
hard-codes `/sbin/ldconfig -p`.  Import this module before NumPy/Torch/MadNIS.
"""

from __future__ import annotations

import ctypes.util as _ctypes_util
import glob as _glob
import os as _os
import subprocess as _subprocess
import sys as _sys
from collections.abc import Sequence
from os import PathLike
from typing import Any

_DRIVER_DIRS = (
    "/run/opengl-driver/lib",
    "/usr/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/lib/x86_64-linux-gnu",
    "/usr/local/cuda/lib64",
    "/usr/lib/wsl/lib",
)

_NATIVE_LIBRARY_PATTERNS = (
    "/nix/store/*gcc*-lib/lib/libstdc++.so.6",
    "/nix/store/*zlib-*/lib/libz.so*",
    "/nix/store/*gmp*/lib/libgmp.so*",
    "/nix/store/*mpfr-*/lib/libmpfr.so*",
    "/nix/store/*libmpc-*/lib/libmpc.so*",
)

_NATIVE_TOOL_PATTERNS = (
    "/nix/store/*gcc-wrapper-*/bin/gcc",
    "/nix/store/*clang-wrapper-*/bin/cc",
)


def _native_library_dirs() -> list[str]:
    dirs = [directory for directory in _DRIVER_DIRS if _os.path.isdir(directory)]
    for pattern in _NATIVE_LIBRARY_PATTERNS:
        dirs.extend(_os.path.dirname(path) for path in _glob.glob(pattern))
    return dirs


def _native_tool_dirs() -> list[str]:
    dirs: list[str] = []
    for pattern in _NATIVE_TOOL_PATTERNS:
        dirs.extend(_os.path.dirname(path) for path in _glob.glob(pattern))
    return dirs


def _first_existing_tool(name: str) -> str | None:
    for directory in _os.environ.get("PATH", "").split(":"):
        path = _os.path.join(directory, name)
        if _os.access(path, _os.X_OK):
            return path
    return None


def _bootstrap_native_environment() -> None:
    changed = False

    existing_libs = [path for path in _os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    merged_libs: list[str] = []
    for path in [*_native_library_dirs(), *existing_libs]:
        if path and _os.path.isdir(path) and path not in merged_libs:
            merged_libs.append(path)
    if merged_libs != existing_libs:
        _os.environ["LD_LIBRARY_PATH"] = ":".join(merged_libs)
        changed = True

    existing_tools = [path for path in _os.environ.get("PATH", "").split(":") if path]
    merged_tools: list[str] = []
    for path in [*_native_tool_dirs(), *existing_tools]:
        if path and _os.path.isdir(path) and path not in merged_tools:
            merged_tools.append(path)
    if merged_tools != existing_tools:
        _os.environ["PATH"] = ":".join(merged_tools)
        changed = True

    cc = _first_existing_tool("gcc") or _first_existing_tool("cc")
    cxx = _first_existing_tool("g++") or _first_existing_tool("c++")
    if cc and "CC" not in _os.environ:
        _os.environ["CC"] = cc
        changed = True
    if cxx and "CXX" not in _os.environ:
        _os.environ["CXX"] = cxx
        changed = True

    if not changed or _os.environ.get("GLNIS_RUNTIME_BOOTSTRAPPED") == "1":
        return
    if not _sys.argv or _sys.argv[0] in {"-c", ""}:
        return

    env = _os.environ.copy()
    env["GLNIS_RUNTIME_BOOTSTRAPPED"] = "1"
    _os.execve(_sys.executable, [_sys.executable, *_sys.argv], env)


def _library_dirs():
    seen: set[str] = set()
    raw_dirs = [*_os.environ.get("LD_LIBRARY_PATH", "").split(":"), *_DRIVER_DIRS]
    for directory in raw_dirs:
        if directory and directory not in seen and _os.path.isdir(directory):
            seen.add(directory)
            yield directory


def _candidate_stems(name: object) -> list[str]:
    raw = str(name)
    stems = [raw]
    if not raw.startswith("lib"):
        stems.append("lib" + raw)
    return [stem if ".so" in stem else stem + ".so" for stem in stems]


def _find_in_library_dirs(name: object) -> str | None:
    for directory in _library_dirs():
        for stem in _candidate_stems(name):
            for match in sorted(_glob.glob(_os.path.join(directory, stem + "*"))):
                if _os.path.isfile(match):
                    return match
    return None


def _ldconfig_p_text() -> str:
    lines: list[str] = []
    for directory in _library_dirs():
        for match in sorted(_glob.glob(_os.path.join(directory, "lib*.so*"))):
            if _os.path.isfile(match):
                lines.append(f"\t{_os.path.basename(match)} (libc6,x86-64) => {match}")
    return "\n".join(lines) + ("\n" if lines else "")


def _is_sbin_ldconfig_p(args: object) -> bool:
    if isinstance(args, (str, bytes, PathLike)):
        return False
    if not isinstance(args, Sequence):
        return False
    return [str(arg) for arg in args] == ["/sbin/ldconfig", "-p"]


def _patch_ctypes_find_library() -> None:
    if hasattr(_ctypes_util, "_findSoname_ldconfig"):
        _ctypes_util._findSoname_ldconfig = _find_in_library_dirs  # type: ignore[attr-defined]


def _patch_triton_ldconfig_probe() -> None:
    _libcuda = _find_in_library_dirs("cuda")
    if _libcuda and "TRITON_LIBCUDA_PATH" not in _os.environ:
        _os.environ["TRITON_LIBCUDA_PATH"] = _os.path.dirname(_libcuda)

    original_check_output = _subprocess.check_output
    if getattr(original_check_output, "_glnis_patched", False):
        return

    def _check_output(args: Any, *pargs: Any, **kwargs: Any):
        if _is_sbin_ldconfig_p(args):
            text = _ldconfig_p_text()
            if kwargs.get("text") or kwargs.get("universal_newlines") or kwargs.get("encoding"):
                return text
            return text.encode()
        return original_check_output(args, *pargs, **kwargs)

    _check_output._glnis_patched = True  # type: ignore[attr-defined]
    _subprocess.check_output = _check_output


def bootstrap() -> None:
    _bootstrap_native_environment()
    _patch_ctypes_find_library()
    _patch_triton_ldconfig_probe()


bootstrap()
