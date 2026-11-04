"""Load optional GammaLoop dependencies only for operations that need them."""

from importlib import import_module
from types import ModuleType


class MissingGammaLoopDependency(ModuleNotFoundError):
    """An optional dependency is absent, rather than installed but broken."""


def require_gammaloop_dependency(module: str, operation: str) -> ModuleType:
    """Import a backend, preserving failures inside an installed dependency.

    In particular, a missing native submodule in an installed companion, binary
    loading errors, Symbolica license errors, and absent transitive dependencies
    must keep their original diagnostics.
    """
    try:
        return import_module(module)
    except ModuleNotFoundError as error:
        if error.name != module.partition(".")[0]:
            raise
        raise MissingGammaLoopDependency(
            f"{operation} requires the optional GammaLoop feature "
            f"(missing {error.name!r}). Install 'glnis_gammaboard_api[gammaloop]'. "
            "For a source checkout, install the companion too: "
            "pip install -e ./native/gammaloop -e '.[gammaloop]'.",
            name=error.name,
        ) from error


def gammaloop_native() -> ModuleType:
    """Load the companion extension, never an old in-tree ``glnis._rust`` binary."""
    return require_gammaloop_dependency(
        "glnis_gammaloop_native._rust", "Reading GammaLoop states"
    )
