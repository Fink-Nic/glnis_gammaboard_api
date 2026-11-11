"""Compatibility interface for the optional GammaLoop native companion.

Importing this module is safe in a core-only installation. A native dependency
is loaded only when one of its operations is requested.
"""

from os import PathLike

from glnis._optional import gammaloop_native


def health_check() -> str:
    return gammaloop_native().health_check()


def classify_state_folder(state_folder: str | PathLike[str]) -> str:
    return gammaloop_native().classify_state_folder(state_folder)


def parse_state_summary_json(state_folder: str | PathLike[str]) -> str:
    return gammaloop_native().parse_state_summary_json(state_folder)


def parse_state_simplified(
    state_folder: str | PathLike[str], process_id: int, integrand_name: str
) -> str:
    return gammaloop_native().parse_state_simplified(
        state_folder, process_id, integrand_name
    )
