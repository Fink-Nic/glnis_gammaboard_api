# type: ignore
"""Print the Rust GammaLoop state parser output for a user-supplied state.

Examples:
    GLNIS_STATE_FOLDER=/path/to/state \
    GLNIS_PROCESS_ID=0 \
    GLNIS_INTEGRAND_NAME=my_integrand \
    pytest -s tests/test_rust_state_parser.py

    python tests/test_rust_state_parser.py \
        --state-folder /path/to/state --process-id 0 --integrand-name my_integrand
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from pprint import pformat
from glnis.core.parser import MetaDataParser
from dataclasses import asdict

import pytest


ENV_STATE_FOLDER = "GLNIS_STATE_FOLDER"
ENV_PROCESS_ID = "GLNIS_PROCESS_ID"
ENV_INTEGRAND_NAME = "GLNIS_INTEGRAND_NAME"


def _resolve_inputs(
    state_folder: str | None,
    process_id: str | int | None,
    integrand_name: str | None,
) -> tuple[Path, int, str]:
    state_folder = state_folder or os.environ.get(ENV_STATE_FOLDER)
    process_id = process_id if process_id is not None else os.environ.get(ENV_PROCESS_ID)
    integrand_name = integrand_name or os.environ.get(ENV_INTEGRAND_NAME)

    missing = []
    if state_folder is None:
        missing.append(f"--state-folder or {ENV_STATE_FOLDER}")
    if process_id is None:
        missing.append(f"--process-id or {ENV_PROCESS_ID}")
    if integrand_name is None:
        missing.append(f"--integrand-name or {ENV_INTEGRAND_NAME}")
    if missing:
        raise ValueError("Missing required input(s): " + ", ".join(missing))

    return Path(state_folder), int(process_id), integrand_name


def _print_summary(json_or_dict: str|dict) -> None:
    if isinstance(json_or_dict, dict):
        data = json_or_dict
    else:
        data = json.loads(json_or_dict)
    for key, value in data.items():
        if key in {"model", "dot_file"}:
            print(f"{key}: <{len(value)} characters>")
        else:
            print(f"{key}: {pformat(value, sort_dicts=False)}")


def _parse_and_print(state_folder: Path, process_id: int, integrand_name: str) -> None:
    from glnis import _rust

    _print_summary(_rust.parse_state_simplified(state_folder, process_id, integrand_name))


def test_rust_state_parser():
    try:
        state_folder, process_id, integrand_name = _resolve_inputs(None, None, None)
    except ValueError as err:
        pytest.skip(str(err))

    _parse_and_print(state_folder, process_id, integrand_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print parse_state_simplified output for a GammaLoop state."
    )
    parser.add_argument("--state-folder", default=None)
    parser.add_argument("--process-id", default=None)
    parser.add_argument("--integrand-name", default=None)
    args = parser.parse_args()

    state_folder, process_id, integrand_name = _resolve_inputs(
        args.state_folder,
        args.process_id,
        args.integrand_name,
    )
    hline = 50*"="
    print(hline)
    print("Rust parsed data")
    print(hline)
    _parse_and_print(state_folder, process_id, integrand_name)
    Parser = MetaDataParser(
        metadata=dict(kind="gammaloop", state_folder=state_folder, process_id=process_id, integrand_name=integrand_name)
    )
    graph_properties = Parser.get_graph_properties()
    print(hline)
    print("glnis parsed graph properties")
    print(hline)
    if isinstance(graph_properties, list):
        graph_properties = graph_properties[0]

    _print_summary(asdict(graph_properties))

if __name__ == "__main__":
    main()
