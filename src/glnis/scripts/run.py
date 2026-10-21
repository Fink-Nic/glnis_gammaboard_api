"""Run a local sampler/integrand pair configured by TOML."""

from __future__ import annotations

import glnis_runtime_bootstrap

glnis_runtime_bootstrap.bootstrap()

import argparse
import json
import math
from pathlib import Path

import numpy as np

from glnis.core.sampler_aggregator import SamplerAggregator


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(*, default_config: Path | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?" if default_config else None,
                        default=default_config, help="Run configuration TOML")
    parser.add_argument("--history", type=Path, help="Write training and inference observables as JSON")
    args = parser.parse_args()
    aggregator = SamplerAggregator.from_toml(args.config)
    result = aggregator.run()
    print(f"Completed {len(aggregator.training_history)} training steps and "
          f"{result.result.n_points} inference samples "
          f"({len(aggregator.inference_history)} accumulator updates).", flush=True)
    if result.result.n_points:
        print(result.str_report())
    elif aggregator.training_history:
        print("Final training observables:")
        print(json.dumps(_json_value(aggregator.training_history[-1]), indent=2, allow_nan=False))
    if args.history is not None:
        destination = args.history.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        history = dict(training=aggregator.training_history, inference=aggregator.inference_history)
        destination.write_text(json.dumps(_json_value(history), indent=2, allow_nan=False) + "\n")
        print(f"Observables written to {destination}")


if __name__ == "__main__":
    main()
