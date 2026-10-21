# type: ignore
"""Compare sequential MadNIS configurations from TOML using recorded history."""

from __future__ import annotations

import glnis_runtime_bootstrap

glnis_runtime_bootstrap.bootstrap()

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gc
from pathlib import Path
import pickle
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from glnis.scripts.comparison_config import ComparisonConfig
from glnis.scripts.comparison_plots import plot_sampler_comp, write_summary
from glnis.utils.types import Result

if TYPE_CHECKING:
    from glnis.core.sampler_aggregator import SamplerAggregator


@dataclass
class TrainingProgress:
    """Per-step measurements from the samples used for each optimizer update.

    The snapshot fields retain their historical names, but no extra evaluation
    draws are taken. Sample-count axes are cumulative across training steps.
    Discrete-probability fields remain empty because history has no such probes.
    """

    losses: list[float | None] | None = field(default_factory=list)
    rsds: list[float] = field(default_factory=list)
    abs_rsds: list[float] = field(default_factory=list)
    tvars: list[float] = field(default_factory=list)
    abs_tvars: list[float] = field(default_factory=list)
    steps_losses: list[int] = field(default_factory=list)
    nspl_losses: list[int] = field(default_factory=list)
    steps_snapshot: list[int] = field(default_factory=list)
    nspl_snapshot: list[int] = field(default_factory=list)
    means: list[float] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    steps_discrete: list[int] = field(default_factory=list)
    nspl_discrete: list[int] = field(default_factory=list)
    all_channels: NDArray = field(default_factory=lambda: np.array([]))
    discrete_probs: list[NDArray] = field(default_factory=list)

    @classmethod
    def from_history(cls, history: list[dict[str, Any]], training_phase: str) -> TrainingProgress:
        progress = cls()
        total_samples = 0
        for record in history:
            total_samples += record["n_samples"]
            step = record["step"]
            selected = record["training_result"]
            observables = record["observables"]
            progress.losses.append(record["loss"])
            progress.steps_losses.append(step)
            progress.nspl_losses.append(total_samples)
            progress.steps_snapshot.append(step)
            progress.nspl_snapshot.append(total_samples)
            progress.means.append(selected["mean"])
            progress.errors.append(selected["error"])
            progress.rsds.append(selected["rsd"])
            progress.tvars.append(selected["tvar"])
            if training_phase == "abs":
                # |complex weight| is already nonnegative; its absolute
                # statistics equal the selected training statistics.
                progress.abs_rsds.append(selected["rsd"])
                progress.abs_tvars.append(selected["tvar"])
            else:
                progress.abs_rsds.append(observables[f"abs_{training_phase}_rsd"])
                progress.abs_tvars.append(observables[f"abs_{training_phase}_tvar"])
        return progress


class SamplerCompData:
    """Comparison results keyed by configured labels.

    Existing result/progress fields and the class's import location are retained
    for saved data. Graph and old integrator metadata are optional legacy fields;
    new runs record their full configuration in ``run_configs``. Optional sampler
    checkpoint metadata lives in ``integrator_states``; results contain no live
    models or GPU tensors. Checkpoints are relative to the comparison file.
    """

    def __init__(
        self,
        integrator_identifiers: list[str],
        graph_properties: Any = None,
        target: Result | None = None,
        settings: dict[str, Any] | None = None,
        madnis_kwargs: dict[str, Any] | None = None,
        madnis_info: dict[str, Any] | None = None,
        integrand_kwargs: dict[str, Any] | None = None,
        param_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.result: dict[str, Result] = {name: Result() for name in integrator_identifiers}
        self.observables: dict[str, dict[str, Any]] = {name: {} for name in integrator_identifiers}
        self.training_progress = {name: TrainingProgress() for name in integrator_identifiers}
        self.graph_properties = graph_properties
        self.target = deepcopy(target)
        self.settings = deepcopy(settings or {})
        self.madnis_info = deepcopy(madnis_info or {})
        self.madnis_kwargs = deepcopy(madnis_kwargs or {})
        self.integrand_kwargs = deepcopy(integrand_kwargs or {})
        self.param_kwargs = deepcopy(param_kwargs or {})
        self.integrator_states: dict[str, Any] = {name: None for name in integrator_identifiers}
        self.run_configs: dict[str, dict[str, Any]] = {}
        self.training_histories: dict[str, list[dict[str, Any]]] = {}
        self.inference_histories: dict[str, list[dict[str, Any]]] = {}
        self.output_file: Path | None = None
        self._source_file: Path | None = None
        self.artifact_paths: list[Path] = []

    def __setstate__(self, state: dict[str, Any]) -> None:
        defaults = SamplerCompData([]).__dict__
        defaults.update(state)
        self.__dict__.update(defaults)
        self.result = {
            name: value if isinstance(value, Result) else Result.from_legacy(value)
            for name, value in self.result.items()
        }
        if self.target is not None and not isinstance(self.target, Result):
            self.target = Result.from_legacy(self.target)

    def restore_aggregator(self, name: str, *, use_gpu: bool | None = None) -> SamplerAggregator:
        """Restore a completed run's sampler and integrand (trusted checkpoints only).

        The restored aggregator starts with empty histories and accumulator;
        recorded comparison statistics remain available on this object.
        ``use_gpu`` can override the saved device preference for plotting.
        """
        from glnis.core.sampler_aggregator import SamplerAggregator

        if name not in self.result:
            raise KeyError(f"Unknown comparison sampler {name!r}")
        if use_gpu is not None and type(use_gpu) is not bool:
            raise ValueError("use_gpu must be bool or None")
        snapshot = deepcopy(self.integrator_states.get(name))
        if not isinstance(snapshot, dict) or not snapshot.get("save_path"):
            raise ValueError(f"No checkpoint recorded for sampler {name!r}; enable comparison.save_checkpoints")
        if name not in self.run_configs:
            raise ValueError(f"No run configuration recorded for sampler {name!r}")
        checkpoint = Path(snapshot["save_path"]).expanduser()
        if not checkpoint.is_absolute():
            source = self._source_file or self.output_file
            if source is None:
                raise ValueError("Relative checkpoint paths require the comparison file location")
            checkpoint = Path(source).parent / checkpoint
        snapshot["save_path"] = str(checkpoint.resolve())
        config = deepcopy(self.run_configs[name])
        if use_gpu is not None:
            config["sampler"]["use_gpu"] = use_gpu
        return SamplerAggregator.from_snapshot(snapshot, config)


def _save_comparison(path: Path, data: SamplerCompData) -> None:
    """Replace the persisted result only after the complete pickle is written."""
    temporary = path.with_suffix(".pkl.tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_comparison(path: str | Path) -> SamplerCompData:
    """Load a comparison pickle created by this script (trusted files only)."""
    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, SamplerCompData):
        raise ValueError(f"Expected SamplerCompData, got {type(data).__name__}")
    data._source_file = source
    data.output_file = source
    return data


def run_sampler_comp(
    file: str | Path | dict[str, Any] | ComparisonConfig,
    comment: str | None = None,
    no_output: bool = False,
    no_plot: bool = False,
) -> SamplerCompData:
    """Run each named configuration to completion before constructing the next.

    Every run has its own sampler, integrand, accumulator, and histories. Only
    recorded training measurements are used; inference starts after training.
    Optional checkpoints are saved after each completed sampler; the comparison
    data is saved once all runs finish. ``no_output`` suppresses all files.
    """
    if isinstance(file, ComparisonConfig):
        config = file
    elif isinstance(file, dict):
        config = ComparisonConfig.from_dict(file)
    else:
        config = ComparisonConfig.from_toml(file)
    settings = config.settings
    if comment is not None:
        settings["comparison"]["comment"] = comment
    comment = settings["comparison"]["comment"]
    data = SamplerCompData([], settings=settings, integrand_kwargs=config.integrand)
    data.run_configs = {
        name: dict(sampler=deepcopy(run.sampler), integrand=deepcopy(run.integrand),
                   aggregator=asdict(run.aggregator))
        for name, run in config.named_runs
    }
    output_file = None
    if not no_output:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d-%H_%M_%S_%fZ")
        output_file = config.output_dir / f"{config.run_name}_{timestamp}.pkl"
        data._source_file = output_file
        if config.save_data:
            data.output_file = output_file
            data.artifact_paths.append(output_file)

    for index, (name, run_config) in enumerate(config.named_runs, start=1):
        print(f"Running {name} ({index}/{len(config.named_runs)})", flush=True)
        aggregator = None
        try:
            aggregator = run_config.build()
            accumulator = aggregator.run()
            if index == 1:
                data.target = deepcopy(aggregator.integrand.target)
            data.result[name] = deepcopy(accumulator.result)
            data.observables[name] = accumulator.get_observables()
            data.training_progress[name] = TrainingProgress.from_history(
                aggregator.training_history, run_config.aggregator.training_phase,
            )
            data.training_histories[name] = deepcopy(aggregator.training_history)
            data.inference_histories[name] = deepcopy(aggregator.inference_history)
            data.madnis_info[name] = deepcopy(aggregator.sampler.get_diagnostics())
            data.integrator_states[name] = None
            if output_file is not None and config.save_checkpoints:
                checkpoint_path = output_file.with_name(output_file.stem + "_checkpoints") / f"{index - 1:03d}.pt"
                snapshot = deepcopy(aggregator.snapshot(checkpoint_path))
                snapshot["save_path"] = str(checkpoint_path.relative_to(output_file.parent))
                data.integrator_states[name] = snapshot
                data.artifact_paths.append(checkpoint_path)
            print(f"{name}: {len(aggregator.training_history)} training steps, "
                  f"{accumulator.result.n_points} inference samples", flush=True)
            print(aggregator.accumulator.str_report())
        except Exception as error:
            error.add_note(f"Comparison run {name!r} failed.")
            raise
        finally:
            # Models contain reference cycles. Release each before building the
            # next configuration, especially when samplers share a GPU.
            del aggregator
            gc.collect()

    if output_file is not None:
        if config.plot and not no_plot:
            data.artifact_paths.extend(plot_sampler_comp(
                output_file, data, comment, asdict(config.plotting),
            ))
        else:
            data.artifact_paths.append(write_summary(output_file, data, comment))
        if config.save_data:
            _save_comparison(output_file, data)
        print("Comparison output: " + str(output_file.parent), flush=True)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="Comparison TOML, or a saved comparison with --replot")
    parser.add_argument("--comment", default=None, help="Comment for the comparison summary")
    parser.add_argument("--no-plot", action="store_true", help="Save data and summary without PNG plots")
    parser.add_argument("--no-output", action="store_true", help="Run without writing any files")
    parser.add_argument("--replot", action="store_true", help="Plot a saved comparison pickle without sampling")
    args = parser.parse_args()
    if args.replot:
        if args.no_output or args.no_plot:
            parser.error("--replot cannot be combined with --no-output or --no-plot")
        data = load_comparison(args.file)
        comparison = data.settings.get("comparison", {})
        comment = comparison.get("comment", "") if args.comment is None else args.comment
        for artifact in plot_sampler_comp(args.file, data, comment, comparison.get("plotting", {})):
            print(artifact)
    else:
        run_sampler_comp(args.file, comment=args.comment, no_output=args.no_output, no_plot=args.no_plot)


if __name__ == "__main__":
    # Save stable module-qualified class names in pickles, including with -m or
    # direct script execution, so another Python process can load the results.
    from glnis.scripts.sampler_comparison import main as run_main

    run_main()
