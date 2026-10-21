# type: ignore
"""Headless plots and text summaries for sequential sampler comparisons."""

from __future__ import annotations

from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np


def _plot_series(axis, samples, values, *, label: str, color) -> None:
    if values is None or not len(values):
        return
    if len(samples) != len(values):
        raise ValueError(f"Training samples and measurements have different lengths for {label!r}")
    xs = np.asarray(samples, dtype=float)
    ys = np.array(values, dtype=float, copy=True)
    ys[~np.isfinite(ys)] = np.nan
    axis.plot(xs, ys, marker=".", linewidth=1.2, markersize=4, color=color, label=label)


def _metric_scale(axis, values) -> None:
    """Keep zeros and signed losses visible; use log scale for wide positive ranges."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size and np.all(finite > 0) and finite.max() / finite.min() >= 10:
        axis.set_yscale("log")
    axis.margins(y=0.15)
    axis.grid(True, alpha=0.18)


def _run_name(data) -> str:
    settings = getattr(data, "settings", {})
    return str(settings.get("comparison", {}).get("run_name", settings.get("run_name", "Sampler comparison")))


def _write_summary(path: Path, data, comment: str) -> None:
    lines = [_run_name(data), "=" * 60]
    if comment:
        lines.extend([comment, ""])
    lines.append("Training curves use the samples evaluated for each training step.")
    lines.append("Time variance uses accumulated processing time, not elapsed wall time.")
    target = getattr(data, "target", None)
    if target is not None:
        lines.append("Reference:")
        for phase in ("real", "imag"):
            value = getattr(target, phase)
            lines.append(f"  {phase}: {value.mean:.8e} +/- {value.error:.8e}")
    run_configs = getattr(data, "run_configs", {})
    for label, result in data.result.items():
        lines.extend(["", label, "-" * 60])
        progress = data.training_progress.get(label)
        if progress is not None:
            steps = progress.steps_snapshot or progress.steps_losses
            samples = progress.nspl_snapshot or progress.nspl_losses
            lines.append(f"Completed training steps: {max(steps, default=0)}")
            lines.append(f"Training samples: {max(samples, default=0)}")
        config = run_configs.get(label, {})
        phase = config.get("aggregator", {}).get("training_phase", "real")
        lines.append(f"Training phase: {phase}")
        lines.append(f"Inference samples: {result.n_points}")
        if result.n_points:
            for phase in ("real", "imag"):
                value = getattr(result, phase)
                absolute = getattr(result, f"abs_{phase}")
                lines.append(
                    f"  {phase}: {value.mean:.8e} +/- {value.error:.8e}; "
                    f"RSD={value.rsd:.6e}; ARSD={absolute.rsd:.6e}; "
                    f"TVAR={value.tvar:.6e}; ATVAR={absolute.tvar:.6e}"
                )
            lines.append(f"Processing microseconds per sample: {result.total_time / result.n_points * 1e6:.6e}")
        if config:
            lines.extend(["Resolved configuration:", pformat(config, sort_dicts=False, width=100)])
    path.write_text("\n".join(lines) + "\n")


def write_summary(file: str | Path, Data: Any, comment: str = "") -> Path:
    """Write a summary beside an output base path; the base file need not exist."""
    source = Path(file).expanduser().resolve()
    source.parent.mkdir(parents=True, exist_ok=True)
    destination = source.with_name(f"{source.stem}_summary.txt")
    _write_summary(destination, Data, comment)
    return destination


def plot_sampler_comp(
    file: str | Path,
    Data: Any,
    comment: str = "",
    plotting_settings: dict[str, Any] | None = None,
) -> list[Path]:
    """Write comparison artifacts beside ``file`` without opening GUI windows.

    The data container supplies ``training_progress``, ``result``, ``target``,
    ``settings`` and ``run_configs``. Empty phases produce no corresponding PNG.
    The returned paths include the text summary and every generated figure.
    """
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    settings = dict(plotting_settings or {})
    unknown = set(settings) - {"dpi", "figsize"}
    if unknown:
        raise ValueError(f"Unknown plotting settings: {', '.join(sorted(unknown))}")
    dpi = settings.get("dpi", 150)
    if isinstance(dpi, bool) or not isinstance(dpi, (int, float)) or not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("plotting.dpi must be a positive number")
    figsize = settings.get("figsize", (10, 8))
    if (
        not isinstance(figsize, (list, tuple)) or len(figsize) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not np.isfinite(value) or value <= 0 for value in figsize)
    ):
        raise ValueError("plotting.figsize must contain two positive numbers")

    source = Path(file).expanduser().resolve()
    summary = write_summary(source, Data, comment)
    artifacts = [summary]
    labels = list(dict.fromkeys([*Data.training_progress, *Data.result]))
    cmap = colormaps["tab10"] if len(labels) <= 10 else colormaps["turbo"]
    colors = {label: cmap(index if len(labels) <= 10 else index / max(len(labels) - 1, 1))
              for index, label in enumerate(labels)}

    def save(figure, suffix: str) -> None:
        destination = source.with_name(f"{source.stem}_{suffix}.png")
        try:
            FigureCanvasAgg(figure)
            figure.get_layout_engine().set(w_pad=0.08, h_pad=0.10)
            figure.canvas.draw()
            # Save at the same DPI and positions used for layout. Re-running
            # constrained layout during saving can shift multiline titles.
            figure.set_layout_engine("none")
            figure.savefig(destination, dpi=dpi)
            artifacts.append(destination)
        finally:
            figure.clear()

    trained = {
        label: progress for label, progress in Data.training_progress.items()
        if len(progress.nspl_losses) or len(progress.nspl_snapshot)
    }
    if trained:
        figure = Figure(figsize=figsize, dpi=dpi, layout="constrained")
        axes = figure.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": (3, 1, 1, 1)})
        series = ("losses", "rsds", "abs_rsds", "tvars")
        for label, progress in trained.items():
            for axis, attribute in zip(axes, series):
                samples = progress.nspl_losses if attribute == "losses" else progress.nspl_snapshot
                _plot_series(axis, samples, getattr(progress, attribute), label=label, color=colors[label])
        for axis, title in zip(axes, ("Loss", "RSD", "ARSD", "Time variance")):
            axis.set_ylabel(title)
            _metric_scale(axis, [value for line in axis.lines for value in line.get_ydata()])
        axes[-1].set_xlabel("Cumulative training samples")
        axes[0].legend(handles=[Line2D([], [], color=colors[label], label=label) for label in trained])
        figure.suptitle(f"{_run_name(Data)}\nTraining-batch statistics")
        save(figure, "training_prog")

    if any(result.n_points for result in Data.result.values()):
        figure = Figure(figsize=figsize, dpi=dpi, layout="constrained")
        axes = figure.subplots(4, 2, sharex=True, gridspec_kw={"height_ratios": (3, 1, 1, 1)})
        result_labels = list(Data.result)
        target = getattr(Data, "target", None)
        for column, phase in enumerate(("real", "imag")):
            axes[0, column].set_title(phase.capitalize())
            metric_values = [[], [], []]
            if target is not None:
                reference = getattr(target, phase)
                if np.isfinite(reference.mean):
                    axes[0, column].axhline(reference.mean, color="0.35", linestyle="--", label="Reference")
                    if np.isfinite(reference.error) and reference.error > 0:
                        axes[0, column].axhspan(reference.mean - reference.error, reference.mean + reference.error,
                                               color="0.5", alpha=0.15)
            for index, (label, result) in enumerate(Data.result.items()):
                if not result.n_points:
                    continue
                value = getattr(result, phase)
                absolute = getattr(result, f"abs_{phase}")
                if np.isfinite(value.mean):
                    error = value.error if np.isfinite(value.error) else None
                    axes[0, column].errorbar(index, value.mean, yerr=error, marker="o", markersize=5,
                                             capsize=4, color=colors[label], label=label)
                for row, metric in enumerate((value.rsd, absolute.rsd, value.tvar), start=1):
                    metric_values[row - 1].append(metric)
                    if np.isfinite(metric):
                        axes[row, column].plot(index, metric, marker="o", color=colors[label], label=label)
            for row, title in enumerate(("Integral", "RSD", "ARSD", "Time variance")):
                axes[row, column].set_ylabel(title)
                if row:
                    _metric_scale(axes[row, column], metric_values[row - 1])
                else:
                    axes[row, column].grid(True, alpha=0.18)
            axes[-1, column].set_xticks(range(len(result_labels)), result_labels, rotation=30, ha="right")
            axes[-1, column].set_xlim(-0.5, len(result_labels) - 0.5)
        if target is not None:
            handles, legend_labels = axes[0, 0].get_legend_handles_labels()
            reference_handles = [(handle, label) for handle, label in zip(handles, legend_labels) if label == "Reference"]
            if reference_handles:
                axes[0, 0].legend(*zip(*reference_handles))
        figure.suptitle(f"{_run_name(Data)}\nInference results")
        save(figure, "integration_result")
    return artifacts
