"""Headless plots of integrands and recovered sampler densities on slices."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from glnis.scripts.slice_plots import Slice, SlicePlotData


def _values(value: Any, size: int) -> np.ndarray | None:
    """Reject stale/mismatched arrays without breaking other sampler plots."""
    if value is None:
        return None
    result = np.asarray(value)
    if result.size != size or np.iscomplexobj(result):
        return None
    result = np.asarray(result, dtype=float).reshape(-1).copy()
    result[~np.isfinite(result)] = np.nan
    return result


def _series(slice_: Slice, size: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series = {}
    functions = getattr(slice_, "func_vals", None) or {}
    fallback = _values(slice_.func_val, size)
    for label, raw_density in (slice_.prob or {}).items():
        density = _values(raw_density, size)
        if density is None:
            continue
        density[density < 0] = np.nan
        function = _values(functions.get(label), size)
        if function is None:
            function = fallback
        if function is None:
            function = np.full(size, np.nan)
        series[label] = (function, density)
    return series


def _ratio(function: np.ndarray, density: np.ndarray) -> np.ndarray:
    result = np.full(function.shape, np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(np.abs(function), density, out=result, where=density > 0)
    result[~np.isfinite(result)] = np.nan
    return result


def _metadata(slice_: Slice, precision: int) -> str:
    def coordinates(value: Any) -> str:
        return np.array2string(np.asarray(value), precision=precision, suppress_small=True)

    directions = "; ".join(
        f"{'uv'[index]} = {coordinates(direction)}"
        for index, direction in enumerate(slice_.dirs)
    )
    strata = getattr(slice_, "strata", None)
    extra = "" if strata is None else f"; strata = {strata}"
    return (
        f"Origin = {coordinates(slice_.origin)}; channels = {slice_.discrete}{extra}\n"
        f"{directions}"
    )


def _figure_title(slice_: Slice, dimension: int, index: int, labels: list[str]) -> str:
    title = slice_.name or f"{dimension}D slice {index + 1}"
    return f"{title} — {labels[0] if len(labels) == 1 else 'sampler comparison'}"


def _plot_1d(
    slice_: Slice,
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    colors: dict[str, Any],
    normalized: bool,
    precision: int,
    index: int,
) -> Figure:
    figure = Figure(figsize=(10, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1, 0.7]})
    grid = slice_.grid[0]
    coordinates = np.linspace(grid.start, grid.stop, grid.num)
    function_label = "|f| / I_abs" if normalized else "|f|"
    for label, (function, density) in series.items():
        color = colors[label]
        axes[0].plot(coordinates, np.abs(function), "--", color=color, label=f"{label}: {function_label}")
        axes[0].plot(coordinates, density, color=color, label=f"{label}: PDF")
        axes[1].plot(coordinates, _ratio(function, density), color=color, label=label)
        axes[2].plot(coordinates, np.sign(function), color=color)
    axes[0].set_ylabel("Function magnitude / density")
    axes[0].legend(fontsize="small", ncols=2)
    axes[1].set_ylabel(f"({function_label}) / PDF")
    axes[2].set_ylabel("Sign of f")
    axes[2].set_yticks([-1, 0, 1], labels=["−", "0", "+"])
    axes[2].set_ylim(-1.25, 1.25)
    axes[2].set_xlabel("u")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(_figure_title(slice_, 1, index, list(series)))
    figure.supxlabel(_metadata(slice_, precision), fontsize="small")
    return figure


def _normalization(arrays: list[np.ndarray]) -> Normalize:
    finite_arrays = [values[np.isfinite(values)] for values in arrays]
    maximum = max((float(values.max()) for values in finite_arrays if values.size), default=0.0)
    return Normalize(vmin=0.0, vmax=maximum if maximum > 0 else 1.0)


def _plot_2d(
    slice_: Slice,
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    normalized: bool,
    precision: int,
    cmap_name: str,
    index: int,
) -> Figure:
    shape = (slice_.grid[1].num, slice_.grid[0].num)
    extent = [slice_.grid[0].start, slice_.grid[0].stop, slice_.grid[1].start, slice_.grid[1].stop]
    rows = [
        [np.abs(function), density, _ratio(function, density), np.sign(function)]
        for function, density in series.values()
    ]
    norms = [_normalization([row[column] for row in rows]) for column in range(3)]
    sign_cmap = ListedColormap(["#c44e52", "#eeeeee", "#55a868"]).with_extremes(bad="#b0b0b0")
    sign_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], sign_cmap.N)
    cmap = colormaps[cmap_name].with_extremes(bad="#b0b0b0")
    figure = Figure(figsize=(14, 3.5 * len(rows) + 1.2), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(rows), 4, squeeze=False, sharex=True, sharey=True)
    function_label = "|f| / I_abs" if normalized else "|f|"
    titles = [function_label, "Sampler PDF", f"({function_label}) / PDF", "Sign of f"]
    images = []
    for row_index, (label, row) in enumerate(zip(series, rows)):
        for column, values in enumerate(row):
            axis = axes[row_index, column]
            image = axis.imshow(
                np.ma.masked_invalid(values.reshape(shape)),
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation="nearest",
                cmap=sign_cmap if column == 3 else cmap,
                norm=sign_norm if column == 3 else norms[column],
            )
            if row_index == 0:
                axis.set_title(titles[column])
                images.append(image)
            if column == 0:
                axis.set_ylabel(f"{label}\nv")
            if row_index == len(rows) - 1:
                axis.set_xlabel("u")
    for column, image in enumerate(images):
        colorbar = figure.colorbar(image, ax=list(axes[:, column]), shrink=0.85, pad=0.02)
        if column == 3:
            colorbar.set_ticks([-1, 0, 1], labels=["−", "0", "+"])
    figure.suptitle(_figure_title(slice_, 2, index, list(series)))
    figure.supxlabel(
        _metadata(slice_, precision) + "\nGray denotes unavailable values or an undefined ratio.",
        fontsize="small",
    )
    return figure


def plot_slices(
    file: str | Path | SlicePlotData,
    force_directory: str | Path | None = None,
) -> list[Path]:
    """Write single-sampler and comparison PNGs, without changing pyplot state.

    Function arrays are already normalized by the slice generation pipeline when
    ``data.itg`` is set. No normalization is inferred from the unweighted grid.
    """
    if isinstance(file, (str, Path)):
        from glnis.scripts.slice_plots import load_slice_data

        data = load_slice_data(file)
        source = Path(file)
    else:
        data = file
        output_file = getattr(data, "output_file", None)
        source = Path(output_file) if output_file is not None else Path("outputs/slices.pkl")
    if not all(hasattr(data, field) for field in ("settings", "slices1d", "slices2d", "itg")):
        raise TypeError("Expected SlicePlotData or a saved slice data path")
    directory = Path(force_directory) if force_directory is not None else source.parent
    settings = data.settings.get("slice_plot", {}).get("plotting", {})
    plot_singles = settings.get("plot_singles", True)
    plot_comparison = settings.get("plot_comparison", True)
    dpi = settings.get("dpi", 150)
    precision = settings.get("table_precision", 3)
    cmap_name = settings.get("cmap_name", "plasma")
    # Colors retain the same meaning in every slice, including incompatible runs.
    labels = list(dict.fromkeys(
        label for slice_ in [*data.slices1d, *data.slices2d] for label in (slice_.prob or {})
    ))
    colors = {label: colormaps["tab10"](index % 10) for index, label in enumerate(labels)}
    paths = []
    for dimension, slices in ((1, data.slices1d), (2, data.slices2d)):
        for index, slice_ in enumerate(slices):
            size = int(np.prod([grid.num for grid in slice_.grid]))
            series = _series(slice_, size)
            if not series:
                continue
            groups = []
            if plot_singles:
                groups.extend(
                    (f"sampler_{model_index:03d}", {label: series[label]})
                    for model_index, label in enumerate(labels) if label in series
                )
            if plot_comparison:
                groups.append(("comparison", series))
            for suffix, group in groups:
                if dimension == 1:
                    figure = _plot_1d(slice_, group, colors, data.itg is not None, precision, index)
                else:
                    figure = _plot_2d(slice_, group, data.itg is not None, precision, cmap_name, index)
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{source.stem}_slice{dimension}d_{index:03d}_{suffix}.png"
                try:
                    figure.savefig(path, dpi=dpi)
                finally:
                    figure.clear()
                paths.append(path)
    return paths
