# type: ignore
"""Evaluate and plot 1D/2D PDF slices from saved sampler comparisons."""

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
import tomllib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from glnis.core.config import _resolve_path, _table, _unknown
from glnis.scripts.sampler_comparison import SamplerCompData, load_comparison
from glnis.scripts.slice_rendering import plot_slices


@dataclass
class LinSpace:
    start: float = 0.0
    stop: float = 1.0
    num: int = 1000

    def __post_init__(self):
        for name in ("start", "stop"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not np.isfinite(value):
                raise ValueError(f"slice grid.{name} must be finite")
        if self.stop <= self.start:
            raise ValueError("slice grid.stop must exceed start")
        if type(self.num) is not int or self.num < 2:
            raise ValueError("slice grid.num must be an integer >= 2")


@dataclass
class Slice:
    name: str
    origin: NDArray
    discrete: list[int]
    dirs: list[NDArray | int]
    grid: list[LinSpace]
    func_val: NDArray | None = None
    prob: dict[str, NDArray] | None = None
    # Different sign channels/domains can have different function overlays.
    func_vals: dict[str, NDArray] = field(default_factory=dict)


class SlicePlotData:
    def __init__(
        self, graph_properties=None, settings=None, madnis_kwargs=None,
        integrand_kwargs=None, param_kwargs=None, slices1d=None, slices2d=None,
        itg: float | None = None, EPS: float = 1e-6,
    ):
        self.graph_properties = graph_properties
        self.settings = deepcopy(settings or {})
        self.madnis_kwargs = deepcopy(madnis_kwargs or {})
        self.integrand_kwargs = deepcopy(integrand_kwargs or {})
        self.param_kwargs = deepcopy(param_kwargs or {})
        self.slices1d: list[Slice] = list(slices1d or [])
        self.slices2d: list[Slice] = list(slices2d or [])
        self.EPS = EPS
        self.itg = itg
        self.skipped: dict[str, dict[str, str]] = {}
        self.output_file: Path | None = None
        self.artifact_paths: list[Path] = []

    def __setstate__(self, state):
        defaults = SlicePlotData().__dict__
        defaults.update(state)
        self.__dict__.update(defaults)


def _real_vector(value, path):
    try:
        array = np.asarray(value)
        if array.ndim != 1 or not array.size or array.dtype.kind not in "iuf":
            raise ValueError
        result = array.astype(float, copy=True)
        if not np.isfinite(result).all():
            raise ValueError
        return result
    except (TypeError, ValueError):
        raise ValueError(f"{path} must be a nonempty vector of finite real numbers") from None


@dataclass
class SlicePlotConfig:
    slices: list[Slice]
    settings: dict[str, Any]
    comparison_file: Path | None = None
    output_dir: Path | None = None
    run_name: str = "slice_plot"
    EPS: float = 1e-6
    domain: str = "auto"
    normalize: bool = True
    normalization: float | None = None
    batch_size: int | None = None
    use_gpu: bool | None = None
    save_data: bool = True
    plot: bool = True

    @classmethod
    def from_dict(cls, settings: dict[str, Any], *, base_dir: str | Path | None = None):
        settings = deepcopy(settings)
        if "slice_plot" not in settings:
            raise ValueError("Missing [slice_plot]; add it to the comparison TOML or provide --settings")
        table = _table(settings["slice_plot"], "slice_plot")
        _unknown(table, {"slices", "comparison_file", "output_dir", "run_name", "EPS", "domain",
                         "normalize", "normalization", "batch_size", "use_gpu", "save_data", "plot", "plotting"},
                 "slice_plot")
        directory = Path.cwd() if base_dir is None else Path(base_dir).expanduser().resolve()
        for name in ("comparison_file", "output_dir"):
            if name in table and (not isinstance(table[name], str) or not table[name].strip()):
                raise ValueError(f"slice_plot.{name} must be a nonempty path string")
            _resolve_path(table, name, directory, "slice_plot")
        name = table.get("run_name", "slice_plot")
        if not isinstance(name, str) or not name.strip() or name in {".", ".."} or any(c in name for c in "/\\\0"):
            raise ValueError("slice_plot.run_name must be a nonempty filename without path separators")
        for key in ("normalize", "use_gpu", "save_data", "plot"):
            if key in table and type(table[key]) is not bool:
                raise ValueError(f"slice_plot.{key} must be bool")
        eps = table.get("EPS", 1e-6)
        if type(eps) not in (float, int) or not np.isfinite(eps) or not 0 <= eps < 0.5:
            raise ValueError("slice_plot.EPS must lie in [0, 0.5)")
        domain = table.get("domain", "auto")
        if not isinstance(domain, str) or domain not in {"auto", "unit_cube", "unbounded"}:
            raise ValueError("slice_plot.domain must be auto, unit_cube, or unbounded")
        if "normalization" in table:
            value = table["normalization"]
            if type(value) not in (float, int) or not np.isfinite(value) or value <= 0:
                raise ValueError("slice_plot.normalization must be positive and finite")
        if "batch_size" in table and (type(table["batch_size"]) is not int or table["batch_size"] <= 0):
            raise ValueError("slice_plot.batch_size must be a positive integer")
        plotting = _table(table.get("plotting", {}), "slice_plot.plotting")
        _unknown(plotting, {"plot_singles", "plot_comparison", "dpi", "cmap_name", "table_precision"},
                 "slice_plot.plotting")
        for key in ("plot_singles", "plot_comparison"):
            if key in plotting and type(plotting[key]) is not bool:
                raise ValueError(f"slice_plot.plotting.{key} must be bool")
        for key, minimum in (("dpi", 1), ("table_precision", 0)):
            if key in plotting and (type(plotting[key]) is not int or plotting[key] < minimum):
                raise ValueError(f"slice_plot.plotting.{key} must be an integer >= {minimum}")
        if "cmap_name" in plotting:
            from matplotlib import colormaps
            if not isinstance(plotting["cmap_name"], str) or plotting["cmap_name"] not in colormaps:
                raise ValueError("slice_plot.plotting.cmap_name must name an installed matplotlib colormap")
        entries = table.get("slices")
        if not isinstance(entries, list) or not entries:
            raise ValueError("slice_plot.slices must contain at least one slice definition")
        slices = []
        names = set()
        for index, entry in enumerate(entries):
            path = f"slice_plot.slices[{index}]"
            entry = _table(entry, path)
            _unknown(entry, {"name", "origin", "discrete", "dirs", "grid"}, path)
            label = entry.get("name", f"slice-{index + 1}")
            if not isinstance(label, str) or not label.strip() or label in names:
                raise ValueError(f"{path}.name must be nonempty and unique")
            names.add(label)
            origin = _real_vector(entry.get("origin"), f"{path}.origin")
            discrete = entry.get("discrete", [])
            if not isinstance(discrete, list) or any(type(value) is not int for value in discrete):
                raise ValueError(f"{path}.discrete must be a list of integers")
            directions = entry.get("dirs", [0])
            if not isinstance(directions, list) or not 1 <= len(directions) <= 2:
                raise ValueError(f"{path}.dirs must contain one or two axis indices or direction vectors")
            directions = [value if type(value) is int else _real_vector(value, f"{path}.dirs")
                          for value in directions]
            grids = entry.get("grid", [{}])
            if not isinstance(grids, list) or not grids or len(grids) not in {1, len(directions)}:
                raise ValueError(f"{path}.grid must contain one grid or one per direction")
            parsed_grids = []
            for grid in grids:
                grid = _table(grid, f"{path}.grid")
                _unknown(grid, {"start", "stop", "num"}, f"{path}.grid")
                parsed_grids.append(LinSpace(**grid))
            if len(parsed_grids) == 1:
                parsed_grids *= len(directions)
            slices.append(Slice(label, origin, list(discrete), directions, parsed_grids, prob={}))
        settings["slice_plot"] = table
        options = {key: value for key, value in table.items() if key not in {"slices", "plotting"}}
        for key in ("output_dir", "comparison_file"):
            if key in options:
                options[key] = Path(options[key])
        return cls(slices=slices, settings=settings, **options)

    @classmethod
    def from_toml(cls, path: str | Path):
        source = Path(path).expanduser().resolve()
        with source.open("rb") as stream:
            return cls.from_dict(tomllib.load(stream), base_dir=source.parent)


def _geometry(slice_: Slice):
    origin = slice_.origin.copy()
    directions = []
    for direction in slice_.dirs:
        if type(direction) is int:
            if not 0 <= direction < len(origin):
                raise ValueError(f"axis {direction} is outside the origin's {len(origin)} dimensions")
            origin[direction] = 0.0
            vector = np.eye(len(origin))[direction]
        else:
            vector = np.asarray(direction)
            if vector.shape != origin.shape or not np.any(vector):
                raise ValueError("direction vectors must match the origin dimension and be nonzero")
        directions.append(vector.copy())
    if len(directions) == 2 and np.linalg.matrix_rank(np.vstack(directions)) < 2:
        raise ValueError("2D slice directions must be linearly independent")
    axes = [np.linspace(**asdict(grid)) for grid in slice_.grid]
    if len(axes) == 1:
        continuous = origin + axes[0][:, None] * directions[0]
        shape = (len(axes[0]),)
    else:
        u, v = np.meshgrid(*axes, indexing="xy")
        continuous = origin + u.ravel()[:, None] * directions[0] + v.ravel()[:, None] * directions[1]
        shape = (len(axes[1]), len(axes[0]))
    slice_.origin, slice_.dirs = origin, directions
    return continuous, shape


def _compatibility(slice_: Slice, aggregator) -> str | None:
    if len(slice_.origin) != aggregator.pdf_continuous_dims:
        return f"origin has {len(slice_.origin)} coordinates; sampler PDF expects {aggregator.pdf_continuous_dims}"
    cardinalities = aggregator.pdf_discrete_cardinalities
    if len(slice_.discrete) != len(cardinalities):
        return f"slice has {len(slice_.discrete)} discrete coordinates; sampler PDF expects {len(cardinalities)}"
    for index, (value, size) in enumerate(zip(slice_.discrete, cardinalities)):
        if not 0 <= value < size:
            return f"discrete coordinate {index}={value} is outside [0, {size})"
    return None


def _domain_mask(points, aggregator, config):
    domain = config.domain
    if domain == "auto":
        sampler = getattr(aggregator.sampler, "madnis_1", aggregator.sampler)
        mapping = getattr(sampler, "mapping", None)
        if mapping is not None:
            domain = "unit_cube" if mapping.mappings[-1].output_space.name == "HCUBE" else "unbounded"
        else:
            domain = "unit_cube" if sampler.cfg.flow_config.uniform_latent else "unbounded"
    valid = np.isfinite(points).all(axis=1)
    if domain == "unit_cube":
        valid &= ((points >= config.EPS) & (points <= 1 - config.EPS)).all(axis=1)
    return valid


def _normalization(comparison, config):
    if not config.normalize:
        return None
    if config.normalization is not None:
        return float(config.normalization)
    phase = comparison.settings.get("aggregator", {}).get("training_phase", "real")
    attribute = {"real": "abs_real", "imag": "abs_imag"}.get(phase)
    candidates = [comparison.target, *comparison.result.values()]
    for result in candidates:
        if result is None:
            continue
        if attribute is not None:
            value = getattr(result, attribute).mean
        elif result.abs_imag.mean == 0:
            value = result.abs_real.mean
        else:
            continue  # Complex magnitude is not reconstructed from separate phases.
        if np.isfinite(value) and value > 0:
            return float(value)
    return None


def _function_values(aggregator, discrete, continuous):
    if continuous.shape[1] != aggregator.integrand.continuous_dims:
        return None
    sampler = getattr(aggregator.sampler, "madnis_1", aggregator.sampler)
    mapping = getattr(sampler, "mapping", None)
    n_mapping = 0 if mapping is None else mapping.num_discrete_dims
    n_strata = len(aggregator.strat_cardinalities or [])
    start = n_strata + (0 if sampler.cfg.condition_integrand_first else n_mapping)
    n_discrete = len(aggregator.integrand.discrete_cardinalities)
    values = aggregator.integrand.evaluate_batch(continuous, discrete[:, start:start + n_discrete])
    phase = aggregator.config.training_phase
    selected = np.abs(values) if phase == "abs" else getattr(values, phase).copy()
    if n_strata:
        strata = discrete[:, :n_strata]
        if n_strata == 1:
            strata = strata[:, 0]
        transformed = aggregator.sampler.strat_channel_transform(selected.copy(), strata.copy())
        selected[np.asarray(transformed) == 0] = 0
    return selected


def load_slice_data(file: str | Path) -> SlicePlotData:
    with Path(file).expanduser().open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, SlicePlotData):
        raise ValueError(f"Expected SlicePlotData, got {type(data).__name__}")
    data.output_file = Path(file).expanduser().resolve()
    return data


def run_slice_plots(
    file: str | Path | SamplerCompData | SlicePlotData, settings_file: str | Path | None = None,
    no_output: bool = False, no_plot: bool = False, only_plot: bool = False,
    force_directory: str | Path | None = None,
) -> SlicePlotData:
    config = SlicePlotConfig.from_toml(settings_file) if settings_file else None
    source = None
    if isinstance(file, (SamplerCompData, SlicePlotData)):
        comparison = file
        source = file.output_file or getattr(file, "_source_file", None)
    else:
        source = Path(file).expanduser().resolve()
        if source.suffix.lower() == ".toml":
            if config is None:
                config = SlicePlotConfig.from_toml(source)
            if config.comparison_file is None:
                raise ValueError("slice_plot.comparison_file is required when running from a TOML")
            source = config.comparison_file
            comparison = load_comparison(source)
        else:
            with source.open("rb") as stream:
                comparison = pickle.load(stream)
            if isinstance(comparison, SamplerCompData):
                comparison._source_file = source
                comparison.output_file = source
            elif not isinstance(comparison, SlicePlotData):
                raise ValueError(f"Expected SamplerCompData or SlicePlotData, got {type(comparison).__name__}")
    if isinstance(comparison, SlicePlotData):
        if config is not None:
            comparison.settings.setdefault("slice_plot", {})["plotting"] = config.settings["slice_plot"].get("plotting", {})
        if source is not None:
            comparison.output_file = source
        if not no_output and not no_plot:
            comparison.artifact_paths.extend(plot_slices(comparison, force_directory=force_directory))
        return comparison
    if only_plot:
        raise ValueError("--replot requires saved SlicePlotData, not a sampler comparison")
    if config is None:
        config = SlicePlotConfig.from_dict(comparison.settings, base_dir=source.parent if source else None)
    if not any(comparison.integrator_states.values()):
        raise ValueError("No sampler checkpoints found; run the comparison with comparison.save_checkpoints = true")
    data = SlicePlotData(settings=config.settings, integrand_kwargs=comparison.integrand_kwargs,
                         EPS=config.EPS, itg=_normalization(comparison, config))
    prepared = []
    for specification in deepcopy(config.slices):
        prepared.append(specification)
        (data.slices1d if len(specification.dirs) == 1 else data.slices2d).append(specification)
    for name in comparison.run_configs:
        aggregator = None
        try:
            if not comparison.integrator_states.get(name):
                for slice_ in prepared:
                    reason = "no checkpoint saved for this sampler"
                    data.skipped.setdefault(slice_.name, {})[name] = reason
                    print(f"Skipping slice {slice_.name!r} for {name!r}: {reason}")
                continue
            aggregator = comparison.restore_aggregator(name, use_gpu=config.use_gpu)
            for slice_ in prepared:
                reason = _compatibility(slice_, aggregator)
                if reason is None:
                    try:
                        points, shape = _geometry(slice_)
                    except ValueError as error:
                        reason = str(error)
                if reason is not None:
                    data.skipped.setdefault(slice_.name, {})[name] = reason
                    print(f"Skipping slice {slice_.name!r} for {name!r}: {reason}")
                    continue
                mask = _domain_mask(points, aggregator, config)
                density = np.full(len(points), np.nan)
                function = np.full(len(points), np.nan)
                has_function = aggregator.pdf_continuous_dims == aggregator.integrand.continuous_dims
                indices = np.flatnonzero(mask)
                size = config.batch_size or aggregator.generation_batch_size
                for start in range(0, len(indices), size):
                    selected = indices[start:start + size]
                    discrete = np.tile(np.array(slice_.discrete, dtype=np.int64), (len(selected), 1))
                    density[selected] = aggregator.pdf(discrete, points[selected], batch_size=size)
                    values = _function_values(aggregator, discrete, points[selected])
                    if values is not None:
                        function[selected] = values if data.itg is None else values / data.itg
                slice_.prob[name] = density.reshape(shape)
                if has_function:
                    slice_.func_vals[name] = function.reshape(shape)
                    if slice_.func_val is None:
                        slice_.func_val = slice_.func_vals[name].copy()
        finally:
            del aggregator
            gc.collect()
    if no_output:
        return data
    directory = Path(force_directory).expanduser().resolve() if force_directory else config.output_dir
    if directory is None:
        directory = (source.parent if source else Path.cwd()) / "slice_plots"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d-%H_%M_%S_%fZ")
    destination = directory / f"{config.run_name}_{timestamp}.pkl"
    data.output_file = destination
    if config.save_data:
        data.artifact_paths.append(destination)
    if config.plot and not no_plot:
        data.artifact_paths.extend(plot_slices(data, force_directory=directory))
    if config.save_data:
        with destination.open("wb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="Comparison pickle, slice TOML, or saved slice data")
    parser.add_argument("--settings", type=Path, help="TOML with [slice_plot] settings")
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--replot", action="store_true", help="Plot saved slice data without restoring samplers")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.replot and (args.no_output or args.no_plot):
        parser.error("--replot cannot be combined with --no-output or --no-plot")
    data = run_slice_plots(args.file, settings_file=args.settings, no_output=args.no_output,
                          no_plot=args.no_plot, only_plot=args.replot, force_directory=args.output_dir)
    for artifact in data.artifact_paths:
        print(artifact)


if __name__ == "__main__":
    # Pickled data must reference the importable module, not __main__.
    from glnis.scripts.slice_plots import main as run_main
    run_main()
