"""Validated settings for comparing named samplers on one integrand."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import tomllib
from typing import Any

from glnis.core.config import AggregatorConfig, RunConfig, _resolve_path, _table, _unknown
from glnis.core.integrand import Integrand


def _deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge nested tables; every other value, including lists, replaces its default."""
    result = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _sampler_paths(settings: dict[str, Any], directory: Path, path: str) -> None:
    _resolve_path(settings, "save_path", directory, path)
    for name, key in (("evaluator_metadata", "state_folder"), ("parser", "gammaloop_directory")):
        if name in settings:
            nested = _table(settings[name], f"{path}.{name}")
            _resolve_path(nested, key, directory, f"{path}.{name}")
            settings[name] = nested


@dataclass(frozen=True)
class ComparisonPlotConfig:
    dpi: int = 150
    figsize: tuple[float, float] = (10.0, 8.0)

    def __post_init__(self) -> None:
        if type(self.dpi) is not int or self.dpi <= 0:
            raise ValueError("comparison.plotting.dpi must be a positive integer")
        if (
            not isinstance(self.figsize, (list, tuple))
            or len(self.figsize) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in self.figsize)
        ):
            raise ValueError("comparison.plotting.figsize must contain two finite positive numbers")
        object.__setattr__(self, "figsize", tuple(float(value) for value in self.figsize))

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ComparisonPlotConfig:
        values = _table(config, "comparison.plotting")
        _unknown(values, {"dpi", "figsize"}, "comparison.plotting")
        return cls(**values)


@dataclass(frozen=True)
class ComparisonConfig:
    """Fully merged, validated run settings; parsing never constructs a sampler."""

    named_runs: list[tuple[str, RunConfig]]
    integrand: dict[str, Any]
    aggregator: AggregatorConfig
    run_name: str = "sampler_comparison"
    output_dir: Path = Path("outputs/sampler_comparison")
    comment: str = ""
    save_data: bool = True
    save_checkpoints: bool = False
    plot: bool = True
    plotting: ComparisonPlotConfig = ComparisonPlotConfig()
    slice_plot: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_name, str)
            or not self.run_name.strip()
            or self.run_name in {".", ".."}
            or any(character in self.run_name for character in ("/", "\\", "\0"))
        ):
            raise ValueError("comparison.run_name must be a nonempty filename without path separators")
        if not isinstance(self.output_dir, (str, Path)) or not str(self.output_dir).strip():
            raise ValueError("comparison.output_dir must be a nonempty path")
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        if not isinstance(self.comment, str):
            raise ValueError("comparison.comment must be a string")
        for name in ("save_data", "save_checkpoints", "plot"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"comparison.{name} must be bool")
        if not isinstance(self.plotting, ComparisonPlotConfig):
            raise ValueError("comparison.plotting must be a ComparisonPlotConfig")
        if not isinstance(self.aggregator, AggregatorConfig):
            raise ValueError("aggregator must be an AggregatorConfig")
        if not isinstance(self.named_runs, list) or not self.named_runs:
            raise ValueError("samplers must be a nonempty list of named configurations")
        seen: set[str] = set()
        for index, entry in enumerate(self.named_runs):
            if not isinstance(entry, tuple) or len(entry) != 2 or not isinstance(entry[1], RunConfig):
                raise ValueError(f"samplers[{index}] must pair a name with a RunConfig")
            _validate_name(entry[0], index, seen)
        object.__setattr__(self, "named_runs", list(self.named_runs))
        object.__setattr__(self, "integrand", _table(self.integrand, "integrand"))
        if self.slice_plot is not None:
            object.__setattr__(self, "slice_plot", _table(self.slice_plot, "slice_plot"))

    @property
    def settings(self) -> dict[str, Any]:
        """Resolved, independent settings suitable for storing with comparison data."""
        settings = {
            "comparison": {
                "run_name": self.run_name,
                "output_dir": str(self.output_dir),
                "comment": self.comment,
                "save_data": self.save_data,
                "save_checkpoints": self.save_checkpoints,
                "plot": self.plot,
                "plotting": asdict(self.plotting),
            },
            "integrand": deepcopy(self.integrand),
            "aggregator": asdict(self.aggregator),
            "samplers": [{"name": name, **deepcopy(run.sampler)} for name, run in self.named_runs],
        }
        if self.slice_plot is not None:
            settings["slice_plot"] = deepcopy(self.slice_plot)
        return settings

    @classmethod
    def from_dict(
        cls, config: dict[str, Any], *, base_dir: str | Path | None = None,
    ) -> ComparisonConfig:
        values = _table(config, "comparison run")
        _unknown(values, {"comparison", "integrand", "aggregator", "sampler", "samplers", "slice_plot"}, "")
        if "integrand" not in values:
            raise ValueError("Missing [integrand] configuration table")
        integrand = _table(values["integrand"], "integrand")
        # Validate before callers can run the first sampler, without allocating
        # any flows or CUDA state. All named runs share this integrand.
        try:
            Integrand.from_config(integrand)
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"integrand: {error}") from error
        aggregator = AggregatorConfig.from_dict(values.get("aggregator", {}))
        defaults = _table(values.get("sampler", {}), "sampler")
        entries = values.get("samplers")
        if not isinstance(entries, list) or not entries:
            raise ValueError("samplers must be a nonempty list of named configurations")
        directory = Path.cwd() if base_dir is None else Path(base_dir).expanduser().resolve()
        slice_plot = values.get("slice_plot")
        if slice_plot is not None:
            slice_plot = _table(slice_plot, "slice_plot")
            for key in ("comparison_file", "output_dir"):
                _resolve_path(slice_plot, key, directory, "slice_plot")

        comparison = _table(values.get("comparison", {}), "comparison")
        _unknown(comparison, {
            "run_name", "output_dir", "comment", "save_data", "save_checkpoints", "plot", "plotting",
        }, "comparison")
        plotting = ComparisonPlotConfig.from_dict(comparison.pop("plotting", {}))
        comparison.setdefault("output_dir", "outputs/sampler_comparison")
        if not isinstance(comparison["output_dir"], str) or not comparison["output_dir"].strip():
            raise ValueError("comparison.output_dir must be a nonempty path string")
        _resolve_path(comparison, "output_dir", directory, "comparison")

        seen: set[str] = set()
        runs: list[tuple[str, RunConfig]] = []
        for index, entry in enumerate(entries):
            path = f"samplers[{index}]"
            overrides = _table(entry, path)
            name = overrides.pop("name", None)
            _validate_name(name, index, seen)
            merged = _deep_merge(defaults, overrides)
            _sampler_paths(merged, directory, path)
            try:
                run = RunConfig.from_dict({
                    "sampler": merged, "integrand": integrand,
                    "aggregator": asdict(aggregator),
                })
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(f"{path} ({name!r}): {error}") from error
            if aggregator.mode == "infer" and run.sampler["training_steps"]:
                raise ValueError(f"{path} ({name!r}): sampler.training_steps must be 0 for aggregator.mode='infer'")
            runs.append((name, run))
        return cls(
            named_runs=runs, integrand=integrand, aggregator=aggregator,
            plotting=plotting, slice_plot=slice_plot, **comparison,
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> ComparisonConfig:
        source = Path(path).expanduser().resolve()
        with source.open("rb") as stream:
            return cls.from_dict(tomllib.load(stream), base_dir=source.parent)


def _validate_name(name: Any, index: int, seen: set[str]) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"samplers[{index}].name must be a nonempty string")
    if name in seen:
        raise ValueError(f"samplers[{index}].name duplicates {name!r}")
    seen.add(name)
