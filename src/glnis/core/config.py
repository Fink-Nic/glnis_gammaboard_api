"""Validated configuration and factories for local sampling runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import tomllib
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from glnis.core.sampler_aggregator import SamplerAggregator

def _table(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a configuration table")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} keys must be strings")
    return deepcopy(value)


def _unknown(values: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        names = ", ".join(f"{path}.{name}" if path else name for name in unknown)
        raise ValueError(f"Unknown configuration settings: {names}")


def _mapping_space(value: Any, path: str):
    from glnis.core.mappings import Space

    if isinstance(value, str):
        try:
            value = Space[value.upper()]
        except KeyError:
            raise ValueError(f"{path} must be 'HCUBE' or 'MOMENTUM'") from None
    if not isinstance(value, Space):
        raise ValueError(f"{path} must be 'HCUBE' or 'MOMENTUM'")
    return value


def _normalize_mapping_layer(
    config: dict[str, Any], path: str, *, nested: bool = False,
) -> dict[str, Any]:
    import inspect
    from glnis.core.mappings import Mapping

    layer = _table(config, path)
    registry = Mapping.kind_registry()
    kind = layer.get("kind")
    if not isinstance(kind, str) or kind not in registry:
        raise ValueError(f"{path}.kind is unknown: {kind!r}")
    # Mapping constructors forward **kwargs through their inheritance chain.
    # Top-level spaces are assigned by LayeredMapping; wrapped mappings may
    # explicitly configure their own input space.
    allowed = {"kind"}
    excluded = {"self", "graph_properties"} | (set() if nested else {"input_space"})
    for mapping_type in registry[kind].__mro__:
        if "__init__" not in mapping_type.__dict__ or mapping_type is object:
            continue
        allowed.update(
            name for name, parameter in inspect.signature(mapping_type.__init__).parameters.items()
            if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
            and name not in excluded
        )
    _unknown(layer, allowed, path)
    if "input_space" in layer:
        layer["input_space"] = _mapping_space(layer["input_space"], f"{path}.input_space").name
    if "mapping_kwargs" in layer and layer["mapping_kwargs"] is not None:
        layer["mapping_kwargs"] = _normalize_mapping_layer(
            layer["mapping_kwargs"], f"{path}.mapping_kwargs", nested=True,
        )
    return layer


@dataclass(frozen=True)
class AggregatorConfig:
    mode: Literal["train", "infer", "train_then_infer"] = "train_then_infer"
    inference_samples: int = 0
    generation_batch_size: int | None = None
    evaluation_batch_size: int | None = None
    evaluation_workers: int = 2
    max_in_flight_batches: int = 4
    log_every: int = 10
    training_phase: Literal["real", "imag", "abs"] = "real"

    def __post_init__(self) -> None:
        for name, allowed in (
            ("mode", ("train", "infer", "train_then_infer")),
            ("training_phase", ("real", "imag", "abs")),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"aggregator.{name} must be one of {allowed!r}")
        for name in (
            "inference_samples", "generation_batch_size", "evaluation_batch_size",
            "evaluation_workers", "max_in_flight_batches", "log_every",
        ):
            value = getattr(self, name)
            if value is None and name in {"generation_batch_size", "evaluation_batch_size"}:
                continue
            minimum = 0 if name in {"inference_samples", "log_every"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"aggregator.{name} must be an integer >= {minimum}")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> AggregatorConfig:
        values = _table(config, "aggregator")
        _unknown(values, {item.name for item in fields(cls)}, "aggregator")
        return cls(**values)


def _normalize_sampler(config: dict[str, Any]) -> dict[str, Any]:
    from gammaboard_sampler import GammaBoardSamplerConfig
    from glnis.core.nn.orchestrator import Orchestrator, RectangularOrchestrator, RectangularOrchestratorConfig
    from glnis.core.nn.sampler_module import RectangularSamplerModule, MixedFlowConfig
    from glnis.core.nn.flow.continuous_flow import Flow, FlowConfig
    from glnis.core.nn.flow.discrete_flow import DiscreteFlow, TransformerConfig
    from glnis.core.nn.flow.spline_transform import SplineTransform
    from glnis.core.nn.flow.subnet_constructor import SubnetConstructor
    from glnis.core.nn.losses import Loss
    from glnis.core.nn.schedulers import Scheduler

    values = _table(config, "sampler")
    kind = values.pop("kind", "default")
    if kind != "default":
        raise ValueError(f"sampler.kind is unknown: {kind!r}")
    _unknown(values, {item.name for item in fields(GammaBoardSamplerConfig)}, "sampler")
    result = asdict(GammaBoardSamplerConfig(**values))
    orchestrator = result["orchestrator"]
    orchestrator_kind = orchestrator.pop("kind", Orchestrator.DEFAULT_KIND)
    implementation = Orchestrator.subclass_for_kind(orchestrator_kind)
    if issubclass(implementation, RectangularOrchestrator):
        settings = RectangularOrchestratorConfig.from_kwargs(**orchestrator)
        orchestrator = settings.to_dict()
        orchestrator.pop("module_kwargs")
        orchestrator.pop("save_path")
        Loss.from_kwargs(**settings.loss)
        if settings.scheduler is not None:
            Scheduler.subclass_for_kind(settings.scheduler.get("kind", Scheduler.DEFAULT_KIND))
    result["orchestrator"] = {"kind": orchestrator_kind, **orchestrator}
    module = result["module"]
    module_kind = module.get("kind", RectangularSamplerModule.DEFAULT_KIND)
    RectangularSamplerModule.subclass_for_kind(module_kind)
    if module_kind == "mixed flow":
        mixed = MixedFlowConfig.from_kwargs(**{key: value for key, value in module.items() if key != "kind"})
        continuous_kind = mixed.flow.get("kind", Flow.DEFAULT_KIND)
        Flow.subclass_for_kind(continuous_kind)
        if continuous_kind == Flow.DEFAULT_KIND:
            flow = FlowConfig.from_kwargs(**{key: value for key, value in mixed.flow.items() if key != "kind"})
            SplineTransform.subclass_for_kind(flow.spline.get("kind", SplineTransform.DEFAULT_KIND))
            SubnetConstructor.subclass_for_kind(flow.subnet.get("kind", SubnetConstructor.DEFAULT_KIND))
        discrete_kind = mixed.discrete.get("kind", DiscreteFlow.DEFAULT_KIND)
        DiscreteFlow.subclass_for_kind(discrete_kind)
        if discrete_kind == "transformer":
            transformer = TransformerConfig.from_kwargs(**{key: value for key, value in mixed.discrete.items() if key != "kind"})
            SubnetConstructor.subclass_for_kind(transformer.subnet.get("kind", SubnetConstructor.DEFAULT_KIND))

    metadata = result.get("evaluator_metadata") or {}
    if "kind" in metadata and not isinstance(metadata["kind"], str):
        raise ValueError("sampler.evaluator_metadata.kind must be a string")
    if metadata.get("kind") == "gammaloop":
        for name in ("state_folder", "integrand_name"):
            if not isinstance(metadata.get(name), str) or not metadata[name]:
                raise ValueError(f"sampler.evaluator_metadata.{name} must be a nonempty string")
        process_id = metadata.get("process_id")
        if type(process_id) is not int or process_id < 0:
            raise ValueError("sampler.evaluator_metadata.process_id must be an integer >= 0")
        if "momentum_space" in metadata and type(metadata["momentum_space"]) is not bool:
            raise ValueError("sampler.evaluator_metadata.momentum_space must be bool")
        if "coordinate_space" in metadata and not isinstance(metadata["coordinate_space"], str):
            raise ValueError("sampler.evaluator_metadata.coordinate_space must be a string")
        if "domain_axes" in metadata and (
            not isinstance(metadata["domain_axes"], list)
            or not all(isinstance(axis, str) for axis in metadata["domain_axes"])
        ):
            raise ValueError("sampler.evaluator_metadata.domain_axes must be a list of strings")

    if result.get("parser") is not None:
        parser = result["parser"]
        _unknown(parser, {"override_lmb_heuristics", "gammaloop_directory"}, "sampler.parser")
        if "override_lmb_heuristics" in parser and type(parser["override_lmb_heuristics"]) is not bool:
            raise ValueError("sampler.parser.override_lmb_heuristics must be bool")
        if "gammaloop_directory" in parser and not isinstance(parser["gammaloop_directory"], str):
            raise ValueError("sampler.parser.gammaloop_directory must be a path string")

    if result.get("graph_properties") is not None:
        from glnis.utils.types import GraphProperties

        _unknown(result["graph_properties"], {item.name for item in fields(GraphProperties)}, "sampler.graph_properties")

    if result.get("mapping") is not None:
        from glnis.core.mappings import Space

        mapping = result["mapping"]
        _unknown(mapping, {"input_space", "layer_cfgs", "use_f128"}, "sampler.mapping")
        mapping["input_space"] = _mapping_space(
            mapping.get("input_space", Space.HCUBE), "sampler.mapping.input_space",
        ).name
        if "use_f128" in mapping and type(mapping["use_f128"]) is not bool:
            raise ValueError("sampler.mapping.use_f128 must be bool")
        layers = mapping.get("layer_cfgs")
        if not isinstance(layers, list) or not layers:
            raise ValueError("sampler.mapping.layer_cfgs must be a nonempty list of configuration tables")
        mapping["layer_cfgs"] = [
            _normalize_mapping_layer(layer, f"sampler.mapping.layer_cfgs[{index}]")
            for index, layer in enumerate(layers)
        ]
        if result.get("graph_properties") is None and (result.get("evaluator_metadata") or {}).get("kind") != "gammaloop":
            raise ValueError("sampler.mapping requires sampler.graph_properties or GammaLoop sampler.evaluator_metadata")
    return result


def _resolve_path(values: dict[str, Any], key: str, directory: Path, path: str) -> None:
    value = values.get(key)
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{path}.{key} must be a path string")
    if not value:
        return
    source = Path(value).expanduser()
    values[key] = str(source if source.is_absolute() else (directory / source).resolve())


@dataclass(frozen=True)
class RunConfig:
    """A run's three sections; sampler dimensions come from the integrand."""

    sampler: dict[str, Any]
    integrand: dict[str, Any]
    aggregator: AggregatorConfig = AggregatorConfig()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampler", _normalize_sampler(self.sampler))
        object.__setattr__(self, "integrand", _table(self.integrand, "integrand"))
        if not isinstance(self.aggregator, AggregatorConfig):
            raise ValueError("aggregator must be an AggregatorConfig")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> RunConfig:
        values = _table(config, "run")
        _unknown(values, {"sampler", "integrand", "aggregator"}, "")
        for name in ("sampler", "integrand"):
            if name not in values:
                raise ValueError(f"Missing [{name}] configuration table")
        return cls(
            sampler=values["sampler"],
            integrand=values["integrand"],
            aggregator=AggregatorConfig.from_dict(values.get("aggregator", {})),
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> RunConfig:
        source = Path(path).expanduser().resolve()
        with source.open("rb") as stream:
            values = tomllib.load(stream)
        sampler = _table(values.get("sampler", {}), "sampler")
        _resolve_path(sampler, "save_path", source.parent, "sampler")
        for name, key in (("evaluator_metadata", "state_folder"), ("parser", "gammaloop_directory")):
            if name in sampler:
                nested = _table(sampler[name], f"sampler.{name}")
                _resolve_path(nested, key, source.parent, f"sampler.{name}")
                sampler[name] = nested
        if "sampler" in values:
            values["sampler"] = sampler
        return cls.from_dict(values)

    def build(self) -> SamplerAggregator:
        from gammaboard_sampler import GammaBoardSampler
        from gammaboard_process import RectangularDomain
        from glnis.core.integrand import Integrand
        from glnis.core.sampler_aggregator import SamplerAggregator

        try:
            integrand = Integrand.from_config(self.integrand)
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"integrand: {error}") from error
        values = _normalize_sampler(self.sampler)
        try:
            sampler = GammaBoardSampler(domain=RectangularDomain(
                discrete_cardinalities=tuple(integrand.discrete_cardinalities),
                continuous_dims=integrand.continuous_dims), **values)
        except (TypeError, ValueError, KeyError, IndexError) as error:
            raise ValueError(f"sampler: {error}") from error
        if sampler.mapping is not None:
            output_dims = sampler.mapping.continuous_dims_out
            if output_dims is not None and output_dims != integrand.continuous_dims:
                raise ValueError(
                    f"sampler.mapping output dimension {output_dims} does not match "
                    f"integrand.continuous_dims={integrand.continuous_dims}"
                )
        return SamplerAggregator(sampler, integrand, self.aggregator)
