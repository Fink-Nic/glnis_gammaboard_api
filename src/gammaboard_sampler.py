"""GammaBoard's public sampler adapter around configurable training orchestrators."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Self, Callable

import numpy as np
from numpy.typing import NDArray

from gammaboard_process import Domain, Feedback, GenerationStatus, SampleBatch, Sampler, RectangularDomain
from glnis.core.nn.orchestrator import Orchestrator, RectangularOrchestrator, RectangularOrchestratorConfig
from glnis.utils.types import Config


@dataclass
class GammaLoopEvaluatorMetadata:
    state_folder: str
    process_id: int
    integrand_name: str
    momentum_space: bool = False
    coordinate_space: str = "x_space"
    domain_axes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, metadata):
        if metadata.get("kind") != "gammaloop":
            raise ValueError("GammaLoop metadata must have kind='gammaloop'")
        return cls(state_folder=str(metadata["state_folder"]), process_id=int(metadata["process_id"]),
                   integrand_name=str(metadata["integrand_name"]),
                   momentum_space=bool(metadata.get("momentum_space", False)),
                   coordinate_space=str(metadata.get("coordinate_space", "x_space")),
                   domain_axes=[str(axis) for axis in metadata.get("domain_axes", [])])


def parse_gammaloop_metadata(evaluator_metadata):
    if not evaluator_metadata or evaluator_metadata.get("kind") != "gammaloop":
        return None
    return GammaLoopEvaluatorMetadata.from_dict(evaluator_metadata)


def _merge_config(base: dict, overrides: dict) -> dict:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            # A different implementation starts with its own settings.
            if "kind" in value and value["kind"] != result[key].get("kind", value["kind"]):
                result[key] = deepcopy(value)
            else:
                result[key] = _merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@dataclass
class GammaBoardSamplerConfig(Config):
    save_path: str | None = None
    sampler: dict[str, Any] | None = None
    orchestrator: dict[str, Any] | None = None
    evaluator_metadata: dict[str, Any] | None = None
    parser: dict[str, Any] | None = None
    mapping: dict[str, Any] | None = None
    graph_properties: dict[str, Any] | None = None

    def __post_init__(self):
        for name in ("sampler", "orchestrator", "evaluator_metadata", "parser", "mapping", "graph_properties"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, dict):
                raise ValueError(f"{name} must be a configuration table")
            setattr(self, name, deepcopy(value))
        self.sampler = self.sampler or {}
        self.orchestrator = self.orchestrator or {}


@dataclass
class GammaBoardSamplerSnapshot:
    kind: str = "default"
    config: dict[str, Any] = field(default_factory=dict)
    orchestrator_snapshot: dict[str, Any] = field(default_factory=dict)
    pending_mapping_weights: list[float] = field(default_factory=list)
    domain: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, snapshot: dict[str, Any]) -> Self:
        values = deepcopy(snapshot)
        # Older checkpoints called the combined Jacobian/partition factor a Jacobian.
        if "pending_mapping_jac" in values:
            values.setdefault("pending_mapping_weights", values.pop("pending_mapping_jac"))
        return cls(**values)


class GammaBoardSampler(Sampler):
    KIND = "default"

    def __init__(self, *, domain: Domain, save_path: str | None = None,
                 evaluator_metadata: dict | None = None, sampler: dict | None = None,
                 orchestrator: dict | None = None, parser: dict | None = None,
                 mapping: dict | None = None, graph_properties: dict | None = None,
                 cfg: GammaBoardSamplerConfig | None = None,
                 snapshot: GammaBoardSamplerSnapshot | None = None,
                 fresh_training: bool = False, kind: str = "default"):
        if kind != self.KIND:
            raise ValueError(f"unknown sampler kind: {kind!r}")
        domain.validate()
        self.domain = domain
        self.discrete_cardinalities, self.continuous_dims = domain.require_rectangular_shape()
        self.cfg = deepcopy(cfg) if cfg is not None else GammaBoardSamplerConfig(
            save_path=save_path, sampler=sampler, orchestrator=orchestrator, parser=parser,
            mapping=mapping, graph_properties=graph_properties, evaluator_metadata=evaluator_metadata)
        self._input_discrete_cardinalities = list(self.discrete_cardinalities)
        self._input_continuous_dims = self.continuous_dims
        if self.cfg.mapping is not None:
            from glnis.core.parser import MetaDataParser
            metadata_parser = MetaDataParser(config=self.cfg.parser, metadata=self.cfg.evaluator_metadata,
                                            graph_properties=self.cfg.graph_properties)
            self.mapping = metadata_parser.get_layered_mapping_instance(self.cfg.mapping)
            output_dims = self.mapping.continuous_dims_out
            if output_dims is not None and output_dims != self._input_continuous_dims:
                raise ValueError("mapping output dimension does not match the public domain")
            self.continuous_dims = self.mapping.continuous_dims_in or self.continuous_dims
        else:
            self.mapping = None
        self._all_discrete_cardinalities = self.discrete_cardinalities + (
            self.mapping.discrete_cardinalities if self.mapping is not None else [])
        self.num_discrete_dims = len(self._all_discrete_cardinalities)
        self._sampled_discrete_indices = np.flatnonzero(np.asarray(self._all_discrete_cardinalities) > 1)
        self._mask_superfluous_cardinalities = len(self._sampled_discrete_indices) != self.num_discrete_dims
        self.orchestrator_domain = RectangularDomain(
            discrete_cardinalities=tuple(int(self._all_discrete_cardinalities[i]) for i in self._sampled_discrete_indices),
            continuous_dims=self.continuous_dims,
            strata_dims=int(sum(i < getattr(domain, "strata_dims", 0) for i in self._sampled_discrete_indices)))
        kwargs: dict[str, Any] = deepcopy(self.cfg.orchestrator or {})
        orchestrator_kind = kwargs.pop("kind", Orchestrator.DEFAULT_KIND)
        kwargs.update(sampler_kwargs=self.cfg.sampler, save_path=self.cfg.save_path)
        if snapshot is None:
            self.orchestrator = Orchestrator.from_kind(orchestrator_kind, domain=self.orchestrator_domain, **kwargs)
        else:
            if snapshot.kind != self.KIND:
                raise ValueError("checkpoint sampler kind does not match requested kind")
            if snapshot.domain is not None and snapshot.domain != domain.to_dict():
                raise ValueError("checkpoint public domain does not match requested domain")
            if orchestrator_kind != snapshot.orchestrator_snapshot["kind"]:
                raise ValueError("cannot change orchestrator kind while transferring a checkpoint")
            factory = Orchestrator.from_stage if fresh_training else Orchestrator.from_snapshot
            self.orchestrator: Orchestrator = factory(snapshot=snapshot.orchestrator_snapshot, domain=self.orchestrator_domain, **kwargs) # type: ignore
        self.cfg.orchestrator = {"kind": self.orchestrator.KIND, **self.orchestrator.cfg.to_dict()} # type: ignore
        self.cfg.orchestrator.pop("save_path", None)
        self.cfg.orchestrator.pop("sampler_kwargs", None)
        self.pending_mapping_weights = (list(snapshot.pending_mapping_weights)
                                    if snapshot is not None and not fresh_training else [])

    @classmethod
    def _checkpoint_config(cls, snapshot, args, evaluator_metadata):
        args = deepcopy(args or {})
        if args.pop("kind", cls.KIND) != cls.KIND:
            raise ValueError("unknown sampler kind")
        implementation = Orchestrator.subclass_for_kind(snapshot.orchestrator_snapshot["kind"])
        if issubclass(implementation, RectangularOrchestrator) and args.get("orchestrator") is not None:
            args["orchestrator"] = RectangularOrchestratorConfig.normalize_kwargs(**args["orchestrator"])
        config = _merge_config(snapshot.config, args)
        if evaluator_metadata is not None:
            config["evaluator_metadata"] = deepcopy(evaluator_metadata)
        return GammaBoardSamplerConfig(**config)

    @classmethod
    def from_snapshot(cls, *, snapshot: Any, domain: Domain,
                      init_args: dict[str, Any] | None = None, evaluator_metadata: Any = None) -> Self:
        """Resume a run, including pending feedback and RNG state.

        init_args overrides output path, device, precision, execution policy and
        generation chunk size. Changing the proposal, mapping, or training settings requires a new stage.
        The checkpoint is read from its recorded source path regardless of the new output path.
        """
        snapshot = GammaBoardSamplerSnapshot.from_dict(snapshot) if isinstance(snapshot, dict) else deepcopy(snapshot)
        if not snapshot.config:
            if not init_args:
                raise ValueError("checkpoint has no configuration; supply init_args")
            snapshot.config = deepcopy(init_args)
            snapshot.config.pop("kind", None)
        cfg = cls._checkpoint_config(snapshot, init_args, evaluator_metadata)
        saved = GammaBoardSamplerConfig(**snapshot.config)
        for name in ("sampler", "mapping", "graph_properties", "parser"):
            if getattr(cfg, name) != getattr(saved, name):
                raise ValueError(f"cannot override {name} during restore; use from_stage")
        return cls(domain=domain, cfg=cfg, snapshot=snapshot)

    @classmethod
    def from_stage(cls, *, source_state: Any, source_args: dict[str, Any],
                   init_args: dict[str, Any], domain: Domain, evaluator_metadata: Any = None) -> Self:
        """Keep the trained proposal and start fresh training/inference state.

        Stage settings inherit the source configuration, with init_args taking
        precedence. Optimizer, scheduler, RNG seed, counters, and pending feedback
        belong to the new stage. The proposal architecture and mapping stay compatible.
        """
        snapshot = GammaBoardSamplerSnapshot.from_dict(source_state) if isinstance(source_state, dict) else deepcopy(source_state)
        if not snapshot.config:
            snapshot.config = deepcopy(source_args)
            snapshot.config.pop("kind", None)
        cfg = cls._checkpoint_config(snapshot, init_args, evaluator_metadata)
        saved = GammaBoardSamplerConfig(**snapshot.config)
        for name in ("sampler", "mapping", "graph_properties", "parser"):
            if getattr(cfg, name) != getattr(saved, name):
                raise ValueError(f"stage handoff cannot change {name}")
        return cls(domain=domain, cfg=cfg, snapshot=snapshot, fresh_training=True)

    def snapshot(self) -> Any:
        self.orchestrator.cfg.save_path = self.cfg.save_path # type: ignore
        return asdict(GammaBoardSamplerSnapshot(
            kind=self.KIND, config=self.cfg.to_dict(),
            orchestrator_snapshot=self.orchestrator.snapshot().to_dict(),
            pending_mapping_weights=list(self.pending_mapping_weights), domain=self.domain.to_dict()))

    def generate(self, max_samples: int) -> SampleBatch | GenerationStatus:
        result = self.orchestrator.generate(max_samples)
        if isinstance(result, GenerationStatus):
            return result
        result.xs_discrete, result.xs_continuous, jac, integration_weight = self._apply_mapping(
            self._unmask_discrete(result.xs_discrete), result.xs_continuous)
        if jac is not None:
            mapping_weight = jac * integration_weight
            result.weights *= mapping_weight
            if result.training_remaining is not None:
                self.pending_mapping_weights.extend(mapping_weight.tolist())
        return result

    def feedback(self, feedback: Feedback) -> None:
        count = len(feedback.unmasked_values)
        if self.mapping is None:
            self.orchestrator.feedback(feedback)
            return
        if not self.pending_mapping_weights:
            self.orchestrator.feedback(feedback)
            return
        if len(self.pending_mapping_weights) < count:
            raise ValueError("feedback values exceed pending mapping weight length")
        mapped = Feedback(feedback.unmasked_values * np.asarray(self.pending_mapping_weights[:count]), feedback.strata)
        self.orchestrator.feedback(mapped)
        del self.pending_mapping_weights[:count]

    def pdf(self, xs_discrete: NDArray, xs_continuous: NDArray):
        discrete = np.asarray(xs_discrete)
        continuous = np.asarray(xs_continuous)
        count = len(continuous)
        if discrete.shape != (count, len(self.discrete_cardinalities)) or continuous.shape != (count, self._input_continuous_dims):
            raise ValueError("PDF coordinates do not match the public domain")
        if discrete.dtype.kind not in "iu":
            raise ValueError("discrete coordinates must be integers")
        if count == 0:
            return np.empty(0)
        valid = np.isfinite(continuous).all(axis=1)
        for index, cardinality in enumerate(self.discrete_cardinalities):
            valid &= (discrete[:, index] >= 0) & (discrete[:, index] < cardinality)
        result = np.zeros(count)
        if not valid.any():
            return result
        if self.mapping is None:
            density = self.orchestrator.pdf(self._mask_discrete(discrete[valid]), continuous[valid])
            if density is None:
                return None
            result[valid] = density
            return result
        # Hidden mapping channels are summed, using geometric Jacobians rather
        # than the partition weights used by multichannel integration estimators.
        for channel in product(*(range(n) for n in self.mapping.discrete_cardinalities)):
            mapping_discrete = np.tile(np.asarray(channel, dtype=np.int64), (int(valid.sum()), 1))
            all_discrete = np.hstack((discrete[valid], mapping_discrete))
            try:
                mapped = self.mapping.backward(mapping_discrete, continuous[valid].copy())
            except NotImplementedError:
                return None
            density = self.orchestrator.pdf(self._mask_discrete(all_discrete), mapped.continuous)
            if density is None:
                return None
            result[valid] += density * mapped.jacobian
        return result

    def discrete_mass(self, subspaces: list[dict[int, int]]) -> list[float | None]:
        results = []
        for subspace in subspaces:
            discrete = self._subspace_to_discrete(subspace)
            if any(value < 0 or value >= self.discrete_cardinalities[index] for index, value in subspace.items()):
                results.append(0.0)
            elif discrete.shape[1] == 0:
                results.append(1.0)
            else:
                results.append(self.orchestrator.discrete_mass(discrete))
        return results

    def _subspace_to_discrete(self, subspace: dict[int, int]) -> NDArray:
        values = np.full((1, len(self.discrete_cardinalities)), -1, dtype=np.int64)
        for index, value in subspace.items():
            if type(index) is not int or not 0 <= index < values.shape[1]:
                raise ValueError(f"discrete dimension index {index!r} is out of bounds")
            if type(value) is not int:
                raise ValueError("discrete constraints must be integers")
            values[0, index] = value
        sampled = self._mask_discrete(values)
        fixed = np.flatnonzero(sampled[0] >= 0)
        return sampled[:, :int(fixed[-1]) + 1] if len(fixed) else sampled[:, :0]

    def _mask_discrete(self, xs_discrete: NDArray) -> NDArray:
        indices = self._sampled_discrete_indices[self._sampled_discrete_indices < xs_discrete.shape[1]]
        return xs_discrete[:, indices]

    def _unmask_discrete(self, xs_discrete: NDArray) -> NDArray:
        if not self._mask_superfluous_cardinalities:
            return xs_discrete
        result = np.zeros((len(xs_discrete), self.num_discrete_dims), dtype=xs_discrete.dtype)
        result[:, self._sampled_discrete_indices] = xs_discrete
        return result

    def _apply_mapping(self, xs_discrete, xs_continuous, inverse=False):
        if self.mapping is None:
            return xs_discrete, xs_continuous, None, None
        public, auxiliary = np.split(xs_discrete, [len(self.discrete_cardinalities)], axis=1)
        method = self.mapping.backward if inverse else self.mapping.forward
        return public, *method(auxiliary, xs_continuous)

    def diagnostics(self) -> dict[str, Any]:
        return {"kind": self.KIND, "orchestrator": self.orchestrator.KIND,
                **self.orchestrator.diagnostics()}

    @property
    def training_status(self):
        return self.orchestrator.training_status

    def training_samples_remaining(self):
        return self.orchestrator.training_samples_remaining()

    def persisted_output(self) -> Any:
        return None
