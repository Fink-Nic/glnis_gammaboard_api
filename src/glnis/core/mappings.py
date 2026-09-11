# type: ignore
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass, field

# import momtrop
import numpy as np
from numpy.typing import NDArray

from glnis.utils.types import GraphProperties, LayerData

type MappingOutput = Tuple[NDArray, NDArray]

type ContinuousDim = int | None


@dataclass
class MappingConfig:
    kind: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        config = dict(config_dict)
        mapping_type = config.pop("kind", None)
        if mapping_type is None:
            raise ValueError("Mapping config must have a 'kind' key.")
        return cls(
            kind=mapping_type,
            kwargs=config or {},
        )


class Space(Enum):
    HCUBE = 0
    MOMENTUM = 1

    def inverse(self) -> "Space":
        if self is Space.HCUBE:
            return Space.MOMENTUM
        return Space.HCUBE


class Mapping(ABC):
    """
    Abstract class for a mapping layer in the parameterisation chain. 
    """

    N_SPATIAL_DIMS = 3
    IDENTIFIER = "ABCMapping"
    TRANSFORMS_SPACE = True
    CONSUMES_NEXT_N_PARAMS = 0

    def __init__(
        self,
        graph_properties: GraphProperties,
        input_space: Space = Space.HCUBE,
        cfg: Dict[str, Any] | None = None,
        **uncaught_kwargs,
    ):
        self.input_space = input_space
        self.output_space = self._get_output_space()
        self.gp = graph_properties
        self.cfg: Dict[str, Any] = cfg if cfg is not None else {}
        self.continuous_dims_in = self._get_continuous_dims(self.input_space)
        self.continuous_dims_out = self._get_continuous_dims(self.output_space)
        self.discrete_cardinalities = self._get_discrete_cardinalities()
        self.num_discrete_dims = len(self.discrete_cardinalities)

    def forward(self, continuous: NDArray, discrete: NDArray) -> MappingOutput:
        if self.input_space is Space.HCUBE:
            return self._map_from_hcube(continuous, discrete)
        return self._map_from_momentum(continuous, discrete)

    def backward(self, continuous: NDArray, discrete: NDArray) -> MappingOutput:
        if self.output_space is Space.MOMENTUM:
            return self._map_from_momentum(continuous, discrete)
        return self._map_from_hcube(continuous, discrete)

    def _map_from_hcube(self, continuous: NDArray, discrete: NDArray) -> MappingOutput:
        """
        Args:
            continuous: continuous parameters
            discrete: discrete parameters
        Returns:
            parameterised continuous output, jacobians
        """
        raise NotImplementedError(
            f"Mapping '{self.IDENTIFIER}' has not implemented the unit hypercube transform."
        )

    def _map_from_momentum(self, momentum: NDArray, discrete: NDArray) -> MappingOutput:
        """
        Args:
            continuous: continuous parameters
            discrete: discrete parameters
        Returns:
            parameterised continuous output, jacobians
        """
        raise NotImplementedError(
            f"Mapping '{self.IDENTIFIER}' has not implemented the momentum space transform."
        )

    def _prior_prob_function(self, discrete: NDArray) -> NDArray:
        num_disc_input = discrete.shape[1]
        if num_disc_input == self.num_discrete_dims:
            return np.zeros_like(discrete, dtype=np.float64)

        disc_dim = self.discrete_cardinalities[num_disc_input]
        return np.ones((len(discrete), disc_dim), dtype=np.float64) / disc_dim

    def _to_generation_lmb(
        self, momenta: NDArray, channel: NDArray, inverse: bool = False
    ) -> NDArray:
        """
        Transforms the loop momenta to the edge momentum basis of the graph, using the
        channel ID to determine the correct transformation.

        Args:
            momenta: shape (n_samples, n_loops*3)
            channel: shape (n_samples, 1)
        Returns:
            shape (n_samples, n_loops*3)
        """
        edges = self.gp.lmb_array[channel]
        shifts = np.array(self.gp.edge_momentum_shifts)
        sample_shifts = shifts[edges].reshape(-1, 3 * self.gp.n_loops)
        if not inverse:
            momenta -= sample_shifts

        if inverse:
            backward = self.gp.channel_inv_transforms[
                self.gp.generation_channel_id
            ]
            transform = backward @ self.gp.channel_transforms
        else:
            forward = self.gp.channel_transforms[
                self.gp.generation_channel_id
            ]
            transform = forward @ self.gp.channel_inv_transforms
        sample_transform = transform[channel.ravel()]
        result = sample_transform @ momenta.reshape(
            -1, self.gp.n_loops, 3
        )
        result = result.reshape(-1, self.gp.n_loops * 3)

        if inverse:
            result += sample_shifts

        return result

    def _get_discrete_cardinalities(self) -> List[int]:
        """
        Returns:
            List of shape of the discrete dimensions of this layer in the parameterisation chain.
            Intended to be used to initialize the value for self.discrete_dims
        """
        return []

    def _get_continuous_dims(self, input_space: Space) -> ContinuousDim:
        match input_space:
            case Space.HCUBE | Space.MOMENTUM:
                return self.N_SPATIAL_DIMS * self.gp.n_loops
            case _:
                raise ValueError(f"Unknown input space: {input_space}")

    def _get_output_space(self) -> Space:
        if self.TRANSFORMS_SPACE:
            return self.input_space.inverse()
        return self.input_space

    @classmethod
    def all_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield subclass
            yield from subclass.all_subclasses()

    @classmethod
    def mapping_registry(cls) -> Dict[str, type["Mapping"]]:
        registry = {}

        for subclass in cls.all_subclasses():
            identifier = subclass.IDENTIFIER

            if identifier in registry:
                raise ValueError(
                    f"Duplicate Mapping class IDENTIFIER {identifier!r}: "
                    f"{registry[identifier].__name__} and {subclass.__name__}"
                )

            registry[identifier] = subclass

        return registry

    @classmethod
    def from_kind(cls, kind: str, graph_properties: GraphProperties, input_space=Space.HCUBE, **kwargs):
        registry = cls.mapping_registry()

        try:
            subclass = registry[kind]
        except KeyError:
            raise ValueError(f"Unknown kind: {kind!r}") from None

        return subclass(graph_properties=graph_properties, input_space=input_space, **kwargs)

    @classmethod
    def from_kwargs(cls, **kwargs):
        return cls.__init__(**kwargs)


class LayeredMapping:

    def __init__(
        self,
        graph_properties,
        layer_cfgs: List[Dict[str, Any]],
        input_space: Space = Space.HCUBE,
        use_f128: bool = False,
        **uncaught_kwargs,
    ):
        self.use_f128 = use_f128
        layer_cfgs = layer_cfgs if isinstance(layer_cfgs, list) else [layer_cfgs]
        layer_cfgs: List[MappingConfig] = [MappingConfig.from_dict(cfg) for cfg in layer_cfgs]
        self.mappings: List[Mapping] = []
        next_input_space = input_space
        for layer_cfg in layer_cfgs:
            next_map = Mapping.from_kind(
                kind=layer_cfg.kind, graph_properties=graph_properties, input_space=next_input_space, **layer_cfg.kwargs
            )
            next_input_space = next_map.output_space
            self.mappings.append(next_map)
        self.continuous_dims_in = self._get_continuous_dims_in()
        self.continuous_dims_out = self._get_continuous_dims_out()
        self.discrete_cardinalities = self._get_discrete_cardinalities()
        self.num_discrete_dims = len(self.discrete_cardinalities)

    def discrete_prior_prob_function(self, discrete: NDArray, dim: int = 0) -> NDArray:
        if discrete.shape[1] == len(self.discrete_cardinalities):
            return np.zeros_like(discrete, dtype=np.float64)

        start = 0
        for mapping in self.mappings:
            if dim < mapping.num_discrete_dims:
                break
            start += mapping.num_discrete_dims
            dim -= mapping.num_discrete_dims
        else:
            raise IndexError("Discrete dimension is outside the mapping cardinalities.")

        return mapping._prior_prob_function(discrete[:, start:])

    def forward(
        self,
        discrete: NDArray | None,
        continuous: NDArray,
        wgt: NDArray,
    ) -> MappingOutput:
        return self._apply_mappings(discrete, continuous, wgt, inverse=False)

    def backward(
        self,
        discrete: NDArray | None,
        continuous: NDArray,
        wgt: NDArray,
    ) -> MappingOutput:
        return self._apply_mappings(discrete, continuous, wgt, inverse=True)

    def _apply_mappings(
        self,
        discrete: NDArray | None,
        continuous: NDArray,
        wgt: NDArray,
        inverse: bool = False,
    ) -> MappingOutput:
        """
        Args:
            discrete: shape (n_samples, n_discrete_dims)
            continuous: shape (n_samples, n_continuous_dim)
            wgt: shape (n_samples,)
        Returns:
            untransformed discrete input: shape (n_samples, n_discrete_dims - len(self.discrete_dims))
            parameterised continuous output: shape (n_samples, n_continuous_out)
            wgt * jacobian: shape (n_samples,)
        """
        if discrete is None:
            discrete = np.zeros((continuous.shape[0], 0), dtype=np.uint64)

        layer_input = LayerData(
            n_points=continuous.shape[0],
            n_cont=continuous.shape[1],
            n_disc=discrete.shape[1],
            dtype=np.float128 if self.use_f128 else np.float64
        )
        layer_input.continuous = continuous
        if inverse:
            split_at = np.cumsum(
                [mapping.num_discrete_dims for mapping in self.mappings[:-1]]
            )
            mapping_discrete_groups = np.split(discrete, split_at, axis=1)
            layer_input.discrete = np.hstack(
                [*reversed(mapping_discrete_groups)]
            )
        else:
            layer_input.discrete = discrete
        layer_input.update("sampled input")

        mappings = self.mappings if not inverse else reversed(self.mappings)
        for mapping in mappings:
            layer_continuous_dims = (
                mapping.continuous_dims_out
                if inverse
                else mapping.continuous_dims_in
            )
            if layer_continuous_dims is None:
                layer_continuous_dims = layer_input.continuous.shape[1]
            layer_cont, pass_cont = np.hsplit(
                layer_input.continuous, [layer_continuous_dims]
            )
            layer_disc, pass_disc = np.hsplit(layer_input.discrete, [mapping.num_discrete_dims])
            if not inverse:
                jac, cont = mapping.forward(layer_cont, layer_disc)
            else:
                jac, cont = mapping.backward(layer_cont, layer_disc)
            layer_input.jac *= jac
            layer_input.continuous = np.hstack([cont, pass_cont])
            layer_input.discrete = pass_disc
            layer_input.update(mapping.IDENTIFIER)

        output_jac = np.zeros((layer_input.n_points,), dtype=layer_input.dtype)
        output_jac[layer_input.success] = layer_input.jac.flatten()
        output_cont = np.zeros((layer_input.n_points, layer_input.continuous.shape[1]), dtype=layer_input.dtype)
        output_cont[layer_input.success, :] = layer_input.continuous

        return (
            output_cont,
            wgt * output_jac.flatten(),
        )

    def _get_continuous_dims_in(self) -> int:
        dim = 0
        for mapping in self.mappings[:-1]:
            dim -= mapping.continuous_dims_out or 0
            dim += mapping.continuous_dims_in or 0
        dim += self.mappings[-1].continuous_dims_in or 0
        return dim

    def _get_continuous_dims_out(self) -> int:
        return self.mappings[-1].continuous_dims_out

    def _get_discrete_cardinalities(self) -> List[int]:
        ddim = []
        for mapping in self.mappings:
            ddim.extend(mapping.discrete_cardinalities or [])
        return ddim


# class MomtropMapping(Mapping):
#     """
#     Wrapper for the momtrop sampler (arxiv.org/abs/2504.09613) rust implementation.
#     """

#     IDENTIFIER = "momtrop"

#     def __init__(
#         self,
#         edge_weight: float | List[float] | None = None,
#         sample_discrete: bool = True,
#         mask_redundant: bool = True,
#         **kwargs: Dict[str, Any],
#     ):
#         """
#         Args:
#             overwrite_edge_weight (float | List[float] | bool): Sets the propagator weights of the Feynman measure to sample from
#             sample_discrete (bool): Enable to expose the edge indices as a discrete input
#             mask_redundant (bool): If sample_discrete is enabled, will not expose the last (n_edges-1) continuous inputs
#         """
#         self.edge_weight = edge_weight
#         self.sample_discrete = sample_discrete
#         self.mask_redundant = mask_redundant and sample_discrete
#         self.gp: GraphProperties = kwargs["graph_properties"]
#         match self.edge_weight:
#             case int() | float():
#                 self.edge_weight = self.gp.n_edges * [
#                     float(self.edge_weight)
#                 ]
#             case [_, *_]:
#                 if not len(self.edge_weight) == self.gp.n_edges:
#                     raise ValueError(
#                         "If provided as a sequence, the number of momtrop edge weights must match the number of propagators."
#                     )
#             case _:
#                 default_weight = (
#                     (3 * self.gp.n_loops + 3 / 2)
#                     / self.gp.n_edges
#                     / 2
#                 )
#                 edge_weight = self.gp.n_edges * [default_weight]

#         mt_edges = [
#             momtrop.Edge(tuple(src_dst), ismassive, weight)
#             for src_dst, ismassive, weight in zip(
#                 self.gp.edge_src_dst_vertices,
#                 self.gp.edge_ismassive,
#                 self.edge_weight,
#             )
#         ]
#         assym_graph = momtrop.Graph(
#             mt_edges, self.gp.graph_external_vertices
#         )
#         momentum_shifts = [
#             momtrop.Vector(*shift)
#             for shift in self.gp.edge_momentum_shifts
#         ]
#         self.momtrop_edge_data = momtrop.EdgeData(
#             self.gp.edge_masses, momentum_shifts
#         )
#         self.momtrop_sampler = momtrop.Sampler(
#             assym_graph, self.gp.graph_signature
#         )
#         self.momtrop_sampler_settings = momtrop.Settings(False, False)
#         super().__init__(**kwargs)

#     def _map_from_hcube(
#         self,
#         continuous: NDArray,
#         discrete: NDArray,
#     ) -> MappingOutput:
#         if self.mask_redundant:
#             continuous = np.hstack(
#                 [
#                     continuous,
#                     np.zeros(
#                         (continuous.shape[0], self.gp.n_edges - 1),
#                         dtype=continuous.dtype,
#                     ),
#                 ]
#             )
#         if discrete.size == 0:
#             samples = self.momtrop_sampler.sample_batch(
#                 continuous,
#                 self.momtrop_edge_data,
#                 self.momtrop_sampler_settings,
#                 None,
#             )
#         else:
#             samples = self.momtrop_sampler.sample_batch(
#                 continuous,
#                 self.momtrop_edge_data,
#                 self.momtrop_sampler_settings,
#                 self._get_graph_from_edges_removed(discrete),
#             )

#         jac = np.array(samples.jacobians, dtype=continuous.dtype).reshape(-1, 1)
#         momentum = np.array(samples.loop_momenta, dtype=continuous.dtype).reshape(
#             len(continuous), -1
#         )

#         return jac, momentum

#     def _get_graph_from_edges_removed(
#         self, edges_removed: NDArray | None = None
#     ) -> List[List[int]]:
#         """
#         Args:
#             edges_removed: List of the edge indices that have already been removed from the graph
#         Returns:
#             List of shape (n_edges,) that appends the as-yet unforced edges to edges_removed
#         """
#         n_edges = self.gp.n_edges
#         n_points, k = edges_removed.shape
#         full_graph = np.arange(n_edges)
#         if edges_removed is None:
#             return [full_graph.tolist()]
#         if k > n_edges:
#             raise ValueError(f"Too many edges removed: {k} > {n_edges}")

#         edges_removed = edges_removed.astype(np.uint64)

#         if k == 0:
#             return np.tile(full_graph, (n_points, 1))

#         result = np.empty((n_points, n_edges), dtype=np.uint64)
#         result[:, :k] = edges_removed
#         # Check if edges_removed contains duplicates, replace with arange, prior will be zero anyways
#         duplicate_mask = np.any(
#             np.diff(np.sort(edges_removed, axis=1), axis=1).reshape(n_points, -1) == 0,
#             axis=1,
#         )
#         if np.any(duplicate_mask):
#             edges_removed[duplicate_mask] = np.arange(k)

#         # If only one edge is left, we can directly return the result without masking (which is more expensive)
#         if k == n_edges - 1:
#             valid_sum = n_edges * (n_edges - 1) / 2
#             result[:, -1] = valid_sum - np.sum(edges_removed, axis=1)
#             return result

#         # mask[i, j] == True ⇔ edge j is still available for sample i
#         mask = np.ones((n_points, n_edges), dtype=bool)
#         mask[np.arange(n_points).reshape(-1, 1), edges_removed] = False

#         # shape: (n, n_edges - k)
#         remaining = np.nonzero(mask)[1].reshape(n_points, -1)
#         result[:, k:] = remaining

#         return result

#     def _prior_prob_function(self, indices: NDArray) -> NDArray:
#         return np.array(self.momtrop_sampler.predict_discrete_probs(indices.tolist()))

#     def _get_continuous_dims(self, input_space: Space) -> int:
#         match input_space:
#             case Space.HCUBE:
#                 if self.mask_redundant:
#                     return (
#                         self.momtrop_sampler.get_dimension() - self.gp.n_edges + 1
#                     )
#                 return self.momtrop_sampler.get_dimension()
#             case Space.MOMENTUM:
#                 return self.N_SPATIAL_DIMS * self.gp.n_loops
#             case _:
#                 raise ValueError(f"Unknown input space: {input_space}")

#     def _get_discrete_cardinalities(self) -> List[int]:
#         if not self.sample_discrete:
#             return []
#         n_edges = self.gp.n_edges
#         if self.mask_redundant:
#             return (n_edges - 1) * [n_edges]
#         return n_edges * [n_edges]


class SphericalMapping(Mapping):
    IDENTIFIER = "spherical"

    def __init__(self, conformal_scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.conformal_scale = conformal_scale
        if self.gp.e_cm > 0.0:
            self.conformal_scale *= self.gp.e_cm
        self.n_loops = self.gp.n_loops

    def _map_from_hcube(
        self,
        continuous: NDArray,
        discrete: NDArray,
    ) -> MappingOutput:
        momentum = np.zeros_like(continuous)
        n_points = continuous.shape[0]

        jac = np.ones((n_points, 1), dtype=continuous.dtype)
        jac *= (4 * np.pi * self.conformal_scale**3) ** self.n_loops

        for i_loop in range(self.n_loops):
            _start = self.N_SPATIAL_DIMS * i_loop
            _end = self.N_SPATIAL_DIMS * (i_loop + 1)

            xs = continuous[:, _start:_end]
            x, y, z = np.hsplit(xs, [1, 2])

            r = x / (1 - x)
            cos_az = 2 * y - 1
            sin_az = np.sqrt(1 - cos_az**2)
            pol = 2 * np.pi * z

            ks = (
                self.conformal_scale
                * r
                * np.hstack([sin_az * np.cos(pol), sin_az * np.sin(pol), cos_az])
            )
            momentum[:, _start:_end] = ks
            jac *= x**2 / (1 - x) ** 4
        
        if discrete.size == 0:
            discrete = np.empty((momentum.shape[0], 1), dtype=np.uint64)
            discrete.fill(self.gp.generation_channel_id)
        momentum = self._to_generation_lmb(momentum, discrete)

        return jac, momentum.reshape(n_points, -1)

    def _map_from_momentum(self, momentum: NDArray, discrete: NDArray) -> MappingOutput:
        # Use the inverse spherical parameterisation logic to map momentum -> unit hypercube
        if discrete.size == 0:
            discrete = np.empty((momentum.shape[0], 1), dtype=np.uint64)
            discrete.fill(self.gp.generation_channel_id)
        continuous = self._to_generation_lmb(momentum, discrete, inverse=True)
        xs = np.zeros_like(continuous)

        jac = np.ones((len(continuous), 1), dtype=continuous.dtype)
        jac /= (4 * np.pi * self.conformal_scale**3) ** self.n_loops

        for i_loop in range(self.n_loops):
            _start = self.N_SPATIAL_DIMS * i_loop
            _end = self.N_SPATIAL_DIMS * (i_loop + 1)
            ks = continuous[:, _start:_end]

            k0, k1, k2 = np.hsplit(ks, [1, 2])

            r = np.linalg.norm(ks, axis=1).reshape(-1, 1)
            cos_az = k2 / r
            tan_pol = (k1 / k0).reshape(-1, 1)
            pol: NDArray = np.arctan(tan_pol)
            pol += np.pi * (1 - np.sign(k0)) / 2 * np.sign(k1)
            pol += np.pi * (1 - np.sign(pol))

            r /= self.conformal_scale
            x = r / (1 + r)
            y = (cos_az + 1) / 2
            z = pol / 2 / np.pi
            xs[:, _start:_end] = np.hstack([x, y, z])

            jac /= x**2 / (1 - x) ** 4

        return jac, xs

    def _jac_from_momentum(self, momentum: NDArray, discrete: NDArray) -> NDArray:
        # Standalone inverse jacobian for multichanneling scheme
        if discrete.size == 0:
            discrete = np.empty((momentum.shape[0], 1), dtype=np.uint64)
            discrete.fill(self.gp.generation_channel_id)
        momentum = self._to_generation_lmb(momentum, discrete, inverse=True)

        jac = np.ones((len(momentum), 1), dtype=momentum.dtype)
        jac /= (4 * np.pi * self.conformal_scale**3) ** self.n_loops

        for i_loop in range(self.n_loops):
            _start = self.N_SPATIAL_DIMS * i_loop
            _end = self.N_SPATIAL_DIMS * (i_loop + 1)
            ks = momentum[:, _start:_end]
            r = np.linalg.norm(ks, axis=1).reshape(-1, 1)
            r /= self.conformal_scale
            x = r / (1 + r)
            jac /= x**2 / (1 - x) ** 4

        return jac


class MadspaceMapping(Mapping):
    IDENTIFIER = "madspace"

    def __init__(
        self,
        return_full_phase_space: bool = False,
        diagram: str = "epem_a_ttxh_LO",
        **kwargs,
    ):
        import madspace as ms

        self.diagram = diagram

        match self.diagram:
            case "epem_a_ttxh_LO":
                self.ms_diagram = ms.Diagram(
                    incoming_masses=[0.0, 0.0],
                    outgoing_masses=[173.0, 173.0, 125.0],
                    propagators=[
                        ms.Propagator(*prop) for prop in [[0.0, 0.0], [173.0, 0.0]]
                    ],
                    vertices=[
                        ["i0", "i1", "p0"],
                        ["p0", "o0", "p1"],
                        ["p1", "o1", "o2"],
                    ],
                )
                self.mapping = ms.PhaseSpaceMapping(
                    ms.Topology(self.ms_diagram),
                    kwargs["graph_properties"].e_cm,
                    permutations=[[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                )
            case "epem_a_ddx_LO":
                self.ms_diagram = ms.Diagram(
                    incoming_masses=[0.0, 0.0],
                    outgoing_masses=[0.0, 0.0],
                    propagators=[ms.Propagator(*prop) for prop in [[0.0, 0.0]]],
                    vertices=[["i0", "i1", "p0"], ["p0", "o0", "o1"]],
                )
                self.mapping = ms.PhaseSpaceMapping(
                    ms.Topology(self.ms_diagram),
                    kwargs["graph_properties"].e_cm,
                )
        self.return_full_phase_space = return_full_phase_space
        self.return_full_phase_space = False
        super().__init__(**kwargs)

    def _get_continuous_dims(self, input_space: Space) -> ContinuousDim:
        match input_space:
            case Space.HCUBE:
                return self.mapping.random_dim()
            case Space.MOMENTUM:
                if self.return_full_phase_space:
                    return self.mapping.particle_count() * self.N_SPATIAL_DIMS
                return (len(self.ms_diagram.outgoing_masses) - 1) * self.N_SPATIAL_DIMS
            case _:
                raise ValueError(f"Unknown input space: {input_space}")

    def _map_from_hcube(
        self,
        continuous: NDArray,
        discrete: NDArray,
    ) -> MappingOutput:
        momenta_permutations = np.array(
            [
                [4, 3],
                [2, 4],
            ]
        )
        permut = np.array(momenta_permutations[discrete], dtype=np.int32)

        result = self.mapping.map_forward(
            [continuous], [discrete.flatten().astype(np.int32)]
        )
        jac = result.det.reshape(-1, 1)
        if self.return_full_phase_space:
            momenta = result.momenta[:, :, 1:].reshape(continuous.shape[0], -1)
        else:
            momenta = np.zeros(
                (continuous.shape[0], len(self.ms_diagram.outgoing_masses) - 1, 3),
                dtype=continuous.dtype,
            )
            momenta[discrete.flatten() == 0] = result.momenta[
                discrete.flatten() == 0, :, 1:
            ][..., [4, 3], :]
            momenta[discrete.flatten() == 1] = result.momenta[
                discrete.flatten() == 1, :, 1:
            ][..., [2, 4], :]
            momenta[:, 0, :] = momenta[:, 0, :] - momenta[:, 1, :]

        momenta = momenta.reshape(continuous.shape[0], -1)

        return jac, momenta


class KaapoParameterisation(Mapping):
    IDENTIFIER = "kaapo"

    def __init__(
        self,
        mu: List[float] | float = np.pi,
        a: float = 0.5,
        b: float = 1.0,
        vary_a: bool = False,
        a_min: float = 0.2,
        angle_shift: float = 0.0,
        **kwargs,
    ):
        self.a = a
        self.b = b if b else self.gp.e_cm
        self.vary_a = vary_a
        self.a_min = a_min
        self.angle_shift = angle_shift

        super().__init__(**kwargs)
        self.mu = mu
        if not type(self.mu) == list:
            self.mu: list[float] = self.gp.n_edges * [self.mu]

    def _map_from_hcube(
        self, continuous: NDArray, discrete: NDArray
    ) -> MappingOutput:
        if discrete.size == 0:
            discrete = np.zeros((continuous.shape[0], 1), dtype=np.uint64)
        # For easier reading
        n_loops = self.gp.n_loops
        n_points = continuous.shape[0]
        dtype = continuous.dtype
        if self.vary_a:
            a = self.a_min + (1.0 - self.a_min) * continuous[:, -1].reshape(-1, 1)
        else:
            a = self.a
        b = self.b

        momentum = np.zeros((n_points, 3 * n_loops), dtype=dtype)

        # The constant part of the jacobian
        jac = np.ones((n_points, 1), dtype=dtype)
        jac *= (4 * np.pi / a / b**a) ** n_loops

        for i_loop in range(n_loops):
            basis_edge = self.gp.lmb_array[discrete, i_loop]
            m_e = np.array(self.gp.edge_masses)[basis_edge]
            mu = np.array(self.mu)[basis_edge]
            p_F = np.clip(mu**2 - m_e**2, a_min=0.0, a_max=None) ** 0.5

            _start = self.N_SPATIAL_DIMS * i_loop
            _end = self.N_SPATIAL_DIMS * (i_loop + 1)

            xs = continuous[:, _start:_end]
            x1, x2, x3 = np.hsplit(xs, [1, 2])
            if self.angle_shift != 0.0:
                x2 = (x2 + self.angle_shift) % 1
                x3 = (x3 + self.angle_shift) % 1

            cos_az = 2 * x2 - 1
            sin_az = np.sqrt(1 - cos_az**2)
            pol = 2 * np.pi * x3

            # Discriminator around the fermi surface and origin
            peak_F: NDArray = b**a * x1 / (1 - x1) - p_F**a

            # Radial component
            h_c = p_F + np.sign(peak_F) * np.abs(peak_F) ** (1 / a)

            # Standard spherical parameterisation, scaled by h_c
            k_vec = h_c * np.hstack(
                [sin_az * np.cos(pol), sin_az * np.sin(pol), cos_az]
            )
            momentum[:, _start:_end] = k_vec

            # Calculate the jacobian
            jac *= h_c**2 * np.abs(peak_F) ** (1 / a - 1)
            jac *= (np.sign(peak_F) * np.abs(peak_F) + p_F**a + b**a) ** 2

        momentum = self._to_generation_lmb(momentum, discrete)

        return jac, momentum.reshape(n_points, -1)

    def _get_continuous_dims(self, input_space: Space) -> ContinuousDim:
        match input_space:
            case Space.HCUBE:
                c_dim = self.N_SPATIAL_DIMS * self.gp.n_loops
                if self.vary_a:
                    c_dim += 1
                return c_dim
            case Space.MOMENTUM:
                return self.N_SPATIAL_DIMS * self.gp.n_loops
            case _:
                raise ValueError(f"Unknown input space: {input_space}")


class RKaapoParameterisation(Mapping):
    IDENTIFIER = "reduced kaapo"

    def __init__(
        self,
        mu: List[float] | float = np.pi,
        a: float = 0.5,
        b: float = 1.0,
        vary_a: bool = False,
        a_min: float = 0.2,
        angle_shift: float = 0.0,
        **kwargs,
    ):
        self.a = a
        self.b = b if b else self.gp.e_cm
        self.vary_a = vary_a
        self.a_min = a_min
        self.angle_shift = angle_shift

        super().__init__(**kwargs)
        self.mu = mu
        if not type(self.mu) == list:
            self.mu: list[float] = self.gp.n_edges * [self.mu]

    def _map_from_hcube(self, continuous: NDArray, discrete: NDArray) -> MappingOutput:
        if discrete.size == 0:
            discrete = np.zeros((continuous.shape[0], 1), dtype=np.uint64)
        # For easier reading
        n_loops = self.gp.n_loops
        n_points = continuous.shape[0]
        dtype = continuous.dtype
        if self.vary_a:
            a = self.a_min + (1.0 - self.a_min) * continuous[:, -1].reshape(-1, 1)
        else:
            a = self.a
        b = self.b

        momentum = np.zeros((n_points, 3 * n_loops), dtype=dtype)

        # The constant part of the jacobian
        jac = np.ones((n_points, 1), dtype=dtype)
        jac *= (4 * np.pi / a / b**a) ** n_loops

        for i_loop in range(n_loops):
            basis_edge = self.gp.lmb_array[discrete, i_loop]
            m_e = np.array(self.gp.edge_masses)[basis_edge]
            mu = np.array(self.mu)[basis_edge]
            p_F = np.clip(mu**2 - m_e**2, a_min=0.0, a_max=None) ** 0.5

            _start = self.N_SPATIAL_DIMS * i_loop
            _end = self.N_SPATIAL_DIMS * (i_loop + 1)

            if i_loop == 0:
                x1 = continuous[:, 0].reshape(-1, 1)
                cos_az = np.ones((continuous.shape[0], 1), dtype=continuous.dtype)
                sin_az = np.zeros((continuous.shape[0], 1), dtype=continuous.dtype)
                pol = 0
            elif i_loop == 1:
                x1 = continuous[:, 1].reshape(-1, 1)
                x2 = continuous[:, 2].reshape(-1, 1)
                if self.angle_shift != 0.0:
                    x2 = (x2 + self.angle_shift) % 1
                cos_az = 2 * x2 - 1
                sin_az = np.sqrt(1 - cos_az**2)
                pol = 0
            else:
                xs = continuous[
                    :, _start - self.N_SPATIAL_DIMS: _end - self.N_SPATIAL_DIMS
                ]
                x1, x2, x3 = np.hsplit(xs, [1, 2])
                if self.angle_shift != 0.0:
                    x2 = (x2 + self.angle_shift) % 1
                    x3 = (x3 + self.angle_shift) % 1

                cos_az = 2 * x2 - 1
                sin_az = np.sqrt(1 - cos_az**2)
                pol = 2 * np.pi * x3

            # Discriminator around the fermi surface and origin
            peak_F: NDArray = b**a * x1 / (1 - x1) - p_F**a

            # Radial component
            h_c = p_F + np.sign(peak_F) * np.abs(peak_F) ** (1 / a)

            # Standard spherical parameterisation, scaled by h_c
            k_vec = h_c * np.hstack(
                [sin_az * np.cos(pol), sin_az * np.sin(pol), cos_az]
            )
            momentum[:, _start:_end] = k_vec

            # Calculate the jacobian
            jac *= h_c**2 * np.abs(peak_F) ** (1 / a - 1)
            jac *= (np.sign(peak_F) * np.abs(peak_F) + p_F**a + b**a) ** 2

        momentum = self._to_generation_lmb(momentum, discrete)

        return jac, momentum.reshape(n_points, -1)

    def _get_continuous_dims(self, input_space: Space) -> ContinuousDim:
        match input_space:
            case Space.HCUBE:
                if self.gp.n_loops == 1:
                    c_dim = 1
                else:
                    c_dim = self.N_SPATIAL_DIMS * (self.gp.n_loops - 1)
                if self.vary_a:
                    c_dim += 1
                return c_dim
            case Space.MOMENTUM:
                return self.N_SPATIAL_DIMS * self.gp.n_loops
            case _:
                raise ValueError(f"Unknown input space: {input_space}")


class SParameterisation(Mapping):
    IDENTIFIER = "s"

    def __init__(self, exponent: float = 2.0, **kwargs):
        """
        Args:
            exponent: Exponent of the S transformation. Higher values lead to stronger smoothing around the edges of the unit cube
        """
        # This transform operates in the unit hypercube only
        self.TRANSFORMS_SPACE = False
        super().__init__(**kwargs)
        self.exponent = max(exponent, 1.0)

    def _map_from_hcube(
        self, continuous: NDArray, discrete: NDArray
    ) -> MappingOutput:
        xn = np.power(continuous, self.exponent)
        denom = xn + np.power(1 - continuous, self.exponent)
        cont = xn / denom

        num = continuous - continuous * continuous
        jac = (
            self.exponent
            * np.power(
                num, self.exponent - 1.0, where=num != 0, out=np.zeros_like(continuous)
            )
            / denom
            / denom
        )
        jac = np.prod(jac, axis=1, keepdims=True)

        return jac, cont

    def _get_continuous_dims(self, input_space: Space) -> ContinuousDim:
        # This mapping does not expose a continuous dimension
        return None


class IdentityParameterisation(Mapping):
    IDENTIFIER = "identity"

    def __init__(
        self,
        seed: int = 42,
        uniform_continuous: bool = False,
        force_dim: int | None = None,
        **kwargs,
    ):
        self.seed = seed
        self.uniform_continuous = uniform_continuous
        self.rng = np.random.default_rng(seed)
        self.force_dim = force_dim
        # Operates in the same space (doesn't switch HCUBE <-> MOMENTUM)
        self.TRANSFORMS_SPACE = False
        super().__init__(**kwargs)

    def _map_from_hcube(self, continuous: NDArray, discrete: NDArray) -> MappingOutput:
        jac = np.ones((continuous.shape[0], 1), dtype=continuous.dtype)
        if self.uniform_continuous:
            return jac, self.rng.random(size=continuous.shape)
        return jac, continuous

    def _map_from_momentum(self, momentum: NDArray, discrete: NDArray) -> MappingOutput:
        # Identity inverse is the same as forward
        return self._map_from_hcube(continuous, discrete)

    def _get_continuous_dims(self, input_space: Space) -> ContinuousDim:
        if self.force_dim is not None:
            return self.force_dim
        # Signal that this mapping does not expose a fixed continuous input dimension
        return None


class MCLayer(Mapping, ABC):
    IDENTIFIER = "multichanneling"

    def __init__(self, mapping_kwargs: Dict[str, Any] | None = None, **kwargs):
        self.gp = kwargs["graph_properties"]
        mapping_cfg = MappingConfig.from_dict(mapping_kwargs or {"kind": "spherical"})
        self.mapping = Mapping.from_kind(mapping_cfg.kind, self.gp, **mapping_cfg.kwargs)
        super().__init__(**kwargs)
        self.IDENTIFIER = (
            f"multichanneling: {self.IDENTIFIER} using {self.mapping.IDENTIFIER}"
        )

        self.lmbs = self.gp.lmb_array
        self.n_channels = self.gp.n_channels
        self.n_loops = self.gp.n_loops

        self.shifts = np.array(self.gp.edge_momentum_shifts)
        self.channel_shifts = self.shifts[self.lmbs]
        self.channel_masses = np.array(self.gp.edge_masses)[self.lmbs]
        backward = self.gp.channel_inv_transforms[
            self.gp.generation_channel_id
        ]
        self.transforms = backward @ self.gp.channel_transforms

    def _map_from_hcube(self, continuous: NDArray, discrete: NDArray) -> MappingOutput:
        jac, momentum = self.mapping.forward(continuous, discrete)
        jac *= self._mc_weight(momentum, discrete).reshape(-1, 1)
        return jac, momentum

    def _map_from_momentum(self, momentum: NDArray, discrete: NDArray) -> MappingOutput:
        jac, continuous = self.mapping.backward(momentum, discrete.copy())
        jac /= self._mc_weight(momentum, discrete).reshape(-1, 1)
        return jac, continuous

    @abstractmethod
    def _mc_weight(self, momentum: NDArray, discrete: NDArray) -> NDArray:
        return np.ones((momentum.shape[0], 1), dtype=momentum.dtype)

    def _get_discrete_cardinalities(self) -> List[int]:
        return [self.gp.n_channels]

    def _get_continuous_dims(self, input_space: Space) -> ContinuousDim:
        return self.mapping._get_continuous_dims(input_space)

    @classmethod
    def from_kwargs(
        cls,
        graph_properties: GraphProperties,
        subtype: str = "ose",
        **kwargs,
    ):
        return cls.from_kind(kind=subtype, graph_properties=graph_properties, **kwargs)


class OSEMCLayer(MCLayer):
    IDENTIFIER = "ose"

    def __init__(self, ose_exponent: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.ose_exponent = ose_exponent

    def _mc_weight(self, momentum: NDArray, discrete: NDArray) -> NDArray:
        # Need to calculate the e-surface term for all lmbs
        mc_weight = np.prod(  # Multiply for each loop
            np.sum(  # Dot product
                (
                    self.transforms[discrete.ravel()]
                    @ momentum.reshape(-1, self.n_loops, 3)
                    + self.channel_shifts[discrete.ravel()]
                )
                ** 2,
                axis=2,
            )
            + self.channel_masses[discrete.ravel()] ** 2,
            axis=1,
        )
        mc_weight = np.power(
            mc_weight,
            -self.ose_exponent / 2.0,
            where=mc_weight != 0,
            out=np.zeros_like(mc_weight),
        )
        norm_factor = np.zeros_like(mc_weight)
        for ch in range(self.n_channels):
            transform = self.transforms[ch]
            shift = self.channel_shifts[ch]
            mass = self.channel_masses[ch]
            weight = np.prod(  # Multiply for each loop
                np.sum(  # Dot product
                    (transform @ momentum.reshape(-1, self.n_loops, 3) + shift) ** 2, axis=2,
                )
                + mass**2,
                axis=1,
            )
            norm_factor += np.power(
                weight,
                -self.ose_exponent / 2.0,
                where=weight != 0,
                out=np.zeros_like(weight),
            )

        return np.divide(
            mc_weight, norm_factor, out=np.zeros_like(mc_weight), where=norm_factor != 0
        )


class FermiMCLayer(MCLayer):
    IDENTIFIER = "fermi"

    def __init__(
        self,
        ose_exponent: float = 4.0,
        fermi_exponent: float = 1.0,
        set_bosonic_edge_to_one: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ose_exponent = ose_exponent
        self.fermi_exponent = fermi_exponent
        self.set_bosonic_edge_to_one = set_bosonic_edge_to_one
        if hasattr(self.mapping, "mu"):
            self.channel_mu = np.array(self.mapping.mu)[self.lmbs]
        else:
            self.channel_mu = np.zeros((self.n_channels, self.n_loops))

    def _mc_weight(self, momentum: NDArray, discrete: NDArray) -> NDArray:
        momentum = momentum.reshape(-1, self.n_loops, 3)
        self.param: KaapoParameterisation
        # Need to calculate the fermi surface term for all lmbs
        e_surface_terms = np.sqrt(
            np.sum(
                (
                    self.transforms[discrete.ravel()]
                    @ momentum.reshape(-1, self.n_loops, 3)
                    + self.channel_shifts[discrete.ravel()]
                )
                ** 2,
                axis=2,
            )
            + self.channel_masses[discrete.ravel()] ** 2
        )
        fermi_weights = np.abs(e_surface_terms - self.channel_mu[discrete.ravel()])
        if self.set_bosonic_edge_to_one:
            bosonic_edge_mask = self.channel_mu[discrete.ravel()] == 0.0
            fermi_weights[bosonic_edge_mask] = 1.0
        mc_weight = np.prod(e_surface_terms, axis=1)
        mc_weight = np.power(
            mc_weight,
            -self.ose_exponent,
            where=mc_weight != 0,
            out=np.zeros_like(mc_weight),
        )
        fermi_weight = np.prod(fermi_weights, axis=1)
        mc_weight *= np.power(
            fermi_weight,
            -self.fermi_exponent,
            where=fermi_weight != 0,
            out=np.zeros_like(fermi_weight),
        )

        norm_factor = np.zeros_like(mc_weight)
        for ch in range(self.n_channels):
            transform = self.transforms[ch]
            shift = self.channel_shifts[ch]
            mass = self.channel_masses[ch]
            mu = self.channel_mu[ch]
            e_surface_terms = np.sqrt(
                np.sum(
                    (transform @ momentum.reshape(-1, self.n_loops, 3) + shift) ** 2,
                    axis=2,
                )
                + mass**2
            )
            fermi_surface_terms = np.abs(e_surface_terms - mu)
            if self.set_bosonic_edge_to_one:
                bosonic_edge_mask = mu == 0.0
                fermi_surface_terms[:, bosonic_edge_mask] = 1.0
            weight = np.prod(e_surface_terms, axis=1)
            weight = np.power(
                weight, -self.ose_exponent, where=weight != 0, out=np.zeros_like(weight)
            )
            fermi_weight = np.prod(fermi_surface_terms, axis=1)
            weight *= np.power(
                fermi_weight,
                -self.fermi_exponent,
                where=fermi_weight != 0,
                out=np.zeros_like(fermi_weight),
            )
            norm_factor += weight

        return np.divide(
            mc_weight, norm_factor, out=np.zeros_like(mc_weight), where=norm_factor != 0
        )


class JacMCLayer(MCLayer):
    IDENTIFIER = "jac"

    def __init__(
        self,
        jac_exponent: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.jac_exponent = jac_exponent
        self.mapping_has_jacobian_method = hasattr(self.mapping, "_jac_from_momentum")

    def _jac_from_momentum(self, momentum: NDArray, discrete: NDArray) -> NDArray:
        if self.mapping_has_jacobian_method:
            return self.mapping._jac_from_momentum(momentum, discrete)
        else:
            return self.mapping._map_from_momentum(momentum, discrete)[0]

    def _mc_weight(self, momentum: NDArray, discrete: NDArray) -> NDArray:
        mc_weight = np.power(
            self._jac_from_momentum(momentum, discrete).reshape(-1),
            self.jac_exponent,
            where=jacobian > 0,
            out=np.zeros_like(jacobian),
        )
        norm_factor = np.zeros_like(mc_weight)
        for ch in range(self.n_channels):
            transform = self.transforms[ch]
            shift = self.channel_shifts[ch]
            mass = self.channel_masses[ch]
            weight = np.power(
                self._jac_from_momentum(
                    momentum, np.full_like(discrete, ch)
                ).reshape(-1),
                self.jac_exponent,
                where=jacobian > 0,
                out=np.zeros_like(jacobian),
            )
            norm_factor += weight

        return np.divide(
            mc_weight, norm_factor, out=np.zeros_like(mc_weight), where=norm_factor != 0
        )
