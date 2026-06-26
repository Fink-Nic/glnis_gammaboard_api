# type: ignore
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from numpy.typing import NDArray
from copy import deepcopy

import momtrop
from glnis.utils.types import LayerData, GraphProperties, ParameterisationLayerConfig


type ParamOutput = Tuple[NDArray, NDArray, NDArray | None]


class Parameterisation(ABC):
    """
    Abstract class for a parameterisation layer in the parameterisation chain. Handles the structure
    of the chain and the passing of data along the chain, while the actual parameterisation step is
    implemented in the _layer_parameterise method of the child classes.
    """
    N_SPATIAL_DIMS = 3
    IDENTIFIER = "ABCParameterisation"

    def __init__(self,
                 graph_properties: GraphProperties,
                 next_param: 'Parameterisation' = None,
                 is_first_layer: bool = False,
                 **uncaught_kwargs):
        self.graph_properties = graph_properties
        self.next_param = next_param
        self.is_first_layer = is_first_layer

        self.layer_continuous_dim_in = self._layer_continuous_dim_in()
        self.layer_continuous_dim_out = self._layer_continuous_dim_out()
        self.layer_discrete_dims = self._layer_discrete_dims()
        self.layer_num_discrete_dims = len(self.layer_discrete_dims)

        self.chain_continuous_dim_in = self.get_chain_continuous_dim()
        self.chain_discrete_dims = self.get_chain_discrete_dims()

    @abstractmethod
    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray,
                            ) -> ParamOutput:
        """
        Args:
            continuous: continuous parameters
            discrete: discrete parameters
        Returns:
            jacobians, parameterised continuous output
        """
        pass

    def parameterise(self, layer_input: LayerData) -> LayerData:
        """
        Args:
            LayerData: Output of the previous Parameterisation layer (or sampler)
        Returns:
            LayerData: Parameterisation and data to be passed along the chain
        Raises:
            ValueError
        """
        param_input = self._to_layer_input(layer_input)
        param_output = self._layer_parameterise(*param_input)
        layer_output = self._to_layer_output(layer_input, *param_output)

        if self.next_param is None:
            return layer_output

        return self.next_param.parameterise(layer_output)

    def discrete_prior_prob_function(self, indices: NDArray, dim: int = 0) -> NDArray:
        """
        Args:
            indices: indices of the discrete channel
            dim: current index on dim=1 of generated indices
        Returns:
            NDArray of shape (indices.shape[0], self.layer_num_discrete_dims):
            probability of the prior distribution for given indices.
            Default is flat probability distribution,
            zero if indices.shape[0] = self.layer_num_discrete_dims
        """
        if dim < self.layer_num_discrete_dims or self.next_param is None:
            return self._layer_prior_prob_function(indices)

        indices = indices[:, self.layer_num_discrete_dims:]
        dim -= self.layer_num_discrete_dims

        return self.next_param.discrete_prior_prob_function(indices, dim)

    def get_chain_discrete_dims(self) -> List[int]:
        """
        Returns:
            List of the discrete dimensions of this and following layers in the
            parameterisation chain.
            Intended to be used to set the discrete dimensions for an Integrator.
        """
        if self.next_param is None:
            return self.layer_discrete_dims
        return self.layer_discrete_dims + self.next_param.get_chain_discrete_dims()

    def get_chain_continuous_dim(self) -> int:
        """
        Returns:
            The continuous dimension of this and following layers in the parameterisation chain.
            Intended to be used to set the number of continuous dimensions for an Integrator.
        """
        if self.next_param is None:
            return self.layer_continuous_dim_in

        layer_dim = self.layer_continuous_dim_in - self.layer_continuous_dim_out

        return layer_dim + self.next_param.get_chain_continuous_dim()

    def _to_layer_input(self, input: LayerData
                        ) -> Tuple[NDArray, NDArray]:
        """
        Returns the part of the input data that is relevant to the current layer, respecting
        the structure of the chain.

        Args:
            input: LayerData of the previous layer
        Returns:
            Continuous Samples: NDArray of shape(n_samples, self.continuous_dim)
            Discrete Samples: NDArray of shape(n_samples, <=len(self.discrete_dims))
        Raises:
            ValueError
        """
        continuous = input.continuous
        discrete = input.discrete
        if (n_dim := continuous.shape[1]) < self.chain_continuous_dim_in:
            raise ValueError(
                f"Layer {self.IDENTIFIER} has received {n_dim}-dimensional continuous input, "
                + f"expected at least {self.chain_continuous_dim_in}.")
        n_disc = len(self.chain_discrete_dims)
        if (n_dim := discrete.shape[1]) < n_disc:
            raise ValueError(
                f"Layer {self.IDENTIFIER} has received {n_dim}-dimensional discrete input, "
                + f"expected at least {n_disc}.")

        continuous = continuous[:, :self.layer_continuous_dim_in]

        if discrete.shape[1] > self.layer_num_discrete_dims:
            discrete = discrete[:, :self.layer_num_discrete_dims]

        return continuous, discrete

    def _to_layer_output(self, layer_input: LayerData,
                         jac_param: NDArray,
                         cont_param: NDArray,
                         disc_param: NDArray | None,) -> LayerData:
        """
        Returns the output of the current layer in a form that respects the structure of the chain,
        in order to be passed down the chain.

        Args:
            layer_input: LayerData of the previous layer
            *param_output: Output generated by the parameterisation step of the current layer
        Returns:
            LayerData
        """
        cont_param = cont_param.reshape(len(layer_input.continuous), -1)
        if disc_param is None:
            disc_param = np.zeros(
                (layer_input.n_points, 0), dtype=layer_input.dtype)
        # Pass along potential additional input that is required down the chain
        n_cont = layer_input._active_structure[layer_input.POSITIONS['continuous']]
        n_disc = layer_input._active_structure[layer_input.POSITIONS['discrete']]

        if n_cont > self.layer_continuous_dim_in:
            cont_pass = layer_input.continuous[:, self.layer_continuous_dim_in:]
            cont_param = np.hstack([cont_param, cont_pass])
        if n_disc > self.layer_num_discrete_dims:
            disc_pass = layer_input.discrete[:, self.layer_num_discrete_dims:]
            disc_param = np.hstack([disc_param, disc_pass])

        # Update the data
        layer_input.jac *= jac_param.reshape(-1, 1)
        layer_input.continuous = cont_param
        layer_input.discrete = disc_param
        layer_input.update(self.IDENTIFIER)

        return layer_input

    def _layer_prior_prob_function(self, indices: NDArray) -> NDArray:
        num_disc_input = indices.shape[1]
        if num_disc_input == self.layer_num_discrete_dims:
            return np.zeros_like(indices, dtype=np.float64)

        disc_dim = self.layer_discrete_dims[num_disc_input]
        return np.ones((len(indices), disc_dim), dtype=np.float64) / disc_dim

    def _to_generation_lmb(self, momenta: NDArray, discrete: NDArray, inverse: bool = False) -> NDArray:
        """
        Transforms the loop momenta to the edge momentum basis of the graph, using the
        discrete channel information to determine the correct transformation.

        Args:
            momenta: shape (n_samples, n_loops*3)
            discrete: shape (n_samples, num_disc_dims)
        Returns:
            shape (n_samples, n_loops*3)
        """
        edges = self.graph_properties.lmb_array[discrete]
        shifts = np.array(self.graph_properties.edge_momentum_shifts)
        sample_shifts = shifts[edges].reshape(-1, 3*self.graph_properties.n_loops)
        if not inverse:
            momenta -= sample_shifts

        if inverse:
            backward = self.graph_properties.channel_inv_transforms[
                self.graph_properties.generation_channel_id]
            transform = backward @ self.graph_properties.channel_transforms
        else:
            forward = self.graph_properties.channel_transforms[
                self.graph_properties.generation_channel_id]
            transform = forward @ self.graph_properties.channel_inv_transforms
        sample_transform = transform[discrete.ravel()]
        result = sample_transform @ momenta.reshape(-1, self.graph_properties.n_loops, 3)
        result = result.reshape(-1, self.graph_properties.n_loops*3)

        if inverse:
            result += sample_shifts

        return result

    def _layer_discrete_dims(self) -> List[int]:
        """
        Returns:
            List of shape of the discrete dimensions of this layer in the parameterisation chain.
            Intended to be used to initialize the value for self.discrete_dims
        """
        return []

    def _layer_continuous_dim_in(self) -> int:
        """
        Intended to be used to initialize the value for self.layer_continuous_dim_in.

        Returns:
            The continuous dimension of the input of this layer in the parameterisation chain.
        """
        return self.N_SPATIAL_DIMS*self.graph_properties.n_loops

    def _layer_continuous_dim_out(self) -> int:
        """
        Intended to be used to initialize the value for self.layer_continuous_dim_out.

        Returns:
            The continuous dimension of the output of this layer in the parameterisation chain.
        """
        return self.N_SPATIAL_DIMS*self.graph_properties.n_loops


class LayeredParameterisation:
    """
    Factory class for a layered parameterisation. Takes a list of parameterisation settings.
    """
    IDENTIFIER = "layered parameterisation"

    def __init__(self, graph_properties: GraphProperties | List[GraphProperties],
                 param_configs: Dict[str, Any],):
        layer_configs = param_configs.get("layer", [])
        if not isinstance(layer_configs, list):
            layer_configs = [layer_configs]
        if isinstance(graph_properties, list):
            graph_properties = graph_properties[0]
        param_layers: List[Parameterisation] = []
        self.condition_integrand_first = param_configs.get("condition_integrand_first", False)
        self.use_f128 = param_configs.get("use_f128", False)
        for i_layer, config in enumerate([ParameterisationLayerConfig.from_dict(config) for config in layer_configs]):
            is_first_layer = (i_layer + 1) == len(layer_configs)
            next_param = None if i_layer == 0 else param_layers[-1]
            config.param_kwargs.update(dict(is_first_layer=is_first_layer,
                                            next_param=next_param,
                                            graph_properties=graph_properties,))

            match config.param_type.lower():
                case "momtrop":
                    p = MomtropParameterisation(**config.param_kwargs)
                case "momtrop_edge_weights":
                    p = MomtropEdgeWeightsParameterisation(**config.param_kwargs)
                case "spherical":
                    p = SphericalParameterisation(**config.param_kwargs)
                case "inv_spherical":
                    p = InverseSphericalParameterisation(**config.param_kwargs)
                case "kaapo":
                    p = KaapoParameterisation(**config.param_kwargs)
                case "rkaapo":
                    p = RKaapoParameterisation(**config.param_kwargs)
                case "s":
                    p = SParameterisation(**config.param_kwargs)
                case "identity":
                    p = IdentityParameterisation(**config.param_kwargs)
                case "epem_ttxh_lo":
                    p = EpEmTTxH_LO(**config.param_kwargs)
                case "mc_layer":
                    if next_param is None:
                        raise ValueError(
                            "MC layer must be passed after a parameterisation.")
                    config.param_kwargs.update(dict(is_first_layer=next_param.is_first_layer,
                                                    next_param=next_param.next_param,
                                                    param=next_param,))
                    match config.param_kwargs.pop('subtype', 'ose').lower():
                        case "ose":
                            p = OSEMCLayer(**config.param_kwargs)
                        case "fermi":
                            p = FermiMCLayer(**config.param_kwargs)
                case _:
                    raise NotImplementedError(
                        f"Parameterisation {config.param_type} has not been implemented.")
            param_layers.append(p)

        if not param_layers:
            param_layers.append(
                IdentityParameterisation(
                    graph_properties=graph_properties,
                    is_first_layer=True,)
            )

        self.param: Parameterisation = param_layers[-1]
        self.continuous_dims = self.param.chain_continuous_dim_in
        self.discrete_dims = self.param.chain_discrete_dims
        self.num_layers = len(param_layers)

    def parameterise(self, discrete: NDArray | None, continuous: NDArray, wgt: NDArray, ) -> Tuple[NDArray, NDArray, NDArray]:
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

        pass_disc_to_integrand = (
            discrete[..., :-len(self.discrete_dims)] if self.condition_integrand_first
            else discrete[..., len(self.discrete_dims):]
        )
        layer_input = LayerData(n_points=continuous.shape[0],
                                n_cont=continuous.shape[1],
                                n_disc=discrete.shape[1])
        layer_input.continuous = continuous
        layer_input.discrete = discrete
        layer_input.update("sampled input")

        layer_output = self.param.parameterise(layer_input)
        return pass_disc_to_integrand, layer_output.continuous, wgt * layer_output.jac.flatten()

    def discrete_prior_prob_function(self, indices: NDArray, dim: int = 0) -> NDArray:
        return self.param.discrete_prior_prob_function(indices, dim)


class MomtropParameterisation(Parameterisation):
    """
    Wrapper for the momtrop sampler (arxiv.org/abs/2504.09613) rust implementation.
    """
    IDENTIFIER = "momtrop param"

    def __init__(self, edge_weight: float | List[float] | None = None,
                 sample_discrete: bool = True,
                 mask_redundant: bool = True,
                 **kwargs: Dict[str, Any]):
        """
        Args:
            overwrite_edge_weight (float | List[float] | bool): Sets the propagator weights of the Feynman measure to sample from
            sample_discrete (bool): Enable to expose the edge indices as a discrete input
            mask_redundant (bool): If sample_discrete is enabled, will not expose the last (n_edges-1) continuous inputs
        """
        self.edge_weight = edge_weight
        self.sample_discrete = sample_discrete
        self.mask_redundant = mask_redundant and sample_discrete
        self.graph_properties: GraphProperties = kwargs["graph_properties"]
        match self.edge_weight:
            case int() | float():
                self.edge_weight = self.graph_properties.n_edges * [float(self.edge_weight)]
            case [_, *_]:
                if not len(self.edge_weight) == self.graph_properties.n_edges:
                    raise ValueError(
                        "If provided as a sequence, the number of momtrop edge weights must match the number of propagators.")
            case _:
                default_weight = (3*self.graph_properties.n_loops + 3/2)/self.graph_properties.n_edges/2
                edge_weight = self.graph_properties.n_edges*[default_weight]

        mt_edges = [
            momtrop.Edge(tuple(src_dst), ismassive, weight) for src_dst, ismassive, weight
            in zip(self.graph_properties.edge_src_dst_vertices,
                   self.graph_properties.edge_ismassive,
                   self.edge_weight)
        ]
        assym_graph = momtrop.Graph(
            mt_edges, self.graph_properties.graph_external_vertices)
        momentum_shifts = [momtrop.Vector(*shift) for shift
                           in self.graph_properties.edge_momentum_shifts]
        self.momtrop_edge_data = momtrop.EdgeData(
            self.graph_properties.edge_masses, momentum_shifts)
        self.momtrop_sampler = momtrop.Sampler(
            assym_graph, self.graph_properties.graph_signature)
        self.momtrop_sampler_settings = momtrop.Settings(False, False)
        super().__init__(**kwargs)

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray,
                            ) -> ParamOutput:
        if self.mask_redundant:
            continuous = np.hstack([
                continuous,
                np.zeros((continuous.shape[0], self.graph_properties.n_edges - 1), dtype=continuous.dtype)
            ])
        if discrete.size == 0:
            samples = self.momtrop_sampler.sample_batch(
                continuous, self.momtrop_edge_data, self.momtrop_sampler_settings, None, )
        else:
            samples = self.momtrop_sampler.sample_batch(
                continuous, self.momtrop_edge_data, self.momtrop_sampler_settings,
                self._get_graph_from_edges_removed(discrete), )

        jac = np.array(samples.jacobians, dtype=continuous.dtype).reshape(-1, 1)
        momentum = np.array(
            samples.loop_momenta, dtype=continuous.dtype).reshape(len(continuous), -1)

        return jac, momentum, None

    def _get_graph_from_edges_removed(self, edges_removed: NDArray | None = None
                                      ) -> List[List[int]]:
        """
        Args:
            edges_removed: List of the edge indices that have already been removed from the graph
        Returns:
            List of shape (n_edges,) that appends the as-yet unforced edges to edges_removed
        """
        n_edges = self.graph_properties.n_edges
        n_points, k = edges_removed.shape
        full_graph = np.arange(n_edges)
        if edges_removed is None:
            return [full_graph.tolist()]
        if k > n_edges:
            raise ValueError(f"Too many edges removed: {k} > {n_edges}")

        edges_removed = edges_removed.astype(np.uint64)

        if k == 0:
            return np.tile(full_graph, (n_points, 1))

        result = np.empty((n_points, n_edges), dtype=np.uint64)
        result[:, :k] = edges_removed
        # Check if edges_removed contains duplicates, replace with arange, prior will be zero anyways
        duplicate_mask = np.any(np.diff(np.sort(edges_removed, axis=1), axis=1).reshape(n_points, -1) == 0, axis=1)
        if np.any(duplicate_mask):
            edges_removed[duplicate_mask] = np.arange(k)

        # If only one edge is left, we can directly return the result without masking (which is more expensive)
        if k == n_edges - 1:
            valid_sum = n_edges * (n_edges - 1) / 2
            result[:, -1] = valid_sum - np.sum(edges_removed, axis=1)
            return result

        # mask[i, j] == True ⇔ edge j is still available for sample i
        mask = np.ones((n_points, n_edges), dtype=bool)
        mask[np.arange(n_points).reshape(-1, 1), edges_removed] = False

        # shape: (n, n_edges - k)
        remaining = np.nonzero(mask)[1].reshape(n_points, -1)
        result[:, k:] = remaining

        return result

    def _layer_prior_prob_function(self, indices: NDArray) -> NDArray:
        return np.array(self.momtrop_sampler.predict_discrete_probs(indices.tolist()))

    def _layer_continuous_dim_in(self) -> int:
        if self.mask_redundant:
            return self.momtrop_sampler.get_dimension() - self.graph_properties.n_edges + 1
        return self.momtrop_sampler.get_dimension()

    def _layer_discrete_dims(self) -> List[int]:
        if not self.sample_discrete:
            return []
        n_edges = self.graph_properties.n_edges
        if self.mask_redundant:
            return (n_edges - 1) * [n_edges]
        return n_edges * [n_edges]


class SphericalParameterisation(Parameterisation):
    IDENTIFIER = "spherical param"

    def __init__(self,
                 conformal_scale: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.conformal_scale = conformal_scale
        if self.graph_properties.e_cm > 0.0:
            self.conformal_scale *= self.graph_properties.e_cm
        self.n_loops = self.graph_properties.n_loops

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray,
                            ) -> ParamOutput:
        momentum = np.zeros_like(continuous)
        n_points = continuous.shape[0]
        if discrete.size == 0:
            discrete = np.empty((continuous.shape[0], 1), dtype=np.uint64)
            discrete.fill(self.graph_properties.generation_channel_id)

        # Constant part of the jacobian
        jac = np.ones((n_points, 1), dtype=continuous.dtype)
        jac *= (4*np.pi * self.conformal_scale**3)**self.n_loops

        for i_loop in range(self.n_loops):
            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)

            xs = continuous[:, _start: _end]
            x, y, z = np.hsplit(xs, [1, 2])

            r = x/(1-x)
            cos_az = (2*y-1)
            sin_az = np.sqrt(1 - cos_az**2)
            pol = 2*np.pi*z

            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)
            ks = self.conformal_scale*r * \
                np.hstack(
                    [sin_az * np.cos(pol), sin_az * np.sin(pol), cos_az])
            momentum[:, _start: _end] = ks
            # Calculate the jacobian determinant
            jac *= x**2 / (1 - x)**4

        # Transform the loop momenta back to the LMB of the graph
        momentum = self._to_generation_lmb(momentum, discrete)

        return jac, momentum.reshape(n_points, -1), None


class EpEmTTxH_LO(Parameterisation):
    IDENTIFIER = "epem_ttxh_lo param"

    def __init__(self,
                 return_full_phase_space: bool = False,
                 **kwargs):
        import madspace as ms
        self.ms_diagram = ms.Diagram(
            incoming_masses=[0.0, 0.0],
            outgoing_masses=[173.0, 173.0, 125.0],
            propagators=[ms.Propagator(*prop) for prop in [[0.0, 0.0], [173.0, 0.0]]],
            vertices=[["i0", "i1", "p0"], ["p0", "o0", "p1"], ["p1", "o1", "o2"]],
        )
        self.mapping = ms.PhaseSpaceMapping(
            ms.Topology(self.ms_diagram), kwargs["graph_properties"].e_cm,
            permutations=[[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]]
        )
        self.return_full_phase_space = return_full_phase_space
        self.return_full_phase_space = False
        super().__init__(**kwargs)

    def _layer_continuous_dim_in(self) -> int:
        return self.mapping.random_dim()

    def _layer_continuous_dim_out(self) -> int:
        if self.return_full_phase_space:
            return self.mapping.particle_count()*self.N_SPATIAL_DIMS
        return (len(self.ms_diagram.outgoing_masses)-1)*self.N_SPATIAL_DIMS

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray,
                            ) -> ParamOutput:
        momenta_permutations = np.array([
            [4, 3],
            [2, 4],
        ])
        permut = np.array(momenta_permutations[discrete], dtype=np.int32)

        result = self.mapping.map_forward([continuous], [discrete.flatten().astype(np.int32)])
        jac = result.det.reshape(-1, 1)
        if self.return_full_phase_space:
            momenta = result.momenta[:, :, 1:].reshape(continuous.shape[0], -1)
        else:
            momenta = np.zeros((continuous.shape[0], len(self.ms_diagram.outgoing_masses)-1, 3), dtype=continuous.dtype)
            momenta[discrete.flatten() == 0] = result.momenta[discrete.flatten() == 0, :, 1:][..., [4, 3], :]
            momenta[discrete.flatten() == 1] = result.momenta[discrete.flatten() == 1, :, 1:][..., [2, 4], :]
            momenta[:, 0, :] = momenta[:, 0, :] - momenta[:, 1, :]

        momenta = momenta.reshape(continuous.shape[0], -1)

        # print(momenta)
        return jac, momenta, None


# class EpEmTTxH_LO(Parameterisation):
#     IDENTIFIER = "epem_ttxh_lo param"

#     def __init__(self,
#                  return_full_phase_space: bool = False,
#                  **kwargs):
#         import madspace as ms
#         self.ms_diagram = ms.Diagram(
#             incoming_masses=[0.0, 0.0],
#             outgoing_masses=[173.0, 173.0, 125.0],
#             propagators=[ms.Propagator(*prop) for prop in [[0.0, 0.0], [173.0, 0.0]]],
#             vertices=[["i0", "i1", "p0"], ["p0", "o0", "p1"], ["p1", "o1", "o2"]],
#         )
#         self.mapping = ms.PhaseSpaceMapping(
#             ms.Topology(self.ms_diagram), kwargs["graph_properties"].e_cm,
#             permutations=[[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]]
#         )
#         self.return_full_phase_space = return_full_phase_space
#         self.return_full_phase_space = False
#         super().__init__(**kwargs)

#     def _layer_continuous_dim_in(self) -> int:
#         return self.mapping.random_dim() + 1

#     def _layer_continuous_dim_out(self) -> int:
#         if self.return_full_phase_space:
#             return self.mapping.particle_count()*self.N_SPATIAL_DIMS
#         return (len(self.ms_diagram.outgoing_masses)-1)*self.N_SPATIAL_DIMS

#     def _layer_parameterise(self, continuous: NDArray, discrete: NDArray,
#                             ) -> ParamOutput:
#         momenta_permutations = np.array([
#             [4, 3],
#             [2, 4],
#         ])
#         permut = np.array(momenta_permutations[discrete], dtype=np.int32)

#         result = self.mapping.map_forward([continuous[:, 1:]], [discrete.flatten().astype(np.int32)])
#         jac_ms = result.det.reshape(-1, 1)
#         print(jac_ms.device)
#         if self.return_full_phase_space:
#             momenta = result.momenta[:, :, 1:].reshape(continuous.shape[0], -1)
#         else:
#             momenta = np.zeros((continuous.shape[0], len(self.ms_diagram.outgoing_masses)-1, 3), dtype=continuous.dtype)
#             momenta[discrete.flatten() == 0] = result.momenta[discrete.flatten() == 0, :, 1:][..., [4, 3], :]
#             momenta[discrete.flatten() == 1] = result.momenta[discrete.flatten() == 1, :, 1:][..., [2, 4], :]
#             momenta[:, 0, :] = momenta[:, 0, :] - momenta[:, 1, :]

#         x = continuous[:, 0].reshape(-1, 1)
#         momenta = momenta.reshape(continuous.shape[0], -1)
#         momenta *= x/(1-x)
#         jac = jac_ms * (1-x)**(2*momenta.shape[1])

#         # print(momenta)
#         return jac, momenta, None


class InverseSphericalParameterisation(Parameterisation):
    IDENTIFIER = "inverse spherical param"

    def __init__(self,
                 conformal_scale: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.conformal_scale = conformal_scale
        if self.graph_properties.e_cm > 0.0:
            self.conformal_scale *= self.graph_properties.e_cm
        self.n_loops = self.graph_properties.n_loops

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray,
                            ) -> Tuple[NDArray, NDArray]:
        if discrete.size == 0:
            discrete = np.empty((continuous.shape[0], 1), dtype=np.uint64)
            discrete.fill(self.graph_properties.generation_channel_id)
        continuous = self._to_generation_lmb(continuous, discrete, inverse=True)
        xs = np.zeros_like(continuous)

        # Constant part of the jacobian
        jac = np.ones((len(continuous), 1), dtype=continuous.dtype)
        jac /= (4*np.pi * self.conformal_scale**3)**self.n_loops

        for i_loop in range(self.n_loops):
            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)
            ks = continuous[:, _start: _end]

            k0, k1, k2 = np.hsplit(ks, [1, 2])

            r = np.linalg.norm(ks, axis=1).reshape(-1, 1)
            cos_az = k2 / r
            tan_pol = (k1 / k0).reshape(-1, 1)
            pol: NDArray = np.arctan(tan_pol)
            # Accounting for missing quadrants of arctan
            pol += np.pi*(1 - np.sign(k0))/2 * np.sign(k1)
            pol += np.pi*(1 - np.sign(pol))

            r /= self.conformal_scale
            x = r / (1 + r)
            y = (cos_az + 1) / 2
            z = pol / 2 / np.pi
            xs[:, _start: _end] = np.hstack([x, y, z])

            # Calculate the jacobian determinant
            jac /= x**2 / (1 - x)**4

        return jac, xs, None


class KaapoParameterisation(Parameterisation):
    IDENTIFIER = "kaapo param"

    def __init__(self, mu: List[float] | float = np.pi,
                 a: float = 0.5,
                 b: float = 1.0,
                 vary_a: bool = False,
                 a_min: float = 0.2,
                 angle_shift: float = 0.0,
                 **kwargs):
        self.a = a
        self.b = b if b else self.graph_properties.e_cm
        self.vary_a = vary_a
        self.a_min = a_min
        self.angle_shift = angle_shift

        super().__init__(**kwargs)
        self.mu = mu
        if not type(self.mu) == list:
            self.mu: list[float] = self.graph_properties.n_edges*[self.mu]

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray
                            ) -> ParamOutput:
        if discrete.size == 0:
            discrete = np.zeros((continuous.shape[0], 1), dtype=np.uint64)
        # For easier reading
        n_loops = self.graph_properties.n_loops
        n_points = continuous.shape[0]
        dtype = continuous.dtype
        if self.vary_a:
            a = self.a_min + (1. - self.a_min)*continuous[:, -1].reshape(-1, 1)
        else:
            a = self.a
        b = self.b

        momentum = np.zeros((n_points, 3*n_loops), dtype=dtype)

        # The constant part of the jacobian
        jac = np.ones((n_points, 1), dtype=dtype)
        jac *= (4 * np.pi / a / b**a)**n_loops

        for i_loop in range(n_loops):
            basis_edge = self.graph_properties.lmb_array[discrete, i_loop]
            m_e = np.array(self.graph_properties.edge_masses)[basis_edge]
            mu = np.array(self.mu)[basis_edge]
            p_F = np.clip(
                mu**2 - m_e**2, a_min=0., a_max=None)**0.5

            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)

            xs = continuous[:, _start:_end]
            x1, x2, x3 = np.hsplit(xs, [1, 2])
            if self.angle_shift != 0.:
                x2 = (x2 + self.angle_shift) % 1
                x3 = (x3 + self.angle_shift) % 1

            cos_az = (2*x2-1)
            sin_az = np.sqrt(1 - cos_az**2)
            pol = 2*np.pi*x3

            # Discriminator around the fermi surface and origin
            peak_F: NDArray = b**a * x1 / (1 - x1) - p_F**a

            # Radial component
            h_c = p_F + np.sign(peak_F) * np.abs(peak_F)**(1 / a)

            # Standard spherical parameterisation, scaled by h_c
            k_vec = h_c * np.hstack(
                [sin_az * np.cos(pol), sin_az * np.sin(pol), cos_az])
            momentum[:, _start: _end] = k_vec

            # Calculate the jacobian
            jac *= h_c**2 * np.abs(peak_F)**(1 / a - 1)
            jac *= (np.sign(peak_F)*np.abs(peak_F) + p_F**a + b**a)**2

        # Transform the loop momenta back to the LMB of the graph
        momentum = self._to_generation_lmb(momentum, discrete)

        return jac, momentum.reshape(n_points, -1), None

    def _layer_continuous_dim_in(self) -> int:
        c_dim = self.N_SPATIAL_DIMS*self.graph_properties.n_loops
        if self.vary_a:
            c_dim += 1
        return c_dim


class RKaapoParameterisation(Parameterisation):
    IDENTIFIER = "reduced kaapo param"

    def __init__(self, mu: List[float] | float = np.pi,
                 a: float = 0.5,
                 b: float = 1.0,
                 vary_a: bool = False,
                 a_min: float = 0.2,
                 angle_shift: float = 0.0,
                 **kwargs):
        self.a = a
        self.b = b if b else self.graph_properties.e_cm
        self.vary_a = vary_a
        self.a_min = a_min
        self.angle_shift = angle_shift

        super().__init__(**kwargs)
        self.mu = mu
        if not type(self.mu) == list:
            self.mu: list[float] = self.graph_properties.n_edges*[self.mu]

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray
                            ) -> ParamOutput:
        if discrete.size == 0:
            discrete = np.zeros((continuous.shape[0], 1), dtype=np.uint64)
        # For easier reading
        n_loops = self.graph_properties.n_loops
        n_points = continuous.shape[0]
        dtype = continuous.dtype
        if self.vary_a:
            a = self.a_min + (1. - self.a_min)*continuous[:, -1].reshape(-1, 1)
        else:
            a = self.a
        b = self.b

        momentum = np.zeros((n_points, 3*n_loops), dtype=dtype)

        # The constant part of the jacobian
        jac = np.ones((n_points, 1), dtype=dtype)
        jac *= (4 * np.pi / a / b**a)**n_loops

        for i_loop in range(n_loops):
            basis_edge = self.graph_properties.lmb_array[discrete, i_loop]
            m_e = np.array(self.graph_properties.edge_masses)[basis_edge]
            mu = np.array(self.mu)[basis_edge]
            p_F = np.clip(
                mu**2 - m_e**2, a_min=0., a_max=None)**0.5

            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)

            if i_loop == 0:
                x1 = continuous[:, 0].reshape(-1, 1)
                cos_az = np.ones(
                    (continuous.shape[0], 1), dtype=continuous.dtype)
                sin_az = np.zeros(
                    (continuous.shape[0], 1), dtype=continuous.dtype)
                pol = 0
            elif i_loop == 1:
                x1 = continuous[:, 1].reshape(-1, 1)
                x2 = continuous[:, 2].reshape(-1, 1)
                if self.angle_shift != 0.:
                    x2 = (x2 + self.angle_shift) % 1
                cos_az = (2*x2 - 1)
                sin_az = np.sqrt(1 - cos_az**2)
                pol = 0
            else:
                xs = continuous[:, _start -
                                self.N_SPATIAL_DIMS: _end-self.N_SPATIAL_DIMS]
                x1, x2, x3 = np.hsplit(xs, [1, 2])
                if self.angle_shift != 0.:
                    x2 = (x2 + self.angle_shift) % 1
                    x3 = (x3 + self.angle_shift) % 1

                cos_az = (2*x2-1)
                sin_az = np.sqrt(1 - cos_az**2)
                pol = 2*np.pi*x3

            # Discriminator around the fermi surface and origin
            peak_F: NDArray = b**a * x1 / (1 - x1) - p_F**a

            # Radial component
            h_c = p_F + np.sign(peak_F) * np.abs(peak_F)**(1 / a)

            # Standard spherical parameterisation, scaled by h_c
            k_vec = h_c * np.hstack(
                [sin_az * np.cos(pol), sin_az * np.sin(pol), cos_az])
            momentum[:, _start: _end] = k_vec

            # Calculate the jacobian
            jac *= h_c**2 * np.abs(peak_F)**(1 / a - 1)
            jac *= (np.sign(peak_F)*np.abs(peak_F) + p_F**a + b**a)**2

        # Transform the loop momenta back to the LMB of the graph
        momentum = self._to_generation_lmb(momentum, discrete)

        return jac, momentum.reshape(n_points, -1), None

    def _layer_continuous_dim_in(self) -> int:
        if self.graph_properties.n_loops == 1:
            c_dim = 1
        else:
            c_dim = self.N_SPATIAL_DIMS*(self.graph_properties.n_loops - 1)
        if self.vary_a:
            c_dim += 1
        return c_dim


class SParameterisation(Parameterisation):
    IDENTIFIER = "S param"

    def __init__(self, exponent: float = 2.0,
                 **kwargs):
        """
        Args:
            exponent: Exponent of the S transformation. Higher values lead to stronger smoothing around the edges of the unit cube
        """
        super().__init__(**kwargs)
        self.exponent = max(exponent, 1.0)

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray) -> ParamOutput:
        xn = np.power(continuous, self.exponent)
        denom = xn + np.power(1 - continuous, self.exponent)
        cont = xn / denom

        num = continuous - continuous*continuous
        jac = self.exponent*np.power(num, self.exponent - 1.0, where=num != 0,
                                     out=np.zeros_like(continuous)) / denom / denom
        jac = np.prod(jac, axis=1, keepdims=True)

        return jac, cont, discrete


class IdentityParameterisation(Parameterisation):
    IDENTIFIER = "identity param"

    def __init__(self,
                 seed: int = 42,
                 uniform_continuous: bool = False,
                 force_dim: int | None = None,
                 **kwargs):
        self.seed = seed
        self.uniform_continuous = uniform_continuous
        self.rng = np.random.default_rng(seed)
        self.force_dim = force_dim
        super().__init__(**kwargs)

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray) -> ParamOutput:
        jac = np.ones((continuous.shape[0], 1), dtype=continuous.dtype)
        if self.uniform_continuous:
            return jac, self.rng.random(size=continuous.shape), discrete
        return jac, continuous, discrete

    def _layer_continuous_dim_in(self) -> int:
        if self.force_dim is not None:
            return self.force_dim
        return self.graph_properties.n_loops * self.N_SPATIAL_DIMS


class MCLayer(Parameterisation, ABC):
    IDENTIFIER = "MC layer"

    def __init__(self,
                 param: Parameterisation,
                 **kwargs):
        self.param = param
        self.IDENTIFIER += f" : {self.param.IDENTIFIER}"
        super().__init__(**kwargs)

        self.lmbs = self.graph_properties.lmb_array
        self.n_channels = self.graph_properties.n_channels
        self.n_loops = self.graph_properties.n_loops

        self.shifts = np.array(self.graph_properties.edge_momentum_shifts)
        self.channel_shifts = self.shifts[self.lmbs]
        self.channel_masses = np.array(
            self.graph_properties.edge_masses)[self.lmbs]
        # Transform to the edge momentum basis
        # shape: (n_samples, n_channels, n_loops)
        backward = self.graph_properties.channel_inv_transforms[
            self.graph_properties.generation_channel_id]
        self.transforms = backward @ self.graph_properties.channel_transforms

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray) -> ParamOutput:
        jac, momentum, _ = self.param._layer_parameterise(
            continuous, discrete)
        jac *= self._mc_weight(momentum, discrete).reshape(-1, 1)

        return jac, momentum, None

    @abstractmethod
    def _mc_weight(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        return np.ones((continuous.shape[0], 1), dtype=continuous.dtype)

    def _layer_discrete_dims(self) -> List[int]:
        return [self.graph_properties.n_channels]

    def _layer_continuous_dim_in(self) -> int:
        return self.param._layer_continuous_dim_in()


class OSEMCLayer(MCLayer):
    IDENTIFIER = "OSE MC Layer"

    def __init__(self,
                 ose_exponent: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.ose_exponent = ose_exponent

    def _mc_weight(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        momentum = continuous.reshape(-1, self.n_loops, 3)
        # Need to calculate the e-surface term for all lmbs
        mc_weight = np.prod(  # Multiply for each loop
            np.sum(  # Dot product
                (self.transforms[discrete.ravel()] @ momentum.reshape(-1, self.n_loops, 3)
                    + self.channel_shifts[discrete.ravel()])**2, axis=2
            ) + self.channel_masses[discrete.ravel()]**2, axis=1
        )
        mc_weight = np.power(mc_weight, -self.ose_exponent/2., where=mc_weight != 0, out=np.zeros_like(mc_weight))
        norm_factor = np.zeros_like(mc_weight)
        for ch in range(self.n_channels):
            transform = self.transforms[ch]
            shift = self.channel_shifts[ch]
            mass = self.channel_masses[ch]
            weight = np.prod(  # Multiply for each loop
                np.sum(  # Dot product
                    (transform @ momentum.reshape(-1, self.n_loops, 3) + shift)**2, axis=2
                ) + mass**2, axis=1
            )
            norm_factor += np.power(weight, -self.ose_exponent/2., where=weight != 0, out=np.zeros_like(weight))

        return np.divide(mc_weight, norm_factor, out=np.zeros_like(mc_weight), where=norm_factor != 0)


class FermiMCLayer(MCLayer):
    IDENTIFIER = "Fermi MC Layer"

    def __init__(self,
                 ose_exponent: float = 4.0,
                 fermi_exponent: float = 1.0,
                 set_bosonic_edge_to_one: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.ose_exponent = ose_exponent
        self.fermi_exponent = fermi_exponent
        self.set_bosonic_edge_to_one = set_bosonic_edge_to_one
        if hasattr(self.param, 'mu'):
            self.channel_mu = np.array(self.param.mu)[self.lmbs]
        else:
            self.channel_mu = np.zeros((self.n_channels, self.n_loops))

    def _mc_weight(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        momentum = continuous.reshape(-1, self.n_loops, 3)
        self.param: KaapoParameterisation
        # Need to calculate the fermi surface term for all lmbs
        e_surface_terms = np.sqrt(np.sum(
            (self.transforms[discrete.ravel()] @ momentum.reshape(-1, self.n_loops, 3)
                + self.channel_shifts[discrete.ravel()])**2, axis=2) + self.channel_masses[discrete.ravel()]**2)
        fermi_weights = np.abs(e_surface_terms - self.channel_mu[discrete.ravel()])
        if self.set_bosonic_edge_to_one:
            bosonic_edge_mask = self.channel_mu[discrete.ravel()] == 0.
            fermi_weights[bosonic_edge_mask] = 1.
        mc_weight = np.prod(e_surface_terms, axis=1)
        mc_weight = np.power(mc_weight, -self.ose_exponent, where=mc_weight != 0, out=np.zeros_like(mc_weight))
        fermi_weight = np.prod(fermi_weights, axis=1)
        mc_weight *= np.power(fermi_weight, -self.fermi_exponent,
                              where=fermi_weight != 0, out=np.zeros_like(fermi_weight))

        norm_factor = np.zeros_like(mc_weight)
        for ch in range(self.n_channels):
            transform = self.transforms[ch]
            shift = self.channel_shifts[ch]
            mass = self.channel_masses[ch]
            mu = self.channel_mu[ch]
            e_surface_terms = np.sqrt(np.sum(
                (transform @ momentum.reshape(-1, self.n_loops, 3) + shift)**2, axis=2) + mass**2)
            fermi_surface_terms = np.abs(e_surface_terms - mu)
            if self.set_bosonic_edge_to_one:
                bosonic_edge_mask = mu == 0.
                fermi_surface_terms[:, bosonic_edge_mask] = 1.
            weight = np.prod(e_surface_terms, axis=1)
            weight = np.power(weight, -self.ose_exponent, where=weight != 0, out=np.zeros_like(weight))
            fermi_weight = np.prod(fermi_surface_terms, axis=1)
            weight *= np.power(fermi_weight, -self.fermi_exponent,
                               where=fermi_weight != 0, out=np.zeros_like(fermi_weight))
            norm_factor += weight

        return np.divide(mc_weight, norm_factor, out=np.zeros_like(mc_weight), where=norm_factor != 0)
