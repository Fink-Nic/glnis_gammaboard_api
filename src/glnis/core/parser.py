# type: ignore
from glnis.utils.types import GraphProperties, ParserConfig
from symbolica import E, S, Expression
from typing import Dict, List, Set
import pydot
import json
from glnis.core.parameterisation import LayeredParameterisation


class MetaDataParser:
    def __init__(self, config: Dict | None = None, metadata: Dict | None = None, graph_properties: Dict | None = None):
        self.config = ParserConfig(**(config or {}))
        self.metadata = metadata if metadata is not None else {}
        self.graph_properties_dict = graph_properties

    def get_layered_parameterisation_instance(
            self, parameterisation_config: Dict) -> LayeredParameterisation:
        return LayeredParameterisation(self.get_graph_properties(), parameterisation_config)

    def get_graph_properties(self) -> GraphProperties | List[GraphProperties]:
        if self.graph_properties_dict is not None:
            return GraphProperties(**self.graph_properties_dict)

        from madnis_sampler import parse_gammaloop_metadata
        self.metadata = parse_gammaloop_metadata(self.metadata)
        if self.metadata is None:
            raise NotImplementedError(
                "Graph properties must be supplied via 'graph_properties' config key if not using GammaLoop evaluator.")

        import gammaloop
        gammaloop_state = gammaloop.GammaLoopAPI(
            self.metadata.state_folder,
            level=gammaloop.LogLevel.Off,
            logfile_level=gammaloop.LogLevel.Off,
            read_only_state=True)
        outputs = dict()
        for o in gammaloop_state.list_outputs():
            outputs.update(o)
        if len(outputs) == 0:
            raise ValueError(
                f"No processes found in GammaLoop state at '{self.metadata.state_folder}'. Perhaps you forgot to generate it?")

        integrand_name = self.metadata.integrand_name
        if integrand_name not in outputs:
            integrand_name = list(outputs)[0]
        process_id = outputs[integrand_name]
        iinfo = gammaloop_state.get_integrand_info(process_id, integrand_name)
        model_as_str = gammaloop_state.get_model()
        dot_as_str = gammaloop_state.get_dot_files(process_id, integrand_name)
        Dot = DotParser(dot_as_str, model_as_str, dot_from_string=True)
        kinematics = gammaloop_state.get_default_runtime_settings().kinematics
        e_cm = kinematics.e_cm
        ext_momenta = kinematics.externals.data.momenta.to_dict()
        graph_properties_list = []
        for group_id, graph_group in enumerate(iinfo.graph_groups):
            master_id = [g.graph_id for g in graph_group.graphs if g.is_master][0]
            graph_properties = Dot.get_graph_properties(master_id, ext_momenta)
            lmbs = graph_group.loop_momentum_bases
            if self.config.override_lmb_heuristics:
                active_lmbs = lmbs
            else:
                active_lmbs = [lmb for lmb in lmbs if lmb.channel_id is not None]
                active_lmbs = active_lmbs if len(active_lmbs) > 0 else lmbs
            try:
                generation_basis_id = [lmb.matches_generation_basis for lmb in lmbs].index(True)
            except:
                generation_basis_id = 0
            generation_channel_id = active_lmbs.index(lmbs[generation_basis_id])
            # Map edge_ids to {0, ..., n_edges-1}
            gl_internal_edge_ids = set(e_id for lmb in lmbs for e_id in lmb.edge_ids)
            if not len(gl_internal_edge_ids) == graph_properties.n_edges:
                raise ValueError(
                    """Number of internal edges inferred from the dot file does not match the number of internal edges in the GammaLoop state. 
                    This should not happen, please report this issue.""")
            e_id_map: Dict[int, int] = dict()
            for my_e_id, gl_e_id in enumerate(gl_internal_edge_ids):
                e_id_map[gl_e_id] = my_e_id
            graph_properties.lmb_array = [[e_id_map[e_id] for e_id in lmb.edge_ids] for lmb in active_lmbs]

            graph_properties.orientation_ids = [o.orientation_id for o in graph_group.orientations]
            graph_properties.orientation_signatures = [o.signature for o in graph_group.orientations]
            graph_properties.generation_channel_id = generation_channel_id
            graph_properties.e_cm = e_cm
            graph_properties.__post_init__()

            graph_properties_list.append(graph_properties)

        return graph_properties_list if len(graph_properties_list) > 1 else graph_properties_list[0]


class ModelParser:
    def __init__(self, model_file: str, from_string=True):
        if from_string:
            self.model_path = "GammaLoop model loaded from string"
            self.model = json.loads(model_file)
        else:
            self.model_path = model_file
            with open(self.model_path, "r") as f:
                self.model = json.load(f)

    def get_particle_from_identifier(self, identifier_name: str, value) -> Dict:
        particle_match = None
        for particle in self.model['particles']:
            try:
                if particle[identifier_name] == value:
                    particle_match = particle
            except:
                pass

        if particle_match is None:
            raise KeyError(
                f"Particle with {identifier_name}='{value}' does not exist in model '{self.model_path}'.")

        return particle_match

    def get_particle_parameter_from_identifier(self,
                                               identifier_name: str,
                                               identifier_value,
                                               parameter_name: str):
        particle_match = self.get_particle_from_identifier(
            identifier_name, identifier_value)
        try:
            model_parameter_name = particle_match[parameter_name]
        except KeyError:
            raise KeyError(
                f"Particle with '{identifier_name}' = '{identifier_value}' "
                + f"does not have parameter '{parameter_name}'.")

        if model_parameter_name == 'ZERO':
            return [0., 0.]

        parameter_match = None
        for parameter in self.model['parameters']:
            try:
                if parameter['name'] == model_parameter_name:
                    parameter_match = parameter['value']
            except:
                pass

        if parameter_match is None:
            raise KeyError(f"The model '{self.model_path}' does not specify a value for "
                           + f"the parameter '{model_parameter_name}'.")

        return parameter_match

    def get_particle_parameter_from_name(self, particle_name: str, parameter_name: str):
        return self.get_particle_parameter_from_identifier('name', particle_name, parameter_name)

    def get_particle_mass_from_name(self, particle_name: str):
        return self.get_particle_parameter_from_identifier('name', particle_name, 'mass')


class DotParser:
    def __init__(self, dot_file: str, model_file: str,
                 dot_from_string: bool = False, model_from_string: bool = True):
        if dot_from_string:
            self.graph_file = pydot.graph_from_dot_data(dot_file)
        else:
            self.graph_file = pydot.graph_from_dot_file(str(dot_file))
        self.Model = ModelParser(model_file, from_string=model_from_string)

    def get_dot_graph(self, graph_id: int):
        return self.graph_file[graph_id]

    def infer_dependent_momentum(self,
                                 ext_momenta: List[List[float]],
                                 ext_sigs: List[int],
                                 dependent_momentum_index: int) -> List[List[float]]:
        # Infering the dependent momentum from momentum conservation
        if len(ext_momenta) == 0:
            return []
        if not len(ext_momenta) == len(ext_sigs):
            raise ValueError("Length of external momenta and external signatures must match.")
        dmi = dependent_momentum_index
        dm_sig = ext_sigs[dmi]
        dependent_momentum = 4*[0.]
        for momentum, sig in zip(ext_momenta, ext_sigs):
            if momentum == 'dependent':
                continue
            dependent_momentum[0] -= dm_sig*sig*momentum[0]
            dependent_momentum[1] -= dm_sig*sig*momentum[1]
            dependent_momentum[2] -= dm_sig*sig*momentum[2]
            dependent_momentum[3] -= dm_sig*sig*momentum[3]
        ext_momenta[dmi] = dependent_momentum

        return ext_momenta

    def get_external_signature(self, graph_id: int = 0) -> List[int]:
        graph = self.get_dot_graph(graph_id)
        edges = graph.get_edges()

        ext_sigs = len(edges)*[None]
        for edge in edges:
            src_split = edge.get_source().split(':')
            dst_split = edge.get_destination().split(':')
            if len(src_split) == 1:
                ext_sigs[int(dst_split[1])] = 1
            elif len(dst_split) == 1:
                ext_sigs[int(src_split[1])] = -1
        ext_sigs = [sig for sig in ext_sigs if sig is not None]

        return ext_sigs

    def get_graph_properties(self, graph_id: int,
                             ext_momenta: List[List[float]],) -> GraphProperties:

        # External momenta
        n_ext_mom = len(ext_momenta)

        # Dot graph
        graph = self.get_dot_graph(graph_id)
        edges: List[pydot.Edge] = graph.get_edges()
        vertices = graph.get_nodes()

        VERTICES: list[pydot.Node] = []
        LMB_EDGES: list[pydot.Edge] = []
        EXT_VERTICES: Set[pydot.Node] = set()
        INT_EDGES: list[pydot.Edge] = []

        # Filter out the external vertices
        for vert in vertices:
            if vert.get('int_id') is not None:
                VERTICES.append(vert)

        # Add vertex ID for momtrop
        for v_id, vert in enumerate(VERTICES):
            vert.set('v_id', v_id)

        # Filter edges and add additional attributes for momtrop
        EXT_SIGNATURES = []
        for edge in edges:
            src_split = edge.get_source().split(':')
            dst_split = edge.get_destination().split(':')
            edge.set('src', src_split[0])
            edge.set('dst', dst_split[0])

            if edge.get('lmb_id') is not None:
                LMB_EDGES.append(edge)

            if len(src_split) == 1:
                # Incoming external momentum
                EXT_VERTICES.add(graph.get_node(edge.get("dst"))[0])
                EXT_SIGNATURES.append(1)
            elif len(dst_split) == 1:
                # Outgoing external momentum
                EXT_VERTICES.add(graph.get_node(edge.get("src"))[0])
                EXT_SIGNATURES.append(-1)
            else:
                if not "K" in (edge.get('lmb_rep') or ""):
                    continue
                INT_EDGES.append(edge)
                particle_name = edge.get('particle')[1:-1]
                edge.set('mass', self.Model.get_particle_mass_from_name(
                    particle_name)[0])
                src_vert = graph.get_node(edge.get('src'))[0]
                dst_vert = graph.get_node(edge.get('dst'))[0]
                edge.set('src_id', src_vert.get('v_id'))
                edge.set('dst_id', dst_vert.get('v_id'))

        # Reconstruct the dependent external momentum from momentum conservation
        if 'dependent' in ext_momenta:
            dmi = ext_momenta.index('dependent')
            ext_momenta = self.infer_dependent_momentum(
                ext_momenta, EXT_SIGNATURES, dependent_momentum_index=dmi)

        # Symbolica setup for LMB representation parsing
        # P: External momenta
        # K: Internal momenta
        # x_, a_: wildcards
        P, K = S('P', 'K')
        x_, a_ = S('x_', 'a_')

        # Set up momtrop sampler
        n_loops = len(LMB_EDGES)

        graph_externals = sorted([v.get("v_id") for v in EXT_VERTICES])
        graph_signature = []
        edge_momentum_shifts = []
        edge_src_dst_vertices = []
        edge_masses = []
        edge_external_sigs = []

        for edge in INT_EDGES:
            # Generate the momtrop edge
            src_id = edge.get('src_id')
            dst_id = edge.get('dst_id')
            mass = edge.get('mass')
            edge_src_dst_vertices.append([src_id, dst_id])
            edge_masses.append(mass)

            # LMB representation parsing
            e: Expression = E(edge.get('lmb_rep')[1:-1])
            e = e.replace(P(x_, a_), P(x_))
            e = e.replace(K(x_, a_), K(x_))
            lmb_sig = [int(e.coefficient(K(lmb_id)).to_sympy())
                       for lmb_id in range(n_loops)]
            graph_signature.append(lmb_sig)

            edge_external_sig = [float(e.coefficient(P(ext_id)).to_sympy())
                                 for ext_id in range(n_ext_mom)]
            momentum_shift = [0. for _ in range(3)]
            for coeff, ext_mom in zip(edge_external_sig, ext_momenta):
                for i in range(3):
                    momentum_shift[i] += coeff*ext_mom[i+1]

            edge_momentum_shifts.append(momentum_shift)
            edge_external_sigs.append(edge_external_sig)

        return GraphProperties(
            edge_src_dst_vertices=edge_src_dst_vertices,
            edge_masses=edge_masses,
            edge_momentum_shifts=edge_momentum_shifts,
            graph_external_vertices=graph_externals,
            graph_signature=graph_signature,
            edge_external_sigs=edge_external_sigs,
            external_momenta=ext_momenta,
        )
