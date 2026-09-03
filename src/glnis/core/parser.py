# type: ignore
import json
from typing import Dict, List, Set

import pydot
from symbolica import E, Expression, S

from glnis.core.mappings import LayeredMapping
from glnis.utils.types import GraphProperties, ParserConfig


class MetaDataParser:
    def __init__(
        self,
        config: Dict | None = None,
        metadata: Dict | None = None,
        graph_properties: Dict | None = None,
    ):
        self.config = ParserConfig(**(config or {}))
        self.metadata = metadata if metadata is not None else {}
        self.graph_properties_dict = graph_properties

    def get_layered_mapping_instance(self, mapping_config: Dict) -> LayeredMapping:
        return LayeredMapping(
            self.get_graph_properties(), **mapping_config
        )

    def get_graph_properties(self) -> GraphProperties | List[GraphProperties]:
        if self.graph_properties_dict is not None:
            return GraphProperties(**self.graph_properties_dict)

        from madnis_sampler import parse_gammaloop_metadata

        self.metadata = parse_gammaloop_metadata(self.metadata)
        if self.metadata is None:
            raise NotImplementedError(
                "Graph properties must be supplied via 'graph_properties' config key if not using GammaLoop evaluator."
            )

        from glnis._rust import parse_state_simplified

        integrand_data = json.loads(parse_state_simplified(
            self.metadata.state_folder, self.metadata.process_id, self.metadata.integrand_name))
        Dot = DotParser(integrand_data["dot_file"], integrand_data["model"], dot_from_string=True)

        """
        #[derive(Debug, Serialize)]
        struct SimplifiedStateData {
            state_folder: String,
            process_id: usize,
            integrand_name: String,
            model: String,
            dot_file: String,
            external_momenta: Vec<ExternalMomentumData>,
            e_cm: f64,
            graph_groups: Vec<SimplifiedGraphGroupData>,
        }

        #[derive(Debug, Serialize)]
        struct SimplifiedGraphGroupData {
            group_id: usize,
            master_graph_id: usize,
            loop_momentum_bases_edge_ids: Vec<Vec<usize>>,
            generation_channel_id: usize,
            orientation_ids: Vec<usize>,
            orientation_signatures: Vec<Vec<i8>>,
        }
        """

        e_cm = integrand_data["e_cm"]
        ext_momenta = integrand_data["external_momenta"]
        graph_properties_list = []
        for graph_group in integrand_data["graph_groups"]:
            master_id = graph_group["master_graph_id"]
            graph_properties = Dot.get_graph_properties(master_id, ext_momenta)
            lmbs = graph_group["loop_momentum_bases_edge_ids"]
            # Map edge_ids to {0, ..., n_edges-1}
            gl_internal_edge_ids = set(e_id for lmb in lmbs for e_id in lmb)
            e_id_map: Dict[int, int] = dict()
            for my_e_id, gl_e_id in enumerate(gl_internal_edge_ids):
                e_id_map[gl_e_id] = my_e_id

            # Technically this should indeed never be the case for bridgeless graphs, but should not be an issue
            # even if it does happen (except for momtrop).
            # if not len(gl_internal_edge_ids) == graph_properties.n_edges:
            #     raise ValueError(
            #         """Number of internal edges inferred from the dot file does not match the number of internal edges in the GammaLoop state.
            #         This should not happen, please report this issue."""
            #     )
            graph_properties.lmb_array = [
                [e_id_map[e_id] for e_id in lmb] for lmb in lmbs
            ]
            graph_properties.orientation_ids = graph_group["orientation_ids"]
            graph_properties.orientation_signatures = graph_group["orientation_signatures"]
            graph_properties.generation_channel_id = graph_group["generation_channel_id"]
            graph_properties.e_cm = e_cm
            graph_properties.__post_init__()

            graph_properties_list.append(graph_properties)

        if len(graph_properties_list) > 1:
            return graph_properties_list

        return graph_properties_list[0]


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
        for particle in self.model["particles"]:
            try:
                if particle[identifier_name] == value:
                    particle_match = particle
            except:
                pass

        if particle_match is None:
            raise KeyError(
                f"Particle with {identifier_name}='{value}' does not exist in model '{self.model_path}'."
            )

        return particle_match

    def get_particle_parameter_from_identifier(
        self, identifier_name: str, identifier_value, parameter_name: str
    ):
        particle_match = self.get_particle_from_identifier(
            identifier_name, identifier_value
        )
        try:
            model_parameter_name = particle_match[parameter_name]
        except KeyError:
            raise KeyError(
                f"Particle with '{identifier_name}' = '{identifier_value}' "
                + f"does not have parameter '{parameter_name}'."
            )

        if model_parameter_name == "ZERO":
            return [0.0, 0.0]

        parameter_match = None
        for parameter in self.model["parameters"]:
            try:
                if parameter["name"] == model_parameter_name:
                    parameter_match = parameter["value"]
            except:
                pass

        if parameter_match is None:
            raise KeyError(
                f"The model '{self.model_path}' does not specify a value for "
                + f"the parameter '{model_parameter_name}'."
            )

        return parameter_match

    def get_particle_parameter_from_name(self, particle_name: str, parameter_name: str):
        return self.get_particle_parameter_from_identifier(
            "name", particle_name, parameter_name
        )

    def get_particle_mass_from_name(self, particle_name: str):
        return self.get_particle_parameter_from_identifier(
            "name", particle_name, "mass"
        )


class DotParser:
    def __init__(
        self,
        dot_file: str,
        model_file: str,
        dot_from_string: bool = False,
        model_from_string: bool = True,
    ):
        if dot_from_string:
            self.graph_file = pydot.graph_from_dot_data(dot_file)
        else:
            self.graph_file = pydot.graph_from_dot_file(str(dot_file))
        self.Model = ModelParser(model_file, from_string=model_from_string)

    def get_dot_graph(self, graph_id: int):
        return self.graph_file[graph_id]

    def infer_dependent_momentum(
        self,
        ext_momenta: List[List[float]],
        ext_sigs: List[int],
        dependent_momentum_index: int,
    ) -> List[List[float]]:
        # Infering the dependent momentum from momentum conservation
        if len(ext_momenta) == 0:
            return []
        if not len(ext_momenta) == len(ext_sigs):
            raise ValueError(
                "Length of external momenta and external signatures must match."
            )
        dmi = dependent_momentum_index
        dm_sig = ext_sigs[dmi]
        dependent_momentum = 4 * [0.0]
        for momentum, sig in zip(ext_momenta, ext_sigs):
            if momentum == "dependent":
                continue
            dependent_momentum[0] -= dm_sig * sig * momentum[0]
            dependent_momentum[1] -= dm_sig * sig * momentum[1]
            dependent_momentum[2] -= dm_sig * sig * momentum[2]
            dependent_momentum[3] -= dm_sig * sig * momentum[3]
        ext_momenta[dmi] = dependent_momentum

        return ext_momenta

    def get_external_signature(self, graph_id: int = 0) -> List[int]:
        graph = self.get_dot_graph(graph_id)
        edges = graph.get_edges()

        ext_sigs = len(edges) * [None]
        for edge in edges:
            src_split = edge.get_source().split(":")
            dst_split = edge.get_destination().split(":")
            if len(src_split) == 1:
                ext_sigs[int(dst_split[1])] = 1
            elif len(dst_split) == 1:
                ext_sigs[int(src_split[1])] = -1
        ext_sigs = [sig for sig in ext_sigs if sig is not None]

        return ext_sigs

    def get_graph_properties(
        self,
        graph_id: int,
        ext_momenta: List[List[float]],
    ) -> GraphProperties:

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
            if vert.get("int_id") is not None:
                VERTICES.append(vert)

        # Add vertex ID for momtrop
        for v_id, vert in enumerate(VERTICES):
            vert.set("v_id", v_id)

        # Filter edges and add additional attributes for momtrop
        EXT_SIGNATURES = []
        for edge in edges:
            src_split = edge.get_source().split(":")
            dst_split = edge.get_destination().split(":")
            edge.set("src", src_split[0])
            edge.set("dst", dst_split[0])

            if edge.get("lmb_id") is not None:
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
                if "K" not in (edge.get("lmb_rep") or ""):
                    continue
                INT_EDGES.append(edge)
                particle_name = edge.get("particle")[1:-1]
                edge.set(
                    "mass", self.Model.get_particle_mass_from_name(particle_name)[0]
                )
                src_vert = graph.get_node(edge.get("src"))[0]
                dst_vert = graph.get_node(edge.get("dst"))[0]
                edge.set("src_id", src_vert.get("v_id"))
                edge.set("dst_id", dst_vert.get("v_id"))

        # Reconstruct the dependent external momentum from momentum conservation
        if "dependent" in ext_momenta:
            dmi = ext_momenta.index("dependent")
            ext_momenta = self.infer_dependent_momentum(
                ext_momenta, EXT_SIGNATURES, dependent_momentum_index=dmi
            )

        # Symbolica setup for LMB representation parsing
        # P: External momenta
        # K: Internal momenta
        # x_, a_: wildcards
        P, K = S("P", "K")
        x_, a_ = S("x_", "a_")

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
            src_id = edge.get("src_id")
            dst_id = edge.get("dst_id")
            mass = edge.get("mass")
            edge_src_dst_vertices.append([src_id, dst_id])
            edge_masses.append(mass)

            # LMB representation parsing
            e: Expression = E(edge.get("lmb_rep")[1:-1])
            e = e.replace(P(x_, a_), P(x_))
            e = e.replace(K(x_, a_), K(x_))
            lmb_sig = [
                int(e.coefficient(K(lmb_id)).to_sympy()) for lmb_id in range(n_loops)
            ]
            graph_signature.append(lmb_sig)

            edge_external_sig = [
                float(e.coefficient(P(ext_id)).to_sympy())
                for ext_id in range(n_ext_mom)
            ]
            momentum_shift = [0.0 for _ in range(3)]
            for coeff, ext_mom in zip(edge_external_sig, ext_momenta):
                for i in range(3):
                    momentum_shift[i] += coeff * ext_mom[i + 1]

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
