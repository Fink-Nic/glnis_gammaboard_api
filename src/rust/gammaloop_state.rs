use std::path::{Path, PathBuf};

use gammaloop_api::state::{ProcessRef, State, StateFolderKind};
use gammalooprs::{
    momentum::ExternalMomenta,
    processes::{DotExportSettings, ProcessCollection},
    settings::{RuntimeSettings, runtime::kinematic::Externals},
    utils::serde_utils::{SerdeFileError, SmartSerde},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct StateSummary {
    state_folder: String,
    folder_kind: String,
    process_count: usize,
    processes: Vec<ProcessSummary>,
}

#[derive(Debug, Serialize)]
struct ProcessSummary {
    process_id: usize,
    process_name: String,
    kind: &'static str,
    integrands: Vec<String>,
}

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

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ExternalMomentumData {
    Components(Vec<f64>),
    Dependent(&'static str),
}

fn folder_kind_label(kind: &StateFolderKind) -> &'static str {
    match kind {
        StateFolderKind::Missing => "missing",
        StateFolderKind::Scratch => "scratch",
        StateFolderKind::Unmanifested => "unmanifested",
        StateFolderKind::Saved => "saved",
        StateFolderKind::Invalid(_) => "invalid",
    }
}

fn state_folder_kind(path: &Path) -> PyResult<StateFolderKind> {
    gammaloop_api::state::classify_state_folder(path)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

fn external_momenta_data(externals: &Externals) -> Vec<ExternalMomentumData> {
    match externals {
        Externals::Constant { momenta, .. } => momenta
            .iter()
            .map(|momentum| match momentum {
                ExternalMomenta::Independent(components) => ExternalMomentumData::Components(
                    components
                        .iter()
                        .map(|component| f64::from(*component))
                        .collect(),
                ),
                ExternalMomenta::Dependent(_) => ExternalMomentumData::Dependent("dependent"),
            })
            .collect(),
    }
}

fn load_saved_state(state_folder: &Path) -> PyResult<State> {
    State::load(state_folder.to_path_buf(), None, None)
        .map_err(|err| PyValueError::new_err(format!("failed to load GammaLoop state: {err:?}")))
}

fn load_state_runtime_settings(state_folder: &Path) -> PyResult<RuntimeSettings> {
    match RuntimeSettings::from_file_typed(state_folder.join("default_runtime_settings.toml")) {
        Ok(settings) => Ok(settings),
        Err(SerdeFileError::FileError(_)) => Ok(RuntimeSettings::default()),
        Err(err) => Err(PyValueError::new_err(format!(
            "failed to load GammaLoop state runtime settings: {err:?}"
        ))),
    }
}

fn model_json_string(state: &State) -> PyResult<String> {
    serde_json::to_string(&state.model.to_serializable())
        .map_err(|err| PyValueError::new_err(format!("failed to serialize GammaLoop model: {err}")))
}

fn dot_file_string(state: &State, process_id: usize, integrand_name: &str) -> PyResult<String> {
    let process = state
        .process_list
        .processes
        .get(process_id)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "process id {process_id} does not exist in GammaLoop state"
            ))
        })?;
    let settings = DotExportSettings::default();
    let mut dot_output = String::new();

    match &process.collection {
        ProcessCollection::Amplitudes(amplitudes) => amplitudes
            .get(integrand_name)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "could not find amplitude named '{integrand_name}' in process id {process_id}"
                ))
            })?
            .write_dot_fmt(&mut dot_output, &settings)
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "failed to write DOT for amplitude '{integrand_name}': {err}"
                ))
            })?,
        ProcessCollection::CrossSections(cross_sections) => cross_sections
            .get(integrand_name)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "could not find cross-section named '{integrand_name}' in process id {process_id}"
                ))
            })?
            .write_dot_fmt(&mut dot_output, &settings)
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "failed to write DOT for cross-section '{integrand_name}': {err}"
                ))
            })?,
    }

    Ok(dot_output)
}

/// Classify a GammaLoop state folder without loading the full state.
#[pyfunction]
pub fn classify_state_folder(state_folder: PathBuf) -> PyResult<String> {
    let kind = state_folder_kind(&state_folder)?;
    Ok(folder_kind_label(&kind).to_string())
}

/// Load a GammaLoop state read-only and return a compact JSON summary.
///
/// This is intentionally small for now: it proves that the Rust API can inspect
/// a state directly and gives Python enough structured data to select processes
/// and integrands before deeper graph-property parsing is wired in.
#[pyfunction]
pub fn parse_state_summary_json(state_folder: PathBuf) -> PyResult<String> {
    let kind = state_folder_kind(&state_folder)?;
    if !matches!(kind, StateFolderKind::Saved) {
        return Err(PyValueError::new_err(format!(
            "'{}' is not a saved GammaLoop state folder (classified as {})",
            state_folder.display(),
            folder_kind_label(&kind),
        )));
    }

    let state = load_saved_state(&state_folder)?;

    let processes = state
        .process_list
        .processes
        .iter()
        .enumerate()
        .map(|(process_id, process)| {
            let (kind, integrands) = match &process.collection {
                ProcessCollection::Amplitudes(amplitudes) => (
                    "amplitude",
                    amplitudes.keys().cloned().collect::<Vec<String>>(),
                ),
                ProcessCollection::CrossSections(cross_sections) => (
                    "cross_section",
                    cross_sections.keys().cloned().collect::<Vec<String>>(),
                ),
            };
            ProcessSummary {
                process_id,
                process_name: process.definition.folder_name.clone(),
                kind,
                integrands,
            }
        })
        .collect::<Vec<_>>();

    let summary = StateSummary {
        state_folder: state_folder.display().to_string(),
        folder_kind: folder_kind_label(&kind).to_string(),
        process_count: processes.len(),
        processes,
    };

    serde_json::to_string(&summary)
        .map_err(|err| PyValueError::new_err(format!("failed to serialize state summary: {err}")))
}

/// Load a GammaLoop state read-only and return integrand metadata needed by GLNIS.
///
/// The graph-group fields mirror the metadata extracted by Python's
/// `MetaDataParser`: LMBs are restricted to active multi-channel bases when
/// present, otherwise all LMBs are returned, and `generation_channel_id` is the
/// index of the generation basis within that active LMB list.
#[pyfunction]
pub fn parse_state_simplified(
    state_folder: PathBuf,
    process_id: usize,
    integrand_name: String,
) -> PyResult<String> {
    let kind = state_folder_kind(&state_folder)?;
    if !matches!(kind, StateFolderKind::Saved) {
        return Err(PyValueError::new_err(format!(
            "'{}' is not a saved GammaLoop state folder (classified as {})",
            state_folder.display(),
            folder_kind_label(&kind),
        )));
    }

    let state = load_saved_state(&state_folder)?;
    let runtime_settings = load_state_runtime_settings(&state_folder)?;

    let process_ref = ProcessRef::Id(process_id);
    let integrand_info = state
        .get_integrand_info(Some(&process_ref), Some(&integrand_name))
        .map_err(|err| PyValueError::new_err(format!("failed to get integrand info: {err}")))?;

    let model = model_json_string(&state)?;
    let dot_file = dot_file_string(
        &state,
        integrand_info.process_id,
        &integrand_info.integrand_name,
    )?;

    let graph_groups = integrand_info
        .graph_groups
        .iter()
        .map(|graph_group| {
            let master_graph_id = graph_group
                .graphs
                .iter()
                .find(|graph| graph.is_master)
                .or_else(|| graph_group.graphs.first())
                .map(|graph| graph.graph_id)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "graph group {} does not contain any graphs",
                        graph_group.group_id
                    ))
                })?;

            let mut active_lmb_indices = graph_group
                .loop_momentum_bases
                .iter()
                .enumerate()
                .filter_map(|(basis_index, lmb)| lmb.channel_id.is_some().then_some(basis_index))
                .collect::<Vec<_>>();
            if active_lmb_indices.is_empty() {
                active_lmb_indices = (0..graph_group.loop_momentum_bases.len()).collect();
            }

            let generation_basis_id = graph_group
                .loop_momentum_bases
                .iter()
                .position(|lmb| lmb.matches_generation_basis)
                .unwrap_or(0);
            let generation_channel_id = if graph_group.loop_momentum_bases.is_empty() {
                0
            } else {
                active_lmb_indices
                    .iter()
                    .position(|basis_index| *basis_index == generation_basis_id)
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "generation basis {} in graph group {} is not part of the active LMB list",
                            generation_basis_id, graph_group.group_id
                        ))
                    })?
            };

            let loop_momentum_bases_edge_ids = active_lmb_indices
                .iter()
                .map(|basis_index| graph_group.loop_momentum_bases[*basis_index].edge_ids.clone())
                .collect();
            let orientation_ids = graph_group
                .orientations
                .iter()
                .map(|orientation| orientation.orientation_id)
                .collect();
            let orientation_signatures = graph_group
                .orientations
                .iter()
                .map(|orientation| orientation.signature.clone())
                .collect();

            Ok(SimplifiedGraphGroupData {
                group_id: graph_group.group_id,
                master_graph_id,
                loop_momentum_bases_edge_ids,
                generation_channel_id,
                orientation_ids,
                orientation_signatures,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let result = SimplifiedStateData {
        state_folder: state_folder.display().to_string(),
        process_id: integrand_info.process_id,
        integrand_name: integrand_info.integrand_name,
        model,
        dot_file,
        external_momenta: external_momenta_data(&runtime_settings.kinematics.externals),
        e_cm: runtime_settings.kinematics.e_cm,
        graph_groups,
    };

    serde_json::to_string(&result).map_err(|err| {
        PyValueError::new_err(format!(
            "failed to serialize simplified state metadata: {err}"
        ))
    })
}
