use std::path::{Path, PathBuf};

use gammaloop_api::{StateLoadOption, state::StateFolderKind};
use gammalooprs::processes::ProcessCollection;
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

    let loaded = StateLoadOption {
        state_folder: Some(state_folder.clone()),
        read_only_state: true,
        ..StateLoadOption::default()
    }
    .load()
    .map_err(|err| PyValueError::new_err(format!("failed to load GammaLoop state: {err:?}")))?;

    let processes = loaded
        .state
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
