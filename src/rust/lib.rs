use pyo3::prelude::*;

mod gammaloop_state;

/// Lightweight sanity check used by the Python test suite to verify that the
/// PyO3 extension was built and imported from the active environment.
#[pyfunction]
fn health_check() -> &'static str {
    "glnis-gammaboard-api rust extension ok"
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(health_check, m)?)?;
    m.add_function(wrap_pyfunction!(gammaloop_state::classify_state_folder, m)?)?;
    m.add_function(wrap_pyfunction!(gammaloop_state::parse_state_summary_json, m)?)?;
    Ok(())
}
