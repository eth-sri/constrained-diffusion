pub mod cfg;
pub mod constraining;
pub mod fa;

use crate::cfg::register_child_module as register_cfg_module;
use crate::constraining::register_child_module as register_constraining_module;
use crate::fa::register_child_module as register_fa_module;
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule(name = "rustformlang")]
fn rustformlang_bindings(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_cfg_module(py, m)?;
    register_fa_module(py, m)?;
    register_constraining_module(py, m)?;
    Ok(())
}
