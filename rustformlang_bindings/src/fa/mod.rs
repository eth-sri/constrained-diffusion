use pyo3::{prelude::*, py_run};
pub mod bytes_dfa;
pub mod dfa;
pub mod epsilon_nfa;

use crate::fa::bytes_dfa::register_child_module as register_bytes_dfa_module;
use crate::fa::dfa::register_child_module as register_dfa_module;
use crate::fa::epsilon_nfa::register_child_module as register_enfa_module;

pub fn register_child_module(py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // let m = PyModule::new(parent_module.py(), "rustformlang_fa")?;
    // py_run!(py, m, "import sys; sys.modules['rustformlang_fa'] = m");
    register_dfa_module(py, &parent_module)?;
    register_enfa_module(py, &parent_module)?;
    register_bytes_dfa_module(py, &parent_module)?;
    // parent_module.add_submodule(&m)?;
    Ok(())
}
