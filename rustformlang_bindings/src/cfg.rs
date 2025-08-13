use std::time::Duration;

use hashbrown::HashMap;

use crate::fa::bytes_dfa::BytesDFA;
use crate::fa::dfa::DFA;
use pyo3::prelude::*;
use rustformlang::cfg::terminal::Terminal;
use rustformlang::cfg::{cfg::CFG as RustCFG, variable::Variable};
use rustformlang::language::Language;

#[pyclass]
pub struct CFG {
    pub cfg: RustCFG,
}

#[pymethods]
impl CFG {
    #[staticmethod]
    pub fn from_text(text: String, start_symbol: String) -> PyResult<CFG> {
        let start_symbol = Variable::new(start_symbol.as_str());
        Ok(CFG {
            cfg: RustCFG::from_text(text.as_str(), start_symbol),
        })
    }

    pub fn accepts(&self, input: Vec<String>) -> PyResult<bool> {
        Ok(self.cfg.accepts(
            &input
                .iter()
                .map(|s| Terminal::from_string(s.clone()))
                .collect::<Vec<Terminal>>(),
        ))
    }

    pub fn __contains__(&self, input: Vec<String>) -> PyResult<bool> {
        self.accepts(input)
    }

    pub fn accepts_string(&self, input: String) -> PyResult<bool> {
        Ok(self.cfg.accepts_string(input.as_str()))
    }

    pub fn is_empty(&self) -> PyResult<bool> {
        Ok(self.cfg.is_empty())
    }

    pub fn get_terminals(&self) -> PyResult<Vec<String>> {
        Ok(self.cfg.terminals.iter().map(|t| t.name.clone()).collect())
    }

    pub fn num_productions(&self) -> PyResult<usize> {
        Ok(self.cfg.productions.len())
    }

    pub fn to_normal_form(&self) -> PyResult<CFG> {
        Ok(CFG {
            cfg: self.cfg.to_normal_form(),
        })
    }

    pub fn intersection(&self, other: &DFA) -> PyResult<CFG> {
        Ok(CFG {
            cfg: self.cfg.intersection(&other.dfa),
        })
    }

    pub fn concatenate(&self, other: &CFG) -> PyResult<CFG> {
        Ok(CFG {
            cfg: self.cfg.clone().concatenate(&other.cfg),
        })
    }

    pub fn to_text(&self) -> PyResult<String> {
        Ok(self.cfg.to_text())
    }

    pub fn substitute(&self, terminal_map: HashMap<String, PyRef<CFG>>) -> PyResult<CFG> {
        let corrected_map: HashMap<Terminal, &RustCFG> = HashMap::from_iter(
            terminal_map
                .iter()
                .map(|(k, v)| (Terminal::from_string(k.clone()), &v.cfg)),
        );
        Ok(CFG {
            cfg: self.cfg.substitute(&corrected_map),
        })
    }

    pub fn is_intersection_empty(&self, other: &DFA, timeout: f64) -> PyResult<bool> {
        Ok(self
            .cfg
            .is_intersection_empty(&other.dfa, Some(Duration::from_secs_f64(timeout))))
    }

    pub fn example_word(&self, dfa: &DFA, timeout: f64) -> PyResult<Option<String>> {
        Ok(self
            .cfg
            .example_word(&dfa.dfa, Some(Duration::from_secs_f64(timeout)))
            .map(|t| {
                t.iter()
                    .map(|s| s.name.clone())
                    .collect::<Vec<String>>()
                    .join(" ")
            }))
    }
}

#[pyfunction]
fn is_intersection_empty_threaded(
    py: Python<'_>,
    cfg: &CFG,
    dfa: &DFA,
    timeout: f64,
) -> PyResult<bool> {
    Ok(py.allow_threads(|| {
        cfg.cfg
            .is_intersection_empty(&dfa.dfa, Some(Duration::from_secs_f64(timeout)))
    }))
}

/// A Python module implemented in Rust.
pub fn register_child_module(py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // let m = PyModule::new(parent_module.py(), "rustformlang.cfg")?;
    // py_run!(py, m, "import sys; sys.modules['rustformlang.cfg'] = m");
    parent_module.add_class::<CFG>()?;
    parent_module.add_function(wrap_pyfunction!(
        is_intersection_empty_threaded,
        parent_module
    )?)?;
    // parent_module.add_submodule(&m)?;
    Ok(())
}
