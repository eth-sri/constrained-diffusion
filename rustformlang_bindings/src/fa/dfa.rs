use pyo3::{prelude::*, py_run};
use regex_syntax::escape;
use rustformlang::fa::dfa::DFA as RustDFA;
use rustformlang::fa::finite_automaton::FiniteAutomaton;
use rustformlang::input_symbol::InputSymbol;
use rustformlang::language::Language;
use rustformlang::regex::regex::regex_to_dfa as rust_regex_to_dfa;

use super::epsilon_nfa::ENFA;

#[pyclass]
pub struct DFA {
    pub dfa: RustDFA,
}

#[pymethods]
impl DFA {
    pub fn accepts_string(&self, input: String) -> PyResult<bool> {
        Ok(self.dfa.accepts_string(input.as_str()))
    }

    pub fn accepts(&self, input: Vec<String>) -> PyResult<bool> {
        Ok(self.dfa.accepts(
            &input
                .iter()
                .map(|s| InputSymbol::from_string(s.clone()))
                .collect::<Vec<InputSymbol>>(),
        ))
    }

    pub fn is_empty(&self) -> PyResult<bool> {
        Ok(self.dfa.is_empty())
    }

    pub fn num_states(&self) -> PyResult<usize> {
        Ok(self.dfa.states.len())
    }

    pub fn to_epsilon_automaton(&self) -> PyResult<ENFA> {
        Ok(ENFA {
            enfa: self.dfa.to_epsilon_automaton(),
        })
    }

    pub fn minimize(&self) -> PyResult<DFA> {
        Ok(DFA {
            dfa: self.dfa.minimize(),
        })
    }

    pub fn intersection(&mut self, other: &DFA) -> PyResult<DFA> {
        Ok(DFA {
            dfa: self.dfa.intersection(&other.dfa),
        })
    }

    pub fn complement(&mut self) -> PyResult<DFA> {
        Ok(DFA {
            dfa: self.dfa.complement(),
        })
    }

    pub fn difference(&mut self, other: &DFA) -> PyResult<DFA> {
        Ok(DFA {
            dfa: self.dfa.difference(&other.dfa),
        })
    }

    pub fn n_prefix_language(&self, n: usize) -> PyResult<DFA> {
        Ok(DFA {
            dfa: self.dfa.clone().n_prefix_language(n),
        })
    }

    pub fn prefix_language(&self) -> PyResult<DFA> {
        Ok(DFA {
            dfa: self.dfa.clone().prefix_language(),
        })
    }

    pub fn true_prefix_language(&self) -> PyResult<DFA> {
        Ok(DFA {
            dfa: self.dfa.clone().true_prefix_language(),
        })
    }

    pub fn accept_prefix(&self, input: Vec<String>) -> PyResult<Option<usize>> {
        Ok(self.dfa.accept_prefix(
            &input
                .iter()
                .map(|s| InputSymbol::from_string(s.clone()))
                .collect::<Vec<InputSymbol>>(),
        ))
    }

    pub fn accept_prefix_string(&self, input: String) -> PyResult<Option<usize>> {
        Ok(self.dfa.accept_prefix_string(input.as_str()))
    }

    pub fn accept_prefix_bytes(&self, input: Vec<u8>) -> PyResult<Option<usize>> {
        Ok(self.dfa.accept_prefix_bytes(&input))
    }

    pub fn to_text(&self) -> PyResult<String> {
        Ok(format!("{}", self.dfa))
    }

    pub fn __eq__(&self, other: &DFA) -> PyResult<bool> {
        Ok(self.dfa.equals(&other.dfa))
    }

    pub fn to_bytes_dfa(&self) -> PyResult<crate::fa::bytes_dfa::BytesDFA> {
        Ok(crate::fa::bytes_dfa::BytesDFA {
            dfa: self.dfa.to_bytes_dfa(),
        })
    }

    pub fn concat(&self, other: &DFA) -> PyResult<DFA> {
        Ok(DFA {
            dfa: self.dfa.clone().concat(&other.dfa).to_deterministic(),
        })
    }
}

#[pyfunction]
pub fn minimize_dfa_threaded(py: Python<'_>, dfa: &DFA) -> PyResult<DFA> {
    Ok(py.allow_threads(|| DFA {
        dfa: dfa.dfa.minimize(),
    }))
}

/// A Python module implemented in Rust.
pub fn register_child_module(py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // let m = PyModule::new(parent_module.py(), "rustformlang_fa_dfa")?;
    // py_run!(py, m, "import sys; sys.modules['rustformlang_fa_dfa'] = m");
    parent_module.add_class::<DFA>()?;
    parent_module.add_function(wrap_pyfunction!(minimize_dfa_threaded, parent_module)?)?;
    // parent_module.add_submodule(&m)?;
    Ok(())
}
