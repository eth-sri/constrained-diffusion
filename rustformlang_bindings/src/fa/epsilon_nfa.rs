use pyo3::{prelude::*, py_run};
use rustformlang::fa::epsilon_nfa::ENFA as RustENFA;
use rustformlang::fa::finite_automaton::FiniteAutomaton;
use rustformlang::fa::state::State;
use rustformlang::input_symbol::{InputSymbol, EPSILON};
use rustformlang::language::Language;

use super::dfa::DFA;

#[pyfunction]
pub fn epsilon() -> PyResult<String> {
    Ok(EPSILON.to_string())
}

#[pyclass]
pub struct ENFA {
    pub enfa: RustENFA,
}

#[pymethods]
impl ENFA {
    #[new]
    pub fn new() -> PyResult<ENFA> {
        Ok(ENFA {
            enfa: RustENFA::empty(),
        })
    }

    pub fn add_transition(
        &mut self,
        from_state: String,
        symbol: String,
        to_state: String,
    ) -> PyResult<()> {
        self.enfa.add_transition(
            &State::from_string(from_state),
            &InputSymbol::from_string(symbol),
            &State::from_string(to_state),
        );
        Ok(())
    }

    pub fn set_start_state(&mut self, start_state: String) -> PyResult<()> {
        self.enfa.set_start_state(State::from_string(start_state));
        Ok(())
    }

    pub fn add_accept_state(&mut self, final_state: String) -> PyResult<()> {
        self.enfa.add_accept_state(State::from_string(final_state));
        Ok(())
    }

    pub fn num_states(&self) -> PyResult<usize> {
        Ok(self.enfa.states.len())
    }

    pub fn accepts_string(&self, input: String) -> PyResult<bool> {
        Ok(self.enfa.accepts_string(input.as_str()))
    }

    pub fn accepts(&self, input: Vec<String>) -> PyResult<bool> {
        Ok(self.enfa.accepts(
            &input
                .iter()
                .map(|s| InputSymbol::from_string(s.clone()))
                .collect::<Vec<InputSymbol>>(),
        ))
    }

    pub fn to_deterministic(&self) -> PyResult<DFA> {
        Ok(DFA {
            dfa: self.enfa.to_deterministic(),
        })
    }

    pub fn minimize(&self) -> PyResult<DFA> {
        Ok(DFA {
            dfa: self.enfa.minimize(),
        })
    }

    pub fn union(&mut self, other: &ENFA) -> PyResult<ENFA> {
        Ok(ENFA {
            enfa: self.enfa.union(&other.enfa),
        })
    }

    pub fn n_suffix_language(&self, n: usize) -> PyResult<ENFA> {
        Ok(ENFA {
            enfa: self.enfa.n_suffix_language(n),
        })
    }

    pub fn suffix_language(&self) -> PyResult<ENFA> {
        Ok(ENFA {
            enfa: self.enfa.suffix_language(),
        })
    }

    pub fn true_suffix_language(&self) -> PyResult<ENFA> {
        Ok(ENFA {
            enfa: self.enfa.true_suffix_language(),
        })
    }

    pub fn is_empty(&self) -> PyResult<bool> {
        Ok(self.enfa.is_empty())
    }

    pub fn to_graphviz(&self) -> PyResult<String> {
        Ok(self.enfa.to_graphviz())
    }

    pub fn concat(&mut self, other: &ENFA) {
        self.enfa.concat(&other.enfa);
    }
}

#[pyfunction]
pub fn minimize_enfa_threaded(py: Python<'_>, enfa: &ENFA) -> PyResult<DFA> {
    Ok(py.allow_threads(|| DFA {
        dfa: enfa.enfa.minimize(),
    }))
}

/// A Python module implemented in Rust.
pub fn register_child_module(py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // let m = PyModule::new(parent_module.py(), "rustformlang_fa_enfa")?;
    // py_run!(py, m, "import sys; sys.modules['rustformlang_fa_dfa'] = m");
    parent_module.add_class::<ENFA>()?;
    parent_module.add_function(wrap_pyfunction!(epsilon, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(minimize_enfa_threaded, parent_module)?)?;
    // parent_module.add_submodule(&m)?;
    Ok(())
}
