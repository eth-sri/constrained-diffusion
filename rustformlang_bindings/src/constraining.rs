use hashbrown::HashMap;

use pyo3::prelude::*;
use rustformlang::constraining::{
    all_lexings as rust_all_lexings, lex as rust_lex, prelex_word as rust_prelex_word,
    reset_lex_cache as rust_reset_lex_cache, Token,
};
use rustformlang::fa::bytes_dfa::BytesDFA;
use rustformlang::input_symbol::InputSymbol;

use crate::fa::bytes_dfa::BytesDFA as PyDFA;
use crate::fa::dfa::DFA as PyRawDFA;

#[pyclass]
pub struct LexMap {
    pub lex_map: HashMap<InputSymbol, (BytesDFA, BytesDFA, BytesDFA, BytesDFA)>,
}

#[pymethods]
impl LexMap {
    #[staticmethod]
    pub fn from_lex_map(
        lex_map: HashMap<String, (PyRef<PyDFA>, PyRef<PyDFA>, PyRef<PyDFA>, PyRef<PyDFA>)>,
    ) -> PyResult<LexMap> {
        Ok(LexMap {
            lex_map: lex_map
                .into_iter()
                .map(|(k, (a, b, c, d))| {
                    (
                        InputSymbol::from_string(k),
                        (a.dfa.clone(), b.dfa.clone(), c.dfa.clone(), d.dfa.clone()),
                    )
                })
                .collect(),
        })
    }

    pub fn keys(&self) -> Vec<String> {
        self.lex_map
            .keys()
            .map(|k| k.name.clone())
            .collect::<Vec<String>>()
    }

    pub fn get(&self, key: String) -> Option<(PyDFA, PyDFA, PyDFA, PyDFA)> {
        self.lex_map
            .get(&InputSymbol::from_string(key))
            .map(|(a, b, c, d)| {
                (
                    PyDFA { dfa: a.clone() },
                    PyDFA { dfa: b.clone() },
                    PyDFA { dfa: c.clone() },
                    PyDFA { dfa: d.clone() },
                )
            })
    }

    pub fn __contains__(&self, key: String) -> bool {
        self.lex_map.contains_key(&InputSymbol::from_string(key))
    }

    pub fn __len__(&self) -> usize {
        self.lex_map.len()
    }

    pub fn __getitem__(&self, key: String) -> PyResult<(PyDFA, PyDFA, PyDFA, PyDFA)> {
        self.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Key not found"))
    }
}

#[pyfunction]
pub fn reset_lex_cache(py: Python<'_>) {
    py.allow_threads(|| {
        rust_reset_lex_cache();
    });
}

#[pyfunction]
pub fn lex_string(
    py: Python<'_>,
    word: String,
    lex_map: PyRef<LexMap>,
    is_first: bool,
    strip_chars: Option<String>,
) -> PyResult<Vec<(Vec<String>, Option<String>, Option<String>)>> {
    let lex_map_clone = lex_map.lex_map.clone();
    Ok(py.allow_threads(|| {
        rust_lex(&Token::Word(word), &lex_map_clone, is_first, &strip_chars)
            .iter()
            .map(|(tokens, prefix, suffix)| {
                (
                    tokens.iter().map(|t| t.name.clone()).collect(),
                    prefix
                        .as_ref()
                        .map(|x| String::from_utf8(x.clone()).unwrap()),
                    suffix
                        .as_ref()
                        .map(|x| String::from_utf8(x.clone()).unwrap()),
                )
            })
            .collect()
    }))
}

#[pyfunction]
pub fn prelex_word(
    py: Python<'_>,
    word: String,
    prelex: String,
    is_first: bool,
    is_last: bool,
) -> String {
    py.allow_threads(|| rust_prelex_word(&word, &prelex, is_first, is_last))
}

#[derive(FromPyObject)]
pub enum TokenEnum<'py> {
    String(String),
    Lexing((Vec<String>, Option<String>, Option<String>)),
    None(Option<usize>),
    CatchAll(Bound<'py, PyAny>),
}

#[pyfunction]
pub fn all_lexings(
    py: Python<'_>,
    vocab: Vec<String>,
    lex_map: PyRef<LexMap>,
    prelex: Option<String>,
    strip_chars: Option<String>,
) -> PyResult<Vec<Vec<(Vec<String>, Option<String>, Option<String>)>>> {
    Ok(rust_all_lexings(
        &vocab,
        &lex_map.lex_map,
        prelex.as_deref(),
        strip_chars.as_deref(),
    )
    .iter()
    .map(|lexing| {
        lexing
            .iter()
            .map(|(tokens, prefix, suffix)| {
                (
                    tokens.iter().map(|t| t.name.clone()).collect(),
                    prefix
                        .as_ref()
                        .map(|x| String::from_utf8(x.clone()).unwrap()),
                    suffix
                        .as_ref()
                        .map(|x| String::from_utf8(x.clone()).unwrap()),
                )
            })
            .collect::<Vec<(Vec<String>, Option<String>, Option<String>)>>()
    })
    .collect())
}

/// A Python module implemented in Rust.
pub fn register_child_module(py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // let m = PyModule::new(parent_module.py(), "rustformlang.cfg")?;
    // py_run!(py, m, "import sys; sys.modules['rustformlang.cfg'] = m");
    parent_module.add_class::<LexMap>()?;
    parent_module.add_function(wrap_pyfunction!(reset_lex_cache, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(lex_string, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(prelex_word, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(all_lexings, parent_module)?)?;
    // parent_module.add_submodule(&m)?;
    Ok(())
}
