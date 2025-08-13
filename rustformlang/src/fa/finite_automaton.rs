use hashbrown::{HashMap, HashSet};

use crate::fa::bytes_dfa::BytesDFA;
use crate::input_symbol::{epsilon, InputSymbol};

use super::dfa::DFA;
use super::epsilon_nfa::ENFA;
use super::nfa::NFA;
use super::state::State;

pub trait FiniteAutomaton {
    fn to_epsilon_automaton(&self) -> ENFA;

    fn remove_epsilon_transitions(&self) -> NFA {
        self.to_epsilon_automaton().remove_epsilon_transitions()
    }

    fn to_deterministic(&self) -> DFA {
        self.remove_epsilon_transitions().to_deterministic()
    }

    fn minimize(&self) -> DFA {
        let dfa = self.to_deterministic();
        dfa.minimize()
    }

    fn to_bytes_dfa(&self) -> BytesDFA {
        let alphabet: Vec<InputSymbol> = (0..std::u8::MAX)
            .map(crate::input_symbol::char_to_symbol)
            .collect();
        let with_alphabet = self.to_deterministic().with_alphabet(&alphabet);
        let transitions: Vec<HashMap<u8, usize>> = with_alphabet
            .transitions
            .iter()
            .map(|x| x.iter().map(|(k, v)| (*k as u8, *v as usize)).collect())
            .collect();

        BytesDFA {
            state_index_map: with_alphabet.state_index_map,
            states: with_alphabet.states,
            transitions,
            start_state: with_alphabet.start_state,
            accept_states: with_alphabet.accept_states,
        }
    }

    fn union(&self, other: &Self) -> ENFA {
        let enfa = self.to_epsilon_automaton();
        let other_enfa = other.to_epsilon_automaton();
        enfa.union(&other_enfa)
    }

    fn intersection(&self, other: &Self) -> DFA {
        let dfa = self.to_deterministic();
        let other_dfa = other.to_deterministic();
        dfa.intersection(&other_dfa)
    }

    fn difference(&self, other: &Self) -> DFA {
        let dfa = self.to_deterministic();
        let other_dfa = other.to_deterministic();
        dfa.difference(&other_dfa)
    }

    fn complement(&self) -> DFA {
        let dfa = self.to_deterministic();
        dfa.complement()
    }

    fn symmetric_difference(&self, other: &Self) -> ENFA {
        let dfa = self.to_deterministic();
        let other_dfa = other.to_deterministic();
        dfa.difference(&other_dfa)
            .to_epsilon_automaton()
            .union(&other_dfa.difference(&dfa).to_epsilon_automaton())
    }

    fn equals(&self, other: &Self) -> bool {
        // Two finite automata are equal if they accept the same language
        // i.e., if the difference between them is empty
        self.symmetric_difference(other).is_empty()
    }

    fn concat(&mut self, other: &Self) -> ENFA {
        let mut own_copy = self.to_epsilon_automaton();
        own_copy.concat(&other.to_epsilon_automaton());
        own_copy
    }
}

impl FiniteAutomaton for ENFA {
    fn to_epsilon_automaton(&self) -> ENFA {
        // ENFA is already an epsilon automaton
        self.clone()
    }
}
impl PartialEq for ENFA {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}
impl Eq for ENFA {}

impl FiniteAutomaton for DFA {
    fn to_epsilon_automaton(&self) -> ENFA {
        // Convert DFA to ENFA
        let transitions = self
            .transitions
            .iter()
            .map(|x| x.iter().map(|(k, v)| (*k, HashSet::from([*v]))).collect())
            .collect();
        ENFA {
            state_index_map: self.state_index_map.clone(),
            alphabet_index_map: self.alphabet_index_map.clone(),
            states: self.states.clone(),
            alphabet: self.alphabet.clone(),
            transitions: transitions,
            start_state: self.start_state,
            accept_states: self.accept_states.clone(),
        }
    }

    fn remove_epsilon_transitions(&self) -> NFA {
        // Convert DFA to ENFA
        let transitions = self
            .transitions
            .iter()
            .map(|x| x.iter().map(|(k, v)| (*k, HashSet::from([*v]))).collect())
            .collect();
        NFA {
            state_index_map: self.state_index_map.clone(),
            alphabet_index_map: self.alphabet_index_map.clone(),
            states: self.states.clone(),
            alphabet: self.alphabet.clone(),
            transitions: transitions,
            start_states: HashSet::from([self.start_state]),
            accept_states: self.accept_states.clone(),
        }
    }

    fn to_deterministic(&self) -> DFA {
        // DFA is already deterministic
        self.clone()
    }
}
impl PartialEq for DFA {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}
impl Eq for DFA {}

impl FiniteAutomaton for NFA {
    fn to_epsilon_automaton(&self) -> ENFA {
        // add a new start state and an epsilon transition to the old start states
        let mut alphabet = self.alphabet.clone();
        let mut alphabet_index_map = self.alphabet_index_map.clone();
        let epsilon_symbol = self
            .alphabet_index_map
            .get(&epsilon())
            .map(|s| *s)
            .unwrap_or_else(|| {
                alphabet.push(epsilon());
                alphabet_index_map.insert(epsilon(), alphabet.len() - 1);
                alphabet.len() - 1
            });

        let mut transitions = self.transitions.clone();
        let new_start_state = self.states.len();
        let new_transitions = HashMap::from([(epsilon_symbol, self.start_states.clone())]);
        transitions.push(new_transitions);
        let mut new_states = self.states.clone();
        new_states.push(State::new("q0"));
        let mut new_state_index_map = self.state_index_map.clone();
        new_state_index_map.insert(State::new("q0"), new_start_state);
        // Convert NFA to ENFA
        ENFA {
            state_index_map: new_state_index_map,
            alphabet_index_map: alphabet_index_map,
            states: new_states,
            alphabet: alphabet,
            transitions: transitions,
            start_state: new_start_state,
            accept_states: self.accept_states.clone(),
        }
    }

    fn remove_epsilon_transitions(&self) -> NFA {
        self.clone()
    }
}
impl PartialEq for NFA {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}
impl Eq for NFA {}

impl FiniteAutomaton for BytesDFA {
    fn to_epsilon_automaton(&self) -> ENFA {
        self.to_deterministic().to_epsilon_automaton()
    }

    fn to_deterministic(&self) -> DFA {
        let alphabet: Vec<InputSymbol> = (0..std::u8::MAX)
            .map(crate::input_symbol::char_to_symbol)
            .collect();
        let alphabet_index_map: HashMap<InputSymbol, usize> = HashMap::from_iter(
            alphabet
                .iter()
                .enumerate()
                .map(|(i, symbol)| (symbol.clone(), i)),
        );
        let transitions: Vec<HashMap<usize, usize>> = self
            .transitions
            .iter()
            .map(|x| x.iter().map(|(k, v)| (*k as usize, *v as usize)).collect())
            .collect();
        DFA {
            state_index_map: self.state_index_map.clone(),
            alphabet_index_map,
            states: self.states.clone(),
            alphabet,
            transitions: transitions,
            start_state: self.start_state,
            accept_states: self.accept_states.clone(),
        }
    }
}
impl PartialEq for BytesDFA {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}
impl Eq for BytesDFA {}
