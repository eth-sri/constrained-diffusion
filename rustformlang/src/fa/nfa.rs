use crate::fa::dfa::DFA;
use crate::fa::state::State;
use crate::input_symbol::InputSymbol;
use crate::language::Language;
use elsa::vec;
use hashbrown::{HashMap, HashSet};
use std::cmp::max;
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct NFA {
    pub state_index_map: HashMap<State, usize>, // Map of state names to State indices
    pub alphabet_index_map: HashMap<InputSymbol, usize>, // Map of input symbols to indices

    pub states: Vec<State>,         // List of states
    pub alphabet: Vec<InputSymbol>, // Input symbols (alphabet)

    pub transitions: Vec<HashMap<usize, HashSet<usize>>>, // Transitions state -> input_symbol -> set of next states
    pub start_states: HashSet<usize>,                     // Start state
    pub accept_states: HashSet<usize>,                    // Accept states
}

impl NFA {
    /// Creates a new NFA
    pub fn new(
        state_index_map: HashMap<State, usize>,
        alphabet_index_map: HashMap<InputSymbol, usize>,
        states: Vec<State>,
        alphabet: Vec<InputSymbol>,
        transitions: Vec<HashMap<usize, HashSet<usize>>>,
        start_states: HashSet<usize>,
        accept_states: HashSet<usize>,
    ) -> Self {
        NFA {
            state_index_map,
            alphabet_index_map,
            states,
            alphabet,
            transitions,
            start_states,
            accept_states,
        }
    }

    pub fn empty() -> Self {
        NFA {
            state_index_map: HashMap::new(),
            alphabet_index_map: HashMap::new(),
            states: vec![],
            alphabet: vec![],
            transitions: vec![],
            start_states: HashSet::new(),
            accept_states: HashSet::new(),
        }
    }

    /// Sets the start state
    pub fn add_start_state(&mut self, start_state: State) {
        let start_index = self
            .state_index_map
            .entry(start_state.clone())
            .or_insert_with(|| {
                let index = self.states.len();
                self.states.push(start_state);
                self.transitions.push(HashMap::new());
                index
            });
        self.start_states.insert(*start_index);
    }

    /// Adds a new accept state
    pub fn add_accept_state(&mut self, accept_state: State) {
        let accept_index = self
            .state_index_map
            .entry(accept_state.clone())
            .or_insert_with(|| {
                let index = self.states.len();
                self.states.push(accept_state);
                self.transitions.push(HashMap::new());
                index
            });
        self.accept_states.insert(*accept_index);
    }

    /// Adds a transition from state `from` to state `to` on input `symbol`
    pub fn add_transition(&mut self, from: &State, symbol: &InputSymbol, to: &State) {
        let from_index = *self.state_index_map.entry(from.clone()).or_insert_with(|| {
            let index = self.states.len();
            self.states.push(from.clone());
            index
        });
        let to_index = *self.state_index_map.entry(to.clone()).or_insert_with(|| {
            let index = self.states.len();
            self.states.push(to.clone());
            index
        });
        let symbol_index = *self
            .alphabet_index_map
            .entry(symbol.clone())
            .or_insert_with(|| {
                let index = self.alphabet.len();
                self.alphabet.push(symbol.clone());
                index
            });

        // Ensure the transition vector is large enough
        while self.transitions.len() <= max(from_index, to_index) {
            self.transitions.push(HashMap::new());
        }

        // Add the transition
        self.transitions[from_index]
            .entry(symbol_index)
            .or_insert(HashSet::new())
            .insert(to_index);
    }

    fn _next_states(&self, states: &HashSet<usize>, symbol: usize) -> HashSet<usize> {
        let mut next_states = HashSet::new();

        for state in states {
            match self.transitions[*state].get(&symbol) {
                Some(next_states_set) => {
                    for next_state in next_states_set {
                        next_states.insert(*next_state);
                    }
                }
                None => {}
            }
        }

        next_states
    }

    fn _next_states_vec(&self, states: &Vec<usize>, symbol: usize) -> BTreeSet<usize> {
        let mut next_states = BTreeSet::new();

        for state in states {
            match self.transitions[*state].get(&symbol) {
                Some(next_states_set) => {
                    for next_state in next_states_set {
                        next_states.insert(*next_state);
                    }
                }
                None => {}
            }
        }

        next_states
    }

    pub fn to_deterministic(&self) -> DFA {
        // Convert NFA to DFA
        let orig_accept_states: HashSet<usize> = self.accept_states.clone();

        let dfa_start_state: Vec<usize> = self
            .start_states
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut dfa_accept_states: HashSet<Vec<usize>> = HashSet::new();
        let mut dfa_states: HashSet<Vec<usize>> =
            vec![dfa_start_state.clone()].into_iter().collect();
        let mut dfa_transitions: HashMap<Vec<usize>, HashMap<usize, BTreeSet<usize>>> =
            HashMap::new();

        let mut to_process: Vec<Vec<usize>> = vec![dfa_start_state.clone()];
        while let Some(current_states) = to_process.pop() {
            // if any state is a final state, add the current state to the DFA accept states
            if current_states
                .iter()
                .any(|state| orig_accept_states.contains(state))
            {
                dfa_accept_states.insert(current_states.clone());
            }

            // collect all symbols that continue from these states
            let all_continuing_symbols = current_states
                .iter()
                .map(|state| self.transitions[*state].keys())
                .flatten()
                .collect::<HashSet<_>>();

            // follow all states reachable from the current states by all symbols
            for symbol in all_continuing_symbols {
                let all_next_states = self._next_states_vec(&current_states, *symbol);

                if all_next_states.is_empty() {
                    continue;
                }
                // add a transition from the current state to the next state
                dfa_transitions
                    .entry(current_states.clone())
                    .or_insert_with(HashMap::new)
                    .entry(*symbol)
                    .or_insert(BTreeSet::new())
                    .extend(all_next_states.clone());
                // add this state to the DFA if not already present (and make sure to process it)
                let all_next_states: Vec<usize> = all_next_states.into_iter().collect();
                if dfa_states.insert(all_next_states.clone()) {
                    to_process.push(all_next_states);
                }
            }
        }
        DFA::from_states_transitions(
            dfa_start_state.into_iter().collect(),
            dfa_accept_states
                .into_iter()
                .map(|s| s.into_iter().collect())
                .collect(),
            dfa_states.into_iter().collect(),
            dfa_transitions
                .into_iter()
                .map(|(k, v)| {
                    (
                        k.into_iter().collect(),
                        v.into_iter()
                            .map(|(k, v)| (k, v.into_iter().collect()))
                            .collect(),
                    )
                })
                .collect(),
            self.alphabet.clone(),
            self.alphabet_index_map.clone(),
        )
    }
}

impl Language for NFA {
    /// Accepts a string if it can be accepted by the ENFA
    /// The input is a vector of input symbols
    /// The function returns true if the ENFA accepts the input string, false otherwise
    fn accepts(&self, input: &Vec<InputSymbol>) -> bool {
        // Start with the epsilon-closure of the start state
        let mut current_states = self.start_states.clone();
        for symbol in input {
            match self.alphabet_index_map.get(symbol) {
                Some(&symbol_index) => {
                    // Get the next states based on the current states and the input symbol
                    current_states = self._next_states(&current_states, symbol_index);
                }
                None => {
                    // If the symbol is not in the alphabet, return false
                    return false;
                }
            }
        }
        // Check if any of the current states are accept states
        !self.accept_states.is_disjoint(&current_states)
    }
}
