use crate::fa::nfa::NFA;
use crate::fa::state::State;
use crate::input_symbol::{epsilon, InputSymbol};
use crate::language::Language;
use hashbrown::{HashMap, HashSet};
use std::cmp::max;

#[derive(Debug, Clone)]
pub struct ENFA {
    pub state_index_map: HashMap<State, usize>, // Map of state names to State indices
    pub alphabet_index_map: HashMap<InputSymbol, usize>, // Map of input symbols to indices

    pub states: Vec<State>,         // List of states
    pub alphabet: Vec<InputSymbol>, // Input symbols (alphabet)

    pub transitions: Vec<HashMap<usize, HashSet<usize>>>, // Transitions state -> input_symbol -> set of next states
    pub start_state: usize,                               // Start state
    pub accept_states: HashSet<usize>,                    // Accept states
}

impl ENFA {
    /// Creates a new ENFA
    pub fn new(
        state_index_map: HashMap<State, usize>,
        alphabet_index_map: HashMap<InputSymbol, usize>,
        states: Vec<State>,
        alphabet: Vec<InputSymbol>,
        transitions: Vec<HashMap<usize, HashSet<usize>>>,
        start_state: usize,
        accept_states: HashSet<usize>,
    ) -> Self {
        ENFA {
            state_index_map,
            alphabet_index_map,
            states,
            alphabet,
            transitions,
            start_state,
            accept_states,
        }
    }

    pub fn empty() -> Self {
        ENFA {
            state_index_map: HashMap::from([(State::new("0"), 0)]),
            alphabet_index_map: HashMap::new(),
            states: vec![State::new("0")],
            alphabet: vec![],
            transitions: vec![HashMap::new()],
            start_state: 0,
            accept_states: HashSet::new(),
        }
    }

    /// Sets the start state
    pub fn set_start_state(&mut self, start_state: State) {
        let start_index = self
            .state_index_map
            .entry(start_state.clone())
            .or_insert_with(|| {
                let index = self.states.len();
                self.states.push(start_state);
                self.transitions.push(HashMap::new());
                index
            });
        self.start_state = *start_index;
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

    /// Epsilon-closure: computes all reachable states from a given set of states using ε-transitions
    fn _epsilon_closure(
        &self,
        states: HashSet<usize>,
        epsilon_index: Option<&usize>,
    ) -> HashSet<usize> {
        let mut closure = states.clone();
        let mut stack: Vec<usize> = Vec::from_iter(states);
        if epsilon_index.is_none() {
            return closure;
        }
        let epsilon_index = *epsilon_index.unwrap();

        while let Some(state) = stack.pop() {
            let next_states = self.transitions[state].get(&epsilon_index);
            match next_states {
                Some(next_states) => {
                    for next_state in next_states {
                        if closure.insert(*next_state) {
                            stack.push(*next_state);
                        }
                    }
                }
                None => {}
            }
        }

        closure
    }

    /// Computes the epsilon-closure of a set of states
    pub fn epsilon_closure(&self, states: &HashSet<State>) -> HashSet<State> {
        let epsilon_index = self.alphabet_index_map.get(&epsilon());
        self._epsilon_closure(
            states.iter().map(|s| self.state_index_map[s]).collect(),
            epsilon_index,
        )
        .iter()
        .map(|&s| self.states[s].clone())
        .collect()
    }

    /// Computes the next states based on the current states and input symbol
    fn _next_states(&self, states: &HashSet<usize>, symbol: usize) -> HashSet<usize> {
        let mut next_states = HashSet::new();

        for state in states {
            match self.transitions[*state].get(&symbol) {
                Some(next_states_set) => {
                    next_states.extend(next_states_set);
                }
                None => {}
            }
        }

        next_states
    }

    /// Computes the next reachable states based on current states
    fn _next_states_all(&self, states: &HashSet<usize>) -> HashSet<usize> {
        let mut next_states = HashSet::new();

        for state in states {
            for (_, next_states_set) in self.transitions[*state].iter() {
                next_states.extend(next_states_set);
            }
        }

        next_states
    }

    /// Computes an equivalent NFA without ε-transitions
    pub fn remove_epsilon_transitions(&self) -> NFA {
        // the start states are the epsilon-closure of the start state
        let epsilon_index = self.alphabet_index_map.get(&epsilon());
        let start_states = self._epsilon_closure(HashSet::from([self.start_state]), epsilon_index);
        let mut final_states = self.accept_states.clone();

        // Also add all transitions from epsilon-closure states to each other state
        let mut new_transitions = vec![HashMap::new(); self.states.len()];
        for state in 0..self.states.len() {
            let closure = self._epsilon_closure(HashSet::from([state]), epsilon_index);
            for closure_state in closure {
                // For each state in the closure, add transitions to the new state
                for (sym, next_states) in self.transitions[closure_state].iter() {
                    if Some(sym) == epsilon_index {
                        continue; // Skip epsilon transitions
                    }
                    new_transitions[state]
                        .entry(*sym)
                        .or_insert(HashSet::new())
                        .extend(next_states);
                }
                // If the closure state is an accept state, add it to the final states
                if self.accept_states.contains(&closure_state) {
                    final_states.insert(state);
                }
            }
        }
        // remove epsilon from the index map
        let mut new_alphabet_index_map = self.alphabet_index_map.clone();
        new_alphabet_index_map.remove(&epsilon());

        // create a new NFA without epsilon transitions
        NFA::new(
            self.state_index_map.clone(),
            new_alphabet_index_map,
            self.states.clone(),
            self.alphabet.clone(),
            new_transitions,
            start_states,
            final_states,
        )
    }

    pub fn union(&self, other: &Self) -> ENFA {
        // merge the states
        let other_states_base_index = self.states.len();
        let mut states = self.states.clone();
        states.extend(other.states.clone());
        let mut state_index_map = self.state_index_map.clone();
        state_index_map.extend(
            other
                .state_index_map
                .iter()
                .map(|(k, v)| (k.clone(), *v + other_states_base_index)),
        );

        // merge the alphabet
        let mut alphabet = self.alphabet.clone();
        let mut alphabet_index_map = self.alphabet_index_map.clone();
        let mut other_alphabet_index_map: Vec<usize> = vec![0; other.alphabet.len()];
        for (i, symbol) in other.alphabet.iter().enumerate() {
            match alphabet_index_map.get(symbol) {
                Some(&symbol_index) => {
                    other_alphabet_index_map[i] = symbol_index;
                }
                None => {
                    alphabet.push(symbol.clone());
                    alphabet_index_map.insert(symbol.clone(), alphabet.len() - 1);
                    other_alphabet_index_map[i] = alphabet.len() - 1;
                }
            }
        }

        // merge the transitions
        let mut transitions = self.transitions.clone();
        for other_transition_map in &other.transitions {
            let mut transition = HashMap::new();
            for (symbol, next_states) in other_transition_map.iter() {
                let new_symbol = other_alphabet_index_map[*symbol];
                let new_next_states = next_states
                    .iter()
                    .map(|state| *state + other_states_base_index)
                    .collect::<HashSet<usize>>();
                transition.insert(new_symbol, new_next_states);
            }
            transitions.push(transition);
        }

        // merge the accept states
        let mut accept_states = self.accept_states.clone();
        accept_states.extend(
            other
                .accept_states
                .iter()
                .map(|state| *state + other_states_base_index),
        );

        // introduce a new start state and add epsilon transitions to the start states of both ENFAs
        let start_state = states.len();
        states.push(State::new("start"));
        state_index_map.insert(State::new("start"), start_state);
        if alphabet_index_map.get(&epsilon()).is_none() {
            alphabet.push(epsilon());
            alphabet_index_map.insert(epsilon(), alphabet.len() - 1);
        }
        let epsilon_index = alphabet_index_map.get(&epsilon()).unwrap().clone();
        transitions.push(HashMap::from([(
            epsilon_index,
            HashSet::from([
                self.start_state,
                other.start_state + other_states_base_index,
            ]),
        )]));

        ENFA {
            state_index_map,
            alphabet_index_map,
            states,
            alphabet,
            transitions,
            start_state,
            accept_states,
        }
    }

    /// Creates a new ENFA that accepts the n-suffix language of the original ENFA
    /// In particular, it accepts all strings s where \exists w \in \Sigma^{n} o \Sigma^{*}: w o s \in L
    pub fn n_suffix_language(&self, n: usize) -> ENFA {
        // Create a new start state
        let new_start_state = State::new("start");
        let new_start_index = self.states.len();
        let mut state_index_map = self.state_index_map.clone();
        state_index_map.insert(new_start_state.clone(), new_start_index);
        let mut new_transitions = self.transitions.clone();
        let mut new_states = self.states.clone();
        new_states.push(new_start_state);

        let mut alphabet_index_map = self.alphabet_index_map.clone();
        let mut alphabet = self.alphabet.clone();
        let epsilon_index = self.alphabet_index_map.get(&epsilon());
        if epsilon_index.is_none() {
            alphabet.push(epsilon());
            alphabet_index_map.insert(epsilon(), alphabet.len() - 1);
        }
        // Add epsilon transitions from the new start state to every state that is reachable in n steps from the original start state

        // Collect all states reachable in n steps from the start state
        let mut start_n = HashSet::from([self.start_state]);
        start_n = self._epsilon_closure(start_n, epsilon_index);
        for _ in 0..n {
            start_n = self._next_states_all(&start_n);
            start_n = self._epsilon_closure(start_n, epsilon_index);
        }
        // Now collect all states reachable from any of the n-steps-away states
        let mut reached = start_n;
        let mut new_reached = reached.clone();
        while !new_reached.is_empty() {
            new_reached = self._next_states_all(&new_reached);
            new_reached = self._epsilon_closure(new_reached, epsilon_index);
            new_reached = new_reached.difference(&reached).cloned().collect();
            reached.extend(new_reached.iter());
        }

        // Add epsilon transition from new start state to the reached states
        let new_epsilon_index = *alphabet_index_map.get(&epsilon()).unwrap();
        new_transitions.push(HashMap::from([(new_epsilon_index, reached)]));

        // Return the modified ENFA
        ENFA {
            state_index_map,
            alphabet_index_map: alphabet_index_map,
            states: new_states,
            alphabet: alphabet,
            transitions: new_transitions,
            start_state: new_start_index,
            accept_states: self.accept_states.clone(),
        }
    }

    /// Return an ENFA that accepts the suffix language of the original ENFA
    /// In particular, it accepts all strings s where \exists w \in \Sigma^{*}: w o s \in L
    pub fn suffix_language(&self) -> ENFA {
        self.n_suffix_language(0)
    }

    // Return an ENFA that accepts true suffixes of the original ENFA
    /// In particular, it accepts all strings s where \exists w \in \Sigma^{+}: w o s \in L
    pub fn true_suffix_language(&self) -> ENFA {
        self.n_suffix_language(1)
    }

    /// Returns the set of reachable states from the given state
    pub fn reachable_states(&self, state: usize) -> HashSet<usize> {
        let mut reachable = HashSet::from_iter(vec![state]);
        let mut new_states = reachable.clone();
        let epsilon_index = self.alphabet_index_map.get(&epsilon());

        while !new_states.is_empty() {
            new_states = self._next_states_all(&new_states);
            new_states = self._epsilon_closure(new_states, epsilon_index);
            new_states = new_states.difference(&reachable).cloned().collect();
            reachable.extend(new_states.iter());
        }

        reachable
    }

    /// Returns whether the ENFA is empty
    pub fn is_empty(&self) -> bool {
        // Check if there are any accepting states that can be reached from the start state
        self.accept_states
            .is_disjoint(&self.reachable_states(self.start_state))
    }

    pub fn to_graphviz(&self) -> String {
        /// Converts the ENFA to a Graphviz representation
        /// The output is a string in the following format:
        /// digraph finite_state_machine {
        // 	fontname="Helvetica,Arial,sans-serif"
        // 	node [fontname="Helvetica,Arial,sans-serif"]
        // 	edge [fontname="Helvetica,Arial,sans-serif"]
        // 	rankdir=LR;
        // 	node [shape = doublecircle]; 0 3 4 8;
        // 	node [shape = circle];
        // 	0 -> 2 [label = "SS(B)"];
        // 	0 -> 1 [label = "SS(S)"];
        // 	1 -> 3 [label = "S($end)"];
        // 	2 -> 6 [label = "SS(b)"];
        // 	2 -> 5 [label = "SS(a)"];
        // 	2 -> 4 [label = "S(A)"];
        // 	5 -> 7 [label = "S(b)"];
        // 	5 -> 5 [label = "S(a)"];
        // 	6 -> 6 [label = "S(b)"];
        // 	6 -> 5 [label = "S(a)"];
        // 	7 -> 8 [label = "S(b)"];
        // 	7 -> 5 [label = "S(a)"];
        // 	8 -> 6 [label = "S(b)"];
        // 	8 -> 5 [label = "S(a)"];
        //  null [label= "", shape=none,height=.0,width=.0]
        //  null -> 0
        // }
        let mut output = String::from("digraph finite_state_machine {\n");
        output.push_str("\tfontname=\"Helvetica,Arial,sans-serif\"\n");
        output.push_str("\tnode [fontname=\"Helvetica,Arial,sans-serif\"]\n");
        output.push_str("\tedge [fontname=\"Helvetica,Arial,sans-serif\"]\n");
        output.push_str("\trankdir=LR;\n");
        // Add accept states
        output.push_str("\tnode [shape = doublecircle]; ");
        for state in &self.accept_states {
            output.push_str(&format!("{} ", state));
        }
        output.push_str(";\n");
        // Add normal states
        output.push_str("\tnode [shape = circle];\n");
        // Add transitions
        for (from_index, transitions) in self.transitions.iter().enumerate() {
            for (symbol_index, next_states) in transitions.iter() {
                let symbol = self.alphabet[*symbol_index].name.clone();
                for &to_index in next_states {
                    output.push_str(&format!(
                        "\t{} -> {} [label = \"{}\"];\n",
                        from_index, to_index, symbol
                    ));
                }
            }
        }
        // Add start state
        output.push_str("\tnull [label= \"\", shape=none,height=.0,width=.0]\n");
        output.push_str(&format!("\tnull -> {};\n", self.start_state));
        output.push_str("}\n");
        // Return
        output
    }

    /// Re-maps all input symbols to the given alphabet
    pub fn with_alphabet(&self, alphabet: &Vec<InputSymbol>) -> Self {
        let alphabet_index_map: HashMap<InputSymbol, usize> =
            HashMap::from_iter(alphabet.iter().enumerate().map(|(i, s)| (s.clone(), i)));
        let old_alphabet_map: Vec<Option<usize>> = Vec::from_iter(
            self.alphabet
                .iter()
                .enumerate()
                .map(|(i, s)| alphabet_index_map.get(s).cloned()),
        );
        let mut new_transitions: Vec<HashMap<usize, HashSet<usize>>> =
            Vec::with_capacity(self.transitions.len());
        for transition_map in self.transitions.iter() {
            new_transitions.push(HashMap::from_iter(transition_map.iter().filter_map(
                |(symbol, new_states)| old_alphabet_map[*symbol].map(|f| (f, new_states.clone())),
            )));
        }
        ENFA::new(
            self.state_index_map.clone(),
            alphabet_index_map,
            self.states.clone(),
            alphabet.clone(),
            new_transitions,
            self.start_state.clone(),
            self.accept_states.clone(),
        )
    }

    pub fn extend_alphabet(&mut self, alphabet: &Vec<InputSymbol>) {
        let own_alphabet: HashSet<InputSymbol> = HashSet::from_iter(self.alphabet.iter().cloned());
        self.alphabet.extend(
            own_alphabet
                .difference(&HashSet::from_iter(alphabet.iter().cloned()))
                .cloned(),
        );
    }

    pub fn concat(&mut self, other: &ENFA) {
        if (self.alphabet != other.alphabet) {
            self.extend_alphabet(&other.alphabet);
            self.concat(&other.with_alphabet(&self.alphabet));
            return;
        }
        let offset = self.states.len();
        self.transitions.extend(other.transitions.iter().map(|x| {
            x.iter()
                .map(|(symbol, target)| (*symbol, target.iter().map(|s| s + offset).collect()))
                .collect()
        }));
        self.states.extend(other.states.clone());
        let epsilon_index = self.alphabet_index_map.get(&epsilon());
        let epsilon_index = match epsilon_index {
            Some(i) => *i,
            None => {
                let new_index = self.alphabet.len();
                self.alphabet_index_map
                    .insert(epsilon(), self.alphabet.len());
                self.alphabet.push(epsilon());
                new_index
            }
        };
        // Add epsilon transitions from the final states to start states of other
        for final_state in self.accept_states.iter() {
            self.transitions[*final_state]
                .entry(epsilon_index)
                .or_insert_with(HashSet::new)
                .insert(other.start_state + offset);
        }
        self.accept_states = other.accept_states.iter().map(|s| s + offset).collect();
    }
}

impl Language for ENFA {
    /// Accepts a string if it can be accepted by the ENFA
    /// The input is a vector of input symbols
    /// The function returns true if the ENFA accepts the input string, false otherwise
    fn accepts(&self, input: &Vec<InputSymbol>) -> bool {
        let epsilon_index = self.alphabet_index_map.get(&epsilon());
        // Start with the epsilon-closure of the start state
        let mut current_states =
            self._epsilon_closure(HashSet::from([self.start_state]), epsilon_index);
        for symbol in input {
            match self.alphabet_index_map.get(symbol) {
                Some(&symbol_index) => {
                    // Get the next states based on the current states and the input symbol
                    current_states = self._next_states(&current_states, symbol_index);
                    current_states = self._epsilon_closure(current_states, epsilon_index);
                }
                None => {
                    // If the symbol is not in the alphabet, return false
                    return false;
                }
            }
        }
        // Check if any of the current states are accept states
        for state in &current_states {
            if self.accept_states.contains(state) {
                return true;
            }
        }
        false
    }
}
