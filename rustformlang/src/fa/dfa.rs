use rustc_hash::FxHashSet;

use crate::fa::state::State;
use crate::input_symbol::{char_to_symbol, InputSymbol};
use crate::language::Language;
use hashbrown::{HashMap, HashSet};
use std::cmp::max;
use std::collections::{BTreeSet, VecDeque};
use std::fmt::Display;
use std::hash::Hash;

#[derive(Debug, Clone)]
pub struct DFA {
    pub state_index_map: HashMap<State, usize>, // Map of state names to State indices
    pub alphabet_index_map: HashMap<InputSymbol, usize>, // Map of input symbols to indices

    pub states: Vec<State>,         // List of states
    pub alphabet: Vec<InputSymbol>, // Input symbols (alphabet)

    pub transitions: Vec<HashMap<usize, usize>>, // Transitions (state -> input_symbol -> next state)
    pub start_state: usize,                      // Start state
    pub accept_states: HashSet<usize>,           // Accept states
}

impl DFA {
    /// Creates a new DFA
    pub fn new(
        state_index_map: HashMap<State, usize>,
        alphabet_index_map: HashMap<InputSymbol, usize>,
        states: Vec<State>,
        alphabet: Vec<InputSymbol>,
        transitions: Vec<HashMap<usize, usize>>,
        start_state: usize,
        accept_states: HashSet<usize>,
    ) -> Self {
        DFA {
            state_index_map,
            alphabet_index_map,
            states,
            alphabet,
            transitions,
            start_state,
            accept_states,
        }
    }

    pub fn get_start_state(&self) -> &State {
        &self.states[self.start_state]
    }

    pub fn get_accept_states(&self) -> HashSet<State> {
        HashSet::from_iter(self.accept_states.iter().map(|&s| &self.states[s]).cloned())
    }

    pub fn get_transitions(&self) -> Vec<(State, InputSymbol, State)> {
        let mut transitions = Vec::new();
        for (from_state, transitions_map) in self.transitions.iter().enumerate() {
            for (symbol_index, &to_state) in transitions_map.iter() {
                let from_state_name = self.states[from_state].clone();
                let to_state_name = self.states[to_state].clone();
                let symbol = self.alphabet[*symbol_index].clone();
                transitions.push((from_state_name, symbol, to_state_name));
            }
        }
        transitions
    }

    pub fn empty() -> Self {
        DFA {
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

        // Add the transition (note this brutally overrides any existing transition)
        self.transitions[from_index].insert(symbol_index, to_index);
    }

    /// Returns the next state given the current state and input symbol
    fn _next_state(&self, state: usize, symbol: usize) -> Option<usize> {
        self.transitions[state].get(&symbol).cloned()
    }

    /// Returns the next states given the current state
    fn _next_states(&self, state: usize) -> HashSet<usize> {
        HashSet::from_iter(self.transitions[state].values().cloned())
    }

    pub fn from_states_transitions(
        dfa_start_state: Vec<usize>,
        dfa_accept_states: HashSet<Vec<usize>>,
        dfa_states: Vec<Vec<usize>>,
        dfa_transitions: HashMap<Vec<usize>, HashMap<usize, Vec<usize>>>,
        dfa_alphabet: Vec<InputSymbol>,
        dfa_alphabet_index_map: HashMap<InputSymbol, usize>,
    ) -> DFA {
        let states: Vec<State> = (0..dfa_states.len())
            .map(|i| {
                let state_name = format!("q{}", i);
                State::new(&state_name)
            })
            .collect();
        // Step 1: Create a mapping from each Vec<usize> in dfa_states to its index.
        let mut states_map: HashMap<Vec<usize>, usize> = HashMap::new();
        for (index, state) in dfa_states.iter().enumerate() {
            states_map.insert(state.clone(), index);
        }

        // Step 2: Convert the start state using the states_map.
        let start_state_index = *states_map
            .get(&dfa_start_state)
            .expect("Start state not found in states map!");

        // Step 3: Convert the accept states to their indices.
        let accept_states_indices: HashSet<usize> = dfa_accept_states
            .iter()
            .map(|state| states_map[state])
            .collect();

        // Step 4: Build the transition table using states_map.
        let mut transitions = vec![HashMap::new(); dfa_states.len()];
        for (from_state, inner_map) in dfa_transitions {
            let from_index = *states_map
                .get(&from_state)
                .expect("From state not found in states map!");
            for (symbol_index, to_state) in inner_map {
                let to_index = *states_map
                    .get(&to_state)
                    .expect("To state not found in states map!");
                transitions[from_index].insert(symbol_index, to_index);
            }
        }

        // Step 5: Construct the resulting DFA.
        DFA {
            state_index_map: HashMap::new(), // This might be built later if needed
            alphabet_index_map: dfa_alphabet_index_map,
            states: states,
            alphabet: dfa_alphabet,
            transitions,
            start_state: start_state_index,
            accept_states: accept_states_indices,
        }
    }

    /// Returns the set of reachable states from the given state
    pub fn reachable_states(&self, state: usize) -> HashSet<usize> {
        let mut reachable = HashSet::from_iter(vec![state]);
        let mut stack = vec![state];

        while let Some(current_state) = stack.pop() {
            for (_, &next_state) in self.transitions[current_state].iter() {
                if reachable.insert(next_state) {
                    stack.push(next_state);
                }
            }
        }

        reachable
    }

    /// Returns the set of states that lead to the accept state
    pub fn leading_to_accept_states(&self) -> HashSet<usize> {
        // invert the transition map
        let mut inverted_transitions: Vec<HashSet<usize>> = vec![HashSet::new(); self.states.len()];
        for (from_state, transitions) in self.transitions.iter().enumerate() {
            for (_, &to_state) in transitions.iter() {
                inverted_transitions[to_state].insert(from_state);
            }
        }

        // Find all states that lead to the accept states
        let mut reachable = self.accept_states.clone();
        let mut stack = Vec::from_iter(self.accept_states.iter().cloned());
        while let Some(current_state) = stack.pop() {
            for &next_state in inverted_transitions[current_state].iter() {
                if reachable.insert(next_state) {
                    stack.push(next_state);
                }
            }
        }

        reachable
    }

    /// Partitions the states into groups of equivalent states
    /// The function returns a vector of sets, where each set contains indices of equivalent states
    pub fn partition_states(&self, states: &HashSet<usize>) -> HashSet<BTreeSet<usize>> {
        // Following https://www.irif.fr/~carton/Enseignement/Complexite/ENS/Redaction/2008-2009/yingjie.xu.pdf

        // invert the transition map
        let mut inverted_transitions: Vec<Vec<HashSet<usize>>> =
            vec![vec![HashSet::new(); self.alphabet.len()]; self.states.len()];
        for (from_state, transitions) in self.transitions.iter().enumerate() {
            for (&to_symbol, &to_state) in transitions.iter() {
                inverted_transitions[to_state][to_symbol].insert(from_state);
            }
        }

        // core algorithm begins here
        // states = Q which might be less than self.states -> we intersect with states to get relevant accept states
        let accept_states = BTreeSet::from_iter(states.intersection(&self.accept_states).cloned());
        let mut w_set = HashSet::from_iter(vec![
            accept_states,
            BTreeSet::from_iter(states.difference(&self.accept_states).cloned()),
        ]);
        let mut p_set: HashSet<BTreeSet<usize>> = w_set.clone();
        while !w_set.is_empty() {
            // pop a set from w_set
            let s_set = w_set.iter().next().unwrap().clone();
            w_set.remove(&s_set);

            for symbol in 0..self.alphabet.len() {
                // this computes the inverse for the set
                let l_a: BTreeSet<usize> = BTreeSet::from_iter(
                    s_set
                        .iter()
                        .flat_map(|&s| inverted_transitions[s][symbol].iter())
                        .cloned(),
                );
                if l_a.is_empty() {
                    continue;
                }
                let mut splits_to_remove = Vec::new();
                for r_set in p_set.iter() {
                    let r_1 = BTreeSet::from_iter(r_set.intersection(&l_a).cloned());
                    if r_1.is_empty() || r_set.is_subset(&l_a) {
                        continue;
                    }
                    let r_2 = BTreeSet::from_iter(r_set.difference(&r_1).cloned());
                    if w_set.contains(r_set) {
                        w_set.remove(r_set);
                        w_set.insert(r_1.clone());
                        w_set.insert(r_2.clone());
                    } else {
                        if r_1.len() <= r_2.len() {
                            w_set.insert(r_1.clone());
                        } else {
                            w_set.insert(r_2.clone());
                        }
                    }
                    splits_to_remove.push((r_set.clone(), r_1, r_2));
                }
                // remove the splits from p_set
                for (r_set, r_1, r_2) in splits_to_remove {
                    p_set.remove(&r_set);
                    p_set.insert(r_1);
                    p_set.insert(r_2);
                }
            }
        }
        // return the partition
        p_set
    }

    /// return a new DFA that is the minimized version of this DFA
    pub fn minimize(&self) -> DFA {
        // remove unreachable states
        let states = HashSet::from_iter(
            self.reachable_states(self.start_state)
                .intersection(&self.leading_to_accept_states())
                .cloned(),
        );
        let accept_states: HashSet<usize> =
            HashSet::from_iter(self.accept_states.intersection(&states).cloned());
        if accept_states.is_empty() {
            return DFA::empty();
        }

        // Group equivalent states
        let partitions = Vec::from_iter(self.partition_states(&states));

        // Map each original state to its partition
        let mut state_to_partition: HashMap<usize, usize> = HashMap::new();
        for (i, partition) in partitions.iter().enumerate() {
            for state in partition {
                state_to_partition.insert(*state, i);
            }
        }

        // Create a new DFA with the partitions as states
        let dfa_start_state: usize = state_to_partition[&self.start_state];
        let dfa_states: Vec<State> = partitions
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let state_name = format!("q{}", i);
                State::new(&state_name)
            })
            .collect();
        let dfa_accept_states: HashSet<usize> =
            HashSet::from_iter(accept_states.iter().map(|s| state_to_partition[s]));
        let mut dfa_transitions: Vec<HashMap<usize, usize>> =
            vec![HashMap::new(); partitions.len()];
        for (from_partition, inner_map) in self.transitions.iter().enumerate() {
            if !states.contains(&from_partition) {
                continue;
            }
            for (symbol_index, to_partition) in inner_map {
                if !states.contains(to_partition) {
                    continue;
                }
                let from_partition_index = state_to_partition[&from_partition];
                let to_partition_index = state_to_partition[to_partition];
                dfa_transitions[from_partition_index].insert(*symbol_index, to_partition_index);
            }
        }
        let dfa_alphabet: Vec<InputSymbol> = self.alphabet.clone();
        let dfa_alphabet_index_map = self.alphabet_index_map.clone();

        DFA {
            state_index_map: HashMap::new(), // This might be built later if needed
            alphabet_index_map: dfa_alphabet_index_map,
            states: dfa_states,
            alphabet: dfa_alphabet,
            transitions: dfa_transitions,
            start_state: dfa_start_state,
            accept_states: dfa_accept_states,
        }
    }

    /// Returns whether the DFA is empty
    pub fn is_empty(&self) -> bool {
        let mut reachable: FxHashSet<usize> = FxHashSet::from_iter(vec![self.start_state]);
        let mut stack = vec![self.start_state];

        while let Some(current_state) = stack.pop() {
            if self.accept_states.contains(&current_state) {
                return false;
            }
            for (_, &next_state) in self.transitions[current_state].iter() {
                if reachable.insert(next_state) {
                    stack.push(next_state);
                }
            }
        }

        true
    }

    fn ensure_alphabet(&mut self, alphabet: &Vec<InputSymbol>) {
        for symbol in alphabet {
            if !self.alphabet.contains(&symbol) {
                self.alphabet.push(symbol.clone());
                self.alphabet_index_map
                    .insert(symbol.clone(), self.alphabet.len() - 1);
            }
        }
    }

    /// Returns a new DFA that accepts the complement of the language of this DFA
    pub fn complement(&self) -> DFA {
        // insert a new final state
        let trash_node_index = self.states.len();
        let trash_node = State::new("trash");
        let mut states = self.states.clone();
        states.push(trash_node.clone());

        let mut state_index_map = self.state_index_map.clone();
        state_index_map.insert(trash_node, trash_node_index);

        // invert the accept states
        let mut accepting_states: HashSet<usize> = HashSet::from_iter(
            HashSet::from_iter(0..self.states.len())
                .difference(&self.accept_states)
                .cloned(),
        );
        accepting_states.insert(trash_node_index);

        // add missing transitions from every state to the trash node
        let mut transitions = self.transitions.clone();
        let all_symbols: HashSet<usize> = HashSet::from_iter(0..self.alphabet.len());
        for transition in transitions.iter_mut() {
            transition.extend(
                all_symbols
                    .difference(&transition.keys().cloned().collect())
                    .map(|symbol| (*symbol, trash_node_index)),
            );
        }

        // add transitions from the trash node to itself
        transitions.push(HashMap::from_iter(
            all_symbols.iter().map(|symbol| (*symbol, trash_node_index)),
        ));

        DFA {
            state_index_map,
            alphabet_index_map: self.alphabet_index_map.clone(),
            states,
            alphabet: self.alphabet.clone(),
            transitions,
            start_state: self.start_state,
            accept_states: accepting_states,
        }
    }

    /// Returns a new DFA that accepts the intersection of the languages of this DFA and the other DFA
    pub fn intersection(&self, other: &DFA) -> DFA {
        let alphabet = self.alphabet.clone();
        let alphabet_index_map = self.alphabet_index_map.clone();
        // map of other alphabet symbols to this DFA's alphabet
        // if the symbol is not in this DFA's alphabet, the result is None
        let other_alphabet_map: HashMap<usize, usize> =
            HashMap::from_iter(other.alphabet.iter().enumerate().filter_map(|(i, symbol)| {
                alphabet_index_map
                    .get(symbol)
                    .cloned()
                    .map(|index| (i, index))
            }));

        let mut states = Vec::new();
        for i in 0..self.states.len() {
            for j in 0..other.states.len() {
                let state_name = format!("{}-{}", self.states[i].name, other.states[j].name);
                states.push(State::new(&state_name));
            }
        }
        let other_states_len = other.states.len();
        let state_index_fn = |i: usize, j: usize| -> usize { i * other_states_len + j };
        let mut to_process = vec![(self.start_state, other.start_state)];
        let mut processed = HashSet::new();

        let mut accept_states: HashSet<usize> = HashSet::new();
        for i in self.accept_states.iter() {
            for j in other.accept_states.iter() {
                let state_index = state_index_fn(*i, *j);
                accept_states.insert(state_index);
            }
        }

        // create the intersection states and transitions
        let mut transitions = vec![HashMap::new(); states.len()];
        while let Some((i, j)) = to_process.pop() {
            let state_index = state_index_fn(i, j);
            for (other_symbol, &next_other_state) in other.transitions[j].iter() {
                if let Some(symbol_index) = other_alphabet_map.get(other_symbol) {
                    if let Some(&own_next_state) = self.transitions[i].get(symbol_index) {
                        // add the transition
                        let next_index = state_index_fn(own_next_state, next_other_state);
                        transitions[state_index].insert(*symbol_index, next_index);
                        // mark for processing
                        let state_tuple = (own_next_state, next_other_state);
                        if processed.insert(state_tuple) {
                            to_process.push(state_tuple);
                        }
                    }
                }
            }
        }
        let state_index_map = HashMap::from_iter(
            states
                .iter()
                .enumerate()
                .map(|(i, state)| (state.clone(), i)),
        );

        DFA {
            state_index_map,
            alphabet_index_map,
            states,
            alphabet,
            transitions,
            start_state: state_index_fn(self.start_state, other.start_state),
            accept_states,
        }
    }

    pub fn difference(&self, other: &Self) -> DFA {
        let mut other_sure = other.clone();
        other_sure.ensure_alphabet(&self.alphabet);
        self.intersection(&other_sure.complement())
    }

    /// Return a DFA that accepts the n prefix language of the original DFA
    /// In particular, it accepts all strings s where \exists w \in \Sigma^{n} o \Sigma^{*}: s o w \in L
    pub fn n_prefix_language(mut self, n: usize) -> DFA {
        // Get all states leading to an accept state
        let leading_states = self.leading_to_accept_states();

        // Make all states accept states that reach a leading state in n steps
        let mut accept_states = HashSet::new();
        for state in 0..self.states.len() {
            let mut reachable = HashSet::from([state]);
            for _ in 0..n {
                let mut next_reachable = HashSet::new();
                for cur_state in reachable.iter() {
                    next_reachable.extend(self._next_states(*cur_state));
                }
                reachable = next_reachable;
            }
            if !reachable.is_disjoint(&leading_states) {
                accept_states.insert(state);
            }
        }

        // Create a new DFA with the same states and transitions, but with the new accept states
        self.accept_states = accept_states;
        self
    }

    /// Return a DFA that accepts the prefix language of the original DFA
    /// In particular, it accepts all strings s where \exists w \in \Sigma^{*}: s o w \in L
    pub fn prefix_language(self) -> DFA {
        self.n_prefix_language(0)
    }

    /// Return a DFA that accepts the true prefix language of the original DFA
    /// In particular, it accepts all strings s where \exists w \in \Sigma^{+}: s o w \in L
    pub fn true_prefix_language(self) -> DFA {
        self.n_prefix_language(1)
    }

    /// return the maximum prefix of the word that is accepted by the DFA
    /// equivalently this is the "span" of the regular expression to the word (greedy, maximum match)

    /// if the word is never accepted, returns None
    /// if the word is accepted returns the number of characters consumed
    pub fn accept_prefix(&self, input: &Vec<InputSymbol>) -> Option<usize> {
        let mut current_state = self.start_state;
        let mut max_accepted = match self.accept_states.contains(&current_state) {
            true => Some(0),
            false => None,
        };
        for (consumed, symbol) in input.iter().enumerate() {
            match self.alphabet_index_map.get(symbol) {
                Some(&symbol_index) => {
                    match self._next_state(current_state, symbol_index) {
                        Some(next_state) => {
                            current_state = next_state;
                            if self.accept_states.contains(&current_state) {
                                max_accepted = Some(consumed + 1);
                            }
                        }
                        None => {
                            // If there is no transition for the symbol, break
                            break;
                        }
                    }
                }
                None => {
                    // If the symbol is not in the alphabet, break;
                    break;
                }
            }
        }
        max_accepted
    }

    pub fn accept_prefix_string(&self, input: &str) -> Option<usize> {
        let input: Vec<InputSymbol> = input.bytes().map(char_to_symbol).collect();
        self.accept_prefix(&input)
    }

    pub fn accept_prefix_bytes(&self, input: &[u8]) -> Option<usize> {
        let input: Vec<InputSymbol> = input.iter().cloned().map(char_to_symbol).collect();
        self.accept_prefix(&input)
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
        let mut new_transitions: Vec<HashMap<usize, usize>> =
            Vec::with_capacity(self.transitions.len());
        for transition_map in self.transitions.iter() {
            new_transitions.push(HashMap::from_iter(transition_map.iter().filter_map(
                |(symbol, new_state)| old_alphabet_map[*symbol].map(|f| (f, *new_state)),
            )));
        }
        DFA::new(
            self.state_index_map.clone(),
            alphabet_index_map,
            self.states.clone(),
            alphabet.clone(),
            new_transitions,
            self.start_state.clone(),
            self.accept_states.clone(),
        )
    }

    pub fn sub_automaton(&self, new_start: usize, new_end: usize) -> Self {
        DFA::new(
            self.state_index_map.clone(),
            self.alphabet_index_map.clone(),
            self.states.clone(),
            self.alphabet.clone(),
            self.transitions.clone(),
            new_start,
            HashSet::from_iter(vec![new_end]),
        )
    }

    pub fn example_word(&self) -> Option<Vec<InputSymbol>> {
        // Does a simple BFS through the automaton to find the shortest possible example word
        let mut state_to_next: HashMap<usize, (usize, usize)> = HashMap::new();

        let mut queue = VecDeque::new();
        queue.push_back(self.start_state);
        while let Some(current_state) = queue.pop_front() {
            if self.accept_states.contains(&current_state) {
                let mut final_word = vec![];
                let mut cur_state = current_state;
                while let Some(next_state) = state_to_next.get(&cur_state) {
                    final_word.push(next_state.0);
                    cur_state = next_state.1;
                }
                return Some(
                    final_word
                        .iter()
                        .rev()
                        .map(|s| self.alphabet[*s].clone())
                        .collect(),
                );
            }
            for (&symbol, &to) in self
                .transitions
                .get(current_state)
                .unwrap_or(&HashMap::new())
                .iter()
            {
                if !state_to_next.contains_key(&to) {
                    state_to_next.insert(to, (symbol, current_state));
                }
            }
        }

        None
    }

    pub fn intersection_example_word(&self, other: &Self) -> Option<Vec<InputSymbol>> {
        // Does a simple BFS through the automaton to find the shortest possible example word
        let mut state_to_next: HashMap<(usize, usize), (usize, (usize, usize))> = HashMap::new();

        let mut queue = VecDeque::new();
        queue.push_back((self.start_state, other.start_state));
        while let Some((current_state, other_state)) = queue.pop_front() {
            if self.accept_states.contains(&current_state)
                && other.accept_states.contains(&other_state)
            {
                // Reconstruct the example word
                let mut final_word = vec![];
                let mut cur_state = (current_state, other_state);
                while let Some(next_state) = state_to_next.get(&cur_state) {
                    final_word.push(self.alphabet[next_state.0].clone());
                    cur_state = next_state.1;
                }
                return Some(final_word.iter().rev().cloned().collect());
            }
            for (&symbol, &own_to) in self
                .transitions
                .get(current_state)
                .unwrap_or(&HashMap::new())
                .iter()
            {
                let other_to_opt = other._next_state(other_state, symbol);
                if let Some(other_to) = other_to_opt {
                    let to = (own_to, other_to);
                    if !state_to_next.contains_key(&to) {
                        state_to_next.insert(to, (symbol, (current_state, other_state)));
                        queue.push_back(to);
                    }
                }
            }
        }

        None
    }
}

impl Language for DFA {
    /// Accepts a string if it can be accepted by the ENFA
    /// The input is a vector of input symbols
    /// The function returns true if the ENFA accepts the input string, false otherwise
    fn accepts(&self, input: &Vec<InputSymbol>) -> bool {
        // Start with the epsilon-closure of the start state
        let mut current_state = self.start_state;
        for symbol in input {
            match self.alphabet_index_map.get(symbol) {
                Some(&symbol_index) => {
                    match self._next_state(current_state, symbol_index) {
                        Some(next_state) => {
                            current_state = next_state;
                        }
                        None => {
                            // If there is no transition for the symbol, return false
                            return false;
                        }
                    }
                }
                None => {
                    // If the symbol is not in the alphabet, return false
                    return false;
                }
            }
        }
        // Check if the current state is an accept state
        self.accept_states.contains(&current_state)
    }
}

impl Display for DFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DFA {{\n")?;
        writeln!(f, "  Start State: {:?}\n", self.start_state)?;
        writeln!(f, "  Accept States: {:?}\n", self.get_accept_states())?;
        writeln!(f, "  States: {:?}\n", self.states)?;
        writeln!(f, "  Alphabet: {:?}\n", self.alphabet)?;
        writeln!(f, "  Transitions:\n")?;
        for transition in self.get_transitions() {
            writeln!(
                f,
                "    {} -- {} --> {}\n",
                transition.0.display(),
                transition.1.display(),
                transition.2.display()
            )?;
        }
        writeln!(f, "}}")
    }
}
