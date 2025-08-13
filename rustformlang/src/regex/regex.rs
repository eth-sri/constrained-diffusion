use hashbrown::{HashMap, HashSet};
use std::vec;

use crate::fa::{bytes_dfa::BytesDFA, state::State};
use regex_dfa::nfa::{Accept, Nfa};

pub fn regex_to_dfa(regex: &str) -> BytesDFA {
    let anchored = match regex {
        "" => "^$".to_string(),
        _ => format!("^({regex})$"),
    };
    let nfa = Nfa::from_regex(anchored.as_str()).expect("Failed to create NFA from regex");
    let nfa = nfa.remove_looks();

    if nfa.is_empty() {
        return BytesDFA::empty();
    }
    if !nfa.is_anchored() {
        panic!("Regex is not anchored");
    }

    let max_states: usize = std::usize::MAX;
    let nfa = nfa
        .byte_me(max_states)
        .expect("Failed to convert NFA to byte NFA");

    let dfa = nfa
        .determinize(max_states)
        .expect("Failed to determinize")
        .optimize();

    let dfa_start_state = dfa
        .init_at_start()
        .expect(format!("Failed to get start state").as_str());

    let mut dfa_states: Vec<State> = (0..dfa.states.len())
        .map(|i| State {
            name: format!("q{i}"),
        })
        .collect();
    let mut dfa_accept_states: HashSet<usize> = HashSet::new();
    let mut dfa_transitions: Vec<HashMap<u8, usize>> = vec![HashMap::new(); dfa.states.len()];
    for (i, state) in dfa.states.iter().enumerate() {
        let state_trans = dfa_transitions.get_mut(i).unwrap();
        // we need to treat AtEoi as a special case
        // These are only accept iff the transition into them is at EOI
        // we translate this back into standard DFAs by adding a new state
        // in which self-edges transition, and make it not-accepting
        for (symbol, next_state) in state.transitions.keys_values() {
            state_trans.insert(symbol, *next_state);
        }
        if state.accept == Accept::Never {
            continue;
        }
        dfa_accept_states.insert(i);
    }

    let dfa_state_index_map: HashMap<State, usize> = HashMap::from_iter(
        dfa_states
            .iter()
            .enumerate()
            .map(|(i, state)| (state.clone(), i)),
    );
    BytesDFA::new(
        dfa_state_index_map,
        dfa_states,
        dfa_transitions,
        dfa_start_state,
        dfa_accept_states,
    )
}
