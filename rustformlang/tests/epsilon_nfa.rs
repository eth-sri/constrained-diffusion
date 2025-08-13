use hashbrown::HashSet;
use rustformlang::fa::epsilon_nfa::ENFA;
use rustformlang::fa::state::State;
use rustformlang::input_symbol::{epsilon, InputSymbol};
use rustformlang::language::Language;

#[test]
fn test_empty_enfa() {
    let enfa = ENFA::empty();

    // Ensuring the ENFA is empty
    assert_eq!(
        enfa.states.len(),
        1,
        "Empty ENFA should have one state only"
    );
    assert_eq!(enfa.alphabet.len(), 0, "Empty ENFA should no symbol");
    assert!(enfa.is_empty(), "Empty ENFA should have no accept states");

    assert!(
        !enfa.accepts(&vec![]),
        "Empty ENFA should not accept the empty string"
    );
}

#[test]
fn test_add_states_and_transitions() {
    let mut enfa = ENFA::empty();

    let state1 = State::new("q1");
    let state2 = State::new("q2");

    let symbol_a = InputSymbol::new("a");

    // Adding start state
    enfa.set_start_state(state1.clone());
    assert_eq!(
        enfa.states.len(),
        2,
        "ENFA should have exactly two states after adding start state"
    );

    // Adding accept state
    enfa.add_accept_state(state2.clone());
    assert_eq!(
        enfa.states.len(),
        3,
        "ENFA should have three states after adding accept state"
    );
    //assert_eq!(enfa.accept_states.len(), 1, "There should be one accept state in ENFA");

    // Adding a transition
    enfa.add_transition(&state1, &symbol_a, &state2);
    //assert_eq!(enfa.transitions.len(), 1, "There should be one transition");

    // // Validate transition
    // let from_index = enfa.state_index_map[&state1];
    // let to_index = enfa.state_index_map[&state2];
    // let symbol_index = enfa.alphabet_index_map[&symbol_a];
    // let transition_set = enfa.transitions[from_index].get(&symbol_index).unwrap();
    // assert!(transition_set.contains(&to_index), "Transition set should contain the target state");
}

#[test]
fn test_epsilon_closure_single_state() {
    let mut enfa = ENFA::empty();

    let state1 = State::new("q1");
    let state2 = State::new("q2");

    // Adding states and epsilon transition
    enfa.set_start_state(state1.clone());
    enfa.add_transition(&state1, &epsilon(), &state2);

    let closure = enfa.epsilon_closure(&HashSet::from_iter(vec![state1.clone()]));
    assert!(
        closure.contains(&state1),
        "Epsilon closure must include the original state"
    );
    assert!(
        closure.contains(&state2),
        "Epsilon closure must include reachable states via epsilon"
    );
    assert_eq!(
        closure.len(),
        2,
        "Epsilon closure should include two states"
    );
}

#[test]
fn test_epsilon_closure_multiple_states() {
    let mut enfa = ENFA::empty();

    let state1 = State::new("q1");
    let state2 = State::new("q2");
    let state3 = State::new("q3");

    // Adding states and epsilon transitions
    enfa.set_start_state(state1.clone());
    enfa.add_transition(&state1, &epsilon(), &state2);
    enfa.add_transition(&state2, &epsilon(), &state3);

    let closure = enfa.epsilon_closure(&HashSet::from_iter(vec![state1.clone(), state2.clone()]));
    assert!(
        closure.contains(&state1),
        "Epsilon closure must include the original state"
    );
    assert!(
        closure.contains(&state2),
        "Epsilon closure must include directly reachable states"
    );
    assert!(
        closure.contains(&state3),
        "Epsilon closure must include indirectly reachable states"
    );
    assert_eq!(
        closure.len(),
        3,
        "Epsilon closure should include all reachable states"
    );
}

#[test]
fn test_remove_epsilon_transitions() {
    let mut enfa = ENFA::empty();

    let state1 = State::new("q1");
    let state2 = State::new("q2");
    let state3 = State::new("q3");

    let symbol_a = InputSymbol::new("a");
    let symbol_b = InputSymbol::new("b");

    // Adding states and transitions
    enfa.set_start_state(state1.clone());
    enfa.add_accept_state(state3.clone());
    enfa.add_transition(&state1, &epsilon(), &state2);
    enfa.add_transition(&state2, &symbol_a, &state3);
    enfa.add_transition(&state1, &symbol_b, &state3);

    assert!(!enfa.is_empty(), "ENFA should not be empty");

    // Removing epsilon transitions
    let nfa = enfa.remove_epsilon_transitions();

    // The resulting NFA should have the same states
    assert_eq!(
        nfa.states.len(),
        enfa.states.len(),
        "NFA should have the same number of states as the original ENFA"
    );

    // The resulting NFA should have the same accepting string
    assert!(
        nfa.accepts(&vec![symbol_a.clone()]),
        "NFA should accept the string 'a'"
    );
    assert!(
        nfa.accepts(&vec![symbol_b.clone()]),
        "NFA should accept the string 'a'"
    );
    assert!(
        !nfa.accepts(&vec![symbol_a.clone(), symbol_a.clone()]),
        "DFA should not accept the string 'aa'"
    );

    // make deterministic
    let dfa = nfa.to_deterministic();
    print!("Determinized DFA: {}", dfa.to_string());
    assert!(
        dfa.accepts(&vec![symbol_a.clone()]),
        "DFA should accept the string 'a'"
    );
    assert!(
        dfa.accepts(&vec![symbol_b.clone()]),
        "DFA should accept the string 'b'"
    );
    assert!(
        !dfa.accepts(&vec![symbol_a.clone(), symbol_a.clone()]),
        "DFA should not accept the string 'aa'"
    );

    // minimize
    let min_dfa = dfa.minimize();
    print!("Minimized DFA: {}", min_dfa.to_string());
    assert!(
        min_dfa.accepts(&vec![symbol_a.clone()]),
        "Minimized DFA should accept the string 'a'"
    );
    assert!(
        min_dfa.accepts(&vec![symbol_b.clone()]),
        "Minimized DFA should accept the string 'b'"
    );
    assert!(
        !min_dfa.accepts(&vec![symbol_a.clone(), symbol_a.clone()]),
        "Minimized DFA should not accept the string 'aa'"
    );
    assert!(!min_dfa.is_empty(), "Minimized DFA should not be empty");
}

#[test]
fn test_accepts_string() {
    let mut enfa = ENFA::empty();

    let state1 = State::new("q1");
    let state2 = State::new("q2");
    let state3 = State::new("q3");

    let symbol_a = InputSymbol::new("a");
    let symbol_b = InputSymbol::new("b");

    // Adding states and transitions
    enfa.set_start_state(state1.clone());
    enfa.add_transition(&state1, &epsilon(), &state2);
    enfa.add_transition(&state2, &symbol_a, &state3);
    enfa.add_accept_state(state3.clone());

    // Validate string acceptance
    assert!(
        enfa.accepts(&vec![symbol_a.clone()]),
        "ENFA should accept string 'a'"
    );
    assert!(
        !enfa.accepts(&vec![symbol_a.clone(), symbol_a.clone()]),
        "ENFA should not accept string 'aa'"
    );
    assert!(
        !enfa.accepts(&vec![symbol_b.clone()]),
        "ENFA should not accept string 'b'"
    );

    enfa.add_accept_state(state2.clone());
    assert!(
        enfa.accepts(&vec![]),
        "ENFA should accept empty string due to epsilon transitions leading to accept state"
    );
}

#[test]
fn test_n_suffix_language() {
    let mut enfa = ENFA::empty();

    let state1 = State::new("q1");
    let state2 = State::new("q2");
    let state3 = State::new("q3");

    let symbol_a = InputSymbol::new("a");
    let symbol_b = InputSymbol::new("b");

    // Adding states and transitions
    enfa.set_start_state(state1.clone());
    enfa.add_transition(&state1, &symbol_a, &state2);
    enfa.add_transition(&state2, &symbol_b, &state3);
    enfa.add_accept_state(state3.clone());

    let suffix_1 = enfa.n_suffix_language(1);

    // Validate suffix language
    assert!(
        suffix_1.accepts_string("b"),
        "ENFA should accept suffix 'b'"
    );
    assert!(
        !suffix_1.accepts_string("a"),
        "ENFA should not accept suffix 'a'"
    );
    assert!(
        !suffix_1.accepts_string("ab"),
        "ENFA should not accept suffix 'ab' (0-length prefix suffix)"
    );
    assert!(
        suffix_1.accepts_string(""),
        "ENFA should accept empty string as suffix"
    );

    let suffix_0 = enfa.n_suffix_language(0);

    assert!(
        suffix_0.accepts_string(""),
        "ENFA should accept empty string as suffix"
    );
    assert!(
        !suffix_0.accepts_string("a"),
        "ENFA should not accept 'a' as suffix"
    );
    assert!(
        suffix_0.accepts_string("b"),
        "ENFA should accept 'b' as suffix"
    );
    assert!(
        suffix_0.accepts_string("ab"),
        "ENFA should accept suffix 'ab' (0-length prefix suffix)"
    );
}
