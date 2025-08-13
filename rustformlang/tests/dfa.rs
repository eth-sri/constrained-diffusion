use hashbrown::HashMap;

use rustformlang::fa::epsilon_nfa::ENFA;
use rustformlang::fa::finite_automaton::FiniteAutomaton;
use rustformlang::input_symbol::InputSymbol;
use rustformlang::{language::Language, regex::regex::regex_to_dfa};

#[test]
fn test_complement() {
    let dfa = regex_to_dfa("a*b*a");
    let complement_dfa = dfa.complement();

    // Test the complement DFA
    assert!(complement_dfa.accepts_string("ab"));
    assert!(!complement_dfa.accepts_string("ba"));
    assert!(!complement_dfa.accepts_string("aaabbbba"));
    assert!(!complement_dfa.accepts_string("a"));
    assert!(!complement_dfa.accepts_string("aaaaaaaaaa"));
    assert!(complement_dfa.accepts_string(""));
    assert!(complement_dfa.accepts_string("aabaaab"));
}

#[test]
fn test_intersection() {
    let dfa1 = regex_to_dfa("a*b*");
    let dfa2 = regex_to_dfa("b*a*b");

    let intersection_dfa = dfa1.intersection(&dfa2);

    // Test the intersection DFA
    assert!(intersection_dfa.accepts_string("ab"));
    assert!(intersection_dfa.accepts_string("aaab"));
    assert!(intersection_dfa.accepts_string("bb"));

    assert!(!intersection_dfa.accepts_string("aaabbbb"));
    assert!(!intersection_dfa.accepts_string("ba"));
    assert!(!intersection_dfa.accepts_string("aaabbbba"));
    assert!(!intersection_dfa.accepts_string("a"));
    assert!(!intersection_dfa.accepts_string("aaaaaaaaaa"));
    assert!(!intersection_dfa.accepts_string(""));
}

#[test]
fn test_union() {
    let mut union_dfa = regex_to_dfa("").to_epsilon_automaton();
    for word in [
        "(var|let|const)",
        "while",
        "true",
        "false",
        "if",
        "else",
        "for",
        "do",
        "switch",
        "case",
    ]
    .iter()
    {
        let dfa = regex_to_dfa(format!(r#"\x02{}\x03"#, word).as_str());
        union_dfa = union_dfa.union(&dfa.to_epsilon_automaton());
        if word.to_string() != "(var|let|const)".to_string() {
            assert!(
                union_dfa.accepts_string(format!("\x02{}\x03", word).as_str()),
                "DFA should accept the string '{}'",
                format!(r#"\x02{}\x03"#, word).as_str()
            );
        }
    }
    let union_dfa_min = union_dfa.minimize();
    print!("Union DFA: {}", union_dfa_min);

    assert!(union_dfa.accepts_string("\x02var\x03"));
    assert!(union_dfa.accepts_string("\x02let\x03"));
    assert!(union_dfa_min.accepts_string("\x02var\x03"));
    assert!(union_dfa_min.accepts_string("\x02let\x03"));

    for word in [
        "(var|let|const)",
        "while",
        "true",
        "false",
        "if",
        "else",
        "for",
        "do",
        "switch",
        "case",
    ]
    .iter()
    {
        if word.to_string() != "(var|let|const)".to_string() {
            assert!(
                union_dfa_min.accepts_string(format!("\x02{}\x03", word).as_str()),
                "MIN DFA should accept the string '{}'",
                format!(r#"\x02{}\x03"#, word).as_str()
            );
        }
    }
}

#[test]
fn test_difference() {
    let dfa1 = regex_to_dfa("a*b*");
    let dfa2 = regex_to_dfa("b*a*b");

    let difference_dfa = dfa1.difference(&dfa2);

    // Test the difference DFA
    assert!(!difference_dfa.accepts_string("ab"));
    assert!(!difference_dfa.accepts_string("aaab"));
    assert!(!difference_dfa.accepts_string("bb"));
    assert!(difference_dfa.accepts_string("abb"));

    assert!(difference_dfa.accepts_string("aaabbbb"));
    assert!(!difference_dfa.accepts_string("ba"));
    assert!(!difference_dfa.accepts_string("aaabbbba"));
    assert!(difference_dfa.accepts_string("a"));
    assert!(difference_dfa.accepts_string("aaaaaaaaaa"));
    assert!(difference_dfa.accepts_string(""));
}

#[test]
fn test_union_diff() {
    let mut union_dfa = regex_to_dfa("").to_epsilon_automaton();
    for word in [
        "(var|let|const)",
        "while",
        "true",
        "false",
        "if",
        "else",
        "for",
        "do",
        "switch",
        "case",
    ]
    .iter()
    {
        let dfa = regex_to_dfa(format!(r#"%{}%"#, word).as_str());
        union_dfa = union_dfa.union(&dfa.to_epsilon_automaton());
        if !(word.to_string() == "(var|let|const)".to_string()) {
            assert!(union_dfa.accepts_string(format!(r#"%{}%"#, word).as_str()));
        }
    }
    let union_dfa_min = union_dfa.minimize();
    print!("Union DFA: {}", union_dfa_min);
    let identifier_dfa = regex_to_dfa(r#"%[a-zA-Z_]\w*%"#);
    print!("Identifier DFA: {}", identifier_dfa);
    let diff_dfa = identifier_dfa
        .to_deterministic()
        .difference(&union_dfa_min)
        .to_bytes_dfa();
    print!("Difference DFA: {}", diff_dfa);
    let diff_dfa_min = diff_dfa.minimize();
    print!("Minimized Difference DFA: {}", diff_dfa_min);
    assert!(diff_dfa_min.accepts_string("%my_variable%"));
    assert!(diff_dfa_min.accepts_string("%myVar%"));
    assert!(diff_dfa_min.accepts_string("%myVar123%"));
    assert!(diff_dfa_min.accepts_string("%myVar_123%"));
    assert!(diff_dfa_min.accepts_string("%myVar123_%"));

    assert!(!diff_dfa_min.accepts_string("%123myVar%"));

    assert!(!diff_dfa_min.accepts_string("%var%"));
    assert!(!diff_dfa_min.accepts_string("%let%"));
    assert!(!diff_dfa_min.accepts_string("%for%"));
    assert!(!diff_dfa_min.accepts_string("%while%"));
    assert!(!diff_dfa_min.accepts_string("%if%"));
    assert!(!diff_dfa_min.accepts_string("%else%"));
    assert!(!diff_dfa_min.accepts_string("%true%"));
    assert!(!diff_dfa_min.accepts_string("%false%"));
    assert!(!diff_dfa_min.accepts_string("%switch%"));
    assert!(!diff_dfa_min.accepts_string("%case%"));
}

#[test]
fn test_n_prefix_language() {
    let dfa = regex_to_dfa(r"a{3}b{5}").n_prefix_language(3);
    assert!(dfa.accepts_string("aaabb"));
    assert!(!dfa.accepts_string("aabbb"));
    assert!(!dfa.accepts_string("abbb"));
    assert!(!dfa.accepts_string("aaabbbb"));
    assert!(dfa.accepts_string("a"));
    assert!(dfa.accepts_string("aaab"));

    let dfa = regex_to_dfa(r"abc+").n_prefix_language(4);
    assert!(dfa.accepts_string("abcccc"));
    assert!(dfa.accepts_string("abcc"));
    assert!(!dfa.accepts_string("bc"));

    let dfa = regex_to_dfa(r"abce{4}").n_prefix_language(0);
    assert!(dfa.accepts_string("abceeee"));
    assert!(dfa.accepts_string("abceee"));
    assert!(dfa.accepts_string("ab"));
    assert!(!dfa.accepts_string("abceeeee"));
}

#[test]
fn test_accept_prefix() {
    let dfa = regex_to_dfa(r"abc+");
    assert_eq!(dfa.accept_prefix_string("abcc"), Some(4));
    assert_eq!(dfa.accept_prefix_string("abc"), Some(3));
    assert_eq!(dfa.accept_prefix_string(""), None);
    assert_eq!(dfa.accept_prefix_string("abccb"), Some(4));
    assert_eq!(dfa.accept_prefix_string("ab"), None);
    assert_eq!(dfa.accept_prefix_string("abcdef"), Some(3));
    assert_eq!(dfa.accept_prefix_string("abdef"), None);
    assert_eq!(dfa.accept_prefix_string("def"), None);
}

#[test]
fn test_prefix_diff() {
    let dfa1 = regex_to_dfa(r"[rnbqkbnrpRNBQKBNRP]")
        .true_prefix_language()
        .minimize();
    let big_union = ENFA::empty();
    let dfa2 = regex_to_dfa(r"b").true_prefix_language().minimize();
    let big_union = big_union.union(&dfa2.to_epsilon_automaton());
    let big_union = big_union.minimize();
    let diff_dfa = dfa1
        .to_deterministic()
        .difference(&big_union)
        .minimize()
        .to_bytes_dfa();
    assert!(dfa2.accepts_string(""));
    assert!(!diff_dfa.accepts_string(""));
    assert!(!diff_dfa.accepts_string("b"));
    assert!(!diff_dfa.accepts_string("R"));
}

#[test]
fn test_example_word() {
    let dfa1 = regex_to_dfa("ba*b*");
    let dfa2 = regex_to_dfa("b*a*b");

    let difference_dfa = dfa1.difference(&dfa2);

    let example_word = difference_dfa.example_word().unwrap();
    let example_word = String::from_utf8(example_word.clone()).unwrap();
    print!("{}", example_word);
    assert!(difference_dfa.accepts_string(example_word.as_str()));
    assert!(dfa1.accepts_string(example_word.as_str()));
    assert!(!dfa2.accepts_string(example_word.as_str()));
}

#[test]
fn test_concat() {
    let dfa1 = regex_to_dfa("ba*").concat(&regex_to_dfa("(a+b|b+)"));

    assert!(!dfa1.accepts_string("baaaaaa"));
    assert!(dfa1.accepts_string("baaaaaabb"));
    assert!(dfa1.accepts_string("baaaaaab"));
}

#[test]
fn test_concat_2() {
    let mut dfa1 = regex_to_dfa(r"\s*").to_epsilon_automaton();
    dfa1.concat(&regex_to_dfa("#").to_epsilon_automaton());
    dfa1.concat(&regex_to_dfa(r"\s*").to_epsilon_automaton());
    dfa1.concat(&regex_to_dfa(r"\x02include\x03").to_epsilon_automaton());
    print!("{}", dfa1.to_graphviz());

    assert!(dfa1.accepts_string("# \x02include\x03"));
    assert!(!dfa1.accepts_string("#"))
}

#[test]
fn test_intersection_example_word() {
    let dfa1 = regex_to_dfa("a*b*");
    let dfa2 = regex_to_dfa("b*a*b");

    // Test the intersection DFA
    let word = dfa1.intersection_example_word(&dfa2);
    assert!(word.is_some());
    let word = word.unwrap();
    assert!(dfa1.accepts_string(&String::from_utf8(word.clone()).unwrap()));
    assert!(dfa2.accepts_string(&String::from_utf8(word.clone()).unwrap()));
}
