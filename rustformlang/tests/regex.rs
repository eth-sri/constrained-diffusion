use rustformlang::language::Language;
use rustformlang::regex::regex::regex_to_dfa;

#[test]
fn test_our_regex() {
    let dfa = regex_to_dfa(r"\d{4}-\d{2}-\d{2}");
    assert!(!dfa.accepts_string("My birthday is 1986-08-22!"));
    assert!(dfa.accepts_string("1986-08-22"));
    assert!(!dfa.accepts_string("No valid date here!")); // No match expected
    assert!(!dfa.accepts_string("1986/08/22")); // Partial match format (incorrect separator)
    assert!(!dfa.accepts_string("19860822")); // Missing separators
    assert!(dfa.accepts_string("1986-08-22")); // Match is valid
    assert!(!dfa.accepts_string("More dates: 1986-08-22 and 1990-01-01")); // DFA doesn't handle sub-matches
    assert!(!dfa.accepts_string("")); // No match for empty input
    assert!(!dfa.accepts_string(" 1986-08-22 ")); // DFA should not ignore leading/trailing whitespace
    assert!(dfa.accepts_string("1986-08-22")); // Exact match
    assert!(!dfa.accepts_string("abcd-ef-gh")); // Entirely invalid
    assert!(!dfa.accepts_string("1234-ab-56")); // Letters in date
}

#[test]
fn test_regex_to_dfa_custom_format() {
    // Test for different date patterns
    let dfa1 = regex_to_dfa(r"\d{2}-\d{2}-\d{4}");
    let dfa2 = regex_to_dfa(r"\d{4}/\d{2}/\d{2}");
    assert!(dfa1.accepts_string("08-22-1986")); // Matches MM-DD-YYYY
    assert!(dfa2.accepts_string("1986/08/22")); // Matches YYYY/MM/DD
    assert!(!dfa2.accepts_string("1986-08-22")); // Incorrect separator for this DFA
}

#[test]
fn test_regex_to_dfa_edge_cases() {
    // Test for edge cases in regex patterns
    let dfa_empty = regex_to_dfa(r""); // Regex that matches an empty string
    assert!(dfa_empty.accepts_string("")); // Should accept empty string
    assert!(!dfa_empty.accepts_string("not empty"));

    let dfa_none = regex_to_dfa(r"\d");
    assert!(dfa_none.accepts_string("1"));
}

#[test]
fn test_regex_repetition_and_alternation() {
    // Regex with repetition (*, +, {min,max}) and alternation (|)
    let dfa = regex_to_dfa(r"(cat|dog)+");
    assert!(dfa.accepts_string("cat")); // Single match
    assert!(dfa.accepts_string("dog")); // Single match
    assert!(dfa.accepts_string("catdog")); // Multiple matches concatenated
    assert!(dfa.accepts_string("dogcatdog")); // Repeated pattern
    assert!(!dfa.accepts_string("rat")); // Invalid word
    assert!(!dfa.accepts_string("catdograt")); // Partially invalid sequence
}

#[test]
fn test_regex_with_anchors() {
    let dfa = regex_to_dfa(r"[A-Z]\d{3}[a-z]");
    assert!(dfa.accepts_string("A123b")); // Valid match
    assert!(!dfa.accepts_string("A12b")); // Fewer digits
    assert!(!dfa.accepts_string("a123b")); // Lowercase start
    assert!(!dfa.accepts_string("A123B")); // Uppercase end
    assert!(!dfa.accepts_string("A123bX")); // Extra character
}

#[test]
fn test_regex_with_negation() {
    let dfa = regex_to_dfa(r"[^A-Z]\d{3}[^']");
    assert!(dfa.accepts_string("a123b")); // Valid match
    assert!(!dfa.accepts_string("A123b")); // Uppercase start
    assert!(!dfa.accepts_string("a123'")); // Invalid end
    assert!(!dfa.accepts_string("a123")); // Missing end character
    assert!(!dfa.accepts_string("A123'")); // Invalid start and end
}

#[test]
fn test_regex_escape_sequences() {
    // Regex with escape sequences
    let dfa = regex_to_dfa(r#"\w+\s\w+"#); // Matches two words separated by a space
    assert!(dfa.accepts_string("A B")); // Minimal match
    assert!(dfa.accepts_string("Rust programming")); // Another valid match
    assert!(dfa.accepts_string("Hello world")); // Simple match
    assert!(!dfa.accepts_string("SingleWord")); // No space
    assert!(!dfa.accepts_string("Hello_world")); // Underscore, no space
}

#[test]
fn test_regex_w() {
    // Regex with escape sequences
    let dfa = regex_to_dfa(r#"w+ w+"#); // Matches two words separated by a space
    assert!(dfa.accepts_string("w w")); // Minimal match
    assert!(dfa.accepts_string("wwww wwwwwwwwwww")); // Another valid match
    assert!(dfa.accepts_string("wwwww wwwww")); // Simple match
    assert!(!dfa.accepts_string("wwwwwwwwww")); // No space
    assert!(!dfa.accepts_string("wwwwwwwwwww")); // Underscore, no space
}

#[test]
fn test_regex_nested_groups_and_quantifiers() {
    // Regex with nested groups
    let dfa = regex_to_dfa(r"(a(b(c(d)?)?)?)+");
    assert!(dfa.accepts_string("abcd")); // Full nested group
    assert!(dfa.accepts_string("abc")); // Partial nested group
    assert!(dfa.accepts_string("ab")); // Only part of the group
    assert!(dfa.accepts_string("a")); // Minimal match
    assert!(!dfa.accepts_string("bcd")); // Missing leading `a`
    assert!(dfa.accepts_string("abcabcabc")); // Repeated valid sequences
}

#[test]
fn test_regex_with_complex_patterns() {
    // Regex with lookalike complex pattern
    let dfa = regex_to_dfa(r"(\d{3}-)?\d{3}-\d{4}");
    assert!(dfa.accepts_string("123-456-7890")); // With area code
    assert!(dfa.accepts_string("456-7890")); // Without area code
    assert!(!dfa.accepts_string("123-45-6789")); // Incorrect grouping
    assert!(dfa.accepts_string("999-9999")); // Minimal valid
    assert!(!dfa.accepts_string("")); // No match for empty

    // Edge case: Input ends with valid portion
    assert!(!dfa.accepts_string("Call me at 123-456-7890")); // DFA doesn't match substrings
}

#[test]
fn test_regex_string() {
    // Regex with lookalike complex pattern
    let dfa = regex_to_dfa("\".*\"");
    assert!(dfa.accepts_string("\"Hello, World!\"")); // Valid string
    assert!(dfa.accepts_string("\"\"")); // Empty string
    assert!(!dfa.accepts_string("Hello, World!")); // No quotes
    assert!(!dfa.accepts_string("\"Hello, World!")); // Missing closing quote
}

#[test]
fn test_ateoi() {
    // This has a special AtEoi case
    // one a transitions into a state that is AtEoi, should only accept if the transition into it is at EOI
    let dfa = regex_to_dfa(r"a*b*a");
    assert!(dfa.accepts_string("aba"));
    assert!(dfa.accepts_string("aaaba"));
    assert!(!dfa.accepts_string("aab"));
    assert!(dfa.accepts_string("aaaa"));
}
#[test]
fn test_ateoi2() {
    // This has a special AtEoi case
    // one a transitions into a state that is AtEoi, should only accept if the transition into it is at EOI
    let dfa = regex_to_dfa(r"a*b+a");
    assert!(dfa.accepts_string("aba"));
    assert!(dfa.accepts_string("aaaba"));
    assert!(!dfa.accepts_string("aab"));
    assert!(!dfa.accepts_string("aaaa"));
}

#[test]
fn test_unicode() {
    let dfa = regex_to_dfa(r"[^\n\r]+");
    assert!(dfa.accepts_string("Hello, World!")); // Valid string
    assert!(dfa.accepts_string("こんにちは")); // Valid Unicode string
    assert!(dfa.accepts_string("Friday: Sunny, high 75°F, low 50°F.")); // Valid Unicode string
}

#[test]
fn test_dollar() {
    let dfa = regex_to_dfa(r"\$hello");
    assert!(dfa.accepts_string("$hello")); // Valid string
    assert!(!dfa.accepts_string("hello")); // misses the dollar (which can not be substituted by epsilon)
}
