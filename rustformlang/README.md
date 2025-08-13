# Rustformlang

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A high-performance Rust implementation of formal language theory constructs, focusing on Context-Free Grammars (CFGs) and their intersection with regular languages. This library is inspired by and partly a port of [pyformlang](https://pyformlang.readthedocs.io/en/latest/), with somewhat similar APIs and heavily optimized CFG/DFA intersection operations.

## Features

### Context-Free Grammars (CFG)
- **CFG-DFA Intersection**: Efficient computation of the intersection between context-free and regular languages
- **Grammar Normalization**: Conversion to Chomsky Normal Form and other canonical forms
- **Grammar Reduction**: Removal of useless symbols, epsilon productions, and unit productions and further heuristics for language size reduction.

### Finite Automata
- **Deterministic Finite Automata (DFA), Non-deterministic Finite Automata (NFA) and epsilon-NFAs**
- **Automata Operations**: Union, intersection, complement, minimization, and determinization
- **Bytes DFA**: Specialized DFA implementation where the alphabet is the range of bytes (0-255)

### Regular Expressions
- **Regex to DFA Conversion**: Direct compilation of regular expressions to DFAs
- **Pattern Matching**: Efficient string matching using compiled automata

### Advanced Features
- **Lexical Analysis**: Sophisticated tokenization with configurable lexing maps
- **Grammar Substitution**: Replace terminals with other grammars for compositional language construction
- **Example Word Generation**: Generate example strings from grammars and automata
- **Performance Optimizations**: Parallel processing, caching, and optimized data structures

## Installation

Copy this directory into your project.
Add this to your `Cargo.toml`:

```toml
[dependencies]
rustformlang = { path = "./rustformlang" }
```

## Quick Start

### Creating a Context-Free Grammar

```rust
use rustformlang::cfg::{cfg::CFG, variable::Variable};

let grammar_text = r#"
    S -> A B | C
    A -> a
    B -> b  
    C -> c
"#;

let cfg = CFG::from_text(grammar_text, Variable::new("S"));
```

### Manually Create Finite Automata

```rust
use rustformlang::fa::{dfa::DFA, state::State};
use rustformlang::input_symbol::InputSymbol;

let mut dfa = DFA::empty();
let start = State::new("q0");
let accept = State::new("q1");

dfa.set_start_state(start.clone());
dfa.add_accept_state(accept.clone());
dfa.add_transition(&start, &InputSymbol::new("a"), &accept);

// Check if DFA accepts a string
let input = vec![InputSymbol::new("a")];
println!("Accepts 'a': {}", dfa.accepts(&input));
```

### Create Finite Automata from Regular Expressions

```rust
use rustformlang::regex::regex::regex_to_dfa;

// Create minimal DFA from regex pattern
let dfa = regex_to_dfa("a(b|c)*d");

// Check if string matches pattern
let input = "abcbcd";
println!("Matches pattern: {}", dfa.accepts_string(input));
```

Note that this creates a ByteDFA, i.e., a DFA specialized for the alphabet of 0-255.

### CFG-DFA Intersection

```rust
use rustformlang::regex::regex::regex_to_dfa;

let cfg_text = r#"
    S -> A B
    A -> a
    B -> b
"#;
let cfg = CFG::from_text(cfg_text, Variable::new("S"));

// Create DFA from regex
let dfa = regex_to_dfa("(ab)*").to_deterministic();

// Compute intersection and check emptiness (fast!)
let intersection_empty = cfg.is_intersection_empty(&dfa, None);
println!("Intersection is empty: {}", intersection_empty);

// Compute intersection and check emptiness (expensive!)
let intersection = cfg.intersection(&dfa);
println!("Intersection is empty: {}", intersection.is_empty());
```

### Lexical Analysis

```rust
use rustformlang::constraining::{LexMap, StringOrDFA, compile_lex_map, lex, Token};
use rustformlang::input_symbol::InputSymbol;
use std::collections::HashMap;

// Define lexing rules
let mut lex_map = LexMap::new();
lex_map.insert(InputSymbol::new("NUMBER"), StringOrDFA::String(r"\d+".to_string()));
lex_map.insert(InputSymbol::new("WORD"), StringOrDFA::String(r"\w+".to_string()));

let compiled_map = compile_lex_map(lex_map, &HashMap::new());

// Tokenize input
let token = Token::Word("hello123".to_string());
let lexings = lex(&token, &compiled_map, true, &None);
```

## Grammar Operations

### Normalization and Cleaning

```rust
// Clean grammar (remove useless symbols, epsilon productions, unit productions)
let clean_cfg = cfg.cleaned();

// Convert to C2F+ε (automatically applies cleaning)
let normal_cfg = cfg.to_normal_form();


// Check properties
println!("Generates epsilon: {}", cfg.generates_epsilon());
println!("Is in normal form: {}", cfg.is_normal_form());
println!("Is empty: {}", cfg.is_empty());
```

### Grammar Composition

```rust
use std::collections::HashMap;

// Substitute terminals with other grammars
let substitution_grammar = CFG::from_text("S -> x y", Variable::new("S"));
let substitutions = HashMap::from([(Terminal::new("a"), &substitution_grammar)]);
let substituted_cfg = cfg.substitute(&substitutions);

// Concatenate grammars
let cfg2 = CFG::from_text("S -> d", Variable::new("S"));
let concatenated = CFG::concat(vec![&cfg, &cfg2]);
```

## Automata Operations

### DFA Operations

```rust
// Minimize DFA
let minimized = dfa.minimize();

// Complement
let complement = dfa.complement();

// Intersection of two DFAs
let intersection = dfa1.intersection(&dfa2);

// Check emptiness
println!("Is empty: {}", dfa.is_empty());

// Generate example word
if let Some(word) = dfa.example_word() {
    println!("Example word: {:?}", word);
}
```

### Language Operations

```rust
// Prefix languages
let prefix_lang = dfa.prefix_language();
let true_prefix_lang = dfa.true_prefix_language();

// Accept prefix of input
let input = "hello world";
if let Some(prefix_len) = dfa.accept_prefix_string(input) {
    println!("Accepted prefix length: {}", prefix_len);
}
```

## Performance Features

- **Parallel Processing**: Lexical analysis uses Rayon for parallel computation
- **Caching**: Intelligent caching of lexing results and grammar computations
- **Optimized Data Structures**: Uses `hashbrown::HashMap` and `smallvec::SmallVec` for performance
- **Memory Efficient**: Careful memory management with reference counting and lazy evaluation

## Advanced Usage

### Custom Lexing with Preprocessing

```rust
use rustformlang::constraining::{prelex_word, all_prelex_words};

// Apply preprocessing to words before lexing
let prelexed = prelex_word("hello", "[]", true, false);
let all_lexings = all_prelex_words("hello", Some("[]"), &compiled_map, Some(" \t\n"));
```

### Grammar Analysis

```rust
// Get generating and nullable symbols
let generating = cfg.get_generating_symbols();
let nullable = cfg.get_nullable_symbols();
let reachable = cfg.get_reachable_symbols();

// Get production rules
let productions = cfg.get_productions();
for production in productions {
    println!("{}", production);
}
```

### Intersection with Timeout

```rust
use std::time::Duration;

// Check intersection emptiness with timeout
let timeout = Duration::from_secs(10);
let is_empty = cfg.is_intersection_empty(&dfa, Some(timeout));

// Generate example word with timeout
let example = cfg.example_word(&dfa, Some(timeout));
```

## Benchmarks

The library includes a JavaScript grammar benchmark that demonstrates performance on real-world grammars:

```bash
cargo run --bin js_grammar_bench --release
```

## Testing

Run the test suite:

```bash
cargo test
```

## Architecture

The library is organized into several key modules:

- **`cfg`**: Context-free grammar implementation
- **`fa`**: Finite automata (DFA, NFA, epsilon-NFA)
- **`regex`**: Regular expression to automata conversion
- **`constraining`**: Lexical analysis and tokenization
- **`input_symbol`**: Symbol representation and utilities
- **`language`**: Language trait definitions

## Contributing

Contributions are welcome! Please ensure all tests pass and follow the existing code style.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This library is inspired by and compatible with [pyformlang](https://github.com/Aunsiels/pyformlang).
