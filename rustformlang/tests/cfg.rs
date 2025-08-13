use hashbrown::HashMap;
use rustformlang::cfg::cfg::CFG;
use rustformlang::cfg::production::{Production, Symbol};
use rustformlang::cfg::terminal::Terminal;
use rustformlang::cfg::variable::Variable;
use rustformlang::fa::finite_automaton::FiniteAutomaton;
use rustformlang::input_symbol::InputSymbol;
use rustformlang::language::{Language, MutLanguage};
use rustformlang::regex::regex::regex_to_dfa;

#[test]
fn test_cfg_to_text() {
    let start_variable = Variable::new("S");
    let a_variable = Variable::new("A");
    let b_variable = Variable::new("B");
    let terminal_a = Terminal::new("a");
    let terminal_b = Terminal::new("b");

    let production1 = Production::new(
        start_variable.clone(),
        vec![
            Symbol::T(terminal_a.clone()),
            Symbol::V(start_variable.clone()),
        ],
    );
    let production2 = Production::new(
        start_variable.clone(),
        vec![
            Symbol::V(b_variable.clone()),
            Symbol::V(a_variable.clone()),
            Symbol::T(terminal_b.clone()),
        ],
    );
    let production3 = Production::new(a_variable.clone(), vec![Symbol::T(terminal_a.clone())]);
    let production4 = Production::new(b_variable.clone(), vec![Symbol::T(terminal_b.clone())]);
    let production5 = Production::new(start_variable.clone(), vec![]);
    let productions = vec![
        production1,
        production2,
        production3,
        production4,
        production5,
    ];

    let cfg = CFG::from_start_and_productions(start_variable, productions);
    let expected_output = "Start Symbol: S\n\
S -> a S | B A b | ε \n\
B -> b \n\
A -> a \n\
Terminals: a b ";

    // assert_eq!(cfg.to_text(), expected_output);
}

#[test]
fn test_from_text() {
    let text = r#"
        S -> A | B
        A -> Bobo r
        B -> a
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    assert_eq!(cfg.productions.len(), 4); // Check number of variables
    assert_eq!(cfg.get_productions().len(), 4); // Check number of productions
    assert_eq!(cfg.terminals.len(), 2); // Check number of terminals
}

#[test]
fn test_from_text2() {
    let text = r#"
        S -> A B
        A -> a
        B -> b
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    // Check if the terminal sequence "a b" is derivable
    //assert!(cfg.contains(&[Terminal::new("a"), Terminal::new("b")]));
    assert_eq!(cfg.productions.len(), 3); // Check number of variables
    assert_eq!(cfg.terminals.len(), 2); // Check number of terminals
    assert_eq!(cfg.get_productions().len(), 3); // Check number of productions
    assert!(!cfg.generates_epsilon()); // Check if the CFG generates epsilon
}

#[test]
fn test_from_text3() {
    let text = r#"
        S ->  A  B
        A -> Bobo r
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    assert_eq!(cfg.productions.len(), 4); // Check number of variables
    assert_eq!(cfg.get_productions().len(), 2); // Check number of productions
    assert_eq!(cfg.terminals.len(), 1); // Check number of terminals
    assert!(!cfg.generates_epsilon()); // Check if the CFG generates epsilon
}

#[test]
fn test_from_text_union() {
    let text = r#"
        "VAR:S" -> "TER:a" | b
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    assert_eq!(cfg.get_productions().len(), 2); // Ensure there are 2 productions for "S -> a | b"
    assert!(!cfg.generates_epsilon()); // Check if the CFG generates epsilon
}

#[test]
fn test_epsilon() {
    let text = r#"
        S -> epsilon
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty
    assert_eq!(cfg.terminals.len(), 0); // No terminals present for epsilon
    assert!(cfg.generates_epsilon()); // Check if the CFG generates epsilon
}

#[test]
fn test_epsilon2() {
    let text = r#"
        S -> A B | a
        A -> $
        B -> b
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty
    assert!(!cfg.generates_epsilon()); // Check if the CFG generates epsilon
}

#[test]
fn test_epsilon3() {
    let text = r#"
        S -> B | a A | a
        A -> $
        B -> b | A
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty
    assert!(cfg.generates_epsilon()); // Check if the CFG generates epsilon
}

/// Test that cleaning preserves the CFG (modulo accepting epsilon)
#[test]
fn test_clean() {
    let text = r#"
        S -> A B | a
        A -> $
        B -> b
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty
    let mut cleaned_cfg = cfg.cleaned();
    assert!(!cleaned_cfg.generates_epsilon()); // Check if the cleaned CFG generates epsilon
    assert!(!cleaned_cfg.is_empty()); // Check if the generated CFG is empty
                                      // TODO add more checks using acceptance as soon as it is implemented
}

#[test]
fn test_clean2() {
    let text = r#"
        S -> A B | a | C A | A B A S
        A -> $
        B -> b
        A -> E
        Z -> u
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty
    let mut cleaned_cfg = cfg.cleaned();
    println!("Cleaned CFG: {}", cleaned_cfg.to_text());
    assert!(!cleaned_cfg.generates_epsilon()); // Check if the cleaned CFG generates epsilon
    assert!(!cleaned_cfg.is_empty()); // Check if the generated CFG is empty
    assert!(!cleaned_cfg.terminals.contains(&Terminal::new("u"))); // Check if the cleaned CFG contains terminal "u"
                                                                   // TODO add more checks using acceptance as soon as it is implemented
}

#[test]
fn test_clean3() {
    let text = r#"
        S -> A B | A B A S
        B -> b
        A -> $
        A -> c
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty
    let mut cleaned_cfg = cfg.cleaned();
    println!("Cleaned CFG: {}", cleaned_cfg.to_text());
    assert!(!cleaned_cfg.generates_epsilon()); // Check if the cleaned CFG generates epsilon
    assert!(!cleaned_cfg.is_empty()); // Check if the generated CFG is empty
                                      // TODO add more checks using acceptance as soon as it is implemented
}

#[test]
fn test_clean4() {
    let text = r#"
        S -> A B | A B A S
        A -> $
        A -> c
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    assert!(cfg.is_empty()); // Check if the generated CFG is empty
    println!("orig CFG: {}", cfg.to_text());
    let mut cleaned_cfg = cfg.cleaned();
    println!("Cleaned CFG: {}", cleaned_cfg.to_text());
    assert!(!cleaned_cfg.generates_epsilon()); // Check if the cleaned CFG generates epsilon
    assert!(cleaned_cfg.is_empty()); // Check if the generated CFG is empty
                                     // TODO add more checks using acceptance as soon as it is implemented
}

#[test]
fn test_to_cnf_simple() {
    // is already in CNF
    let text = r#"
        S -> A B | a
        A -> a
        B -> b
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    let mut cnf_cfg = cfg.to_normal_form(); // Assuming this converts to CNF
                                            // assert!(cfg.to_text() == cnf_cfg.to_text()); // Check if the CNF CFG is the same as the original CFG
    assert!(cfg.terminals.len() == 2); // Check number of terminals
                                       // visual inspection checks out, add asserts for contained word later
}

#[test]
fn test_to_cnf_with_epsilon() {
    let text = r#"
        S -> A B | epsilon | A B A S
        A -> a
        B -> b
        B -> $
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    // println!("orig CFG: {}", cfg.to_text());
    let mut cnf_cfg = cfg.to_normal_form(); // Assuming this converts to CNF

    // println!("CNF CFG: {}", cnf_cfg.to_text());
    // visual inspection checks out, add asserts for contained word later
}

#[test]
fn test_to_cnf_removal_of_useless_symbols() {
    let text = r#"
        S -> A B | C
        A -> a
        B -> b
        C -> D
        D -> E
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    // println!("orig CFG: {}", cfg.to_text());
    let mut cnf_cfg = cfg.to_normal_form();
    // println!("CNF CFG: {}", cnf_cfg.to_text());

    // Check that useless symbols are removed
    assert_eq!(cnf_cfg.productions.len(), 3); // Check number of variables
}

#[test]
fn test_to_cnf_large_grammar() {
    let text = r#"
        S -> A B C | D E
        A -> a
        B -> b
        C -> c
        D -> d
        E -> e
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    println!("orig CFG: {}", cfg.to_text());
    let mut cnf_cfg = cfg.to_normal_form(); // Assuming this converts to CNF

    println!("CNF CFG: {}", cnf_cfg.to_text());
}

#[test]
fn test_intersection() {
    let text1 = r#"
        S -> A B | C
        A -> a
        B -> b
        C -> c
        "#;
    let mut cfg = CFG::from_text(text1, Variable::new("S"));
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty

    for word in ["ab", "c"] {
        let mut dfa = regex_to_dfa(word).to_deterministic();
        assert!(!dfa.is_empty()); // Check if the generated DFA is empty

        let mut intersection_cfg = cfg.intersection(&mut dfa).cleaned();
        println!("Intersection CFG: {}", intersection_cfg.to_text());
        assert!(!intersection_cfg.is_empty()); // Check if the intersection CFG is empty
    }
}

#[test]
fn test_intersection2() {
    let text1 = r#"
        S -> A B | C | A B A S
        A -> a
        B -> b
        C -> c
        "#;
    let mut cfg = CFG::from_text(text1, Variable::new("S")).to_normal_form();
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty
    print!("orig CFG: {}", cfg.to_text());

    for (pattern, isempty) in [
        ("(aab)*", true),
        ("a*b*", false),
        ("a*b", false),
        ("abac", false),
        ("a*", true),
        ("aac", true),
    ] {
        let mut dfa = regex_to_dfa(pattern).to_deterministic();
        assert!(!dfa.is_empty()); // Check if the generated DFA is empty

        let mut intersection_cfg = cfg.intersection(&mut dfa).cleaned();
        println!("Intersection CFG: {}", intersection_cfg.to_text());
        assert!(
            intersection_cfg.is_empty() == isempty,
            "failed for {}",
            pattern
        ); // Check if the intersection CFG is empty
    }
}

#[test]
fn test_intersection_eps() {
    let text1 = r#"
        S -> A B | C
        A -> a
        B -> b
        C -> $
        "#;
    let mut cfg = CFG::from_text(text1, Variable::new("S"));
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty

    for word in [""] {
        let mut dfa = regex_to_dfa(word).to_deterministic();
        assert!(!dfa.is_empty()); // Check if the generated DFA is empty

        let mut intersection_cfg = cfg.intersection(&mut dfa).cleaned();
        println!("Intersection CFG: {}", intersection_cfg.to_text());
        assert!(!intersection_cfg.is_empty()); // Check if the intersection CFG is empty
    }
}

#[test]
fn test_accept() {
    let text = r#"
        S -> A B | C | A B A S
        A -> a
        B -> b
        B -> $
        C -> c
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    cfg = cfg.to_normal_form();
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty
    println!("CFG: {}", cfg.to_text());

    for word in ["ab", "c", "aac", "abaab", "aaab", "abac"] {
        assert!(
            cfg.accepts_string(word),
            "CFG should accept the word '{}', but it does not.",
            word
        ); // Check if the CFG accepts the word
    }
    for word in ["e", "abc", "abaca", "bb", "aa", "aab"] {
        assert!(
            !cfg.accepts_string(word),
            "CFG should not accept the word '{}', but it does.",
            word
        ); // Check if the CFG does not accept the word
    }
}

#[test]
fn test_substitution() {
    let text = r#"
        S -> A B | C
        A -> a
        B -> b
        C -> c
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty

    let text2 = r#"
        S -> c S c | e
        "#;

    let mut cfg2 = CFG::from_text(text2, Variable::new("S"));
    assert!(!cfg2.is_empty()); // Check if the generated CFG is empty

    let mut substituted_cfg = cfg.substitute(&HashMap::from([(Terminal::new("a"), &cfg2)]));
    println!("Substituted CFG: {}", substituted_cfg.to_text());
    assert!(!substituted_cfg.is_empty()); // Check if the substituted CFG is empty
    assert!(
        substituted_cfg.accepts_string("cecb"),
        "Substituted CFG should accept the word 'ccb', but it does not."
    );
    assert!(
        !substituted_cfg.accepts_string("a"),
        "Substituted CFG should not accept the word 'a', but it does."
    );
    assert!(
        !substituted_cfg.accepts_string("cb"),
        "Substituted CFG should not accept the word 'a', but it does."
    );
}

#[test]
fn test_concat() {
    let text = r#"
        S -> A B | C
        A -> a
        B -> b
        C -> c
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S"));
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty

    let text2 = r#"
        S -> c S c | e
        "#;

    let mut cfg2 = CFG::from_text(text2, Variable::new("S"));
    assert!(!cfg2.is_empty()); // Check if the generated CFG is empty

    let mut concatenated_cfg = CFG::concat(vec![&cfg, &cfg2]);
    println!("Concatenated CFG: {}", concatenated_cfg.to_text());
    assert!(!concatenated_cfg.is_empty()); // Check if the concatenated CFG is empty
    assert!(
        concatenated_cfg.accepts_string("abcec"),
        "Concatenated CFG should accept the word 'abcec', but it does not."
    );
    assert!(
        !concatenated_cfg.accepts_string("a"),
        "Concatenated CFG should not accept the word 'a', but it does."
    );
}
const JS_GRAMMAR: &str = r#"
S -> Source_elements | $

Source_elements -> Statement | Statement Source_elements

Statement -> Block
    | Variable_statement
    | Expression_statement
    | If_statement
    | Iteration_statement
    | Continue_statement
    | Break_statement
    | Return_statement
    | With_statement
    | Switch_statement
    | Throw_statement
    | Try_statement

Block -> { Statement_list }

Empty_braces -> { }

Statement_list -> Statement | Statement Statement_list

Variable_statement  -> variable_decl_type Variable_declaration_list EOS

Variable_declaration_list -> Variable_declaration | Variable_declaration_list , Variable_declaration

Variable_declaration  -> identifier Type_declaration Initialiser | identifier Type_declaration | identifier Initialiser | identifier

Type_declaration  -> : Type_identifier

Param_list -> identifier Type_declaration | Param_list , identifier Type_declaration
Identifier_list -> identifier | Identifier_list , identifier

Type_identifier  -> identifier | Type_identifier [ ]
    | ( Param_list ) Type_declaration | ( ) Type_declaration
    | [ Identifier_list ]

Initialiser -> = Single_expression

Expression_statement -> Expression_sequence EOS

If_statement -> if ( Expression_sequence ) Statement | if ( Expression_sequence ) Statement else Statement

Iteration_statement -> do Statement while ( Expression_sequence )
    | while ( Expression_sequence ) Statement
    | for ( Expression_sequence ; Expression_sequence ; Expression_sequence ) Statement
    | for ( variable_decl_type Variable_declaration_list ; Expression_sequence ; Expression_sequence ) Statement
    | for ( Single_expression in Expression_sequence ) Statement
    | for ( variable_decl_type Variable_declaration in Expression_sequence ) Statement 
    | for ( Single_expression of Expression_sequence ) Statement
    | for ( variable_decl_type Variable_declaration of Expression_sequence ) Statement 

Continue_statement  -> continue

Break_statement  -> break

Return_statement  -> return | return Expression_sequence

With_statement -> with ( Expression_sequence ) Statement

Switch_statement  -> switch ( Expression_sequence ) Case_block

Case_block  -> { Case_clauses Default_clause } | { Case_clauses }

Case_clauses -> Case_clause | Case_clauses Case_clause

Case_clause -> case Expression_sequence : Statement_list | case Expression_sequence :

Default_clause  -> default : Statement_list | default :

Throw_statement  -> throw Expression_sequence

Try_statement  -> try Block Catch_production | try Block Finally_production | try Block Catch_production Finally_production

Catch_production -> catch ( identifier ) Block

Finally_production -> finally Block

Function_declaration  -> function identifier ( OptFormal_parameter_list ) OptType_declaration { Function_body }
    | function ( ) { Function_body }
    | ( OptFormal_parameter_list ) => { Function_body }
    | ( OptFormal_parameter_list ) => Single_expression

OptType_declaration -> Type_declaration | $
OptFormal_parameter_list -> Formal_parameter_list | $
Formal_parameter_list -> Formal_parameter | Formal_parameter_list , Formal_parameter
Formal_parameter -> identifier | identifier Type_declaration

Function_body -> S

Array_literal -> [ Element_list ] | [ ]

Element_list -> Single_expression | Element_list , Single_expression

Object_literal -> { }
    | { Property_name_and_value_list }

Property_name_and_value_list -> Property_assignment | Property_name_and_value_list , Property_assignment

Property_assignment -> Property_name : Single_expression

Property_name -> identifier | string_literal | numeric_literal

Arguments -> ( Expression_sequence ) | ( )

Expression_sequence -> Single_expression | Expression_sequence , Single_expression

Identifier_expression  -> identifier

Unary_expression  -> new Single_expression 
    | delete Single_expression
    | void Single_expression
    | typeof Single_expression
    | post_op Single_expression
    | unary_op Single_expression
    | unary_and_binary_op Single_expression

Single_expression  -> Function_declaration
    | Single_expression [ Expression_sequence ]
    | Single_expression . Identifier_expression
    | Single_expression Arguments
    | Single_expression post_op
    | Unary_expression
    | Single_expression binary_op Single_expression
    | Single_expression unary_and_binary_op Single_expression
    | Single_expression instanceof Single_expression
    | Single_expression in Single_expression
    | Single_expression ? Single_expression : Single_expression
    | Single_expression = Expression_sequence
    | Single_expression assignment_operator Expression_sequence
    | Identifier_expression
    | Array_literal
    | Literal
    | Object_literal
    | ( Expression_sequence )

Literal -> string_literal | numeric_literal | other_literal

EOS -> ; | $
"#;

fn get_js_grammar() -> String {
    let mut lines: Vec<String> = vec![];
    for line in JS_GRAMMAR.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('|') {
            if let Some(last) = lines.last_mut() {
                last.push(' ');
                last.push_str(trimmed);
            }
        } else {
            lines.push(trimmed.to_string());
        }
    }
    lines.join("\n")
}

fn _valid_ts_programs() -> Vec<Vec<String>> {
    vec![
        vec![
            "variable_decl_type",
            "identifier",
            "=",
            "numeric_literal",
            ";",
        ],
        vec![
            "variable_decl_type",
            "identifier",
            "=",
            "numeric_literal",
            "unary_and_binary_op",
            "numeric_literal",
            ";",
        ],
        vec![
            "variable_decl_type",
            "identifier",
            "=",
            "[",
            "numeric_literal",
            ",",
            "numeric_literal",
            ",",
            "numeric_literal",
            "]",
            ";",
        ],
        vec![
            "variable_decl_type",
            "identifier",
            "=",
            "{",
            "identifier",
            ":",
            "numeric_literal",
            ",",
            "identifier",
            ":",
            "numeric_literal",
            "}",
            ";",
        ],
        vec![
            "variable_decl_type",
            "identifier",
            "=",
            "string_literal",
            ";",
        ],
        vec![
            "variable_decl_type",
            "identifier",
            ":",
            "identifier",
            "=",
            "numeric_literal",
            ";",
        ],
        vec!["function", "identifier", "(", ")", "{", "}"],
        vec![
            "function",
            "identifier",
            "(",
            "identifier",
            ":",
            "identifier",
            ")",
            "{",
            "}",
        ],
        vec![
            "function",
            "identifier",
            "(",
            "identifier",
            ":",
            "identifier",
            ",",
            "identifier",
            ":",
            "identifier",
            ")",
            "{",
            "}",
        ],
        vec![
            "variable_decl_type",
            "identifier",
            ":",
            "identifier",
            "=",
            "numeric_literal",
            ";",
            "variable_decl_type",
            "identifier",
            ":",
            "identifier",
            "=",
            "string_literal",
            ";",
            "variable_decl_type",
            "identifier",
            ":",
            "identifier",
            "=",
            "other_literal",
            ";",
        ],
        vec![
            "function",
            "identifier",
            "(",
            "identifier",
            ":",
            "identifier",
            ",",
            "identifier",
            ":",
            "identifier",
            ")",
            ":",
            "identifier",
            "{",
            "return",
            "identifier",
            "unary_and_binary_op",
            "identifier",
            ";",
            "}",
        ],
        vec![
            "variable_decl_type",
            "identifier",
            ":",
            "identifier",
            "[",
            "]",
            "=",
            "[",
            "numeric_literal",
            ",",
            "numeric_literal",
            ",",
            "numeric_literal",
            ",",
            "numeric_literal",
            ",",
            "numeric_literal",
            "]",
            ";",
            "variable_decl_type",
            "identifier",
            "=",
            "identifier",
            ".",
            "identifier",
            "(",
            "(",
            "identifier",
            ")",
            "=>",
            "identifier",
            "binary_op",
            "numeric_literal",
            ")",
            ";",
        ],
    ]
    .into_iter()
    .map(|v| v.into_iter().map(String::from).collect())
    .collect()
}

#[test]
fn test_js_grammar() {
    let text = get_js_grammar();
    let mut cfg = CFG::from_text(&text, Variable::new("S")).to_normal_form();
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty
    for word in _valid_ts_programs() {
        assert!(
            cfg.accepts(&Vec::from_iter(word.iter().map(|s| Terminal::new(s)))),
            "CFG should accept the word '{:?}', but it does not.",
            word
        ); // Check if the CFG accepts the word
    }
}

#[test]
fn test_to_normal_form() {
    let text = r#"
C#CNF#1 -> F
C#CNF#1 -> C#CNF#1 * F
E -> C#CNF#1
E -> E + C#CNF#1
F -> I
F -> ( E )
I -> b
I -> I 0
I -> a
I -> I 1
I -> I b
I -> I a
"#;
    let mut cfg = CFG::from_text(&text, Variable::new("E"));
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty
    let mut cnf_cfg = cfg.to_normal_form(); // Assuming this converts to CNF
    print!("CNF CFG: {}", cnf_cfg.to_text());
    assert!(!cnf_cfg.is_empty()); // Check if the CNF CFG is empty
    assert_eq!(cnf_cfg.terminals.len(), 8); // Check number of terminals
    assert_eq!(cnf_cfg.productions.len(), 12); // Check number of non-terminals
    assert_eq!(cnf_cfg.get_productions().len(), 20); // Check number of productions
    assert!(cnf_cfg.accepts_string("a0b1")); // Check if the CFG accepts the word
}

#[test]
fn test_example_word() {
    let text = r#"
        S -> A | B
        A -> B a
        B -> b
        "#;
    let dfa = regex_to_dfa("(a|b)*").to_deterministic();

    let cfg = CFG::from_text(text, Variable::new("S"));
    let example_word = cfg.example_word(&dfa, None).unwrap();
    println!("{:?}", example_word);
    assert!(cfg.accepts(&example_word));
    let text = r#"
        S -> A | B
        A -> B | a b
        B -> A
        "#;

    let cfg = CFG::from_text(text, Variable::new("S"));
    let example_word = cfg.example_word(&dfa, None).unwrap();
    println!("{:?}", example_word);
    assert!(cfg.accepts(&example_word));
}

#[test]
#[ignore]
fn test_caching_speedup() {
    let text = r#"
        S -> $ | A M X S X A
        A -> a
        M -> B M | B D
        B -> b
        D -> d
        X -> c
        "#;

    let mut cfg = CFG::from_text(text, Variable::new("S")).to_normal_form();
    let text_left = "abbbbdcabbdc";
    let text_right = "cacacaca";
    for i in 0..100 {
        let word = format!(
            "{}[abdc]*{}[abdc]*{}",
            text_left,
            "ca".repeat(i),
            text_right
        );
        let dfa = regex_to_dfa(&word).to_deterministic();
        assert!(!cfg.is_intersection_empty(&dfa, None));
    }
}
