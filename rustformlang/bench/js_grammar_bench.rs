use rustformlang::cfg::cfg::CFG;
use rustformlang::cfg::terminal::Terminal;
use rustformlang::cfg::variable::Variable;
use rustformlang::language::Language;

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
    | for ( Single_expression_or_variable_decl_type in_of Expression_sequence ) Statement

Single_expression_or_variable_decl_type -> Single_expression | variable_decl_type Variable_declaration

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

fn test_js_grammar(cfg: &mut CFG) {
    for word in _valid_ts_programs() {
        assert!(
            cfg.accepts(&Vec::from_iter(word.iter().map(|s| Terminal::new(s)))),
            "CFG should accept the word '{:?}', but it does not.",
            word
        ); // Check if the CFG accepts the word
    }
}

fn main() {
    let text = get_js_grammar();
    let mut cfg = CFG::from_text(&text, Variable::new("S"));
    println!("CFG size (productions): {}", cfg.get_productions().len());
    assert!(!cfg.is_empty()); // Check if the generated CFG is empty
    cfg = cfg.to_normal_form();
    println!(
        "CFG in normal form size (productions): {}",
        cfg.get_productions().len()
    );
    //println!("{}", cfg.to_text());
    for _ in 0..40 {
        test_js_grammar(&mut cfg);
    }
    println!("All tests passed!");
}
