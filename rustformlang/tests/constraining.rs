use hashbrown::{HashMap, HashSet};

use lazy_static::lazy_static;
use rustformlang::{
    cfg::{cfg::CFG, variable::Variable},
    constraining::{
        compile_lex_map, derive_supertokens, lex as lex_raw, CompiledLexMap, LexMap, Lexing,
        StringOrDFA, Token,
    },
    fa::dfa::DFA,
    input_symbol::InputSymbol,
    language::Language,
};
macro_rules! set {
    ($($x:expr),* $(,)?) => {
        HashSet::from([$($x),*])
    };
}
macro_rules! isvec {
    ($($x:expr),* $(,)?) => {
        vec![$(InputSymbol::new($x)),*]
    };
}

macro_rules! str_to_vec {
    ($s:expr) => {
        $s.as_bytes().to_vec()
    };
}

///
/// LEX_MAP = {
///     # both original and reversed
///     "lexNumber": r"((-?[1-9]\d*)|0)",
///     "lexString": r'"[^\n\r"]*"',
///     "lexNull": r"null",
///     "lexTrue": r"true",
///     "lexFalse": r"false",
///     "lexFence": r"```",
/// }
/// LEX_TOKENS = {"{", "}", ",", "[", "]", ":"}
/// for token in LEX_TOKENS:
///     LEX_MAP[token] = regex_escape(token)

lazy_static! {
    static ref LEX_MAP: LexMap = HashMap::from([
        (
            InputSymbol::new("lexNumber"),
            StringOrDFA::String("((-?[1-9]\\d*)|0)".to_string())
        ),
        (
            InputSymbol::new("lexString"),
            StringOrDFA::String(r#""[^\n\r"]*""#.to_string())
        ),
        (
            InputSymbol::new("lexNull"),
            StringOrDFA::String("null".to_string())
        ),
        (
            InputSymbol::new("lexTrue"),
            StringOrDFA::String("true".to_string())
        ),
        (
            InputSymbol::new("lexFalse"),
            StringOrDFA::String("false".to_string())
        ),
        (
            InputSymbol::new("lexFence"),
            StringOrDFA::String("```".to_string())
        ),
        (
            InputSymbol::new("{"),
            StringOrDFA::String(regex_syntax::escape("{").to_string())
        ),
        (
            InputSymbol::new("}"),
            StringOrDFA::String(regex_syntax::escape("}").to_string())
        ),
        (
            InputSymbol::new(","),
            StringOrDFA::String(regex_syntax::escape(",").to_string())
        ),
        (
            InputSymbol::new("["),
            StringOrDFA::String(regex_syntax::escape("[").to_string())
        ),
        (
            InputSymbol::new("]"),
            StringOrDFA::String(regex_syntax::escape("]").to_string())
        ),
        (
            InputSymbol::new(":"),
            StringOrDFA::String(regex_syntax::escape(":").to_string())
        ),
    ]);
}

pub fn lex(s: &str, lex_map: &CompiledLexMap, is_first: bool) -> HashSet<Lexing> {
    lex_raw(&Token::Word(s.to_string()), lex_map, is_first, &None)
}

#[test]
pub fn test_lex() {
    let lex_map = compile_lex_map(LEX_MAP.clone(), &HashMap::new());
    assert_eq!(
        lex("null { } 123 \"hi!\" ", &lex_map, false),
        set![(
            isvec!["lexNull", "{", "}", "lexNumber", "lexString",],
            None,
            None,
        ),]
    );

    assert_eq!(
        lex(r#"null" "#, &lex_map, false),
        set![
            (isvec!["lexString"], Some(str_to_vec!("null\"")), None),
            (
                isvec!["lexNull", "lexString"],
                None,
                Some(str_to_vec!("\" "))
            ),
        ]
    );

    assert_eq!(
        lex(r#"lhio! " : 123, ""#, &lex_map, false),
        set![(
            isvec!["lexString", ":", "lexNumber", ",", "lexString"],
            Some(str_to_vec!("lhio! \"")),
            Some(str_to_vec!("\""))
        )]
    );

    assert_eq!(
        lex("test", &lex_map, false),
        set![(
            isvec!["lexString"],
            Some(str_to_vec!("test")),
            Some(str_to_vec!("test"))
        ),]
    );

    assert_eq!(
        lex(r#""test""#, &lex_map, false),
        set![(isvec!["lexString"], None, None),]
    );

    assert_eq!(
        lex(r#"ing":"#, &lex_map, false),
        set![(isvec!["lexString", ":"], Some(str_to_vec!("ing\"")), None),]
    );

    assert_eq!(
        lex(r#"":"#, &lex_map, false),
        set![
            (isvec!["lexString", ":"], Some(str_to_vec!("\"")), None),
            (isvec!["lexString"], None, Some(str_to_vec!("\":"))),
        ]
    );

    assert_eq!(
        lex(" ", &lex_map, false),
        set![
            (isvec![], None, None),
            (
                isvec!["lexString"],
                Some(str_to_vec!(" ")),
                Some(str_to_vec!(" "))
            ),
        ]
    );

    assert_eq!(
        lex("123", &lex_map, true),
        set![
            (isvec!["lexNumber"], None, Some(str_to_vec!("123"))),
            (isvec!["lexNumber"], None, None),
        ]
    );

    assert_eq!(
        lex("123", &lex_map, false),
        set![
            (
                isvec!["lexString"],
                Some(str_to_vec!("123")),
                Some(str_to_vec!("123"))
            ),
            (
                isvec!["lexNumber"],
                Some(str_to_vec!("123")),
                Some(str_to_vec!("123"))
            ),
            (isvec!["lexNumber"], None, Some(str_to_vec!("123"))),
            (isvec!["lexNumber"], Some(str_to_vec!("123")), None),
            (isvec!["lexNumber"], None, None),
        ]
    );
}
