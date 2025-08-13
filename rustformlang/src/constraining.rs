use hashbrown::{HashMap, HashSet};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::{
    cfg::{cfg::CFG, variable::Variable},
    fa::{
        bytes_dfa::BytesDFA, dfa::DFA, epsilon_nfa::ENFA, finite_automaton::FiniteAutomaton,
        state::State,
    },
    input_symbol::{epsilon, InputSymbol},
    regex::regex::regex_to_dfa,
};

use fancy_regex::Regex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringOrDFA {
    String(String),
    DFA(BytesDFA),
}

pub type LexMap = HashMap<InputSymbol, StringOrDFA>;
pub type CompiledLexMap = HashMap<InputSymbol, (BytesDFA, BytesDFA, BytesDFA, BytesDFA)>;
pub type Lexing = (Vec<InputSymbol>, Option<Vec<u8>>, Option<Vec<u8>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Word(String),
    EOS,
    Lexing(Lexing),
    None,
}

lazy_static! {
    // match any boundary unless at the start or end of the string (to allow suffixes/prefixes)
    static ref  word_boundary_prefix_nostart: Regex = Regex::new(r"(?<!^)\b\w").unwrap();
    static ref  word_boundary_suffix_noend: Regex = Regex::new(r"\w\b(?!$)").unwrap();
    //  match any boundary
    static ref  word_boundary_prefix: Regex = Regex::new(r"\b\w").unwrap();
    static ref  word_boundary_suffix: Regex = Regex::new(r"\w\b").unwrap();


}

fn lex_cache() -> MutexGuard<'static, HashMap<(String, bool), HashSet<Lexing>>> {
    static map: OnceLock<Mutex<HashMap<(String, bool), HashSet<Lexing>>>> = OnceLock::new();
    map.get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("Let's hope the lock isn't poisoned")
}

pub fn reset_lex_cache() {
    let mut cache = lex_cache();
    cache.clear();
}

pub fn relevant_automata_from_regex(
    regex: &StringOrDFA,
) -> (BytesDFA, BytesDFA, BytesDFA, BytesDFA) {
    let dfa = match regex {
        StringOrDFA::String(symbol) => regex_to_dfa(symbol),
        StringOrDFA::DFA(dfa) => dfa.clone(),
    };
    let suffix_lang = dfa
        .to_epsilon_automaton()
        .true_suffix_language()
        .minimize()
        .to_bytes_dfa();
    (
        dfa.clone(),
        dfa.true_prefix_language().minimize().to_bytes_dfa(),
        suffix_lang.clone(),
        suffix_lang.true_prefix_language().minimize().to_bytes_dfa(),
    )
}

pub fn remove_subtokens(d: &BytesDFA, subautomata: Vec<BytesDFA>) -> BytesDFA {
    // remove the subtoken languages from the supertokens
    let mut big_union = ENFA::empty();
    for subautomaton in subautomata {
        big_union = big_union.union(&subautomaton.to_epsilon_automaton());
    }
    d.to_deterministic()
        .difference(&big_union.minimize())
        .minimize()
        .to_bytes_dfa()
}

pub fn compile_lex_map(
    lex_map: LexMap,
    subtokens: &HashMap<InputSymbol, Vec<InputSymbol>>,
) -> CompiledLexMap {
    let mut compiled_map: CompiledLexMap = lex_map
        .into_iter()
        .map(|(name, input_symbol_or_dfa)| {
            (name, relevant_automata_from_regex(&input_symbol_or_dfa))
        })
        .collect::<HashMap<_, _>>();
    let compiled_map_clone = compiled_map.clone();
    // remove the subtoken languages from the supertokens
    for (name, automata) in compiled_map.iter_mut() {
        if let Some(subtokens) = subtokens.get(name) {
            let subautomata_list = subtokens
                .iter()
                .map(|s| {
                    compiled_map_clone
                        .get(s)
                        .expect("Subtoken not found in compiled map")
                })
                .collect::<Vec<_>>();
            automata.0 = remove_subtokens(
                &automata.0,
                subautomata_list.iter().map(|s| s.0.clone()).collect(),
            );
        }
    }
    compiled_map
}

pub fn prelex_word(word: &str, prelex: &str, is_first: bool, is_last: bool) -> String {
    // TODO need to check if this is correct
    let (prelex_prefix, prelex_suffix) = (prelex.as_bytes()[0], prelex.as_bytes()[1]);
    let add_prefix;
    if is_first {
        add_prefix = word_boundary_prefix.replace_all(word, |caps: &fancy_regex::Captures| {
            format!("{}{}", prelex_prefix as char, &caps[0])
        });
    } else {
        add_prefix = word_boundary_prefix_nostart
            .replace_all(word, |caps: &fancy_regex::Captures| {
                format!("{}{}", prelex_prefix as char, &caps[0])
            });
    }
    let add_suffix;
    if is_last {
        add_suffix = word_boundary_suffix
            .replace_all(&add_prefix, |caps: &fancy_regex::Captures| {
                format!("{}{}", &caps[0], prelex_suffix as char)
            });
    } else {
        add_suffix = word_boundary_suffix_noend
            .replace_all(&add_prefix, |caps: &fancy_regex::Captures| {
                format!("{}{}", &caps[0], prelex_suffix as char)
            });
    }
    return add_suffix.into_owned();
}

pub fn lex(
    word: &Token,
    lex_map: &CompiledLexMap,
    is_first: bool,
    strip_chars: &Option<String>,
) -> HashSet<Lexing> {
    // Generate all tokens (partially) represented by this word
    if let Token::Lexing(lexing) = word {
        return vec![lexing.clone()].into_iter().collect();
    }
    let Token::Word(word_str) = word else {
        panic!("Expected a Word token, got: {:?}", word);
    };
    match lex_cache().get(&(word_str.clone(), is_first)) {
        Some(cached) => return cached.clone(),
        None => {}
    }
    let word_str = word_str.clone().into_bytes();
    let strip_chars = strip_chars.clone().map(|s| s.into_bytes()).unwrap_or(
        vec![' ', '\t', '\n', '\r']
            .iter()
            .map(|c| *c as u8)
            .collect::<Vec<u8>>(),
    );
    let mut potential_splits: HashSet<(Vec<InputSymbol>, usize, Option<Vec<u8>>, Option<Vec<u8>>)> =
        HashSet::from([(vec![], 0, None, None)]);
    // does it partially match a suffix of any token?
    if !is_first && word_str.len() > 0 {
        for (lex_name, (_, _, lex_regex_suffix, lex_regex_suffix_prefix)) in lex_map.iter() {
            let suffix_partial_match = lex_regex_suffix_prefix.accept_prefix_bytes(&word_str);
            if suffix_partial_match == Some(word_str.len()) {
                potential_splits.insert((
                    vec![lex_name.clone()],
                    word_str.len(),
                    Some(word_str.clone()),
                    Some(word_str.clone()),
                ));
            }
            let suffix_match = lex_regex_suffix.accept_prefix_bytes(&word_str);
            if let Some(suffix_match) = suffix_match {
                if suffix_match > 0 {
                    potential_splits.insert((
                        vec![lex_name.clone()],
                        suffix_match,
                        Some(word_str[..suffix_match].to_vec()),
                        None,
                    ));
                }
            }
        }
    }
    // now check how we can lex the remaining string
    let mut final_splits = HashSet::new();
    while let Some(s) = potential_splits.iter().next().cloned() {
        potential_splits.remove(&s);
        let (current_split, position, first_partial, last_partial) = s;
        // remove whitespace in front
        let position = if position < word_str.len() {
            let mut pos = position;
            while pos < word_str.len() && strip_chars.contains(&word_str[pos]) {
                pos += 1;
            }
            pos
        } else {
            position
        };
        // if we are at the end of the word, we can add the prefix
        if position == word_str.len() {
            final_splits.insert((current_split, first_partial, last_partial));
            continue;
        }
        // Try to lex the remaining string
        for (lex_name, (lex_regex, lex_regex_prefix, _, _)) in lex_map.iter() {
            // Check if the prefix matches the regex
            if let Some(prefix_match) = lex_regex_prefix.accept_prefix_bytes(&word_str[position..])
            {
                // if the match is partial, it has to cover the remaining string
                if prefix_match == word_str.len() - position {
                    // If it matches, add the token to the prefix and continue
                    potential_splits.insert((
                        current_split
                            .iter()
                            .cloned()
                            .chain(vec![lex_name.clone()])
                            .collect(),
                        position + prefix_match,
                        first_partial.clone(),
                        Some(word_str[position..].to_vec()),
                    ));
                }
            }
            // the match can also be complete and we just remove the matching prefix
            if let Some(prefix_match) = lex_regex.accept_prefix_bytes(&word_str[position..]) {
                if prefix_match > 0 {
                    // If it matches, add the token to the prefix and continue
                    potential_splits.insert((
                        current_split
                            .iter()
                            .cloned()
                            .chain(vec![lex_name.clone()])
                            .collect(),
                        position + prefix_match,
                        first_partial.clone(),
                        None,
                    ));
                }
            }
        }
    }
    // cache the results
    lex_cache().insert(
        (
            word_str.clone().into_iter().map(char::from).collect(),
            is_first,
        ),
        final_splits.clone().into_iter().collect(),
    );
    // return the final splits
    final_splits.into_iter().collect()
}

pub fn collect_subtokens(
    grammar: &CFG,
    lex_map: &LexMap,
) -> (CFG, LexMap, HashMap<InputSymbol, Vec<InputSymbol>>) {
    let mut to_process: Vec<InputSymbol> = lex_map.keys().cloned().collect();
    let mut subtokens: HashMap<InputSymbol, Vec<InputSymbol>> = HashMap::new();
    let mut new_grammar = grammar.clone();

    while let Some(super_token) = to_process.pop() {
        let super_automaton = lex_map.get(&super_token).cloned();
        if super_automaton.is_none() {
            continue;
        }
        let super_automaton = match super_automaton.unwrap() {
            StringOrDFA::String(regex) => regex_to_dfa(&regex),
            StringOrDFA::DFA(dfa) => dfa,
        };

        let mut sub_tokens = Vec::new();
        // find all tokens that are a subset of the string token
        for (key, value) in lex_map.iter() {
            if key == &super_token {
                continue;
            }
            let other_automaton = match value {
                StringOrDFA::String(regex) => regex_to_dfa(regex),
                StringOrDFA::DFA(dfa) => dfa.clone(),
            };
            // is a subset iff the difference is empty
            if other_automaton.difference(&super_automaton).is_empty() {
                sub_tokens.push(key.clone());
            }
        }
        if sub_tokens.is_empty() {
            // nothing to do
            continue;
        }
        new_grammar = new_grammar.substitute(&HashMap::from_iter(vec![(
            super_token.clone(),
            &CFG::from_text(
                &format!(
                    "S -> {} | {}",
                    super_token.get_name(),
                    sub_tokens
                        .iter()
                        .map(|t| t.get_name())
                        .collect::<Vec<_>>()
                        .join(" | ")
                ),
                Variable {
                    name: "S".to_string(),
                },
            ),
        )]));
        subtokens.insert(super_token.clone(), sub_tokens);
    }
    (new_grammar, lex_map.clone(), subtokens)
}

pub fn derive_supertokens(
    subtokens: &HashMap<InputSymbol, Vec<InputSymbol>>,
) -> HashMap<InputSymbol, Vec<InputSymbol>> {
    let mut supertokens = HashMap::new();
    for (k, v) in subtokens {
        for subtoken in v {
            supertokens
                .entry(subtoken.clone())
                .or_insert_with(Vec::new)
                .push(k.clone());
        }
    }
    supertokens
}

/*
ef all_prefix_lexings(
    token: str,
    prelex: str | None,
    lex_map: RustLexMap,
    strip_chars: str | None = None,
):
    # prelex the token if prelex is provided
    lexings = set()
    for possible_pos in (
        (True, True),
        (True, False),
        (False, False),
        (False, True),
    ):
        token = prelex_word(
            token, prelex, is_first=possible_pos[0], is_last=possible_pos[1]
        )
        lexings.update(
            lex(
                token,
                lex_map,
                is_first=possible_pos[0],
                strip_chars=strip_chars,
            )
        )
    return lexings

 */

pub fn all_prelex_words(
    token: &str,
    prelex: Option<&str>,
    lex_map: &CompiledLexMap,
    strip_chars: Option<&str>,
) -> HashSet<Lexing> {
    let mut lexings = HashSet::new();
    let possible_poss = match prelex {
        Some(_) => vec![(true, true), (true, false), (false, false), (false, true)],
        None => vec![(false, false)],
    };
    for possible_pos in possible_poss {
        let prelexed_token = match prelex {
            Some(prelex) => prelex_word(token, prelex, possible_pos.0, possible_pos.1),
            None => token.to_string(),
        };
        let lexed = lex(
            &Token::Word(prelexed_token.clone()),
            lex_map,
            possible_pos.0,
            &strip_chars.map(|s| s.to_string()),
        );
        lexings.extend(lexed);
    }
    lexings
}

pub fn all_lexings(
    vocab: &Vec<String>,
    lex_map: &CompiledLexMap,
    prelex: Option<&str>,
    strip_chars: Option<&str>,
) -> Vec<Vec<Lexing>> {
    // Generate all lexings for each word in the vocabulary
    let all_lexings = vocab
        .par_iter()
        .map(|word| {
            let lexings = all_prelex_words(word, prelex, lex_map, strip_chars);
            lexings.into_iter().collect()
        })
        .collect::<Vec<_>>();
    all_lexings
}

pub fn lexing_to_dfa(lexing: Vec<InputSymbol>, lex_map: &CompiledLexMap) -> BytesDFA {
    let mut overall_dfa = regex_to_dfa("").to_epsilon_automaton();
    for lex in lexing {
        overall_dfa.concat(&lex_map[&lex].0.to_epsilon_automaton());
    }
    return overall_dfa.to_bytes_dfa();
}

pub fn string_to_dfa(string: String) -> BytesDFA {
    regex_to_dfa(string.as_str())
}

pub fn lexing_dfa_aligned_example_string(
    lexing: Vec<InputSymbol>,
    lex_map: &CompiledLexMap,
    string: String,
) -> Option<String> {
    lexing_to_dfa(lexing, lex_map)
        .intersection(&string_to_dfa(string))
        .example_word()
        .map(|x| String::from_utf8(x).unwrap())
}
