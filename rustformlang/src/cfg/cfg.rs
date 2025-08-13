use hashbrown::{HashMap, HashSet};
use priority_queue::PriorityQueue;
use std::cmp::{Ordering, Reverse};
use std::fmt::{Display, Formatter};
use std::hint::unreachable_unchecked;
use std::time::{Duration, Instant};
use std::{any, vec};

use crate::cfg::terminal::TERMINAL_EPSILON_SYMBOLS;
use crate::cfg::{self};
use crate::fa::dfa::DFA;
use crate::fa::state::State;
use crate::input_symbol::InputSymbol;
use crate::language::{Language, MutLanguage};
use cfg::production::{Production, Symbol};
use cfg::terminal::Terminal;
use cfg::variable::Variable;
use once_cell::sync::OnceCell;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolIndex {
    TerminalIndex(usize),
    VariableIndex(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntersectionSymbolIndex {
    TerminalIndex(usize),
    VariableIndex((usize, usize, usize)),
}

pub fn insert_or_append<T>(vec: &mut Vec<Vec<T>>, index: usize, value: T) {
    if index < vec.len() {
        vec[index].push(value);
    } else {
        vec.push(vec![value]);
    }
}

#[derive(Debug, Clone)]
pub struct ImpactsAndRemainingList {
    pub impacts_terminals: Vec<Vec<(usize, usize)>>, // Maps symbols to their impacts
    pub impacts_non_terminals: Vec<Vec<(usize, usize)>>, // Maps symbols to their impacts
    pub remaining_list: Vec<Vec<usize>>,             // Maps symbols to their remaining list
    pub added_impacts: HashSet<usize>,               // Set of added impacts
}

#[derive(Clone)]
pub struct CFG {
    pub terminal_map: OnceCell<FxHashMap<Terminal, usize>>, // Maps terminal symbols to indices

    pub terminals: Vec<Terminal>, // List of terminal symbols

    pub start_symbol: usize,                     // Start symbol
    pub productions: Vec<Vec<Vec<SymbolIndex>>>, // Production rules (non-terminal -> list<body>)

    _impacts_and_remaining_list: OnceCell<ImpactsAndRemainingList>, // Impacts and remaining list

    _is_cleaned: OnceCell<bool>, // Flag to indicate if the CFG has been cleaned (no useless symbols, epsilon, and unit productions)
    _in_normal_form: OnceCell<bool>, // Flag to indicate if the CFG is in normal form
}

impl CFG {
    /// Creates a new Context-Free Grammar
    pub fn new(
        terminal_map: Option<FxHashMap<Terminal, usize>>,
        terminals: Vec<Terminal>,
        start_symbol: usize,
        productions: Vec<Vec<Vec<SymbolIndex>>>,
    ) -> Self {
        CFG {
            terminal_map: match terminal_map {
                Some(map) => OnceCell::with_value(map),
                None => OnceCell::new(),
            },
            terminals,
            start_symbol,
            productions,
            _impacts_and_remaining_list: OnceCell::new(),
            _is_cleaned: OnceCell::new(),
            _in_normal_form: OnceCell::new(),
        }
    }

    /// Creates an empty CFG
    pub fn empty() -> Self {
        let empty_cfg = CFG::new(None, Vec::new(), 0, vec![vec![]]);
        empty_cfg._is_cleaned.set(true).unwrap();
        empty_cfg._in_normal_form.set(true).unwrap();
        empty_cfg
    }

    pub fn get_variable_at(&self, index: usize) -> Variable {
        Variable::from_string(format!("S_{}", index))
    }

    /// Returns the start symbol of the grammar
    pub fn get_start_symbol(&self) -> Variable {
        self.get_variable_at(self.start_symbol)
    }

    pub fn get_terminal_map(&self) -> &FxHashMap<Terminal, usize> {
        self.terminal_map.get_or_init(|| {
            let terminal_map: FxHashMap<Terminal, usize> = self
                .terminals
                .iter()
                .enumerate()
                .map(|(i, terminal)| (terminal.clone(), i))
                .collect();
            terminal_map
        })
    }

    pub fn get_non_terminal_map(&self) -> HashMap<Variable, usize> {
        HashMap::from_iter((0..self.productions.len()).map(|i| (self.get_variable_at(i), i)))
    }

    pub fn get_impacts_and_remaining_list(&self) -> &ImpactsAndRemainingList {
        return self._impacts_and_remaining_list.get_or_init(|| {
            // Create new impacts and remaining list
            self._get_impacts_and_remaining_list()
        });
    }

    /// Returns the production rules for a given non-terminal symbol
    pub fn get_productions_of(&self, non_terminal: &Variable) -> Vec<Production> {
        let index = self.get_non_terminal_map()[non_terminal];
        let productions = &self.productions[index];
        productions
            .iter()
            .map(|body| {
                Production::new(
                    non_terminal.clone(),
                    body.iter()
                        .map(|symbol_index| match symbol_index {
                            SymbolIndex::TerminalIndex(index) => {
                                Symbol::T(self.terminals[*index].clone())
                            }
                            SymbolIndex::VariableIndex(index) => {
                                Symbol::V(self.get_variable_at(*index).clone())
                            }
                        })
                        .collect(),
                )
            })
            .collect()
    }

    /// Returns the production rules for the grammar
    pub fn get_productions(&self) -> Vec<Production> {
        let mut all_productions = Vec::new();
        for (non_terminal, _) in self.get_non_terminal_map().clone() {
            let productions = self.get_productions_of(&non_terminal);
            all_productions.extend(productions);
        }
        all_productions
    }

    /// A convenience constructor to create a CFG from a start symbol and a map of productions.
    ///
    /// `start_variable` - The start symbol for the grammar (of type `Variable`).
    /// `productions_map` - A map of non-terminal symbols (`Variable`) to their production rules.
    /// Each production rule is represented as a vector of `Symbols` (terminals or variables).
    pub fn from_start_and_productions(
        start_variable: Variable,
        productions: Vec<Production>,
    ) -> Self {
        let mut non_terminal_map = HashMap::new();
        let mut terminal_map: FxHashMap<Terminal, usize> = FxHashMap::default();
        let mut non_terminals = Vec::new();
        let mut terminals = Vec::new();
        let mut productions_remap = HashMap::new();

        // Process the productions (and collect terminals and non-terminals on the way)
        for production in &productions {
            // Add the head of the production to the non-terminal map
            let non_terminal = production.head.clone();
            if !non_terminal_map.contains_key(&non_terminal) {
                let index = non_terminals.len();
                non_terminals.push(non_terminal.clone());

                non_terminal_map.insert(non_terminal.clone(), index);
            }

            // Add the production to the productions map
            let mut current_production = Vec::new();
            for symbol in production.body.iter() {
                match symbol {
                    Symbol::V(var) => {
                        let variable_index =
                            non_terminal_map.entry(var.clone()).or_insert_with(|| {
                                let index = non_terminals.len();
                                non_terminals.push(var.clone());
                                index
                            });
                        current_production.push(SymbolIndex::VariableIndex(*variable_index));
                    }
                    Symbol::T(term) => {
                        if TERMINAL_EPSILON_SYMBOLS.contains(&term.name.as_str()) {
                            continue;
                        }
                        let terminal_index =
                            terminal_map.entry(term.clone()).or_insert_with(|| {
                                let index = terminals.len();
                                terminals.push(term.clone());
                                index
                            });
                        current_production.push(SymbolIndex::TerminalIndex(*terminal_index));
                    }
                }
            }
            productions_remap
                .entry(non_terminal)
                .or_insert_with(Vec::new)
                .push(current_production);
        }

        // Determine the start symbol index
        let start_symbol = *non_terminal_map
            .get(&start_variable)
            .expect("Start symbol not found!");

        // Sort the productions by indices of the head non-terminal
        let mut flattened_productions = vec![vec![]; non_terminals.len()];
        for (non_terminal, production_list) in productions_remap {
            let head_index = *non_terminal_map.get(&non_terminal).unwrap();
            flattened_productions[head_index] = production_list;
        }

        CFG::new(
            Some(terminal_map),
            terminals,
            start_symbol,
            flattened_productions,
        )
    }

    pub fn to_text(&self) -> String {
        let mut result = String::new();
        result.push_str("Start Symbol: ");
        result.push_str(self.get_variable_at(self.start_symbol).display().as_str());
        result.push('\n');
        for (i, productions) in self.productions.iter().enumerate() {
            let non_terminal = self.get_variable_at(i);
            result.push_str(&format!("{} -> ", non_terminal.display()));
            for production in productions {
                if production.is_empty() {
                    result.push_str("ε | ");
                    continue;
                }
                for symbol in production {
                    match symbol {
                        SymbolIndex::TerminalIndex(index) => {
                            let terminal = &self.terminals[*index];
                            result.push_str(&format!("{} ", terminal.display()));
                        }
                        SymbolIndex::VariableIndex(index) => {
                            let variable = &self.get_variable_at(*index);
                            result.push_str(&format!("{} ", variable.display()));
                        }
                    }
                }
                result.push_str("| ");
            }
            if !productions.is_empty() {
                result.pop(); // Remove the last "|"
                result.pop(); // Remove the last "|"
            }
            result.push('\n');
        }
        result.push_str("Terminals: ");
        for terminal in &self.terminals {
            result.push_str(&format!("{} ", terminal.display()));
        }
        result
    }

    pub fn is_normal_form(&self) -> bool {
        self._in_normal_form
            .get_or_init(|| self._is_normal_form())
            .clone()
    }

    fn _is_normal_form(&self) -> bool {
        for productions in &self.productions {
            for production in productions {
                if production.len() > 2 {
                    return false; // More than two symbols is not in normal form
                }
                if production.len() == 1 {
                    match &production[0] {
                        SymbolIndex::TerminalIndex(_) => continue, // Single terminal
                        SymbolIndex::VariableIndex(_) => return false, // Single variable
                    }
                } else if production.len() == 2 {
                    match (&production[0], &production[1]) {
                        (SymbolIndex::VariableIndex(_), SymbolIndex::VariableIndex(_)) => continue, // Variable followed by Variable
                        _ => return false,
                    }
                }
            }
        }
        true
    }

    fn _get_impacts_and_remaining_list(&self) -> ImpactsAndRemainingList {
        let mut added_impacts = HashSet::new();
        // map variable -> "remaining size of its production"
        let mut remaining_list = self
            .productions
            .iter()
            .map(|p| Vec::with_capacity(p.len()))
            .collect::<Vec<_>>();
        // map symbol -> impacts
        let mut impacts_terminals: Vec<Vec<(usize, usize)>> = vec![vec![]; self.terminals.len()];
        let mut impacts_non_terminals = vec![Vec::with_capacity(300); self.productions.len()];
        for (head, production) in self.productions.iter().enumerate() {
            for body in production {
                if body.is_empty() {
                    added_impacts.insert(head);
                    continue;
                }

                remaining_list[head].push(body.len());
                let index_impact = remaining_list[head].len() - 1;
                for symbol in body {
                    match symbol {
                        SymbolIndex::TerminalIndex(index) => {
                            impacts_terminals[*index].push((head, index_impact));
                        }
                        SymbolIndex::VariableIndex(index) => {
                            impacts_non_terminals[*index].push((head, index_impact));
                        }
                    }
                }
            }
        }
        // Initial capacities are optimized based on the expected number of impacts
        // for impact in impacts_non_terminals.iter() {
        //     eprintln!("{}", impact.len());
        // }
        ImpactsAndRemainingList {
            impacts_terminals,
            impacts_non_terminals,
            remaining_list,
            added_impacts,
        }
    }

    /// Checks if the CFG generates epsilon (the empty string).
    pub fn generates_epsilon(&self) -> bool {
        let start_symbol = self.start_symbol;
        // Check if the CFG generates epsilon
        let mut generates_epsilon = HashSet::new();
        let mut to_process = vec![];

        let impact_and_remaining_list = self.get_impacts_and_remaining_list();

        for symbol in &impact_and_remaining_list.added_impacts {
            if start_symbol == *symbol {
                return true;
            }
            let symbol_index = SymbolIndex::VariableIndex(*symbol);
            if generates_epsilon.insert(symbol_index) {
                to_process.push(symbol_index);
            }
        }

        let mut remaining_lists = impact_and_remaining_list.remaining_list.clone();
        let impacts_terminals = &impact_and_remaining_list.impacts_terminals;
        let impacts_non_terminals = &impact_and_remaining_list.impacts_non_terminals;

        while let Some(symbol) = to_process.pop() {
            let impacts = match symbol {
                SymbolIndex::TerminalIndex(index) => impacts_terminals[index].as_slice(),
                SymbolIndex::VariableIndex(index) => impacts_non_terminals[index].as_slice(),
            };
            for (symbol_impact, index_impact) in impacts {
                let symbol_impact = *symbol_impact;
                let index_impact = *index_impact;
                let symbol_impact_variable_index = SymbolIndex::VariableIndex(symbol_impact);
                if generates_epsilon.contains(&symbol_impact_variable_index) {
                    continue;
                }
                remaining_lists[symbol_impact][index_impact] -= 1;
                if remaining_lists[symbol_impact][index_impact] == 0 {
                    if symbol_impact == start_symbol {
                        return true;
                    }
                    generates_epsilon.insert(symbol_impact_variable_index);
                    to_process.push(symbol_impact_variable_index);
                }
            }
        }
        false
    }

    /// Returns the intersection language of this CFG with a Regular Language (represented by a DFA).
    pub fn intersection(&self, other: &DFA) -> Self {
        // Placeholder for the intersection logic
        // This would involve creating a new CFG that represents the intersection of this CFG
        // with the DFA's language.
        // The actual implementation would depend on the specific requirements and properties
        // of the CFG and DFA.
        if other.is_empty() {
            return CFG::empty();
        }
        if !self.is_normal_form() {
            return self.to_normal_form().intersection(other);
        }
        if other.alphabet != self.terminals {
            return self.intersection(&other.with_alphabet(&self.terminals));
        }
        self._compute_intersection(other)
    }

    pub fn _compute_intersection(&self, other: &DFA) -> Self {
        let generates_epsilon = self.generates_epsilon() && other.accepts(&vec![]);
        // Note: this internal function assumes that the alphabets are exactly the same, i.e. no remapping of terminals is warranted

        // collect all transitions in other d(p, o) = q --> mpp[o] = {(p, q)}
        // make sure to also map between different variable indices
        let terminals_len = self.terminals.len();

        let mut mpp: Vec<Vec<(usize, usize)>> = vec![vec![]; terminals_len];
        for (p, symbol_map) in other.transitions.iter().enumerate() {
            for (o_dfa, q) in symbol_map {
                mpp[*o_dfa].push((p, *q));
            }
        }

        // following https://www.cs.umd.edu/~gasarch/COURSES/452/F14/cfgreg.pdf Thm 1.4
        let states_len = other.states.len();
        let nonterminals_len = self.productions.len();
        let get_non_terminal_index = |p: usize, V: usize, r: usize| -> usize {
            V * (states_len * states_len) + p * states_len + r
        };
        // Step 1 in Thm 1.4
        let new_non_terminals = states_len * nonterminals_len * states_len + 1;

        let mut new_productions = vec![Vec::with_capacity(1); new_non_terminals];
        for (A, productions) in self.productions.iter().enumerate() {
            for production in productions {
                match production.as_slice() {
                    [SymbolIndex::VariableIndex(B), SymbolIndex::VariableIndex(C)] => {
                        // Non-terminal production A -> B C
                        // Step 2 in Thm 1.4
                        for p in 0..states_len {
                            for r in 0..states_len {
                                new_productions[get_non_terminal_index(p, A, r)].extend(
                                    (0..states_len).map(|q| {
                                        vec![
                                            SymbolIndex::VariableIndex(get_non_terminal_index(
                                                p, *B, q,
                                            )),
                                            SymbolIndex::VariableIndex(get_non_terminal_index(
                                                q, *C, r,
                                            )),
                                        ]
                                    }),
                                );
                            }
                        }
                    }
                    [SymbolIndex::VariableIndex(B)] => {
                        // Variable productions A -> B
                        // Not covered by usual theorems because C2F, but we can simply map to pBq
                        for p in 0..states_len {
                            for q in 0..states_len {
                                new_productions[get_non_terminal_index(p, A, q)].push(vec![
                                    SymbolIndex::VariableIndex(get_non_terminal_index(p, *B, q)),
                                ]);
                            }
                        }
                    }
                    [SymbolIndex::TerminalIndex(o)] => {
                        // Terminal productions A -> a
                        // Step 3 in Thm 1.4
                        for (p, q) in mpp[*o].iter() {
                            new_productions[get_non_terminal_index(*p, A, *q)]
                                .push(vec![SymbolIndex::TerminalIndex(*o)]);
                        }
                    }
                    [] => {
                        // Epsilon productions A -> ε
                        // Not covered by usual theorems
                        // results in pAp -> ε (other combinations non-producing)
                        for p in 0..states_len {
                            new_productions[get_non_terminal_index(p, A, p)].push(vec![]);
                        }
                    }
                    _ => {}
                }
            }
        }
        // Generate the start symbol productions
        // Step 4 in Thm 1.4
        let start_symbol_index = new_non_terminals - 1;
        let mut start_productions = Vec::with_capacity(other.accept_states.len() + 1);
        for final_state in &other.accept_states {
            start_productions.push(vec![SymbolIndex::VariableIndex(get_non_terminal_index(
                other.start_state,
                self.start_symbol,
                *final_state,
            ))]);
        }
        if generates_epsilon {
            start_productions.push(vec![]);
        }
        new_productions[start_symbol_index] = start_productions;

        CFG::new(
            None,
            self.terminals.clone(),
            start_symbol_index,
            new_productions,
        )
    }

    /// Ensures that the CFG has only single terminal productions. (part 1 of normal form construction)
    ///
    /// Returns a grammar of the form `A -> B C D E` and `X_i -> a` where `a` is a terminal
    pub fn with_only_single_terminal_productions(&self) -> Self {
        let mut productions_list: Vec<Vec<Vec<SymbolIndex>>> =
            Vec::with_capacity(self.productions.len());
        // keep track of which symbols needed to be introduced
        let mut added_non_terminals: Vec<Option<usize>> = vec![None; self.terminals.len()];
        let mut new_productions = Vec::with_capacity(self.productions.len());

        for productions in &self.productions {
            let mut productions_list_head = vec![];
            for production in productions {
                if production.len() == 1 {
                    productions_list_head.push(production.clone());
                    continue;
                }
                // replace the terminals with new variables
                let new_production = production
                    .iter()
                    .map(|symbol| match symbol {
                        SymbolIndex::TerminalIndex(index) => {
                            if let Some(new_index) = added_non_terminals[*index] {
                                SymbolIndex::VariableIndex(new_index)
                            } else {
                                let new_index = self.productions.len() + new_productions.len();
                                added_non_terminals[*index] = Some(new_index);
                                new_productions
                                    .push(vec![vec![SymbolIndex::TerminalIndex(*index)]]);
                                SymbolIndex::VariableIndex(new_index)
                            }
                        }
                        _ => *symbol,
                    })
                    .collect::<Vec<_>>();
                productions_list_head.push(new_production.clone());
            }
            productions_list.push(productions_list_head);
        }

        // add productions for the new variables
        productions_list.extend(new_productions);

        // create a new CFG with the new productions
        CFG::new(
            self.terminal_map.get().cloned(),
            self.terminals.clone(),
            self.start_symbol,
            productions_list,
        )
    }

    /// Decomposes productions of the form `A -> B C D E` into `A -> B X`, `X -> C Y` and `Y -> D E`
    pub fn with_decomposed_productions(&self) -> Self {
        let mut productions_list: Vec<Vec<Vec<SymbolIndex>>> = vec![vec![]; self.productions.len()];
        let mut done: HashMap<&[SymbolIndex], usize> = HashMap::new();

        // only iterate over the "old" productions
        for (head, productions) in self.productions.iter().enumerate() {
            for production in productions.iter() {
                if production.len() <= 2 {
                    productions_list[head].push(production.clone());
                    continue;
                }

                // Create the new productions
                let mut cur_head = head;
                let mut stopped = false;
                for i in 0..production.len() - 2 {
                    let temp = &production[i + 1..];
                    if let Some(index) = done.get(temp) {
                        let new_production =
                            vec![production[i], SymbolIndex::VariableIndex(*index)];
                        insert_or_append(&mut productions_list, cur_head, new_production);
                        stopped = true;
                        break;
                    }
                    let new_index = productions_list.len() + (cur_head != head) as usize;
                    let new_production = vec![production[i], SymbolIndex::VariableIndex(new_index)];
                    insert_or_append(&mut productions_list, cur_head, new_production);
                    done.insert(temp, new_index);
                    cur_head = new_index;
                }
                if !stopped {
                    let new_production = vec![
                        production[production.len() - 2],
                        production[production.len() - 1],
                    ];
                    insert_or_append(&mut productions_list, cur_head, new_production);
                }
            }
        }
        // create a new CFG with the new productions
        CFG::new(
            self.terminal_map.get().cloned(),
            self.terminals.clone(),
            self.start_symbol,
            productions_list,
        )
    }

    /// Converts the CFG to a normal form (e.g., Chomsky Normal Form).
    ///
    /// Warnings
    //  ---------
    //  As described in Hopcroft's textbook, a normal form does not generate \
    //  the epsilon word. So, the grammar generated by this function is \
    //  equivalent to the original grammar except if this grammar generates \
    //  the epsilon word. In that case, the language of the generated grammar \
    //  contains the same word as before, except the epsilon word.
    pub fn to_normal_form(&self) -> Self {
        if self.is_normal_form() {
            return self.clone();
        }
        if !self.is_cleaned() {
            return self.cleaned().to_normal_form();
        }
        // Remove terminals
        let no_terminal_cfg = self.with_only_single_terminal_productions();
        let decomposed_cfg = no_terminal_cfg.with_decomposed_productions();

        decomposed_cfg
    }

    /// Computes the set of generating and nullable (if enabled) symbols in the CFG.
    fn _get_generating_symbols(&self, nullable: bool) -> HashSet<SymbolIndex> {
        let mut to_process = vec![];
        let mut generating_symbols: HashSet<SymbolIndex> = HashSet::from([]);

        let terminals_len = self.terminals.len();
        let impact_and_remaining_list = self.get_impacts_and_remaining_list();

        for symbol in &impact_and_remaining_list.added_impacts {
            let symbol_index = SymbolIndex::VariableIndex(*symbol);
            if generating_symbols.insert(symbol_index) {
                to_process.push(symbol_index);
            }
        }
        if !nullable {
            for i in 0..terminals_len {
                generating_symbols.insert(SymbolIndex::TerminalIndex(i));
                to_process.push(SymbolIndex::TerminalIndex(i));
            }
        }

        let mut remaining_lists = impact_and_remaining_list.remaining_list.clone();
        let impacts_non_terminals = &impact_and_remaining_list.impacts_non_terminals;
        let impacts_terminals = &impact_and_remaining_list.impacts_terminals;
        while let Some(symbol) = to_process.pop() {
            let impacts = match symbol {
                SymbolIndex::TerminalIndex(index) => impacts_terminals[index].as_slice(),
                SymbolIndex::VariableIndex(index) => impacts_non_terminals[index].as_slice(),
            };
            for (symbol_impact, index_impact) in impacts {
                let symbol_impact = *symbol_impact;
                let index_impact = *index_impact;
                let symbol_impact_variable_index = SymbolIndex::VariableIndex(symbol_impact);
                if generating_symbols.contains(&symbol_impact_variable_index) {
                    continue;
                }
                remaining_lists[symbol_impact][index_impact] -= 1;
                if remaining_lists[symbol_impact][index_impact] == 0 {
                    generating_symbols.insert(symbol_impact_variable_index);
                    to_process.push(symbol_impact_variable_index);
                }
            }
        }

        return generating_symbols;
    }

    pub fn get_generating_symbols(&self) -> HashSet<SymbolIndex> {
        self._get_generating_symbols(false)
    }

    pub fn get_nullable_symbols(&self) -> HashSet<SymbolIndex> {
        self._get_generating_symbols(true)
    }

    pub fn get_reachable_symbols(&self) -> HashSet<SymbolIndex> {
        let mut reachable_symbols = HashSet::from([SymbolIndex::VariableIndex(self.start_symbol)]);
        let mut reachable_transitions: HashMap<usize, HashSet<SymbolIndex>> = HashMap::new();
        for (head, productions) in self.productions.iter().enumerate() {
            for production in productions {
                let temp = reachable_transitions
                    .entry(head)
                    .or_insert_with(|| HashSet::new());
                for symbol in production {
                    temp.insert(symbol.clone());
                }
            }
        }
        let mut to_process = vec![self.start_symbol];
        while let Some(symbol) = to_process.pop() {
            if !reachable_transitions.contains_key(&symbol) {
                continue;
            }
            for next_symbol in &reachable_transitions[&symbol] {
                if !reachable_symbols.insert(*next_symbol) {
                    continue;
                }
                if let SymbolIndex::VariableIndex(index) = next_symbol {
                    to_process.push(*index);
                }
            }
        }

        reachable_symbols
    }

    /// Returns a new CFG without the specified symbols.
    pub fn reduced_to_symbols(&self, symbols: &HashSet<SymbolIndex>) -> Self {
        // TODO rewrite this to only reduce non-terminals
        assert!(
            symbols.contains(&SymbolIndex::VariableIndex(self.start_symbol)),
            "must retain start symbol"
        );
        // Filter out terminals and non-terminals that are not in the symbols set
        let mut non_term_diff = vec![0; self.productions.len()];
        let mut running_diff = 0;
        let mut non_terminals = 0;
        for i in 0..self.productions.len() {
            if !symbols.contains(&SymbolIndex::VariableIndex(i)) {
                running_diff += 1;
                continue;
            }
            non_term_diff[i] = running_diff;
            non_terminals += 1;
        }

        let mut term_diff = vec![0; self.terminals.len()];
        running_diff = 0;
        let mut terminals = Vec::with_capacity(self.terminals.len());
        for (i, symbol) in self.terminals.iter().enumerate() {
            if !symbols.contains(&SymbolIndex::TerminalIndex(i)) {
                running_diff += 1;
                continue;
            }
            term_diff[i] = running_diff;
            terminals.push(symbol.clone());
        }

        let mut productions_list: Vec<Vec<Vec<SymbolIndex>>> =
            Vec::with_capacity(self.productions.len());
        for (head, productions) in self.productions.iter().enumerate() {
            if !symbols.contains(&SymbolIndex::VariableIndex(head)) {
                continue;
            }
            let mut new_productions = Vec::with_capacity(productions.len());
            for production in productions {
                // check if all symbols in the production are generating
                let all_generating = production.iter().all(|x| symbols.contains(x));
                if all_generating {
                    new_productions.push(
                        production
                            .iter()
                            .map(|symbol| match symbol {
                                SymbolIndex::TerminalIndex(index) => {
                                    SymbolIndex::TerminalIndex(index - term_diff[*index])
                                }
                                SymbolIndex::VariableIndex(index) => {
                                    SymbolIndex::VariableIndex(index - non_term_diff[*index])
                                }
                            })
                            .collect(),
                    );
                }
            }
            productions_list.push(new_productions);
        }

        CFG::new(
            None,
            terminals,
            self.start_symbol - non_term_diff[self.start_symbol],
            productions_list,
        )
    }

    pub fn without_useless_symbols(&self) -> Self {
        let generating_symbols = self.get_generating_symbols();
        // Filter out non-generating symbols from the productions
        let mut cfg_temp = self.reduced_to_symbols(&generating_symbols);
        let reachable_symbols = cfg_temp.get_reachable_symbols();
        cfg_temp.reduced_to_symbols(&reachable_symbols)
    }

    /// Generates all production variants when removing nullable symbols.
    fn _production_without_nullable(
        production: &[SymbolIndex],
        nullable_symbols: &HashSet<SymbolIndex>,
    ) -> Vec<Vec<SymbolIndex>> {
        if production.is_empty() {
            return vec![vec![]];
        }
        let first_nullable_index = production.iter().position(|x| nullable_symbols.contains(x));
        match first_nullable_index {
            Some(index) => {
                let mut all_productions = vec![];
                let all_further_productions =
                    CFG::_production_without_nullable(&production[index + 1..], nullable_symbols);
                let nullable_symbol = production[index].clone();
                let mut all_up_to_index = production[..index].to_vec();
                // drop the symbol
                all_productions.extend(all_further_productions.iter().map(|x| {
                    let mut new_production = all_up_to_index.clone();
                    new_production.extend(x.clone());
                    new_production
                }));
                // add the symbol
                all_up_to_index.push(nullable_symbol);
                all_productions.extend(all_further_productions.iter().map(|x| {
                    let mut new_production = all_up_to_index.clone();
                    new_production.extend(x.clone());
                    new_production
                }));
                return all_productions;
            }
            None => {
                return vec![production.to_vec()];
            }
        }
    }

    /// Removes epsilon productions from the CFG
    pub fn without_epsilon(&self) -> Self {
        let nullable_symbols = self.get_nullable_symbols();
        let mut new_productions: Vec<Vec<Vec<SymbolIndex>>> = vec![vec![]; self.productions.len()];
        for (head, productions) in self.productions.iter().enumerate() {
            for production in productions {
                let new_productions_wonull =
                    CFG::_production_without_nullable(production.as_slice(), &nullable_symbols);
                new_productions[head].extend(
                    new_productions_wonull
                        .iter()
                        .filter(|x| {
                            // filter out empty productions
                            !x.is_empty()
                        })
                        .map(|x| x.clone()),
                );
            }
        }
        CFG::new(
            self.terminal_map.get().cloned(),
            self.terminals.clone(),
            self.start_symbol,
            new_productions,
        )
    }

    /// Returns a set of unit pairs (head, body) for unit productions
    pub fn get_unit_pairs(&self) -> HashSet<(usize, usize)> {
        let mut unit_pairs = HashSet::new();
        for var in 0..self.productions.len() {
            unit_pairs.insert((var, var));
        }
        let mut unit_productions = HashMap::new();
        for (head, productions) in self.productions.iter().enumerate() {
            for production in productions {
                if production.len() == 1 {
                    match &production[0] {
                        SymbolIndex::VariableIndex(index) => {
                            unit_productions
                                .entry(head)
                                .or_insert_with(HashSet::new)
                                .insert(*index);
                        }
                        _ => {}
                    }
                }
            }
        }
        let mut to_process = Vec::from_iter(unit_pairs.clone());
        while let Some((head, body)) = to_process.pop() {
            if !unit_productions.contains_key(&body) {
                continue;
            }
            for next_body in &unit_productions[&body] {
                if unit_pairs.insert((head, *next_body)) {
                    to_process.push((head, *next_body));
                }
            }
        }

        unit_pairs
    }

    /// Removes unit productions from the CFG
    ///
    /// For every production A -> B with unit production B -> YYY, add A -> YYY
    pub fn without_unit_productions(&self) -> Self {
        let unit_pairs = self.get_unit_pairs();

        // Build the map of all preceding non-terminals
        let mut unit_map = HashMap::new();
        for (head, body) in unit_pairs {
            unit_map.entry(body).or_insert_with(Vec::new).push(head);
        }
        // Build the new productions
        // For every production B -> YYY with unit production A -> B, add A -> YYY
        let mut new_productions: Vec<Vec<Vec<SymbolIndex>>> = vec![vec![]; self.productions.len()];
        for (head, productions) in self.productions.iter().enumerate() {
            for production in productions {
                if production.len() != 1 || matches!(production[0], SymbolIndex::TerminalIndex(_)) {
                    for new_head in &unit_map[&head] {
                        new_productions[*new_head].push(production.clone())
                    }
                }
            }
        }

        CFG::new(
            self.terminal_map.get().cloned(),
            self.terminals.clone(),
            self.start_symbol,
            new_productions,
        )
    }

    pub fn is_empty(&self) -> bool {
        let generating_symbols = self.get_generating_symbols();
        let start_is_generating =
            generating_symbols.contains(&SymbolIndex::VariableIndex(self.start_symbol));
        !start_is_generating
    }

    /// Returns a new CFG with a new start symbol to eliminate the start symbol from RHS
    pub fn with_new_start(mut self) -> Self {
        // Add a production for the new start symbol
        self.productions
            .push(vec![vec![SymbolIndex::VariableIndex(self.start_symbol)]]);
        self._impacts_and_remaining_list = OnceCell::new();
        self
    }

    fn _is_cleaned(&self) -> bool {
        // TODO should also check if there are shared prefixes
        let nullables = self.get_nullable_symbols();
        let unit_pairs = self.get_unit_pairs();
        let generating_symbols = self.get_generating_symbols();
        let reachable_symbols = self.get_reachable_symbols();
        return nullables.is_empty()
            && unit_pairs.len() == self.productions.len()
            && reachable_symbols.len() == self.productions.len() + self.terminals.len()
            && generating_symbols.len() == self.productions.len() + self.terminals.len();
    }

    pub fn is_cleaned(&self) -> bool {
        self._is_cleaned.get_or_init(|| self._is_cleaned()).clone()
    }

    /// Cleans the CFG by removing useless symbols, epsilon productions, and unit productions.
    /// This also turns the grammar into C2F normal Form (Sippu et. al.) which is sufficient to compute the intersection
    pub fn cleaned(&self) -> Self {
        if self.is_cleaned() {
            return self.clone();
        }
        if self.is_empty() {
            return CFG::empty();
        }
        // Otherwise, clean the CFG
        // NOTE: the order of this is fine-tuned for performance
        // also following https://www.informaticadidactica.de/uploads/Artikel/LangeLeiss2009/LangeLeiss2009.pdf for minimal increase
        // Results based on LangeLeiss https://www.informaticadidactica.de/uploads/Artikel/LangeLeiss2009/LangeLeiss2009.pdf
        // chosen order (START;TERM;BIN;DEL;UNIT) (removed UNIT for C2F)
        let mut cfg_clean = self.clone().with_new_start();
        cfg_clean = cfg_clean.without_useless_symbols();
        cfg_clean = cfg_clean.inline_single_use_symbols();
        cfg_clean = cfg_clean.without_useless_symbols();
        (cfg_clean, _) = cfg_clean.eliminate_common_prefix();
        cfg_clean = cfg_clean.without_useless_symbols();
        cfg_clean = cfg_clean.eliminate_shared_2_grams();
        cfg_clean = cfg_clean.without_useless_symbols();
        cfg_clean = cfg_clean.with_only_single_terminal_productions();
        cfg_clean = cfg_clean.without_useless_symbols();
        cfg_clean = cfg_clean.with_decomposed_productions();
        cfg_clean = cfg_clean.without_useless_symbols();
        // cfg_clean = cfg_clean.without_epsilon();
        // cfg_clean = cfg_clean.without_useless_symbols();
        // cfg_clean = cfg_clean.without_unit_productions();
        // cfg_clean = cfg_clean.without_useless_symbols();
        cfg_clean._is_cleaned = OnceCell::with_value(true);
        cfg_clean._in_normal_form = OnceCell::with_value(true);
        cfg_clean
    }

    /// Substitute the terminals in the CFG with the given CFGs.
    pub fn substitute(&self, substitutions: &HashMap<Terminal, &CFG>) -> Self {
        // Add all the non-terminals from the substitutions to the new CFG
        let mut new_terminals = self.terminals.clone();
        let mut own_terminal_sub = (0..self.terminals.len())
            .map(SymbolIndex::TerminalIndex)
            .collect::<Vec<_>>();

        let old_own_nonterminals = self.productions.len();
        let own_terminal_map = self.get_terminal_map();
        let mut new_terminal_map = own_terminal_map.clone();

        let mut new_grammar_productions = vec![];
        for (terminal, cfg) in substitutions {
            if !own_terminal_map.contains_key(terminal) {
                continue;
            }
            // Meanwhile collect which symbols terminals are being replaced with (start symbol of substitution CFG)
            let new_nonterminal_base_offset = old_own_nonterminals + new_grammar_productions.len();
            let start_symbol_index = cfg.start_symbol + new_nonterminal_base_offset;
            own_terminal_sub[own_terminal_map[terminal]] =
                SymbolIndex::VariableIndex(start_symbol_index);

            // Check which terminals are new and add the rest
            let mut terminal_map: HashMap<usize, usize> = HashMap::new();
            for (i, symbol) in cfg.terminals.iter().enumerate() {
                match new_terminal_map.get(symbol) {
                    Some(index) => {
                        terminal_map.insert(i, *index);
                    }
                    None => {
                        terminal_map.insert(i, new_terminals.len());
                        new_terminal_map.insert(symbol.clone(), new_terminals.len());
                        new_terminals.push(symbol.clone());
                    }
                }
            }

            // Add the productions from the substitution CFG (with re-mapped non-terminals)
            for productions in cfg.productions.iter() {
                let mut new_productions = vec![];
                for production in productions {
                    let mut new_production = vec![];
                    for symbol in production {
                        // Remapping the symbols
                        match symbol {
                            SymbolIndex::TerminalIndex(index) => {
                                new_production
                                    .push(SymbolIndex::TerminalIndex(terminal_map[index]));
                            }
                            SymbolIndex::VariableIndex(index) => {
                                new_production.push(SymbolIndex::VariableIndex(
                                    index + new_nonterminal_base_offset,
                                ));
                            }
                        }
                    }
                    new_productions.push(new_production);
                }
                new_grammar_productions.push(new_productions);
            }
        }

        // Now replace every terminal in the original CFG with the new terminals
        let mut own_grammar_productions = Vec::with_capacity(old_own_nonterminals);
        for productions in self.productions.iter() {
            let mut new_productions = vec![];
            for production in productions {
                let mut new_production = vec![];
                for symbol in production {
                    match symbol {
                        SymbolIndex::TerminalIndex(index) => {
                            new_production.push(own_terminal_sub[*index]);
                        }
                        s => {
                            new_production.push(s.clone());
                        }
                    }
                }
                new_productions.push(new_production);
            }
            own_grammar_productions.push(new_productions);
        }
        // Add the new productions to the original CFG
        own_grammar_productions.extend(new_grammar_productions);

        // Create a new CFG with the new productions
        CFG::new(
            None,
            new_terminals,
            self.start_symbol,
            own_grammar_productions,
        )
    }

    /// Returns the concatenation of a list of CFGs.
    pub fn concat(cfgs: Vec<&CFG>) -> Self {
        let mut new_cfg = CFG::from_start_and_productions(
            Variable::new("S"),
            vec![Production::new(
                Variable::new("S"),
                (0..cfgs.len())
                    .map(|i| {
                        Symbol::T(Terminal {
                            name: format!("x_{}", i),
                        })
                    })
                    .collect::<Vec<_>>(),
            )],
        );
        let substitution_map = cfgs
            .iter()
            .enumerate()
            .map(|(i, cfg)| {
                let terminal = Terminal::from_string(format!("x_{}", i));
                (terminal, *cfg)
            })
            .collect::<HashMap<_, _>>();
        new_cfg.substitute(&substitution_map)
    }

    pub fn concatenate(self, other: &CFG) -> Self {
        CFG::concat(vec![&self, other])
    }

    /// Eliminates common prefixes, following https://pages.cs.wisc.edu/~fischer/cs536.s15/lectures/L9.4up.pdf
    fn eliminate_common_prefix(&self) -> (Self, bool) {
        let mut new_productions = vec![vec![]; self.productions.len()];
        let mut overall_changed = false;
        for (production_index, productions) in self.productions.iter().enumerate() {
            let mut new_production = productions.clone();
            let mut changed = true;
            while changed {
                changed = false;
                let mut prefixes: HashMap<SymbolIndex, Vec<usize>> = HashMap::new();
                for (i, production) in new_production.iter().enumerate() {
                    if production.len() < 2 {
                        continue;
                    }
                    prefixes
                        .entry(production[0])
                        .or_insert_with(|| vec![])
                        .push(i);
                }
                // check if any prefix has more than one production
                for (prefix, indices) in prefixes.iter() {
                    if indices.len() < 2 {
                        continue;
                    }
                    let indices_map: HashSet<usize> = HashSet::from_iter(indices.iter().cloned());
                    let mut cur_production = vec![];
                    // append all productions without the prefix
                    for (i, production) in new_production.iter().enumerate() {
                        if indices_map.contains(&i) {
                            continue;
                        }
                        cur_production.push(production.clone());
                    }
                    // append a single production to the new symbol
                    let new_symbol_index = new_productions.len();
                    cur_production.push(vec![
                        prefix.clone(),
                        SymbolIndex::VariableIndex(new_symbol_index),
                    ]);
                    // append the productions fo the new symbol
                    let new_symbol_productions = Vec::from_iter(indices.iter().map(|x| {
                        new_production[*x]
                            .iter()
                            .skip(1)
                            .cloned()
                            .collect::<Vec<_>>()
                    }));
                    new_productions.push(new_symbol_productions);
                    new_production = cur_production;
                    changed = true;
                    overall_changed = true;
                    break;
                }
            }
            new_productions[production_index] = new_production;
        }
        (
            CFG::new(
                self.terminal_map.get().cloned(),
                self.terminals.clone(),
                self.start_symbol,
                new_productions,
            ),
            overall_changed,
        )
    }

    /// Eliminates shared 2-grams (single pass)
    pub fn eliminate_shared_2_grams_single_pass(
        productions: Vec<Vec<Vec<SymbolIndex>>>,
    ) -> (Vec<Vec<Vec<SymbolIndex>>>, bool) {
        let overall_changed = false;
        // Step 1: collect all 2-grams
        let mut all_two_grams: HashMap<(SymbolIndex, SymbolIndex), Vec<usize>> = HashMap::new();
        for (symbol_index, productions) in productions.iter().enumerate() {
            for production in productions {
                // We don't care about 2-grams appearing in normal-form style rules
                if production.len() < 3 {
                    continue;
                }
                // Iterate over pairs of adjacent symbols
                let mut prev_symbol = None;
                for i in 0..production.len() - 1 {
                    let two_gram = (production[i], production[i + 1]);
                    // if the 2-grams overlap, don't count
                    if Some(two_gram) == prev_symbol {
                        prev_symbol = None;
                        continue;
                    }
                    // otherwise count
                    all_two_grams
                        .entry(two_gram)
                        .or_insert_with(|| vec![])
                        .push(symbol_index);
                }
            }
        }
        if all_two_grams.is_empty() {
            return (productions, overall_changed);
        }
        // Step 2: replace the most frequent 2-gram with a new symbol
        let (most_frequent_2_gram, occurring_rules) = all_two_grams
            .iter()
            .max_by_key(|(_, count)| count.len())
            .unwrap();
        let most_frequent_2_gram = *most_frequent_2_gram;
        // if the maximum is 1 or less, we don't need to do anything
        if occurring_rules.len() <= 1 {
            return (productions, overall_changed);
        }
        // Check if any existing symbol is already the 2-gram
        let mut existing_symbol_index = None;
        for (symbol_index, productions) in productions.iter().enumerate() {
            match productions.as_slice() {
                [rule] => {
                    match rule.as_slice() {
                        [a, b] => {
                            // Check if the production is a single 2-gram
                            if (*a, *b) == most_frequent_2_gram {
                                // If the production is a single 2-gram, we can reuse the symbol
                                existing_symbol_index = Some(symbol_index);
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        // Otherwise, we need to replace the 2-gram with a new symbol
        let mut new_productions = productions.clone();
        let mut new_symbol_index = 0;
        match existing_symbol_index {
            Some(index) => {
                new_symbol_index = index;
            }
            None => {
                new_productions.push(vec![vec![most_frequent_2_gram.0, most_frequent_2_gram.1]]);
                new_symbol_index = productions.len();
            }
        }
        let mut symbols_to_change: Vec<usize> = Vec::from_iter(
            HashSet::<usize>::from_iter(occurring_rules.iter().cloned())
                .iter()
                .cloned(),
        );
        symbols_to_change.sort();
        for symbol_index in symbols_to_change {
            let production = productions[symbol_index].clone();
            let mut new_production = production.clone();
            for (i, rule) in production.iter().enumerate() {
                if rule.len() < 3 {
                    continue;
                }
                // Replace the 2-gram with the new symbol
                let mut new_rule = Vec::with_capacity(rule.len());
                let mut j = 0;
                while j < rule.len() - 1 {
                    let two_gram = (rule[j], rule[j + 1]);
                    if two_gram == most_frequent_2_gram {
                        // Replace the 2-gram with a new symbol
                        new_rule.push(SymbolIndex::VariableIndex(new_symbol_index));
                        // Skip the next symbol
                        j += 1;
                    } else {
                        new_rule.push(rule[j]);
                    }
                    j += 1;
                }
                // Add the last symbol if it wasn't part of the 2-gram
                if j < rule.len() {
                    new_rule.push(rule[j]);
                }
                new_production[i] = new_rule;
            }
            new_productions[symbol_index] = new_production;
        }
        // Add the new symbol to the productions
        (new_productions, true)
    }

    /// Eliminates shared 2-grams (multiple passes)
    pub fn eliminate_shared_2_grams(self) -> Self {
        let mut overall_changed = true;
        let mut productions = self.productions.clone();
        while overall_changed {
            (productions, overall_changed) = CFG::eliminate_shared_2_grams_single_pass(productions);
        }
        CFG::new(
            self.terminal_map.get().cloned(),
            self.terminals.clone(),
            self.start_symbol,
            productions,
        )
    }

    /// Inline single productions with only one rule i.e. A -> B C D with B -> X --> A -> X C D
    /// iff B is used only once
    fn inline_single_use_symbols_single_pass(
        productions: Vec<Vec<Vec<SymbolIndex>>>,
        start_symbol: usize,
    ) -> (Vec<Vec<Vec<SymbolIndex>>>, bool) {
        // Step 1: collect all usages of symbols
        let mut usages: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); productions.len()];
        for (symbol_index, productions) in productions.iter().enumerate() {
            for production in productions {
                for used_symbol in production {
                    match used_symbol {
                        SymbolIndex::TerminalIndex(_) => continue,
                        SymbolIndex::VariableIndex(index) => {
                            // if it was already used twice, ignore it
                            if usages[*index].len() >= 2 {
                                continue;
                            }
                            // otherwise add the symbol to the usages
                            usages[*index].push(symbol_index);
                        }
                    }
                }
            }
        }
        if usages.is_empty() {
            return (productions, false);
        }
        // Step 2: find the first symbol with only one usage, only one production, not the start symbol
        let mut single_use_symbol_index = None;
        for (symbol_index, usages) in usages.iter().enumerate() {
            if symbol_index == start_symbol {
                continue;
            }
            if usages.len() == 1 && productions[symbol_index].len() == 1 {
                single_use_symbol_index = Some(symbol_index);
                break;
            }
        }
        if single_use_symbol_index.is_none() {
            return (productions, false);
        }
        // Step 3: inline the single use symbol
        let used_symbol = single_use_symbol_index.unwrap();
        let using_symbol = usages[used_symbol][0];
        let mut new_productions = productions.clone();
        let mut new_using_production = productions[using_symbol].clone();
        let used_production = productions[used_symbol][0].clone();
        for (i, rule) in productions[using_symbol].iter().enumerate() {
            // if the symbol does not occur, not need to change the rule
            if rule
                .iter()
                .all(|x| x != &SymbolIndex::VariableIndex(used_symbol))
            {
                continue;
            }
            // otherwise replace the symbol with the production
            let mut new_rule = Vec::with_capacity(rule.len() + used_production.len() - 1);
            for symbol in rule {
                if symbol == &SymbolIndex::VariableIndex(used_symbol) {
                    // replace the symbol with the production
                    new_rule.extend(used_production.iter().cloned());
                } else {
                    new_rule.push(symbol.clone());
                }
            }
            new_using_production[i] = new_rule;
        }
        new_productions[using_symbol] = new_using_production;
        // Remove the single use symbol from the productions
        new_productions[used_symbol] = vec![];
        // Add the new symbol to the productions
        (new_productions, true)
    }

    /// Eliminates shared 2-grams (multiple passes)
    pub fn inline_single_use_symbols(self) -> Self {
        let mut overall_changed = true;
        let mut productions = self.productions.clone();
        while overall_changed {
            (productions, overall_changed) =
                CFG::inline_single_use_symbols_single_pass(productions, self.start_symbol);
        }
        CFG::new(
            self.terminal_map.get().cloned(),
            self.terminals.clone(),
            self.start_symbol,
            productions,
        )
    }

    fn _get_intersection_impacts_and_remaining_list(&self, dfa: &DFA) -> ImpactsAndRemainingList {
        let generates_epsilon = self.generates_epsilon() && dfa.accepts(&vec![]);

        let states_len = dfa.states.len();
        let nonterminals_len = self.productions.len();
        let terminals_len = self.terminals.len();

        let get_non_terminal_index = |p: usize, V: usize, r: usize| -> usize {
            V * (states_len * states_len) + p * states_len + r
        };
        let new_non_terminals = states_len * nonterminals_len * states_len + 1;

        let mut trans_exists: Vec<bool> = vec![false; states_len * states_len * terminals_len];
        let get_trans_index = |p: usize, o: usize, q: usize| -> usize {
            o * (states_len * states_len) + p * states_len + q
        };
        for (p, symbol_map) in dfa.transitions.iter().enumerate() {
            for (&o, &q) in symbol_map {
                trans_exists[get_trans_index(p, o, q)] = true;
            }
        }

        // map variable -> "remaining size of its production"
        let mut remaining_list: Vec<Vec<usize>> =
            vec![Vec::with_capacity(states_len); new_non_terminals];
        // map symbol -> impacts
        let mut impacts_terminals: Vec<Vec<(usize, usize)>> = vec![vec![]; self.terminals.len()];
        let mut impacts_non_terminals = vec![Vec::with_capacity(300); new_non_terminals];

        let mut added_impacts = HashSet::new();
        // handle all "normal" productions occuring from the intersection, generating the relevant productions on the fly
        for (A, productions) in self.productions.iter().enumerate() {
            for p in 0..states_len {
                for r in 0..states_len {
                    let head = get_non_terminal_index(p, A, r);
                    for production in productions {
                        match production.as_slice() {
                            [SymbolIndex::VariableIndex(B), SymbolIndex::VariableIndex(C)] => {
                                // TODO this loop might be straightforward to optimize
                                for body in (0..states_len).map(|q| {
                                    [
                                        get_non_terminal_index(p, *B, q),
                                        get_non_terminal_index(q, *C, r),
                                    ]
                                }) {
                                    remaining_list[head].push(body.len());
                                    let index_impact = remaining_list[head].len() - 1;
                                    for symbol in body {
                                        impacts_non_terminals[symbol].push((head, index_impact));
                                    }
                                }
                            }
                            [SymbolIndex::VariableIndex(B)] => {
                                // Case for C2F
                                let body = [get_non_terminal_index(p, *B, r)];
                                remaining_list[head].push(body.len());
                                let index_impact = remaining_list[head].len() - 1;
                                for symbol in body {
                                    impacts_non_terminals[symbol].push((head, index_impact));
                                }
                            }
                            [SymbolIndex::TerminalIndex(o)] => {
                                // Terminal productions A -> a
                                // Step 3 in Thm 1.4
                                if !trans_exists[get_trans_index(p, *o, r)] {
                                    continue;
                                }
                                remaining_list[head].push(1);
                                let index_impact = remaining_list[head].len() - 1;
                                impacts_terminals[*o].push((head, index_impact));
                            }
                            [] => {
                                // Epsilon productions A -> $
                                // Not handled by standard theorem.
                                // Results in pAp -> $ and pAq ->
                                added_impacts.extend(
                                    (0..states_len).map(|p| get_non_terminal_index(p, A, p)),
                                );
                            }
                            _ => unsafe { unreachable_unchecked() },
                        }
                    }
                }
            }
        }
        // Handle start symbol productions
        let start_symbol_index = new_non_terminals - 1;
        {
            let head = start_symbol_index;
            for final_state in &dfa.accept_states {
                let body = [get_non_terminal_index(
                    dfa.start_state,
                    self.start_symbol,
                    *final_state,
                )];
                remaining_list[head].push(body.len());
                let index_impact = remaining_list[head].len() - 1;
                for symbol in body {
                    impacts_non_terminals[symbol].push((head, index_impact));
                }
            }
            if generates_epsilon {
                added_impacts.insert(head);
            }
        }
        ImpactsAndRemainingList {
            impacts_terminals,
            impacts_non_terminals,
            remaining_list,
            added_impacts,
        }
    }

    /// Computes the set of generating symbols in the intersection of CFG and DFa.
    fn _is_intersection_generating_symbol(&self, dfa: &DFA, generating_q: SymbolIndex) -> bool {
        let mut to_process = vec![];
        let mut generating_symbols: HashSet<SymbolIndex> = HashSet::from([]);

        let terminals_len = self.terminals.len();
        let impact_and_remaining_list = self._get_intersection_impacts_and_remaining_list(dfa);

        for symbol in &impact_and_remaining_list.added_impacts {
            let symbol_index = SymbolIndex::VariableIndex(*symbol);
            if generating_symbols.insert(symbol_index) {
                to_process.push(symbol_index);
            }
        }
        for i in 0..terminals_len {
            generating_symbols.insert(SymbolIndex::TerminalIndex(i));
            to_process.push(SymbolIndex::TerminalIndex(i));
        }
        if generating_symbols.contains(&generating_q) {
            return true;
        }

        let mut remaining_lists = impact_and_remaining_list.remaining_list.clone();
        let impacts_non_terminals = &impact_and_remaining_list.impacts_non_terminals;
        let impacts_terminals = &impact_and_remaining_list.impacts_terminals;
        while let Some(symbol) = to_process.pop() {
            let impacts = match symbol {
                SymbolIndex::TerminalIndex(index) => impacts_terminals[index].as_slice(),
                SymbolIndex::VariableIndex(index) => impacts_non_terminals[index].as_slice(),
            };
            for (symbol_impact, index_impact) in impacts {
                let symbol_impact = *symbol_impact;
                let index_impact = *index_impact;
                let symbol_impact_variable_index = SymbolIndex::VariableIndex(symbol_impact);
                if generating_symbols.contains(&symbol_impact_variable_index) {
                    continue;
                }
                remaining_lists[symbol_impact][index_impact] -= 1;
                if remaining_lists[symbol_impact][index_impact] == 0 {
                    if symbol_impact_variable_index == generating_q {
                        return true;
                    }
                    generating_symbols.insert(symbol_impact_variable_index);
                    to_process.push(symbol_impact_variable_index);
                }
            }
        }

        return false;
    }

    /// Computes whether the intersection is empty using DFS
    /// Follows the alternative path in this SO answer: https://cs.stackexchange.com/a/92314
    /// Returns the vector of generating symbols, allowing to reconstruct an example word of the language
    fn _is_intersection_empty(&self, dfa: &DFA, timeout: Option<Duration>) -> (bool, Vec<bool>) {
        // A vector that is true for final states of the DFA
        let dfa_final_states = (0..dfa.states.len())
            .map(|i| dfa.accept_states.contains(&i))
            .collect::<Vec<_>>();
        let dfa_start_state = dfa.start_state;
        let cfg_start_symbol = self.start_symbol;
        let is_final = |p: usize, v: usize, q: usize| -> bool {
            p == dfa_start_state && v == cfg_start_symbol && dfa_final_states[q]
        };

        let states_len = dfa.states.len();
        let nonterminals_len = self.productions.len();
        let terminals_len = self.terminals.len();

        let get_non_terminal_index = |p: usize, v: usize, r: usize| -> usize {
            v * (states_len * states_len) + p * states_len + r
        };
        let new_non_terminals = states_len * nonterminals_len * states_len + 1;

        // Processing queue for search
        let mut to_process: Vec<(usize, usize, usize, usize)> =
            Vec::with_capacity(new_non_terminals);
        let mut generating_symbols: Vec<bool> = vec![false; new_non_terminals];

        // Map once for every terminal the transition \delta(p, o) = q that exists in the DFA
        let mut trans_exists: Vec<HashSet<(usize, usize)>> = vec![HashSet::new(); terminals_len];
        for (p, symbol_map) in dfa.transitions.iter().enumerate() {
            for (&o, &q) in symbol_map {
                trans_exists[o].insert((p, q));
            }
        }
        // Collect once all nonterminals A that have a production A -> B by B
        let mut single_production_map: Vec<Vec<usize>> = vec![vec![]; nonterminals_len];
        // Collect once all A/C that have a production A -> B C by B
        let mut two_production_map_r: Vec<Vec<(usize, usize)>> = vec![vec![]; nonterminals_len];
        // Collect once all A/B that have a production A -> B C by C
        let mut two_production_map_l: Vec<Vec<(usize, usize)>> = vec![vec![]; nonterminals_len];

        for (a_symbol, rules) in self.productions.iter().enumerate() {
            for production in rules {
                match production.as_slice() {
                    [SymbolIndex::VariableIndex(b_symbol), SymbolIndex::VariableIndex(c_symbol)] => {
                        // Non-terminal productions are not generating symbols
                        // But are collected
                        two_production_map_r[*b_symbol].push((*c_symbol, a_symbol));
                        two_production_map_l[*c_symbol].push((*b_symbol, a_symbol));
                    }
                    [SymbolIndex::VariableIndex(b_symbol)] => {
                        // Single non-terminal productions are not generating symbols
                        // But are collected
                        single_production_map[*b_symbol].push(a_symbol);
                    }
                    [SymbolIndex::TerminalIndex(terminal_index)] => {
                        // This handles Step 3 in Thm 1.4
                        // Mark all productions that generate terminals as generating symbols
                        // Terminal productions are generating symbols
                        for (p, q) in trans_exists[*terminal_index].iter() {
                            let new_symbol_index = get_non_terminal_index(*p, a_symbol, *q);
                            if generating_symbols[new_symbol_index] {
                                continue;
                            }
                            if is_final(*p, a_symbol, *q) {
                                return (false, generating_symbols);
                            }
                            generating_symbols[new_symbol_index] = true;
                            to_process.push((*p, a_symbol, *q, new_symbol_index));
                        }
                    }
                    [] => {
                        // Epsilon productions are generating symbols
                        for p in 0..states_len {
                            let symbol_index = get_non_terminal_index(p, a_symbol, p);
                            if generating_symbols[symbol_index] {
                                continue;
                            }
                            if is_final(p, a_symbol, p) {
                                return (false, generating_symbols);
                            }
                            generating_symbols[symbol_index] = true;
                            to_process.push((p, a_symbol, p, symbol_index));
                        }
                    }
                    _ => unsafe {
                        unreachable_unchecked();
                    },
                }
            }
        }

        let start_time = Instant::now();
        let mut i = 0;
        while let Some((p_old, b_old, q_old, _new_symbol)) = to_process.pop() {
            i += 1;
            // Check for timeout
            if let Some(timeout) = timeout {
                if start_time.elapsed() > timeout {
                    return (true, generating_symbols);
                }
            }
            if to_process.len() > 64_000_000 || i > 64_000_000 {
                // If the queue is too large, return the current generation map
                return (true, generating_symbols);
            }
            // Iterate through all A such that A -> B and mark as generating
            for a_symbol in &single_production_map[b_old] {
                let symbol_index = get_non_terminal_index(p_old, *a_symbol, q_old);
                if generating_symbols[symbol_index] {
                    continue;
                }
                if is_final(p_old, *a_symbol, q_old) {
                    return (false, generating_symbols);
                }
                generating_symbols[symbol_index] = true;
                to_process.push((p_old, *a_symbol, q_old, symbol_index));
            }
            // Iterate through all A/C such that A -> B C and mark as generating
            for (c_symbol, a_symbol) in &two_production_map_r[b_old] {
                for r_old in 0..states_len {
                    let c_symbol_index = get_non_terminal_index(q_old, *c_symbol, r_old);
                    // Only mark A as generating if C is generating
                    if !generating_symbols[c_symbol_index] {
                        continue;
                    }
                    let a_symbol_index = get_non_terminal_index(p_old, *a_symbol, r_old);
                    if generating_symbols[a_symbol_index] {
                        continue;
                    }
                    if is_final(p_old, *a_symbol, r_old) {
                        return (false, generating_symbols);
                    }
                    generating_symbols[a_symbol_index] = true;
                    to_process.push((p_old, *a_symbol, r_old, a_symbol_index));
                }
            }
            // Iterate through all A/B such that A -> B C and mark as generating
            for (b_symbol, a_symbol) in &two_production_map_l[b_old] {
                for r_old in 0..states_len {
                    let b_symbol_index = get_non_terminal_index(r_old, *b_symbol, p_old);
                    // Only mark A as generating if B is generating
                    if !generating_symbols[b_symbol_index] {
                        continue;
                    }
                    let a_symbol_index = get_non_terminal_index(r_old, *a_symbol, q_old);
                    if generating_symbols[a_symbol_index] {
                        continue;
                    }
                    if is_final(r_old, *a_symbol, q_old) {
                        return (false, generating_symbols);
                    }
                    generating_symbols[a_symbol_index] = true;
                    to_process.push((r_old, *a_symbol, q_old, a_symbol_index));
                }
            }
        }

        (true, generating_symbols)
    }

    /// Finds a mapping from generating symbol to which symbol generates it
    /// the last element is a map (intersection non-terminal) -> (generating production in intersection language)
    /// to be used for example word
    fn _find_generating_word(
        &self,
        dfa: &DFA,
        timeout: Option<Duration>,
    ) -> (
        bool,
        HashMap<(usize, usize, usize), Vec<IntersectionSymbolIndex>>,
    ) {
        // A vector that is true for final states of the DFA
        let dfa_final_states = (0..dfa.states.len())
            .map(|i| dfa.accept_states.contains(&i))
            .collect::<Vec<_>>();
        let dfa_start_state = dfa.start_state;
        let cfg_start_symbol = self.start_symbol;
        let is_final = |p: usize, v: usize, q: usize| -> bool {
            p == dfa_start_state && v == cfg_start_symbol && dfa_final_states[q]
        };

        let states_len = dfa.states.len();
        let nonterminals_len = self.productions.len();
        let terminals_len = self.terminals.len();

        let get_non_terminal_index = |p: usize, v: usize, r: usize| -> usize {
            v * (states_len * states_len) + p * states_len + r
        };
        let new_non_terminals = states_len * nonterminals_len * states_len + 1;

        // Processing queue for search
        let mut to_process = PriorityQueue::with_capacity(new_non_terminals);
        // Track the shortest generated string for each (p, v, q)
        let mut generating_symbols: Vec<Option<usize>> = vec![None; new_non_terminals];

        // Map once for every terminal the transition \delta(p, o) = q that exists in the DFA
        let mut trans_exists: Vec<HashSet<(usize, usize)>> = vec![HashSet::new(); terminals_len];
        for (p, symbol_map) in dfa.transitions.iter().enumerate() {
            for (&o, &q) in symbol_map {
                trans_exists[o].insert((p, q));
            }
        }
        // Collect once all nonterminals A that have a production A -> B by B
        let mut single_production_map: Vec<Vec<usize>> = vec![vec![]; nonterminals_len];
        // Collect once all A/C that have a production A -> B C by B
        let mut two_production_map_r: Vec<Vec<(usize, usize)>> = vec![vec![]; nonterminals_len];
        // Collect once all A/B that have a production A -> B C by C
        let mut two_production_map_l: Vec<Vec<(usize, usize)>> = vec![vec![]; nonterminals_len];

        let mut generation_map: HashMap<(usize, usize, usize), Vec<IntersectionSymbolIndex>> =
            HashMap::new();

        for (a_symbol, rules) in self.productions.iter().enumerate() {
            for production in rules {
                match production.as_slice() {
                    [SymbolIndex::VariableIndex(b_symbol), SymbolIndex::VariableIndex(c_symbol)] => {
                        // Non-terminal productions are not generating symbols
                        // But are collected
                        two_production_map_r[*b_symbol].push((*c_symbol, a_symbol));
                        two_production_map_l[*c_symbol].push((*b_symbol, a_symbol));
                    }
                    [SymbolIndex::VariableIndex(b_symbol)] => {
                        // Single non-terminal productions are not generating symbols
                        // But are collected
                        single_production_map[*b_symbol].push(a_symbol);
                    }
                    [SymbolIndex::TerminalIndex(terminal_index)] => {
                        // This handles Step 3 in Thm 1.4
                        // Mark all productions that generate terminals as generating symbols
                        // Terminal productions are generating symbols
                        for (p, q) in trans_exists[*terminal_index].iter() {
                            let new_symbol_index = get_non_terminal_index(*p, a_symbol, *q);
                            if !generating_symbols[new_symbol_index].is_none() {
                                continue;
                            }
                            generation_map.insert(
                                (*p, a_symbol, *q),
                                vec![IntersectionSymbolIndex::TerminalIndex(*terminal_index)],
                            );
                            if is_final(*p, a_symbol, *q) {
                                return (false, generation_map);
                            }
                            generating_symbols[new_symbol_index] = Some(1);
                            to_process.push((*p, a_symbol, *q, new_symbol_index), Reverse(1));
                        }
                    }
                    [] => {
                        // Epsilon productions are generating symbols
                        for p in 0..states_len {
                            let symbol_index = get_non_terminal_index(p, a_symbol, p);
                            if let Some(x) = generating_symbols[symbol_index] {
                                if x == 0 {
                                    continue;
                                }
                            }
                            generation_map.insert((p, a_symbol, p), vec![]);
                            if is_final(p, a_symbol, p) {
                                return (false, generation_map);
                            }
                            generating_symbols[symbol_index] = Some(0);
                            to_process.push_increase((p, a_symbol, p, symbol_index), Reverse(0));
                        }
                    }
                    _ => unsafe {
                        unreachable_unchecked();
                    },
                }
            }
        }

        let start_time = Instant::now();
        let mut i = 0;
        while let Some(((p_old, b_old, q_old, _new_symbol), Reverse(cur_length))) = to_process.pop()
        {
            i += 1;
            if let Some(timeout) = timeout {
                if start_time.elapsed() > timeout {
                    // Timeout reached, return the current generation map
                    return (true, generation_map);
                }
            }
            if to_process.len() > 64_000_000 || i > 64_000_000 {
                // If the queue is too large, return the current generation map
                return (true, generation_map);
            }
            // Iterate through all A such that A -> B and mark as generating
            for a_symbol in &single_production_map[b_old] {
                let symbol_index = get_non_terminal_index(p_old, *a_symbol, q_old);
                if generating_symbols[symbol_index].is_some() {
                    continue;
                }
                generation_map.insert(
                    (p_old, *a_symbol, q_old),
                    vec![IntersectionSymbolIndex::VariableIndex((
                        p_old, b_old, q_old,
                    ))],
                );
                if is_final(p_old, *a_symbol, q_old) {
                    return (false, generation_map);
                }
                generating_symbols[symbol_index] = Some(cur_length);
                to_process
                    .push_increase((p_old, *a_symbol, q_old, symbol_index), Reverse(cur_length));
            }
            // Iterate through all A/C such that A -> B C and mark as generating
            for (c_symbol, a_symbol) in &two_production_map_r[b_old] {
                for r_old in 0..states_len {
                    let c_symbol_index = get_non_terminal_index(q_old, *c_symbol, r_old);
                    // Only mark A as generating if C is generating
                    if let Some(c_length) = generating_symbols[c_symbol_index] {
                        let a_symbol_index = get_non_terminal_index(p_old, *a_symbol, r_old);
                        // Only mark A as generating if it is not already generating
                        if generating_symbols[a_symbol_index].is_some() {
                            continue;
                        }
                        generation_map.insert(
                            (p_old, *a_symbol, r_old),
                            vec![
                                IntersectionSymbolIndex::VariableIndex(((p_old, b_old, q_old))),
                                IntersectionSymbolIndex::VariableIndex((q_old, *c_symbol, r_old)),
                            ],
                        );
                        if is_final(p_old, *a_symbol, r_old) {
                            return (false, generation_map);
                        }
                        generating_symbols[a_symbol_index] = Some(c_length + cur_length);
                        to_process.push_increase(
                            (p_old, *a_symbol, r_old, a_symbol_index),
                            Reverse(cur_length + c_length),
                        );
                    } else {
                        // C is not generating, skip
                        continue;
                    }
                }
            }
            // Iterate through all A/B such that A -> B C and mark as generating
            for (b_symbol, a_symbol) in &two_production_map_l[b_old] {
                for r_old in 0..states_len {
                    let b_symbol_index = get_non_terminal_index(r_old, *b_symbol, p_old);
                    // Only mark A as generating if B is generating
                    if let Some(b_length) = generating_symbols[b_symbol_index] {
                        let a_symbol_index = get_non_terminal_index(r_old, *a_symbol, q_old);
                        if generating_symbols[a_symbol_index].is_some() {
                            // A is already generating, skip
                            continue;
                        }
                        generation_map.insert(
                            (r_old, *a_symbol, q_old),
                            vec![
                                IntersectionSymbolIndex::VariableIndex((r_old, *b_symbol, p_old)),
                                IntersectionSymbolIndex::VariableIndex((p_old, b_old, q_old)),
                            ],
                        );
                        if is_final(r_old, *a_symbol, q_old) {
                            return (false, generation_map);
                        }
                        generating_symbols[a_symbol_index] = Some(b_length + cur_length);
                        to_process.push_increase(
                            (r_old, *a_symbol, q_old, a_symbol_index),
                            Reverse(b_length + cur_length),
                        );
                    } else {
                        // B is not generating, skip
                        continue;
                    }
                }
            }
        }

        (true, generation_map)
    }

    /// Returns an example word of the language of the CFG that intersects with the DFA.
    pub fn example_word(&self, dfa: &DFA, timeout: Option<Duration>) -> Option<Vec<InputSymbol>> {
        if dfa.is_empty() {
            return None;
        }
        if !self.is_normal_form() {
            return self.to_normal_form().example_word(dfa, timeout);
        }
        if dfa.alphabet != self.terminals {
            return self.example_word(&dfa.with_alphabet(&self.terminals), timeout);
        }

        let start_time = Instant::now();
        let (is_empty, generating_map) = self._find_generating_word(dfa, timeout);
        if is_empty {
            // If the language is empty, there is no example word
            return None;
        }
        // Find the path of generating symbols from the start
        // Starts at the generating start symbol and continues through the all-generating production
        let mut generating_start = None;
        for dfa_final_state in dfa.accept_states.iter() {
            let state_index = (dfa.start_state, self.start_symbol, *dfa_final_state);
            if generating_map.contains_key(&state_index) {
                generating_start = Some(state_index);
                break;
            }
        }
        let generating_start = generating_start.unwrap();

        let mut current_word = vec![IntersectionSymbolIndex::VariableIndex(generating_start)];
        let mut any_variable_left = vec![0];
        let mut i = 0;
        while !any_variable_left.is_empty() {
            i += 1;
            if let Some(timeout) = timeout {
                // If the timeout is reached, return None
                if start_time.elapsed() > timeout {
                    // Timeout reached, return Nothing
                    return None;
                }
            }
            if any_variable_left.len() > 64_000_000 || i > 64_000_000 {
                // If the queue is too large, return None
                return None;
            }
            // Iterate through the current word and expand any remaining variables
            let mut expanded_word: Vec<IntersectionSymbolIndex> = vec![];
            let mut prev_index = 0;
            for current_index in any_variable_left.iter() {
                // copy over all from prev index until current_index (exclusive)
                expanded_word.extend(
                    current_word
                        .iter()
                        .skip(prev_index)
                        .take(*current_index - prev_index),
                );
                prev_index = current_index + 1;
                let symbol_to_expand = match current_word[*current_index] {
                    IntersectionSymbolIndex::VariableIndex(s) => s,
                    _ => {
                        unreachable!()
                    }
                };
                // replace the variable at current index with a valid expansion
                if let Some(symbols) = generating_map.get(&symbol_to_expand) {
                    // if its cached, use that
                    expanded_word.extend(symbols.iter().cloned());
                } else {
                    unreachable!();
                }
            }
            expanded_word.extend(
                current_word
                    .iter()
                    .skip(*any_variable_left.last().unwrap() + 1),
            );
            any_variable_left = expanded_word
                .iter()
                .enumerate()
                .filter_map(|(i, x)| match x {
                    IntersectionSymbolIndex::TerminalIndex(_) => None,
                    IntersectionSymbolIndex::VariableIndex(_) => Some(i),
                })
                .collect();
            current_word = expanded_word;
        }

        Some(
            current_word
                .into_iter()
                .map(|s| match s {
                    IntersectionSymbolIndex::TerminalIndex(t) => self.terminals[t].clone(),
                    _ => {
                        unreachable!()
                    }
                })
                .collect(),
        )
    }

    /// Determines whether the intersection with the DFA is empty
    pub fn is_intersection_empty(&self, dfa: &DFA, timeout: Option<Duration>) -> bool {
        if dfa.is_empty() {
            return true;
        }
        if !self.is_normal_form() {
            return self.to_normal_form().is_intersection_empty(dfa, timeout);
        }
        if dfa.alphabet != self.terminals {
            return self.is_intersection_empty(&dfa.with_alphabet(&self.terminals), timeout);
        }
        self._is_intersection_empty(dfa, timeout).0
    }
}

impl Language for CFG {
    fn accepts(&self, input: &Vec<InputSymbol>) -> bool {
        if input.is_empty() {
            return self.generates_epsilon();
        }

        // build a DFA from the input
        let mut dfa = DFA::empty();
        let start_state = State::new("start");
        dfa.set_start_state(start_state.clone());
        let mut prev_state = start_state;
        for (i, symbol) in input.iter().enumerate() {
            let state = State::new(&format!("state_{}", i));
            dfa.add_transition(&prev_state, &symbol, &state);
            prev_state = state;
        }
        dfa.add_accept_state(prev_state.clone());
        dfa = dfa.minimize();

        // Build the intersection adn check if it is empty
        !self.is_intersection_empty(&mut dfa, Some(Duration::from_secs(60)))
    }
}

impl CFG {
    /// Reads a context-free grammar from a string of text.
    ///
    /// Each rule in the grammar is represented as one line in the following format:
    ///   `head -> body1 | body2 | ... | bodyn`
    ///
    /// Non-terminals should start with a capital letter, while terminals start with a lowercase letter.
    /// Special cases for epsilon symbols include `$`, `ε`, `ϵ`, `Є`, or `epsilon`.
    ///
    /// # Parameters
    /// - `text`: The text representing the CFG rules.
    /// - `start_symbol`: The start symbol of the grammar (default is `S`).
    ///
    /// # Returns
    /// Returns a `CFG` object initialized from the given textual representation.
    pub fn from_text(text: &str, start_symbol: Variable) -> Self {
        let mut variables = HashSet::new();
        let mut terminals = HashSet::new();
        let mut productions = Vec::new();

        for line in text.lines() {
            let line = line.trim();
            if !line.is_empty() {
                CFG::read_line(line, &mut productions, &mut terminals, &mut variables);
            }
        }

        Self::from_start_and_productions(start_symbol, productions)
    }

    /// Internal helper to process a single line of a grammar rule.
    ///
    /// Splits the rule into its head and body parts, parsing the head as a variable
    /// and the body parts as a series of terminals and/or variables.
    ///
    /// # Parameters
    /// - `line`: The textual representation of a production rule.
    /// - `productions`: A mutable vector to store parsed productions.
    /// - `terminals`: A mutable hash set to collect all terminal symbols.
    /// - `variables`: A mutable hash set to collect all variable symbols.
    fn read_line(
        line: &str,
        productions: &mut Vec<Production>,
        terminals: &mut HashSet<Terminal>,
        variables: &mut HashSet<Variable>,
    ) {
        if let Some((head_s, body_s)) = line.split_once("->") {
            let mut head_text = head_s.trim();
            if let Some(head_text_stripped) = head_text.strip_prefix("\"VAR:") {
                head_text = head_text_stripped
                    .strip_suffix("\"")
                    .expect("Invalid variable format, needs to end with a quote");
            }

            let head = Variable::new(head_text);
            variables.insert(head.clone());

            for sub_body in body_s.split('|') {
                let mut body = Vec::new();
                for body_component in sub_body.split_whitespace() {
                    // Handle epsilon or escaped symbols
                    if TERMINAL_EPSILON_SYMBOLS.contains(&body_component) {
                        // Epsilon productions are represented with empty bodies
                        continue;
                    }

                    if let Some(variable) = body_component.strip_prefix("\"VAR:") {
                        let var = Variable::new(
                            variable
                                .strip_suffix("\"")
                                .expect("Invalid variable format, needs to end with a quote"),
                        );
                        variables.insert(var.clone());
                        body.push(Symbol::V(var));
                    } else if let Some(terminal) = body_component.strip_prefix("\"TER:") {
                        let term = Terminal::new(
                            terminal
                                .strip_suffix("\"")
                                .expect("Invalid terminal format, needs to end with a quote"),
                        );
                        terminals.insert(term.clone());
                        body.push(Symbol::T(term));
                    } else {
                        if body_component
                            .chars()
                            .next()
                            .unwrap_or_default()
                            .is_uppercase()
                        {
                            let var = Variable::new(body_component);
                            variables.insert(var.clone());
                            body.push(Symbol::V(var));
                        } else {
                            let term = Terminal::new(body_component);
                            terminals.insert(term.clone());
                            body.push(Symbol::T(term));
                        }
                    }
                }

                productions.push(Production::new(head.clone(), body));
            }
        } else {
            panic!("Malformed grammar rule: {}", line);
        }
    }
}

// Example usage
fn _cfg_demo() {
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
    println!("{}", cfg.to_text());
}

impl Display for CFG {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}
