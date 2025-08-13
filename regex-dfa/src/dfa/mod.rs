// Copyright 2015-2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod minimizer;
mod prefix_searcher;
mod trie;

use crate::dfa::minimizer::Minimizer;
use crate::dfa::prefix_searcher::PrefixSearcher;
use crate::graph::Graph;
use crate::look::Look;
use crate::nfa::{Accept, StateIdx};
use crate::refinery::Partition;
use crate::runner::program::TableInsts;
use itertools::Itertools;
use range_map::{RangeMap, RangeMultiMap};
use std;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::mem;
use std::u32;

pub use crate::dfa::prefix_searcher::PrefixPart;

#[derive(Clone, PartialEq, Debug)]
pub struct State<Ret> {
    pub transitions: RangeMap<u8, StateIdx>,
    pub accept: Accept,
    pub ret: Option<Ret>,
}

impl<Ret> State<Ret> {
    pub fn new(accept: Accept, ret: Option<Ret>) -> State<Ret> {
        State {
            transitions: RangeMap::new(),
            accept: accept,
            ret: ret,
        }
    }
}

pub trait RetTrait: Clone + Copy + Debug + Eq + Hash {}
impl<T: Clone + Copy + Debug + Eq + Hash> RetTrait for T {}

#[derive(Clone, PartialEq)]
pub struct Dfa<Ret: 'static> {
    pub states: Vec<State<Ret>>,

    /// This is a vector of length `Look::num()` containing all possible starting positions.
    ///
    /// `init[Look::Boundary]` is the starting position if we are at the beginning of the
    /// input.
    ///
    /// `init[Look::Full]` is the default starting position.
    ///
    /// All other positions in `init` are only used if we are specifically asked to start
    /// there; this is mainly useful in the forward-backward engine.
    pub init: Vec<Option<StateIdx>>,
}

impl<Ret: RetTrait> Dfa<Ret> {
    /// Returns a `Dfa` with no states.
    pub fn new() -> Dfa<Ret> {
        Dfa {
            states: Vec::new(),
            init: vec![None; Look::num()],
        }
    }

    /// Returns the number of states.
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub fn add_state(&mut self, accept: Accept, ret: Option<Ret>) -> StateIdx {
        self.states.push(State::new(accept, ret));
        self.states.len() - 1
    }

    pub fn set_transitions(&mut self, from: StateIdx, transitions: RangeMap<u8, StateIdx>) {
        self.states[from].transitions = transitions;
    }

    pub fn init_state(&self, look: Look) -> Option<StateIdx> {
        self.init[look.as_usize()]
    }

    pub fn init_at_start(&self) -> Option<StateIdx> {
        self.init_state(Look::Boundary)
    }

    pub fn init_otherwise(&self) -> Option<StateIdx> {
        self.init_state(Look::Full)
    }

    /// Returns true if this `Dfa` only matches things at the beginning of the input.
    pub fn is_anchored(&self) -> bool {
        self.init_otherwise().is_none() && self.init_at_start().is_some()
    }

    /// Get transitions from a given state.
    pub fn transitions(&self, state: StateIdx) -> &RangeMap<u8, StateIdx> {
        &self.states[state].transitions
    }

    /// Returns the conditions under which the given state accepts.
    pub fn accept(&self, state: StateIdx) -> &Accept {
        &self.states[state].accept
    }

    /// The value that will be returned if we accept in state `state`.
    pub fn ret(&self, state: StateIdx) -> Option<&Ret> {
        self.states[state].ret.as_ref()
    }

    /// Changes the return value.
    pub fn map_ret<T: RetTrait, F: FnMut(Ret) -> T>(self, mut f: F) -> Dfa<T> {
        let mut ret: Dfa<T> = Dfa::new();
        ret.init = self.init;

        for st in self.states {
            let new_st = State {
                transitions: st.transitions,
                accept: st.accept,
                ret: st.ret.map(&mut f),
            };
            ret.states.push(new_st);
        }
        ret
    }

    /// Returns an equivalent DFA with a minimal number of states.
    ///
    /// Uses Hopcroft's algorithm.
    fn minimize(&self) -> Dfa<Ret> {
        Minimizer::minimize(self)
    }

    /// Returns the transitions of this automaton, reversed.
    fn reversed_transitions(&self) -> Vec<RangeMultiMap<u8, StateIdx>> {
        let mut ret = vec![RangeMultiMap::new(); self.states.len()];

        for (source, st) in self.states.iter().enumerate() {
            for &(range, target) in st.transitions.ranges_values() {
                ret[target].insert(range, source);
            }
        }

        ret
    }

    /// Returns a set of strings that match the beginning of this `Dfa`.
    ///
    /// If the set is non-empty, every match of this `Dfa` is guaranteed to start with one of these
    /// strings.
    pub fn prefix_strings(&self) -> Vec<PrefixPart> {
        // It might seem silly to look for prefixes starting at the anchored state, but it's useful
        // for forward-backward matching. In cases where the regex is honestly anchored, we won't
        // ask to make a prefix anyway.
        if let Some(state) = self.init_state(Look::Boundary) {
            PrefixSearcher::extract(self, state)
        } else {
            Vec::new()
        }
    }

    /*
    pub fn critical_strings(&self) -> Vec<(Vec<u8>, StateIdx)> {
        unimplemented!();
    }
    */

    // Finds the bytes that are treated equivalently by this Dfa.
    //
    // Returns a Vec of length 256 such that vec[i] == vec[j] when i and j are two equivalent
    // bytes. Also returns the log of the number of classes, rounded up.
    fn byte_equivalence_classes(&self) -> (Vec<u8>, u32) {
        let mut part = Partition::new(Some(0..256).into_iter(), 256);
        let mut buf = Vec::with_capacity(256);

        for st in &self.states {
            let group = st.transitions.keys_values().group_by_lazy(|x| x.1);
            for (_, keys_values) in &group {
                buf.clear();
                for (key, _) in keys_values {
                    buf.push(key as usize);
                }
                part.refine(&buf);
            }
        }

        let mut ret = vec![0; 256];
        for (i, p) in part.iter().enumerate() {
            for &x in p {
                ret[x] = i as u8;
            }
        }
        let size = (part.num_parts() - 1) as u32;

        (ret, 32 - size.leading_zeros())
    }

    /// Compiles this `Dfa` into instructions for execution.
    pub fn compile(&self) -> TableInsts<Ret> {
        let (byte_class, log_num_classes) = self.byte_equivalence_classes();

        let mut table = vec![u32::MAX; self.num_states() << log_num_classes];
        let accept: Vec<Option<Ret>> = self
            .states
            .iter()
            .map(|st| {
                if st.accept == Accept::Always {
                    st.ret
                } else {
                    None
                }
            })
            .collect();
        let accept_at_eoi: Vec<Option<Ret>> = self
            .states
            .iter()
            .map(|st| {
                if st.accept != Accept::Never {
                    st.ret
                } else {
                    None
                }
            })
            .collect();

        for (idx, st) in self.states.iter().enumerate() {
            for (ch, &tgt_state) in st.transitions.keys_values() {
                let class = byte_class[ch as usize];
                table[(idx << log_num_classes) + class as usize] = tgt_state as u32;
            }
        }

        TableInsts {
            log_num_classes: log_num_classes,
            byte_class: byte_class,
            accept: accept,
            accept_at_eoi: accept_at_eoi,
            table: table,
        }
    }

    /// Finds an equivalent DFA with the minimal number of states.
    pub fn optimize(self) -> Dfa<Ret> {
        let mut ret = self.minimize();
        ret.sort_states();
        ret
    }

    /// Deletes any transitions that return to the initial state.
    ///
    /// This results in a new Dfa with the following properties:
    /// - if the original Dfa has a match then the new Dfa also has a match that ends in the same
    ///   position (and vice versa), and
    /// - the new Dfa doesn't need to backtrack to find matches: if it fails then it can be
    ///   restarted at the same position it failed in.
    ///
    /// The reason for this method is that it makes prefixes more effective: where the original Dfa
    /// would just loop back to the start state, the new Dfa will signal a failure. Then we can use
    /// a `Prefix` to scan ahead for a good place to resume matching.
    ///
    /// # Panics
    /// - if `self` is not anchored.
    pub fn cut_loop_to_init(mut self) -> Dfa<Ret> {
        if !self.is_anchored() {
            panic!("only anchored Dfas can be cut");
        }

        // The unwrap is safe because we just checked that we are anchored.
        let init = self.init_at_start().unwrap();
        for st in &mut self.states {
            st.transitions.retain_values(|x| *x != init);
        }
        self
    }

    fn map_states<F: FnMut(StateIdx) -> StateIdx>(&mut self, mut map: F) {
        for st in &mut self.states {
            st.transitions.map_values(|x| map(*x));
        }
        let init: Vec<_> = self.init.iter().map(|x| x.map(&mut map)).collect();
        self.init = init;
    }

    /// Sorts states in depth-first alphabetical order.
    ///
    /// This has the following advantages:
    /// - the construction of a `Dfa` becomes deterministic: without sorting, the states aren't in
    ///   deterministic order because `minimize` using hashing.
    /// - better locality: after sorting, many transitions just go straight to the next state.
    /// - we prune unreachable states.
    fn sort_states(&mut self) {
        let sorted = self.dfs_order(self.init.iter().filter_map(|x| *x));

        // Not every old state will necessary get mapped to a new one (unreachable states won't).
        let mut state_map: Vec<Option<StateIdx>> = vec![None; self.states.len()];
        let mut old_states = vec![State::new(Accept::Never, None); self.states.len()];
        mem::swap(&mut old_states, &mut self.states);

        for (new_idx, old_idx) in sorted.into_iter().enumerate() {
            state_map[old_idx] = Some(new_idx);
            mem::swap(&mut old_states[old_idx], &mut self.states[new_idx]);
        }

        // Fix the transitions and initialization to point to the new states. The `unwrap` here is
        // basically the assertion that all reachable states should be mapped to new states.
        self.map_states(|s| state_map[s].unwrap());
    }

    /*
    // Finds all the transitions between states that only match a single byte.
    fn single_byte_transitions(&self) -> HashMap<(StateIdx, StateIdx), u8> {
        use std::collections::hash_map::Entry::*;

        let mut ret = HashMap::new();
        let mut seen = HashSet::new();
        for (src_idx, st) in self.states.iter().enumerate() {
            for &(range, tgt_idx) in st.transitions.ranges_values() {
                if range.start == range.end && !seen.contains(&(src_idx, tgt_idx)) {
                    match ret.entry((src_idx, tgt_idx)) {
                        Occupied(e) => {
                            e.remove();
                            seen.insert((src_idx, tgt_idx));
                        },
                        Vacant(e) => { e.insert(range.start); },
                    }
                }
            }
        }
        ret
    }

    // Finds all the single-byte transitions that must be traversed in order to get to an accepting
    // state.
    fn mandatory_single_byte_transitions(&self, max_steps: usize) -> Vec<(StateIdx, StateIdx, u8)> {
        let map = self.single_byte_transitions();
        let interesting_bytes: HashSet<u8> = map.values().cloned().collect();

        // In order to get from the initial state to state i, we need to see all the bytes in
        // mandatory_bytes[i] at least once. (At least, that's the goal of mandatory_bytes; we
        // start out with too many elements in it and gradually remove them.)
        let mut mandatory_bytes = vec![interesting_bytes.clone(); self.num_states()];
        mandatory_bytes[0] = HashSet::new();

        let mut visited = HashSet::<StateIdx>::new();
        let mut active = HashSet::<StateIdx>::new();
        let mut next = HashSet::<StateIdx>::new();
        next.insert(0);
        let mut steps_left = max_steps;

        fn intersect(a: &mut HashSet<u8>, b: &HashSet<u8>) -> bool {
            let old_size = a.len();
            *a = a.intersection(b).cloned().collect();
            a.len() < old_size
        }

        while steps_left > 0 {
            steps_left -= 1;
            mem::swap(&mut active, &mut next);
            next.clear();

            for &src in &active {
                // If we found an accepting state, keep it in the active set but don't go any
                // further.
                if self.accept(src) != &Accept::Never {
                    next.insert(src);
                    continue;
                }

                visited.insert(src);
                for tgt in self.transitions(src).ranges_values().map(|x| x.1).dedup() {
                    let mut bytes = mandatory_bytes[src].clone();
                    if let Some(b) = map.get(&(src, tgt)) {
                        bytes.insert(*b);
                    }
                    if intersect(&mut mandatory_bytes[tgt], &bytes) || !visited.contains(&tgt) {
                        next.insert(tgt);
                    }
                }
            }
        }

        let critical_bytes = next.into_iter()
            .fold(interesting_bytes,
                  |x, state| x.intersection(&mandatory_bytes[state]).cloned().collect());

        let mut ret: Vec<_> = map.into_iter()
            .filter(|&(pair, byte)| critical_bytes.contains(&byte) && visited.contains(&pair.0))
            .map(|(pair, byte)| (pair.0, pair.1, byte))
            .collect();
        ret.sort();
        ret
    }
    */
}

impl<Ret: Debug> Debug for Dfa<Ret> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.write_fmt(format_args!("Dfa ({} states):\n", self.states.len()))?;

        f.write_fmt(format_args!("Init: {:?}\n", self.init))?;

        for (st_idx, st) in self.states.iter().enumerate().take(40) {
            f.write_fmt(format_args!(
                "\tState {} (accepting: {:?}):\n",
                st_idx, st.accept
            ))?;
            if let Some(ref ret) = st.ret {
                f.write_fmt(format_args!("\t\t{:?}\n", ret))?;
            }

            if !st.transitions.is_empty() {
                f.write_str("\t\tTransitions:\n")?;
                // Cap it at 5 transitions, since it gets unreadable otherwise.
                for &(range, target) in st.transitions.ranges_values().take(5) {
                    f.write_fmt(format_args!(
                        "\t\t\t{} -- {} => {}\n",
                        range.start, range.end, target
                    ))?;
                }
                if st.transitions.num_ranges() > 5 {
                    f.write_str("\t\t\t...\n")?;
                }
            }
        }
        if self.states.len() > 40 {
            f.write_fmt(format_args!(
                "\t...({} more states)\n",
                self.states.len() - 40
            ))?;
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod tests {
    use crate::dfa::*;
    use crate::error::Error;
    use crate::look::Look;
    use crate::nfa::{Accept, Nfa, StateIdx};
    use itertools::Itertools;
    use range_map::{Range, RangeMap};
    use std::usize;

    // Creates a non-backtracking dfa from a regex string.
    pub fn make_dfa_bounded(re: &str, max_states: usize) -> Result<Dfa<(Look, u8)>, Error> {
        let nfa = Nfa::from_regex(re)?;
        let nfa = nfa.remove_looks();
        println!("after remove_looks: {:?}", nfa);
        let nfa = nfa.byte_me(max_states)?;
        println!("after byte: {:?}", nfa);

        let dfa = nfa.determinize(max_states)?;
        Ok(dfa.optimize())
    }

    pub fn make_dfa(re: &str) -> Result<Dfa<(Look, u8)>, Error> {
        make_dfa_bounded(re, usize::MAX)
    }

    pub fn make_anchored(re: &str) -> Dfa<(Look, u8)> {
        let nfa = Nfa::from_regex(re)
            .unwrap()
            .remove_looks()
            .byte_me(usize::MAX)
            .unwrap()
            .anchor(usize::MAX)
            .unwrap();

        nfa.determinize(usize::MAX)
            .unwrap()
            .optimize()
            .cut_loop_to_init()
            .optimize()
    }

    pub fn trans_dfa_anchored(
        size: usize,
        trans: &[(StateIdx, StateIdx, Range<u8>)],
    ) -> Dfa<(Look, u8)> {
        let mut ret = Dfa::new();
        for _ in 0..size {
            ret.add_state(Accept::Never, None);
        }
        for (src, trans) in trans.iter().group_by(|x| x.0) {
            let rm: RangeMap<u8, usize> = trans.into_iter().map(|x| (x.2, x.1)).collect();
            ret.set_transitions(src, rm);
        }
        ret
    }

    #[test]
    fn test_anchored_dfa_simple() {
        let dfa = make_anchored("a");
        let mut tgt = trans_dfa_anchored(2, &[(0, 1, Range::new(b'a', b'a'))]);
        tgt.init[Look::Boundary.as_usize()] = Some(0);
        tgt.states[1].accept = Accept::Always;
        tgt.states[1].ret = Some((Look::Full, 0));

        assert_eq!(dfa, tgt);
    }

    #[test]
    fn test_forward_backward_simple() {
        // TODO
    }

    #[test]
    fn test_anchored_dfa_anchored_end() {
        let dfa = make_anchored("a$");
        let mut tgt = trans_dfa_anchored(
            2,
            &[
                (0, 1, Range::new(b'a', b'a')),
                (1, 1, Range::new(b'a', b'a')),
            ],
        );
        tgt.init[Look::Boundary.as_usize()] = Some(0);
        tgt.states[1].accept = Accept::AtEoi;
        tgt.states[1].ret = Some((Look::Boundary, 0));

        assert_eq!(dfa, tgt);
    }

    #[test]
    fn test_anchored_dfa_literal_prefix() {
        let dfa = make_anchored("abc[A-z]");
        let pref = dfa
            .prefix_strings()
            .into_iter()
            .map(|p| p.0)
            .collect::<Vec<_>>();
        assert_eq!(pref, vec!["abc".as_bytes()]);
    }

    #[test]
    fn test_minimize() {
        let auto = make_dfa("a*?b*?").unwrap();
        // 1, because our highest-priority match is an empty string.
        assert_eq!(auto.states.len(), 1);

        let auto = make_dfa(r"^a").unwrap();
        assert_eq!(auto.states.len(), 2);

        let mut auto = make_dfa("[cgt]gggtaaa|tttaccc[acg]").unwrap();
        // Since `minimize` is non-deterministic (involving random hashes), run this a bunch of
        // times.
        for _ in 0..100 {
            auto = auto.optimize();
            assert_eq!(auto.states.len(), 16);
        }
    }

    #[test]
    fn test_class_normalized() {
        let mut re = make_dfa("[abcdw]").unwrap();
        re.sort_states();
        assert_eq!(re.states.len(), 2);
        assert_eq!(re.states[0].transitions.num_ranges(), 2)
    }

    #[test]
    fn test_max_states() {
        assert!(make_dfa_bounded("foo", 3).is_err());
        assert!(make_dfa_bounded("foo", 4).is_ok());
    }

    #[test]
    fn test_adjacent_predicates() {
        assert!(make_dfa_bounded(r"\btest\b\B", 100)
            .unwrap()
            .states
            .is_empty());
        assert!(make_dfa_bounded(r"\btest\B\b", 100)
            .unwrap()
            .states
            .is_empty());
        assert!(make_dfa_bounded(r"test1\b\Btest2", 100)
            .unwrap()
            .states
            .is_empty());
    }

    #[test]
    fn test_syntax_error() {
        assert!(make_dfa_bounded("(abc", 10).is_err());
    }

    #[test]
    fn match_priority() {
        macro_rules! eq {
            ($re1:expr, $re2:expr) => {{
                let dfa1 = make_dfa($re1).unwrap();
                let dfa2 = make_dfa($re2).unwrap();
                assert_eq!(dfa1, dfa2);
            }};
        }
        eq!("(a|aa)", "a");
        eq!("abcd*?", "abc");
        //eq!("a*?", ""); // TODO: figure out how empty regexes should behave
    }

    // TODO: add a test checking that minimize() doesn't clobber return values.

    /*
    #[test]
    fn critical_transitions() {
        fn crit(max_steps: usize, re: &str, answer: &[(StateIdx, StateIdx, u8)]) {
            let dfa = make_dfa(re).unwrap();
            println!("{:?}", dfa);
            assert_eq!(&dfa.mandatory_single_byte_transitions(max_steps)[..], answer);
        }

        fn crit_anchored(max_steps: usize, re: &str, answer: &[(StateIdx, StateIdx, u8)]) {
            let dfa = make_anchored(re);
            println!("{:?}", dfa);
            assert_eq!(&dfa.mandatory_single_byte_transitions(max_steps)[..], answer);
        }

        crit(10, "a", &[(0, 1, b'a')]);
        crit(10, "aaa", &[(0, 1, b'a'), (1, 2, b'a'), (2, 3, b'a')]);
        crit(2, "aaa", &[(0, 1, b'a'), (1, 2, b'a')]);
        crit(10, "a*|ab", &[]);
        crit(10, "a+|ab", &[(0, 1, b'a')]);
        crit(10, "brown|fox", &[(2, 3, b'o'), (6, 7, b'o')]);
        crit(10, "quick|brown", &[]);
        crit(10, "zzzzzzzzzz|abracadabraz", &[]);
        crit(10, "eeeeeeeeez|abracadabz", &[(9, 10, b'z')]);
        crit(10, ".*x", &[(0, 1, b'x')]);
        crit_anchored(10, "\\bx", &[(0, 260, b'x')]);
    }
    */
}
