// Copyright 2015-2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::dfa::Dfa;
use crate::error::Error;
use crate::look::Look;
use crate::nfa::{Accept, Nfa, NoLooks, State, StateIdx, StateSet};
use crate::Result;
use itertools::Itertools;
use num_traits::PrimInt;
use range_map::{Range, RangeMap, RangeMultiMap};
use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::swap;
use std::{char, u8, usize};
use utf8_ranges::{Utf8Range, Utf8Sequence, Utf8Sequences};

// This provides a more compact way of representing UTF-8 sequences.
//
// A sequence of bytes belongs to this set if its first byte is in `head[0]`, its second byte is
// in `head[1]`, etc., and its last byte belongs to one of the ranges in `last_byte`.
//
// This representation is handy for making NFAs because compared to the representation in
// `Utf8Sequences`, it adds many fewer states. Basically, we are doing some crude minimization
// before creating the states.
struct MergedUtf8Sequences {
    pub head: Vec<Utf8Range>,
    pub last_byte: Vec<Utf8Range>,
}

// Returns this range as a pair of chars, or none if this is an empty range.
fn to_char_pair(r: Range<u32>) -> Option<(char, char)> {
    // Round up self.start to the nearest legal codepoint.
    let start = if r.start > 0xD7FF && r.start < 0xE000 {
        0xE000
    } else {
        r.start
    };

    // Round down self.end.
    let end = if r.end > 0x10FFFF {
        0x10FFFF
    } else if r.end < 0xE000 && r.end > 0xD7FF {
        0xD7FF
    } else {
        r.end
    };

    if start > end {
        None
    } else {
        Some((char::from_u32(start).unwrap(), char::from_u32(end).unwrap()))
    }
}

impl MergedUtf8Sequences {
    // Panics if not all the input sequences have the same leading byte ranges.
    fn merge<I>(iter: I) -> MergedUtf8Sequences
    where
        I: Iterator<Item = Utf8Sequence>,
    {
        let mut head = Vec::new();
        let mut last_byte = Vec::new();

        for seq in iter {
            let len = seq.len();
            let h = &seq.as_slice()[..len - 1];
            if head.is_empty() {
                head.extend_from_slice(h);
            } else if &head[..] != h {
                panic!("invalid sequences to merge");
            }

            last_byte.push(seq.as_slice()[len - 1]);
        }

        MergedUtf8Sequences {
            head: head,
            last_byte: last_byte,
        }
    }

    fn from_sequences<'a, I>(iter: I) -> Box<dyn Iterator<Item = MergedUtf8Sequences> + 'a>
    where
        I: Iterator<Item = Utf8Sequence> + 'a,
    {
        fn head(u: &Utf8Sequence) -> Vec<Utf8Range> {
            let len = u.len();
            u.as_slice()[..len - 1].to_owned()
        }

        Box::new(
            iter.group_by(head)
                .into_iter()
                .map(|(_, seqs)| MergedUtf8Sequences::merge(seqs.into_iter())),
        )
    }

    fn from_ranges<'a, I>(iter: I) -> Box<dyn Iterator<Item = MergedUtf8Sequences> + 'a>
    where
        I: Iterator<Item = Range<u32>> + 'a,
    {
        MergedUtf8Sequences::from_sequences(
            iter.filter_map(to_char_pair)
                .flat_map(|r| Utf8Sequences::new(r.0, r.1)),
        )
    }

    fn num_bytes(&self) -> u8 {
        (self.head.len() + 1) as u8
    }
}

// Creates a byte-based Dfa that matches all the chars in `look.as_set()`.
fn make_char_dfa(look: Look) -> Dfa<(Look, u8)> {
    let mut nfa: Nfa<u32, NoLooks> = Nfa::with_capacity(2);
    nfa.add_state(Accept::Never);
    nfa.add_look_ahead_state(look, 1, 0);
    // TODO: shouldn't adding both Full and Boundary be redundant?
    nfa.init.push((Look::Full, 0));
    nfa.init.push((Look::Boundary, 0));
    nfa.states[0].consuming =
        RangeMultiMap::from_vec(look.as_set().ranges().map(|x| (x, 1)).collect());

    // These unwraps are OK because the only failures are caused by having too many states.
    nfa.byte_me(usize::MAX)
        .unwrap()
        .determinize(usize::MAX)
        .unwrap()
        .optimize()
}

// Creates a byte-based Dfa that matches backwards all the chars in `look.as_set()`.
fn make_rev_char_dfa(look: Look) -> Dfa<(Look, u8)> {
    let mut nfa: Nfa<u8, NoLooks> = Nfa::with_capacity(0); // TODO: better capacity
    nfa.add_state(Accept::Never);
    nfa.init.push((Look::Full, 0));
    nfa.init.push((Look::Boundary, 0));

    // This is more-or-less C&P from add_utf8_sequence.
    for seq in MergedUtf8Sequences::from_ranges(look.as_set().ranges()) {
        let mut last_state = nfa.add_state(Accept::Never);

        for range in &seq.last_byte {
            nfa.add_transition(0, last_state, Range::new(range.start, range.end));
        }
        for range in seq.head.iter().rev() {
            let cur_state = nfa.add_state(Accept::Never);

            nfa.add_transition(last_state, cur_state, Range::new(range.start, range.end));
            last_state = cur_state;
        }

        nfa.states[last_state].accept = Accept::Always;
        nfa.states[last_state].accept_look = look;
        nfa.states[last_state].accept_state = 0;
        nfa.states[last_state].accept_tokens = seq.num_bytes();
    }

    // This unwrap is OK because the only failures are caused by having too many states.
    nfa.determinize(usize::MAX).unwrap().optimize()
}

// We cache optimized Dfas for the expensive looks. See `Nfa<u8, NoLooks>::add_min_utf8_sequences`
// for an explanation.
lazy_static! {
    static ref WORD_CHAR_DFA: Dfa<(Look, u8)> = make_char_dfa(Look::WordChar);
    static ref NOT_WORD_CHAR_DFA: Dfa<(Look, u8)> = make_char_dfa(Look::NotWordChar);
    static ref REV_WORD_CHAR_DFA: Dfa<(Look, u8)> = make_rev_char_dfa(Look::WordChar);
    static ref REV_NOT_WORD_CHAR_DFA: Dfa<(Look, u8)> = make_rev_char_dfa(Look::NotWordChar);
}

impl<Tok: Debug + PrimInt> Nfa<Tok, NoLooks> {
    // Returns the set of all states that can be reached from some initial state.
    fn reachable_from<I>(&self, states: I) -> HashSet<StateIdx>
    where
        I: Iterator<Item = StateIdx>,
    {
        let mut active: HashSet<StateIdx> = states.collect();
        let mut next_active: HashSet<StateIdx> = HashSet::new();
        let mut ret = active.clone();

        while !active.is_empty() {
            for &s in &active {
                for &(_, t) in self.states[s].consuming.ranges_values() {
                    if !ret.contains(&t) {
                        ret.insert(t);
                        next_active.insert(t);
                    }
                }
            }
            swap(&mut active, &mut next_active);
            next_active.clear();
        }
        ret
    }

    // Reverses this Nfa, but only the transitions (i.e. doesn't do anything about initial and
    // final states).
    fn reversed_simple(&self) -> Nfa<Tok, NoLooks> {
        let rev_transitions = self.reversed_transitions();
        let mut ret: Nfa<Tok, NoLooks> = Nfa::with_capacity(self.states.len());

        for trans in rev_transitions {
            let idx = ret.add_state(Accept::Never);
            ret.states[idx].consuming = trans;
        }

        ret
    }

    // Returns the set of all states that can be reached from an initial state and that can reach
    // some accepting state.
    fn reachable_states(&self) -> HashSet<StateIdx> {
        let init_states = self.init.iter().map(|pair| pair.1);
        let final_states = self
            .states
            .iter()
            .enumerate()
            .filter(|&(_, state)| state.accept != Accept::Never)
            .map(|(idx, _)| idx);

        let forward = self.reachable_from(init_states);
        let backward = self.reversed_simple().reachable_from(final_states);
        forward.intersection(&backward).cloned().collect()
    }

    /// Optimizes this Nfa by removing all states that cannot be reached from an initial state
    /// and all states that cannot lead to an accepting state.
    pub fn trim_unreachable(&mut self) {
        let reachable = self.reachable_states();

        let mut old_states = Vec::new();
        swap(&mut self.states, &mut old_states);
        let mut old_to_new = vec![None; old_states.len()];

        let (new_to_old, new_states): (Vec<_>, Vec<State<Tok>>) = old_states
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| reachable.contains(&i))
            .unzip();
        self.states = new_states;

        for (new, &old) in new_to_old.iter().enumerate() {
            old_to_new[old] = Some(new);
        }

        self.map_states(|s| old_to_new[s]);
    }

    // Returns an `Accept` that will accept whenever anything in `states` would accept.
    fn accept_union(&self, states: &StateSet) -> Accept {
        states
            .iter()
            .map(|s| self.states[*s].accept)
            .max()
            .unwrap_or(Accept::Never)
    }
}

impl Nfa<u32, NoLooks> {
    /// Converts this `Nfa` into one that consumes the input byte-by-byte.
    pub fn byte_me(self, max_states: usize) -> Result<Nfa<u8, NoLooks>> {
        let mut ret = Nfa::<u8, NoLooks> {
            states: self
                .states
                .iter()
                .map(|s| State {
                    accept: s.accept,
                    accept_look: s.accept_look,
                    accept_state: s.accept_state,
                    accept_tokens: s.accept_tokens,
                    consuming: RangeMultiMap::new(),
                    looking: Vec::new(),
                })
                .collect(),
            init: self.init,
            phantom: PhantomData,
        };

        for (i, state) in self.states.into_iter().enumerate() {
            // Group transitions by the target state, and add them in batches. Most of the time, we
            // can merge a bunch of Utf8Sequences before adding them, which saves a bunch of
            // states.
            for (tgt, transitions) in state.consuming.ranges_values().group_by(|x| x.1) {
                ret.add_utf8_sequences(i, transitions.into_iter().map(|x| x.0), tgt, max_states)?;
            }
        }
        Ok(ret)
    }
}

impl Nfa<u8, NoLooks> {
    /// Converts this `Nfa` into a `Dfa`.
    pub fn determinize(&self, max_states: usize) -> Result<Dfa<(Look, u8)>> {
        Determinizer::determinize(
            self,
            max_states,
            MatchChoice::TransitionOrder,
            self.init.clone(),
        )
    }

    /// Converts this `Nfa` into a `Dfa`.
    ///
    /// Whenever this `Nfa` matches some text, the `Dfa` also will. But if this `Nfa` has multiple
    /// possible endpoints for a match then the returned `Dfa` is only guaranteed to match the
    /// longest one.
    pub fn determinize_longest(&self, max_states: usize) -> Result<Dfa<(Look, u8)>> {
        Determinizer::determinize(
            self,
            max_states,
            MatchChoice::LongestMatch,
            self.init.clone(),
        )
    }

    /// Returns the reversal of this `Nfa`.
    ///
    /// If `self` matches some string of bytes, then the return value of this method will match
    /// the same strings of bytes reversed.
    ///
    /// Note that this loses information about match priorities.
    pub fn reverse(&self, max_states: usize) -> Result<Nfa<u8, NoLooks>> {
        let mut ret = self.reversed_simple();

        // Turn our initial states into ret's accepting states.
        for &(look, i) in &self.init {
            match look {
                Look::Full => {
                    ret.states[i].accept = Accept::Always;
                    ret.states[i].accept_look = Look::Full;
                }
                Look::Boundary => {
                    ret.states[i].accept = max(ret.states[i].accept, Accept::AtEoi);
                    ret.states[i].accept_look = max(ret.states[i].accept_look, Look::Boundary);
                }
                Look::NewLine => {
                    let accept_state = ret.add_look_ahead_state(Look::NewLine, 1, i);
                    ret.add_transition(i, accept_state, Range::new(b'\n', b'\n'));
                    ret.states[i].accept = max(ret.states[i].accept, Accept::AtEoi);
                    ret.states[i].accept_look = max(ret.states[i].accept_look, Look::Boundary);
                }
                Look::WordChar | Look::NotWordChar => {
                    // It would make more sense to put this outside the loop, but having it inside
                    // prevents a deadlock: constructing REV_*_DFA ends up calling reverse(), but
                    // with no look-ahead so it never gets inside this loop.
                    let dfa: &Dfa<_> = if look == Look::WordChar {
                        &REV_WORD_CHAR_DFA
                    } else {
                        ret.states[i].accept = max(ret.states[i].accept, Accept::AtEoi);
                        ret.states[i].accept_look = max(ret.states[i].accept_look, Look::Boundary);
                        &REV_NOT_WORD_CHAR_DFA
                    };
                    let accept_state = ret.add_look_ahead_state(look, 1, i);
                    ret.add_min_utf8_sequences(i, dfa, accept_state, max_states)?;
                }
                Look::Empty => {
                    panic!("Empty cannot be an init look");
                }
            }
        }

        // Turn our accepting states into ret's initial states.
        ret.init.clear();
        for st in &self.states {
            if st.accept != Accept::Never {
                ret.init.push((st.accept_look, st.accept_state));
            }
        }
        Ok(ret)
    }

    /// Can we accept immediately if the beginning of the input matches `look`?
    fn init_accept(&self, look: Look) -> Accept {
        let set = self
            .init
            .iter()
            .filter(|pair| look <= pair.0)
            .map(|pair| pair.1)
            .collect::<Vec<_>>();
        self.accept_union(&set)
    }

    /// This essentially modifies `self` by adding a `^.*` at the beginning.
    ///
    /// The result is actually a little bit different, because `.` matches a whole code point,
    /// whereas the `^.*` that we add works at the byte level.
    pub fn anchor(mut self, max_states: usize) -> Result<Nfa<u8, NoLooks>> {
        let loop_accept = self.init_accept(Look::Full);
        let loop_state = self.add_state(loop_accept);
        let init_accept = self.init_accept(Look::Boundary);
        let init_state = self.add_state(init_accept);

        // Swap out init so that we can iterate over it while modifying `self`.
        let mut init = Vec::new();
        swap(&mut init, &mut self.init);

        for &(look, st_idx) in &init {
            if look.allows_eoi() {
                // TODO: shouldn't need to clone here.
                for &(range, target) in self.states[st_idx].consuming.clone().ranges_values() {
                    self.add_transition(init_state, target, range);
                }
            }

            match look {
                Look::Boundary => {}
                Look::Full => {
                    for &(range, target) in self.states[st_idx].consuming.clone().ranges_values() {
                        self.add_transition(loop_state, target, range);
                    }
                }
                Look::NewLine => {
                    self.add_transition(init_state, st_idx, Range::new(b'\n', b'\n'));
                    self.add_transition(loop_state, st_idx, Range::new(b'\n', b'\n'));
                }
                Look::WordChar | Look::NotWordChar => {
                    let dfa: &Dfa<_> = if look == Look::WordChar {
                        &WORD_CHAR_DFA
                    } else {
                        &NOT_WORD_CHAR_DFA
                    };

                    self.add_min_utf8_sequences(loop_state, dfa, st_idx, max_states)?;
                    self.add_min_utf8_sequences(init_state, dfa, st_idx, max_states)?;
                }
                Look::Empty => {
                    panic!("Cannot start with an empty look");
                }
            }

            // Once we've found an init state that accepts immediately, don't look for any others
            // (since any matches that we find starting from them are lower priority that the one
            // we've found already). This check is *almost* unnecessary, since similar pruning
            // happens when we turn the NFA into a DFA. The important case that needs to be handled
            // here is the case that a high-priority init state has no transitions out of it. Such
            // a state will be completely removed by this function, and so we need to acknowledge
            // its existence here.
            if self.states[st_idx].accept == Accept::Always {
                break;
            }
        }

        // Wire up the initial and loop states, but only if they aren't accepting. That's because
        // if they are accepting then the accept should take priority over the transition (since
        // making the transition means that we are searching for a match that starts later).
        if init_accept != Accept::Always {
            self.add_transition(init_state, loop_state, Range::full());
        }
        if loop_accept != Accept::Always {
            self.add_transition(loop_state, loop_state, Range::full());
        }

        // The new Nfa is only allowed to start at the beginning of the input, and only at the new
        // initial state.
        self.init.push((Look::Boundary, init_state));
        self.trim_unreachable();
        Ok(self)
    }

    // This does the same thing as add_utf8_sequences, but it gets the transitions from a dfa,
    // which should have zero as its only starting state, and for which every accepting state
    // should be Accept::Always.
    //
    // This is probably used in conjunction with make_char_dfa, which ends up having the same
    // effect as add_utf8_sequences, but adds fewer states.
    fn add_min_utf8_sequences(
        &mut self,
        start_state: StateIdx,
        dfa: &Dfa<(Look, u8)>,
        end_state: StateIdx,
        max_states: usize,
    ) -> Result<()> {
        let offset = self.states.len();
        // If end_accept is true, then it isn't actually important that we end in state
        // `end_state`: we can create a new look_ahead state to end in.
        let end_accept = self.states[end_state].accept_tokens > 0;

        if self.states.len() + dfa.num_states() > max_states {
            return Err(Error::TooManyStates);
        }
        for _ in 0..dfa.num_states() {
            self.add_state(Accept::Never);
        }
        for d_idx in 0..dfa.num_states() {
            let n_src = if d_idx == 0 {
                start_state
            } else {
                d_idx + offset
            };
            for &(range, d_tgt) in dfa.transitions(d_idx).ranges_values() {
                let n_tgt = if dfa.accept(d_tgt) == &Accept::Always && !end_accept {
                    end_state
                } else {
                    let n_tgt = d_tgt + offset;
                    self.states[n_tgt].accept = *dfa.accept(d_tgt);
                    if let Some(&(look, bytes)) = dfa.ret(d_tgt) {
                        self.states[n_tgt].accept_look = look;
                        self.states[n_tgt].accept_state = start_state;
                        self.states[n_tgt].accept_tokens = bytes;
                    }
                    n_tgt
                };
                self.add_transition(n_src, n_tgt, range);
            }
        }

        Ok(())
    }

    // Adds a path from `start_state` to `end_state` for all byte sequences matching `seq`.
    //
    // If `end_state` is a look-ahead state, makes a new accepting state instead (so that we know
    // how many bytes of look-ahead we used).
    fn add_utf8_sequence(
        &mut self,
        start_state: StateIdx,
        mut end_state: StateIdx,
        seq: MergedUtf8Sequences,
    ) {
        let mut last_state = start_state;
        for range in &seq.head {
            let cur_state = self.add_state(Accept::Never);

            self.add_transition(last_state, cur_state, Range::new(range.start, range.end));
            last_state = cur_state;
        }

        if self.states[end_state].accept_tokens > 0 {
            let look = self.states[end_state].accept_look;
            let acc_state = self.states[end_state].accept_state;
            end_state = self.add_look_ahead_state(look, seq.num_bytes(), acc_state);
        }
        for range in &seq.last_byte {
            self.add_transition(last_state, end_state, Range::new(range.start, range.end));
        }
    }

    // Adds a byte path from `start_state` to `end_state` for every char in `ranges`.
    fn add_utf8_sequences<I>(
        &mut self,
        start_state: StateIdx,
        ranges: I,
        end_state: StateIdx,
        max_states: usize,
    ) -> Result<()>
    where
        I: Iterator<Item = Range<u32>>,
    {
        for m in MergedUtf8Sequences::from_ranges(ranges) {
            self.add_utf8_sequence(start_state, end_state, m);
            if self.states.len() > max_states {
                return Err(Error::TooManyStates);
            }
        }
        Ok(())
    }

    // Finds the transitions out of the given set of states, as a RangeMap.
    fn transition_map(&self, states: &[StateIdx]) -> RangeMap<u8, Vec<usize>> {
        let mut transitions = states
            .into_iter()
            .flat_map(|s| self.states[*s].consuming.ranges_values().cloned())
            .collect::<RangeMultiMap<u8, StateIdx>>()
            .group();

        // `scratch` is large enough to be indexed by anything in `elts`. It is full of `false`.
        fn uniquify(elts: &mut Vec<StateIdx>, scratch: &mut Vec<bool>) {
            elts.retain(|&e| {
                let ret = !scratch[e];
                scratch[e] = true;
                ret
            });

            // Clean up scratch, so that it is full of `false` again.
            for e in elts {
                scratch[*e] = false;
            }
        }

        let mut scratch = vec![false; self.num_states()];
        for pair in transitions.as_mut_slice() {
            uniquify(&mut pair.1, &mut scratch);
        }

        transitions
    }
}

#[derive(PartialEq)]
enum MatchChoice {
    TransitionOrder,
    LongestMatch,
}

// This contains all the intermediate data structures that we need when turning an `Nfa` into a
// `Dfa`.
struct Determinizer<'a> {
    nfa: &'a Nfa<u8, NoLooks>,
    dfa: Dfa<(Look, u8)>,
    state_map: HashMap<StateSet, StateIdx>,
    active_states: Vec<StateSet>,
    max_states: usize,
    match_choice: MatchChoice,
}

impl<'a> Determinizer<'a> {
    // Turns an Nfa into an almost-equivalent (up to the difference between shortest and longest
    // matches) Dfa.
    //
    // `init` is a vector of length Look::num(). Each entry gives a set of initial states that
    // will be turned into the initial states of the dfa.
    fn determinize(
        nfa: &Nfa<u8, NoLooks>,
        max_states: usize,
        match_choice: MatchChoice,
        init: Vec<(Look, StateIdx)>,
    ) -> Result<Dfa<(Look, u8)>> {
        let mut det = Determinizer::new(nfa, max_states, match_choice);
        det.run(init)?;
        Ok(det.dfa)
    }

    fn new(
        nfa: &'a Nfa<u8, NoLooks>,
        max_states: usize,
        match_choice: MatchChoice,
    ) -> Determinizer<'a> {
        Determinizer {
            nfa: nfa,
            dfa: Dfa::new(),
            state_map: HashMap::new(),
            active_states: Vec::new(),
            max_states: max_states,
            match_choice: match_choice,
        }
    }

    // Checks whether we should accept in the given set of states.
    //
    // Returns a tuple: the first element says when we accept, the second says what look-ahead (if
    // any) led to us accepting, and the third says how many bytes of look-ahead we needed before
    // knowing that we can accept.
    //
    // There is one annoying corner case: there could be two states in the set `s` with different
    // values of `accept_tokens`, where the higher priority state says `Accept::AtEoi` and the
    // lower priority state says `Accept::Always`. In this case, we return `(AtEoi, look, bytes)`
    // where `look` and `bytes` come from the lower priority state. This doesn't lose any
    // information, since if a state says `Accept::AtEoi` then its `accept_look` and
    // `accept_tokens` are guaranteed to be `Boundary` and `0`.
    fn accept(&self, s: &[StateIdx]) -> (Accept, Look, u8) {
        let mut accept_states = s
            .iter()
            .cloned()
            .filter(|i| self.nfa.states[*i].accept != Accept::Never);
        let mut accept_always_states = s
            .iter()
            .cloned()
            .filter(|i| self.nfa.states[*i].accept == Accept::Always);

        let (first_accept, other_accept) = if self.match_choice == MatchChoice::TransitionOrder {
            (accept_states.next(), accept_always_states.next())
        } else {
            (
                accept_states.min_by_key(|i| self.nfa.states[*i].accept_tokens),
                accept_always_states.min_by_key(|i| self.nfa.states[*i].accept_tokens),
            )
        };

        // Returns the intersection of state.accept_look over all states in s that accept
        // unconditionally and have the given number of look-ahead bytes.
        let look_intersection = |toks: u8| {
            s.iter()
                .cloned()
                .filter(|i| self.nfa.states[*i].accept == Accept::Always)
                .filter(|i| self.nfa.states[*i].accept_tokens == toks)
                .fold(Look::Full, |x, y| {
                    x.intersection(&self.nfa.states[y].accept_look)
                })
        };

        if let Some(first_accept) = first_accept {
            let st = &self.nfa.states[first_accept];

            if st.accept == Accept::AtEoi {
                // Check if there is a lower-priority Accept::Always.
                if let Some(other_accept) = other_accept {
                    let other_st = &self.nfa.states[other_accept];
                    if other_st.accept_tokens > 0 {
                        let look = look_intersection(other_st.accept_tokens);
                        return (Accept::AtEoi, look, other_st.accept_tokens);
                    }
                }
                (Accept::AtEoi, Look::Boundary, 0)
            } else {
                (
                    Accept::Always,
                    look_intersection(st.accept_tokens),
                    st.accept_tokens,
                )
            }
        } else {
            // There are no accepting states.
            (Accept::Never, Look::Empty, 0)
        }
    }

    // Tries to add a new state to the Dfa.
    //
    // If the state already exists, returns the index of the old one. If there are too many states,
    // returns an error.
    fn add_state(&mut self, mut s: StateSet) -> Result<StateIdx> {
        // When we choose our matches by transition order, discard any states that have lower
        // priority than the best match we've found.
        if self.match_choice == MatchChoice::TransitionOrder {
            if let Some(accept_idx) = s
                .iter()
                .position(|&i| self.nfa.states[i].accept == Accept::Always)
            {
                s.truncate(accept_idx + 1);
            }
        }

        if self.state_map.contains_key(&s) {
            Ok(*self.state_map.get(&s).unwrap())
        } else if self.dfa.num_states() >= self.max_states {
            Err(Error::TooManyStates)
        } else {
            let (acc, look, bytes_ago) = self.accept(&s);
            let ret = if acc != Accept::Never {
                Some((look, bytes_ago))
            } else {
                None
            };
            let new_state = self.dfa.add_state(acc, ret);

            self.active_states.push(s.clone());
            self.state_map.insert(s, new_state);
            Ok(new_state)
        }
    }

    // Creates a deterministic automaton representing the same language as our `nfa`.
    // Puts the new Dfa in self.dfa.
    fn run(&mut self, init: Vec<(Look, StateIdx)>) -> Result<()> {
        if self.nfa.states.is_empty() {
            return Ok(());
        }

        for &look in Look::all() {
            let init_states: StateSet = init
                .iter()
                .cloned()
                .filter(|&(x, _)| look == x)
                .map(|(_, y)| y)
                .collect();
            if !init_states.is_empty() {
                let new_state_idx = self.add_state(init_states)?;
                self.dfa.init[look.as_usize()] = Some(new_state_idx);
            }
        }

        while !self.active_states.is_empty() {
            let state = self.active_states.pop().unwrap();
            // This unwrap is ok because anything in active_states must also be in state_map.
            let state_idx = *self.state_map.get(&state).unwrap();
            let trans = self.nfa.transition_map(&state);

            let mut dfa_trans = Vec::new();
            for &(range, ref target) in trans.ranges_values() {
                let target_idx = self.add_state(target.clone())?;
                dfa_trans.push((range, target_idx));
            }
            self.dfa
                .set_transitions(state_idx, dfa_trans.into_iter().collect());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::dfa::Dfa;
    use crate::look::Look;
    use crate::nfa::tests::{re_nfa, trans_nfa, trans_range_nfa};
    use crate::nfa::{Accept, Nfa, NoLooks};
    use range_map::Range;
    use std::usize;

    fn re_nfa_anchored(re: &str) -> Nfa<u8, NoLooks> {
        re_nfa(re)
            .byte_me(usize::MAX)
            .unwrap()
            .anchor(usize::MAX)
            .unwrap()
    }

    fn re_dfa(re: &str) -> Dfa<(Look, u8)> {
        re_nfa(re)
            .byte_me(usize::MAX)
            .unwrap()
            .determinize(usize::MAX)
            .unwrap()
    }

    #[test]
    fn anchor_simple() {
        let nfa = re_nfa_anchored("a");
        let mut target = trans_range_nfa(
            3,
            &[
                (2, 0, Range::new(b'a', b'a')),
                (2, 1, Range::full()),
                (1, 0, Range::new(b'a', b'a')),
                (1, 1, Range::full()),
            ],
        );
        target.init.push((Look::Boundary, 2));
        target.states[0].accept = Accept::Always;

        assert_eq!(nfa, target);
    }

    #[test]
    fn anchor_nl() {
        let nfa = re_nfa_anchored(r"(?m)^a");
        let mut target = trans_nfa(4, &[(3, 1, 'a'), (0, 1, 'a'), (2, 0, '\n'), (3, 0, '\n')]);
        target.init.push((Look::Boundary, 3));
        target.states[1].accept = Accept::Always;

        let mut target = target.byte_me(usize::MAX).unwrap();
        target.states[2].consuming.insert(Range::full(), 2);
        target.states[3].consuming.insert(Range::full(), 2);

        assert_eq!(nfa, target);
    }

    #[test]
    fn anchor_already_anchored() {
        let nfa = re_nfa_anchored("^a");
        let mut target = trans_nfa(2, &[(1, 0, 'a')]);
        target.init.push((Look::Boundary, 1));
        target.states[0].accept = Accept::Always;

        assert_eq!(nfa, target);
    }

    #[test]
    fn determinize_pruning() {
        assert_eq!(re_dfa("a|aa"), re_dfa("a"));
    }

    macro_rules! check_rev_inits {
        ($name:ident, $re:expr, $inits:expr) => {
            #[test]
            fn $name() {
                let rev = re_nfa($re)
                    .byte_me(usize::MAX)
                    .unwrap()
                    .reverse(usize::MAX)
                    .unwrap();
                println!("{:?}", rev.init);
                for &look in Look::all() {
                    println!("checking look {:?}", look);
                    if $inits.contains(&look) {
                        assert!(rev.init.iter().any(|pair| pair.0 == look));
                    } else {
                        assert!(!rev.init.iter().any(|pair| pair.0 == look));
                    }
                }
            }
        };
    }

    check_rev_inits!(rev_init_simple, "abc", [Look::Full]);
    check_rev_inits!(rev_init_boundary, "abc$", [Look::Boundary]);
    check_rev_inits!(
        rev_init_simple_and_boundary,
        "(abc$|abc)",
        [Look::Full, Look::Boundary]
    );
    check_rev_inits!(
        rev_init_new_line,
        "(?m)abc$",
        [Look::Boundary, Look::NewLine]
    );
    check_rev_inits!(rev_init_word, r"  \b", [Look::WordChar]);
    check_rev_inits!(
        rev_init_not_word,
        r"abc\b",
        [Look::Boundary, Look::NotWordChar]
    );
    check_rev_inits!(
        rev_init_word_or_not_word,
        r".\b",
        [Look::Boundary, Look::NotWordChar, Look::WordChar]
    );
}
