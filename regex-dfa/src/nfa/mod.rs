// Copyright 2015-2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::look::Look;
use num_traits::PrimInt;
use range_map::{Range, RangeMultiMap};
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;

mod has_looks;
mod no_looks;

// TODO: it would be nice to make StateIdx a new type instead of a type alias. The problem is that
// we need to be able to index Vecs with it, and we can't impl<T> Index<StateIdx> for Vec<T>
// because of coherence rules.
pub type StateIdx = usize;

/// How we represent a set of states. The two important criteria are:
///
/// - it should be reasonably fast even when there are thousands of states (this knocks out
///   BitSet), and
/// - it should be hashable (this knocks out HashSet).
///
/// Note that efficient insertion and O(1) queries are not important. Therefore, we use a sorted
/// Vec. (But be careful to keep it sorted!)
pub type StateSet = Vec<StateIdx>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
struct LookPair {
    pub behind: Look,
    pub ahead: Look,
    pub target_state: StateIdx,
}

impl LookPair {
    fn is_empty(&self) -> bool {
        self.behind == Look::Empty || self.ahead == Look::Empty
    }

    fn intersection(&self, other: &LookPair) -> LookPair {
        LookPair {
            behind: self.behind.intersection(&other.behind),
            ahead: self.ahead.intersection(&other.ahead),
            target_state: self.target_state,
        }
    }
}

/// The enum for determining whether a state is accepting. Classical NFAs would only allow `Never`
/// and `Always` here, but we also allow `AtEoi`, which means that the state should accept if and
/// only if we've reached the end of the input.
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Accept {
    Never,
    AtEoi,
    Always,
}

#[derive(Clone, Eq, PartialEq)]
struct State<Tok> {
    accept: Accept,
    // If accept == Always and accept_tokens > 0, then we had to do some look-ahead in order to
    // determine that we have a match. In that case, accept_state is the index of the state that
    // should have accepted.
    accept_state: StateIdx,
    // In the case that we had some look-ahead, `accept_look` says what kind of char was involved
    // in the look-ahead, and `accept_tokens` says how many input tokens were consumed while
    // looking ahead. There are some restrictions on these values:
    // - if accept is Never then accept_look is Full and accept_tokens is zero
    // - if accept is AtEoi then accept_look is Boundary and accept_tokens is zero
    // - accept_look is never Empty
    // - if accept is Always then accept_look is neither empty nor Boundary
    // - if Tok is u32 then accept_tokens is either 0 or 1
    // - if Tok is u8 then accept_tokens is at most 4.
    accept_look: Look,
    accept_tokens: u8,

    // The transitions that consume input.
    consuming: RangeMultiMap<Tok, StateIdx>,
    // Transitions that do not consume input, but that are allowed to look forward and backward one
    // token.
    looking: Vec<LookPair>,
}

/// A non-deterministic finite automaton.
///
/// `Tok` is the type of symbol that the automaton consumes. For most operations, only `u8` and
/// `u32` are supported.
///
/// Match priority
/// ==============
///
/// The whole point of a *non-deterministic* finite automaton is that it can match an input in
/// multiple ways. This implementation supports match priorities, meaning that in the event of
/// multiple matches, there is exactly one that is preferred to all the others. In this
/// implementation, the transitions out of each state are ordered and we prefer a match that makes
/// an earlier transition over one that makes a later one.
///
/// The `Variant` parameter
/// =======================
///
/// There are basically two versions of this struct with different representations and invariants,
/// but they share enough code in common that it made more sense to write one struct and use a type
/// parameter to determine which version it is. This is the meaning of the `Variant` type
/// parameter, and it has two possible values: `HasLooks` and `NoLooks`.
///
/// If `Variant == HasLooks` then the `init` field is unused. The only legal values for a state's
/// `accept` field are `Always` and `Never`, and all the `accept_*` fields are unused. The
/// automaton implicitly has a single initial state (state 0). Methods specific to
/// `Nfa<_, HasLooks>` are in `has_looks.rs`.
///
/// If `Variant == NoLooks` then the states' `looking` fields are unused. Initial states are
/// explicitly given in `init` and in the states' `accept.*` fields. Methods specific to
/// `Nfa<_, NoLooks>` are in `no_looks.rs`.
///
/// The typical life-cycle of an `Nfa` is as follows:
///
/// - First, create an `Nfa<u32, HasLooks>` using `from_regex`.
/// - Call `nfa.remove_looks()` to turn the `Nfa<u32, HasLooks>` to an `Nfa<u32, NoLooks>`.
/// - Call `nfa.byte_me()` to turn the `Nfa<u32, NoLooks>` into an `Nfa<u8, NoLooks>`.
/// - Call one of the `nfa.determinize_*()` methods to make a `Dfa`.
///
/// There are also some operations modifying `Nfa<u8, NoLooks>` that can be called between the last
/// two steps.
#[derive(Clone, Eq, PartialEq)]
pub struct Nfa<Tok, Variant> {
    states: Vec<State<Tok>>,
    // The various possible sets of states that the automaton can start in, depending on what the
    // most recent `char` of input was.
    //
    // We decide the initial state by looking at the previous char of input. For every element of
    // `self.init` whose first entry matches that char, we start in the corresponding NFA state.
    // Note that these states are ordered: states that appear earlier are given higher priority for
    // matching.
    init: Vec<(Look, StateIdx)>,
    phantom: PhantomData<Variant>,
}

pub trait Lookability {}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HasLooks;
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct NoLooks;

impl Lookability for HasLooks {}
impl Lookability for NoLooks {}

impl<Tok: Debug + PrimInt, L: Lookability> Nfa<Tok, L> {
    pub fn new() -> Nfa<Tok, L> {
        Nfa::with_capacity(0)
    }

    /// Creates a new `Nfa` that can `add_state()` `n` times without re-allocating.
    pub fn with_capacity(n: usize) -> Nfa<Tok, L> {
        Nfa {
            states: Vec::with_capacity(n),
            init: Vec::new(),
            phantom: PhantomData,
        }
    }

    /// Returns my consuming transitions, but with the source and destination swapped.
    ///
    /// If I have a transition from state `i` to state `j` that consumes token `c`, then
    /// `ret[j]` will contain a mapping from `c` to `i`, where `ret` is the value returned by this
    /// method.
    ///
    /// Note that information about match priorities is lost.
    pub fn reversed_transitions(&self) -> Vec<RangeMultiMap<Tok, StateIdx>> {
        let mut ret = vec![RangeMultiMap::new(); self.states.len()];

        for (source_idx, st) in self.states.iter().enumerate() {
            for &(range, target_idx) in st.consuming.ranges_values() {
                ret[target_idx].insert(range, source_idx);
            }
        }
        ret
    }

    /// Adds a new state and returns its index.
    pub fn add_state(&mut self, accept: Accept) -> StateIdx {
        let state_idx = self.states.len();
        self.states.push(State {
            accept: accept,
            accept_state: state_idx,
            accept_look: if accept == Accept::AtEoi {
                Look::Boundary
            } else {
                Look::Full
            },
            accept_tokens: 0,
            consuming: RangeMultiMap::new(),
            looking: Vec::new(),
        });
        state_idx
    }

    /// Adds a new state and returns its index.
    ///
    /// The new state is always accepting; it represents the case that we accept after looking
    /// ahead a few tokens.
    pub fn add_look_ahead_state(
        &mut self,
        look: Look,
        tokens: u8,
        accept_state: StateIdx,
    ) -> StateIdx {
        debug_assert!(look != Look::Boundary && look != Look::Full && look != Look::Empty);
        debug_assert!(tokens > 0);

        let state_idx = self.states.len();
        self.states.push(State {
            accept: Accept::Always,
            accept_state: accept_state,
            accept_look: look,
            accept_tokens: tokens,
            consuming: RangeMultiMap::new(),
            looking: Vec::new(),
        });
        state_idx
    }

    /// Adds a transition that moves from `source` to `target` on consuming a token in `range`.
    pub fn add_transition(&mut self, source: StateIdx, target: StateIdx, range: Range<Tok>) {
        self.states[source].consuming.insert(range, target);
    }

    /// Returns the set of consuming transitions out of the given state.
    pub fn consuming(&self, i: StateIdx) -> &RangeMultiMap<Tok, StateIdx> {
        &self.states[i].consuming
    }

    /// Returns the number of states.
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    // You've just done some operation that has changed state indices (probably by deleting
    // un-needed states). Now re-label the existing transitions according to the new state indices.
    fn map_states<F>(&mut self, map: F)
    where
        F: Fn(StateIdx) -> Option<StateIdx>,
    {
        for st in &mut self.states {
            st.consuming.retain_values(|x| map(*x).is_some());
            // The unwrap is ok because we've just filtered all the `None`s (and `map` is Fn, not
            // FnMut).
            st.consuming.map_values(|x| map(*x).unwrap());

            st.looking = st
                .looking
                .iter()
                .filter(|look| map(look.target_state).is_some())
                // The unwrap is ok because we've just filtered all the `None`s.
                .map(|look| LookPair {
                    target_state: map(look.target_state).unwrap(),
                    ..*look
                })
                .collect();

            st.accept_state = map(st.accept_state).expect("bug in map_states");
        }

        self.init = self
            .init
            .iter()
            .filter_map(|pair| map(pair.1).map(|idx| (pair.0, idx)))
            .collect();
    }

    // Changes the `Lookability` marker without allocating anything.
    fn transmuted<NewL: Lookability>(self) -> Nfa<Tok, NewL> {
        Nfa {
            states: self.states,
            init: self.init,
            phantom: PhantomData,
        }
    }

    /// Returns true if this Nfa only matches things at the beginning of the input.
    pub fn is_anchored(&self) -> bool {
        self.init.iter().all(|pair| pair.0 == Look::Boundary)
    }

    /// Returns true if this Nfa never matches anything.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

impl<Tok: Debug + PrimInt, L: Lookability> Debug for Nfa<Tok, L> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_fmt(format_args!("Nfa ({} states):\n", self.states.len()))?;

        f.write_fmt(format_args!("Init: {:?}\n", self.init))?;

        for (st_idx, st) in self.states.iter().enumerate().take(40) {
            f.write_fmt(format_args!("\tState {} ({:?}):\n", st_idx, st.accept))?;

            if st.accept != Accept::Never {
                f.write_fmt(format_args!(
                    "\t\tlook {:?}, tokens {:?}, state {:?}\n",
                    st.accept_look, st.accept_tokens, st.accept_state
                ))?;
            }
            if !st.consuming.is_empty() {
                f.write_str("\t\tConsuming:\n")?;
                // Cap it at 10 transitions, since it gets unreadable otherwise.
                for &(range, target) in st.consuming.ranges_values().take(10) {
                    f.write_fmt(format_args!(
                        "\t\t\t{:?} -- {:?} => {}\n",
                        range.start, range.end, target
                    ))?;
                }
                if st.consuming.num_ranges() > 10 {
                    f.write_str("\t\t\t...\n")?;
                }
            }
            if !st.looking.is_empty() {
                f.write_str("\t\tLooking:\n")?;
                for look in &st.looking {
                    f.write_fmt(format_args!(
                        "\t\t\t({:?},{:?}) => {}\n",
                        look.behind, look.ahead, look.target_state
                    ))?;
                }
            }
        }
        if self.states.len() > 40 {
            f.write_fmt(format_args!(
                "\t... ({} more states)\n",
                self.states.len() - 40
            ))?;
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod tests {
    use crate::nfa::{Accept, Nfa, NoLooks, StateIdx};
    use num_traits::PrimInt;
    use range_map::Range;
    use std::fmt::Debug;

    // Creates an Nfa from a regular expression string.
    pub fn re_nfa(re: &str) -> Nfa<u32, NoLooks> {
        let nfa = Nfa::from_regex(re).unwrap();
        println!("before remove looks: {:?}", nfa);
        let nfa = nfa.remove_looks();
        println!("after remove looks: {:?}", nfa);
        nfa
        //Nfa::from_regex(re).unwrap().remove_looks()
    }

    // Creates an Nfa with the given transitions.
    pub fn trans_range_nfa<Tok>(
        size: usize,
        transitions: &[(StateIdx, StateIdx, Range<Tok>)],
    ) -> Nfa<Tok, NoLooks>
    where
        Tok: Debug + PrimInt,
    {
        let mut ret: Nfa<Tok, NoLooks> = Nfa::with_capacity(size);
        for _ in 0..size {
            ret.add_state(Accept::Never);
        }
        for &(src, tgt, range) in transitions {
            ret.add_transition(src, tgt, range);
        }
        ret
    }

    // Creates an Nfa with the given transitions, each of which only takes a single char.
    pub fn trans_nfa<Tok>(
        size: usize,
        transitions: &[(StateIdx, StateIdx, char)],
    ) -> Nfa<Tok, NoLooks>
    where
        Tok: Debug + PrimInt,
    {
        let tok = |x: char| -> Tok { Tok::from(x as u32).unwrap() };
        let range_trans: Vec<_> = transitions
            .iter()
            .map(|x| (x.0, x.1, Range::new(tok(x.2), tok(x.2))))
            .collect();
        trans_range_nfa(size, &range_trans)
    }
}
