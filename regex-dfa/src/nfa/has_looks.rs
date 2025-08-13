// Copyright 2015-2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains two main pieces of functionality: building an NFA from a regular
//! expression and processing an NFA to remove all non-consuming transitions. The first of these is
//! carried out by the `from_regex` function and it is fairly straightforward. The second is
//! possibly more unusual, and so we describe it here in some detail.
//!
//! For your standard classroom NFA, it's trivial to remove non-consuming transitions: for every
//! consuming transition with source state `s` and target state `t`, take the eps-closure of `t`
//! and then add a transition from `s` to everything in that eps-closure. Finally, all
//! non-consuming transitions are deleted. Here it is in ASCII art, where a non-consuming
//! transition transition is denoted by an epsilon (ε):
//!
//! ```text
//!               ε           b
//!     a     /-------> 3 -------> 4
//! 1 -----> 2    ε
//!           \-------> 5
//! ```
//!
//! becomes
//!
//! ```text
//!           a               b
//!   /---------------> 3 -------> 4
//!  /  a
//! 1 -----> 2
//!  \        a
//!   \---------------> 5
//! ```
//!
//! The situation becomes (just a little) tricker when the non-consuming transitions are allowed to
//! have predicates that look forward or back by one token. We need to support this sort of
//! transition if we want to support word boundaries (and the fact that doing so is a bit tricky is
//! probably the main reason that the standard `regex` crate doesn't support DFA simulation if the
//! regex contains word boundaries). So now we allow our non-consuming transitions to be of the
//! form `(P, Q)`, where `P` and `Q` are sets of tokens. You can pass through such a non-consuming
//! transition if and only if the previous token belonged to `P` and the next token belongs to `Q`.
//! (The code for this is in `nfa::LookPair`, which is a teeny bit more complicated because it also
//! allows checking for the edge (beginning for `P`, end for `Q`) of the input.)
//!
//! If `Q` is the set of all tokens, then supporting these kinds of non-consuming transitions is
//! almost the same as the previous case. The first difference is that when we take the
//! eps-closure, we also need to keep track of the predicates on the non-consuming transitions that
//! we passed through. For example, if we have a configuration like
//!
//! ```text
//!                   (P2, Q2)
//!    (P1, Q1)    /------------> 3
//! 1 ----------> 2   (P3, Q3)
//!                \------------> 4
//! ```
//!
//! then states 2, 3, and 4 all belong to the eps-closure of 1. In order to get from 1 to 3, we
//! need to pass through the predicate `(P1 ∩ P2, Q1 ∩ Q2)`; in order to get from 1 to 4, we need
//! to pass through the predicate `(P1 ∩ P3, Q1 ∩ Q3)`.
//!
//! Assuming for now that all of the `Q` predicates are the set of all possible tokens, we remove
//! the non-consuming transitions as follows: take every consuming transition with source state `s`
//! and target state `t`. Then for every `u` in the eps-closure of `t` with predicate `(P, Q)`
//! leading from `t` to `u`, we add a consuming transition from `s` to `u` *if and only if the
//! consumed token belongs to `P`*. Then we delete all the non-consuming transitions. Going back to
//! the first example, suppose that `P1` contains `a` but `P2` does not. Then
//!
//! ```text
//!               (P1, Q1)           b
//!     a     /--------------> 3 -------> 4
//! 1 -----> 2    (P2, Q2)
//!           \--------------> 5
//! ```
//!
//! becomes
//!
//! ```text
//!           a               b
//!   /---------------> 3 -------> 4
//!  /  a
//! 1 -----> 2
//!                     5
//! ```
//!
//! There is actually one more complication that we won't discuss in detail here: the procedure
//! above doesn't account properly for the eps-closure of the initial state, since it only does
//! things to the eps-closure of a state that follows a transition. In order to handle the
//! eps-closure of the initial state, we actually introduce a collection of initial states, some of
//! which are only active if the previous character of the input satisfied some predicate.
//!
//! Finally, in the case that the `Q` predicates are not the set of all possible tokens, we need to
//! add extra states. For every consuming transition from `s` to `t` and every `u` in the
//! eps-closure of `t` with predicate `(P, Q)` leading from `t` to `u`, we add a new state `u'`.
//! The consuming transitions leading out from `u'` are those consuming transitions leading out
//! from `u` whose tokens belong to `Q`. Then we add a consuming transition from `s` to `u'` if the
//! token that was consumed in going from `s` to `t` belongs to `P`. In ASCII art, if `P` contains
//! `a` but not `b`, and if `Q` contains `c` but not `d` then
//!
//! ```text
//!     a          (P, Q)           c
//! 1 -----> 2 -------------> 3 --------> 4
//!     b   ^                  \    d
//! 5 -----/                    \-------> 5
//! ```
//!
//! becomes
//!
//! ```text
//!              a                  c
//!    /--------------------> 3' -----\
//!   / a                           c  \
//! 1 -----> 2                3 --------> 4
//!     b   ^                  \    d
//! 5 -----/                    \-------> 5
//! ```
//!
//! There are a couple of caveats to this transformation also. The first is that we process *all*
//! of the look-behind (i.e. `P`) predicates before we process any of the look-ahead (i.e. `Q`)
//! predicates. The reason for this can be seen in the example above: if state 4 had any
//! non-consuming transitions leading out of it, then in processing that non-consuming transition
//! we might need to add more consuming transitions leading out of 3. That would in turn affect the
//! consuming transitions that we add to 3'. Therefore, we need to add the extra transitions coming
//! out of 3 (which are due to a look-behind predicate) before we add the transitions coming
//! out of 3' (which are due to a look-ahead predicate).
//!
//! The second caveat to the transformation above comes in the handling of accepting states. When a
//! non-consuming transition leads to an accepting state, it means that the source of that
//! transition should become a conditionally accepting state.

use crate::look::Look;
use crate::nfa::{Accept, HasLooks, LookPair, Nfa, NoLooks, StateIdx};
use crate::Result;
use range_map::{Range, RangeSet};
use regex_syntax::{ByteClass, CharClass, ClassRange, Expr, Repeater};
use std::collections::HashSet;
use std::ops::Deref;

// Converts a `CharClass` into a `RangeSet`
fn class_to_set(cc: &CharClass) -> RangeSet<u32> {
    cc.iter()
        .map(|r| Range::new(r.start as u32, r.end as u32))
        .collect()
}
fn byte_class_to_set(cc: &ByteClass) -> RangeSet<u32> {
    cc.iter()
        .map(|r| Range::new(r.start as u32, r.end as u32))
        .collect()
}

impl Nfa<u32, HasLooks> {
    /// Asserts that the invariants that are supposed to hold do.
    fn check_invariants(&self) {
        // The init state is implicitly the first one, so there are no explicit init states.
        debug_assert!(self.init.is_empty());

        // The final state is accepting, and no others are.
        debug_assert!(self.states.last().unwrap().accept == Accept::Always);
        debug_assert!(self
            .states
            .iter()
            .rev()
            .skip(1)
            .all(|s| s.accept == Accept::Never));

        // No state has both a look transition and a consuming transition.
        debug_assert!(self
            .states
            .iter()
            .all(|s| s.looking.is_empty() || s.consuming.is_empty()));

        // All targets of a consuming transition are just the next state.
        debug_assert!(self
            .states
            .iter()
            .enumerate()
            .all(|(idx, s)| s.consuming.ranges_values().all(|&(_, val)| val == idx + 1)));
    }

    /// Creates a new Nfa from a regex string.
    pub fn from_regex(re: &str) -> Result<Nfa<u32, HasLooks>> {
        let expr = Expr::parse(re)?;
        let mut ret = Nfa::new();

        ret.add_state(Accept::Never);
        ret.add_expr(&expr);
        ret.add_eps(0, 1);

        let len = ret.num_states();
        ret.states[len - 1].accept = Accept::Always;

        ret.check_invariants();
        Ok(ret)
    }

    /// Adds a non-input consuming transition between states `source` and `target`.
    ///
    /// The transition will be traversed if the last consumed byte matches `behind` and the next
    /// available byte matches `ahead`.
    pub fn add_look(&mut self, source: StateIdx, target: StateIdx, behind: Look, ahead: Look) {
        let look = LookPair {
            behind: behind,
            ahead: ahead,
            target_state: target,
        };
        self.states[source].looking.push(look);
    }

    /// Removes all look transitions, converting this Nfa into an `Nfa<u32, NoLooks>`.
    pub fn remove_looks(mut self) -> Nfa<u32, NoLooks> {
        if self.states.is_empty() {
            return Nfa::with_capacity(0);
        }

        // For every state with out transitions, add transitions from it to everything in the closure
        // of the target. Note that (according to `check_invariants`) the target state is always
        // the next state.
        let old_len = self.num_states();
        let mut new_states: Vec<(StateIdx, Look, StateIdx)> = Vec::new();
        for src_idx in 0..self.states.len() {
            if !self.states[src_idx].consuming.is_empty() {
                let consuming = self.states[src_idx].consuming.clone();
                for look in self.closure(src_idx + 1) {
                    // Add transitions into the look target.
                    let new_idx = self.add_look_state(look);
                    let filtered_consuming = consuming.intersection(look.behind.as_set());
                    for &(range, _) in filtered_consuming.ranges_values() {
                        self.add_transition(src_idx, new_idx, range);
                    }
                    // If the look target is actually a new state, hold off on adding transitions
                    // out of it, because we need to make sure that all the transitions from
                    // look.target_state have been added first.
                    if new_idx >= old_len {
                        new_states.push((new_idx, look.ahead, look.target_state));
                    }
                }
            }
        }

        // Add the new initial states: everything that was immediately reachable from state 0 is now
        // an initial state.
        for look in self.closure(0) {
            let new_idx = self.add_look_state(look);
            self.init.push((look.behind, new_idx));
            if new_idx >= old_len {
                new_states.push((new_idx, look.ahead, look.target_state));
            }
        }

        // Now add transitions out of the new states.
        for (src_idx, look, tgt_idx) in new_states {
            let out_consuming = self.states[tgt_idx].consuming.intersection(look.as_set());
            for &(range, tgt) in out_consuming.ranges_values() {
                self.states[src_idx].consuming.insert(range, tgt);
            }
        }

        // Get rid of all looking transitions.
        for st in &mut self.states {
            st.looking.clear();
        }

        let mut ret: Nfa<u32, NoLooks> = self.transmuted();
        ret.trim_unreachable();
        ret
    }

    // Adds a new state for a LookPair, if necessary. It is necessary to add a new state if and
    // only if the LookPair needs to look ahead.
    //
    // Returns the index of the new state.
    fn add_look_state(&mut self, look: LookPair) -> StateIdx {
        if look.ahead.is_full() {
            look.target_state
        } else {
            let tgt_idx = look.target_state;
            let new_idx = self.add_state(Accept::Never);

            // If the target states accepts at end of input and the look allows eoi, then the new
            // state must also accept at eoi.
            if self.states[tgt_idx].accept != Accept::Never && look.ahead.allows_eoi() {
                self.states[new_idx].accept = Accept::AtEoi;
                self.states[new_idx].accept_look = Look::Boundary;
            }

            // If the target state of the look is accepting, add a new look-ahead accepting state.
            if self.states[tgt_idx].accept == Accept::Always && !look.ahead.as_set().is_empty() {
                let acc_idx = self.add_look_ahead_state(look.ahead, 1, new_idx);
                for range in look.ahead.as_set().ranges() {
                    self.add_transition(new_idx, acc_idx, range);
                }
            }
            new_idx
        }
    }

    /// Finds (transitively) the set of all non-consuming transitions that can be made starting
    /// from `state`.
    ///
    /// The search is done depth-first so that priority is preserved.
    fn closure(&self, state: StateIdx) -> Vec<LookPair> {
        let mut stack: Vec<LookPair> = Vec::new();
        let mut seen: HashSet<LookPair> = HashSet::new();
        let mut ret: Vec<LookPair> = Vec::new();
        let mut next_looks: Vec<LookPair> = Vec::new();

        stack.extend(self.states[state].looking.iter().cloned().rev());
        while let Some(last_look) = stack.pop() {
            ret.push(last_look);
            next_looks.clear();

            for next_look in &self.states[last_look.target_state].looking {
                let int = next_look.intersection(&last_look);
                if !int.is_empty() && !seen.contains(&int) {
                    seen.insert(int);
                    next_looks.push(int);
                }
            }

            stack.extend(next_looks.drain(..).rev());
        }

        ret
    }

    /// Adds an eps transition between the given states.
    fn add_eps(&mut self, from: StateIdx, to: StateIdx) {
        self.add_look(from, to, Look::Full, Look::Full);
    }

    /// Appends a single state that transitions to the next state on observing one of the chars in
    /// the given range.
    fn add_state_with_chars(&mut self, chars: &RangeSet<u32>) {
        let idx = self.num_states();
        self.add_state(Accept::Never);
        for range in chars.ranges() {
            self.add_transition(idx, idx + 1, range);
        }
    }

    /// Appends two states, with a given transition between them.
    fn add_single_transition(&mut self, chars: &RangeSet<u32>) {
        self.add_state_with_chars(chars);
        self.add_state(Accept::Never);
    }

    /// Appends a sequence of states that recognizes a literal.
    fn add_literal<C, I>(&mut self, chars: I, case_insensitive: bool)
    where
        C: Deref<Target = char>,
        I: Iterator<Item = C>,
    {
        for ch in chars {
            let ranges = if case_insensitive {
                let cc = CharClass::new(vec![ClassRange {
                    start: *ch,
                    end: *ch,
                }]);
                class_to_set(&cc.case_fold())
            } else {
                RangeSet::single(*ch as u32)
            };
            self.add_state_with_chars(&ranges);
        }
        self.add_state(Accept::Never);
    }

    /// Appends a sequence of states that recognizes the concatenation of `exprs`.
    fn add_concat_exprs(&mut self, exprs: &[Expr]) {
        if let Some((expr, rest)) = exprs.split_first() {
            self.add_expr(expr);

            for expr in rest {
                let cur_len = self.num_states();
                self.add_eps(cur_len - 1, cur_len);
                self.add_expr(expr);
            }
        } else {
            self.add_state(Accept::Never);
        }
    }

    /// Appends a sequence of states that recognizes one of the expressions in `alts`.
    ///
    /// The earlier expressions in `alts` get higher priority when matching.
    fn add_alternate_exprs(&mut self, alts: &[Expr]) {
        // Add the new initial state that feeds into the alternate.
        let init_idx = self.num_states();
        self.add_state(Accept::Never);

        let mut expr_end_indices = Vec::<StateIdx>::with_capacity(alts.len());
        for expr in alts {
            let expr_init_idx = self.states.len();
            self.add_eps(init_idx, expr_init_idx);
            self.add_expr(expr);
            expr_end_indices.push(self.states.len() - 1);
        }

        // Make the final state of each alternative point to our new final state.
        self.add_state(Accept::Never);
        let final_idx = self.states.len() - 1;
        for idx in expr_end_indices {
            self.add_eps(idx, final_idx);
        }
    }

    /// Appends new states, representing multiple copies of `expr`.
    fn add_repeat(&mut self, expr: &Expr, rep: Repeater, greedy: bool) {
        match rep {
            Repeater::ZeroOrOne => {
                self.add_repeat_up_to(expr, 1, greedy);
            }
            Repeater::ZeroOrMore => {
                self.add_repeat_zero_or_more(expr, greedy);
            }
            Repeater::OneOrMore => {
                self.add_repeat_min_max(expr, 1, None, greedy);
            }
            Repeater::Range { min, max } => {
                self.add_repeat_min_max(expr, min, max, greedy);
            }
        }
    }

    /// Repeats `expr` a fixed number of times (which must be positive).
    fn add_repeat_exact(&mut self, expr: &Expr, n: u32) {
        assert!(n > 0);
        self.add_expr(expr);
        for _ in 1..n {
            let idx = self.states.len();
            self.add_expr(expr);
            self.add_eps(idx - 1, idx);
        }
    }

    /// Repeats `expr` between zero and `n` times (`n` must be positive).
    fn add_repeat_up_to(&mut self, expr: &Expr, n: u32, greedy: bool) {
        assert!(n > 0);

        self.add_state(Accept::Never);
        let mut init_indices = Vec::<StateIdx>::with_capacity(n as usize);
        for _ in 0..n {
            init_indices.push(self.states.len() as StateIdx);
            self.add_expr(expr);
        }
        let final_idx = self.states.len() - 1;
        for idx in init_indices {
            self.add_alt_eps(idx - 1, idx, final_idx, greedy);
        }
    }

    /// Adds an eps transition from `from` to both `to1` and `to2`. If `greedy` is true, `to1` is
    /// preferred, and otherwise `to2` is preferred.
    fn add_alt_eps(&mut self, from: usize, to1: usize, to2: usize, greedy: bool) {
        if greedy {
            self.add_eps(from, to1);
            self.add_eps(from, to2);
        } else {
            self.add_eps(from, to2);
            self.add_eps(from, to1);
        }
    }

    /// Appends new states, representing multiple copies of `expr`.
    ///
    /// The new states represent a language that accepts at least `min` and at most `maybe_max`
    /// copies of `expr`. (If `maybe_max` is `None`, there is no upper bound.)
    fn add_repeat_min_max(&mut self, expr: &Expr, min: u32, maybe_max: Option<u32>, greedy: bool) {
        if min == 0 && maybe_max == Some(0) {
            // We add a state anyway, in order to maintain the convention that every expr should
            // add at least one state (otherwise keeping track of indices becomes much more
            // tedious).
            self.add_state(Accept::Never);
            return;
        }

        if min > 0 {
            self.add_repeat_exact(expr, min);

            // If anything else comes after this, we need to connect the two parts.
            if maybe_max != Some(min) {
                let len = self.num_states();
                self.add_eps(len - 1, len);
            }
        }

        if let Some(max) = maybe_max {
            if max > min {
                self.add_repeat_up_to(expr, max - min, greedy);
            }
        } else {
            self.add_repeat_zero_or_more(expr, greedy);
        }
    }

    /// Repeats the given expression zero or more times.
    fn add_repeat_zero_or_more(&mut self, expr: &Expr, greedy: bool) {
        let start_idx = self.num_states();
        self.add_state(Accept::Never);
        self.add_expr(expr);
        self.add_state(Accept::Never);
        let end_idx = self.num_states() - 1;

        self.add_alt_eps(start_idx, start_idx + 1, end_idx, greedy);
        self.add_alt_eps(end_idx - 1, start_idx + 1, end_idx, greedy);
    }

    /// Adds two new states, with a look connecting them.
    fn add_look_pair(&mut self, behind: Look, ahead: Look) {
        let idx = self.add_state(Accept::Never);
        self.add_look(idx, idx + 1, behind, ahead);
        self.add_state(Accept::Never);
    }

    /// Adds an extra predicate between the last two states (there must be at least two states).
    fn extra_look(&mut self, behind: Look, ahead: Look) {
        let len = self.states.len();
        self.add_look(len - 2, len - 1, behind, ahead);
    }

    /// Appends a bunch of new states, representing `expr`.
    ///
    /// This maintains the invariant that the last state is always empty (i.e. it doesn't have any
    /// transitions leading out of it). It is also guaranteed to add at least one new state.
    fn add_expr(&mut self, expr: &Expr) {
        use regex_syntax::Expr::*;

        match expr {
            Empty => {
                self.add_state(Accept::Never);
            }
            Class(ref c) => self.add_single_transition(&class_to_set(c)),
            ClassBytes(ref c) => {
                let ranges = byte_class_to_set(c);
                self.add_single_transition(&ranges);
            }
            AnyChar => self.add_single_transition(&RangeSet::full()),
            AnyByte => {
                let ranges = RangeSet::full();
                self.add_single_transition(&ranges);
            }
            AnyCharNoNL => {
                let nls = b"\n\r".into_iter().map(|b| *b as u32);
                self.add_single_transition(&RangeSet::except(nls))
            }
            AnyByteNoNL => {
                let nls = b"\n\r".into_iter().map(|b| *b as u32);
                self.add_single_transition(&RangeSet::except(nls))
            }
            Concat(ref es) => self.add_concat_exprs(es),
            Alternate(ref es) => self.add_alternate_exprs(es),
            Literal { ref chars, casei } => self.add_literal(chars.iter(), *casei),
            LiteralBytes { bytes, casei } => {
                let chars = Vec::from_iter(bytes.iter().map(|b| *b as char));
                self.add_literal(chars.iter(), *casei);
            }
            StartLine => self.add_look_pair(Look::NewLine, Look::Full),
            StartText => self.add_look_pair(Look::Boundary, Look::Full),
            EndLine => self.add_look_pair(Look::Full, Look::NewLine),
            EndText => self.add_look_pair(Look::Full, Look::Boundary),
            WordBoundary => {
                self.add_look_pair(Look::WordChar, Look::NotWordChar);
                self.extra_look(Look::NotWordChar, Look::WordChar);
            }
            WordBoundaryAscii => {
                self.add_look_pair(Look::WordChar, Look::NotWordChar);
                self.extra_look(Look::NotWordChar, Look::WordChar);
            }
            NotWordBoundary => {
                self.add_look_pair(Look::WordChar, Look::WordChar);
                self.extra_look(Look::NotWordChar, Look::NotWordChar);
            }
            NotWordBoundaryAscii => {
                self.add_look_pair(Look::WordChar, Look::WordChar);
                self.extra_look(Look::NotWordChar, Look::NotWordChar);
            }
            Repeat { ref e, r, greedy } => self.add_repeat(e, r.clone(), greedy.clone()),

            // We don't support capture groups, so there is no need to keep track of
            // the group name or number.
            Group { ref e, .. } => self.add_expr(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::look::Look;
    use crate::nfa::tests::{re_nfa, trans_nfa};
    use crate::nfa::{Accept, Nfa, NoLooks, StateIdx};

    // Creates an Nfa with the given transitions, with initial state zero, and with the final
    // state the only accepting state.
    fn trans_nfa_extra(
        size: usize,
        transitions: &[(StateIdx, StateIdx, char)],
    ) -> Nfa<u32, NoLooks> {
        let mut ret: Nfa<u32, NoLooks> = trans_nfa(size, transitions);

        ret.states[size - 1].accept = Accept::Always;
        ret.init.push((Look::Full, 0));
        ret
    }

    #[test]
    fn single() {
        let nfa = re_nfa("a");
        let target = trans_nfa_extra(2, &[(0, 1, 'a')]);

        assert_eq!(nfa, target);
    }

    #[test]
    fn alternate() {
        let nfa = re_nfa("a|b");
        let mut target = trans_nfa_extra(3, &[(0, 2, 'a'), (1, 2, 'b')]);
        target.init.push((Look::Full, 1));

        assert_eq!(nfa, target);
    }

    // TODO: once remove_looks supports laziness, test it.

    #[test]
    fn plus() {
        let nfa = re_nfa("a+");
        // It's possible to generate a smaller NFA for '+', but we don't currently do it.
        let target = trans_nfa_extra(3, &[(0, 1, 'a'), (0, 2, 'a'), (1, 1, 'a'), (1, 2, 'a')]);

        assert_eq!(nfa, target);
    }

    #[test]
    fn star() {
        let nfa = re_nfa("a*");
        // It's possible to generate a smaller NFA for '*', but we don't currently do it.
        let mut target = trans_nfa_extra(2, &[(0, 0, 'a'), (0, 1, 'a')]);
        target.init.push((Look::Full, 1));

        assert_eq!(nfa, target);
    }

    #[test]
    fn rep_fixed() {
        assert_eq!(re_nfa("a{3}"), re_nfa("aaa"));
    }

    #[test]
    fn rep_range() {
        assert_eq!(re_nfa("a{2,4}"), re_nfa("aaa{0,2}"));
    }

    #[test]
    fn sequence() {
        let nfa = re_nfa("ab");
        let target = trans_nfa_extra(3, &[(0, 1, 'a'), (1, 2, 'b')]);

        assert_eq!(nfa, target);
    }

    #[test]
    fn anchored_start() {
        let nfa = re_nfa("^a");
        let mut target = trans_nfa(2, &[(0, 1, 'a')]);
        target.init.push((Look::Boundary, 0));
        target.states[1].accept = Accept::Always;

        assert_eq!(nfa, target);
    }

    #[test]
    fn anchored_end() {
        let nfa = re_nfa("a$");
        let mut target = trans_nfa_extra(2, &[(0, 1, 'a')]);
        target.states[1].accept = Accept::AtEoi;
        target.states[1].accept_look = Look::Boundary;
        target.states[1].accept_state = 1;

        assert_eq!(nfa, target);
    }

    #[test]
    fn word_boundary_start() {
        let nfa = re_nfa(r"\ba");
        let mut target = trans_nfa(2, &[(1, 0, 'a')]);
        target.init.push((Look::NotWordChar, 1));
        target.states[0].accept = Accept::Always;

        assert_eq!(nfa, target);
    }

    #[test]
    fn word_boundary_end() {
        let nfa = re_nfa(r"a\b");
        let mut target = trans_nfa_extra(3, &[(0, 1, 'a')]);
        for range in Look::NotWordChar.as_set().ranges() {
            target.add_transition(1, 2, range);
        }
        target.states[1].accept = Accept::AtEoi;
        target.states[1].accept_look = Look::Boundary;
        target.states[1].accept_state = 1;
        target.states[2].accept = Accept::Always;
        target.states[2].accept_look = Look::NotWordChar;
        target.states[2].accept_state = 1;
        target.states[2].accept_tokens = 1;

        assert_eq!(nfa, target);
    }

    #[test]
    fn word_boundary_ambiguous() {
        let nfa = re_nfa(r"\b(a| )");
        let mut target = trans_nfa(3, &[(1, 0, ' '), (2, 0, 'a')]);
        target.states[0].accept = Accept::Always;
        target.init.push((Look::WordChar, 1));
        target.init.push((Look::NotWordChar, 2));

        assert_eq!(nfa, target);
    }

    #[test]
    fn empty() {
        assert_eq!(re_nfa(""), trans_nfa_extra(1, &[]));
    }
}
