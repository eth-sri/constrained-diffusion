// Copyright 2015-2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::error::Error;
use crate::nfa::{Nfa, NoLooks};
use crate::runner::anchored::AnchoredEngine;
use crate::runner::forward_backward::{ForwardBackwardEngine, Prefix};
use crate::runner::Engine;
use crate::Result;
use std;
use std::fmt::Debug;

#[derive(Debug)]
pub struct Regex {
    engine: Box<dyn Engine<u8>>,
}

// An engine that doesn't match anything.
#[derive(Clone, Debug)]
struct EmptyEngine;

impl<Ret: Debug> Engine<Ret> for EmptyEngine {
    fn find(&self, _: &str) -> Option<(usize, usize, Ret)> {
        None
    }
    fn clone_box(&self) -> Box<dyn Engine<Ret>> {
        Box::new(EmptyEngine)
    }
}

impl Clone for Regex {
    fn clone(&self) -> Regex {
        Regex {
            engine: self.engine.clone_box(),
        }
    }
}

impl Regex {
    /// Creates a new `Regex` from a regular expression string.
    pub fn new(re: &str) -> Result<Regex> {
        Regex::new_bounded(re, std::usize::MAX)
    }

    /// Creates a new `Regex` from a regular expression string, but only if it doesn't require too
    /// many states.
    pub fn new_bounded(re: &str, max_states: usize) -> Result<Regex> {
        let nfa = Nfa::from_regex(re)?;
        let nfa = nfa.remove_looks();

        let eng = if nfa.is_empty() {
            Box::new(EmptyEngine) as Box<dyn Engine<u8>>
        } else if nfa.is_anchored() {
            Box::new(Regex::make_anchored(nfa, max_states)?) as Box<dyn Engine<u8>>
        } else {
            Box::new(Regex::make_forward_backward(nfa, max_states)?) as Box<dyn Engine<u8>>
        };

        Ok(Regex { engine: eng })
    }

    fn make_anchored(nfa: Nfa<u32, NoLooks>, max_states: usize) -> Result<AnchoredEngine<u8>> {
        let nfa = nfa.byte_me(max_states)?;
        let dfa = nfa
            .determinize(max_states)?
            .optimize()
            .map_ret(|(_, bytes)| bytes);
        let prog = dfa.compile();

        Ok(AnchoredEngine::new(prog))
    }

    fn make_forward_backward(
        nfa: Nfa<u32, NoLooks>,
        max_states: usize,
    ) -> Result<ForwardBackwardEngine<u8>> {
        if nfa.is_anchored() {
            return Err(Error::InvalidEngine(
                "anchors rule out the forward-backward engine",
            ));
        }

        let f_nfa = nfa.clone().byte_me(max_states)?.anchor(max_states)?;
        let b_nfa = nfa.byte_me(max_states)?.reverse(max_states)?;

        let f_dfa = f_nfa.determinize(max_states)?.optimize();
        let b_dfa = b_nfa.determinize_longest(max_states)?.optimize();
        let b_dfa = b_dfa.map_ret(|(_, bytes)| bytes);

        let b_prog = b_dfa.compile();
        let f_dfa = f_dfa.map_ret(|(look, bytes)| {
            let b_dfa_state =
                b_dfa.init[look.as_usize()].expect("BUG: back dfa must have this init");
            (b_dfa_state, bytes)
        });

        let mut f_prog = f_dfa.compile();
        let prefix = Prefix::from_parts(f_dfa.prefix_strings());
        match prefix {
            Prefix::Empty => {}
            _ => {
                // If there is a non-trivial prefix, we can usually speed up matching by deleting
                // transitions that return to the start state. That way, instead of returning to
                // the start state, we will just fail to match. Then we get to search for the
                // prefix before trying to match again.
                let f_dfa = f_dfa.cut_loop_to_init().optimize();
                f_prog = f_dfa.compile();
            }
        }

        Ok(ForwardBackwardEngine::new(f_prog, prefix, b_prog))
    }

    /// Returns the index range of the first match, if there is a match. The indices returned are
    /// byte indices of the string. The first index is inclusive; the second is exclusive.
    pub fn find(&self, s: &str) -> Option<(usize, usize)> {
        if let Some((start, end, look_behind)) = self.engine.find(s) {
            Some((start + look_behind as usize, end))
        } else {
            None
        }
    }

    pub fn is_match(&self, s: &str) -> bool {
        // TODO: for the forward-backward engine, this could be faster because we don't need
        // to run backward.
        self.find(s).is_some()
    }
}
