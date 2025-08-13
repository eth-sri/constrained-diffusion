// Copyright 2015-2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Debug;
//use dfa::{Dfa, PrefixPart, RetTrait};
use crate::dfa::PrefixPart;
use crate::runner::program::TableInsts;
use crate::runner::Engine;
use itertools::Itertools;
use memchr::memchr;

#[derive(Clone, Debug)]
pub struct ForwardBackwardEngine<Ret> {
    forward: TableInsts<(usize, u8)>,
    backward: TableInsts<Ret>,
    prefix: Prefix,
}

impl<Ret: Copy + Debug> ForwardBackwardEngine<Ret> {
    pub fn new(
        forward: TableInsts<(usize, u8)>,
        prefix: Prefix,
        backward: TableInsts<Ret>,
    ) -> Self {
        ForwardBackwardEngine {
            forward: forward,
            backward: backward,
            prefix: prefix,
        }
    }

    fn find_with_searcher<SearchFn>(
        &self,
        input: &[u8],
        search: SearchFn,
    ) -> Option<(usize, usize, Ret)>
    where
        SearchFn: Fn(&[u8], usize) -> Option<usize>,
    {
        let mut pos = 0;
        while let Some(start) = search(input, pos) {
            match self.forward.find_from(input, start, 0) {
                Ok((end, (rev_state, look_ahead))) => {
                    let rev_pos = end.saturating_sub(look_ahead as usize);
                    let (start_pos, ret) = self
                        .backward
                        .longest_backward_find_from(input, rev_pos, rev_state)
                        .expect("BUG: matched forward but failed to match backward");
                    return Some((start_pos, rev_pos, ret));
                }
                Err(end) => {
                    pos = end + 1;
                }
            }
        }

        None
    }
}

impl<Ret: Copy + Debug + 'static> Engine<Ret> for ForwardBackwardEngine<Ret> {
    fn find(&self, s: &str) -> Option<(usize, usize, Ret)> {
        let input = s.as_bytes();
        if self.forward.is_empty() {
            return None;
        }

        match self.prefix {
            Prefix::Empty => {
                self.find_with_searcher(
                    input,
                    |s, pos| if pos <= s.len() { Some(pos) } else { None },
                )
            }
            Prefix::ByteSet { ref bytes, offset } => self.find_with_searcher(input, |s, pos| {
                if pos + offset <= s.len() {
                    s[(pos + offset)..]
                        .iter()
                        .position(|c| bytes[*c as usize])
                        .map(|x| x + pos)
                } else {
                    None
                }
            }),
            Prefix::Byte { byte, offset } => self.find_with_searcher(input, |s, pos| {
                if pos + offset <= s.len() {
                    memchr(byte, &input[(pos + offset)..]).map(|x| x + pos)
                } else {
                    None
                }
            }),
            //Prefix::ByteBackwards { .. } => unimplemented!(),
        }
    }

    fn clone_box(&self) -> Box<dyn Engine<Ret>> {
        Box::new(self.clone())
    }
}

/// A `Prefix` is the first part of a DFA. Anything matching the DFA should start with
/// something matching the `Prefix`.
///
/// The purpose of a `Prefix` is that scanning through the input looking for the `Prefix` should be
/// much faster than running the DFA naively.
#[derive(Clone, Debug)]
pub enum Prefix {
    // Matches every position.
    Empty,
    // Matches a single byte in a particular set and then rewinds some number of bytes.
    ByteSet { bytes: Vec<bool>, offset: usize },
    // Matches a specific byte and then rewinds some number of bytes.
    Byte { byte: u8, offset: usize },
    // Matches a specific byte and then runs a DFA backwards.
    //ByteBackwards { byte: u8, rev: Dfa<()> },
}

// How big we allow the byte sets to be. In order for byte sets to be a performance win, finding a
// byte in the set needs to be sufficiently rare; therefore, we only use small sets. There might be
// room for a better heuristic, though: we could use large sets that only have rare bytes.
const MAX_BYTE_SET_SIZE: usize = 16;

impl Prefix {
    fn byte_prefix(parts: &[PrefixPart]) -> Option<Prefix> {
        fn common_prefix<'a>(s1: &'a [u8], s2: &'a [u8]) -> &'a [u8] {
            let prefix_len = s1
                .iter()
                .zip(s2.iter())
                .take_while(|pair| pair.0 == pair.1)
                .count();
            &s1[0..prefix_len]
        }

        let mut parts = parts.iter();
        if let Some(first) = parts.next() {
            let lit = parts.fold(&first.0[..], |acc, p| common_prefix(acc, &p.0));
            if !lit.is_empty() {
                // See if the common prefix contains a full codepoint. If it does, search for the last
                // byte of that codepoint.
                let cp_last_byte = ((!lit[0]).leading_zeros() as usize).saturating_sub(1);
                if cp_last_byte < lit.len() {
                    return Some(Prefix::Byte {
                        byte: lit[cp_last_byte],
                        offset: cp_last_byte,
                    });
                }
            }
        }

        None
    }

    fn byte_set_prefix(parts: &[PrefixPart]) -> Option<Prefix> {
        let crit_byte_pos = |p: &PrefixPart| ((!p.0[0]).leading_zeros() as usize).saturating_sub(1);
        let crit_byte_posns: Vec<usize> = parts.iter().map(crit_byte_pos).dedup().collect();

        if crit_byte_posns.len() == 1 {
            let crit_byte = crit_byte_posns[0];
            if parts.iter().all(|x| x.0.len() > crit_byte) {
                let mut crit_bytes: Vec<u8> = parts.iter().map(|x| x.0[crit_byte]).collect();
                crit_bytes.sort();
                crit_bytes.dedup();

                if crit_bytes.len() <= MAX_BYTE_SET_SIZE {
                    let mut ret = vec![false; 256];
                    for &b in &crit_bytes {
                        ret[b as usize] = true;
                    }
                    return Some(Prefix::ByteSet {
                        bytes: ret,
                        offset: crit_byte,
                    });
                }
            }
        }

        None
    }

    /*
    pub fn from_dfa<Ret: RetTrait>(dfa: &Dfa<Ret>) -> Prefix {
        let parts = dfa.prefix_strings();
        let first_try = Prefix::from_parts(parts);

        /*
        match first_try {
            Prefix::Byte {..} => first_try,
            _ => {
                let crit_strings = dfa.critical_strings();
                unimplemented!();
                first_try
            },
        }
        */
        unimplemented!();
    }
    */

    /// Converts a set of `PrefixParts` into a `Prefix` that matches any of the strings.
    pub fn from_parts(mut parts: Vec<PrefixPart>) -> Prefix {
        parts.retain(|x| !x.0.is_empty());

        if let Some(pref) = Prefix::byte_prefix(&parts) {
            pref
        } else if let Some(pref) = Prefix::byte_set_prefix(&parts) {
            pref
        } else {
            Prefix::Empty
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dfa::PrefixPart;

    fn pref(strs: Vec<&str>) -> Prefix {
        Prefix::from_parts(
            strs.into_iter()
                .enumerate()
                .map(|(i, s)| PrefixPart(s.as_bytes().to_vec(), i))
                .collect(),
        )
    }

    #[test]
    fn test_prefix_choice() {
        use super::Prefix::*;

        assert!(matches!(pref(vec![]), Empty));
        assert!(matches!(pref(vec![""]), Empty));
        assert!(matches!(pref(vec!["a"]), Byte { .. }));
        assert!(matches!(pref(vec!["", "a", ""]), Byte { .. }));
        assert!(matches!(pref(vec!["abc"]), Byte { .. }));
        assert!(matches!(pref(vec!["abc", ""]), Byte { .. }));
        assert!(matches!(pref(vec!["a", "b", "c"]), ByteSet { .. }));
        assert!(matches!(pref(vec!["a", "b", "", "c"]), ByteSet { .. }));
        assert!(matches!(pref(vec!["a", "baa", "", "c"]), ByteSet { .. }));
        assert!(matches!(pref(vec!["ab", "baa", "", "cb"]), ByteSet { .. }));
        assert!(matches!(pref(vec!["ab", "aaa", "", "acb"]), Byte { .. }));
        assert!(matches!(pref(vec!["ab", "abc", "abd"]), Byte { .. }));
    }
}
