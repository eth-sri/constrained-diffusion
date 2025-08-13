// Copyright 2015-2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::dfa::trie::Trie;
use crate::dfa::{Dfa, RetTrait};
use crate::nfa::{Accept, StateIdx};
use std::cmp::{Ordering, PartialOrd};
use std::collections::{HashSet, VecDeque};
use std::mem::swap;

// TODO: These limits are pretty arbitrary (copied from the regex crate).
const NUM_PREFIX_LIMIT: usize = 30;
const PREFIX_LEN_LIMIT: usize = 15;

/// A pair of a byte sequence and the index of the state that we are in after encountering that
/// sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixPart(pub Vec<u8>, pub StateIdx);

pub struct PrefixSearcher {
    active: VecDeque<PrefixPart>,
    current: PrefixPart,
    suffixes: Trie,
    finished: Vec<PrefixPart>,

    // The set of prefixes is complete if:
    //  - we're done with active prefixes before we go over any of our limits, and
    //  - we didn't encounter any states that accept conditionally.
    complete: bool,

    max_prefixes: usize,
    max_len: usize,
}

impl PrefixSearcher {
    pub fn extract<T: RetTrait>(dfa: &Dfa<T>, state: StateIdx) -> Vec<PrefixPart> {
        let mut searcher = PrefixSearcher::new();
        searcher.search(dfa, state);
        searcher.finished
    }

    fn new() -> PrefixSearcher {
        PrefixSearcher {
            active: VecDeque::new(),
            current: PrefixPart(Vec::new(), 0),
            suffixes: Trie::new(),
            finished: Vec::new(),
            complete: true,
            max_prefixes: NUM_PREFIX_LIMIT,
            max_len: PREFIX_LEN_LIMIT,
        }
    }

    fn bail_out(&mut self) {
        let mut current = PrefixPart(Vec::new(), 0);
        let mut active = VecDeque::new();
        swap(&mut current, &mut self.current);
        swap(&mut active, &mut self.active);

        self.finished.extend(active.into_iter());
        self.finished.push(current);
        self.complete = false;
    }

    fn add(&mut self, new_prefs: Vec<PrefixPart>) {
        debug_assert!(
            new_prefs.len() + self.active.len() + self.finished.len() <= self.max_prefixes
        );

        for p in new_prefs.into_iter() {
            if p.0.len() >= self.max_len {
                self.finished.push(p);
            } else {
                self.active.push_back(p);
            }
        }
    }

    fn too_many(&mut self, more: usize) -> bool {
        self.active.len() + self.finished.len() + more > self.max_prefixes
    }

    fn search<T: RetTrait>(&mut self, dfa: &Dfa<T>, state: StateIdx) {
        self.active.push_back(PrefixPart(Vec::new(), state));
        self.suffixes.insert(vec![].into_iter(), state);
        while !self.active.is_empty() {
            self.current = self.active.pop_front().unwrap();

            let trans = dfa.transitions(self.current.1);
            let mut next_prefs = Vec::new();
            for (ch, next_state) in trans.keys_values() {
                let mut next_pref = self.current.0.clone();
                next_pref.push(ch);
                next_prefs.push(PrefixPart(next_pref, *next_state));
            }

            // Discard any new prefix that is the suffix of some existing prefix.
            next_prefs.retain(|pref| {
                let rev_bytes = pref.0.iter().cloned().rev();
                !self.suffixes.prefixes(rev_bytes).any(|s| s == pref.1)
            });
            for pref in &next_prefs {
                self.suffixes.insert(pref.0.iter().cloned().rev(), pref.1);
            }

            // Stop searching if we have too many prefixes already, or if we've run into an accept
            // state. In principle, we could continue expanding the other prefixes even after we
            // run into an accept state, but there doesn't seem much point in having some short
            // prefixes and other long prefixes.
            if self.too_many(next_prefs.len()) || *dfa.accept(self.current.1) != Accept::Never {
                self.bail_out();
                break;
            }

            self.add(next_prefs);
        }
    }
}

// A critical segment is a sequence of bytes that we must match if we want to get to a particular
// state. That sequence need not necessarily correspond to a unique path in the DFA, however.
// Therefore, we store the sequence of bytes and also a set of possible paths that we might have
// traversed while reading those bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct CriticalSegment {
    bytes: Vec<u8>,
    paths: HashSet<Vec<StateIdx>>,
}

// The stdlib seems to have searching functions for &str, but not for &[u8]. If they get added, we
// can remove this.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .enumerate()
        .find(|x| x.1 == needle)
        .map(|y| y.0)
}

// For two critical segments a and b, we say a <= b if it is more specific than b: either a's
// byte sequence contains b's byte sequence or else the byte sequences are the same and a's set of
// paths is a subset of b's set of paths.
// TODO: not sure if this is necessary
impl PartialOrd for CriticalSegment {
    fn partial_cmp(&self, other: &CriticalSegment) -> Option<Ordering> {
        fn less(a: &CriticalSegment, b: &CriticalSegment) -> bool {
            let a_len = a.bytes.len();
            let b_len = b.bytes.len();
            (a_len > b_len && find(&a.bytes, &b.bytes).is_some())
                || (a.bytes == b.bytes && a.paths.is_subset(&b.paths))
        }
        if less(self, other) {
            Some(Ordering::Less)
        } else if less(other, self) {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}

/*
impl CriticalSegment {
    pub fn intersection(xs: &[CriticalSegment], ys: &[CriticalSegment]) -> Vec<CriticalSegment> {
        let common = maximal_common_substrings(
            xs.iter().map(|x| &x.bytes[..]),
            ys.iter().map(|y| &y.bytes[..]));
        let mut ret = Vec::new();

        for s in common {
            let mut paths = HashSet::new();
            for x in xs.iter().chain(ys.iter()) {
                // We look for only the first occurence of the substring in x.
                if let Some(pos) = find(&x.bytes, &s) {
                    paths.extend(x.paths.iter().map(|p| p[pos..(pos + s.len())].to_vec()));
                }
            }
            ret.push(CriticalSegment { bytes: s, paths: paths });
        }
        ret
    }
}

// Finds all strings that are
// - a substring of some element of xs,
// - a substring of some element of ys, and
// - maximal among all strings satisfying the first two conditions.
//
// Note that this implementation is *extremely* naive -- an efficient implementation would probably
// want to use a generalized suffix tree. But since the strings we deal with here are small, we can
// sort of get away with it.
fn maximal_common_substrings<'a, I, J>(xs: I, ys: J) -> HashSet<Vec<u8>>
where I: Iterator<Item=&'a [u8]>, J: Iterator<Item=&'a [u8]> {
    let mut ys_substrings = HashSet::new();
    let mut common_substrings = HashSet::new();

    for y in ys {
        let len = y.len();
        for i in 0..len {
            for j in i..len {
                ys_substrings.insert(&y[i..(j + 1)]);
            }
        }
    }

    for x in xs {
        let len = x.len();
        for i in 0..len {
            for j in i..len {
                if ys_substrings.contains(&x[i..(j + 1)]) {
                    common_substrings.insert(x[i..(j + 1)].to_vec());
                }
            }
        }
    }

    // Now prune out anything that isn't maximal.
    let mut ret = common_substrings.clone();
    for s in &common_substrings {
        let len = s.len();
        for i in 0..len {
            for j in i..len {
                // Make sure we're only looking at proper substrings of s.
                if i > 0 || j < len - 1 {
                    ret.remove(&s[i..(j + 1)]);
                }
            }
        }
    }
    ret
}
*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dfa;
    use crate::look::Look;
    use quickcheck::{quickcheck, QuickCheck, StdGen, TestResult};
    use rand;
    //use super::{find, maximal_common_substrings};

    fn qc(size: usize) -> QuickCheck<StdGen<rand::ThreadRng>> {
        QuickCheck::new().gen(StdGen::new(rand::thread_rng(), size))
    }

    macro_rules! test_prefix {
        ($name:ident, $re_str:expr, $answer:expr, $max_num:expr, $max_len:expr) => {
            #[test]
            fn $name() {
                let dfa = dfa::tests::make_dfa($re_str).unwrap();
                println!("{:?}", dfa);
                let mut pref = PrefixSearcher::new();
                pref.max_prefixes = $max_num;
                pref.max_len = $max_len;
                pref.search(&dfa, dfa.init_state(Look::Full).unwrap());
                let mut prefs = pref.finished.into_iter().map(|x| x.0).collect::<Vec<_>>();
                prefs.sort();

                let answer: Vec<Vec<u8>> =
                    $answer.iter().map(|s| s.as_bytes().to_owned()).collect();
                assert_eq!(prefs, answer);
            }
        };
    }

    test_prefix!(
        long,
        "[XYZ]ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        vec![
            "XABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "YABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "ZABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ],
        3,
        30
    );

    test_prefix!(
        case_insensitive,
        "(?i)abc[a-z]",
        vec!["ABC", "ABc", "AbC", "Abc", "aBC", "aBc", "abC", "abc"],
        30,
        5
    );

    test_prefix!(byte_set, "[ac]", vec!["a", "c"], 30, 5);

    test_prefix!(pruned_repetition, "a+bc", vec!["abc"], 10, 10);

    test_prefix!(pruned_empty_repetition, "[a-zA-Z]*bc", vec!["bc"], 10, 10);

    /*
    #[test]
    fn common_substrings() {
        fn sound(xs: Vec<Vec<u8>>, ys: Vec<Vec<u8>>) -> bool {
            let result = maximal_common_substrings(xs.iter().map(|x| &x[..]), ys.iter().map(|y| &y[..]));

            // Everything in the result should be a substring of something in xs.
            result.iter().all(|x| xs.iter().any(|y| find(&y, &x).is_some()))
                // Everything in the result should be a substring of something in xs.
                && result.iter().all(|x| ys.iter().any(|y| find(&y, &x).is_some()))
                // Nothing in the result should be a strict substring of anything else.
                && result.iter().all(
                    |x| !result.iter().any(|y| y.len() > x.len() && find(&y, &x).is_some()))
        }

        // If z is a substring of something in xs and something in ys then it must be a substring
        // of something in result.
        fn complete(xs: Vec<Vec<u8>>, ys: Vec<Vec<u8>>, z: Vec<u8>) -> TestResult {
            if z.is_empty()
                    || !xs.iter().any(|x| find(&x, &z).is_some())
                    || !ys.iter().any(|y| find(&y, &z).is_some()) {
                return TestResult::discard();
            }

            let result = maximal_common_substrings(xs.iter().map(|x| &x[..]), ys.iter().map(|y| &y[..]));
            TestResult::from_bool(result.iter().any(|x| find(&x, &z).is_some()))
        }

        qc(10).quickcheck(sound as fn(_, _) -> _);
        qc(10).quickcheck(complete as fn(_, _, _) -> _);
    }
    */
}
