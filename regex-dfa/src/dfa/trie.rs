// Copyright 2015-2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is a fairly simple-minded implementation of a trie. Since we don't really have any special
// needs, this module could be replaced by a different crate (if we can find one that's
// well-supported).

#[derive(Clone, Debug)]
pub struct Trie {
    value: Option<usize>,
    sub_tries: Vec<Trie>,
}

impl Trie {
    pub fn new() -> Trie {
        Trie {
            value: None,
            sub_tries: Vec::new(),
        }
    }

    pub fn insert<I: Iterator<Item = u8>>(&mut self, mut key: I, value: usize) {
        if let Some(head) = key.next() {
            if self.sub_tries.is_empty() {
                self.sub_tries = vec![Trie::new(); 256];
            }
            self.sub_tries[head as usize].insert(key, value);
        } else {
            if self.value.is_some() {
                panic!("tried to insert the same key twice");
            }
            self.value = Some(value);
        }
    }

    pub fn prefixes<'a, I: Iterator<Item = u8>>(&'a self, input: I) -> TrieIter<'a, I> {
        TrieIter {
            trie: Some(self),
            input: input,
        }
    }
}

pub struct TrieIter<'a, I: Iterator<Item = u8>> {
    trie: Option<&'a Trie>,
    input: I,
}

impl<'a, I: Iterator<Item = u8>> Iterator for TrieIter<'a, I> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next_trie = self.trie;
        let mut ret = None;
        while let Some(t) = next_trie {
            next_trie = self.input.next().and_then(|c| t.sub_tries.get(c as usize));
            if let Some(v) = t.value {
                ret = Some(v);
                break;
            }
        }
        self.trie = next_trie;
        ret
    }
}
