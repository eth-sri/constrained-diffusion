// Copyright 2015-2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
This crate provides tools for converting regular expressions into deterministic finite automata
(DFAs). The most interesting type is `Regex`, which is a virtual machine for executing a DFA.

# Example: creating and running a `Regex`

```rust
use regex_dfa::Regex;
let re = Regex::new(r"\d{4}-\d{2}-\d{2}").unwrap();
assert_eq!(re.find("My birthday is 1986-08-22!"), Some((15, 25)));
```

The most useful function in this crate is `Regex::find`, which looks for the first substring of the
given string that match the language of the DFA.

# Comparison to the `regex` crate

Compared to rust's standard `regex` crate, the main feature of `regex_dfa` is that `regex_dfa`
*eagerly* compiles a regular expression into a DFA, whereas `regex` does so lazily. There are
advantages and disadvantages to the eager approach. To begin with, doing all the compilation
up-front means that there is less to do at match time. If we get around to writing a compiler
plugin for compiling the regular expression at compile time, this would be an even bigger win.
Another advantage is that since we don't care so much about compilation speed, we have more
opportunities to look for optimizations.

The main disadvantage to eager compilation is memory usage. Even fairly simple regular expressions
may take several tens of kilobytes to represent as a DFA. More complicated ones (especially regular
expressions that use unicode word boundaries or character classes) may require much more. This
disadvantage is specific to eager compilation, since lazy DFA compilation only needs to create DFA
states for those characters that are actually seen (i.e., probably a tiny fraction of the entire
unicode character class). For this reason, `regex_dfa` allows you to restrict the amount of memory
it uses: simply use the method `Regex::new_bounded`, which will fail and report an error if it
would otherwise need to use too much memory.

# Roadmap

There are two substantial features that need to be added before this crate can be considered
feature-complete.

## SIMD optimizations

There are some nice tricks available for using SIMD instructions to quickly scan over uninteresting
parts of the input. The `regex` crate is capable (with a nightly compiler) of doing some of these
already, and we should imitate it.

## Compiler plugin

Since the main advantage of this crate is that it can do work ahead of time, it would make total
sense to do it all at the program's compile time. This feature will probably wait until the rust's
compiler plugin story stabilizes a bit.
*/

#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use]
extern crate matches;

#[cfg(test)]
extern crate rand;

extern crate itertools;
extern crate memchr;
extern crate num_traits;
extern crate range_map;
extern crate regex_syntax;
extern crate utf8_ranges;

#[macro_use]
extern crate lazy_static;

pub mod dfa;
pub mod error;
pub mod graph;
pub mod look;
pub mod nfa;
pub mod regex;
pub mod runner;
pub mod unicode;

pub mod refinery;

pub use crate::error::Error;
pub use crate::regex::Regex;
pub type Result<T> = ::std::result::Result<T, Error>;
