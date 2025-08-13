// Copyright 2015-2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

use crate::unicode::PERLW;
use range_map::{Range, RangeSet};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord)]
pub enum Look {
    Full,
    WordChar,
    NotWordChar,
    NewLine,
    Boundary,
    Empty,
}

lazy_static! {
    static ref FULL: RangeSet<u32> = RangeSet::full();
    static ref WORD_CHAR: RangeSet<u32> = PERLW
        .iter()
        .map(|&(x, y)| Range::new(x as u32, y as u32))
        .collect();
    static ref NOT_WORD_CHAR: RangeSet<u32> = WORD_CHAR.negated();
    static ref NEW_LINE: RangeSet<u32> = RangeSet::single('\n' as u32);
    static ref EMPTY: RangeSet<u32> = RangeSet::new();
}

static ALL: [Look; 6] = [
    Look::Full,
    Look::WordChar,
    Look::NotWordChar,
    Look::NewLine,
    Look::Boundary,
    Look::Empty,
];

impl PartialOrd for Look {
    fn partial_cmp(&self, other: &Look) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else if self.intersection(other) == *self {
            Some(Ordering::Less)
        } else if self.intersection(other) == *other {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}

impl Look {
    pub fn intersection(&self, other: &Look) -> Look {
        use self::Look::*;
        match *self {
            Full => *other,
            WordChar => match *other {
                Full => WordChar,
                WordChar => WordChar,
                _ => Empty,
            },
            NotWordChar => match *other {
                Full => NotWordChar,
                NotWordChar => NotWordChar,
                NewLine => NewLine,
                Boundary => Boundary,
                _ => Empty,
            },
            NewLine => match *other {
                Full => NewLine,
                NotWordChar => NewLine,
                NewLine => NewLine,
                Boundary => Boundary,
                _ => Empty,
            },
            Boundary => match *other {
                WordChar => Empty,
                Empty => Empty,
                _ => Boundary,
            },
            Empty => Empty,
        }
    }

    pub fn supersets(&self) -> Vec<Look> {
        ALL.iter().cloned().filter(|x| *self <= *x).collect()
    }

    pub fn as_set(&self) -> &RangeSet<u32> {
        use self::Look::*;

        match *self {
            Full => &FULL,
            WordChar => &WORD_CHAR,
            NotWordChar => &NOT_WORD_CHAR,
            NewLine => &NEW_LINE,
            Boundary => &EMPTY,
            Empty => &EMPTY,
        }
    }

    pub fn allows_eoi(&self) -> bool {
        use self::Look::*;

        match *self {
            Full => true,
            WordChar => false,
            NotWordChar => true,
            NewLine => true,
            Boundary => true,
            Empty => false,
        }
    }

    pub fn is_full(&self) -> bool {
        match *self {
            Look::Full => true,
            _ => false,
        }
    }

    pub fn as_usize(&self) -> usize {
        use self::Look::*;

        match *self {
            Full => 0,
            WordChar => 1,
            NotWordChar => 2,
            NewLine => 3,
            Boundary => 4,
            Empty => 5,
        }
    }

    pub fn num() -> usize {
        6
    }

    pub fn all() -> &'static [Look] {
        &ALL
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::look::Look::*;
    use quickcheck::{quickcheck, Arbitrary, Gen};

    impl Arbitrary for Look {
        fn arbitrary<G: Gen>(g: &mut G) -> Look {
            *g.choose(&[Full, WordChar, NotWordChar, NewLine, Boundary, Empty])
                .unwrap()
        }
    }

    #[test]
    fn intersection_commutes() {
        fn prop(a: Look, b: Look) -> bool {
            a.intersection(&b) == b.intersection(&a)
        }
        quickcheck(prop as fn(_, _) -> _);
    }

    #[test]
    fn intersection_ordering() {
        fn prop(a: Look, b: Look) -> bool {
            a.intersection(&b) <= a
        }
        quickcheck(prop as fn(_, _) -> _);
    }

    #[test]
    fn intersection_eoi() {
        fn prop(a: Look, b: Look) -> bool {
            a.intersection(&b).allows_eoi() == (a.allows_eoi() && b.allows_eoi())
        }
        quickcheck(prop as fn(_, _) -> _);
    }

    #[test]
    fn intersection_set() {
        fn prop(a: Look, b: Look) -> bool {
            a.intersection(&b).as_set() == &a.as_set().intersection(b.as_set())
        }
        quickcheck(prop as fn(_, _) -> _);
    }
}
