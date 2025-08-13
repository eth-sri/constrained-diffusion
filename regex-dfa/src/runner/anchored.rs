// Copyright 2015-2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::runner::program::TableInsts;
use crate::runner::Engine;
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct AnchoredEngine<Ret> {
    prog: TableInsts<Ret>,
}

impl<Ret: Copy + Debug> AnchoredEngine<Ret> {
    pub fn new(prog: TableInsts<Ret>) -> AnchoredEngine<Ret> {
        AnchoredEngine { prog: prog }
    }
}

impl<Ret: Copy + Debug + 'static> Engine<Ret> for AnchoredEngine<Ret> {
    fn find(&self, s: &str) -> Option<(usize, usize, Ret)> {
        let input = s.as_bytes();
        if self.prog.is_empty() {
            None
        } else if let Ok(end) = self.prog.find_from(input, 0, 0) {
            Some((0, end.0, end.1))
        } else {
            None
        }
    }

    fn clone_box(&self) -> Box<dyn Engine<Ret>> {
        Box::new(self.clone())
    }
}
