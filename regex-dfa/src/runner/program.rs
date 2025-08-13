// Copyright 2015-2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::{Debug, Error as FmtError, Formatter};
use std::u32;

pub type TableStateIdx = u32;

/// A DFA program implemented as a lookup table.
#[derive(Clone)]
pub struct TableInsts<Ret> {
    /// The log (rounded up) of the number of different equivalence classes of bytes.
    // We could save a bit more memory by storing the actual number instead of the log, because
    // then `table` could have length num_classes x num_instructions. However, then we need to
    // multiply (instead of just shifting) to look up the next state, and that slows us down by
    // 10-20%.
    //
    // TODO: we can probably save more memory by splitting classes into ASCII/non-ASCII. Often,
    // many states share the same non-ASCII transitions, so those tables can be merged.
    pub log_num_classes: u32,
    /// A vec of length 256 mapping from bytes to their class indices.
    pub byte_class: Vec<u8>,
    /// A `(1 << log_num_classes) x num_instructions`-long table.
    ///
    /// For a given input byte `b` in state `state`, we look up the next state using
    /// `table[state << log_num_classes + b]`.
    pub table: Vec<TableStateIdx>,
    /// If `accept[st]` is not `None` then `st` is accepting, and `accept[st]` is the data
    /// to return.
    pub accept: Vec<Option<Ret>>,
    /// Same as `accept`, but applies only at the end of the input.
    pub accept_at_eoi: Vec<Option<Ret>>,
}

impl<Ret: Debug> Debug for TableInsts<Ret> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        f.write_fmt(format_args!(
            "TableInsts ({} log_classes, {} instructions):\n",
            self.log_num_classes,
            self.accept.len()
        ))?;
        f.write_str("Byte classes: ")?;
        f.debug_map()
            .entries((0..256).map(|b| (b, self.byte_class[b])))
            .finish()?;

        let num_classes = 1 << self.log_num_classes;
        for idx in 0..self.accept.len() {
            f.write_fmt(format_args!("State {}:\n", idx))?;
            f.debug_map()
                .entries(
                    (0usize..num_classes)
                        .map(|c| (c, self.table[(idx << self.log_num_classes) + c]))
                        .filter(|x| x.1 != u32::MAX),
                )
                .finish()?;
            f.write_str("\n")?;
        }

        f.write_str("Accept: ")?;
        for idx in 0..self.accept.len() {
            if let Some(ref ret) = self.accept[idx] {
                f.write_fmt(format_args!("{} -> {:?}, ", idx, ret))?;
            }
        }

        f.write_str("Accept_at_eoi: ")?;
        for idx in 0..self.accept_at_eoi.len() {
            if let Some(ref ret) = self.accept_at_eoi[idx] {
                f.write_fmt(format_args!("{} -> {:?}, ", idx, ret))?;
            }
        }
        Ok(())
    }
}

impl<Ret: Copy + Debug> TableInsts<Ret> {
    fn next_state(&self, state: usize, input: u8) -> Option<usize> {
        let class = self.byte_class[input as usize];
        let next_state = self.table[(state << self.log_num_classes) + class as usize];
        if next_state != u32::MAX {
            Some(next_state as usize)
        } else {
            None
        }
    }

    pub fn num_states(&self) -> usize {
        self.accept.len()
    }

    pub fn find_from(&self, input: &[u8], pos: usize, state: usize) -> Result<(usize, Ret), usize> {
        let mut state = state as u32;
        let mut ret = Err(input.len());

        if state as usize >= self.accept.len() {
            panic!("BUG");
        }
        for pos in pos..input.len() {
            if let Some(accept_ret) = self.accept[state as usize] {
                ret = Ok((pos, accept_ret));
            }

            // We've manually inlined next_state here, for better performance (measurably better
            // than using #[inline(always)]).
            // For some reason, these bounds checks (even though LLVM leaves them in) don't seem to
            // hurt performance.
            let class = self.byte_class[input[pos] as usize];
            state = self.table[((state as usize) << self.log_num_classes) + class as usize];

            // Since everything in `self.table` is either a valid state or u32::MAX, this is the
            // same as checking if state == u32::MAX. We write it this way in the hope that
            // rustc/LLVM will be able to elide the bounds check at the top of the loop.
            if state as usize >= self.accept.len() {
                if ret.is_err() {
                    return Err(pos);
                }
                break;
            }
        }

        // If we made it to the end of the input, prefer a return value that is specific to EOI
        // over one that can occur anywhere.
        if (state as usize) < self.accept.len() {
            if let Some(accept_ret) = self.accept_at_eoi[state as usize] {
                return Ok((input.len(), accept_ret));
            }
        }
        ret
    }

    pub fn longest_backward_find_from(
        &self,
        input: &[u8],
        pos: usize,
        mut state: usize,
    ) -> Option<(usize, Ret)> {
        let mut ret = None;
        for pos in (0..pos).rev() {
            if let Some(next_ret) = self.accept[state] {
                ret = Some((pos + 1, next_ret));
            }
            if let Some(next_state) = self.next_state(state, input[pos]) {
                state = next_state;
            } else {
                return ret;
            }
        }

        if let Some(end_ret) = self.accept_at_eoi[state] {
            Some((0, end_ret))
        } else {
            ret
        }
    }

    pub fn is_empty(&self) -> bool {
        self.num_states() == 0
    }
}
