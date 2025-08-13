// Copyright 2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::dfa::{Dfa, RetTrait};
use crate::nfa::{Nfa, NoLooks, StateIdx};
use num_traits::PrimInt;
use std::collections::HashSet;
use std::fmt::Debug;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DfsInstruction {
    Continue,
    #[allow(dead_code)]
    TurnBack,
    Stop,
}

pub trait Graph {
    fn num_states(&self) -> usize;

    fn neighbors<'a>(&'a self, i: StateIdx) -> Box<dyn Iterator<Item = StateIdx> + 'a>;

    /// Does a depth-first search of this graph.
    ///
    /// Every time the search visits a new state, `visit` will be called. Every time the search
    /// detects a loop, `cycle` will be called. These return value of these callbacks tell the
    /// search how to proceed:
    /// - on `Continue`, the search will proceed normally
    /// - on `TurnBack`, the search will stop searching the current branch
    /// - on `Stop`, the search will terminate early.
    fn dfs<Inits, Visit, Cycle>(&self, init: Inits, mut visit: Visit, mut cycle: Cycle)
    where
        Visit: FnMut(&[StateIdx]) -> DfsInstruction,
        Cycle: FnMut(&[StateIdx]) -> DfsInstruction,
        Inits: Iterator<Item = StateIdx>,
    {
        // Pairs of (state, children_left_to_explore).
        let mut stack: Vec<StateIdx> = Vec::with_capacity(self.num_states());
        let mut remaining_children_stack: Vec<Box<dyn Iterator<Item = StateIdx>>> =
            Vec::with_capacity(self.num_states());
        let mut visiting: Vec<bool> = vec![false; self.num_states()];
        let mut done: Vec<bool> = vec![false; self.num_states()];

        // For nodes that we are currently visiting, this is their position on the stack.
        let mut stack_pos: Vec<usize> = vec![0; self.num_states()];

        let start_states: Vec<StateIdx> = init.collect();

        for &start_idx in &start_states {
            if !done[start_idx] {
                match visit(&[start_idx][..]) {
                    DfsInstruction::Continue => {}
                    DfsInstruction::TurnBack => {
                        done[start_idx] = true;
                        continue;
                    }
                    DfsInstruction::Stop => {
                        return;
                    }
                }

                visiting[start_idx] = true;
                stack.push(start_idx);
                remaining_children_stack.push(self.neighbors(start_idx));
                stack_pos[start_idx] = 0;

                while !stack.is_empty() {
                    // We keep stack and remaining_children_stack synchronized.
                    debug_assert!(!remaining_children_stack.is_empty());

                    let cur = *stack.last().unwrap();
                    let next_child = remaining_children_stack.last_mut().unwrap().next();

                    if let Some(child) = next_child {
                        if visiting[child] {
                            // We found a cycle: report it (and maybe terminate early).
                            // Since we turn back on finding a cycle anyway, we treat Continue
                            // and TurnBack the same (i.e. we don't need to handle either one
                            // explicitly).
                            if cycle(&stack[stack_pos[child]..]) == DfsInstruction::Stop {
                                return;
                            }
                        } else if !done[child] {
                            // This is a new state: report it and push it onto the stack.
                            stack.push(child);
                            match visit(&stack[stack_pos[child]..]) {
                                DfsInstruction::Stop => {
                                    return;
                                }
                                DfsInstruction::TurnBack => {
                                    stack.pop();
                                    done[child] = true;
                                }
                                DfsInstruction::Continue => {
                                    remaining_children_stack.push(self.neighbors(child));
                                    visiting[child] = true;
                                    stack_pos[child] = stack.len() - 1;
                                }
                            }
                        }
                        continue;
                    }

                    // If we got this far, the current node is out of children. Pop it from the
                    // stack.
                    visiting[cur] = false;
                    done[cur] = true;
                    stack.pop();
                    remaining_children_stack.pop();
                }
            }
        }
    }

    /// The same as `dfs`, but runs on a graph with cuts in it.
    ///
    /// Instead of running on the full graph, runs on the graph where pairs in `cuts` are
    /// disconnected.
    fn dfs_with_cut<Inits, Cuts, Visit, Cycle>(
        &self,
        init: Inits,
        cuts: &HashSet<(StateIdx, StateIdx)>,
        mut visit: Visit,
        mut cycle: Cycle,
    ) where
        Visit: FnMut(&[StateIdx]) -> DfsInstruction,
        Cycle: FnMut(&[StateIdx]) -> DfsInstruction,
        Inits: Iterator<Item = StateIdx>,
    {
        let should_cut = |s: &[StateIdx]| {
            let len = s.len();
            len >= 2 && cuts.contains(&(s[len - 2], s[len - 1]))
        };
        let my_visit = |s: &[StateIdx]| {
            if should_cut(s) {
                DfsInstruction::TurnBack
            } else {
                visit(s)
            }
        };
        let my_cycle = |s: &[StateIdx]| {
            if should_cut(s) {
                DfsInstruction::TurnBack
            } else {
                cycle(s)
            }
        };
        self.dfs(init, my_visit, my_cycle);
    }

    /// Returns a list of states, visited in depth-first order.
    fn dfs_order<I: Iterator<Item = StateIdx>>(&self, init: I) -> Vec<StateIdx> {
        use self::DfsInstruction::*;

        let mut ret: Vec<StateIdx> = Vec::new();
        // The unwrap is ok because dfa guarantees never to pass an empty slice.
        self.dfs(
            init,
            |st| {
                ret.push(*st.last().unwrap());
                Continue
            },
            |_| Continue,
        );
        ret
    }

    /// Checks whether this graph has any cycles.
    #[allow(unused)]
    fn has_cycles(&self) -> bool {
        use self::DfsInstruction::*;

        let mut found = false;
        self.dfs(
            0..self.num_states(),
            |_| Continue,
            |_| {
                found = true;
                Stop
            },
        );
        found
    }
}

impl<T: RetTrait> Graph for Dfa<T> {
    fn num_states(&self) -> usize {
        Dfa::num_states(self)
    }

    fn neighbors<'a>(&'a self, i: StateIdx) -> Box<dyn Iterator<Item = StateIdx> + 'a> {
        Box::new(self.transitions(i).ranges_values().map(|x| x.1))
    }
}

impl<Tok: Debug + PrimInt> Graph for Nfa<Tok, NoLooks> {
    fn num_states(&self) -> usize {
        Nfa::num_states(self)
    }

    fn neighbors<'a>(&'a self, i: usize) -> Box<dyn Iterator<Item = usize> + 'a> {
        Box::new(self.consuming(i).ranges_values().map(|x| x.1))
    }
}

#[cfg(test)]
mod tests {
    use crate::dfa::tests::make_dfa;
    use crate::graph::Graph;

    #[test]
    fn cycles() {
        macro_rules! cyc {
            ($re:expr, $res:expr) => {{
                let dfa = make_dfa($re).unwrap();
                println!("{:?}", dfa);
                assert_eq!(dfa.has_cycles(), $res);
            }};
        }

        cyc!("abcde", false);
        cyc!("ab*d", true);
        cyc!("ab*", true);
        cyc!("ab*?", false);
        cyc!("ab+", true);
        cyc!("ab+?", false);
        cyc!("(ab*?|cde)", false);
        cyc!("(ab*?|cde)f", true);
        cyc!("(abc)*?", false);
        cyc!("(abc)*?def", true);
    }
}
