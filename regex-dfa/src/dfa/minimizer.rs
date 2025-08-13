// Copyright 2015-2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::dfa::{Dfa, RetTrait};
use crate::nfa::{Accept, StateIdx, StateSet};
use crate::refinery::Partition;
use range_map::{RangeMultiMap, RangeSet};
use std::collections::{HashMap, HashSet};

pub struct Minimizer {
    partition: Partition,
    distinguishers: HashSet<usize>,
    // The reversed transitions of the dfa.
    rev: Vec<RangeMultiMap<u8, StateIdx>>,
}

impl Minimizer {
    // Partition the states according to
    // - when they accept,
    // - what they return if they do sometimes accept, and
    // - what set of bytes do we expect to see next.
    fn initial_partition<Ret: RetTrait>(dfa: &Dfa<Ret>) -> Vec<Vec<StateIdx>> {
        let mut part: HashMap<(Accept, Option<&Ret>, RangeSet<u8>), Vec<StateIdx>> = HashMap::new();
        for (idx, st) in dfa.states.iter().enumerate() {
            let chars = st.transitions.to_range_set();
            part.entry((st.accept, dfa.ret(idx), chars))
                .or_insert_with(Vec::new)
                .push(idx);
        }
        part.into_iter().map(|x| x.1).collect()
    }

    // Refine the current partition based on the fact that everything in `splitter` is distinct
    // from everything not in it.
    fn refine(&mut self, splitter: &[StateIdx]) {
        let dists = &mut self.distinguishers;

        self.partition
            .refine_with_callback(splitter, |p, int_idx, diff_idx| {
                if dists.contains(&int_idx) || p.part(diff_idx).len() < p.part(int_idx).len() {
                    dists.insert(diff_idx);
                } else {
                    dists.insert(int_idx);
                }
            });
    }

    fn next_distinguisher(&mut self) -> Option<usize> {
        let maybe_elt = self.distinguishers.iter().next().cloned();
        if let Some(elt) = maybe_elt {
            self.distinguishers.remove(&elt);
        }
        maybe_elt
    }

    fn get_input_sets(&mut self, part_idx: usize) -> Vec<StateSet> {
        let inputs: Vec<_> = self
            .partition
            .part(part_idx)
            .iter()
            .flat_map(|s| self.rev[*s].ranges_values().cloned())
            .collect();
        if inputs.is_empty() {
            return Vec::new();
        }

        let inputs = RangeMultiMap::from_vec(inputs);
        let mut sets: Vec<StateSet> = inputs
            .group()
            .ranges_values()
            .map(|&(_, ref x)| x.clone())
            .collect();
        for set in &mut sets {
            set.sort();
        }
        sets.sort();
        sets.dedup();
        sets
    }

    fn compute_partition(&mut self) {
        while let Some(dist) = self.next_distinguisher() {
            let sets = self.get_input_sets(dist);

            for set in &sets {
                self.refine(set);
            }
        }
    }

    pub fn minimize<Ret: RetTrait>(dfa: &Dfa<Ret>) -> Dfa<Ret> {
        let mut min = Minimizer::new(dfa);

        min.compute_partition();

        let mut ret = Dfa::new();

        // We need to re-index the states: build a map that maps old indices to
        // new indices.
        let mut old_state_to_new = vec![0; dfa.num_states()];
        for part in min.partition.iter() {
            // This unwrap is safe because we don't allow any empty sets into the partition.
            let rep_idx = *part.iter().next().unwrap();
            ret.states.push(dfa.states[rep_idx].clone());

            for &state in part.iter() {
                old_state_to_new[state] = ret.states.len() - 1;
            }
        }

        ret.map_states(|s: StateIdx| old_state_to_new[s]);
        ret.init = dfa
            .init
            .iter()
            .map(|x| x.map(|s: StateIdx| old_state_to_new[s]))
            .collect();
        ret
    }

    fn new<Ret: RetTrait>(dfa: &Dfa<Ret>) -> Minimizer {
        let init = Minimizer::initial_partition(dfa);
        let part = Partition::new(
            init.into_iter().map(|set| set.into_iter()),
            dfa.num_states(),
        );

        // According to Hopcroft's algorithm, we're allowed to leave out one of the distinguishers
        // (at least, as long as it isn't a set of accepting states). Choose the one with the
        // most states to leave out.
        let mut dists: HashSet<usize> = (0..part.num_parts()).collect();
        let worst = (0..dists.len())
            .filter(|i| dfa.states[part.part(*i)[0]].accept == Accept::Never)
            .max_by_key(|i| part.part(*i).len());
        if let Some(worst) = worst {
            dists.remove(&worst);
        }

        Minimizer {
            partition: part,
            distinguishers: dists,
            rev: dfa.reversed_transitions(),
        }
    }
}
