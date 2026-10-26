// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_bigram::{
    BigramModel, BigramModelError, BigramStats, BigramStatsError, RandomSource, TokenId,
};

struct FixedRandom;

impl RandomSource for FixedRandom {
    fn next_f64(&mut self) -> f64 {
        0.0
    }
}

#[test]
fn dead_end() {
    let [first, middle, last] = [0, 1, 2].map(|i| TokenId::from_index(i).unwrap());
    let mut stats = BigramStats::new(3).unwrap();
    stats.observe(&[first, middle, last]).unwrap();
    let model = BigramModel::new(stats);

    assert!(matches!(
        model.generate(first, 3, &mut FixedRandom),
        Err(BigramModelError::Stats(BigramStatsError::NoOutgoingTransitions(id))) if id == last
    ));
}
