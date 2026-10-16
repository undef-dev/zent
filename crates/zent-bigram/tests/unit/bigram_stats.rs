// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use super::{BigramStats, BigramStatsError, TokenId};

#[test]
fn overflow() {
    let current = TokenId::from_index(0).unwrap();
    let next = TokenId::from_index(1).unwrap();
    let mut stats = BigramStats::new(2).unwrap();
    stats.counts[1] = u64::MAX;

    assert_eq!(
        stats.observe(&[current, next]),
        Err(BigramStatsError::CountOverflow { current, next })
    );
    assert_eq!(stats.count(current, next), Ok(u64::MAX));

    stats.counts[0] = 1;
    assert_eq!(
        stats.probabilities(current),
        Err(BigramStatsError::OutgoingCountOverflow(current))
    );
}
