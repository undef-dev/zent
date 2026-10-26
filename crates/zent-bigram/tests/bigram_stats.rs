// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_bigram::{BigramStats, BigramStatsError, TokenId};

fn token(index: usize) -> TokenId {
    TokenId::from_index(index).unwrap()
}

#[test]
fn observe() {
    let mut stats = BigramStats::new(2).unwrap();
    stats
        .observe(&[token(0), token(1), token(1), token(0)])
        .unwrap();
    stats.observe(&[token(0), token(1)]).unwrap();

    assert_eq!(stats.count(token(0), token(1)), Ok(2));
    assert_eq!(stats.probabilities(token(1)).unwrap(), [0.5, 0.5]);
}

#[test]
fn errors() {
    assert_eq!(
        BigramStats::new(usize::MAX),
        Err(BigramStatsError::MatrixSizeOverflow)
    );

    let mut stats = BigramStats::new(2).unwrap();
    let invalid = token(2);
    let expected = BigramStatsError::TokenOutOfVocabulary {
        token: invalid,
        vocab_size: 2,
    };

    for (current, next) in [(invalid, token(0)), (token(0), invalid)] {
        assert_eq!(stats.observe(&[current, next]).unwrap_err(), expected);
    }
}
