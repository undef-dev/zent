// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_bigram::{BigramStats, BigramStatsError, TokenId};

fn token(index: usize) -> TokenId {
    TokenId::from_index(index).unwrap()
}

#[test]
fn invalid_size() {
    assert_eq!(BigramStats::new(0), Err(BigramStatsError::EmptyVocabulary));
    assert_eq!(
        BigramStats::new(usize::MAX),
        Err(BigramStatsError::MatrixSizeOverflow)
    );
}

#[test]
fn observe() {
    let mut stats = BigramStats::new(2).unwrap();
    stats
        .observe(&[token(0), token(1), token(1), token(0)])
        .unwrap();
    stats.observe(&[]).unwrap();
    stats.observe(&[token(0)]).unwrap();
    stats.observe(&[token(0), token(1)]).unwrap();

    for (current, next, count) in [(0, 0, 0), (0, 1, 2), (1, 0, 1), (1, 1, 1)] {
        assert_eq!(stats.count(token(current), token(next)), Ok(count));
    }
    assert_eq!(stats.probabilities(token(0)).unwrap(), [0.0, 1.0]);
    assert_eq!(stats.probabilities(token(1)).unwrap(), [0.5, 0.5]);
}

#[test]
fn invalid_transitions() {
    let mut stats = BigramStats::new(2).unwrap();
    let invalid = token(2);
    let expected = BigramStatsError::TokenOutOfVocabulary {
        token: invalid,
        vocab_size: 2,
    };

    for (current, next) in [(invalid, token(0)), (token(0), invalid)] {
        assert_eq!(stats.count(current, next).unwrap_err(), expected);
        assert_eq!(stats.observe(&[current, next]).unwrap_err(), expected);
    }
    assert_eq!(stats.probabilities(invalid).unwrap_err(), expected);
    assert_eq!(
        stats.probabilities(token(0)),
        Err(BigramStatsError::NoOutgoingTransitions(token(0)))
    );
}
