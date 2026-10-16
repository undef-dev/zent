// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_bigram::{SamplingError, sample_index};

#[test]
fn intervals() {
    for (random, expected) in [
        (0.0, 0),
        (0.199999999, 0),
        (0.2, 2),
        (0.499999999, 2),
        (0.5, 3),
        (0.999999999, 3),
    ] {
        assert_eq!(
            sample_index(&[0.2, 0.0, 0.3, 0.5], random),
            Ok(expected),
            "{random}"
        );
    }
    assert_eq!(sample_index(&[0.0, 1.0], 0.0), Ok(1));
}

#[test]
fn invalid_values() {
    for random in [-0.1, 1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                sample_index(&[1.0], random),
                Err(SamplingError::InvalidRandomValue(_))
            ),
            "{random}"
        );
    }
    for probability in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                sample_index(&[0.2, probability, 0.5], 0.5),
                Err(SamplingError::InvalidProbability { index: 1, .. })
            ),
            "{probability}"
        );
    }
}

#[test]
fn invalid_distribution() {
    assert_eq!(
        sample_index(&[], 0.5),
        Err(SamplingError::EmptyDistribution)
    );
    for probabilities in [[0.2, 0.3, 0.4], [0.5, 0.5, 0.5]] {
        assert!(
            matches!(
                sample_index(&probabilities, 0.5),
                Err(SamplingError::InvalidProbabilitySum(_))
            ),
            "{probabilities:?}"
        );
    }
}
