// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_bigram::{
    BigramModel, BigramModelError, BigramStats, BigramStatsError, CharTokenizer, RandomSource, Rng,
    SamplingError, TokenId,
};

struct FixedRandom(f64);

impl RandomSource for FixedRandom {
    fn next_f64(&mut self) -> f64 {
        self.0
    }
}

#[test]
fn generate() {
    let tokenizer = CharTokenizer::from_corpus("abba").unwrap();
    let tokens = tokenizer.encode("abba").unwrap();
    let mut stats = BigramStats::new(tokenizer.vocab_size()).unwrap();
    stats.observe(&tokens).unwrap();
    let model = BigramModel::new(stats);
    let start = *tokens.last().unwrap();
    let mut rng = Rng::new(42);

    let empty = model.generate(start, 0, &mut rng).unwrap();
    assert!(empty.is_empty());
    let generated = model.generate(start, 8, &mut rng).unwrap();
    assert_eq!(tokenizer.decode(&generated).unwrap(), "bababbab");
}

#[test]
fn errors() {
    let [first, middle, last] = [0, 1, 2].map(|i| TokenId::from_index(i).unwrap());
    let mut stats = BigramStats::new(3).unwrap();
    stats.observe(&[first, middle, last]).unwrap();
    let model = BigramModel::new(stats);

    assert!(matches!(
        model.generate(first, 3, &mut FixedRandom(0.0)),
        Err(BigramModelError::Stats(BigramStatsError::NoOutgoingTransitions(id))) if id == last
    ));
    assert!(matches!(
        model.next_token(first, &mut FixedRandom(1.0)),
        Err(BigramModelError::Sampling(SamplingError::InvalidRandomValue(value))) if value == 1.0
    ));
}
