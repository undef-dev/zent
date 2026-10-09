// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_bigram::bigram::BigramStats;
use zent_bigram::model::BigramModel;
use zent_bigram::rng::Rng;
use zent_bigram::tokenizer::CharTokenizer;

#[test]
fn public_level_zero_pipeline_remains_available() {
    let text = "abba";
    let tokenizer = CharTokenizer::from_corpus(text).unwrap();
    let tokens = tokenizer.encode(text).unwrap();
    let token_indices: Vec<usize> = tokens.iter().map(|token| token.index().unwrap()).collect();

    assert_eq!(token_indices, [0, 1, 1, 0]);

    let mut stats = BigramStats::new(tokenizer.vocab_size()).unwrap();
    stats.observe(&tokens).unwrap();

    let model = BigramModel::new(stats);
    let mut rng = Rng::new(42);
    let generated_tokens = model
        .generate(*tokens.last().unwrap(), 8, &mut rng)
        .unwrap();
    let generated_text = tokenizer.decode(&generated_tokens).unwrap();

    assert_eq!(generated_text, "bababbab");
}
