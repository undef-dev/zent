// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

//! Символьная биграммная модель

mod bigram_model;
mod bigram_stats;
mod rng;
mod sampling;
mod tokenizer;

pub use bigram_model::{BigramModel, BigramModelError};
pub use bigram_stats::{BigramStats, BigramStatsError};
pub use rng::{GOLDEN_GAMMA, MIX_MULTIPLIER_1, MIX_MULTIPLIER_2, RandomSource, Rng};
pub use sampling::{SamplingError, sample_index};
pub use tokenizer::{CharTokenizer, TokenId, TokenizerError};
