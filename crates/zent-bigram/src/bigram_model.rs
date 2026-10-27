// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use crate::RandomSource;
use crate::bigram_stats::{BigramStats, BigramStatsError};
use crate::sampling::{SamplingError, sample_index};
use crate::tokenizer::TokenId;
use std::error::Error;
use std::fmt;

/// Биграммная модель со статистикой переходов и внешним источником случайных чисел
pub struct BigramModel {
    stats: BigramStats,
}

impl BigramModel {
    /// Создаёт модель из статистики переходов
    pub fn new(stats: BigramStats) -> Self {
        Self { stats }
    }

    /// Выбирает следующий токен по статистике; ошибки передаёт через [`BigramModelError`]
    pub fn next_token(
        &self,
        current: TokenId,
        rng: &mut impl RandomSource,
    ) -> Result<TokenId, BigramModelError> {
        let index = sample_index(&self.stats.probabilities(current)?, rng.next_f64())?;

        TokenId::from_index(index).ok_or(BigramModelError::InvalidTokenIndex(index))
    }

    /// Генерирует `count` новых токенов, не включая начальный `current`
    ///
    /// Каждый токен задаёт следующий шаг; при нулевой длине результат пуст.
    /// Ошибка любого шага прерывает генерацию.
    pub fn generate(
        &self,
        current: TokenId,
        count: usize,
        rng: &mut impl RandomSource,
    ) -> Result<Vec<TokenId>, BigramModelError> {
        let mut generated = Vec::with_capacity(count);
        let mut current = current;

        for _ in 0..count {
            current = self.next_token(current, rng)?;
            generated.push(current);
        }

        Ok(generated)
    }
}

/// Ошибка биграммной модели
#[derive(Debug)]
pub enum BigramModelError {
    /// Ошибка статистики переходов
    Stats(BigramStatsError),

    /// Ошибка выборки
    Sampling(SamplingError),

    /// Индекс не помещается в [`TokenId`]
    InvalidTokenIndex(usize),
}

impl From<BigramStatsError> for BigramModelError {
    fn from(error: BigramStatsError) -> Self {
        Self::Stats(error)
    }
}

impl From<SamplingError> for BigramModelError {
    fn from(error: SamplingError) -> Self {
        Self::Sampling(error)
    }
}

impl Error for BigramModelError {}

impl fmt::Display for BigramModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stats(error) => {
                write!(formatter, "bigram statistics error: {error}")
            }
            Self::Sampling(error) => {
                write!(formatter, "sampling error: {error}")
            }
            Self::InvalidTokenIndex(index) => {
                write!(formatter, "invalid token index: {index}")
            }
        }
    }
}
