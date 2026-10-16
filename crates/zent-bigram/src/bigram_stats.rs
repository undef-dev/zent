// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use crate::tokenizer::TokenId;
use std::error::Error;
use std::fmt;

/// Счётчики переходов между токенами, хранящиеся построчно
#[derive(Debug, PartialEq)]
pub struct BigramStats {
    counts: Vec<u64>,
    vocab_size: usize,
}

impl BigramStats {
    /// Создаёт нулевую таблицу; пустой словарь или переполнение размера вызывает ошибку
    pub fn new(vocab_size: usize) -> Result<Self, BigramStatsError> {
        if vocab_size == 0 {
            return Err(BigramStatsError::EmptyVocabulary);
        }

        Ok(Self {
            counts: vec![
                0;
                vocab_size
                    .checked_mul(vocab_size)
                    .ok_or(BigramStatsError::MatrixSizeOverflow)?
            ],
            vocab_size,
        })
    }

    /// Добавляет переходы соседних токенов к накопленным счётчикам
    ///
    /// Срез короче двух токенов ничего не меняет. Неизвестный токен или переполнение
    /// вызывает ошибку; уже учтённые переходы сохраняются.
    pub fn observe(&mut self, tokens: &[TokenId]) -> Result<(), BigramStatsError> {
        for window in tokens.windows(2) {
            let current = window[0];
            let next = window[1];
            let index = self.index(current, next)?;

            self.counts[index] = self.counts[index]
                .checked_add(1)
                .ok_or(BigramStatsError::CountOverflow { current, next })?;
        }

        Ok(())
    }

    /// Возвращает число наблюдений перехода `current -> next`
    pub fn count(&self, current: TokenId, next: TokenId) -> Result<u64, BigramStatsError> {
        Ok(self.counts[self.index(current, next)?])
    }

    /// Возвращает вероятности следующего токена в порядке словаря
    ///
    /// Неизвестный токен, отсутствие переходов или переполнение суммы вызывает ошибку.
    pub fn probabilities(&self, current: TokenId) -> Result<Vec<f64>, BigramStatsError> {
        let current_idx = self.validate_token(current)?;
        let start = current_idx * self.vocab_size;
        let end = start + self.vocab_size;
        let row = &self.counts[start..end];

        let total = row.iter().try_fold(0_u64, |acc, &count| {
            acc.checked_add(count)
                .ok_or(BigramStatsError::OutgoingCountOverflow(current))
        })?;

        if total == 0 {
            return Err(BigramStatsError::NoOutgoingTransitions(current));
        }

        let probabilities = row
            .iter()
            .map(|&count| count as f64 / total as f64)
            .collect();

        Ok(probabilities)
    }

    /// Возвращает плоский индекс перехода
    fn index(&self, current: TokenId, next: TokenId) -> Result<usize, BigramStatsError> {
        let current_idx = self.validate_token(current)?;
        let next_idx = self.validate_token(next)?;

        let index = current_idx
            .checked_mul(self.vocab_size)
            .and_then(|row| row.checked_add(next_idx))
            .ok_or(BigramStatsError::MatrixIndexOverflow)?;

        Ok(index)
    }

    /// Проверяет токен и возвращает его индекс в словаре
    fn validate_token(&self, token: TokenId) -> Result<usize, BigramStatsError> {
        let token_idx = token
            .index()
            .ok_or(BigramStatsError::TokenIndexConversionOverflow)?;

        if token_idx >= self.vocab_size {
            return Err(BigramStatsError::TokenOutOfVocabulary {
                token,
                vocab_size: self.vocab_size,
            });
        }

        Ok(token_idx)
    }
}

/// Ошибка статистики переходов
#[derive(Debug, PartialEq, Eq)]
pub enum BigramStatsError {
    /// Токен вне словаря
    TokenOutOfVocabulary { token: TokenId, vocab_size: usize },

    /// Переполнение счётчика перехода
    CountOverflow { current: TokenId, next: TokenId },

    /// Нет исходящих переходов
    NoOutgoingTransitions(TokenId),

    /// Пустой словарь
    EmptyVocabulary,

    /// Переполнение размера матрицы
    MatrixSizeOverflow,

    /// Переполнение индекса матрицы
    MatrixIndexOverflow,

    /// Индекс токена не помещается в `usize`
    TokenIndexConversionOverflow,

    /// Переполнение суммы исходящих переходов
    OutgoingCountOverflow(TokenId),
}

impl Error for BigramStatsError {}

impl fmt::Display for BigramStatsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenOutOfVocabulary { token, vocab_size } => {
                write!(
                    formatter,
                    "Token id `{token}` is outside vocabulary of size `{vocab_size}`"
                )
            }

            Self::CountOverflow { current, next } => {
                write!(
                    formatter,
                    "Transition count overflow for `{current} -> {next}`"
                )
            }

            Self::NoOutgoingTransitions(token) => {
                write!(
                    formatter,
                    "No outgoing transitions observed for token `{token}`"
                )
            }

            Self::EmptyVocabulary => {
                write!(formatter, "Vocabulary must contain at least one token")
            }

            Self::MatrixSizeOverflow => {
                write!(
                    formatter,
                    "Bigram matrix size exceeds the maximum addressable `usize` range"
                )
            }

            Self::MatrixIndexOverflow => {
                write!(
                    formatter,
                    "Bigram matrix index exceeds the maximum `usize` value"
                )
            }

            Self::TokenIndexConversionOverflow => {
                write!(
                    formatter,
                    "Token id cannot be represented as `usize` on this platform"
                )
            }

            Self::OutgoingCountOverflow(token) => {
                write!(
                    formatter,
                    "Total outgoing transition count overflow for token `{token}`"
                )
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/bigram_stats.rs"]
mod tests;
