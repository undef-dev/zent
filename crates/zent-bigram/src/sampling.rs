// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::fmt;

const EPSILON: f64 = 1e-12;

/// Выбирает индекс по вероятностям
///
/// Требует непустой срез, `random` в `[0, 1)` и вероятности в `[0, 1]`.
/// Сумма должна отличаться от 1 меньше чем на `1e-12`, иначе возвращается ошибка.
pub fn sample_index(probabilities: &[f64], random: f64) -> Result<usize, SamplingError> {
    if probabilities.is_empty() {
        return Err(SamplingError::EmptyDistribution);
    }

    if !(0.0..1.0).contains(&random) {
        return Err(SamplingError::InvalidRandomValue(random));
    }

    let mut sum = 0.0;

    for (index, &value) in probabilities.iter().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(SamplingError::InvalidProbability { index, value });
        }

        sum += value;
    }

    if (sum - 1.0).abs() >= EPSILON {
        return Err(SamplingError::InvalidProbabilitySum(sum));
    }

    let mut cumulative = 0.0;
    for (index, &value) in probabilities.iter().enumerate() {
        cumulative += value;
        if random < cumulative {
            return Ok(index);
        }
    }

    Ok(probabilities.len() - 1)
}

/// Ошибка выборки
#[derive(Debug, PartialEq)]
pub enum SamplingError {
    /// Недопустимая вероятность
    InvalidProbability { index: usize, value: f64 },

    /// Случайное число вне `[0, 1)`
    InvalidRandomValue(f64),

    /// Недопустимая сумма вероятностей
    InvalidProbabilitySum(f64),

    /// Пустое распределение
    EmptyDistribution,
}

impl Error for SamplingError {}

impl fmt::Display for SamplingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDistribution => {
                write!(formatter, "Probability distribution must not be empty")
            }

            Self::InvalidRandomValue(value) => {
                write!(
                    formatter,
                    "Random value `{value}` must be in the range [0.0, 1.0)"
                )
            }

            Self::InvalidProbability { index, value } => {
                write!(
                    formatter,
                    "Invalid probability `{value}` at index `{index}`"
                )
            }

            Self::InvalidProbabilitySum(sum) => {
                write!(
                    formatter,
                    "Probability distribution must sum to 1.0, got `{sum}`"
                )
            }
        }
    }
}
