// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::fmt;

/// Ошибки квантования
#[derive(Debug, PartialEq, Eq)]
pub enum QuantizationError {
    /// Значение по указанному индексу не является конечным
    NonFiniteValue { index: usize },
    /// Масштаб ненулевых значений округлился до нуля
    ScaleUnderflow,
    /// Масштаб не является конечным положительным числом
    InvalidScale,
    /// Входное значение не является конечным
    NonFiniteInput,
    /// Результат восстановления не является конечным
    NonFiniteResult,
}

impl Error for QuantizationError {}

impl fmt::Display for QuantizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteValue { index } => {
                write!(formatter, "non-finite value at index {index}")
            }
            Self::ScaleUnderflow => {
                write!(formatter, "quantization scale rounded to zero")
            }
            Self::InvalidScale => {
                write!(formatter, "scale must be finite and positive")
            }
            Self::NonFiniteInput => {
                write!(formatter, "non-finite input value")
            }
            Self::NonFiniteResult => {
                write!(formatter, "dequantized value is not finite")
            }
        }
    }
}
