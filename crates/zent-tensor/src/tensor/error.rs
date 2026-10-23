// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use crate::{ArithmeticError, Shape, StridesError};
use std::error::Error;
use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum TensorError {
    /// Арифметическая ошибка элемента с линейным индексом
    Arithmetic {
        index: usize,
        source: ArithmeticError,
    },
    /// Формы тензоров не совпадают
    ShapeMismatch { left: Shape, right: Shape },
    /// Число элементов хранилища не соответствует форме
    ElementCountMismatch { expected: usize, actual: usize },
    /// Не удалось вычислить шаги
    Strides(StridesError),
}

impl From<StridesError> for TensorError {
    fn from(value: StridesError) -> Self {
        Self::Strides(value)
    }
}

impl Error for TensorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Arithmetic { source, .. } => Some(source),
            Self::Strides(error) => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for TensorError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Arithmetic { index, .. } => {
                write!(formatter, "arithmetic error at element {index}")
            }

            Self::ElementCountMismatch { expected, actual } => {
                write!(
                    formatter,
                    "tensor shape requires {expected} elements, but storage contains {actual}"
                )
            }

            Self::Strides(_) => {
                write!(formatter, "failed to compute tensor strides")
            }

            Self::ShapeMismatch { left, right } => {
                write!(
                    formatter,
                    "tensor shapes do not match: {:?} and {:?}",
                    left.dims(),
                    right.dims()
                )
            }
        }
    }
}
