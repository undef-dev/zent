// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::fmt;

/// Описывает форму тензора
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
    numel: usize,
}

impl Shape {
    /// Создаёт форму тензора по указанным размерам
    pub fn new(dims: Vec<usize>) -> Result<Self, ShapeError> {
        let numel = if dims.contains(&0) {
            0
        } else {
            dims.iter().try_fold(1usize, |acc, &e| {
                acc.checked_mul(e).ok_or(ShapeError::NumelOverflow)
            })?
        };

        Ok(Self { dims, numel })
    }

    /// Возвращает размеры тензора
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Возвращает число осей
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Возвращает число элементов
    pub fn numel(&self) -> usize {
        self.numel
    }
}

/// Ошибки Shape
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// Число элементов не помещается в `usize`
    NumelOverflow,
}

impl Error for ShapeError {}

impl fmt::Display for ShapeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NumelOverflow => {
                write!(
                    formatter,
                    "Number of tensor elements exceeds usize capacity"
                )
            }
        }
    }
}
