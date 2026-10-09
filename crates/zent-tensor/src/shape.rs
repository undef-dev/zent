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
    ///
    /// Пустой список `[]` соответствует скаляру с одним элементом
    /// Если хотя бы один размер равен нулю, число элементов также равно нулю
    ///
    /// Возвращает [`ShapeError::NumelOverflow`], если при подсчёте элементов возникает переполнение
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_reports_dimensions_rank_and_numel() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();

        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.numel(), 24);
    }

    #[test]
    fn scalar_shape_has_rank_zero_and_one_element() {
        let shape = Shape::new(vec![]).unwrap();

        assert_eq!(shape.dims(), &[]);
        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.numel(), 1);
    }

    #[test]
    fn zero_sized_shape_has_zero_elements() {
        let shape = Shape::new(vec![2, 0, 3]).unwrap();

        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.numel(), 0);
    }

    #[test]
    fn shape_rejects_numel_overflow() {
        let result = Shape::new(vec![usize::MAX, 2]);

        assert_eq!(result, Err(ShapeError::NumelOverflow));
    }

    #[test]
    fn zero_sized_shape_does_not_overflow() {
        let shape = Shape::new(vec![usize::MAX, 2, 0]).unwrap();

        assert_eq!(shape.numel(), 0);
    }
}
