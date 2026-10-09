// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use crate::Shape;
use std::error::Error;
use std::fmt;

/// Шаги по осям тензора в элементах
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strides {
    steps: Vec<usize>,
}

impl Strides {
    // pub fn new (shape: &Shape) -> Result<Self, StridesError>

    /// Вычисляет шаги для расположения элементов подряд
    /// Последняя ось меняется быстрее остальных
    pub fn from_shape(shape: &Shape) -> Result<Self, StridesError> {
        // Шаг последней оси = 1 элемент
        // Шаг каждой предыдущей оси = размер следующей оси * шаг следующей оси

        let mut steps = Vec::with_capacity(shape.rank());

        // У скаляра нет осей и шагов
        if shape.rank() == 0 {
            return Ok(Self { steps });
        }

        let mut step = 1_usize;
        steps.push(step);

        shape.dims()[1..].iter().rev().try_for_each(|&d| {
            step = step.checked_mul(d).ok_or(StridesError::StepOverflow)?;
            steps.push(step);
            Ok::<(), StridesError>(())
        })?;

        steps.reverse();

        Ok(Self { steps })
    }

    /// Возвращает шаги по осям
    pub fn as_slice(&self) -> &[usize] {
        &self.steps
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StridesError {
    /// Шаг не помещается в `usize`
    StepOverflow,
}

impl Error for StridesError {}

impl fmt::Display for StridesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StepOverflow => {
                write!(formatter, "Stride exceeds usize capacity")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_has_no_strides() {
        let shape = Shape::new(vec![]).unwrap();
        let strides = Strides::from_shape(&shape).unwrap();

        assert_eq!(strides.as_slice(), &[]);
    }

    #[test]
    fn vector_has_unit_stride() {
        let shape = Shape::new(vec![3]).unwrap();
        let strides = Strides::from_shape(&shape).unwrap();

        assert_eq!(strides.as_slice(), &[1]);
    }

    #[test]
    fn matrix_strides_are_row_major() {
        let shape = Shape::new(vec![2, 3]).unwrap();
        let strides = Strides::from_shape(&shape).unwrap();

        assert_eq!(strides.as_slice(), &[3, 1]);
    }

    #[test]
    fn three_dimensional_strides_are_row_major() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        let strides = Strides::from_shape(&shape).unwrap();

        assert_eq!(strides.as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn unit_dimensions_preserve_axes() {
        let shape = Shape::new(vec![2, 1, 3]).unwrap();
        let strides = Strides::from_shape(&shape).unwrap();

        assert_eq!(strides.as_slice(), &[3, 3, 1]);
    }

    #[test]
    fn zero_dimension_sets_only_preceding_strides_to_zero() {
        let cases = [
            (vec![0], vec![1]),
            (vec![0, 3, 4], vec![12, 4, 1]),
            (vec![2, 0, 3], vec![0, 3, 1]),
            (vec![2, 3, 0], vec![0, 0, 1]),
        ];

        for (dims, expected) in cases {
            let shape = Shape::new(dims).unwrap();
            let strides = Strides::from_shape(&shape).unwrap();

            assert_eq!(strides.as_slice(), expected.as_slice(), "{shape:?}");
        }
    }

    #[test]
    fn stride_can_equal_usize_max() {
        let shape = Shape::new(vec![1, usize::MAX]).unwrap();
        let strides = Strides::from_shape(&shape).unwrap();

        assert_eq!(strides.as_slice(), &[usize::MAX, 1]);
    }

    #[test]
    fn zero_sized_shape_can_have_stride_overflow() {
        let shape = Shape::new(vec![0, usize::MAX, 2]).unwrap();

        assert_eq!(Strides::from_shape(&shape), Err(StridesError::StepOverflow));
    }

    #[test]
    fn trailing_zero_prevents_stride_overflow() {
        let shape = Shape::new(vec![usize::MAX, 2, 0]).unwrap();
        let strides = Strides::from_shape(&shape).unwrap();

        assert_eq!(strides.as_slice(), &[0, 0, 1]);
    }
}
