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
