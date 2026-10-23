// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use super::{Tensor, TensorError};
use crate::{ScalarAbs, ScalarExp, ScalarLn, ScalarNeg, ScalarSqrt};

impl<T: Copy> Tensor<T> {
    /// Возвращает новый тензор с изменёнными знаками элементов
    pub fn negated(&self) -> Result<Self, TensorError>
    where
        T: ScalarNeg,
    {
        todo!()
    }

    /// Возвращает новый тензор с модулями элементов
    pub fn try_abs(&self) -> Result<Self, TensorError>
    where
        T: ScalarAbs,
    {
        todo!()
    }

    /// Заменяет элементы их модулями
    pub fn try_abs_assign(&mut self) -> Result<(), TensorError>
    where
        T: ScalarAbs,
    {
        todo!()
    }

    /// Записывает модули элементов в готовый тензор
    pub fn try_abs_into(&self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarAbs,
    {
        let _ = output;
        todo!()
    }

    /// Возвращает новый тензор с квадратными корнями элементов
    pub fn try_sqrt(&self) -> Result<Self, TensorError>
    where
        T: ScalarSqrt,
    {
        todo!()
    }

    /// Заменяет элементы их квадратными корнями
    pub fn try_sqrt_assign(&mut self) -> Result<(), TensorError>
    where
        T: ScalarSqrt,
    {
        todo!()
    }

    /// Записывает квадратные корни элементов в готовый тензор
    pub fn try_sqrt_into(&self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarSqrt,
    {
        let _ = output;
        todo!()
    }

    /// Возвращает новый тензор с экспонентами элементов
    pub fn try_exp(&self) -> Result<Self, TensorError>
    where
        T: ScalarExp,
    {
        todo!()
    }

    /// Заменяет элементы их экспонентами
    pub fn try_exp_assign(&mut self) -> Result<(), TensorError>
    where
        T: ScalarExp,
    {
        todo!()
    }

    /// Записывает экспоненты элементов в готовый тензор
    pub fn try_exp_into(&self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarExp,
    {
        let _ = output;
        todo!()
    }

    /// Возвращает новый тензор с натуральными логарифмами элементов
    pub fn try_ln(&self) -> Result<Self, TensorError>
    where
        T: ScalarLn,
    {
        todo!()
    }

    /// Заменяет элементы их натуральными логарифмами
    pub fn try_ln_assign(&mut self) -> Result<(), TensorError>
    where
        T: ScalarLn,
    {
        todo!()
    }

    /// Записывает натуральные логарифмы элементов в готовый тензор
    pub fn try_ln_into(&self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarLn,
    {
        let _ = output;
        todo!()
    }
}
