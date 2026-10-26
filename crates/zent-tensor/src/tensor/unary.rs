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
        self.try_map(ScalarNeg::try_neg)
    }

    /// Возвращает новый тензор с модулями элементов
    pub fn try_abs(&self) -> Result<Self, TensorError>
    where
        T: ScalarAbs,
    {
        self.try_map(ScalarAbs::try_abs)
    }

    /// Заменяет элементы их модулями
    pub fn try_abs_assign(&mut self) -> Result<(), TensorError>
    where
        T: ScalarAbs,
    {
        self.try_map_assign(ScalarAbs::try_abs)
    }

    /// Записывает модули элементов в готовый тензор
    pub fn try_abs_into(&self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarAbs,
    {
        self.try_map_into(output, ScalarAbs::try_abs)
    }

    /// Возвращает новый тензор с квадратными корнями элементов
    pub fn try_sqrt(&self) -> Result<Self, TensorError>
    where
        T: ScalarSqrt,
    {
        self.try_map(ScalarSqrt::try_sqrt)
    }

    /// Заменяет элементы их квадратными корнями
    pub fn try_sqrt_assign(&mut self) -> Result<(), TensorError>
    where
        T: ScalarSqrt,
    {
        self.try_map_assign(ScalarSqrt::try_sqrt)
    }

    /// Записывает квадратные корни элементов в готовый тензор
    pub fn try_sqrt_into(&self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarSqrt,
    {
        self.try_map_into(output, ScalarSqrt::try_sqrt)
    }

    /// Возвращает новый тензор с экспонентами элементов
    pub fn try_exp(&self) -> Result<Self, TensorError>
    where
        T: ScalarExp,
    {
        self.try_map(ScalarExp::try_exp)
    }

    /// Заменяет элементы их экспонентами
    pub fn try_exp_assign(&mut self) -> Result<(), TensorError>
    where
        T: ScalarExp,
    {
        self.try_map_assign(ScalarExp::try_exp)
    }

    /// Записывает экспоненты элементов в готовый тензор
    pub fn try_exp_into(&self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarExp,
    {
        self.try_map_into(output, ScalarExp::try_exp)
    }

    /// Возвращает новый тензор с натуральными логарифмами элементов
    pub fn try_ln(&self) -> Result<Self, TensorError>
    where
        T: ScalarLn,
    {
        self.try_map(ScalarLn::try_ln)
    }

    /// Заменяет элементы их натуральными логарифмами
    pub fn try_ln_assign(&mut self) -> Result<(), TensorError>
    where
        T: ScalarLn,
    {
        self.try_map_assign(ScalarLn::try_ln)
    }

    /// Записывает натуральные логарифмы элементов в готовый тензор
    pub fn try_ln_into(&self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarLn,
    {
        self.try_map_into(output, ScalarLn::try_ln)
    }
}
