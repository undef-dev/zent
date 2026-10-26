// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use super::{Tensor, TensorError};
use crate::ScalarPow;

impl<T: Copy> Tensor<T> {
    /// Возвращает новый тензор с поэлементными степенями
    pub fn try_pow<Exponent>(&self, exponents: &Tensor<Exponent>) -> Result<Self, TensorError>
    where
        T: ScalarPow<Exponent>,
        Exponent: Copy,
    {
        self.try_zip_map(exponents, |base, exponent| base.try_pow(exponent))
    }

    /// Возводит элементы в степени из правого тензора
    pub fn try_pow_assign<Exponent>(
        &mut self,
        exponents: &Tensor<Exponent>,
    ) -> Result<(), TensorError>
    where
        T: ScalarPow<Exponent>,
        Exponent: Copy,
    {
        self.try_zip_map_assign(exponents, |base, exponent| base.try_pow(exponent))
    }

    /// Записывает поэлементные степени в готовый тензор
    pub fn try_pow_into<Exponent>(
        &self,
        exponents: &Tensor<Exponent>,
        output: &mut Self,
    ) -> Result<(), TensorError>
    where
        T: ScalarPow<Exponent>,
        Exponent: Copy,
    {
        self.try_zip_map_into(exponents, output, |base, exponent| base.try_pow(exponent))
    }

    /// Возвращает новый тензор, возводя элементы в одну степень
    pub fn pow_scalar<Exponent>(&self, exponent: Exponent) -> Result<Self, TensorError>
    where
        T: ScalarPow<Exponent>,
        Exponent: Copy,
    {
        self.try_map(|base| base.try_pow(exponent))
    }

    /// Возводит все элементы в одну степень
    pub fn pow_scalar_assign<Exponent>(&mut self, exponent: Exponent) -> Result<(), TensorError>
    where
        T: ScalarPow<Exponent>,
        Exponent: Copy,
    {
        self.try_map_assign(|base| base.try_pow(exponent))
    }

    /// Записывает степени с общим показателем в готовый тензор
    pub fn pow_scalar_into<Exponent>(
        &self,
        exponent: Exponent,
        output: &mut Self,
    ) -> Result<(), TensorError>
    where
        T: ScalarPow<Exponent>,
        Exponent: Copy,
    {
        self.try_map_into(output, |base| base.try_pow(exponent))
    }
}
