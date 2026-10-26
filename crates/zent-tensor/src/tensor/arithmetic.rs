// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use super::{Tensor, TensorError};
use crate::{ScalarAdd, ScalarDiv, ScalarMul, ScalarSub};

/// Поэлементные операции
impl<T: Copy> Tensor<T> {
    /// Возвращает новый тензор с поэлементной суммой
    pub fn try_add(&self, rhs: &Self) -> Result<Self, TensorError>
    where
        T: ScalarAdd,
    {
        self.try_zip_map(rhs, |left, right| left.try_add(right))
    }

    /// Поэлементно прибавляет правый тензор к текущему
    pub fn try_add_assign(&mut self, rhs: &Self) -> Result<(), TensorError>
    where
        T: ScalarAdd,
    {
        self.try_zip_map_assign(rhs, |left, right| left.try_add(right))
    }

    /// Записывает поэлементную сумму в готовый тензор
    pub fn try_add_into(&self, rhs: &Self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarAdd,
    {
        self.try_zip_map_into(rhs, output, |left, right| left.try_add(right))
    }

    /// Возвращает новый тензор с поэлементной разностью
    pub fn try_sub(&self, rhs: &Self) -> Result<Self, TensorError>
    where
        T: ScalarSub,
    {
        self.try_zip_map(rhs, |left, right| left.try_sub(right))
    }

    /// Поэлементно вычитает правый тензор из текущего
    pub fn try_sub_assign(&mut self, rhs: &Self) -> Result<(), TensorError>
    where
        T: ScalarSub,
    {
        self.try_zip_map_assign(rhs, |left, right| left.try_sub(right))
    }

    /// Записывает поэлементную разность в готовый тензор
    pub fn try_sub_into(&self, rhs: &Self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarSub,
    {
        self.try_zip_map_into(rhs, output, |left, right| left.try_sub(right))
    }

    /// Возвращает новый тензор с поэлементным произведением
    pub fn try_mul(&self, rhs: &Self) -> Result<Self, TensorError>
    where
        T: ScalarMul,
    {
        self.try_zip_map(rhs, |left, right| left.try_mul(right))
    }

    /// Поэлементно умножает текущий тензор на правый
    pub fn try_mul_assign(&mut self, rhs: &Self) -> Result<(), TensorError>
    where
        T: ScalarMul,
    {
        self.try_zip_map_assign(rhs, |left, right| left.try_mul(right))
    }

    /// Записывает поэлементное произведение в готовый тензор
    pub fn try_mul_into(&self, rhs: &Self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarMul,
    {
        self.try_zip_map_into(rhs, output, |left, right| left.try_mul(right))
    }

    /// Возвращает новый тензор с поэлементным частным
    pub fn try_div(&self, rhs: &Self) -> Result<Self, TensorError>
    where
        T: ScalarDiv,
    {
        self.try_zip_map(rhs, |left, right| left.try_div(right))
    }

    /// Поэлементно делит текущий тензор на правый
    pub fn try_div_assign(&mut self, rhs: &Self) -> Result<(), TensorError>
    where
        T: ScalarDiv,
    {
        self.try_zip_map_assign(rhs, |left, right| left.try_div(right))
    }

    /// Записывает поэлементное частное в готовый тензор
    pub fn try_div_into(&self, rhs: &Self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarDiv,
    {
        self.try_zip_map_into(rhs, output, |left, right| left.try_div(right))
    }
}
