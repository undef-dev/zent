// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use super::{Tensor, TensorError};
use crate::{ScalarAdd, ScalarDiv, ScalarMul, ScalarSub};

impl<T: Copy> Tensor<T> {
    /// Возвращает новый тензор, прибавляя число к каждому элементу
    pub fn add_scalar(&self, value: T) -> Result<Self, TensorError>
    where
        T: ScalarAdd,
    {
        self.try_map(|element| element.try_add(value))
    }

    /// Прибавляет число к каждому элементу
    pub fn add_scalar_assign(&mut self, value: T) -> Result<(), TensorError>
    where
        T: ScalarAdd,
    {
        self.try_map_assign(|element| element.try_add(value))
    }

    /// Записывает суммы элементов с числом в готовый тензор
    pub fn add_scalar_into(&self, value: T, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarAdd,
    {
        self.try_map_into(output, |element| element.try_add(value))
    }

    /// Возвращает новый тензор, вычитая число из каждого элемента
    pub fn sub_scalar(&self, value: T) -> Result<Self, TensorError>
    where
        T: ScalarSub,
    {
        self.try_map(|element| element.try_sub(value))
    }

    /// Вычитает число из каждого элемента
    pub fn sub_scalar_assign(&mut self, value: T) -> Result<(), TensorError>
    where
        T: ScalarSub,
    {
        self.try_map_assign(|element| element.try_sub(value))
    }

    /// Записывает разности элементов и числа в готовый тензор
    pub fn sub_scalar_into(&self, value: T, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarSub,
    {
        self.try_map_into(output, |element| element.try_sub(value))
    }

    /// Возвращает новый тензор, умножая каждый элемент на число
    pub fn mul_scalar(&self, value: T) -> Result<Self, TensorError>
    where
        T: ScalarMul,
    {
        self.try_map(|element| element.try_mul(value))
    }

    /// Умножает каждый элемент на число
    pub fn mul_scalar_assign(&mut self, value: T) -> Result<(), TensorError>
    where
        T: ScalarMul,
    {
        self.try_map_assign(|element| element.try_mul(value))
    }

    /// Записывает произведения элементов и числа в готовый тензор
    pub fn mul_scalar_into(&self, value: T, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarMul,
    {
        self.try_map_into(output, |element| element.try_mul(value))
    }

    /// Возвращает новый тензор, деля каждый элемент на число
    pub fn div_scalar(&self, value: T) -> Result<Self, TensorError>
    where
        T: ScalarDiv,
    {
        self.try_map(|element| element.try_div(value))
    }

    /// Делит каждый элемент на число
    pub fn div_scalar_assign(&mut self, value: T) -> Result<(), TensorError>
    where
        T: ScalarDiv,
    {
        self.try_map_assign(|element| element.try_div(value))
    }

    /// Записывает частные элементов и числа в готовый тензор
    pub fn div_scalar_into(&self, value: T, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarDiv,
    {
        self.try_map_into(output, |element| element.try_div(value))
    }
}
