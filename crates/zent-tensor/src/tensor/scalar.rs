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
        let _ = value;
        todo!()
    }

    /// Прибавляет число к каждому элементу
    pub fn add_scalar_assign(&mut self, value: T) -> Result<(), TensorError>
    where
        T: ScalarAdd,
    {
        let _ = value;
        todo!()
    }

    /// Записывает суммы элементов с числом в готовый тензор
    pub fn add_scalar_into(&self, value: T, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarAdd,
    {
        let _ = (value, output);
        todo!()
    }

    /// Возвращает новый тензор, вычитая число из каждого элемента
    pub fn sub_scalar(&self, value: T) -> Result<Self, TensorError>
    where
        T: ScalarSub,
    {
        let _ = value;
        todo!()
    }

    /// Вычитает число из каждого элемента
    pub fn sub_scalar_assign(&mut self, value: T) -> Result<(), TensorError>
    where
        T: ScalarSub,
    {
        let _ = value;
        todo!()
    }

    /// Записывает разности элементов и числа в готовый тензор
    pub fn sub_scalar_into(&self, value: T, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarSub,
    {
        let _ = (value, output);
        todo!()
    }

    /// Возвращает новый тензор, умножая каждый элемент на число
    pub fn mul_scalar(&self, value: T) -> Result<Self, TensorError>
    where
        T: ScalarMul,
    {
        let _ = value;
        todo!()
    }

    /// Умножает каждый элемент на число
    pub fn mul_scalar_assign(&mut self, value: T) -> Result<(), TensorError>
    where
        T: ScalarMul,
    {
        let _ = value;
        todo!()
    }

    /// Записывает произведения элементов и числа в готовый тензор
    pub fn mul_scalar_into(&self, value: T, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarMul,
    {
        let _ = (value, output);
        todo!()
    }

    /// Возвращает новый тензор, деля каждый элемент на число
    pub fn div_scalar(&self, value: T) -> Result<Self, TensorError>
    where
        T: ScalarDiv,
    {
        let _ = value;
        todo!()
    }

    /// Делит каждый элемент на число
    pub fn div_scalar_assign(&mut self, value: T) -> Result<(), TensorError>
    where
        T: ScalarDiv,
    {
        let _ = value;
        todo!()
    }

    /// Записывает частные элементов и числа в готовый тензор
    pub fn div_scalar_into(&self, value: T, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarDiv,
    {
        let _ = (value, output);
        todo!()
    }
}
