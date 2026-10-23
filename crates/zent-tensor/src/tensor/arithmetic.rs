// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use super::{Tensor, TensorError};
use crate::{ArithmeticError, ScalarAdd, ScalarDiv, ScalarMul, ScalarSub};

/// Поэлементные операции
impl<T: Copy> Tensor<T> {
    /// Возвращает новый тензор с поэлементной суммой
    pub fn try_add(&self, rhs: &Self) -> Result<Self, TensorError>
    where
        T: ScalarAdd,
    {
        if self.shape() != rhs.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape().clone(),
                right: rhs.shape().clone(),
            });
        }

        let mut new_storage = Vec::<T>::with_capacity(self.numel());
        self.storage
            .as_slice()
            .iter()
            .zip(rhs.storage.as_slice())
            .enumerate()
            .try_for_each(|(index, (&a, &b))| {
                new_storage.push(
                    a.try_add(b)
                        .map_err(|source| TensorError::Arithmetic { index, source })?,
                );
                Ok::<(), TensorError>(())
            })?;

        Self::from_vec(self.shape.clone(), new_storage)
    }

    /// Поэлементно прибавляет правый тензор к текущему
    pub fn try_add_assign(&mut self, rhs: &Self) -> Result<(), TensorError>
    where
        T: ScalarAdd,
    {
        let _ = rhs;
        todo!()
    }

    /// Записывает поэлементную сумму в готовый тензор
    pub fn try_add_into(&self, rhs: &Self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarAdd,
    {
        let _ = (rhs, output);
        todo!()
    }

    /// Возвращает новый тензор с поэлементной разностью
    pub fn try_sub(&self, rhs: &Self) -> Result<Self, TensorError>
    where
        T: ScalarSub,
    {
        let _ = rhs;
        todo!()
    }

    /// Поэлементно вычитает правый тензор из текущего
    pub fn try_sub_assign(&mut self, rhs: &Self) -> Result<(), TensorError>
    where
        T: ScalarSub,
    {
        let _ = rhs;
        todo!()
    }

    /// Записывает поэлементную разность в готовый тензор
    pub fn try_sub_into(&self, rhs: &Self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarSub,
    {
        let _ = (rhs, output);
        todo!()
    }

    /// Возвращает новый тензор с поэлементным произведением
    pub fn try_mul(&self, rhs: &Self) -> Result<Self, TensorError>
    where
        T: ScalarMul,
    {
        let _ = rhs;
        todo!()
    }

    /// Поэлементно умножает текущий тензор на правый
    pub fn try_mul_assign(&mut self, rhs: &Self) -> Result<(), TensorError>
    where
        T: ScalarMul,
    {
        let _ = rhs;
        todo!()
    }

    /// Записывает поэлементное произведение в готовый тензор
    pub fn try_mul_into(&self, rhs: &Self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarMul,
    {
        let _ = (rhs, output);
        todo!()
    }

    /// Возвращает новый тензор с поэлементным частным
    pub fn try_div(&self, rhs: &Self) -> Result<Self, TensorError>
    where
        T: ScalarDiv,
    {
        let _ = rhs;
        todo!()
    }

    /// Поэлементно делит текущий тензор на правый
    pub fn try_div_assign(&mut self, rhs: &Self) -> Result<(), TensorError>
    where
        T: ScalarDiv,
    {
        let _ = rhs;
        todo!()
    }

    /// Записывает поэлементное частное в готовый тензор
    pub fn try_div_into(&self, rhs: &Self, output: &mut Self) -> Result<(), TensorError>
    where
        T: ScalarDiv,
    {
        let _ = (rhs, output);
        todo!()
    }

    /// Возвращает новый тензор, применяя операцию к парам элементов
    #[expect(dead_code)]
    fn try_zip_map<F>(&self, rhs: &Self, operation: F) -> Result<Self, TensorError>
    where
        F: FnMut(T, T) -> Result<T, ArithmeticError>,
    {
        let _ = (rhs, operation);
        todo!()
    }
}
