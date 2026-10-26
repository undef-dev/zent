// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use super::{Tensor, TensorError};
use crate::ArithmeticError;

impl<T: Copy> Tensor<T> {
    /// Возвращает новый тензор, применяя операцию к каждому элементу
    pub(super) fn try_map<F>(&self, mut operation: F) -> Result<Self, TensorError>
    where
        F: FnMut(T) -> Result<T, ArithmeticError>,
    {
        let mut values = Vec::with_capacity(self.numel());

        self.as_slice()
            .iter()
            .enumerate()
            .try_for_each(|(index, &value)| {
                values.push(
                    operation(value).map_err(|source| TensorError::Arithmetic { index, source })?,
                );
                Ok::<(), TensorError>(())
            })?;

        Self::from_vec(self.shape.clone(), values)
    }

    /// Меняет элементы после проверки всех вычислений
    pub(super) fn try_map_assign<F>(&mut self, mut operation: F) -> Result<(), TensorError>
    where
        F: FnMut(T) -> Result<T, ArithmeticError>,
    {
        self.as_slice()
            .iter()
            .enumerate()
            .try_for_each(|(index, &value)| {
                operation(value)
                    .map(|_| ())
                    .map_err(|source| TensorError::Arithmetic { index, source })
            })?;

        self.as_mut_slice().iter_mut().for_each(|value| {
            *value = operation(*value).expect("scalar operation failed after validation");
        });

        Ok(())
    }

    /// Записывает результат в готовый тензор после проверки всех вычислений
    pub(super) fn try_map_into<F>(
        &self,
        output: &mut Self,
        mut operation: F,
    ) -> Result<(), TensorError>
    where
        F: FnMut(T) -> Result<T, ArithmeticError>,
    {
        self.ensure_same_shape(output)?;

        self.as_slice()
            .iter()
            .enumerate()
            .try_for_each(|(index, &value)| {
                operation(value)
                    .map(|_| ())
                    .map_err(|source| TensorError::Arithmetic { index, source })
            })?;

        output
            .as_mut_slice()
            .iter_mut()
            .zip(self.as_slice())
            .for_each(|(target, &value)| {
                *target = operation(value).expect("scalar operation failed after validation");
            });

        Ok(())
    }

    /// Возвращает новый тензор, применяя операцию к парам элементов
    pub(super) fn try_zip_map<U: Copy, F>(
        &self,
        rhs: &Tensor<U>,
        mut operation: F,
    ) -> Result<Self, TensorError>
    where
        F: FnMut(T, U) -> Result<T, ArithmeticError>,
    {
        self.ensure_same_shape(rhs)?;

        let mut values = Vec::with_capacity(self.numel());

        self.as_slice()
            .iter()
            .zip(rhs.as_slice())
            .enumerate()
            .try_for_each(|(index, (&left, &right))| {
                values.push(
                    operation(left, right)
                        .map_err(|source| TensorError::Arithmetic { index, source })?,
                );
                Ok::<(), TensorError>(())
            })?;

        Self::from_vec(self.shape.clone(), values)
    }

    /// Меняет элементы по парам после проверки всех вычислений
    pub(super) fn try_zip_map_assign<U: Copy, F>(
        &mut self,
        rhs: &Tensor<U>,
        mut operation: F,
    ) -> Result<(), TensorError>
    where
        F: FnMut(T, U) -> Result<T, ArithmeticError>,
    {
        self.ensure_same_shape(rhs)?;

        self.as_slice()
            .iter()
            .zip(rhs.as_slice())
            .enumerate()
            .try_for_each(|(index, (&left, &right))| {
                operation(left, right)
                    .map(|_| ())
                    .map_err(|source| TensorError::Arithmetic { index, source })
            })?;

        self.as_mut_slice()
            .iter_mut()
            .zip(rhs.as_slice())
            .for_each(|(left, &right)| {
                *left = operation(*left, right).expect("scalar operation failed after validation");
            });

        Ok(())
    }

    /// Записывает результат парных операций после проверки всех вычислений
    pub(super) fn try_zip_map_into<U: Copy, F>(
        &self,
        rhs: &Tensor<U>,
        output: &mut Self,
        mut operation: F,
    ) -> Result<(), TensorError>
    where
        F: FnMut(T, U) -> Result<T, ArithmeticError>,
    {
        self.ensure_same_shape(rhs)?;
        self.ensure_same_shape(output)?;

        self.as_slice()
            .iter()
            .zip(rhs.as_slice())
            .enumerate()
            .try_for_each(|(index, (&left, &right))| {
                operation(left, right)
                    .map(|_| ())
                    .map_err(|source| TensorError::Arithmetic { index, source })
            })?;

        output
            .as_mut_slice()
            .iter_mut()
            .zip(self.as_slice().iter().zip(rhs.as_slice()))
            .for_each(|(target, (&left, &right))| {
                *target = operation(left, right).expect("scalar operation failed after validation");
            });

        Ok(())
    }
}
