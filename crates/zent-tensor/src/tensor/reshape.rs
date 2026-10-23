// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use super::{Tensor, TensorError};
use crate::{Shape, Strides};

impl<T> Tensor<T> {
    /// Меняет форму без копирования элементов, сохраняя тензор при ошибке
    pub fn reshape(&mut self, new_shape: Shape) -> Result<(), TensorError> {
        let expected = new_shape.numel();
        let actual = self.numel();

        if actual != expected {
            return Err(TensorError::ElementCountMismatch { expected, actual });
        }

        self.strides = Strides::from_shape(&new_shape)?;
        self.shape = new_shape;

        Ok(())
    }

    /// Возвращает тензор с новой формой, при ошибке возвращает исходный
    #[allow(clippy::result_large_err)] // Возвращаем исходный тензор без упаковки ошибки в Box
    pub fn into_shape(mut self, new_shape: Shape) -> Result<Self, (Self, TensorError)> {
        match self.reshape(new_shape) {
            Ok(()) => Ok(self),
            Err(error) => Err((self, error)),
        }
    }
}
