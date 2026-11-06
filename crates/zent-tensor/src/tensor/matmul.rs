// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use super::{Tensor, TensorError};
use crate::{Scalar, ScalarAdd, ScalarMul, Shape, Storage, Strides};

impl<T: Copy> Tensor<T> {
    /// Возвращает новый тензор с произведением двух матриц
    pub fn try_matmul(&self, rhs: &Self) -> Result<Self, TensorError>
    where
        T: Scalar + ScalarMul + ScalarAdd,
    {
        let (rows, inner, cols) = match (self.shape().dims(), rhs.shape().dims()) {
            ([rows, left_cols], [right_rows, cols]) if left_cols == right_rows => {
                (*rows, *left_cols, *cols)
            }
            _ => {
                return Err(TensorError::MatmulMismatch {
                    left: self.shape().clone(),
                    right: rhs.shape().clone(),
                });
            }
        };

        let shape = Shape::new(vec![rows, cols])?;
        let strides = Strides::from_shape(&shape)?;
        let mut values = Vec::with_capacity(shape.numel());

        // При cols == 0 результат пуст, поэтому деление на cols не выполняется
        (0..shape.numel()).try_for_each(|index| {
            let row = index / cols;
            let col = index % cols;

            let value = (0..inner)
                .try_fold(T::ZERO, |accumulator, position| {
                    // Строку левой матрицы читаем подряд, столбец правой с шагом cols
                    let left_index = row * inner + position;
                    let right_index = position * cols + col;
                    let product =
                        self.as_slice()[left_index].try_mul(rhs.as_slice()[right_index])?;

                    accumulator.try_add(product)
                })
                .map_err(|source| TensorError::Arithmetic { index, source })?;

            values.push(value);
            Ok::<(), TensorError>(())
        })?;

        Ok(Self {
            shape,
            strides,
            storage: Storage::from_vec(values),
        })
    }
}
