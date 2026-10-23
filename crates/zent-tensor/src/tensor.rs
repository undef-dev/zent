// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

mod arithmetic;
mod error;
mod index;
mod power;
mod reshape;
mod scalar;
mod unary;

use crate::{Scalar, Shape, Storage, Strides};

pub use error::TensorError;

/// Тензор с собственным хранилищем
pub struct Tensor<T> {
    shape: Shape,
    strides: Strides,
    storage: Storage<T>,
}

impl<T> Tensor<T> {
    /// Создаёт тензор из формы и хранилища
    pub fn new(shape: Shape, storage: Storage<T>) -> Result<Self, TensorError> {
        let expected = shape.numel();
        let actual = storage.len();

        if actual != expected {
            return Err(TensorError::ElementCountMismatch { expected, actual });
        }

        let strides = Strides::from_shape(&shape)?;

        Ok(Self {
            shape,
            strides,
            storage,
        })
    }

    /// Создаёт тензор из формы и вектора
    pub fn from_vec(shape: Shape, values: Vec<T>) -> Result<Self, TensorError> {
        Self::new(shape, Storage::from_vec(values))
    }

    /// Форма тензора
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Шаги по осям тензора
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Возвращает число осей тензора
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Возвращает число элементов тензора
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Проверяет, что хранилище тензора не содержит элементов
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Чтение значений тензора
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Изменение значений тензора
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }

    /// Возвращает ошибку, если формы тензоров различаются
    #[expect(dead_code)]
    fn ensure_same_shape<U>(&self, rhs: &Tensor<U>) -> Result<(), TensorError> {
        let _ = rhs;
        todo!()
    }
}

impl<T: Scalar> Tensor<T> {
    /// Создаёт тензор заданной формы, заполненный нулями
    pub fn zeros(shape: Shape) -> Result<Self, TensorError> {
        let values = vec![T::ZERO; shape.numel()];
        Self::new(shape, Storage::from_vec(values))
    }

    /// Создаёт тензор заданной формы, заполненный единицами
    pub fn ones(shape: Shape) -> Result<Self, TensorError> {
        let values = vec![T::ONE; shape.numel()];
        Self::new(shape, Storage::from_vec(values))
    }
}
