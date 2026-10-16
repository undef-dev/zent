// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use crate::Storage;
use crate::shape::Shape;
use crate::strides::{Strides, StridesError};
use std::error::Error;
use std::fmt;
use std::ops::{Index, IndexMut};

/// Тензор с собственным хранилищем элементов
pub struct Tensor<T> {
    shape: Shape,
    strides: Strides,
    storage: Storage<T>,
}

impl<T> Tensor<T> {
    /// Создаёт тензор из формы и хранилища
    ///
    /// Принимает форму и хранилище во владение без копирования элементов
    /// Проверяет число элементов и вычисляет шаги по осям
    ///
    /// # Errors
    ///
    /// Возвращает [`TensorError`], если:
    /// - число элементов хранилища не соответствует форме
    /// - шаг по оси не помещается в `usize`
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

    /// Возвращает ссылку на элемент по полному набору координат
    pub fn get(&self, coords: &[usize]) -> Option<&T> {
        let index = self.flat_index(coords)?;

        Some(&self.as_slice()[index])
    }

    /// Возвращает изменяемую ссылку на элемент по полному набору координат
    pub fn get_mut(&mut self, coords: &[usize]) -> Option<&mut T> {
        let index = self.flat_index(coords)?;

        Some(&mut self.as_mut_slice()[index])
    }

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
    pub fn into_shape(mut self, new_shape: Shape) -> Result<Self, (Self, TensorError)> {
        match self.reshape(new_shape) {
            Ok(()) => Ok(self),
            Err(error) => Err((self, error)),
        }
    }

    /// Переводит полный набор координат в линейный индекс элемента
    fn flat_index(&self, coords: &[usize]) -> Option<usize> {
        if coords.len() != self.rank() || self.numel() == 0 {
            return None;
        }

        coords
            .iter()
            .zip(self.shape().dims())
            .zip(self.strides().as_slice())
            .try_fold(0_usize, |index, ((&coord, &dim), &stride)| {
                if coord >= dim {
                    return None;
                }

                index.checked_add(coord.checked_mul(stride)?)
            })
    }
}

impl<T, const N: usize> Index<[usize; N]> for Tensor<T> {
    type Output = T;

    /// Возвращает ссылку на элемент, паникует при неверных координатах
    fn index(&self, coords: [usize; N]) -> &Self::Output {
        self.get(&coords).expect("invalid tensor coordinates")
    }
}

impl<T, const N: usize> IndexMut<[usize; N]> for Tensor<T> {
    /// Возвращает изменяемую ссылку на элемент, паникует при неверных координатах
    fn index_mut(&mut self, coords: [usize; N]) -> &mut Self::Output {
        self.get_mut(&coords).expect("invalid tensor coordinates")
    }
}

impl Tensor<f32> {
    /// Создаёт тензор заданной формы, заполненный нулями
    pub fn zeros(shape: Shape) -> Result<Self, TensorError> {
        let values = vec![0.0; shape.numel()];
        Self::new(shape, Storage::from_vec(values))
    }

    /// Создаёт тензор заданной формы, заполненный единицами
    pub fn ones(shape: Shape) -> Result<Self, TensorError> {
        let values = vec![1.0; shape.numel()];
        Self::new(shape, Storage::from_vec(values))
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum TensorError {
    /// Число элементов хранилища не соответствует форме
    ElementCountMismatch { expected: usize, actual: usize },
    /// Не удалось вычислить шаги
    Strides(StridesError),
}

impl From<StridesError> for TensorError {
    fn from(value: StridesError) -> Self {
        Self::Strides(value)
    }
}

impl Error for TensorError {}

impl fmt::Display for TensorError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ElementCountMismatch { expected, actual } => {
                write!(
                    formatter,
                    "tensor shape requires {expected} elements, but storage contains {actual}"
                )
            }

            Self::Strides(error) => {
                write!(formatter, "strides error: {error}")
            }
        }
    }
}
