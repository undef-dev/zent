// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use crate::Storage;
use crate::scalar::{
    ArithmeticError, Scalar, ScalarAdd, ScalarDiv, ScalarMul, ScalarNeg, ScalarSub,
};
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

/// Поэлементные операции
impl<T: Copy> Tensor<T> {
    /// Возвращает новый тензор с поэлементной суммой
    pub fn try_add(&self, rhs: &Self) -> Result<Self, TensorError>
    where
        T: ScalarAdd,
    {
        let _ = rhs;
        todo!()
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
        let _ = rhs;
        let _ = output;
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
        let _ = rhs;
        let _ = output;
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
        let _ = rhs;
        let _ = output;
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
        let _ = rhs;
        let _ = output;
        todo!()
    }

    /// Возвращает новый тензор, прибавляя число к каждому элементу
    pub fn add_scalar(&self, value: T) -> Result<Self, TensorError>
    where
        T: ScalarAdd,
    {
        let _ = value;
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

    /// Возвращает новый тензор с изменёнными знаками элементов
    pub fn negated(&self) -> Result<Self, TensorError>
    where
        T: ScalarNeg,
    {
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

#[derive(Debug, PartialEq, Eq)]
pub enum TensorError {
    /// Арифметическая ошибка элемента с линейным индексом
    Arithmetic {
        index: usize,
        source: ArithmeticError,
    },
    /// Формы тензоров не совпадают
    ShapeMismatch { left: Shape, right: Shape },
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

impl Error for TensorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Arithmetic { source, .. } => Some(source),
            Self::Strides(error) => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for TensorError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Arithmetic { index, .. } => {
                write!(formatter, "arithmetic error at element {index}")
            }

            Self::ElementCountMismatch { expected, actual } => {
                write!(
                    formatter,
                    "tensor shape requires {expected} elements, but storage contains {actual}"
                )
            }

            Self::Strides(_) => {
                write!(formatter, "failed to compute tensor strides")
            }

            Self::ShapeMismatch { left, right } => {
                write!(
                    formatter,
                    "tensor shapes do not match: {:?} and {:?}",
                    left.dims(),
                    right.dims()
                )
            }
        }
    }
}
