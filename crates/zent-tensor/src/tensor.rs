// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use crate::Storage;
use crate::shape::Shape;
use crate::strides::{Strides, StridesError};
use std::error::Error;
use std::fmt;

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

    pub fn get(&self, _coords: &[usize]) -> Option<&T> {
        // Для матрицы формы [2, 3]:
        //
        //               столбцы
        //              0   1   2
        //             -----------
        // строка 0:    1   2   3
        // строка 1:    4   5   6
        // ----------------------------------------------
        // Шаги: [3, 1]
        // Координаты [1, 1] -> смещение 1 * 3 + 1 * 1 = 4
        todo!()
    }

    pub fn get_mut(&mut self, _coords: &[usize]) -> Option<&mut T> {
        todo!()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_preserves_values_and_reports_layout() {
        let cases = [
            (vec![], vec![], 1),
            (vec![1], vec![1], 1),
            (vec![1, 1], vec![1, 1], 1),
            (vec![3], vec![1], 3),
            (vec![2, 3], vec![3, 1], 6),
            (vec![2, 3, 4], vec![12, 4, 1], 24),
        ];

        for (dims, steps, numel) in cases {
            let values: Vec<usize> = (0..numel).collect();
            let tensor = Tensor::new(
                Shape::new(dims.clone()).unwrap(),
                Storage::from_vec(values.clone()),
            )
            .unwrap();

            assert_eq!(tensor.shape().dims(), dims);
            assert_eq!(tensor.rank(), dims.len());
            assert_eq!(tensor.strides().as_slice(), steps);
            assert_eq!(tensor.numel(), numel);
            assert_eq!(tensor.as_slice(), values);
            assert!(!tensor.is_empty());
        }
    }

    #[test]
    fn empty_tensor_preserves_shape_and_strides() {
        let cases = [
            (vec![0], vec![1]),
            (vec![0, 3, 4], vec![12, 4, 1]),
            (vec![2, 0, 3], vec![0, 3, 1]),
            (vec![2, 3, 0], vec![0, 0, 1]),
            (vec![usize::MAX, 2, 0], vec![0, 0, 1]),
        ];

        for (dims, steps) in cases {
            let tensor = Tensor::new(
                Shape::new(dims.clone()).unwrap(),
                Storage::<i32>::from_vec(vec![]),
            )
            .unwrap();

            assert_eq!(tensor.shape().dims(), dims);
            assert_eq!(tensor.rank(), dims.len());
            assert_eq!(tensor.strides().as_slice(), steps);
            assert_eq!(tensor.numel(), 0);
            assert!(tensor.is_empty());
            assert!(tensor.as_slice().is_empty());
        }
    }

    #[test]
    fn tensor_rejects_wrong_storage_length() {
        let cases = [
            (vec![2, 3], 5, 6),
            (vec![2, 3], 7, 6),
            (vec![], 0, 1),
            (vec![0], 1, 0),
        ];

        for (dims, actual, expected) in cases {
            let result = Tensor::new(
                Shape::new(dims).unwrap(),
                Storage::from_vec(vec![0_i32; actual]),
            );

            assert_eq!(
                result.err(),
                Some(TensorError::ElementCountMismatch { expected, actual })
            );
        }
    }

    #[test]
    fn tensor_propagates_stride_overflow() {
        let result = Tensor::new(
            Shape::new(vec![0, usize::MAX, 2]).unwrap(),
            Storage::<i32>::from_vec(vec![]),
        );

        assert_eq!(
            result.err(),
            Some(TensorError::Strides(StridesError::StepOverflow))
        );
    }

    #[test]
    fn length_mismatch_precedes_stride_overflow() {
        let result = Tensor::new(
            Shape::new(vec![0, usize::MAX, 2]).unwrap(),
            Storage::from_vec(vec![42]),
        );

        assert_eq!(
            result.err(),
            Some(TensorError::ElementCountMismatch {
                expected: 0,
                actual: 1,
            })
        );
    }

    #[test]
    fn changing_values_preserves_shape_strides_and_length() {
        let mut tensor = Tensor::new(
            Shape::new(vec![2, 2]).unwrap(),
            Storage::from_vec(vec![1, 2, 3, 4]),
        )
        .unwrap();

        tensor.as_mut_slice()[2] = 99;

        assert_eq!(tensor.as_slice(), &[1, 2, 99, 4]);
        assert_eq!(tensor.shape().dims(), &[2, 2]);
        assert_eq!(tensor.strides().as_slice(), &[2, 1]);
        assert_eq!(tensor.numel(), 4);
    }

    #[test]
    fn tensor_accepts_non_clone_elements_and_keeps_the_buffer() {
        struct Value(i32);

        let values = vec![Value(10), Value(20)];
        let pointer = values.as_ptr();
        let tensor = Tensor::new(Shape::new(vec![2]).unwrap(), Storage::from_vec(values)).unwrap();

        assert_eq!(tensor.as_slice().as_ptr(), pointer);
        assert_eq!(tensor.as_slice()[0].0, 10);
        assert_eq!(tensor.as_slice()[1].0, 20);
    }

    #[test]
    fn from_vec_preserves_order_and_layout() {
        let tensor = Tensor::from_vec(
            Shape::new(vec![2, 3]).unwrap(),
            vec![10, 20, 30, 40, 50, 60],
        )
        .unwrap();

        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.strides().as_slice(), &[3, 1]);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.as_slice(), &[10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn from_vec_moves_non_clone_elements_without_copying() {
        struct Item(i32);

        let values = vec![Item(10), Item(20)];
        let pointer = values.as_ptr();

        let tensor = Tensor::from_vec(Shape::new(vec![2]).unwrap(), values).unwrap();

        assert_eq!(tensor.as_slice().as_ptr(), pointer);
        assert_eq!(tensor.as_slice()[0].0, 10);
        assert_eq!(tensor.as_slice()[1].0, 20);
    }

    #[test]
    fn from_vec_rejects_wrong_element_count() {
        let cases = [
            (vec![2, 3], 5, 6),
            (vec![2, 3], 7, 6),
            (vec![], 0, 1),
            (vec![0], 1, 0),
            // Несовпадение длины проверяется раньше переполнения шагов
            (vec![0, usize::MAX, 2], 1, 0),
        ];

        for (dims, actual, expected) in cases {
            let result = Tensor::from_vec(Shape::new(dims).unwrap(), vec![42_i32; actual]);

            assert_eq!(
                result.err(),
                Some(TensorError::ElementCountMismatch { expected, actual })
            );
        }
    }

    #[test]
    fn zeros_and_ones_preserve_shape_and_fill_values() {
        let cases = [
            (vec![2, 3], vec![3, 1], vec![0.0_f32; 6], vec![1.0_f32; 6]),
            (vec![], vec![], vec![0.0], vec![1.0]),
            (vec![1], vec![1], vec![0.0], vec![1.0]),
            (vec![2, 0, 3], vec![0, 3, 1], vec![], vec![]),
            (vec![usize::MAX, 2, 0], vec![0, 0, 1], vec![], vec![]),
        ];

        for (dims, steps, zero_values, one_values) in cases {
            let shape = Shape::new(dims.clone()).unwrap();

            let zeros = Tensor::<f32>::zeros(shape.clone()).unwrap();
            let ones = Tensor::<f32>::ones(shape).unwrap();

            for (tensor, expected) in [(zeros, zero_values), (ones, one_values)] {
                assert_eq!(tensor.shape().dims(), dims.as_slice());
                assert_eq!(tensor.strides().as_slice(), steps.as_slice());
                assert_eq!(tensor.rank(), dims.len());
                assert_eq!(tensor.numel(), expected.len());
                assert_eq!(tensor.is_empty(), expected.is_empty());
                assert_eq!(tensor.as_slice(), expected.as_slice());
            }
        }
    }

    #[test]
    fn convenience_constructors_propagate_stride_overflow() {
        let shape = Shape::new(vec![0, usize::MAX, 2]).unwrap();

        let results = [
            Tensor::<f32>::from_vec(shape.clone(), vec![]),
            Tensor::<f32>::zeros(shape.clone()),
            Tensor::<f32>::ones(shape),
        ];

        for result in results {
            assert_eq!(
                result.err(),
                Some(TensorError::Strides(StridesError::StepOverflow))
            );
        }
    }
}
