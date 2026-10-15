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
    fn flat_index_and_get_follow_row_major_order() {
        let cases = [
            (vec![3], vec![0], 0),
            (vec![3], vec![2], 2),
            (vec![2, 3], vec![0, 2], 2),
            (vec![2, 3], vec![1, 0], 3),
            (vec![2, 3], vec![1, 1], 4),
            (vec![2, 3], vec![1, 2], 5),
            (vec![2, 3, 4], vec![0, 2, 3], 11),
            (vec![2, 3, 4], vec![1, 0, 0], 12),
            (vec![2, 3, 4], vec![1, 1, 2], 18),
            (vec![2, 3, 4], vec![1, 2, 3], 23),
            (vec![2, 1, 3], vec![1, 0, 2], 5),
        ];

        for (dims, coords, expected) in cases {
            let shape = Shape::new(dims).unwrap();
            let values: Vec<usize> = (100..100 + shape.numel()).collect();
            let tensor = Tensor::from_vec(shape, values).unwrap();

            assert_eq!(tensor.flat_index(&coords), Some(expected));
            assert_eq!(tensor.get(&coords).copied(), Some(100 + expected));
            assert!(std::ptr::eq(
                tensor.get(&coords).unwrap(),
                &tensor.as_slice()[expected],
            ));
        }
    }

    #[test]
    fn indexing_rejects_wrong_coordinate_count() {
        let mut tensor = Tensor::from_vec(
            Shape::new(vec![2, 3]).unwrap(),
            vec![10, 20, 30, 40, 50, 60],
        )
        .unwrap();

        for coords in [vec![], vec![1], vec![1, 1, 0]] {
            assert_eq!(tensor.flat_index(&coords), None);
            assert_eq!(tensor.get(&coords), None);
            assert!(tensor.get_mut(&coords).is_none());
        }

        assert_eq!(tensor.as_slice(), &[10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn indexing_rejects_out_of_bounds_coordinates() {
        let mut tensor = Tensor::from_vec(
            Shape::new(vec![2, 3]).unwrap(),
            vec![10, 20, 30, 40, 50, 60],
        )
        .unwrap();

        let cases = [[0, 3], [2, 0], [usize::MAX, 0], [0, usize::MAX]];

        for coords in cases {
            assert_eq!(tensor.flat_index(&coords), None);
            assert_eq!(tensor.get(&coords), None);
            assert!(tensor.get_mut(&coords).is_none());
        }

        assert_eq!(tensor.as_slice(), &[10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn scalar_indexing_uses_empty_coordinates() {
        let mut tensor = Tensor::from_vec(Shape::new(vec![]).unwrap(), vec![42]).unwrap();

        assert_eq!(tensor.flat_index(&[]), Some(0));
        assert_eq!(tensor.get(&[]), Some(&42));

        *tensor.get_mut(&[]).unwrap() = 99;

        assert_eq!(tensor.get(&[]), Some(&99));
        assert_eq!(tensor.as_slice(), &[99]);
        assert_eq!(tensor.flat_index(&[0]), None);
        assert_eq!(tensor.get(&[0]), None);
        assert!(tensor.get_mut(&[0]).is_none());
        assert_eq!(tensor.shape().dims(), &[]);
        assert_eq!(tensor.strides().as_slice(), &[]);
        assert_eq!(tensor.rank(), 0);
        assert_eq!(tensor.numel(), 1);
    }

    #[test]
    fn empty_tensors_have_no_valid_coordinates() {
        let cases = [
            vec![0],
            vec![0, 3],
            vec![2, 0, 3],
            vec![2, 3, 0],
            vec![usize::MAX, 2, 0],
        ];

        for dims in cases {
            let coords = vec![0; dims.len()];
            let mut tensor = Tensor::<i32>::from_vec(Shape::new(dims).unwrap(), vec![]).unwrap();

            assert_eq!(tensor.flat_index(&coords), None);
            assert_eq!(tensor.get(&coords), None);
            assert!(tensor.get_mut(&coords).is_none());
            assert_eq!(tensor.numel(), 0);
            assert!(tensor.as_slice().is_empty());
        }
    }

    #[test]
    fn get_mut_changes_only_the_selected_element() {
        let mut tensor = Tensor::from_vec(
            Shape::new(vec![2, 3]).unwrap(),
            vec![10, 20, 30, 40, 50, 60],
        )
        .unwrap();

        *tensor.get_mut(&[1, 1]).unwrap() = 99;

        assert_eq!(tensor.get(&[1, 1]), Some(&99));
        assert_eq!(tensor.as_slice(), &[10, 20, 30, 40, 99, 60]);
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.strides().as_slice(), &[3, 1]);
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    fn indexing_does_not_require_copy_or_clone() {
        struct Value(i32);

        let mut tensor =
            Tensor::from_vec(Shape::new(vec![2]).unwrap(), vec![Value(10), Value(20)]).unwrap();

        assert_eq!(tensor.get(&[1]).unwrap().0, 20);
        tensor.get_mut(&[1]).unwrap().0 = 30;

        assert_eq!(tensor.as_slice()[0].0, 10);
        assert_eq!(tensor.as_slice()[1].0, 30);

        let value: &Value = &tensor[[1]];
        assert_eq!(value.0, 30);
        assert!(std::ptr::eq(value, &tensor.as_slice()[1]));

        let value: &mut Value = &mut tensor[[1]];
        value.0 = 40;
        tensor[[0]] = Value(50);

        assert_eq!(tensor.as_slice()[0].0, 50);
        assert_eq!(tensor.as_slice()[1].0, 40);
    }

    #[test]
    fn bracket_indexing_reads_elements_at_different_ranks() {
        let vector = Tensor::from_vec(Shape::new(vec![3]).unwrap(), vec![10, 20, 30]).unwrap();
        assert_eq!(vector[[0]], 10);
        assert_eq!(vector[[2]], 30);

        let matrix = Tensor::from_vec(
            Shape::new(vec![2, 3]).unwrap(),
            vec![10, 20, 30, 40, 50, 60],
        )
        .unwrap();
        assert_eq!(matrix[[0, 2]], 30);
        assert_eq!(matrix[[1, 0]], 40);
        assert_eq!(matrix[[1, 1]], 50);
        assert_eq!(matrix[[1, 2]], 60);

        let tensor =
            Tensor::from_vec(Shape::new(vec![2, 3, 4]).unwrap(), (0..24).collect()).unwrap();
        assert_eq!(tensor[[0, 2, 3]], 11);
        assert_eq!(tensor[[1, 0, 0]], 12);
        assert_eq!(tensor[[1, 1, 2]], 18);
        assert_eq!(tensor[[1, 2, 3]], 23);
    }

    #[test]
    fn bracket_assignment_changes_only_the_selected_element() {
        let mut tensor = Tensor::from_vec(
            Shape::new(vec![2, 3]).unwrap(),
            vec![10, 20, 30, 40, 50, 60],
        )
        .unwrap();

        tensor[[1, 1]] = 99;
        tensor[[1, 1]] += 1;

        assert_eq!(tensor[[1, 1]], 100);
        assert_eq!(tensor.get(&[1, 1]), Some(&100));
        assert_eq!(tensor.as_slice(), &[10, 20, 30, 40, 100, 60]);
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.strides().as_slice(), &[3, 1]);
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    fn bracket_indexing_reads_and_changes_a_scalar() {
        let mut tensor = Tensor::from_vec(Shape::new(vec![]).unwrap(), vec![42]).unwrap();

        assert_eq!(tensor[[]], 42);
        tensor[[]] = 99;

        assert_eq!(tensor[[]], 99);
        assert_eq!(tensor.as_slice(), &[99]);
        assert_eq!(tensor.shape().dims(), &[]);
        assert_eq!(tensor.strides().as_slice(), &[]);
        assert_eq!(tensor.rank(), 0);
        assert_eq!(tensor.numel(), 1);
    }

    #[test]
    #[should_panic(expected = "invalid tensor coordinates")]
    fn bracket_indexing_panics_on_out_of_bounds_coordinates() {
        let tensor = Tensor::from_vec(Shape::new(vec![2, 3]).unwrap(), (0..6).collect()).unwrap();

        let _value = tensor[[0, 3]];
    }

    #[test]
    #[should_panic(expected = "invalid tensor coordinates")]
    fn bracket_assignment_panics_on_out_of_bounds_coordinates() {
        let mut tensor =
            Tensor::from_vec(Shape::new(vec![2, 3]).unwrap(), (0..6).collect()).unwrap();

        tensor[[0, 3]] = 99;
    }

    #[test]
    #[should_panic(expected = "invalid tensor coordinates")]
    fn bracket_indexing_panics_on_wrong_coordinate_count() {
        let tensor = Tensor::from_vec(Shape::new(vec![2, 3]).unwrap(), (0..6).collect()).unwrap();

        let _value = tensor[[1]];
    }

    #[test]
    #[should_panic(expected = "invalid tensor coordinates")]
    fn bracket_assignment_panics_on_wrong_coordinate_count() {
        let mut tensor =
            Tensor::from_vec(Shape::new(vec![2, 3]).unwrap(), (0..6).collect()).unwrap();

        tensor[[1, 1, 0]] = 99;
    }

    #[test]
    #[should_panic(expected = "invalid tensor coordinates")]
    fn bracket_indexing_panics_on_an_empty_tensor() {
        let tensor = Tensor::<i32>::from_vec(Shape::new(vec![2, 0, 3]).unwrap(), vec![]).unwrap();

        let _value = tensor[[0, 0, 0]];
    }

    #[test]
    #[should_panic(expected = "invalid tensor coordinates")]
    fn bracket_assignment_panics_on_an_empty_tensor() {
        let mut tensor =
            Tensor::<i32>::from_vec(Shape::new(vec![2, 0, 3]).unwrap(), vec![]).unwrap();

        tensor[[0, 0, 0]] = 99;
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
