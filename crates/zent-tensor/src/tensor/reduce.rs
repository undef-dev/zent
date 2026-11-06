// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;

use super::{Tensor, TensorError};
use crate::{ArithmeticError, Scalar, ScalarAdd, Shape, Storage, Strides};

impl<T: Copy> Tensor<T> {
    /// Возвращает сумму всех элементов как скалярный тензор
    pub fn try_sum(&self) -> Result<Self, TensorError>
    where
        T: Scalar + ScalarAdd,
    {
        self.try_reduce(None, false, Some(T::ZERO), ScalarAdd::try_add)
    }

    /// Возвращает суммы по оси, сохраняя её размером 1 при keep_dims
    pub fn try_sum_axis(&self, axis: usize, keep_dims: bool) -> Result<Self, TensorError>
    where
        T: Scalar + ScalarAdd,
    {
        self.try_reduce(Some(axis), keep_dims, Some(T::ZERO), ScalarAdd::try_add)
    }

    /// Возвращает минимум всех элементов как скалярный тензор
    pub fn try_min(&self) -> Result<Self, TensorError>
    where
        T: PartialOrd,
    {
        self.try_reduce(None, false, None, |current, value| {
            Ok(select_extreme(current, value, Ordering::Less))
        })
    }

    /// Возвращает минимумы по оси, сохраняя её размером 1 при keep_dims
    pub fn try_min_axis(&self, axis: usize, keep_dims: bool) -> Result<Self, TensorError>
    where
        T: PartialOrd,
    {
        self.try_reduce(Some(axis), keep_dims, None, |current, value| {
            Ok(select_extreme(current, value, Ordering::Less))
        })
    }

    /// Возвращает максимум всех элементов как скалярный тензор
    pub fn try_max(&self) -> Result<Self, TensorError>
    where
        T: PartialOrd,
    {
        self.try_reduce(None, false, None, |current, value| {
            Ok(select_extreme(current, value, Ordering::Greater))
        })
    }

    /// Возвращает максимумы по оси, сохраняя её размером 1 при keep_dims
    pub fn try_max_axis(&self, axis: usize, keep_dims: bool) -> Result<Self, TensorError>
    where
        T: PartialOrd,
    {
        self.try_reduce(Some(axis), keep_dims, None, |current, value| {
            Ok(select_extreme(current, value, Ordering::Greater))
        })
    }

    /// Проверяет номер оси и возвращает её размер
    fn axis_size(&self, axis: usize) -> Result<usize, TensorError> {
        self.shape()
            .dims()
            .get(axis)
            .copied()
            .ok_or(TensorError::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            })
    }

    /// Собирает по одному результату для каждой группы элементов
    fn try_reduce<F>(
        &self,
        axis: Option<usize>,
        keep_dims: bool,
        initial: Option<T>,
        mut operation: F,
    ) -> Result<Self, TensorError>
    where
        F: FnMut(T, T) -> Result<T, ArithmeticError>,
    {
        let (shape, count, stride) = match axis {
            Some(axis) => {
                let count = self.axis_size(axis)?;
                let mut dims = self.shape().dims().to_vec();

                if keep_dims {
                    dims[axis] = 1;
                } else {
                    dims.remove(axis);
                }

                (Shape::new(dims)?, count, self.strides().as_slice()[axis])
            }
            None => (Shape::new(vec![])?, self.numel(), 1),
        };

        if count == 0 && initial.is_none() {
            return Err(TensorError::EmptyReduction);
        }

        let strides = Strides::from_shape(&shape)?;
        let mut values = Vec::with_capacity(shape.numel());

        // При stride == 0 результат пуст, поэтому деление на stride не выполняется
        (0..shape.numel()).try_for_each(|index| {
            let block = index / stride;
            let offset = index % stride;
            let start = block * count * stride + offset;
            let (first, accumulator) = match initial {
                Some(value) => (0, value),
                None => (1, self.as_slice()[start]),
            };

            let value = (first..count).try_fold(accumulator, |accumulator, position| {
                let input_index = start + position * stride;
                operation(accumulator, self.as_slice()[input_index])
                    .map_err(|source| TensorError::Arithmetic { index, source })
            })?;

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

/// Выбирает крайнее значение и сохраняет встретившийся NaN
fn select_extreme<T: Copy + PartialOrd>(current: T, value: T, desired: Ordering) -> T {
    match value.partial_cmp(&current) {
        Some(ordering) if ordering == desired => value,
        None if current.partial_cmp(&current).is_some() => value,
        _ => current,
    }
}

macro_rules! impl_mean {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl Tensor<$ty> {
                /// Возвращает среднее всех элементов как скалярный тензор
                pub fn try_mean(&self) -> Result<Self, TensorError> {
                    self.mean_over(None, false)
                }

                /// Возвращает средние по оси, сохраняя её размером 1 при keep_dims
                pub fn try_mean_axis(&self, axis: usize, keep_dims: bool) -> Result<Self, TensorError> {
                    self.mean_over(Some(axis), keep_dims)
                }

                /// Делит суммы групп на число элементов каждой группы
                fn mean_over(&self, axis: Option<usize>, keep_dims: bool) -> Result<Self, TensorError> {
                    let count = match axis {
                        Some(axis) => self.axis_size(axis)?,
                        None => self.numel(),
                    };

                    if count == 0 {
                        return Err(TensorError::EmptyReduction);
                    }

                    let mut result = self.try_reduce(axis, keep_dims, Some(0.0), ScalarAdd::try_add)?;
                    let divisor = count as $ty;

                    result.as_mut_slice().iter_mut().for_each(|sum| *sum /= divisor);

                    Ok(result)
                }
            }
        )+
    };
}

impl_mean!(f32, f64);
