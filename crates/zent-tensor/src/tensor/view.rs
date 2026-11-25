use std::ops::Range;

use super::Tensor;
use crate::{Shape, Strides};

/// Представление тензора для чтения
pub struct TensorView<'a, T> {
    data: &'a [T],
    /// Смещение от начала data в элементах
    offset: usize,
    shape: Shape,
    strides: Strides,
}

impl<'a, T> TensorView<'a, T> {
    /// Фиксирует координату `index` на оси `axis` и возвращает представление без этой оси
    pub fn select_axis(&self, axis: usize, index: usize) -> Option<Self> {
        // Если оси нет или index достигает её размера, вернём None
        let axis_size = *self.shape.dims().get(axis)?;
        if index >= axis_size {
            return None;
        }

        // Посчитем новое смещение по старым шагам:
        // новое смещение = текущее смещение + index * шаг выбранной оси
        let new_offset = self
            .offset
            .checked_add(self.strides.as_slice().get(axis)?.checked_mul(index)?)?;

        // Удалим одну позицию из обоих списков
        let mut dims = self.shape.dims().to_vec();
        dims.remove(axis);

        let new_shape = Shape::new(dims).ok()?;
        let new_strides = self.strides.without_axis(axis);

        Some(Self {
            data: self.data, // ссылка на исходный срез
            offset: new_offset,
            shape: new_shape,
            strides: new_strides,
        })
    }

    /// Возвращает ссылку на элемент представления по координатам
    pub fn get(&self, coords: &[usize]) -> Option<&'a T> {
        // При неверном числе координат или пустом представлении возвращаем None
        if coords.len() != self.shape.rank() || self.shape.numel() == 0 {
            return None;
        }

        // Для каждой оси прибавляем к текущему смещению координату, умноженную на шаг этой оси
        let index = coords
            .iter()
            .zip(self.shape.dims())
            .zip(self.strides.as_slice())
            .try_fold(self.offset, |offset, ((&coord, &dim), &stride)| {
                if coord >= dim {
                    return None;
                }

                offset.checked_add(coord.checked_mul(stride)?)
            })?;

        self.data.get(index)
    }

    /// Возвращает срез, если элементы представления лежат подряд в порядке обхода
    pub fn as_slice(&self) -> Option<&'a [T]> {
        // У пустого представления смещение может оказаться за концом буфера
        if self.shape.numel() == 0 {
            return Some(&[]);
        }

        // Начинаем с последней оси и шага 1 между соседними элементами среза
        let len = self
            .shape
            .dims()
            .iter()
            .zip(self.strides.as_slice())
            .rev()
            .try_fold(1_usize, |expected_stride, (&dim, &stride)| {
                // У оси длины 1 единственная координата, поэтому подходит любой шаг
                if dim > 1 && stride != expected_stride {
                    return None;
                }

                // Длина блока по проверенным осям задаёт шаг для следующей оси
                expected_stride.checked_mul(dim)
            })?;

        // Смещение + число элементов указывает на позицию сразу за срезом
        let end = self.offset.checked_add(len)?;
        self.data.get(self.offset..end)
    }

    pub fn slice_axis(&self, axis: usize, range: Range<usize>, step: usize) -> Option<Self> {
        // 1. Проверить ось и границы
        // 2. Заменить один размер в shape
        // 3. Вычислить начало и шаг
        // 4. Собрать TensorView

        let _ = step;
        let _ = range;
        let _ = axis;
        todo!()
    }
}

impl<T> Tensor<T> {
    pub fn view(&self) -> TensorView<'_, T> {
        TensorView::<'_, T> {
            data: self.as_slice(), // Срез всех значений
            offset: 0_usize,       // индекс, с которого начинается представление
            shape: self.shape().clone(),
            strides: self.strides().clone(),
        }
    }
}
