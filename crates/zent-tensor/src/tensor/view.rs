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
    pub fn select_axis(&self, axis: usize, index: usize) -> Option<Self> {
        let _ = axis;
        let _ = index;
        todo!()
    }

    pub fn get(&self, coords: &[usize]) -> Option<&'a T> {
        if coords.len() != self.shape.rank() || self.shape.numel() == 0 {
            return None;
        }

        let index = coords
            .iter()
            .zip(self.shape.dims())
            .zip(self.strides.as_slice())
            .try_fold(self.offset, |index, ((&coord, &dim), &stride)| {
                if coord >= dim {
                    return None;
                }

                index.checked_add(coord.checked_mul(stride)?)
            })?;

        self.data.get(index)
    }

    pub fn as_slice(&self) -> Option<&'a [T]> {
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
