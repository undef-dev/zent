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
        let _ = coords;
        todo!()
    }

    pub fn as_slice(&self) -> Option<&'a [T]> {
        todo!()
    }
}
