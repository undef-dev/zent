// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use super::Tensor;
use std::ops::{Index, IndexMut};

impl<T> Tensor<T> {
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
