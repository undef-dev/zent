// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

/// Владеет элементами тензора, расположенными подряд в памяти
pub struct Storage<T> {
    data: Vec<T>,
}

impl<T> Storage<T> {
    /// Принимает вектор во владение без копирования элементов
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Возвращает число элементов
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Проверяет, что хранилище не содержит элементов
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Предоставляет доступ к элементам для чтения
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Предоставляет доступ к элементам для изменения
    ///
    /// Позволяет менять значения, сохраняя число элементов
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}
