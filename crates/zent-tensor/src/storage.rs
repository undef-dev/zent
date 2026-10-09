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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_preserves_elements() {
        let storage = Storage::from_vec(vec![10, 20, 30]);

        assert_eq!(storage.len(), 3);
        assert!(!storage.is_empty());
        assert_eq!(storage.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn empty_storage_has_no_elements() {
        let storage = Storage::<i32>::from_vec(vec![]);

        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
        assert_eq!(storage.as_slice(), &[]);
    }

    #[test]
    fn mutable_slice_changes_stored_elements() {
        let mut storage = Storage::from_vec(vec![10, 20, 30]);

        {
            let elements = storage.as_mut_slice();
            elements[1] = 99;
        }

        assert_eq!(storage.as_slice(), &[10, 99, 30]);
        assert_eq!(storage.len(), 3);
    }
}
