// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::Storage;

#[test]
fn storage_keeps_values_and_the_buffer() {
    for values in [vec![], vec![10, 20, 30]] {
        let expected = values.clone();
        let pointer = values.as_ptr();
        let storage = Storage::from_vec(values);
        assert_eq!(storage.len(), expected.len());
        assert_eq!(storage.is_empty(), expected.is_empty());
        assert_eq!(storage.as_slice(), expected);
        assert_eq!(storage.as_slice().as_ptr(), pointer);
    }
}

#[test]
fn mutable_slice_changes_values_without_changing_length() {
    let mut storage = Storage::from_vec(vec![10, 20, 30]);
    storage.as_mut_slice()[1] = 99;
    assert_eq!(storage.as_slice(), &[10, 99, 30]);
    assert_eq!(storage.len(), 3);
}
