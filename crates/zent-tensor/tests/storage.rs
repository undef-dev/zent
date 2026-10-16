// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::Storage;

#[test]
fn buffer() {
    for values in [vec![], vec![10, 20, 30]] {
        let expected = values.clone();
        let pointer = values.as_ptr();
        let mut storage = Storage::from_vec(values);
        assert_eq!(storage.as_slice().as_ptr(), pointer);
        assert_eq!(storage.as_slice(), expected);
        assert_eq!(storage.len(), expected.len());
        assert_eq!(storage.is_empty(), expected.is_empty());
        storage.as_mut_slice().fill(99);
        assert_eq!(storage.as_slice(), vec![99; expected.len()]);
    }
}
