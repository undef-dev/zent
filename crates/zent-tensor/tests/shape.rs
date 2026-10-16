// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::{Shape, ShapeError};

#[test]
fn dimensions_rank_and_element_count() {
    for (dims, numel) in [
        (vec![], 1),
        (vec![1], 1),
        (vec![2, 3, 4], 24),
        (vec![2, 0, 3], 0),
        (vec![usize::MAX], usize::MAX),
        (vec![usize::MAX, 2, 0], 0),
        (vec![0, usize::MAX, 2], 0),
    ] {
        let shape = Shape::new(dims.clone()).unwrap();
        assert_eq!(shape.dims(), dims);
        assert_eq!(shape.rank(), dims.len());
        assert_eq!(shape.numel(), numel, "{dims:?}");
    }
}

#[test]
fn element_count_overflow() {
    assert_eq!(
        Shape::new(vec![usize::MAX, 2]),
        Err(ShapeError::NumelOverflow)
    );
}
