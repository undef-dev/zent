// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::{Shape, Strides, StridesError};

#[test]
fn from_shape() {
    for (dims, steps) in [
        (vec![], vec![]),
        (vec![3], vec![1]),
        (vec![2, 3, 4], vec![12, 4, 1]),
        (vec![2, 1, 3], vec![3, 3, 1]),
        (vec![0, 3, 4], vec![12, 4, 1]),
        (vec![2, 0, 3], vec![0, 3, 1]),
        (vec![1, usize::MAX], vec![usize::MAX, 1]),
        (vec![usize::MAX, 2, 0], vec![0, 0, 1]),
    ] {
        let shape = Shape::new(dims).unwrap();
        assert_eq!(
            Strides::from_shape(&shape).unwrap().as_slice(),
            steps,
            "{shape:?}"
        );
    }
    let shape = Shape::new(vec![0, usize::MAX, 2]).unwrap();
    assert_eq!(Strides::from_shape(&shape), Err(StridesError::StepOverflow));
}
