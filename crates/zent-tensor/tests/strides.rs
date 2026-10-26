// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::{Shape, Strides, StridesError};

#[test]
fn bounds() {
    let shape = Shape::new(vec![usize::MAX, 2, 0]).unwrap();
    assert_eq!(Strides::from_shape(&shape).unwrap().as_slice(), &[0, 0, 1]);

    let shape = Shape::new(vec![0, usize::MAX, 2]).unwrap();
    assert_eq!(Strides::from_shape(&shape), Err(StridesError::StepOverflow));
}
