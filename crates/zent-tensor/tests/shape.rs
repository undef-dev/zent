// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::{Shape, ShapeError};

#[test]
fn bounds() {
    assert_eq!(Shape::new(vec![]).unwrap().numel(), 1);
    assert_eq!(Shape::new(vec![usize::MAX, 2, 0]).unwrap().numel(), 0);

    assert_eq!(
        Shape::new(vec![usize::MAX, 2]),
        Err(ShapeError::NumelOverflow)
    );
}
