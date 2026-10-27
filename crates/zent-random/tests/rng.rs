// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_random::Rng;

#[test]
fn sequence() {
    let mut integers = Rng::new(0);
    let mut floats32 = Rng::new(0);
    let mut floats64 = Rng::new(0);

    for _ in 0..32 {
        assert!((0.0..1.0).contains(&floats32.next_f32()));
        assert!((0.0..1.0).contains(&floats64.next_f64()));

        integers.next_u64();
        let next = integers.next_u64();
        assert_eq!(floats32.next_u64(), next);
        assert_eq!(floats64.next_u64(), next);
    }
}
