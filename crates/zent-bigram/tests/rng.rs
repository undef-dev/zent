// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_bigram::Rng;

#[test]
fn sequence() {
    let mut first = Rng::new(0);
    let mut second = Rng::new(0);

    for _ in 0..32 {
        let value = first.next_f64();
        assert_eq!(value, second.next_f64());
        assert!((0.0..1.0).contains(&value));
    }
}
