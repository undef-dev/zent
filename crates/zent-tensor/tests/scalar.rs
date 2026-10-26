// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::{ArithmeticError::Overflow, ScalarPow};

#[test]
fn integer_pow() {
    for (base, exponent, expected) in [
        (0_i8, 0_u8, Ok(1)),
        (i8::MIN, 1, Ok(i8::MIN)),
        (2, 7, Err(Overflow)),
    ] {
        assert_eq!(base.try_pow(exponent), expected);
    }
    assert_eq!(0_i32.try_pow(1_u64 << 32), Ok(0));
}

#[test]
fn float_pow() {
    assert_eq!((-1.0_f64).try_pow(i128::MIN), Ok(1.0));
    assert_eq!((-1.0_f64).try_pow(u128::MAX), Ok(-1.0));
    assert_eq!(f32::NAN.try_pow(0_u8), Ok(1.0));
    assert_eq!((-0.0_f32).try_pow(-3_i8), Ok(f32::NEG_INFINITY));
}
