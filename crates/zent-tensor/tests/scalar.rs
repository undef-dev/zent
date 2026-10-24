// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::{ArithmeticError::Overflow, ScalarPow};

#[test]
fn integer_pow() {
    for (base, exponent, expected) in [
        (0_i8, 0_u8, Ok(1)),
        (-2, 3, Ok(-8)),
        (i8::MIN, 1, Ok(i8::MIN)),
        (2, 7, Err(Overflow)),
    ] {
        assert_eq!(base.try_pow(exponent), expected);
    }
    assert_eq!(2_u8.try_pow(7_u16), Ok(128));
    assert_eq!(2_u8.try_pow(8_u16), Err(Overflow));
    assert_eq!((-1_i32).try_pow(u128::MAX), Ok(-1));
}

#[test]
fn float_pow() {
    for (base, exponent, expected) in [(2.0_f32, -3_i8, 0.125), (-2.0, -3, -0.125)] {
        assert_eq!(base.try_pow(exponent), Ok(expected));
    }
    assert!((10.0_f32.try_pow(-2_i16).unwrap() - 0.01).abs() < 1e-8);
    assert_eq!((-1.0_f64).try_pow(i128::MIN), Ok(1.0));
    assert_eq!((-1.0_f64).try_pow(u128::MAX), Ok(-1.0));
    assert_eq!(f32::NAN.try_pow(0_u8), Ok(1.0));
    assert_eq!(f64::MAX.try_pow(2_u32), Ok(f64::INFINITY));
    assert_eq!((-0.0_f32).try_pow(-3_i8), Ok(f32::NEG_INFINITY));
    assert_eq!(
        (-0.0_f32).try_pow(3_u8).unwrap().to_bits(),
        (-0.0_f32).to_bits()
    );
    assert!((4.0_f64.try_pow(0.5_f64).unwrap() - 2.0).abs() < 1e-12);
    assert!((-4.0_f32).try_pow(0.5_f32).unwrap().is_nan());
    assert_eq!(0.0_f32.try_pow(0.0_f32), Ok(1.0));
}
