// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_quant::int8::{dequantize, quantize};
use zent_quant::{QuantizationError, symmetric_scale};

#[test]
fn scale() {
    assert_eq!(symmetric_scale(&[]), Ok(1.0));
    assert_eq!(symmetric_scale(&[0.0, -0.0]), Ok(1.0));
    assert_eq!(symmetric_scale(&[12.0, -254.0, 0.0, 127.0]), Ok(2.0));

    // Очень маленький положительный масштаб допустим, если он не округлился до нуля
    for value in [f32::MIN_POSITIVE, f32::MAX] {
        let scale = symmetric_scale(&[value]).unwrap();
        assert!(scale.is_finite() && scale > 0.0);
    }
}

#[test]
fn errors() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            symmetric_scale(&[0.0, value]),
            Err(QuantizationError::NonFiniteValue { index: 1 })
        );
    }

    // Ближайшее к нулю положительное значение f32
    let smallest_positive = 0.0_f32.next_up();

    assert_eq!(
        symmetric_scale(&[smallest_positive]),
        Err(QuantizationError::ScaleUnderflow)
    );
}

#[test]
fn conversion() {
    let scale = 2.0;

    // Округление половин от нуля и ограничение кодов с обеих сторон
    for (value, code, restored) in [
        (-300.0, -127, -254.0),
        (-5.0, -3, -6.0),
        (0.0, 0, 0.0),
        (5.0, 3, 6.0),
        (300.0, 127, 254.0),
    ] {
        assert_eq!(quantize(value, scale), Ok(code));
        assert_eq!(dequantize(code, scale), Ok(restored));
    }

    let smallest_positive = 0.0_f32.next_up();
    assert_eq!(quantize(smallest_positive, smallest_positive), Ok(1));
    assert_eq!(dequantize(1, smallest_positive), Ok(smallest_positive));

    // Переполнение деления при допустимом масштабе ограничивается крайним кодом
    assert_eq!(quantize(1.0, smallest_positive), Ok(i8::MAX));
}

#[test]
fn conversion_errors() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(quantize(value, 2.0), Err(QuantizationError::NonFiniteInput));
    }

    assert_eq!(
        dequantize(2, f32::MAX),
        Err(QuantizationError::NonFiniteResult)
    );

    for scale in [
        0.0,
        -0.0,
        -1.0,
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        assert_eq!(
            (quantize(5.0, scale), dequantize(3, scale)),
            (
                Err(QuantizationError::InvalidScale),
                Err(QuantizationError::InvalidScale)
            )
        );
    }
}
