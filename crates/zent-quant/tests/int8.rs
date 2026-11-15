// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

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
