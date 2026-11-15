// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

//! Симметричное восьмибитное квантование

use std::ops::Neg;

use crate::error::QuantizationError;

/// Вычисляет общий масштаб симметричного квантования в i8
pub fn symmetric_scale(values: &[f32]) -> Result<f32, QuantizationError> {
    // 1. Все обычные конечные веса участвуют в поиске максимального модуля
    // 2. NaN и бесконечности дают ошибку
    // 3. Для пустого набора и набора из одних нулей возвращается масштаб 1.0
    // 4. Успешный результат должен быть конечным и положительным

    let max_abs_value = values
        .iter()
        .enumerate()
        .try_fold(0.0_f32, |max, (index, &value)| {
            if value.is_finite() {
                Ok(max.max(value.abs()))
            } else {
                Err(QuantizationError::NonFiniteValue { index })
            }
        })?;

    // Масштаб (размер шага) = максимальный модуль веса / 127
    let scale = max_abs_value / f32::from(i8::MAX);

    match (max_abs_value, scale) {
        (0.0, _) => Ok(1.0),
        (_, 0.0) => Err(QuantizationError::ScaleUnderflow),
        (_, scale) => Ok(scale),
    }
}

/// Переводит значение в код i8 с заданным масштабом
pub fn quantize(value: f32, scale: f32) -> Result<i8, QuantizationError> {
    if !(scale.is_finite() && scale > 0.0) {
        return Err(QuantizationError::InvalidScale);
    } else if !value.is_finite() {
        return Err(QuantizationError::NonFiniteInput);
    }

    // 1. Количество шагов = вес / масштаб
    // 2. Округляем количество шагов
    // 3. Ограничиваем код диапазоном от -127 до 127
    // 4. Переводим округлённый код из f32 в i8
    
    Ok((value / scale)
        .round()
        .clamp(f32::from(i8::MAX).neg(), f32::from(i8::MAX)) as i8)
}

/// Восстанавливает приближённое значение по коду и масштабу
pub fn dequantize(code: i8, scale: f32) -> Result<f32, QuantizationError> {
    if !(scale.is_finite() && scale > 0.0) {
        return Err(QuantizationError::InvalidScale);
    }

    // Восстановленный вес = код * масштаб
    let restored = f32::from(code) * scale;

    // При большом масштабе результат может стать бесконечным
    if !restored.is_finite() {
        return Err(QuantizationError::NonFiniteResult);
    }

    Ok(restored)
}
