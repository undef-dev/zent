// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use zent_quant::int8::{dequantize_slice, quantize_slice};
use zent_tensor::{Shape, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    // Один объект с тремя признаками и один выход слоя
    let input = Tensor::from_vec(Shape::new(vec![1, 3])?, vec![1.0_f32, 2.0, 3.0])?;
    let weights = Tensor::from_vec(Shape::new(vec![3, 1])?, vec![0.5, 1.0, -0.5])?;
    let bias = 0.5;
    let original = input.try_matmul(&weights)?.as_slice()[0] + bias;

    let (codes, scale) = quantize_slice(weights.as_slice())?;
    let values = dequantize_slice(&codes, scale)?;
    let restored_weights = Tensor::from_vec(weights.shape().clone(), values)?;
    let restored = input.try_matmul(&restored_weights)?.as_slice()[0] + bias;

    // Масштаб общий для всех весов, поэтому умножаем на него всю сумму
    let mut sum = 0.0_f32;
    for (&feature, code) in input.as_slice().iter().zip(codes) {
        sum += feature * f32::from(code);
    }
    let direct = sum * scale + bias;
    let quant_error = (original - restored).abs();
    let method_diff = (direct - restored).abs();

    println!("Original | Restored | From i8  | Quant error | i8 vs restored");
    println!("{original:8.6} | {restored:8.6} | {direct:8.6} | {quant_error:11.6} | {method_diff}");
    Ok(())
}
