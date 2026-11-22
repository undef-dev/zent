// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use zent_quant::int8::{dequantize, quantize};
use zent_quant::symmetric_scale;

fn main() -> Result<(), Box<dyn Error>> {
    // Максимальный модуль 254 даёт масштаб: 254 / 127 = 2
    let weights = [-254.0_f32, -5.0, 0.0, 5.0, 254.0];
    let scale = symmetric_scale(&weights)?;

    println!("Scale: {scale:.1}");
    println!();
    println!(
        "{:>8} | {:>4} | {:>8} | {:>5}",
        "Weight", "Code", "Restored", "Error"
    );
    println!("---------+------+----------+------");

    for weight in weights {
        let code = quantize(weight, scale)?;
        let restored = dequantize(code, scale)?;

        // Абсолютная ошибка показывает расстояние до исходного веса
        let error = (restored - weight).abs();

        println!("{weight:8.1} | {code:4} | {restored:8.1} | {error:5.1}");
    }

    Ok(())
}
