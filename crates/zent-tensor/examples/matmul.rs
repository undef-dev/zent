// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use zent_tensor::{Shape, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    // Два объекта, у каждого три входных признака
    let inputs = Tensor::from_vec(
        Shape::new(vec![2, 3])?,
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )?;

    // Каждый столбец содержит три веса для одного из двух выходов
    let weights = Tensor::from_vec(
        Shape::new(vec![3, 2])?,
        vec![0.5, -1.0, 1.0, 0.0, -0.5, 1.0],
    )?;
    let bias = Tensor::from_vec(Shape::new(vec![2])?, vec![0.5, -1.0])?;

    // Одни и те же смещения прибавляются к выходам каждого объекта
    let output = inputs.try_matmul(&weights)?.try_add(&bias)?;

    println!("Output:");
    output
        .as_slice()
        .chunks(2)
        .enumerate()
        .for_each(|(index, row)| println!("Sample {}: {row:?}", index + 1));

    Ok(())
}
