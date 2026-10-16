// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use zent_tensor::{Shape, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Filling tensors with zeros and ones");

    let zeros = Tensor::<f32>::zeros(Shape::new(vec![2, 3])?)?;
    let ones = Tensor::<i32>::ones(Shape::new(vec![2, 3])?)?;

    println!("Zeros: {:?}", zeros.as_slice());
    println!("Ones: {:?}", ones.as_slice());

    Ok(())
}
