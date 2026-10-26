// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use zent_tensor::{Shape, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    let shape = Shape::new(vec![2, 2])?;
    let mut left = Tensor::from_vec(shape.clone(), vec![1_i32, 2, 3, 4])?;
    let right = Tensor::from_vec(shape.clone(), vec![10, 20, 30, 40])?;
    let mut output = Tensor::zeros(shape)?;
    let left_pointer = left.as_slice().as_ptr();
    let output_pointer = output.as_slice().as_ptr();

    let sum = left.try_add(&right)?;
    left.try_add_into(&right, &mut output)?;
    left.try_add_assign(&right)?;

    println!("New tensor: {:?}", sum.as_slice());
    println!("Existing output: {:?}", output.as_slice());
    println!("Changed input: {:?}", left.as_slice());
    println!(
        "Input buffer reused: {}",
        left.as_slice().as_ptr() == left_pointer
    );
    println!(
        "Output buffer reused: {}",
        output.as_slice().as_ptr() == output_pointer
    );

    Ok(())
}
