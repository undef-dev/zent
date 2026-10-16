// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use zent_tensor::{Shape, Storage, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Creating a tensor from storage");

    let shape = Shape::new(vec![2, 3])?;
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Storage::from_vec(values);
    let tensor = Tensor::new(shape, storage)?;

    println!("Shape: {:?}", tensor.shape().dims());
    println!("Strides: {:?}", tensor.strides().as_slice());
    println!("Rank: {}", tensor.rank());
    println!("Number of elements: {}", tensor.numel());
    println!("Tensor is empty: {}", tensor.is_empty());
    println!("Values in row-major order: {:?}", tensor.as_slice());

    Ok(())
}
