// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use zent_tensor::{Shape, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Changing shape in place");

    let mut tensor = Tensor::from_vec(Shape::new(vec![2, 3])?, vec![1, 2, 3, 4, 5, 6])?;
    let pointer = tensor.as_slice().as_ptr();
    println!("Original shape: {:?}", tensor.shape().dims());
    println!("Value at [1, 1]: {}", tensor[[1, 1]]);

    tensor.reshape(Shape::new(vec![3, 2])?)?;

    println!("New shape: {:?}", tensor.shape().dims());
    println!("New strides: {:?}", tensor.strides().as_slice());
    println!("The same value at [2, 0]: {}", tensor[[2, 0]]);
    println!("Values keep their order: {:?}", tensor.as_slice());
    println!(
        "Buffer unchanged: {}",
        tensor.as_slice().as_ptr() == pointer
    );

    let error = tensor
        .reshape(Shape::new(vec![2, 2])?)
        .expect_err("four positions cannot hold six elements");
    println!("Rejected shape: {error}");
    println!("Shape after rejection: {:?}", tensor.shape().dims());
    println!("Values after rejection: {:?}", tensor.as_slice());

    Ok(())
}
