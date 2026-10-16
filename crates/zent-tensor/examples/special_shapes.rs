// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use zent_tensor::{Shape, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Scalars and empty tensors");

    let mut scalar = Tensor::from_vec(Shape::new(vec![])?, vec![42])?;
    println!(
        "Scalar rank: {}, elements: {}",
        scalar.rank(),
        scalar.numel()
    );
    println!("Scalar at []: {:?}", scalar.get(&[]));
    scalar[[]] = 99;

    scalar.reshape(Shape::new(vec![1, 1])?)?;
    println!("Scalar as a 1x1 matrix: {}", scalar[[0, 0]]);
    scalar.reshape(Shape::new(vec![])?)?;
    println!("Back to scalar: {}", scalar[[]]);

    let mut empty = Tensor::<f32>::zeros(Shape::new(vec![2, 0, 3])?)?;
    empty.reshape(Shape::new(vec![0, 6])?)?;
    println!("Empty shape: {:?}", empty.shape().dims());
    println!("Empty strides: {:?}", empty.strides().as_slice());
    println!(
        "Empty elements: {}, is_empty: {}",
        empty.numel(),
        empty.is_empty()
    );
    println!("No element at [0, 0]: {:?}", empty.get(&[0, 0]));

    Ok(())
}
