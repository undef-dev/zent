// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use zent_tensor::{Shape, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Reading and modifying elements");

    let mut tensor = Tensor::from_vec(Shape::new(vec![2, 3])?, vec![1, 2, 3, 4, 5, 6])?;

    println!("Before modification: {:?}", tensor.as_slice());
    println!("get([1, 1]): {:?}", tensor.get(&[1, 1]));
    println!("tensor[[1, 1]]: {}", tensor[[1, 1]]);
    println!("get([0, 3]): {:?}", tensor.get(&[0, 3]));

    *tensor
        .get_mut(&[1, 1])
        .expect("coordinates [1, 1] are valid for shape [2, 3]") = 50;

    println!("After get_mut: {:?}", tensor.as_slice());
    println!("tensor[[1, 1]]: {}", tensor[[1, 1]]);

    tensor[[1, 1]] = 500;
    println!("After indexed assignment: {:?}", tensor.as_slice());
    println!("get([1, 1]): {:?}", tensor.get(&[1, 1]));

    for value in tensor.as_mut_slice() {
        *value += 1;
    }
    println!(
        "After adding 1 through as_mut_slice: {:?}",
        tensor.as_slice()
    );

    Ok(())
}
