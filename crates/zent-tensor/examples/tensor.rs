// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::{Shape, Storage, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Матрица из двух строк и трёх столбцов, значения записаны по строкам
    let shape = Shape::new(vec![2, 3])?;
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Storage::from_vec(values);

    let mut tensor = Tensor::new(shape, storage)?;

    println!("Shape: {:?}", tensor.shape().dims());
    println!("Strides: {:?}", tensor.strides().as_slice());
    println!("Rank: {}", tensor.rank());
    println!("Number of elements: {}", tensor.numel());
    println!("Tensor is empty: {}", tensor.is_empty());
    println!("Before modification: {:?}", tensor.as_slice());
    println!("get([1, 1]): {:?}", tensor.get(&[1, 1]));
    println!("tensor[[1, 1]]: {}", tensor[[1, 1]]);
    println!("get([0, 3]): {:?}", tensor.get(&[0, 3]));

    *tensor
        .get_mut(&[1, 1])
        .expect("coordinates [1, 1] are valid for shape [2, 3]") = 50.0;

    println!("After get_mut: {:?}", tensor.as_slice());
    println!("tensor[[1, 1]]: {}", tensor[[1, 1]]);

    tensor[[1, 1]] = 500.0;
    println!("After indexed assignment: {:?}", tensor.as_slice());
    println!("get([1, 1]): {:?}", tensor.get(&[1, 1]));

    let mut scalar = Tensor::from_vec(Shape::new(vec![])?, vec![42.0_f32])?;
    println!("Scalar at []: {:?}", scalar.get(&[]));

    *scalar.get_mut(&[]).expect("a scalar has one element") = 99.0;
    println!("Scalar after modification: {:?}", scalar.get(&[]));
    println!("scalar[[]]: {}", scalar[[]]);

    scalar[[]] = 123.0;
    println!("Scalar after indexed assignment: {}", scalar[[]]);

    Ok(())
}
