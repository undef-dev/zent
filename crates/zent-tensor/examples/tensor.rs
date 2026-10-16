// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use zent_tensor::{Shape, Storage, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    creation_and_layout()?;
    reading_and_modification()?;
    zeros_and_ones()?;
    reshape_in_place()?;
    reshape_with_ownership()?;
    scalars_and_empty_tensors()?;

    Ok(())
}

fn creation_and_layout() -> Result<(), Box<dyn Error>> {
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

fn reading_and_modification() -> Result<(), Box<dyn Error>> {
    println!("\nReading and modifying elements");

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

fn zeros_and_ones() -> Result<(), Box<dyn Error>> {
    println!("\nFilling tensors with zeros and ones");

    let zeros = Tensor::zeros(Shape::new(vec![2, 3])?)?;
    let ones = Tensor::ones(Shape::new(vec![2, 3])?)?;

    println!("Zeros: {:?}", zeros.as_slice());
    println!("Ones: {:?}", ones.as_slice());

    Ok(())
}

fn reshape_in_place() -> Result<(), Box<dyn Error>> {
    println!("\nChanging shape in place");

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

fn reshape_with_ownership() -> Result<(), Box<dyn Error>> {
    println!("\nChanging shape with ownership and recovering from an error");

    let mut tensor = Tensor::from_vec(Shape::new(vec![2, 3])?, vec![1, 2, 3, 4, 5, 6])?;
    let pointer = tensor.as_slice().as_ptr();

    for dims in [vec![6], vec![2, 2], vec![3, 2]] {
        tensor = match tensor.into_shape(Shape::new(dims)?) {
            Ok(reshaped) => {
                println!("New shape: {:?}", reshaped.shape().dims());
                reshaped
            }
            Err((original, error)) => {
                println!("Rejected shape: {error}");
                println!("Recovered original shape: {:?}", original.shape().dims());
                println!("Recovered values: {:?}", original.as_slice());
                original
            }
        };
    }

    println!("Values after retry: {:?}", tensor.as_slice());
    println!(
        "Buffer unchanged: {}",
        tensor.as_slice().as_ptr() == pointer
    );

    Ok(())
}

fn scalars_and_empty_tensors() -> Result<(), Box<dyn Error>> {
    println!("\nScalars and empty tensors");

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

    let mut empty = Tensor::zeros(Shape::new(vec![2, 0, 3])?)?;
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
