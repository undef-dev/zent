// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use zent_tensor::{Shape, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Changing shape with ownership and recovering from an error");

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
