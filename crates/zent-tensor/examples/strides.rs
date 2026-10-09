// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::{Shape, Strides};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape::new(vec![2, 3, 4])?;
    let strides = Strides::from_shape(&shape)?;

    println!("Shape: {:?}", shape.dims());
    println!("Strides: {:?}", strides.as_slice()); // [12, 4, 1]

    Ok(())
}
