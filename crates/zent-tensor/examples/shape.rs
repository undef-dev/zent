// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::Shape;

fn main() {
    let shape = match Shape::new(vec![2, 3, 4]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    println!("Shape: {:?}", shape.dims());
    println!("Rank: {}", shape.rank());
    println!("Number of elements: {}", shape.numel());

    // let strides = Strides::from_shape(&shape).unwrap();
    // println!("Strides: {:?}", strides.as_slice());
}
