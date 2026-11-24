// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use zent_tensor::{Shape, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    let matrix = Tensor::from_vec(Shape::new(vec![2, 3])?, vec![1, 2, 3, 4, 5, 6])?;
    let view = matrix.view();

    // Вторая строка занимает три соседние позиции в буфере
    let row = view.select_axis(0, 1).ok_or("missing second row")?;
    println!("row.as_slice() = {:?}", row.as_slice());

    // Элементы второго столбца лежат на позициях 1 и 4
    let column = view.select_axis(1, 1).ok_or("missing second column")?;
    let first = column.get(&[0]).ok_or("missing first element")?;
    let second = column.get(&[1]).ok_or("missing second element")?;

    println!("column.as_slice() = {:?}", column.as_slice());
    println!("column via get = [{first}, {second}]");

    Ok(())
}
