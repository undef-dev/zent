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

    println!("Форма: {:?}", tensor.shape().dims());
    println!("Шаги по осям: {:?}", tensor.strides().as_slice());
    println!("Количество осей: {}", tensor.rank());
    println!("Количество элементов: {}", tensor.numel());
    println!("Тензор пустой: {}", tensor.is_empty());
    println!("До изменения: {:?}", tensor.as_slice());

    tensor.as_mut_slice()[4] = 50.0;

    println!("После изменения: {:?}", tensor.as_slice());

    Ok(())
}
