// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::Storage;

fn main() {
    // Данные для будущей матрицы из двух строк и трёх столбцов
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // Передаём владение вектором хранилищу
    let mut storage = Storage::from_vec(values);

    println!("Количество элементов: {}", storage.len());
    println!("Хранилище пустое: {}", storage.is_empty());
    println!("До изменения: {:?}", storage.as_slice());

    {
        let elements = storage.as_mut_slice();

        // Удваиваем каждый элемент непосредственно в хранилище
        for value in elements {
            *value *= 2.0;
        }
    }

    println!("После изменения: {:?}", storage.as_slice());
}
