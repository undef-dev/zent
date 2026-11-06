// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::{ArithmeticError, Shape, ShapeError, Tensor, TensorError};

#[test]
fn product() {
    let left = Tensor::from_vec(Shape::new(vec![2, 3]).unwrap(), vec![1, 2, 3, 4, 5, 6]).unwrap();
    let right =
        Tensor::from_vec(Shape::new(vec![3, 2]).unwrap(), vec![10, 1, 20, 1, 30, 1]).unwrap();

    let result = left.try_matmul(&right).unwrap();
    assert_eq!(result.shape().dims(), &[2, 2]);
    assert_eq!(result.as_slice(), &[140, 6, 320, 15]);

    // Нулевая длина суммы даёт нули, отсутствие строк или столбцов даёт пустой результат
    for (left_dims, right_dims, dims, expected) in [
        ([2, 0], [0, 3], [2, 3], vec![0; 6]),
        ([0, 3], [3, 2], [0, 2], vec![]),
        ([2, 3], [3, 0], [2, 0], vec![]),
    ] {
        let left = Tensor::<i32>::zeros(Shape::new(left_dims.to_vec()).unwrap()).unwrap();
        let right = Tensor::zeros(Shape::new(right_dims.to_vec()).unwrap()).unwrap();

        let result = left.try_matmul(&right).unwrap();
        assert_eq!(result.shape().dims(), dims);
        assert_eq!(result.as_slice(), expected);
    }
}

#[test]
fn bounds() {
    let input = Tensor::<i32>::zeros(Shape::new(vec![2, 3]).unwrap()).unwrap();

    // Неверное число осей и разные внутренние размеры проверяем для обоих входов
    for dims in [vec![3], vec![4, 4]] {
        let invalid = Tensor::zeros(Shape::new(dims).unwrap()).unwrap();

        for (left, right) in [(&input, &invalid), (&invalid, &input)] {
            assert_eq!(
                left.try_matmul(right).unwrap_err(),
                TensorError::MatmulMismatch {
                    left: left.shape().clone(),
                    right: right.shape().clone(),
                }
            );
        }
    }

    let left =
        Tensor::from_vec(Shape::new(vec![2, 2]).unwrap(), vec![1_i8, 0, i8::MAX, 1]).unwrap();

    // Во второй строке переполняется сумма либо отдельное произведение
    for values in [vec![1, 1], vec![2, 0]] {
        let right = Tensor::from_vec(Shape::new(vec![2, 1]).unwrap(), values).unwrap();

        assert_eq!(
            left.try_matmul(&right).unwrap_err(),
            TensorError::Arithmetic {
                index: 1,
                source: ArithmeticError::Overflow,
            }
        );
    }

    // Входы пусты, но число элементов результата не помещается в usize
    let left = Tensor::<i32>::zeros(Shape::new(vec![usize::MAX, 0]).unwrap()).unwrap();
    let right = Tensor::zeros(Shape::new(vec![0, 2]).unwrap()).unwrap();
    assert_eq!(
        left.try_matmul(&right).unwrap_err(),
        TensorError::Shape(ShapeError::NumelOverflow)
    );
}
