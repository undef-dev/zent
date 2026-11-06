// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::{ArithmeticError, Shape, Tensor, TensorError};

#[test]
fn axes() {
    // Два блока, в каждом две строки по два числа
    let input = Tensor::from_vec(
        Shape::new(vec![2, 2, 2]).unwrap(),
        vec![1, 2, 3, 4, 5, 6, 7, 8],
    )
    .unwrap();

    for (result, dims, expected) in [
        (input.try_sum(), vec![], vec![36]),
        (input.try_min(), vec![], vec![1]),
        (input.try_max(), vec![], vec![8]),
        (
            input.try_sum_axis(1, true),
            vec![2, 1, 2],
            vec![4, 6, 12, 14],
        ),
        (input.try_min_axis(0, false), vec![2, 2], vec![1, 2, 3, 4]),
        (input.try_max_axis(2, false), vec![2, 2], vec![2, 4, 6, 8]),
    ] {
        let result = result.unwrap();
        assert_eq!(result.shape().dims(), dims);
        assert_eq!(result.as_slice(), expected);
    }

    let input = Tensor::from_vec(Shape::new(vec![2]).unwrap(), vec![1.0_f32, 2.0]).unwrap();
    let mean = input.try_mean().unwrap();
    assert_eq!(mean.shape().dims(), &[]);
    assert_eq!(mean.as_slice(), [1.5]);

    let input = Tensor::from_vec(
        Shape::new(vec![2, 2]).unwrap(),
        vec![1.0_f64, 3.0, 5.0, 7.0],
    )
    .unwrap();
    let mean = input.try_mean_axis(1, true).unwrap();
    assert_eq!(mean.shape().dims(), &[2, 1]);
    assert_eq!(mean.as_slice(), &[2.0, 6.0]);
}

#[test]
fn bounds() {
    let empty = Tensor::<f32>::zeros(Shape::new(vec![2, 0, 3]).unwrap()).unwrap();
    let sum = empty.try_sum_axis(1, false).unwrap();
    assert_eq!(sum.shape().dims(), &[2, 3]);
    assert_eq!(sum.as_slice(), &[0.0; 6]);
    assert_eq!(empty.try_sum().unwrap().as_slice(), &[0.0]);
    assert!(empty.try_min_axis(0, false).unwrap().is_empty());

    for result in [
        empty.try_mean(),
        empty.try_mean_axis(1, false),
        empty.try_min_axis(1, false),
        empty.try_max(),
    ] {
        assert!(matches!(result, Err(TensorError::EmptyReduction)));
    }
    assert_eq!(
        empty.try_sum_axis(3, false).unwrap_err(),
        TensorError::AxisOutOfBounds { axis: 3, rank: 3 }
    );

    let input =
        Tensor::from_vec(Shape::new(vec![2, 2]).unwrap(), vec![1_i8, 2, i8::MAX, 1]).unwrap();
    assert_eq!(
        input.try_sum_axis(1, false).unwrap_err(),
        TensorError::Arithmetic {
            index: 1,
            source: ArithmeticError::Overflow,
        }
    );

    let input =
        Tensor::from_vec(Shape::new(vec![3]).unwrap(), vec![1.0_f64, f64::NAN, 2.0]).unwrap();
    assert!(input.try_min().unwrap().as_slice()[0].is_nan());
    assert!(input.try_max().unwrap().as_slice()[0].is_nan());
}
