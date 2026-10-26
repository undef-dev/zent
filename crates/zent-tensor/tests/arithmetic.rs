// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use zent_tensor::{ArithmeticError, Shape, Strides, Tensor, TensorError};

fn tensor<T>(values: Vec<T>) -> Tensor<T> {
    Tensor::from_vec(Shape::new(vec![values.len()]).unwrap(), values).unwrap()
}

fn state<T: Copy>(value: &Tensor<T>) -> (Shape, Strides, Vec<T>, *const T) {
    (
        value.shape().clone(),
        value.strides().clone(),
        value.as_slice().to_vec(),
        value.as_slice().as_ptr(),
    )
}

#[test]
fn errors() {
    let mut left = tensor(vec![1_i8, 2, 127]);
    let right = tensor(vec![1; 3]);
    let mut output = tensor(vec![50; 3]);
    let original = state(&left);
    let destination = state(&output);
    let error = TensorError::Arithmetic {
        index: 2,
        source: ArithmeticError::Overflow,
    };

    assert_eq!(left.try_add_assign(&right).unwrap_err(), error);
    assert_eq!(left.try_add_into(&right, &mut output).unwrap_err(), error);
    assert_eq!(left.add_scalar_assign(1).unwrap_err(), error);
    assert_eq!(left.add_scalar_into(1, &mut output).unwrap_err(), error);
    assert_eq!(state(&left), original);
    assert_eq!(state(&output), destination);

    assert!(matches!(
        left.div_scalar(0),
        Err(TensorError::Arithmetic {
            index: 0,
            source: ArithmeticError::DivisionByZero,
        })
    ));

    let mut empty = tensor::<i8>(vec![]);
    empty.div_scalar_assign(0).unwrap();
}

#[test]
fn shapes() {
    let left = tensor(vec![i8::MAX; 2]);
    let right = Tensor::from_vec(Shape::new(vec![2, 1]).unwrap(), vec![1_i8; 2]).unwrap();
    let mut output = Tensor::from_vec(Shape::new(vec![1, 2]).unwrap(), vec![7_i8; 2]).unwrap();
    let destination = state(&output);

    assert_eq!(
        left.try_add_into(&right, &mut output),
        Err(TensorError::ShapeMismatch {
            left: left.shape().clone(),
            right: right.shape().clone(),
        })
    );
    assert_eq!(
        left.try_add_into(&left, &mut output),
        Err(TensorError::ShapeMismatch {
            left: left.shape().clone(),
            right: output.shape().clone(),
        })
    );
    assert_eq!(state(&output), destination);
}
