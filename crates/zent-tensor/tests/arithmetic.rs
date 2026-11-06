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
    let right = tensor(vec![1]);
    let mut output = tensor(vec![50; 3]);
    let original = state(&left);
    let destination = state(&output);
    let error = TensorError::Arithmetic {
        index: 2,
        source: ArithmeticError::Overflow,
    };

    for result in [
        left.try_add(&right).map(|_| ()),
        left.try_add_assign(&right),
        left.try_add_into(&right, &mut output),
        left.add_scalar_assign(1),
        left.add_scalar_into(1, &mut output),
    ] {
        assert_eq!(result.unwrap_err(), error);
    }

    assert!(matches!(
        left.div_scalar(0),
        Err(TensorError::Arithmetic {
            index: 0,
            source: ArithmeticError::DivisionByZero,
        })
    ));

    let mut empty = tensor::<i8>(vec![]);
    empty.div_scalar_assign(0).unwrap();

    let row = Tensor::from_vec(Shape::new(vec![1, 3]).unwrap(), vec![1_i8; 3]).unwrap();
    for result in [
        left.try_add_assign(&row),
        left.try_add_into(&row, &mut output),
    ] {
        assert_eq!(
            result,
            Err(TensorError::ShapeMismatch {
                left: row.shape().clone(),
                right: left.shape().clone(),
            })
        );
    }

    let incompatible = tensor(vec![1_i8; 2]);

    assert_eq!(
        left.try_add(&incompatible).unwrap_err(),
        TensorError::BroadcastMismatch {
            left: left.shape().clone(),
            right: incompatible.shape().clone(),
        }
    );
    assert_eq!(state(&left), original);
    assert_eq!(state(&output), destination);
}

#[test]
fn broadcast() {
    let left = Tensor::from_vec(Shape::new(vec![2, 2, 1]).unwrap(), vec![10, 20, 30, 40]).unwrap();
    let right = tensor(vec![1, 2]);
    let shape = Shape::new(vec![2, 2, 2]).unwrap();
    let expected = [9, 8, 19, 18, 29, 28, 39, 38];

    let mut output = Tensor::zeros(shape.clone()).unwrap();
    let output_buffer = output.as_slice().as_ptr();
    let mut assigned =
        Tensor::from_vec(shape.clone(), vec![10, 10, 20, 20, 30, 30, 40, 40]).unwrap();
    let assigned_buffer = assigned.as_slice().as_ptr();

    let result = left.try_sub(&right).unwrap();
    left.try_sub_into(&right, &mut output).unwrap();
    assigned.try_sub_assign(&right).unwrap();

    for value in [&result, &output, &assigned] {
        assert_eq!(value.shape(), &shape);
        assert_eq!(value.as_slice(), expected);
    }
    assert_eq!(output.as_slice().as_ptr(), output_buffer);
    assert_eq!(assigned.as_slice().as_ptr(), assigned_buffer);

    let exponent = Tensor::from_vec(Shape::new(vec![]).unwrap(), vec![2_u8]).unwrap();
    assert_eq!(
        left.try_pow(&exponent).unwrap().as_slice(),
        [100, 400, 900, 1600]
    );

    let base = Tensor::from_vec(Shape::new(vec![]).unwrap(), vec![3_i32]).unwrap();
    let scalar = base.try_pow(&exponent).unwrap();
    assert_eq!(scalar.shape(), base.shape());
    assert_eq!(scalar.as_slice(), [9]);

    let empty = Tensor::<i32>::zeros(Shape::new(vec![2, 0, 3]).unwrap()).unwrap();
    let divisor = Tensor::zeros(Shape::new(vec![1, 3]).unwrap()).unwrap();
    let result = empty.try_div(&divisor).unwrap();
    assert_eq!(result.shape(), empty.shape());
    assert!(result.is_empty());

    let incompatible = Tensor::zeros(Shape::new(vec![2, 2, 3]).unwrap()).unwrap();
    assert!(matches!(
        empty.try_add(&incompatible),
        Err(TensorError::BroadcastMismatch { .. })
    ));
}
