// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;
use std::panic::catch_unwind;
use zent_tensor::{Shape, StridesError, Tensor, TensorError};

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.to_vec()).unwrap()
}

fn make_tensor(dims: &[usize]) -> Tensor<usize> {
    let shape = shape(dims);
    let values = (0..shape.numel()).collect();
    Tensor::from_vec(shape, values).unwrap()
}

fn assert_tensor<T: Debug + PartialEq>(
    tensor: &Tensor<T>,
    dims: &[usize],
    steps: &[usize],
    values: &[T],
) {
    assert_eq!(tensor.shape().dims(), dims);
    assert_eq!(tensor.strides().as_slice(), steps);
    assert_eq!(tensor.as_slice(), values);
}

#[test]
fn constructor_errors() {
    assert_eq!(
        Tensor::from_vec(shape(&[2, 3]), vec![42]).err(),
        Some(TensorError::ElementCountMismatch {
            expected: 6,
            actual: 1,
        })
    );
}

#[test]
fn access() {
    let mut tensor = make_tensor(&[2, 3, 4]);
    tensor[[1, 1, 2]] = 99;
    assert_eq!(tensor.as_slice()[18], 99);
    assert_eq!(make_tensor(&[]).get(&[]), Some(&0));
}

#[test]
fn invalid_access() {
    for (dims, coords) in [
        (vec![2, 3], vec![1]),
        (vec![2, 3], vec![0, 3]),
        (vec![2, 0, 3], vec![0, 0, 0]),
    ] {
        assert_eq!(
            make_tensor(&dims).get(&coords),
            None,
            "{dims:?}: {coords:?}"
        );
    }
    assert!(catch_unwind(|| make_tensor(&[2, 3])[[0, 3]]).is_err());
}

#[test]
fn view() {
    let tensor = Tensor::from_vec(shape(&[2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
    let view = tensor.view();

    assert_eq!(view.get(&[1, 1]), Some(&5));
    assert_eq!(view.get(&[0, 3]), None);
}

#[test]
fn select_axis() {
    let tensor = Tensor::from_vec(shape(&[2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
    let view = tensor.view();

    let row = view.select_axis(0, 1).unwrap();
    assert_eq!(row.get(&[0]), Some(&4));
    assert_eq!(row.get(&[1]), Some(&5));
    assert_eq!(row.get(&[2]), Some(&6));

    let column = view.select_axis(1, 1).unwrap();
    assert_eq!(column.get(&[0]), Some(&2));
    assert_eq!(column.get(&[1]), Some(&5));
}

#[test]
fn as_slice() {
    let tensor = Tensor::from_vec(shape(&[2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
    let view = tensor.view();
    let row = view.select_axis(0, 1).unwrap();
    let column = view.select_axis(1, 1).unwrap();

    assert_eq!(view.as_slice(), Some(tensor.as_slice()));
    assert_eq!(row.as_slice(), Some(&[4, 5, 6][..]));
    assert_eq!(column.as_slice(), None);
}

#[test]
fn reshape() {
    #[derive(Debug, PartialEq)]
    struct Value(usize);

    for (dims, new_dims, steps) in [
        (vec![2, 3], vec![3, 2], vec![2, 1]),
        (vec![1], vec![], vec![]),
        (vec![0], vec![2, 0, 3], vec![0, 3, 1]),
    ] {
        let original_shape = shape(&dims);
        let expected: Vec<_> = (0..original_shape.numel()).map(Value).collect();
        let values: Vec<_> = (0..original_shape.numel()).map(Value).collect();
        let pointer = values.as_ptr();
        let mut tensor = Tensor::from_vec(original_shape.clone(), values).unwrap();
        let original_steps = tensor.strides().as_slice().to_vec();

        tensor.reshape(shape(&new_dims)).unwrap();
        assert_tensor(&tensor, &new_dims, &steps, &expected);
        assert_eq!(tensor.as_slice().as_ptr(), pointer);

        let tensor = tensor
            .into_shape(original_shape)
            .map_err(|(_, error)| error)
            .unwrap();
        assert_tensor(&tensor, &dims, &original_steps, &expected);
        assert_eq!(tensor.as_slice().as_ptr(), pointer);
    }
}

#[test]
fn reshape_errors() {
    for (dims, new_dims, expected) in [
        (
            vec![],
            vec![0, usize::MAX, 2],
            TensorError::ElementCountMismatch {
                expected: 0,
                actual: 1,
            },
        ),
        (
            vec![2, 0, 3],
            vec![0, usize::MAX, 2],
            TensorError::Strides(StridesError::StepOverflow),
        ),
    ] {
        let mut tensor = make_tensor(&dims);
        let steps = tensor.strides().as_slice().to_vec();
        let values = tensor.as_slice().to_vec();
        let pointer = tensor.as_slice().as_ptr();
        let shape = shape(&new_dims);

        assert_eq!(tensor.reshape(shape.clone()).unwrap_err(), expected);
        assert_tensor(&tensor, &dims, &steps, &values);
        assert_eq!(tensor.as_slice().as_ptr(), pointer);

        let (tensor, error) = tensor.into_shape(shape).err().unwrap();
        assert_eq!(error, expected);
        assert_tensor(&tensor, &dims, &steps, &values);
        assert_eq!(tensor.as_slice().as_ptr(), pointer);
    }
}
