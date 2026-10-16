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
    for (dims, expected) in [(&[2, 3][..], 6), (&[0, usize::MAX, 2][..], 0)] {
        assert_eq!(
            Tensor::from_vec(shape(dims), vec![42]).err(),
            Some(TensorError::ElementCountMismatch {
                expected,
                actual: 1
            })
        );
    }
    assert_eq!(
        Tensor::<i32>::from_vec(shape(&[0, usize::MAX, 2]), vec![]).err(),
        Some(TensorError::Strides(StridesError::StepOverflow))
    );
}

#[test]
fn fill() {
    let zeros = Tensor::<i32>::zeros(shape(&[2, 3])).unwrap();
    let ones = Tensor::<f32>::ones(shape(&[2, 3])).unwrap();
    assert_eq!(zeros.as_slice(), &[0; 6]);
    assert_eq!(ones.as_slice(), &[1.0; 6]);

    let scalar = Tensor::<u8>::ones(shape(&[])).unwrap();
    assert_eq!(scalar.as_slice(), &[1]);
    let empty = Tensor::<f64>::zeros(shape(&[2, 0, 3])).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.shape().dims(), &[2, 0, 3]);
}

#[test]
fn access() {
    let mut tensor = make_tensor(&[2, 3, 4]);
    assert_eq!(tensor.get(&[1, 1, 2]), Some(&18));
    *tensor.get_mut(&[1, 1, 2]).unwrap() = 99;
    assert_eq!(tensor.as_slice()[18], 99);
    tensor.as_mut_slice()[18] = 100;
    assert_eq!(tensor[[1, 1, 2]], 100);
    tensor[[1, 1, 2]] = 200;
    assert_eq!(tensor.get(&[1, 1, 2]), Some(&200));
    assert_eq!(make_tensor(&[]).get(&[]), Some(&0));
}

#[test]
fn invalid_access() {
    for (dims, coords) in [
        (vec![2, 3], vec![1]),
        (vec![2, 3], vec![1, 1, 0]),
        (vec![2, 3], vec![0, 3]),
        (vec![2, 3], vec![2, 0]),
        (vec![2, 3], vec![usize::MAX, 0]),
        (vec![2, 0, 3], vec![0, 0, 0]),
    ] {
        let mut tensor = make_tensor(&dims);
        assert_eq!(tensor.get(&coords), None, "{dims:?}: {coords:?}");
        assert!(tensor.get_mut(&coords).is_none(), "{dims:?}: {coords:?}");
    }
    assert!(catch_unwind(|| make_tensor(&[2, 3])[[0, 3]]).is_err());
    assert!(catch_unwind(|| make_tensor(&[2, 3])[[1]] = 99).is_err());
}

#[test]
fn reshape() {
    #[derive(Debug, PartialEq)]
    struct Value(usize);

    for (dims, new_dims, steps) in [
        (vec![2, 3], vec![3, 2], vec![2, 1]),
        (vec![6], vec![2, 1, 3], vec![3, 3, 1]),
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
            vec![2, 3],
            vec![2, 2],
            TensorError::ElementCountMismatch {
                expected: 4,
                actual: 6,
            },
        ),
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
