// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;
use std::panic::{UnwindSafe, catch_unwind};
use zent_tensor::{Shape, Storage, StridesError, Tensor, TensorError};

const LAYOUTS: &[(&[usize], &[usize], usize)] = &[
    (&[], &[], 1),
    (&[1], &[1], 1),
    (&[1, 1], &[1, 1], 1),
    (&[3], &[1], 3),
    (&[2, 3], &[3, 1], 6),
    (&[2, 3, 4], &[12, 4, 1], 24),
    (&[0], &[1], 0),
    (&[0, 3, 4], &[12, 4, 1], 0),
    (&[2, 0, 3], &[0, 3, 1], 0),
    (&[2, 3, 0], &[0, 0, 1], 0),
    (&[usize::MAX, 2, 0], &[0, 0, 1], 0),
];

fn assert_tensor<T: Debug + PartialEq>(
    tensor: &Tensor<T>,
    dims: &[usize],
    steps: &[usize],
    values: &[T],
) {
    assert_eq!(tensor.shape().dims(), dims);
    assert_eq!(tensor.strides().as_slice(), steps);
    assert_eq!(tensor.rank(), dims.len());
    assert_eq!(tensor.numel(), values.len());
    assert_eq!(tensor.is_empty(), values.is_empty());
    assert_eq!(tensor.as_slice(), values);
}

fn make_tensor(dims: &[usize]) -> Tensor<usize> {
    let shape = Shape::new(dims.to_vec()).unwrap();
    let values = (0..shape.numel()).collect();
    Tensor::from_vec(shape, values).unwrap()
}

#[test]
fn constructors_preserve_values_and_layout() {
    for &(dims, steps, numel) in LAYOUTS {
        let shape = Shape::new(dims.to_vec()).unwrap();
        let values: Vec<usize> = (0..numel).collect();
        for tensor in [
            Tensor::new(shape.clone(), Storage::from_vec(values.clone())).unwrap(),
            Tensor::from_vec(shape, values.clone()).unwrap(),
        ] {
            assert_tensor(&tensor, dims, steps, &values);
        }
    }
}

#[test]
fn constructors_report_errors_in_the_expected_order() {
    for (dims, actual, expected) in [
        (vec![2, 3], 5, 6),
        (vec![2, 3], 7, 6),
        (vec![], 0, 1),
        (vec![0], 1, 0),
        (vec![0, usize::MAX, 2], 1, 0),
    ] {
        let shape = Shape::new(dims).unwrap();
        for result in [
            Tensor::new(shape.clone(), Storage::from_vec(vec![42; actual])),
            Tensor::from_vec(shape, vec![42; actual]),
        ] {
            assert_eq!(
                result.err(),
                Some(TensorError::ElementCountMismatch { expected, actual })
            );
        }
    }

    let shape = Shape::new(vec![0, usize::MAX, 2]).unwrap();
    for result in [
        Tensor::new(shape.clone(), Storage::<f32>::from_vec(vec![])),
        Tensor::from_vec(shape.clone(), vec![]),
        Tensor::zeros(shape.clone()),
        Tensor::ones(shape),
    ] {
        assert_eq!(
            result.err(),
            Some(TensorError::Strides(StridesError::StepOverflow))
        );
    }
}

#[test]
fn zeros_and_ones_fill_values_and_keep_layout() {
    for &(dims, steps, numel) in LAYOUTS {
        let shape = Shape::new(dims.to_vec()).unwrap();
        for (tensor, value) in [
            (Tensor::zeros(shape.clone()).unwrap(), 0.0),
            (Tensor::ones(shape).unwrap(), 1.0),
        ] {
            assert_tensor(&tensor, dims, steps, &vec![value; numel]);
        }
    }
}

#[test]
fn get_and_get_mut_follow_row_major_order() {
    for (dims, coords, offset) in [
        (vec![], vec![], 0),
        (vec![3], vec![0], 0),
        (vec![3], vec![2], 2),
        (vec![2, 3], vec![0, 2], 2),
        (vec![2, 3], vec![1, 0], 3),
        (vec![2, 3], vec![1, 1], 4),
        (vec![2, 3], vec![1, 2], 5),
        (vec![2, 3, 4], vec![0, 2, 3], 11),
        (vec![2, 3, 4], vec![1, 0, 0], 12),
        (vec![2, 3, 4], vec![1, 1, 2], 18),
        (vec![2, 3, 4], vec![1, 2, 3], 23),
        (vec![2, 1, 3], vec![1, 0, 2], 5),
    ] {
        let mut tensor = make_tensor(&dims);
        let steps = tensor.strides().as_slice().to_vec();
        let mut values = tensor.as_slice().to_vec();
        assert_eq!(tensor.get(&coords), Some(&offset));
        assert!(std::ptr::eq(
            tensor.get(&coords).unwrap(),
            &tensor.as_slice()[offset]
        ));
        *tensor.get_mut(&coords).unwrap() = 99;
        values[offset] = 99;
        assert_tensor(&tensor, &dims, &steps, &values);
    }
}

#[test]
fn invalid_coordinates_return_none_and_leave_values_unchanged() {
    for (dims, coords) in [
        (vec![2, 3], vec![]),
        (vec![2, 3], vec![1]),
        (vec![2, 3], vec![1, 1, 0]),
        (vec![2, 3], vec![0, 3]),
        (vec![2, 3], vec![2, 0]),
        (vec![2, 3], vec![usize::MAX, 0]),
        (vec![2, 3], vec![0, usize::MAX]),
        (vec![], vec![0]),
        (vec![0], vec![0]),
        (vec![0, 3], vec![0, 0]),
        (vec![2, 0, 3], vec![0, 0, 0]),
        (vec![2, 3, 0], vec![0, 0, 0]),
        (vec![usize::MAX, 2, 0], vec![0, 0, 0]),
    ] {
        let mut tensor = make_tensor(&dims);
        let steps = tensor.strides().as_slice().to_vec();
        let values = tensor.as_slice().to_vec();
        assert_eq!(tensor.get(&coords), None, "{dims:?}: {coords:?}");
        assert!(tensor.get_mut(&coords).is_none(), "{dims:?}: {coords:?}");
        assert_tensor(&tensor, &dims, &steps, &values);
    }
}

#[test]
fn bracket_indexing_reads_and_changes_values() {
    fn check<const N: usize>(dims: &[usize], coords: [usize; N], offset: usize) {
        let mut tensor = make_tensor(dims);
        let steps = tensor.strides().as_slice().to_vec();
        let mut values = tensor.as_slice().to_vec();
        assert_eq!(tensor[coords], offset);
        tensor[coords] = 99;
        tensor[coords] += 1;
        values[offset] = 100;
        assert_eq!(tensor.get(&coords), Some(&100));
        assert_tensor(&tensor, dims, &steps, &values);
    }

    check(&[], [], 0);
    for (coords, offset) in [([0], 0), ([2], 2)] {
        check(&[3], coords, offset);
    }
    for (coords, offset) in [([0, 2], 2), ([1, 0], 3), ([1, 1], 4), ([1, 2], 5)] {
        check(&[2, 3], coords, offset);
    }
    for (coords, offset) in [
        ([0, 2, 3], 11),
        ([1, 0, 0], 12),
        ([1, 1, 2], 18),
        ([1, 2, 3], 23),
    ] {
        check(&[2, 3, 4], coords, offset);
    }
}

#[test]
fn bracket_indexing_panics_on_invalid_coordinates() {
    fn assert_panics(action: impl FnOnce() + UnwindSafe) {
        let panic = catch_unwind(action).expect_err("invalid coordinates must panic");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied());
        assert_eq!(message, Some("invalid tensor coordinates"));
    }

    fn check<const N: usize>(dims: &[usize], coords: [usize; N]) {
        let tensor = make_tensor(dims);
        assert_panics(move || {
            let _value = tensor[coords];
        });
        let mut tensor = make_tensor(dims);
        assert_panics(move || {
            tensor[coords] = 99;
        });
    }

    check(&[2, 3], [0, 3]);
    check(&[2, 3], [2, 0]);
    check(&[2, 3], [1]);
    check(&[2, 3], [1, 1, 0]);
    check(&[2, 0, 3], [0, 0, 0]);
}

#[test]
fn mutable_slice_preserves_layout() {
    let mut tensor = make_tensor(&[2, 2]);
    tensor.as_mut_slice()[2] = 99;
    assert_tensor(&tensor, &[2, 2], &[2, 1], &[0, 1, 99, 3]);
}

#[test]
fn shape_changes_preserve_values_and_the_buffer() {
    for (dims, new_dims, steps) in [
        (vec![2, 3], vec![3, 2], vec![2, 1]),
        (vec![2, 3], vec![6], vec![1]),
        (vec![6], vec![2, 1, 3], vec![3, 3, 1]),
        (vec![2, 3], vec![2, 3], vec![3, 1]),
        (vec![], vec![1], vec![1]),
        (vec![1], vec![1, 1], vec![1, 1]),
        (vec![1, 1], vec![], vec![]),
        (vec![], vec![], vec![]),
        (vec![0], vec![2, 0, 3], vec![0, 3, 1]),
        (vec![2, 0, 3], vec![0, 6], vec![6, 1]),
        (vec![0, 6], vec![usize::MAX, 2, 0], vec![0, 0, 1]),
        (vec![usize::MAX, 2, 0], vec![0], vec![1]),
    ] {
        for take_ownership in [false, true] {
            let mut tensor = make_tensor(&dims);
            let mut values = tensor.as_slice().to_vec();
            let pointer = tensor.as_slice().as_ptr();
            let shape = Shape::new(new_dims.clone()).unwrap();
            if take_ownership {
                tensor = tensor
                    .into_shape(shape)
                    .unwrap_or_else(|(_, error)| panic!("valid shape rejected: {error}"));
            } else {
                tensor.reshape(shape).unwrap();
            }
            assert_tensor(&tensor, &new_dims, &steps, &values);
            assert_eq!(tensor.as_slice().as_ptr(), pointer);
            if let Some(last) = values.last_mut() {
                let coords: Vec<usize> = new_dims.iter().map(|dim| dim - 1).collect();
                assert_eq!(tensor.get(&coords), Some(&*last));
                *tensor.get_mut(&coords).unwrap() = 99;
                *last = 99;
                assert_tensor(&tensor, &new_dims, &steps, &values);
            } else {
                assert_eq!(tensor.get(&vec![0; new_dims.len()]), None);
            }
        }
    }
}

#[test]
fn failed_shape_changes_preserve_the_tensor_and_allow_retry() {
    for (dims, new_dims, expected_error) in [
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
            vec![0],
            TensorError::ElementCountMismatch {
                expected: 0,
                actual: 1,
            },
        ),
        (
            vec![0],
            vec![],
            TensorError::ElementCountMismatch {
                expected: 1,
                actual: 0,
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
        for take_ownership in [false, true] {
            let mut tensor = make_tensor(&dims);
            let steps = tensor.strides().as_slice().to_vec();
            let values = tensor.as_slice().to_vec();
            let pointer = tensor.as_slice().as_ptr();
            let shape = Shape::new(new_dims.clone()).unwrap();
            let (original, error) = if take_ownership {
                tensor
                    .into_shape(shape)
                    .err()
                    .expect("invalid shape must fail")
            } else {
                let error = tensor.reshape(shape).unwrap_err();
                (tensor, error)
            };
            assert_eq!(error, expected_error);
            assert_tensor(&original, &dims, &steps, &values);
            assert_eq!(original.as_slice().as_ptr(), pointer);

            let recovered = original
                .into_shape(Shape::new(vec![values.len()]).unwrap())
                .unwrap_or_else(|(_, error)| panic!("valid retry failed: {error}"));
            assert_tensor(&recovered, &[values.len()], &[1], &values);
            assert_eq!(recovered.as_slice().as_ptr(), pointer);
        }
    }
}

#[test]
fn constructors_access_and_shape_changes_support_non_clone_elements() {
    #[derive(Debug, PartialEq)]
    struct Value(i32);

    for from_vec in [false, true] {
        let values = vec![Value(10), Value(20)];
        let pointer = values.as_ptr();
        let shape = Shape::new(vec![2]).unwrap();
        let mut tensor = if from_vec {
            Tensor::from_vec(shape, values)
        } else {
            Tensor::new(shape, Storage::from_vec(values))
        }
        .unwrap();
        assert_eq!(tensor.as_slice().as_ptr(), pointer);
        assert_eq!(tensor.get(&[1]), Some(&Value(20)));
        tensor.get_mut(&[1]).unwrap().0 = 30;
        let value: &Value = &tensor[[1]];
        assert!(std::ptr::eq(value, &tensor.as_slice()[1]));
        let value: &mut Value = &mut tensor[[1]];
        value.0 = 40;
        tensor[[0]] = Value(50);
        tensor.reshape(Shape::new(vec![1, 2]).unwrap()).unwrap();
        assert_tensor(&tensor, &[1, 2], &[2, 1], &[Value(50), Value(40)]);
        assert_eq!(tensor.as_slice().as_ptr(), pointer);

        let (original, error) = tensor
            .into_shape(Shape::new(vec![3]).unwrap())
            .err()
            .unwrap();
        assert_eq!(
            error,
            TensorError::ElementCountMismatch {
                expected: 3,
                actual: 2
            }
        );
        assert_tensor(&original, &[1, 2], &[2, 1], &[Value(50), Value(40)]);
        assert_eq!(original.as_slice().as_ptr(), pointer);
        let reshaped = original
            .into_shape(Shape::new(vec![2, 1]).unwrap())
            .unwrap_or_else(|(_, error)| panic!("valid retry failed: {error}"));
        assert_tensor(&reshaped, &[2, 1], &[1, 1], &[Value(50), Value(40)]);
        assert_eq!(reshaped.as_slice().as_ptr(), pointer);
    }
}
