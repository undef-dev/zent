// Zent - https://github.com/undef-dev/zent
// Copyright (c) 2026 Denis Plyasunov and other Zent contributors.
// SPDX-License-Identifier: Apache-2.0

mod shape;
mod storage;
mod strides;
mod tensor;

pub use shape::{Shape, ShapeError};
pub use storage::Storage;
pub use strides::{Strides, StridesError};
pub use tensor::{Tensor, TensorError};
