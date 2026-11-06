use super::{Tensor, TensorError};
use crate::{Scalar, ScalarAdd, ScalarMul};

impl<T: Copy> Tensor<T> {
    pub fn try_matmul(&self, rhs: &Self) -> Result<Self, TensorError>
    where
        T: Scalar + ScalarMul + ScalarAdd,
    {
        todo!()
    }
}
