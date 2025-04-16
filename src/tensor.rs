use std::ops::{Index, IndexMut};

use crate::{error::TensorError, shape::Shape, storage::Storage, tensorizable::Tensorizable};

pub struct Tensor<T> {
    // we want to _loan_ this tensor out
    storage: Storage<T>,
    shape: Shape,
    requires_grad: bool,
    grad: Option<Storage<T>>,
}

impl<T> Tensor<T> {
    pub fn new(data: impl Tensorizable<T>) -> Result<Self, TensorError> {
        data.to_tensor()
    }
}
