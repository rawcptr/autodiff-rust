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

    pub fn from_raw(
        storage: Storage<T>,
        shape: Shape,
        requires_grad: bool,
        grad: Option<Storage<T>>,
    ) -> Self {
        Self {
            storage,
            shape,
            requires_grad,
            grad,
        }
    }

    pub fn storage(&self) -> &Storage<T> {
        &self.storage
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn grad(&self) -> Option<&Storage<T>> {
        self.grad.as_ref()
    }
}

impl<T, const D: usize> Index<[usize; D]> for Tensor<T> {
    type Output = T;

    fn index(&self, index: [usize; D]) -> &Self::Output {
        // SAFETY:
        // linear index has been validated by shape
        // `as_slice` ensures validity by itself
        unsafe { &self.storage.as_slice()[self.shape.linear_index(index)] }
    }
}

impl<T, const D: usize> IndexMut<[usize; D]> for Tensor<T> {
    fn index_mut(&mut self, index: [usize; D]) -> &mut Self::Output {
        // SAFETY:
        // linear index has been validated by shape
        // `as_mut_slice` ensures validity
        unsafe { &mut self.storage.as_mut_slice()[self.shape.linear_index(index)] }
    }
}
