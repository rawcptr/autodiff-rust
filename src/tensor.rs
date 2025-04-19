//! Defines the main `Tensor` struct and its core functionalities.

use std::ops::{Index, IndexMut};

use crate::{error::TensorError, shape::Shape, storage::Storage, tensorizable::Tensorizable};

/// A multi-dimensional array (tensor) with support for automatic differentiation.
///
/// Tensors store their data in a contiguous, aligned memory block (`Storage`)
/// and keep track of their shape (`Shape`). They can optionally track gradients.
#[derive(Debug)]
pub struct Tensor<T> {
    /// Raw, aligned storage for the tensor's elements.
    storage: Storage<T>,
    /// Describes the dimensions and layout of the tensor data.
    shape: Shape,
    /// Flag indicating whether gradient calculation is required for this tensor.
    requires_grad: bool,
    /// Stores the gradient of this tensor, if calculated. Uses the same storage type.
    grad: Option<Storage<T>>,
}

impl<T> Tensor<T> {
    /// Creates a new Tensor backed with a [`Storage`] from a collection
    /// that implements [`Tensorizable`]
    ///
    /// This is the primary way to create tensors from Rust collections like `Vec`, arrays, etc.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError`] if the input data cannot be converted into a valid tensor
    /// (e.g., inconsistent dimensions in nested vectors).
    pub fn new(data: impl Tensorizable<T>) -> Result<Self, TensorError> {
        data.to_tensor()
    }

    /// Creates a `Tensor` directly from its constituent parts.
    ///
    /// This is primarily used internally or where `Storage` and `Shape`
    /// are managed manually. The caller is responsible for ensuring that the `storage`
    /// contains [`Storage::len`] initialized elements and that the `grad` storage (if provided)
    /// matches the shape and contains initialized elements.
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

    #[inline]
    /// Returns an immutable reference to the underlying [`Storage`].
    pub fn storage(&self) -> &Storage<T> {
        &self.storage
    }

    #[inline]
    /// Returns an immutable reference to the tensor's [`Shape`].
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns `true` if this tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    #[inline]
    /// Returns an optional immutable reference to the gradient's [`Storage`].
    /// Returns `None` if gradients are not required or haven't been computed yet.
    pub fn grad(&self) -> Option<&Storage<T>> {
        self.grad.as_ref()
    }

    #[inline]
    /// Returns the total number of elements in the tensor.
    /// Equivalent to the product of its dimensions.
    pub fn len(&self) -> usize {
        self.storage().len()
    }

    #[inline]
    /// Directly reads an element from the underlying storage at the given linear `index`.
    ///
    /// This bypasses shape calculations and accesses the raw buffer.
    /// **Note:** Prefer using the `Index` trait (`tensor[[d0, d1, ...]]`) for safer,
    /// dimension-aware access.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds (`index >= self.len()`).
    pub(crate) fn direct_index(&self, index: usize) -> &T {
        self.storage().direct_read(index)
    }

    #[inline]
    /// Directly writes a `val` to the underlying storage at the given linear `index`.
    ///
    /// This bypasses shape calculations and accesses the raw buffer.
    /// **Note:** Use with extreme caution. Prefer `IndexMut` for safer access.
    ///
    /// # Safety
    /// The caller *must* ensure that `index` is within the bounds `[0, self.len())`.
    /// Writing out of bounds results in undefined behavior.
    pub(crate) unsafe fn direct_write(&mut self, index: usize, val: T) {
        unsafe {
            self.storage.direct_write(index, val);
        }
    }
}

impl<T, const D: usize> Index<[usize; D]> for Tensor<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; D]) -> &Self::Output {
        // SAFETY:
        // - `self.shape.linear_index(index)` computes the offset based on the shape's dimensions
        //   and panics if `index` is out of bounds for any dimension.
        // - `self.storage` is guaranteed to be allocated with at least `self.len()` elements.
        // - The tensor's elements `0..self.len()` are guaranteed to be initialized upon creation
        //   (via `Tensorizable` or `from_raw`'s contract).
        self.storage.direct_read(self.shape.linear_index(index))
    }
}

impl<T, const D: usize> IndexMut<[usize; D]> for Tensor<T> {
    fn index_mut(&mut self, index: [usize; D]) -> &mut Self::Output {
        let linear_index = self.shape.linear_index(index);
        // SAFETY:
        // - `self.shape.linear_index(index)` computes the offset based on the shape's dimensions
        //   and panics if `index` is out of bounds for any dimension, ensuring `linear_index < self.len()`.
        // - `self.storage.as_mut_ptr()` returns a valid, aligned pointer to initialized memory of `self.len()` elements.
        // - The pointer offset by `linear_index` points to a valid, initialized element within the allocation.
        unsafe { &mut *self.storage.as_mut_ptr().add(linear_index) }
    }
}
