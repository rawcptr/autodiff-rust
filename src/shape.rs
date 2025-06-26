//! Shape
//!
//! # Broadcasting Semantics
//!
//! This crate follows the broadcasting semantics of `PyTorch` for familiarity.
//!
//! See: [Broadcasting Semantics in `PyTorch`](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
//!
//! Two tensors are broadcast-able if:
//!
//! - Each tensor has non-zero [`Shape::ndims`]
//! - While iterating over dimensions in reverse, the dimension sizes
//!   are either:
//!     - Equal
//!     - 1
//!     - Do not exist
//!
//! If two tensors are broadcast-able, the dimensions of the result
//! from their operation is as follows:
//!
//! 1. If the number of dimensions of tensor A and tensor B are not equal,
//!    prepend 1 to the dimensions of the **shorter** tensor until their ranks match.
//!    so that both of them are of same rank, (i.e `A.shape().ndims() == B.shape.ndims()`)
//! 2. The batch dimensions (all but the last two) are broadcasted according to `PyTorch` semantics.
//! 3. The last two dims follow matrix multiplication rules:
//!    - A: `[..., M, K]`
//!    - B: `[..., K, N]`
//!    - output: `[...broadcasted, M, N]`

use crate::error::TensorError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(Box<[usize]>);

impl Shape {
    pub fn ndims(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn volume(&self) -> usize {
        self.0.iter().product()
    }

    #[must_use]
    pub fn strides(&self) -> Self {
        let (mut strides, _) = self.0.iter().rfold(
            (Vec::with_capacity(self.ndims()), 1usize),
            |(mut vec, acc), &dim| {
                vec.push(acc);
                (vec, acc.saturating_mul(dim))
            },
        );
        strides.reverse();
        Shape(strides.into_boxed_slice())
    }

    /// Returns the linear index from a given N dim index.
    ///
    /// # Panics
    ///
    /// Panics in debug profile if `indices.len() != self.ndims()`
    pub fn linear_index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(indices.len(), self.ndims());

        indices
            .iter()
            .zip(&self.0)
            .try_fold(0, |acc, (dim, i)| (i < dim).then_some(acc * dim + i))
            .expect("invalid indices")
    }

    /// Checks if `Self` can matrix multiply with `other` after broadcasting.
    ///
    /// For general broadcasting semantics, see: [`crate::shape`]
    ///
    /// # Errors
    ///
    /// Returns an error if either self or other are of length 0 or cannot be broadcasted.
    pub fn can_broadcast_matmul(&self, other: &Self) -> Result<Self, TensorError> {
        let (a, b) = (self.dims(), other.dims());

        if a.is_empty() || b.is_empty() {
            return Err(TensorError::InvalidOp(
                "matmul requires at least 1D tensors".to_string(),
            ));
        }

        let a_last = a[a.len().saturating_sub(1)];

        let b_snd_last = if b.len() == 1 { b[0] } else { b[b.len() - 2] };
        if a_last != b_snd_last {
            return Err(TensorError::InvalidOp(format!(
                "cannot matmul\na: {a:?}\nb: {b:?}"
            )));
        }

        let mut output = try_broadcast(
            &a[..a.len().saturating_sub(2)],
            &b[..b.len().saturating_sub(2)],
        )?;

        let m = if a.len() >= 2 { a[a.len() - 2] } else { 1 };
        let n = if b.len() >= 2 { b[b.len() - 1] } else { 1 };

        output.push(m);
        output.push(n);

        Ok(Shape(output.into_boxed_slice()))
    }
}

fn try_broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, TensorError> {
    let max_len = a.len().max(b.len());
    let mut ret = Vec::with_capacity(max_len);

    let dimension = |i: usize, c: &[usize]| *c.get(c.len().wrapping_sub(i + 1)).unwrap_or(&1);

    for i in 0..max_len {
        let (d1, d2) = (dimension(i, a), dimension(i, b));
        match (d1, d2) {
            (1, n) | (n, 1) => ret.push(n),
            (m, n) if m == n => ret.push(m),
            _ => return Err(TensorError::Broadcast { d1, d2 }),
        }
    }

    Ok(ret)
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Self(value.to_vec().into_boxed_slice())
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shape({:?})", &self.0)
    }
}
