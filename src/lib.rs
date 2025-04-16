#![feature(allocator_api)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod error;
mod shape;
mod storage;
mod tensor;
mod tensorizable;

pub use tensorizable::Tensorizable;
pub use tensor::Tensor;