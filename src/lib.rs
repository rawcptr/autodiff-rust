#![feature(allocator_api)]

mod error;
mod shape;
mod storage;
mod tensor;
mod tensorizable;

pub use tensor::Tensor;