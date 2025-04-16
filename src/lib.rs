#![feature(allocator_api)]

mod storage;
mod tensor;
mod shape;
mod error;

pub use tensor::*;

pub use tensor::Tensor;