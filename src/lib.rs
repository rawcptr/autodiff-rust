//! # autodiff
//!
//! A minimal reverse-mode automatic differentiation (autodiff) engine in Rust.
//!
//! ## Features
//!
//! - Designed for direct manipulation and understanding of the underlying mechanisms.
//! - Uses 32-byte aligned memory allocation suitable for AVX2/SIMD optimizations.
//! - Provides a basic `Tensor` type with shape tracking.
//! - Keeps external dependencies to a minimum.
//!
//! ## Motivation
//!
//! This crate serves as a learning project to:
//! - Understand backpropagation from first principles.
//! - Build an ergonomic tensor abstraction over raw pointers.
//! - Experiment with graph-based autodiff and memory management techniques.
//!
//! **Note:** This is a work-in-progress and primarily for educational purposes. It is **not** production-ready.

#![feature(allocator_api, min_specialization)]
#![warn(
    clippy::perf,
    clippy::correctness,
    clippy::complexity,
    clippy::style,
    clippy::suspicious,
    // clippy::pedantic
)]
#![deny(clippy::undocumented_unsafe_blocks, clippy::cast_possible_truncation)]
#![allow(clippy::float_cmp)]

mod error;
mod shape;
mod storage;
mod tensor;
mod tensorizable;

// Re-export core types for convenience.
pub use tensor::Tensor;
pub use tensorizable::Tensorizable;