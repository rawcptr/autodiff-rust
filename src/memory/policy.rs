//! Memory allocation policies for buffers.
//!
//! This module provides strategy traits that control how memory is allocated and aligned.

use std::{
    alloc::{AllocError, Allocator, Layout},
    ptr::NonNull,
};

/// Strategy for initializing allocated memory.
pub trait InitStrategy {
    /// Returns a pointer to the allocated memory slice.
    /// 
    /// Allocates memory according to the strategy's initialization policy.
    /// The allocates the given `layout` by using the provided `allocator`
    /// 
    /// # Errors
    ///
    /// Returns an error if the given allocation fails.
    fn allocate<A: Allocator>(allocator: A, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;
}

/// Strategy for determining memory alignment requirements.
///
/// Implementations define alignment based on target architecture, SIMD capabilities,
/// or custom requirements. All alignment calculations are performed at compile time.
pub trait AlignmentStrategy {
    /// Returns the required memory alignment for type `T`.
    ///
    /// # Returns
    ///
    /// Memory alignment in bytes, always a power of two.
    fn alignment<T>() -> usize;
}

/// SIMD-optimized alignment strategy.
///
/// Automatically selects optimal alignment based on target architecture and available
/// SIMD instruction sets:
/// - **`ARM64 with NEON`**: 16-byte alignment
/// - **`x86/x86_64`**: with AVX2**: 32-byte alignment  
/// - **Fallback**: Uses `align_of::<T>()`
///
/// All alignment decisions are made at compile time using `cfg!` macros.
///
/// # Examples
/// ```ignore
/// use your_crate::memory::policy::{AlignmentStrategy, SimdAlignment};
///
/// // On x86_64 with AVX2, returns 32
/// let alignment = SimdAlignment::alignment::<f32>();
/// ```
pub struct SimdAlignment;

/// 16-byte alignment for ARM NEON SIMD operations.
const NEON_ALIGN: usize = 16;

/// 32-byte alignment for x86 AVX2 SIMD operations.  
const AVX2_ALIGN: usize = 32;

impl AlignmentStrategy for SimdAlignment {
    /// Returns SIMD-optimal alignment for the target architecture.
    ///
    /// # Panics
    ///
    /// Panics if the computed alignment is not a power of two (which should never
    /// happen with valid SIMD alignments).
    fn alignment<T>() -> usize {
        let ret = if cfg!(all(target_feature = "neon", target_arch = "aarch64")) {
            NEON_ALIGN
        } else if cfg!(all(
            target_feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        )) {
            AVX2_ALIGN
        } else {
            std::mem::align_of::<T>()
        };
        assert!(ret.is_power_of_two());
        ret
    }
}

/// Custom alignment strategy with compile-time specified alignment.
///
/// Provides a fixed alignment value specified as a const generic parameter.
/// Useful when you need specific alignment requirements that differ from
/// SIMD defaults.
///
/// # Examples
///
/// ```ignore
/// use your_crate::memory::policy::{AlignmentStrategy, CustomAlignment};
///
/// // 64-byte alignment (e.g., for cache line alignment)
/// let alignment = CustomAlignment::<64>::alignment::<f64>();
/// assert_eq!(alignment, 64);
/// ```
pub struct CustomAlignment<const ALIGN: usize>;

impl<const ALIGN: usize> AlignmentStrategy for CustomAlignment<ALIGN> {
    /// Returns the custom alignment value.
    ///
    /// # Panics
    /// 
    /// Panics if `ALIGN` is not a power of two.
    fn alignment<T>() -> usize {
        assert!(ALIGN.is_power_of_two());
        ALIGN
    }
}

/// Uninitialized memory allocation strategy.
///
/// Allocates memory without initializing it, leaving the contents undefined.
/// This is the fastest allocation strategy. 
///
/// # Safety
/// 
/// Memory allocated with this strategy contains undefined values. Users must
/// initialize all memory before reading from it.
pub struct Uninitialized;
impl InitStrategy for Uninitialized {
    fn allocate<A: Allocator>(allocator: A, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        allocator.allocate(layout)
    }
}


/// Zero-initialized memory allocation strategy.
///
/// Allocates memory and initializes all bytes to zero. This is safer than
/// uninitialized allocation but has a performance cost due to the zeroing
/// operation.
///
/// Useful when you need guaranteed clean memory or when working with types
/// where zero-initialization provides meaningful default values.
pub struct Zeroed;
impl InitStrategy for Zeroed {
    fn allocate<A: Allocator>(allocator: A, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        allocator.allocate_zeroed(layout)
    }
}
