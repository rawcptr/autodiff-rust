use std::{
    alloc::{Allocator, Layout},
    ptr::NonNull,
    rc::Rc,
};

/// `x86_64` AVX2 32-byte alignment
#[allow(unused)]
pub const AVX2_ALIGN: usize = 32;

/// `AArch64` NEON 16-byte alignment
#[allow(unused)]
pub const NEON_ALIGN: usize = 16;

/// Raw, aligned heap storage for elements of type `T`.
///
/// Owns the allocated memory and handles deallocation.
/// Ensures specific memory alignment with AVX and NEON requirements.
/// defaulting to 32 bytes for `x86_64` with AVX2 feature
/// and 16 bytes for `aarch64` with NEON feature.
///
/// # Note
///
/// `RawStorage` only drops the underlying allocation. It will **NOT** drop the `T` present in the allocated memory.
/// This storage is intended to be a low-surface-area unsafe pool of aligned memory that can later be layered on with a safe abstraction
#[derive(Debug)]
pub struct Buffer<T, Alloc: Allocator + ?Sized> {
    /// Pointer to start of allocation.
    ptr: NonNull<T>,
    /// Number of elements originally requested (`numel`).
    numel: usize,
    /// Full layout used during allocation (includes padding).
    layout: Layout,
    /// Reference to underlying storage allocator.
    allocator: Rc<Alloc>,
}

/// Builder for constructing a [`Buffer`] with custom settings.
///
/// This allows customizing number of elements, memory alignment,
/// and whether the memory should be zero-initialized.
pub struct BufferBuilder {
    numel: usize,
    zeroed: bool,
    align: usize,
}

impl BufferBuilder {
    pub fn new(numel: usize) -> Self {
        Self {
            numel,
            zeroed: false,
            align: std::mem::align_of::<()>(),
        }
    }

    /// If true, the buffer will be allocated with all bytes set to zero.
    #[must_use]
    pub fn zeroed(mut self, z: bool) -> Self {
        self.zeroed = z;
        self
    }

    /// Override default alignment (see [`Buffer`] docs for platform defaults).
    ///
    /// Must be a power of two.
    #[must_use]
    pub fn with_alignment(mut self, align: usize) -> Self {
        self.align = align;
        self
    }

    pub fn build<T, A: Allocator + ?Sized>(self, alloc: &Rc<A>) -> Buffer<T, A> {
        let Self { numel, zeroed, .. } = self;
        let align = if self.align == std::mem::align_of::<()>() {
            Self::alignment::<T>()
        } else {
            self.align
        };

        Buffer::with_alignment(numel, align, zeroed, alloc)
    }

    const fn alignment<T>() -> usize {
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

impl<T, A: Allocator + ?Sized> Buffer<T, A> {
    /// Returns a `RawStorage` with specified attributes.
    ///
    /// # Arguments
    ///
    /// * `numel` - number of elements to allocate for.
    /// * `align` - required alignment of memory.
    /// * `zeroed` - if allocated memory should be zeroed out.
    /// * `allocator` - The allocator to use.
    ///
    /// # Panics
    ///
    /// Panics if `T` is a Zero-Sized Type, `numel` is 0, or `align` is not a power of two.
    fn with_alignment(numel: usize, align: usize, zeroed: bool, allocator: &Rc<A>) -> Self {
        assert!((std::mem::size_of::<T>() != 0), "ZSTs are not supported.");
        assert!(
            (numel != 0),
            "zero-sized buffers (numel=0) are not supported."
        );
        assert!(align.is_power_of_two(), "Alignment must be a power of two");

        let size = self::utils::align_to::<T>(numel, align);

        let layout = Layout::from_size_align(size, align).unwrap_or_else(|_| {
            panic!("layout creation should have valid alignment: {align} and length: {numel}")
        });

        let ptr = {
            if zeroed {
                // SAFETY:
                // - layout is non-zero size and valid alignment (guaranteed by assertions).
                // - Trusting the allocator to return a valid pointer on success.
                allocator
                    .allocate_zeroed(layout)
                    .unwrap_or_else(|_| panic!("allocator failed to allocate layout: {layout:#?}"))
                    .cast()
            } else {
                // SAFETY:
                // - layout is non-zero size and valid alignment (guaranteed by assertions).
                // - Trusting the allocator to return a valid pointer on success.
                let tmp: NonNull<T> = allocator
                    .allocate(layout)
                    .unwrap_or_else(|_| panic!("allocator failed to allocate layout: {layout:#?}"))
                    .cast();

                self::utils::zero_trailing_bytes::<T>(tmp.as_ptr().cast(), numel, size);
                tmp
            }
        };
        #[cfg(debug_assertions)]
        // SAFETY:
        // - this code is only ran in debug builds.
        // - `ptr.as_ptr()` is a valid non-null aligned pointer to allocated memory.
        // - `size` is the number of *bytes* in the array.
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0xAB, size);
        }

        Buffer {
            ptr,
            layout,
            numel,
            allocator: allocator.clone(),
        }
    }

    /// Returns the internal pointer to the underlying memory.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a mutable internal pointer to the underlying memory
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Return the underlying layout: [`std::alloc::Layout`]
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Return the total allocated size of storage in bytes.
    #[inline]
    pub fn allocated_size_bytes(&self) -> usize {
        self.layout().size()
    }

    /// Return the total number of elements `T` that can fit in the allocated memory.
    /// This includes space for padding beyond the requested number of elements.
    /// This is the total capacity in terms of number of `T` elements.
    #[inline]
    pub fn allocated_capacity(&self) -> usize {
        self.layout().size() / std::mem::size_of::<T>()
    }

    /// Returns the number of elements originally requested (logical length).
    #[inline]
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Returns a slice over the logical allocated region.
    ///
    /// # Safety
    /// Caller must ensure that all elements within the returned slice
    /// are fully initialized.
    ///
    /// Calling this function with any element uninitialized is **undefined behavior**.
    pub unsafe fn as_slice(&self) -> &[T] {
        // SAFETY:
        // - `self.as_ptr()` returns a valid, non-null, aligned pointer.
        // - `self.allocated_capacity()` returns the correct number of elements
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.numel()) }
    }

    /// Returns a mutable slice over the logical allocated region.
    ///
    /// # Safety
    /// Caller must ensure that all elements within the returned slice
    /// are fully initialized.
    ///
    /// Calling this function with any element uninitialized is **undefined behavior**.
    pub unsafe fn as_slice_mut(&mut self) -> &mut [T] {
        // SAFETY:
        // - `as_mut_ptr` is a pointer to a valid, non-null, aligned pointer.
        // - `self.allocated_capacity()` returns the correct number of elements
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.numel()) }
    }
}

impl<T, A: Allocator + ?Sized> Drop for Buffer<T, A> {
    /// Deallocates the buffer. Does **not** drop any `T`s.
    fn drop(&mut self) {
        // SAFETY:
        // - `self.as_mut_ptr()` is not modified from the original allocation
        // - `self.layout()` is the same layout used for the original allocation
        unsafe {
            self.allocator.deallocate(self.ptr.cast(), self.layout());
        }
    }
}

mod utils {
    /// Returns allocation size (in bytes) for `numel` elements of `T`,
    /// rounded up to the nearest multiple of `align`.
    #[inline]
    pub fn align_to<T>(numel: usize, align: usize) -> usize {
        let tsize = std::mem::size_of::<T>();

        let size_in_bytes = numel
            .checked_mul(tsize)
            .unwrap_or_else(|| panic!("numel {numel} * tsize {tsize} overflowed."));

        (size_in_bytes + align - 1) & !(align - 1)
    }

    /// Fills trailing padding bytes with zeroes (if any).
    ///
    /// This is useful when SIMD loads might read past initialized data.
    /// Does nothing if `length * size_of::<T>() >= size`.
    #[inline]
    pub fn zero_trailing_bytes<T>(ptr: *mut u8, length: usize, size: usize) {
        let start_offset = length * std::mem::size_of::<T>();
        if start_offset >= size {
            return;
        }

        let pad_bytes = size - start_offset;

        // SAFETY:
        // - `base.add(start_offset)` is within allocation of `size` bytes
        unsafe {
            std::ptr::write_bytes(ptr.add(start_offset), 0, pad_bytes);
        }
    }
}
