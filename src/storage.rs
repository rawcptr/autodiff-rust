//! Provides raw, aligned memory storage [`Storage`] for tensor data.
//! Handles allocation, deallocation, and basic access, with memory alignment.

use std::{
    alloc::Layout,
    marker::PhantomData,
    ptr::{self, NonNull},
};

use crate::error::TensorError;

/// Raw, aligned heap storage for elements of type `T`.
///
/// Owns the allocated memory and handles dropping elements and deallocation.
/// Ensures specific memory alignment (defaulting to 32 bytes for AVX2 compatibility).
#[derive(Debug)]
pub struct Storage<T> {
    /// Pointer to heap allocation.
    ptr: NonNull<T>,
    /// The number of elements the storage holds.
    len: usize,
    /// The memory layout (size and alignment) used for allocation.
    layout: Layout,

    // marker to indicate Storage owns the underlying allocation
    // even when it only hold a raw pointer.
    _marker: PhantomData<T>,
}

impl<T> Storage<T> {
    /// AVX2 32-byte alignment
    pub const ALIGN: usize = 32;

    /// Allocates aligned storage for elements of type `T`.
    ///
    /// The allocation is 32-byte aligned. The actual allocated size might be
    /// larger than `numel * size_of::<T>()` due to padding for alignment.
    ///
    /// # Panics
    ///
    /// Panics if memory allocation fails or if the calculated layout is invalid
    /// (e.g., size overflows `isize`).
    ///
    /// # Errors
    /// Returns [`TensorError::MemoryViolation`] if the container size mismatches the requested `numel`
    /// during initialization (which shouldn't happen with this constructor variant, but is relevant
    /// for `with_container`).
    pub fn new(container: impl AsRef<[T]>) -> Result<Self, TensorError> {
        let size = container.as_ref().len();
        let mut storage = Self::uninitialized(size);
        storage.with_container(container)?;
        Ok(storage)
    }

    /// Allocates aligned, **uninitialized** storage for `numel` elements of type `T`.
    ///
    /// The allocated memory is aligned to [`Storage::ALIGN`] bytes. Padding may be added
    /// to meet alignment requirements; this padding is zeroed.
    ///
    /// **Warning:** The primary storage area for the `numel` elements is **uninitialized**.
    /// It's the caller's responsibility to initialize these elements before reading them.
    ///
    /// # Panics
    ///
    /// Panics if memory allocation fails or if the calculated layout is invalid.
    pub fn uninitialized(numel: usize) -> Self {
        // align padding for avx2
        let size = align_to::<T>(numel, Self::ALIGN);

        let ptr = if size == 0 {
            // Use dangling pointer for zero-sized allocations, aligning it.
            NonNull::dangling()
        } else {
            let layout = Layout::from_size_align(size, Self::ALIGN)
                .expect("layout creation should have valid alignment and length");

            // SAFETY:
            // - `layout` is guaranteed to be valid and non-zero size by the `Layout::from_size_align`
            //   call above and the `if size == 0` check.
            // - `std::alloc::alloc` is the correct function for allocating raw memory.
            let ptr = unsafe { std::alloc::alloc(layout) };
            // Check for allocation failure.
            assert!(!ptr.is_null(), "allcation failed.");

            // SAFETY:
            // - `ptr` is non-null, as asserted above.
            // - `ptr` points to memory allocated with `layout`, which has the correct alignment (`Self::ALIGN`).
            NonNull::new(ptr.cast::<T>())
                .expect("NonNull::new failed after successful allocation check")
        };

        let layout = Layout::from_size_align(size, Self::ALIGN)
            .expect("layout creation should succeed even for size 0");

        let mut storage = Self {
            ptr,
            len: numel,
            layout,
            _marker: PhantomData,
        };

        storage.pad_zeros();
        storage
    }

    /// Initializes the allocated storage by copying data from a container.
    ///
    /// This should only be called on storage created with `uninitialized` or similar methods
    /// where the elements `0..self.len` are known to be uninitialized.
    ///
    /// # Errors
    /// Returns [`TensorError::MemoryViolation`] if the `container` length does not exactly match
    /// `self.len()`.
    ///
    fn with_container(&mut self, container: impl AsRef<[T]>) -> Result<(), TensorError> {
        let slice = container.as_ref();

        debug_assert_eq!(
            slice.as_ptr() as usize % std::mem::align_of::<T>(),
            0,
            "Source slice not properly aligned"
        );

        let slice_size = slice.len();
        let tensor_capacity = self.len();

        if slice_size != tensor_capacity {
            return Err(TensorError::MemoryViolation {
                why: format!(
                    "tried to assign {slice_size} elements to a buffer that can hold {tensor_capacity} elements"
                ),
            });
        }

        // SAFETY:
        // - `slice.as_ptr()` points to valid memory for `slice.len()` elements of `T`,
        //   as guaranteed by `AsRef<[T]>`.
        // - `self.as_mut_ptr()` points to valid, aligned memory allocated for at least `self.len()` elements.
        // - `slice.len()` equals `self.len()` due to the check above, ensuring the copy is within bounds
        //   for both source and destination.
        // - The source and destination memory regions are non-overlapping because `self` owns
        //   newly allocated memory and `slice` comes from an external source.
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), self.as_mut_ptr(), slice.len());
        }

        Ok(())
    }

    /// Returns a raw const pointer to the start of the storage buffer.
    ///
    /// # Safety
    /// The caller must ensure the pointer is used safely (e.g., not dereferenced beyond `len`,
    /// respecting aliasing rules). The storage guarantees the pointer is valid and aligned
    /// for `len` elements.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a raw mutable pointer to the start of the storage buffer.
    ///
    /// # Safety
    /// The caller must ensure the pointer is used safely (e.g., not read/written beyond `len`,
    /// respecting aliasing rules, ensuring proper initialization before reads). The storage
    /// guarantees the pointer is valid and aligned for `len` elements.
    #[inline]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns the number of elements the storage is allocated for (excluding padding).
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Returns an immutable slice covering the initialized elements `0..self.len`.
    ///
    /// # Safety
    ///
    /// The caller *must* ensure that all elements from index `0` up to (but not including)
    /// `self.len()` have been properly initialized before calling this function. Accessing
    /// elements via the returned slice invokes undefined behavior if they are uninitialized.
    #[inline(always)]
    pub(crate) unsafe fn as_slice(&self) -> &[T] {
        // SAFETY:
        // - `self.as_ptr()` returns a valid, non-null, aligned pointer allocated for at least `self.len` elements.
        // - `self.len` holds the number of elements intended to be stored.
        // - The caller guarantees that elements `0..self.len` are initialized, fulfilling the contract
        //   of `std::slice::from_raw_parts`.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Returns a mutable slice covering the initialized elements `0..self.len`.
    ///
    /// # Safety
    ///
    /// The caller *must* ensure that all elements from index `0` up to (but not including)
    /// `self.len()` have been properly initialized before calling this function. Accessing
    /// elements via the returned slice invokes undefined behavior if they are uninitialized.
    /// Additionally, the caller must ensure that no other mutable references to the slice elements exist.
    #[inline(always)]
    pub(crate) unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY:
        // - `self.as_mut_ptr()` returns a valid, non-null, aligned pointer allocated for at least `self.len` elements.
        // - `self.len` holds the number of elements intended to be stored.
        // - The caller guarantees that elements `0..self.len` are initialized, fulfilling the contract
        //   of `std::slice::from_raw_parts_mut`.
        // - The caller guarantees exclusive mutable access.
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    #[inline]
    /// Fills any padding bytes (beyond `self.len` up to `self.layout.size()`) with zeros.
    /// This is often done after initialization to ensure predictable values in padding areas,
    /// which might be read by SIMD operations that load full alignment blocks.
    ///
    /// Calling this function where `T` is zero sized will result in a panic.
    pub(crate) fn pad_zeros(&mut self) {
        if self.layout.size() != 0 {
            let padding_len_elements = self.layout.size() / std::mem::size_of::<T>() - self.len();
            // SAFETY:
            // - `self.as_mut_ptr()` is a valid pointer to the start of the allocation.
            // - `add(self.ptr.add(self.len()).as_ptr())` calculates the pointer to the first padding element.
            //   This offset is valid because `self.len * size_of<T>() <= self.layout.size()`.
            // - `padding_len_elements` is the number of *elements* (not bytes) in the padding area.
            //   This calculation is correct based on the total allocated size (`layout.size()`)
            //   and the used size (`self.len * size_of<T>`).
            // - `ptr::write_bytes` correctly zeroes the specified number of *bytes*. We write `padding_len_elements`
            //   times the size of T bytes.
            unsafe {
                let padding_start = self.ptr.add(self.len()).as_ptr();
                ptr::write_bytes(padding_start, 0, padding_len_elements);
            }
        }
    }

    /// Directly reads a reference to the element at the specified `offset`.
    ///
    /// # Panics
    /// Panics if `offset >= self.len()`.
    /// Assumes the element at `offset` is initialized.
    #[inline]
    pub(crate) fn direct_read(&self, offset: usize) -> &T {
        assert!(offset < self.len(), "Direct read offset out of bounds");
        // SAFETY:
        // - `self.as_ptr()` returns a valid, non-null, aligned pointer.
        // - The `offset` is checked to be within the bounds `[0, self.len())`.
        // - `add(offset)` computes a pointer to a valid element within the allocation.
        // - Dereferencing this pointer (`*...`) is safe because the element is assumed to be initialized
        //   (as `Storage` is typically used within `Tensor` which manages initialization).
        unsafe { &*self.as_ptr().add(offset) }
    }

    /// Directly writes `val` to the element at the specified `offset`.
    ///
    /// # Safety
    /// The caller *must* ensure that `offset` is within the bounds `[0, self.len())`.
    /// Writing out of bounds causes undefined behavior.
    #[inline]
    pub(crate) unsafe fn direct_write(&mut self, offset: usize, val: T) {
        // SAFETY:
        // - `self.as_mut_ptr()` returns a valid, non-null, aligned pointer.
        // - The caller guarantees that `offset` is within the bounds `[0, self.len())`.
        // - `add(offset)` computes a pointer to a valid element location within the allocation.
        // - `write(val)` performs a volatile write, overwriting the memory at that location. This is
        //   safe because the location is guaranteed valid by the caller. It assumes ownership of `val`
        unsafe {
            self.as_mut_ptr().add(offset).write(val);
        }
    }
}

#[inline]
/// Calculates the required allocation size in bytes for `numel` elements of type `T`,
/// ensuring the size is a multiple of `align` bytes.
fn align_to<T>(numel: usize, align: usize) -> usize {
    let tsize = std::mem::size_of::<T>();
    if tsize == 0 || numel == 0 {
        return 0;
    }
    let size_in_bytes = numel * tsize;

    (size_in_bytes + align - 1) & !(align - 1)
}

impl<T> Drop for Storage<T> {
    fn drop(&mut self) {
        if self.layout.size() == 0 {
            return;
        }

        if std::mem::needs_drop::<T>() {
            let mut curr_ptr = self.ptr.as_ptr();

            for _ in 0..self.len {
                // SAFETY:
                // - `curr_ptr` points to an element within the allocated buffer (`0..self.len`).
                // - We iterate exactly `self.len` times, ensuring we only drop initialized elements
                //   (as per the contract of how `Storage` is typically used and initialized).
                // - The pointer `curr_ptr` is valid for reads and writes (as it's part of the allocation).
                // - `drop_in_place` correctly handles the drop logic for type `T`.
                // - `curr_ptr` is advanced correctly within the bounds.
                unsafe {
                    std::ptr::drop_in_place(curr_ptr);
                    curr_ptr = curr_ptr.add(1);
                }
            }
        }

        // SAFETY:
        // - `self.ptr` points to the memory block previously allocated by `std::alloc::alloc` or `alloc_zeroed`.
        // - `self.layout` is the exact `Layout` instance that was used for the allocation call.
        // - The memory block has not been deallocated yet.
        // - All elements within the block that required dropping have been dropped in the loop above.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr().cast(), self.layout) }
    }
}

impl<T: Copy> Storage<T> {
    /// Returns the copy storage of this [`Storage<T>`].
    pub fn copy_storage(&self) -> Self {
        if self.len() == 0 {
            return Storage::<T>::uninitialized(0);
        }

        let mut new_storage = Storage::<T>::uninitialized(self.len());

        // SAFETY:
        // - `self.as_ptr()` points to valid memory containing `self.len()` initialized elements of type `T`.
        // - `new_storage.as_mut_ptr()` points to valid, aligned, allocated memory sufficient for `self.len()` elements.
        // - `self.len()` ensures the copy stays within the bounds of both source and destination.
        // - Source and destination are non-overlapping (new allocation).
        // - `T: Copy` guarantees that a bitwise copy (`copy_nonoverlapping`) is a valid way to duplicate `T`.
        unsafe {
            ptr::copy_nonoverlapping(self.as_ptr(), new_storage.as_mut_ptr(), self.len());
        }

        new_storage
    }
}

impl<T: Clone> Storage<T> {
    pub fn clone_storage(&self) -> Self {
        if self.len() == 0 {
            return Storage::<T>::uninitialized(self.len());
        }
        // SAFETY: `self` contains `len` initialized elements, so `as_slice` is safe here.
        let src_slice = unsafe { self.as_slice() };
        // Note: This involves an intermediate Vec allocation. Could be optimized
        // The double allocation is necessary because Vec correctly handles panic
        // during initializatin.
        // if T: Clone exits, and we clone it, but we panic during copy from self to other
        // we will start dropping the values and try to drop uninitialized memory.
        // vector allocation ensures we only drop what was copied.
        let cloned_data = src_slice.to_vec();
        Storage::new(cloned_data).expect("failed to clone storage")
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import Storage, align_to etc.
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    // Helper struct for Drop test
    #[derive(Clone)]
    struct DropTracker {
        _id: usize, // Optional: helps debugging
        counter: Arc<AtomicUsize>,
    }
    impl Drop for DropTracker {
        fn drop(&mut self) {
            // println!("Dropping tracker {}", self.id); // For debugging
            self.counter.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct ZeroSizedType; // Example ZST

    #[test]
    fn test_allocation_deallocation() {
        // Test various types and sizes
        let _s1 = Storage::<f32>::uninitialized(10);
        let _s2 = Storage::<u8>::uninitialized(128);
        let _s3 = Storage::<()>::uninitialized(5); // ZST via unit type
        let _s4 = Storage::<ZeroSizedType>::uninitialized(20);
        // Just letting them drop is the test (miri checks dealloc)
    }

    #[test]
    fn test_zero_elements() {
        let s = Storage::<f32>::new(vec![]).unwrap();
        assert_eq!(s.len(), 0);
        assert_eq!(s.layout.size(), 0);
        // Miri checks drop behavior
    }

    #[test]
    fn test_alignment() {
        let s_f32 = Storage::<f32>::uninitialized(100);
        let s_u8 = Storage::<u8>::uninitialized(100);
        let s_u64 = Storage::<u64>::uninitialized(100);

        assert_eq!(s_f32.as_ptr() as usize % Storage::<f32>::ALIGN, 0);
        assert_eq!(s_u8.as_ptr() as usize % Storage::<u8>::ALIGN, 0);
        assert_eq!(s_u64.as_ptr() as usize % Storage::<u64>::ALIGN, 0);
    }

    #[test]
    fn test_element_drop() {
        let counter = Arc::new(AtomicUsize::new(0));
        let numel = 10;
        {
            let mut storage: Storage<DropTracker> = Storage::uninitialized(numel);
            // Initialize elements safely if possible, otherwise unsafe write
            for i in 0..numel {
                let tracker = DropTracker {
                    _id: i,
                    counter: counter.clone(),
                };
                // IMPORTANT: Need unsafe write because storage is uninitialized
                unsafe {
                    std::ptr::write(storage.as_mut_ptr().add(i), tracker);
                }
            }

            assert_eq!(storage.len(), numel);
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        } // storage drops here
        // Check if all elements were dropped
        assert_eq!(counter.load(Ordering::SeqCst), numel);
    }

    #[test]
    fn test_data_integrity() {
        let numel = 10;
        let mut storage = Storage::<i32>::uninitialized(numel);
        // Write data
        for i in 0..numel {
            // SAFETY: we only write numel elements that have allocation behind them
            unsafe {
                storage.as_mut_ptr().add(i).write(i as i32 * 10);
            }
        }
        // Read data back
        for i in 0..numel {
            // SAFETY: we only read within num elements that are allocated for
            let value = unsafe { storage.as_ptr().add(i).read() };
            assert_eq!(value, i as i32 * 10);
        }
    }

    #[test]
    fn test_simd_padding_zeroed() {
        let numel = 3; // Not a multiple of 8 (for f32 AVX)
        let storage = Storage::<f32>::uninitialized(numel);

        // SAFETY:
        // index only within bounds.
        // Check padding is zeroed
        unsafe {
            for i in numel..storage.layout.size() / std::mem::size_of::<f32>() {
                assert_eq!(storage.as_ptr().add(i).read(), 0.0);
            }
        }
    }

    #[test]
    #[should_panic(expected = "tried to assign 3 elements to a buffer that can hold 2 elements")]
    fn test_overfilled_storage() {
        let mut storage = Storage::<u8>::uninitialized(2);
        if let TensorError::MemoryViolation { why } =
            storage.with_container(vec![1, 2, 3]).unwrap_err()
        {
            panic!("{why}")
        } else {
            unreachable!("what")
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // proptest! {
    //     #[test]
    //     fn test_any_length(len in 0..1000usize) {
    //         let storage = Storage::<u32>::uninitialized(len);
    //         assert_eq!(storage.len(), len);
    //     }
    // }
}
