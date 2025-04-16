use std::{
    alloc::Layout,
    marker::PhantomData,
    ptr::{self, NonNull},
};

use crate::error::TensorError;

/// Raw, aligned heap storage for elements of type `T`.
///
/// Owns the allocated memory and handles dropping elements and deallocation.
/// Ensures specific memory alignment (32 bytes for AVX2).
pub struct Storage<T> {
    /// Pointer to heap allocation.
    ptr: NonNull<T>,
    /// The number of elements the storage holds.
    len: usize,
    /// The memory layout (size and alignment) used for allocation.
    layout: Layout,

    // storage should own the underlying data.
    _marker: PhantomData<T>,
}

impl<T> Storage<T> {
    const ALIGN: usize = 32;
    /// Allocates aligned storage for `numel` elements of type `T`.
    ///
    /// The allocation is 32-byte aligned. The actual allocated size might be
    /// larger than `numel * size_of::<T>()` due to padding for alignment.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails or if the calculated layout is invalid.
    pub fn new(size: usize, container: impl AsRef<[T]>) -> Result<Self, TensorError> {
        let mut storage = Self::uninitialized(size);
        storage.with_container(container)?;
        Ok(storage)
    }

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
            // - Layout is valid from Layout::size_align
            // - alloc() -> null is acceptable. we just panic.
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                panic!("allcation failed.")
            }
            // SAFETY: Case is safe.
            // - Layout matches T's alignment (align_to<T> ensures this.)
            // - ptr is non-null. checked above.
            NonNull::new(ptr.cast::<T>()).expect("allocation should not return a null pointer")
        };

        let layout = Layout::from_size_align(size, Self::ALIGN)
            .expect("layout creation should succeed even for size 0");

        if layout.size() != 0 {
            // SAFETY: Padding is zeroed to avoid any weird things happening
            unsafe {
                let padding_start = ptr.add(numel).as_ptr();
                ptr::write_bytes(
                    padding_start,
                    0,
                    (layout.size() / std::mem::size_of::<T>()) - numel,
                );
            }
        }

        Self {
            ptr,
            len: numel,
            layout,
            _marker: PhantomData,
        }
    }

    fn with_container(&mut self, container: impl AsRef<[T]>) -> Result<(), TensorError> {
        let slice = container.as_ref();

        debug_assert_eq!(
            slice.as_ptr() as usize % std::mem::align_of::<T>(),
            0,
            "Source slice not properly aligned"
        );

        let slice_size = std::mem::size_of_val(slice);
        let layout_size = self.layout.size();

        if slice_size != layout_size {
            return Err(TensorError::MemoryViolation {
                why: format!("{layout_size} buffer cannot hold {slice_size} bytes"),
            });
        }

        // SAFETY:
        // Slice is guaranteed to be valid.
        // Self is guaranteed to be valid (see: self::unaligned)
        // size is guaranteed to be exact due to check above
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), self.as_mut_ptr(), slice.len());
        }

        Ok(())
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Get a alice from the initalized region.
    ///
    /// # Safety
    /// - Must ensure all elements upto `self.len` are initialized.
    pub(crate) unsafe fn as_slice(&self) -> &[T] {
        // SAFETY:
        // self.len is always allocated during creation of Self
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Get a mutable alice from the initalized region.
    ///
    /// # Safety
    /// - Must ensure all elements upto `self.len` are initialized.
    pub(crate) unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY:
        // self.len is always allocated during creation of Self.
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }
}

#[inline]
/// Calculates allocation size in bytes,
/// and adds padding for alignment if necessary.
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
                // - `curr_ptr` is valid for drops (points to initialized `T`)
                // - `self.len` ensures we stay within initialized bounds
                // - No double-drop: `drop_in_place` takes ownership
                unsafe {
                    std::ptr::drop_in_place(curr_ptr);
                    curr_ptr = curr_ptr.add(1);
                }
            }
        }

        // SAFETY:
        // - `self.ptr` was allocated via `std::alloc::alloc` with `self.layout`
        // - `self.layout` matches the original allocation (stored in the struct)
        // - No elements remain to be dropped (handled above)
        unsafe { std::alloc::dealloc(self.ptr.as_ptr().cast(), self.layout) }
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
        let s = Storage::<f32>::new(0, vec![]);
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
            unsafe {
                storage.as_mut_ptr().add(i).write(i as i32 * 10);
            }
        }
        // Read data back
        for i in 0..numel {
            let value = unsafe { storage.as_ptr().add(i).read() };
            assert_eq!(value, i as i32 * 10);
        }
    }
}
