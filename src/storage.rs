//! Provides raw, aligned memory storage [`Storage`] for tensor data.
//! Handles allocation, deallocation, and basic access, with memory alignment.

use std::rc::Rc;

use crate::buffer::{Buffer, BufferBuilder};

/// `Storage<T, A>` is a partially-initialized memory container.
///
/// It wraps [`Buffer<T, A>`], which handles allocation and layout.
/// - The uninitialized tail (if any) of the `Buffer` is never exposed directly.
pub struct Storage<T, A = std::alloc::Global>
where
    A: std::alloc::Allocator,
{
    /// See [`crate::buffer::Buffer`].
    buffer: Buffer<T, A>,
    /// The number of elements guaranteed to be initialized.
    init: usize,
}

impl<T, A: std::alloc::Allocator> Storage<T, A> {
    /// Creates a new storage buffer for `numel` elements using the given allocator.
    ///
    /// Allocated memory is uninitialized. no elements are considered initialized yet.
    pub fn new(numel: usize, alloc: &Rc<A>) -> Self {
        let buffer: Buffer<T, A> = BufferBuilder::new(numel).build(alloc);
        Self { buffer, init: 0 }
    }

    /// Returns a reference to the element at `index` if it has been initialized.
    ///
    /// Returns `None` if `index >= self.init`.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.init {
            return None;
        }

        // SAFETY:
        // - `buffer.as_ptr()` is a valid, non-null, aligned pointer to
        //   a allocated buffer.
        // - index is bounds-checked against init, and init guarantees
        //   that elements [0..init) are properly initialized.
        unsafe { self.buffer.as_ptr().add(index).as_ref() }
    }

    /// writes a value to the next uninitialized slot, extending `init` by 1.
    ///
    /// # Safety
    ///
    /// - `init < allocated_len()` must hold.
    /// - caller must not use old pointers that alias the written memory.
    pub unsafe fn write_unchecked(&mut self, value: T) {
        debug_assert!(self.init < self.allocated_len());
        // SAFETY:
        // - `self.as_mut_ptr()` is a valid, non-null, aligned pointer.
        // - `self.init` < `self.allocated_len()`
        unsafe {
            std::ptr::write(self.as_mut_ptr().add(self.init), value);
        }
        self.init += 1;
    }

    /// Unsafely sets `init = len`. 
    /// Caller must ensure elements `[0..len)` are valid.
    ///
    /// # Safety
    ///
    /// - `len <= allocated_len()`
    /// - Elements in `[0, len)` must be initialized.
    pub unsafe fn assume_init(&mut self, len: usize) {
        debug_assert!(len <= self.allocated_len());
        self.init = len;
    }

    /// Drops all initialized elements and resets the init counter.
    ///
    /// Keeps the allocation alive.
    pub fn clear(&mut self) {
        for i in 0..self.init {
            // SAFETY:
            // - `ptr + i` is within the slice region since we
            //   allocate exact memory.
            // - `val` is cloned beforehand so panic is separated from
            //   the write.
            unsafe {
                std::ptr::drop_in_place(self.buffer.as_mut_ptr().add(i));
            }
        }
        self.init = 0;
    }

    /// Returns a mutable reference to the element at `index` if it has been initialized.
    ///
    /// Returns `None` if `index >= self.init`.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.init {
            return None;
        }
        // SAFETY:
        // - `buffer.as_mut_ptr()` is a valid, non-null, aligned pointer to
        //   a allocated buffer.
        // - index is bounds-checked against init, and init guarantees
        //   that elements [0..init) are properly initialized.
        unsafe { self.buffer.as_mut_ptr().add(index).as_mut() }
    }

    /// Returns the number of initialized elements.
    ///
    /// Only elements in `[0, len())` are safe to access.
    pub fn len(&self) -> usize {
        self.init
    }

    /// Returns the number of elements the buffer was originally allocated for.
    ///
    /// May be larger than `len()`; uninitialized tail must not be accessed.
    pub fn allocated_len(&self) -> usize {
        self.buffer.numel()
    }

    /// Returns `true` if no elements are initialized.
    pub fn is_empty(&self) -> bool {
        self.init == 0
    }

    /// Returns the actual capacity in elements, accounting for allocator alignment.
    ///
    /// This may differ from `allocated_len()` if padding or over-allocation occurs.
    pub fn capacity(&self) -> usize {
        self.buffer.allocated_capacity()
    }

    /// Returns a raw const pointer to the start of the buffer.
    ///
    /// Only valid for reads within `[0, init)`.
    pub fn as_ptr(&self) -> *const T {
        self.buffer.as_ptr()
    }

    /// Returns a raw mut pointer to the start of the buffer.
    ///
    /// Only valid for writes within `[0, init)` or for manual initialization.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.buffer.as_mut_ptr()
    }

    /// Returns a shared slice of all initialized elements `[0, init)`.
    ///
    /// # Safety
    ///
    /// Does not include uninitialized memory.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY:
        // - `self.as_ptr()` is a valid non-null, aligned pointer to
        //   allocated memory.
        // - `self.init` is a valid number of initialized elements
        //   within the allocated region.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.init) }
    }

    /// Returns a mutable slice of all initialized elements `[0, init)`.
    ///
    /// # Safety
    ///
    /// Does not include uninitialized memory.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY:
        // - `self.as_mut_ptr()` is a valid non-null, aligned pointer to
        //   allocated memory.
        // - `self.init` is a valid number of initialized elements
        //   within the allocated region.
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.init) }
    }
}

impl<T: Clone, A: std::alloc::Allocator> Storage<T, A> {
    /// Creates a new storage buffer and clones each element from the given slice.
    ///
    /// All elements are immediately initialized.
    fn from_slice(slice: &[T], alloc: &Rc<A>) -> Self {
        let mut buffer: Buffer<T, A> = BufferBuilder::new(slice.len()).build(alloc);
        let mut init = 0;
        for (i, val) in slice.iter().enumerate() {
            let val = val.clone();
            // SAFETY:
            // - `ptr + i` is within the slice region since we
            //   allocate exact memory.
            // - `val` is cloned beforehand so panic is separated from
            //   the write.
            unsafe {
                std::ptr::write(buffer.as_mut_ptr().add(i), val);
            }
            init += 1;
        }
        Self { buffer, init }
    }

    /// Creates a new storage buffer of `numel` elements, each cloned from `value`.
    ///
    /// All elements are immediately initialized.
    pub fn filled_with(numel: usize, value: T, alloc: &Rc<A>) -> Self {
        let mut buffer: Buffer<T, A> = BufferBuilder::new(numel).build(alloc);
        let mut init = 0;
        for i in 0..numel {
            let val = value.clone();
            // SAFETY:
            // - `ptr + i` is within the slice region since we
            //   allocate exact memory.
            // - `val` is cloned beforehand so panic is separated from
            //   the write.
            unsafe {
                std::ptr::write(buffer.as_mut_ptr().add(i), val);
            }
            init += 1;
        }

        Self { buffer, init }
    }
}

impl<T, A: std::alloc::Allocator> Drop for Storage<T, A> {
    fn drop(&mut self) {
        // Drop all initialized elements
        for i in 0..self.init {
            // SAFETY:
            // - `buffer.as_mut_ptr()` is a valid, aligned non-null pointer.
            // - `ptr + i` is valid within initialized elements.
            // - `T` at `ptr + i` is initialized.
            unsafe {
                std::ptr::drop_in_place(self.buffer.as_mut_ptr().add(i));
            }
        }
    }
}
