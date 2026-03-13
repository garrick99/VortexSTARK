//! GPU device memory buffer with RAII semantics.

use crate::cuda::ffi;
use std::ffi::c_void;
use std::marker::PhantomData;

/// Owning handle to a contiguous GPU allocation.
/// Automatically freed on drop.
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for DeviceBuffer<T> {}

impl<T> DeviceBuffer<T> {
    /// Allocate `len` elements on the GPU (uninitialized).
    pub fn alloc(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                _marker: PhantomData,
            };
        }
        let bytes = len * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { ffi::cudaMalloc(&mut ptr, bytes) };
        assert!(err == 0, "cudaMalloc failed: error {err}");
        Self {
            ptr: ptr as *mut T,
            len,
            _marker: PhantomData,
        }
    }

    /// Upload host slice to a new GPU buffer.
    pub fn from_host(data: &[T]) -> Self {
        let buf = Self::alloc(data.len());
        if !data.is_empty() {
            let bytes = data.len() * std::mem::size_of::<T>();
            let err = unsafe {
                ffi::cudaMemcpy(
                    buf.ptr as *mut c_void,
                    data.as_ptr() as *const c_void,
                    bytes,
                    ffi::MEMCPY_H2D,
                )
            };
            assert!(err == 0, "cudaMemcpy H2D failed: error {err}");
        }
        buf
    }

    /// Download GPU buffer contents to a host Vec.
    pub fn to_host(&self) -> Vec<T>
    where
        T: Default + Clone,
    {
        let mut host = vec![T::default(); self.len];
        if self.len > 0 {
            let bytes = self.len * std::mem::size_of::<T>();
            let err = unsafe {
                ffi::cudaMemcpy(
                    host.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    bytes,
                    ffi::MEMCPY_D2H,
                )
            };
            assert!(err == 0, "cudaMemcpy D2H failed: error {err}");
        }
        host
    }

    /// Zero-fill the buffer.
    pub fn zero(&mut self) {
        if self.len > 0 {
            let bytes = self.len * std::mem::size_of::<T>();
            unsafe { ffi::cudaMemset(self.ptr as *mut c_void, 0, bytes) };
        }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy device data into a new device buffer (D2D).
    pub fn clone_device(&self) -> Self {
        let new_buf = Self::alloc(self.len);
        if self.len > 0 {
            let bytes = self.len * std::mem::size_of::<T>();
            let err = unsafe {
                ffi::cudaMemcpy(
                    new_buf.ptr as *mut c_void,
                    self.ptr as *const c_void,
                    bytes,
                    ffi::MEMCPY_D2D,
                )
            };
            assert!(err == 0, "cudaMemcpy D2D failed: error {err}");
        }
        new_buf
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cudaFree(self.ptr as *mut c_void) };
        }
    }
}
