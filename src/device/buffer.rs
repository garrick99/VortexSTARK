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
// SAFETY: GPU memory is globally visible to all host threads.
// CUDA API calls are serialized by the driver when accessing the same stream.
unsafe impl<T: Send> Sync for DeviceBuffer<T> {}

/// Whether to use synchronous CUDA malloc (needed for WSL2 compatibility).
/// cudaMallocAsync relies on the CUDA memory pool API which has issues
/// with WSL2's GPU-PV driver translation layer.
static USE_SYNC_MALLOC: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Force synchronous cudaMalloc/cudaFree instead of async pool-based allocation.
/// Call this before any GPU allocation if running under WSL2.
pub fn force_sync_malloc() {
    USE_SYNC_MALLOC.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Auto-detect WSL2 and enable sync malloc if needed.
pub fn detect_wsl2_and_configure() {
    if std::path::Path::new("/proc/sys/fs/binfmt_misc/WSLInterop").exists()
        || std::env::var("WSL_DISTRO_NAME").is_ok()
    {
        eprintln!("[CUDA] WSL2 detected — using synchronous cudaMalloc (pool API not supported)");
        force_sync_malloc();
    }
}

pub fn use_sync() -> bool {
    USE_SYNC_MALLOC.load(std::sync::atomic::Ordering::Relaxed)
}

impl<T> DeviceBuffer<T> {
    /// Allocate `len` elements on the GPU (uninitialized).
    /// Uses cudaMallocAsync for pool-based allocation on native systems,
    /// falls back to cudaMalloc on WSL2.
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
        let err = if use_sync() {
            unsafe { ffi::cudaMalloc(&mut ptr, bytes) }
        } else {
            unsafe { ffi::cudaMallocAsync(&mut ptr, bytes, std::ptr::null_mut()) }
        };
        if err != 0 {
            panic!(
                "cudaMalloc failed: error {err} (bytes={bytes}, {:.1}MB requested, sync={})",
                bytes as f64 / 1e6, use_sync()
            );
        }
        Self {
            ptr: ptr as *mut T,
            len,
            _marker: PhantomData,
        }
    }

    /// Upload from a pinned host buffer (faster DMA transfer).
    /// The caller must provide a pinned (page-locked) slice allocated via cudaMallocHost.
    /// SAFETY: `data` must point to pinned host memory.
    pub unsafe fn from_pinned(data: *const T, len: usize) -> Self {
        let buf = Self::alloc(len);
        if len > 0 {
            let bytes = len * std::mem::size_of::<T>();
            let err = unsafe { ffi::cudaMemcpy(
                buf.ptr as *mut c_void,
                data as *const c_void,
                bytes,
                ffi::MEMCPY_H2D,
            ) };
            assert!(err == 0, "cudaMemcpy H2D (pinned) failed: error {err}");
        }
        buf
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

    /// Download GPU buffer to host via pinned staging for maximum PCIe throughput.
    /// ~6-12x faster than to_host() for large buffers (uses DMA instead of CPU paging).
    pub fn to_host_fast(&self) -> Vec<T>
    where
        T: Default + Clone,
    {
        if self.len == 0 {
            return Vec::new();
        }
        let bytes = self.len * std::mem::size_of::<T>();

        // Allocate pinned staging buffer
        let mut pinned: *mut c_void = std::ptr::null_mut();
        let err = unsafe { ffi::cudaMallocHost(&mut pinned, bytes) };
        if err != 0 {
            // Fall back to regular to_host if pinned alloc fails
            return self.to_host();
        }

        // D2H into pinned memory (fast DMA path)
        let err = unsafe {
            ffi::cudaMemcpy(pinned, self.ptr as *const c_void, bytes, ffi::MEMCPY_D2H)
        };
        assert!(err == 0, "cudaMemcpy D2H pinned failed: {err}");

        // Copy from pinned to Vec (memcpy, no page faults)
        let mut host = vec![T::default(); self.len];
        unsafe {
            std::ptr::copy_nonoverlapping(pinned as *const T, host.as_mut_ptr(), self.len);
            ffi::cudaFreeHost(pinned);
        }
        host
    }

    /// Async upload: copy host data to an existing GPU buffer on a stream.
    /// Caller must sync the stream before reading from the buffer.
    pub fn upload_async(&mut self, data: &[T], stream: &crate::cuda::ffi::CudaStream) {
        assert!(data.len() <= self.len);
        if !data.is_empty() {
            let bytes = data.len() * std::mem::size_of::<T>();
            let err = unsafe {
                ffi::cudaMemcpyAsync(
                    self.ptr as *mut c_void,
                    data.as_ptr() as *const c_void,
                    bytes,
                    ffi::MEMCPY_H2D,
                    stream.ptr,
                )
            };
            assert!(err == 0, "cudaMemcpyAsync H2D failed: {err}");
        }
    }

    /// Download GPU buffer directly into a caller-provided slice (no allocation).
    pub fn download_into(&self, dst: &mut [T]) {
        assert!(dst.len() >= self.len, "dst too small: {} < {}", dst.len(), self.len);
        if self.len > 0 {
            let bytes = self.len * std::mem::size_of::<T>();
            let err = unsafe {
                ffi::cudaMemcpy(
                    dst.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    bytes,
                    ffi::MEMCPY_D2H,
                )
            };
            assert!(err == 0, "cudaMemcpy D2H (into) failed: error {err}");
        }
    }

    /// Async download into caller-provided buffer on a stream.
    pub fn download_into_async(&self, dst: &mut [T], stream: &crate::cuda::ffi::CudaStream) {
        assert!(dst.len() >= self.len);
        if self.len > 0 {
            let bytes = self.len * std::mem::size_of::<T>();
            let err = unsafe {
                ffi::cudaMemcpyAsync(
                    dst.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    bytes,
                    ffi::MEMCPY_D2H,
                    stream.ptr,
                )
            };
            assert!(err == 0, "cudaMemcpyAsync D2H failed: {err}");
        }
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

/// Page-locked (pinned) host memory buffer.
/// Enables DMA transfers (faster H2D/D2H) and is required for async memcpy.
/// Reusable across calls to avoid per-batch allocation overhead.
pub struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for PinnedBuffer<T> {}

impl<T> PinnedBuffer<T> {
    /// Allocate `len` elements of pinned host memory.
    pub fn alloc(len: usize) -> Self {
        if len == 0 {
            return Self { ptr: std::ptr::null_mut(), len: 0, _marker: PhantomData };
        }
        let bytes = len * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { ffi::cudaMallocHost(&mut ptr, bytes) };
        assert!(err == 0, "cudaMallocHost failed: {err}");
        Self { ptr: ptr as *mut T, len, _marker: PhantomData }
    }

    /// Grow if needed (realloc).
    pub fn ensure_capacity(&mut self, needed: usize) {
        if self.len >= needed { return; }
        // Free old
        if !self.ptr.is_null() {
            unsafe { ffi::cudaFreeHost(self.ptr as *mut c_void) };
        }
        let bytes = needed * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { ffi::cudaMallocHost(&mut ptr, bytes) };
        assert!(err == 0, "cudaMallocHost failed: {err}");
        self.ptr = ptr as *mut T;
        self.len = needed;
    }

    pub fn as_ptr(&self) -> *const T { self.ptr }
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }

    pub fn as_slice(&self, len: usize) -> &[T] {
        assert!(len <= self.len);
        unsafe { std::slice::from_raw_parts(self.ptr, len) }
    }

    pub fn as_mut_slice(&mut self, len: usize) -> &mut [T] {
        assert!(len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cudaFreeHost(self.ptr as *mut c_void) };
        }
    }
}

impl<T> Clone for DeviceBuffer<T> {
    fn clone(&self) -> Self {
        self.clone_device()
    }
}

impl<T> std::fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("len", &self.len)
            .field("ptr", &self.ptr)
            .finish()
    }
}

impl<T> AsRef<DeviceBuffer<T>> for DeviceBuffer<T> {
    fn as_ref(&self) -> &DeviceBuffer<T> {
        self
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if use_sync() {
                unsafe { ffi::cudaFree(self.ptr as *mut c_void) };
            } else {
                unsafe { ffi::cudaFreeAsync(self.ptr as *mut c_void, std::ptr::null_mut()) };
            }
        }
    }
}
