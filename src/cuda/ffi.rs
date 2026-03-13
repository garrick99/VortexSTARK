//! Raw FFI bindings to CUDA runtime and kraken-stark kernels.

use std::ffi::c_void;

// CUDA runtime
unsafe extern "C" {
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(dev_ptr: *mut c_void) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    pub fn cudaMemset(dev_ptr: *mut c_void, value: i32, count: usize) -> i32;
    pub fn cudaDeviceSynchronize() -> i32;
    pub fn cudaGetLastError() -> i32;
}

// cudaMemcpyKind
pub const MEMCPY_H2D: i32 = 1;
pub const MEMCPY_D2H: i32 = 2;
pub const MEMCPY_D2D: i32 = 3;

// Our kernels
unsafe extern "C" {
    pub fn cuda_m31_add(a: *const u32, b: *const u32, out: *mut u32, n: u32);
    pub fn cuda_m31_mul(a: *const u32, b: *const u32, out: *mut u32, n: u32);
    pub fn cuda_device_sync();
}
