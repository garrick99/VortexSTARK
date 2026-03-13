#include "include/qm31.cuh"

// Basic field operation kernels for testing and general use.

// Elementwise M31 add
__global__ void m31_add_kernel(const uint32_t* a, const uint32_t* b, uint32_t* out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = m31_add(a[i], b[i]);
}

// Elementwise M31 mul
__global__ void m31_mul_kernel(const uint32_t* a, const uint32_t* b, uint32_t* out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = m31_mul(a[i], b[i]);
}

// Batch M31 inverse using Montgomery's trick
// Product tree → invert single element → unwind
__global__ void batch_inverse_pass1_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ prefix,
    uint32_t n
) {
    // Single-thread prefix product (for small n, or per-block for large n)
    // For MVP, run this on CPU side. GPU batch inverse comes in Phase 2.
}

extern "C" {

void cuda_m31_add(const uint32_t* a, const uint32_t* b, uint32_t* out, uint32_t n) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    m31_add_kernel<<<blocks, threads>>>(a, b, out, n);
}

void cuda_m31_mul(const uint32_t* a, const uint32_t* b, uint32_t* out, uint32_t n) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    m31_mul_kernel<<<blocks, threads>>>(a, b, out, n);
}

void cuda_device_sync() {
    cudaDeviceSynchronize();
}

} // extern "C"
