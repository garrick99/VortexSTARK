// Batched gather kernels for CudaColumn decommitment.
//
// Replaces the O(N) individual cudaMemcpy calls in MerkleProverLifted::decommit
// (one per query per column) with a single kernel launch + transfer per column.

#include <stdint.h>

// Gather u32 elements: dst[i] = src[idx[i]]
__global__ void gather_u32_kernel(
    const uint32_t* __restrict__ src,
    const uint32_t* __restrict__ idx,
    uint32_t* __restrict__ dst,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

// Gather 8-word (256-bit) elements: dst[i*8..i*8+8] = src[idx[i]*8..idx[i]*8+8]
// Used for Blake2sHash columns (each hash = 32 bytes = 8 u32s).
__global__ void gather_u256_kernel(
    const uint32_t* __restrict__ src,
    const uint32_t* __restrict__ idx,
    uint32_t* __restrict__ dst,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t src_off = idx[i] * 8;
        uint32_t dst_off = i * 8;
        dst[dst_off + 0] = src[src_off + 0];
        dst[dst_off + 1] = src[src_off + 1];
        dst[dst_off + 2] = src[src_off + 2];
        dst[dst_off + 3] = src[src_off + 3];
        dst[dst_off + 4] = src[src_off + 4];
        dst[dst_off + 5] = src[src_off + 5];
        dst[dst_off + 6] = src[src_off + 6];
        dst[dst_off + 7] = src[src_off + 7];
    }
}

extern "C" {

void cuda_gather_u32(
    const uint32_t* src,
    const uint32_t* idx,
    uint32_t* dst,
    uint32_t n
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    gather_u32_kernel<<<blocks, threads>>>(src, idx, dst, n);
}

void cuda_gather_u256(
    const uint32_t* src,
    const uint32_t* idx,
    uint32_t* dst,
    uint32_t n
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    gather_u256_kernel<<<blocks, threads>>>(src, idx, dst, n);
}

} // extern "C"
