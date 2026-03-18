// GPU barycentric evaluation: result = sum_i(evals[i] * weights[i])
//
// evals:   n M31 values (1 u32 each, in bit-reversed circle evaluation order)
// weights: n QM31 values in AoS layout: [w[0].v[0], w[0].v[1], w[0].v[2], w[0].v[3], w[1].v[0], ...]
// output:  [n_blocks * 4] QM31 partial sums — caller reduces on CPU
//
// This replaces the CPU fallback in poly_ops.rs::barycentric_eval_at_point.
// For n = 2^22, this avoids downloading 4+16 MB from GPU and saves ~5ms.
//
// The caller launches at most MAX_BLOCKS blocks and does a trivial final
// reduction over the partial sums on CPU.

#include "include/m31.cuh"
#include "include/qm31.cuh"

// Warp-level QM31 reduction using shuffle.
__device__ __forceinline__
QM31 warp_reduce_qm31(QM31 val) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        QM31 other;
        other.v[0] = __shfl_xor_sync(0xFFFFFFFFu, val.v[0], mask);
        other.v[1] = __shfl_xor_sync(0xFFFFFFFFu, val.v[1], mask);
        other.v[2] = __shfl_xor_sync(0xFFFFFFFFu, val.v[2], mask);
        other.v[3] = __shfl_xor_sync(0xFFFFFFFFu, val.v[3], mask);
        val = qm31_add(val, other);
    }
    return val;
}

__global__ void barycentric_eval_kernel(
    const uint32_t* __restrict__ evals,    // n M31 values
    const uint32_t* __restrict__ weights,  // n * 4 u32 (QM31 AoS)
    uint32_t n,
    uint32_t* __restrict__ out             // n_blocks * 4 QM31 partial sums
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    // Thread-local accumulator
    QM31 local = qm31_zero();

    // Grid-stride loop over elements
    for (uint32_t i = tid; i < n; i += stride) {
        uint32_t e = evals[i];
        uint32_t base = i * 4;
        QM31 w = {{weights[base], weights[base+1], weights[base+2], weights[base+3]}};
        local = qm31_add(local, qm31_mul_m31(w, e));
    }

    // Warp-level reduction
    local = warp_reduce_qm31(local);

    // Block-level reduction via shared memory (one slot per warp)
    __shared__ QM31 shmem[32]; // supports up to 1024 threads (32 warps)
    int warp_id = threadIdx.x >> 5;   // threadIdx.x / 32
    int lane_id = threadIdx.x & 0x1F; // threadIdx.x % 32

    if (lane_id == 0) {
        shmem[warp_id] = local;
    }
    __syncthreads();

    // Only the first warp does the final block reduction
    if (warp_id == 0) {
        int n_warps = blockDim.x >> 5;
        QM31 val = (lane_id < n_warps) ? shmem[lane_id] : qm31_zero();
        val = warp_reduce_qm31(val);
        if (lane_id == 0) {
            uint32_t base = blockIdx.x * 4;
            out[base + 0] = val.v[0];
            out[base + 1] = val.v[1];
            out[base + 2] = val.v[2];
            out[base + 3] = val.v[3];
        }
    }
}

extern "C" {

/// Compute sum_i(evals[i] * weights[i]) using a parallel reduction.
///
/// Writes `n_blocks` QM31 partial sums to `out`. Caller must final-reduce on CPU.
/// `n_blocks` = the number of blocks launched (caller allocates out[n_blocks * 4]).
void cuda_barycentric_eval(
    const uint32_t* evals,
    const uint32_t* weights,
    uint32_t n,
    uint32_t* out,
    uint32_t n_blocks
) {
    uint32_t threads = 256;
    barycentric_eval_kernel<<<n_blocks, threads>>>(evals, weights, n, out);
    cudaDeviceSynchronize();
}

} // extern "C"
