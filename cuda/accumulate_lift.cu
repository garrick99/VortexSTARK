// GPU kernel for AccumulationOps::lift_and_accumulate.
//
// Implements one iteration of the lift-and-accumulate loop (stwo CPU impl):
//
//   log_ratio = col.len().ilog2() - curr.len().ilog2()
//   for i in 0..col.len():
//       src_idx = (i >> (log_ratio + 1) << 1) | (i & 1)
//       col[i] += curr[src_idx]
//
// GPU kernel: one thread per element of `col` (size col_n).
// Both `col` and `curr` are QM31 arrays (4 u32 per element).
// The update is in-place: col[i] += curr[src_idx].

#include "include/qm31.cuh"

__global__ void accumulate_lift_kernel(
    uint32_t* __restrict__ col,         // in/out: col_n QM31 elements
    const uint32_t* __restrict__ curr,  // read-only: curr_n QM31 elements
    uint32_t col_n,
    uint32_t log_ratio                  // = log2(col_n) - log2(curr_n)
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= col_n) return;

    // src_idx = (i >> (log_ratio + 1) << 1) | (i & 1)
    uint32_t shift = log_ratio + 1;
    uint32_t src_idx = ((i >> shift) << 1) | (i & 1u);

    QM31 col_val  = {{col[4*i],         col[4*i+1],         col[4*i+2],         col[4*i+3]}};
    QM31 curr_val = {{curr[4*src_idx],   curr[4*src_idx+1],  curr[4*src_idx+2],  curr[4*src_idx+3]}};
    QM31 result   = qm31_add(col_val, curr_val);

    col[4*i + 0] = result.v[0];
    col[4*i + 1] = result.v[1];
    col[4*i + 2] = result.v[2];
    col[4*i + 3] = result.v[3];
}

extern "C" {

void cuda_accumulate_lift(
    uint32_t* col,
    const uint32_t* curr,
    uint32_t col_n,
    uint32_t log_ratio
) {
    uint32_t blocks = (col_n + 255) / 256;
    accumulate_lift_kernel<<<blocks, 256>>>(col, curr, col_n, log_ratio);
}

} // extern "C"
