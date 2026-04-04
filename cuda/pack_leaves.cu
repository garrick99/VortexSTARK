// GPU kernel for PackLeavesOps::pack_leaves_input.
//
// Transposes 4 input columns (BaseField/M31, size N each) into
// 64 output columns (BaseField/M31, size N/16 each).
//
// From stwo CPU impl:
//   output_col_idx = coord * PACKED_LEAF_SIZE + leaf_pos  (c in 0..4, p in 0..16)
//   output[c*16+p][i] = input[c][i*16 + p]
//
// GPU mapping (one thread per element of each output column):
//   total threads = N (covers all coord/pos combinations via flat indexing)
//   thread idx maps to: coord = idx / N, in_pos = idx % N
//   out_col = coord * 16 + (in_pos % 16)
//   out_pos = in_pos / 16
//   output_cols[out_col][out_pos] = input_cols[coord][in_pos]

#include <stdint.h>

#define PACKED_LEAF_SIZE 16u
#define SECURE_EXTENSION_DEGREE 4u

__global__ void pack_leaves_kernel(
    const uint32_t* const* __restrict__ input_cols,   // [4] device pointers to M31 columns
    uint32_t* const* __restrict__ output_cols,         // [64] device pointers to output M31 columns
    uint32_t N                                          // input column length (must be multiple of 16)
) {
    // One thread per (coord, in_pos) pair across all 4 input columns
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= SECURE_EXTENSION_DEGREE * N) return;

    uint32_t coord  = global_idx / N;
    uint32_t in_pos = global_idx % N;

    uint32_t out_col = coord * PACKED_LEAF_SIZE + (in_pos % PACKED_LEAF_SIZE);
    uint32_t out_pos = in_pos / PACKED_LEAF_SIZE;

    output_cols[out_col][out_pos] = input_cols[coord][in_pos];
}

extern "C" {

// input_col_ptrs:  device pointer to array of 4 device pointers (M31 columns, size N each)
// output_col_ptrs: device pointer to array of 64 device pointers (M31 columns, size N/16 each)
void cuda_pack_leaves(
    const uint32_t* const* input_col_ptrs,
    uint32_t* const* output_col_ptrs,
    uint32_t N
) {
    uint32_t total = SECURE_EXTENSION_DEGREE * N;
    uint32_t blocks = (total + 255) / 256;
    pack_leaves_kernel<<<blocks, 256>>>(input_col_ptrs, output_col_ptrs, N);
}

} // extern "C"
