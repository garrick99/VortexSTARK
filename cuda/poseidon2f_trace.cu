// Poseidon2-Full GPU trace kernel: RF=8 full rounds, RP=0.
// Each CUDA thread computes one full permutation (8 rounds, 8 rows).
//
// Round structure (identical for all 8 rounds):
//   state[j] += RC[r][j]   (add round constants)
//   state[j]  = state[j]^5 (full S-box: all 8 elements)
//   state      = M_E(state) (external MDS: s=sum; out[i]=2*x[i]+s)
//   → write row r to trace
//
// Compared to standard Poseidon2 (30 rows):
//   8 rows instead of 30  →  3.75× more permutations per trace
//   Identical MDS (M_E = circ(3,1,...,1)) and S-box (x^5)
//   No partial rounds, no round-type dispatch in the kernel

#include "include/m31.cuh"

#define P2F_STATE_WIDTH  8
#define P2F_RF           8    // full rounds only
#define P2F_ROWS_PER_PERM 8   // == P2F_RF

// RF × STATE_WIDTH = 64 round constants in GPU constant memory.
__constant__ uint32_t P2F_RC[P2F_RF * P2F_STATE_WIDTH];  // 64 values

// x → x^5: 2 squarings + 1 multiply.
__device__ __forceinline__ uint32_t p2f_sbox(uint32_t x) {
    uint32_t x2 = m31_mul(x, x);
    uint32_t x4 = m31_mul(x2, x2);
    return m31_mul(x4, x);
}

// M_E: circ(3,1,...,1) → s = sum(state); out[i] = 2*state[i] + s.
__device__ __forceinline__ void p2f_mds(uint32_t state[P2F_STATE_WIDTH]) {
    uint32_t s = 0;
    for (int j = 0; j < P2F_STATE_WIDTH; j++) s = m31_add(s, state[j]);
    for (int j = 0; j < P2F_STATE_WIDTH; j++)
        state[j] = m31_add(m31_add(state[j], state[j]), s);
}

// Each thread: one permutation, 8 rounds, writes 8 trace rows.
__global__ void p2f_trace_kernel(
    const uint32_t* __restrict__ block_inputs,
    uint32_t* const* __restrict__ trace_cols,
    uint32_t n_blocks
) {
    uint32_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= n_blocks) return;

    uint32_t state[P2F_STATE_WIDTH];
    const uint32_t* inp = block_inputs + (uint64_t)block_idx * P2F_STATE_WIDTH;
    for (int j = 0; j < P2F_STATE_WIDTH; j++) state[j] = inp[j];

    uint32_t row_base = block_idx * P2F_ROWS_PER_PERM;

    for (int r = 0; r < P2F_RF; r++) {
        const uint32_t* rc = &P2F_RC[r * P2F_STATE_WIDTH];
        for (int j = 0; j < P2F_STATE_WIDTH; j++) state[j] = m31_add(state[j], rc[j]);
        for (int j = 0; j < P2F_STATE_WIDTH; j++) state[j] = p2f_sbox(state[j]);
        p2f_mds(state);
        uint32_t row = row_base + r;
        for (int j = 0; j < P2F_STATE_WIDTH; j++) trace_cols[j][row] = state[j];
    }
}

extern "C" {

void cuda_p2f_upload_consts(const uint32_t* host_rc) {
    cudaMemcpyToSymbol(P2F_RC, host_rc, P2F_RF * P2F_STATE_WIDTH * sizeof(uint32_t));
}

void cuda_p2f_trace(
    const uint32_t* block_inputs,
    uint32_t* const* trace_cols,
    uint32_t n_blocks
) {
    uint32_t threads = 256;
    uint32_t blocks  = (n_blocks + threads - 1) / threads;
    p2f_trace_kernel<<<blocks, threads>>>(block_inputs, trace_cols, n_blocks);
}

} // extern "C"
