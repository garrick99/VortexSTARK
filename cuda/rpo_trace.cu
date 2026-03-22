// GPU RPO-M31 trace generation.
// Paper: "RPO-M31 and XHash-M31: Efficient Hash Functions for Circle STARKs"
// (eprint 2024/1635, Ashur & Tariq).
//
// Each CUDA thread computes one full RPO-M31 permutation block (7 rounds):
//   For r = 0..7:
//     FM: state = x^5(MDS(state) + RC_FM_r)      → write trace row 2r
//     BM: state = x^(1/5)(MDS(state) + RC_BM_r)  → write trace row 2r+1
//   CLS: state = MDS(state) + RC_CLS              (no trace row)
//
// Trace layout: STATE_WIDTH columns, each with n_blocks * ROWS_PER_PERM elements.
// Row base for block b: b * ROWS_PER_PERM. Row 2r = FM output, row 2r+1 = BM output.
//
// Constants:
//   MDS_FLAT: STATE_WIDTH × STATE_WIDTH = 576 u32 (row-major)
//   RPO_RC:   NUM_STEPS × STATE_WIDTH = 360 u32
//   Layout: [FM_0(24), BM_0(24), FM_1(24), BM_1(24), ..., FM_6(24), BM_6(24), CLS(24)]

#include "include/m31.cuh"

#define RPO_STATE_WIDTH  24
#define RPO_ROUNDS        7
#define RPO_ROWS_PER_PERM 14  // RPO_ROUNDS * 2
#define RPO_NUM_STEPS    15   // RPO_ROUNDS * 2 + 1 (includes CLS)

// GPU constant memory: MDS matrix (row-major, 576 u32) and round constants (360 u32).
__constant__ uint32_t MDS_FLAT[RPO_STATE_WIDTH * RPO_STATE_WIDTH];  // 576 values
__constant__ uint32_t RPO_RC[RPO_NUM_STEPS * RPO_STATE_WIDTH];      // 360 values

// Apply the 24×24 MDS matrix in-place using circulant structure.
// Sum is accumulated in 64-bit to avoid overflow, then reduced to M31.
__device__ __forceinline__ void rpo_mds(uint32_t state[RPO_STATE_WIDTH]) {
    uint32_t tmp[RPO_STATE_WIDTH];
    for (int i = 0; i < RPO_STATE_WIDTH; i++) {
        uint64_t acc = 0;
        const uint32_t* row = &MDS_FLAT[i * RPO_STATE_WIDTH];
        for (int j = 0; j < RPO_STATE_WIDTH; j++) {
            // m31_mul result < p < 2^31; 24 such terms < 24*2^31 < 2^36 — safe in uint64_t.
            acc += m31_mul(row[j], state[j]);
        }
        tmp[i] = m31_reduce(acc);
    }
    for (int i = 0; i < RPO_STATE_WIDTH; i++) state[i] = tmp[i];
}

// Forward S-box: x → x^5. Cost: 2 squarings + 1 multiply.
__device__ __forceinline__ uint32_t rpo_sbox_fwd(uint32_t x) {
    uint32_t x2 = m31_mul(x, x);
    uint32_t x4 = m31_mul(x2, x2);
    return m31_mul(x4, x);
}

// Inverse S-box: x → x^1717986917 = x^(5^{-1} mod (p-1)).
// 1717986917 = 0x66666665 in binary: ~30 squarings, ~16 multiplies.
__device__ __forceinline__ uint32_t rpo_sbox_inv(uint32_t x) {
    uint32_t result = 1u;
    uint32_t base = x;
    uint32_t exp = 1717986917u;
    while (exp > 0) {
        if (exp & 1u) result = m31_mul(result, base);
        base = m31_mul(base, base);
        exp >>= 1;
    }
    return result;
}

// Main kernel: each thread computes one RPO-M31 permutation and writes 14 trace rows.
// block_inputs: [n_blocks * STATE_WIDTH] flattened input states
// trace_cols:   [STATE_WIDTH] pointers to output columns, each n_blocks * ROWS_PER_PERM elements
__global__ void rpo_trace_kernel(
    const uint32_t* __restrict__ block_inputs,
    uint32_t* const* __restrict__ trace_cols,
    uint32_t n_blocks
) {
    uint32_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= n_blocks) return;

    // Load input state into registers.
    uint32_t state[RPO_STATE_WIDTH];
    const uint32_t* inp = block_inputs + (uint64_t)block_idx * RPO_STATE_WIDTH;
    for (int j = 0; j < RPO_STATE_WIDTH; j++) state[j] = inp[j];

    uint32_t row_base = block_idx * RPO_ROWS_PER_PERM;
    int rc_idx = 0;

    for (int r = 0; r < RPO_ROUNDS; r++) {
        // ── FM step: MDS → add RC → x^5 ──────────────────────────────────
        rpo_mds(state);
        const uint32_t* rc_fm = &RPO_RC[rc_idx * RPO_STATE_WIDTH];
        for (int j = 0; j < RPO_STATE_WIDTH; j++) state[j] = m31_add(state[j], rc_fm[j]);
        for (int j = 0; j < RPO_STATE_WIDTH; j++) state[j] = rpo_sbox_fwd(state[j]);
        rc_idx++;

        // Write FM trace row.
        uint32_t fm_row = row_base + 2 * r;
        for (int j = 0; j < RPO_STATE_WIDTH; j++) trace_cols[j][fm_row] = state[j];

        // ── BM step: MDS → add RC → x^(1/5) ─────────────────────────────
        rpo_mds(state);
        const uint32_t* rc_bm = &RPO_RC[rc_idx * RPO_STATE_WIDTH];
        for (int j = 0; j < RPO_STATE_WIDTH; j++) state[j] = m31_add(state[j], rc_bm[j]);
        for (int j = 0; j < RPO_STATE_WIDTH; j++) state[j] = rpo_sbox_inv(state[j]);
        rc_idx++;

        // Write BM trace row.
        uint32_t bm_row = row_base + 2 * r + 1;
        for (int j = 0; j < RPO_STATE_WIDTH; j++) trace_cols[j][bm_row] = state[j];
    }

    // ── CLS step: MDS → add RC (no trace row) ───────────────────────────
    rpo_mds(state);
    const uint32_t* rc_cls = &RPO_RC[rc_idx * RPO_STATE_WIDTH];
    for (int j = 0; j < RPO_STATE_WIDTH; j++) state[j] = m31_add(state[j], rc_cls[j]);
    // Final state available but not written to trace.
    (void)state;
}

extern "C" {

// Upload MDS matrix (576 u32, row-major) and round constants (360 u32) to GPU constant memory.
// Call once before running the trace kernel.
void cuda_rpo_upload_constants(const uint32_t* host_mds, const uint32_t* host_rc) {
    cudaMemcpyToSymbol(MDS_FLAT, host_mds,
        RPO_STATE_WIDTH * RPO_STATE_WIDTH * sizeof(uint32_t));
    cudaMemcpyToSymbol(RPO_RC, host_rc,
        RPO_NUM_STEPS * RPO_STATE_WIDTH * sizeof(uint32_t));
}

// Generate RPO-M31 trace on GPU.
void cuda_rpo_trace(
    const uint32_t* block_inputs,
    uint32_t* const* trace_cols,
    uint32_t n_blocks
) {
    uint32_t threads = 128;
    uint32_t blocks  = (n_blocks + threads - 1) / threads;
    rpo_trace_kernel<<<blocks, threads>>>(block_inputs, trace_cols, n_blocks);
}

} // extern "C"
