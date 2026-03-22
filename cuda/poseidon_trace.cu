// GPU Poseidon2 trace generation.
// Each CUDA thread computes one full Poseidon2 permutation block (30 rounds):
//   4 full rounds (M_E) → 22 partial rounds (M_I, only x[0] S-box) → 4 full rounds (M_E)
// Writes 30 rows × 8 columns to the trace output.

#include "include/m31.cuh"

#define POSEIDON_STATE_WIDTH 8
#define POSEIDON_RF_BEFORE   4
#define POSEIDON_RP         22
#define POSEIDON_RF_AFTER    4
#define POSEIDON_RF          8   // RF_BEFORE + RF_AFTER
#define POSEIDON_NUM_ROUNDS 30   // RF + RP

// Full round constants: RF * STATE_WIDTH = 64 values
// Layout: full_rcs[r][j] = FULL_RCONSTS[r * STATE_WIDTH + j]
// First RF_BEFORE rounds are "begin" full rounds, next RF_AFTER are "end" full rounds.
__constant__ uint32_t FULL_RCONSTS[POSEIDON_RF * POSEIDON_STATE_WIDTH];  // 64 values

// Partial round constants: RP = 22 values (applied to x[0] only)
__constant__ uint32_t PARTIAL_RCONSTS[POSEIDON_RP];  // 22 values

__device__ __forceinline__ uint32_t p2_sbox(uint32_t x) {
    uint32_t x2 = m31_mul(x, x);
    uint32_t x4 = m31_mul(x2, x2);
    return m31_mul(x4, x);
}

// M_E: circ(3,1,...,1): s = sum(state); out[i] = 2*state[i] + s
__device__ __forceinline__ void m_ext(uint32_t state[POSEIDON_STATE_WIDTH]) {
    uint32_t s = 0;
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) s = m31_add(s, state[j]);
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
        state[j] = m31_add(m31_add(state[j], state[j]), s);
    }
}

// M_I: circ(2,1,...,1): s = sum(state); out[i] = state[i] + s
__device__ __forceinline__ void m_int(uint32_t state[POSEIDON_STATE_WIDTH]) {
    uint32_t s = 0;
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) s = m31_add(s, state[j]);
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
        state[j] = m31_add(state[j], s);
    }
}

// Each thread computes one full Poseidon2 block (30 rounds).
// block_inputs: [n_blocks * STATE_WIDTH] flattened input states
// trace_cols:   [STATE_WIDTH] pointers to output columns, each n_blocks * NUM_ROUNDS elements
__global__ void poseidon2_trace_kernel(
    const uint32_t* __restrict__ block_inputs,
    uint32_t* const* __restrict__ trace_cols,
    uint32_t n_blocks
) {
    uint32_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= n_blocks) return;

    uint32_t state[POSEIDON_STATE_WIDTH];
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
        state[j] = block_inputs[block_idx * POSEIDON_STATE_WIDTH + j];
    }

    uint32_t row_base = block_idx * POSEIDON_NUM_ROUNDS;

    // Phase 1: RF_BEFORE full rounds
    for (int r = 0; r < POSEIDON_RF_BEFORE; r++) {
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            state[j] = m31_add(state[j], FULL_RCONSTS[r * POSEIDON_STATE_WIDTH + j]);
        }
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            state[j] = p2_sbox(state[j]);
        }
        m_ext(state);
        uint32_t row = row_base + r;
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            trace_cols[j][row] = state[j];
        }
    }

    // Phase 2: RP partial rounds
    for (int r = 0; r < POSEIDON_RP; r++) {
        state[0] = m31_add(state[0], PARTIAL_RCONSTS[r]);
        state[0] = p2_sbox(state[0]);
        m_int(state);
        uint32_t row = row_base + POSEIDON_RF_BEFORE + r;
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            trace_cols[j][row] = state[j];
        }
    }

    // Phase 3: RF_AFTER full rounds
    for (int r = 0; r < POSEIDON_RF_AFTER; r++) {
        int full_r = POSEIDON_RF_BEFORE + r;
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            state[j] = m31_add(state[j], FULL_RCONSTS[full_r * POSEIDON_STATE_WIDTH + j]);
        }
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            state[j] = p2_sbox(state[j]);
        }
        m_ext(state);
        uint32_t row = row_base + POSEIDON_RF_BEFORE + POSEIDON_RP + r;
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            trace_cols[j][row] = state[j];
        }
    }
}

extern "C" {

// Upload round constants to GPU constant memory.
// host_rc layout: [full_rcs: RF*STATE_WIDTH=64, partial_rcs: RP=22] = 86 total values
void cuda_poseidon_upload_round_consts(const uint32_t* host_rc) {
    cudaMemcpyToSymbol(FULL_RCONSTS, host_rc,
        POSEIDON_RF * POSEIDON_STATE_WIDTH * sizeof(uint32_t));
    cudaMemcpyToSymbol(PARTIAL_RCONSTS, host_rc + POSEIDON_RF * POSEIDON_STATE_WIDTH,
        POSEIDON_RP * sizeof(uint32_t));
}

// Generate Poseidon2 trace on GPU.
void cuda_poseidon_trace(
    const uint32_t* block_inputs,
    uint32_t* const* trace_cols,
    uint32_t n_blocks
) {
    uint32_t threads = 256;
    uint32_t blocks = (n_blocks + threads - 1) / threads;
    poseidon2_trace_kernel<<<blocks, threads>>>(block_inputs, trace_cols, n_blocks);
}

} // extern "C"
