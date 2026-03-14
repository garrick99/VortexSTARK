// GPU Poseidon trace generation.
// Each CUDA thread computes one full Poseidon permutation block (22 rounds),
// writing 22 rows × 8 columns to the trace output.
// Block inputs are pre-computed on CPU and uploaded.

#include "include/m31.cuh"

#define POSEIDON_STATE_WIDTH 8
#define POSEIDON_NUM_ROUNDS 22

// MDS first row (circulant)
__constant__ uint32_t TRACE_MDS_ROW[POSEIDON_STATE_WIDTH] = {3, 1, 1, 1, 1, 1, 1, 2};

// Round constants in constant memory (22 * 8 = 176 values, fits easily)
__constant__ uint32_t TRACE_ROUND_CONSTS[POSEIDON_NUM_ROUNDS * POSEIDON_STATE_WIDTH];

__device__ __forceinline__ uint32_t trace_sbox(uint32_t x) {
    uint32_t x2 = m31_mul(x, x);
    uint32_t x4 = m31_mul(x2, x2);
    return m31_mul(x4, x);
}

__device__ void trace_mds(const uint32_t in[POSEIDON_STATE_WIDTH],
                           uint32_t out[POSEIDON_STATE_WIDTH]) {
    for (int i = 0; i < POSEIDON_STATE_WIDTH; i++) {
        uint32_t acc = 0;
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            uint32_t mds_val = TRACE_MDS_ROW[(POSEIDON_STATE_WIDTH + j - i) % POSEIDON_STATE_WIDTH];
            acc = m31_add(acc, m31_mul(mds_val, in[j]));
        }
        out[i] = acc;
    }
}

// Each thread computes one full Poseidon block.
// block_inputs: [n_blocks * STATE_WIDTH] — flattened input states
// trace_cols: [STATE_WIDTH] pointers to output columns, each of length n_blocks * NUM_ROUNDS
__global__ void poseidon_trace_kernel(
    const uint32_t* __restrict__ block_inputs,  // [n_blocks * 8] flattened
    uint32_t* const* __restrict__ trace_cols,   // [8] column pointers
    uint32_t n_blocks
) {
    uint32_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= n_blocks) return;

    // Load block input
    uint32_t state[POSEIDON_STATE_WIDTH];
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
        state[j] = block_inputs[block_idx * POSEIDON_STATE_WIDTH + j];
    }

    uint32_t row_base = block_idx * POSEIDON_NUM_ROUNDS;

    for (int r = 0; r < POSEIDON_NUM_ROUNDS; r++) {
        // Add round constants
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            state[j] = m31_add(state[j], TRACE_ROUND_CONSTS[r * POSEIDON_STATE_WIDTH + j]);
        }

        // S-box: x -> x^5
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            state[j] = trace_sbox(state[j]);
        }

        // MDS
        uint32_t tmp[POSEIDON_STATE_WIDTH];
        trace_mds(state, tmp);
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            state[j] = tmp[j];
        }

        // Write row to trace columns
        uint32_t row = row_base + r;
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            trace_cols[j][row] = state[j];
        }
    }
}

extern "C" {

// Upload round constants to GPU constant memory.
void cuda_poseidon_upload_round_consts(const uint32_t* host_rc) {
    cudaMemcpyToSymbol(TRACE_ROUND_CONSTS, host_rc,
        POSEIDON_NUM_ROUNDS * POSEIDON_STATE_WIDTH * sizeof(uint32_t));
}

// Generate Poseidon trace on GPU.
// block_inputs: [n_blocks * 8] on device — pre-computed block input states
// trace_cols: [8] device pointers to output columns, each n_blocks * NUM_ROUNDS elements
void cuda_poseidon_trace(
    const uint32_t* block_inputs,
    uint32_t* const* trace_cols,
    uint32_t n_blocks
) {
    uint32_t threads = 256;
    uint32_t blocks = (n_blocks + threads - 1) / threads;
    poseidon_trace_kernel<<<blocks, threads>>>(block_inputs, trace_cols, n_blocks);
}

} // extern "C"
