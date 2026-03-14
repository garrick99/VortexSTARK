// GPU constraint evaluation for Poseidon AIR over M31.
//
// State width: 8, S-box: x^5, MDS: 8x8 circulant.
// Evaluates STATE_WIDTH transition constraints at each row and combines
// with random alpha coefficients into a single QM31 quotient value.

#include "include/qm31.cuh"

#define POSEIDON_STATE_WIDTH 8
#define POSEIDON_NUM_ROUNDS 22

// MDS first row (circulant matrix)
__constant__ uint32_t MDS_ROW[POSEIDON_STATE_WIDTH] = {3, 1, 1, 1, 1, 1, 1, 2};

// S-box: x -> x^5
__device__ __forceinline__ uint32_t sbox(uint32_t x) {
    uint32_t x2 = m31_mul(x, x);
    uint32_t x4 = m31_mul(x2, x2);
    return m31_mul(x4, x);
}

// MDS matrix-vector multiply (circulant)
__device__ void mds_multiply(const uint32_t state[POSEIDON_STATE_WIDTH],
                              uint32_t out[POSEIDON_STATE_WIDTH]) {
    for (int i = 0; i < POSEIDON_STATE_WIDTH; i++) {
        uint32_t acc = 0;
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            uint32_t mds_val = MDS_ROW[(POSEIDON_STATE_WIDTH + j - i) % POSEIDON_STATE_WIDTH];
            acc = m31_add(acc, m31_mul(mds_val, state[j]));
        }
        out[i] = acc;
    }
}

// Poseidon transition constraint kernel.
// For each row i, checks: trace[i+1] = MDS(sbox(trace[i] + round_const[round]))
// Combines STATE_WIDTH constraints with alpha coefficients into one QM31.
__global__ void poseidon_quotient_kernel(
    const uint32_t* const* __restrict__ trace_cols,  // [STATE_WIDTH] column pointers
    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2, uint32_t* __restrict__ out3,
    const uint32_t* __restrict__ round_consts,  // [NUM_ROUNDS * STATE_WIDTH] flattened
    const uint32_t* __restrict__ alpha_coeffs,  // [STATE_WIDTH * 4] QM31 coefficients
    uint32_t n  // eval domain size
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t next_i = (i + 1) % n;
    uint32_t round = i % POSEIDON_NUM_ROUNDS;
    uint32_t next_round = (round + 1) % POSEIDON_NUM_ROUNDS;

    // Load current state + add round constants for the NEXT round
    uint32_t state[POSEIDON_STATE_WIDTH];
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
        state[j] = m31_add(trace_cols[j][i], round_consts[next_round * POSEIDON_STATE_WIDTH + j]);
    }

    // S-box
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
        state[j] = sbox(state[j]);
    }

    // MDS
    uint32_t expected[POSEIDON_STATE_WIDTH];
    mds_multiply(state, expected);

    // Constraint: next_state - expected for each column
    // Combine with alpha coefficients: Q = sum_j alpha_j * C_j
    QM31 result = {{0, 0, 0, 0}};
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
        uint32_t actual_next = trace_cols[j][next_i];
        uint32_t constraint = m31_sub(actual_next, expected[j]);

        QM31 alpha_j = {{
            alpha_coeffs[j * 4 + 0], alpha_coeffs[j * 4 + 1],
            alpha_coeffs[j * 4 + 2], alpha_coeffs[j * 4 + 3]
        }};
        result = qm31_add(result, qm31_mul_m31(alpha_j, constraint));
    }

    out0[i] = result.v[0];
    out1[i] = result.v[1];
    out2[i] = result.v[2];
    out3[i] = result.v[3];
}

// Chunked variant for streaming quotient computation.
__global__ void poseidon_quotient_chunk_kernel(
    const uint32_t* const* __restrict__ trace_cols,
    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2, uint32_t* __restrict__ out3,
    const uint32_t* __restrict__ round_consts,
    const uint32_t* __restrict__ alpha_coeffs,
    uint32_t offset, uint32_t chunk_n, uint32_t global_n
) {
    uint32_t local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i >= chunk_n) return;

    uint32_t i = offset + local_i;
    uint32_t next_i = (i + 1) % global_n;
    uint32_t next_round = ((i % POSEIDON_NUM_ROUNDS) + 1) % POSEIDON_NUM_ROUNDS;

    uint32_t state[POSEIDON_STATE_WIDTH];
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
        state[j] = m31_add(trace_cols[j][i], round_consts[next_round * POSEIDON_STATE_WIDTH + j]);
    }

    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
        state[j] = sbox(state[j]);
    }

    uint32_t expected[POSEIDON_STATE_WIDTH];
    mds_multiply(state, expected);

    QM31 result = {{0, 0, 0, 0}};
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
        uint32_t actual_next = trace_cols[j][next_i];
        uint32_t constraint = m31_sub(actual_next, expected[j]);
        QM31 alpha_j = {{
            alpha_coeffs[j * 4 + 0], alpha_coeffs[j * 4 + 1],
            alpha_coeffs[j * 4 + 2], alpha_coeffs[j * 4 + 3]
        }};
        result = qm31_add(result, qm31_mul_m31(alpha_j, constraint));
    }

    out0[local_i] = result.v[0];
    out1[local_i] = result.v[1];
    out2[local_i] = result.v[2];
    out3[local_i] = result.v[3];
}

extern "C" {

void cuda_poseidon_quotient(
    const uint32_t* const* trace_cols,
    uint32_t* out0, uint32_t* out1, uint32_t* out2, uint32_t* out3,
    const uint32_t* round_consts,
    const uint32_t* alpha_coeffs,
    uint32_t n
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    poseidon_quotient_kernel<<<blocks, threads>>>(
        trace_cols, out0, out1, out2, out3, round_consts, alpha_coeffs, n
    );
}

void cuda_poseidon_quotient_chunk(
    const uint32_t* const* trace_cols,
    uint32_t* out0, uint32_t* out1, uint32_t* out2, uint32_t* out3,
    const uint32_t* round_consts,
    const uint32_t* alpha_coeffs,
    uint32_t offset, uint32_t chunk_n, uint32_t global_n
) {
    uint32_t threads = 256;
    uint32_t blocks = (chunk_n + threads - 1) / threads;
    poseidon_quotient_chunk_kernel<<<blocks, threads>>>(
        trace_cols, out0, out1, out2, out3, round_consts, alpha_coeffs,
        offset, chunk_n, global_n
    );
}

} // extern "C"
