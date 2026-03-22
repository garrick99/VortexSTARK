// GPU constraint evaluation for Poseidon2 AIR over M31.
//
// State width: 8, S-box: x^5
// Full rounds (M_E):    4 at start + 4 at end, all 8 elements get S-box
// Partial rounds (M_I): 22 in middle, only x[0] gets S-box
// Total: 30 rounds per permutation block.
//
// Transition constraint at row i:
//   next_round = (i % NUM_ROUNDS + 1) % NUM_ROUNDS
//   if next_round is full (< RF_BEFORE or >= RF_BEFORE+RP):
//     s[j] = trace[i][j] + full_rc[full_r][j]  for all j
//     s[j] = s[j]^5                              for all j
//     expected = M_E(s)
//   if next_round is partial:
//     s[0] = trace[i][0] + partial_rc[p]; s[j>0] = trace[i][j]
//     s[0] = s[0]^5
//     expected = M_I(s)
//   constraint[j] = trace[i+1][j] - expected[j]
//
// round_consts layout: [full_rcs: RF*8=64, partial_rcs: RP=22] = 86 total values
// (same flat layout as produced by poseidon::round_constants_flat() in Rust)

#include "include/qm31.cuh"

#define POSEIDON_STATE_WIDTH 8
#define POSEIDON_RF_BEFORE   4
#define POSEIDON_RP         22
#define POSEIDON_RF_AFTER    4
#define POSEIDON_RF          8
#define POSEIDON_NUM_ROUNDS 30

__device__ __forceinline__ uint32_t p2_sbox_c(uint32_t x) {
    uint32_t x2 = m31_mul(x, x);
    uint32_t x4 = m31_mul(x2, x2);
    return m31_mul(x4, x);
}

// M_E: circ(3,1,...,1): s = sum(state); out[i] = 2*state[i] + s
__device__ __forceinline__ void m_ext_c(uint32_t state[POSEIDON_STATE_WIDTH]) {
    uint32_t s = 0;
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) s = m31_add(s, state[j]);
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
        state[j] = m31_add(m31_add(state[j], state[j]), s);
    }
}

// M_I: circ(2,1,...,1): s = sum(state); out[i] = state[i] + s
__device__ __forceinline__ void m_int_c(uint32_t state[POSEIDON_STATE_WIDTH]) {
    uint32_t s = 0;
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) s = m31_add(s, state[j]);
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
        state[j] = m31_add(state[j], s);
    }
}

// Read full round constant for round full_r (0..RF), element j.
// full_rcs start at round_consts[0].
__device__ __forceinline__ uint32_t full_rc(const uint32_t* round_consts, int full_r, int j) {
    return round_consts[full_r * POSEIDON_STATE_WIDTH + j];
}

// Read partial round constant for partial round p (0..RP).
// partial_rcs start at round_consts[RF * STATE_WIDTH].
__device__ __forceinline__ uint32_t partial_rc(const uint32_t* round_consts, int p) {
    return round_consts[POSEIDON_RF * POSEIDON_STATE_WIDTH + p];
}

__global__ void poseidon2_quotient_kernel(
    const uint32_t* const* __restrict__ trace_cols,
    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2, uint32_t* __restrict__ out3,
    const uint32_t* __restrict__ round_consts,  // [RF*8=64, RP=22] = 86 values
    const uint32_t* __restrict__ alpha_coeffs,  // [STATE_WIDTH * 4] QM31 coefficients
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t next_i = (i + 1) % n;
    uint32_t round_in_block = i % POSEIDON_NUM_ROUNDS;
    uint32_t next_round = (round_in_block + 1) % POSEIDON_NUM_ROUNDS;

    bool next_is_partial = (next_round >= POSEIDON_RF_BEFORE) &&
                           (next_round < POSEIDON_RF_BEFORE + POSEIDON_RP);

    uint32_t state[POSEIDON_STATE_WIDTH];
    uint32_t expected[POSEIDON_STATE_WIDTH];

    if (next_is_partial) {
        int p = next_round - POSEIDON_RF_BEFORE;
        state[0] = m31_add(trace_cols[0][i], partial_rc(round_consts, p));
        state[0] = p2_sbox_c(state[0]);
        for (int j = 1; j < POSEIDON_STATE_WIDTH; j++) state[j] = trace_cols[j][i];
        m_int_c(state);
    } else {
        int fr = (next_round < POSEIDON_RF_BEFORE) ? next_round : (next_round - POSEIDON_RP);
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            state[j] = m31_add(trace_cols[j][i], full_rc(round_consts, fr, j));
        }
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) state[j] = p2_sbox_c(state[j]);
        m_ext_c(state);
    }
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) expected[j] = state[j];

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

__global__ void poseidon2_quotient_chunk_kernel(
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
    uint32_t round_in_block = i % POSEIDON_NUM_ROUNDS;
    uint32_t next_round = (round_in_block + 1) % POSEIDON_NUM_ROUNDS;

    bool next_is_partial = (next_round >= POSEIDON_RF_BEFORE) &&
                           (next_round < POSEIDON_RF_BEFORE + POSEIDON_RP);

    uint32_t state[POSEIDON_STATE_WIDTH];
    uint32_t expected[POSEIDON_STATE_WIDTH];

    if (next_is_partial) {
        int p = next_round - POSEIDON_RF_BEFORE;
        state[0] = m31_add(trace_cols[0][i], partial_rc(round_consts, p));
        state[0] = p2_sbox_c(state[0]);
        for (int j = 1; j < POSEIDON_STATE_WIDTH; j++) state[j] = trace_cols[j][i];
        m_int_c(state);
    } else {
        int fr = (next_round < POSEIDON_RF_BEFORE) ? next_round : (next_round - POSEIDON_RP);
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) {
            state[j] = m31_add(trace_cols[j][i], full_rc(round_consts, fr, j));
        }
        for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) state[j] = p2_sbox_c(state[j]);
        m_ext_c(state);
    }
    for (int j = 0; j < POSEIDON_STATE_WIDTH; j++) expected[j] = state[j];

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
    poseidon2_quotient_kernel<<<blocks, threads>>>(
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
    poseidon2_quotient_chunk_kernel<<<blocks, threads>>>(
        trace_cols, out0, out1, out2, out3, round_consts, alpha_coeffs,
        offset, chunk_n, global_n
    );
}

} // extern "C"
