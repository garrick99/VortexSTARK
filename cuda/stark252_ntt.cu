// Stark252 NTT kernel — standard (non-Circle) FRI over the 252-bit STARK prime.
//
// Layout: SoA — data is stored as 4 contiguous blocks of n u64s, one block per limb.
//   d_data[0*n .. 1*n-1] = limb 0 of all elements
//   d_data[1*n .. 2*n-1] = limb 1 of all elements
//   d_data[2*n .. 3*n-1] = limb 2 of all elements
//   d_data[3*n .. 4*n-1] = limb 3 of all elements
//
// Twiddle factors are stored in the same SoA layout (n/2 entries).
// All arithmetic uses fp_mul (iterative Barrett) from fp252.cuh — no Montgomery form
// conversion needed since the prover works in standard representation.
//
// NTT variant: Cooley-Tukey DIT.
//   - Input is bit-reversed before the kernel launches (host-side).
//   - Each kernel invocation handles one butterfly stage (stride = half_len).
//   - After all log_n stages, output is in natural order.

#include "include/fp252.cuh"
#include <stdint.h>

// ─────────────────────────────────────────────────────────────────
// Load / store helpers for SoA layout
// ─────────────────────────────────────────────────────────────────

__device__ __forceinline__ Fp252 soa_load(const uint64_t* d, uint32_t i, uint32_t n) {
    Fp252 r;
    r.v[0] = d[0*n + i];
    r.v[1] = d[1*n + i];
    r.v[2] = d[2*n + i];
    r.v[3] = d[3*n + i];
    return r;
}

__device__ __forceinline__ void soa_store(uint64_t* d, uint32_t i, uint32_t n, Fp252 x) {
    d[0*n + i] = x.v[0];
    d[1*n + i] = x.v[1];
    d[2*n + i] = x.v[2];
    d[3*n + i] = x.v[3];
}

// ─────────────────────────────────────────────────────────────────
// Single-stage DIT butterfly kernel
//
// For each butterfly pair (k, k+half_len) in the current group:
//   u = a[k]
//   v = a[k + half_len] * twiddle[j]   (j = k mod half_len)
//   a[k]          = u + v
//   a[k + half_len] = u - v
//
// half_len: half the group size for this stage (1, 2, 4, ..., n/2)
// n:        total number of elements
// ─────────────────────────────────────────────────────────────────

__global__ void stark252_ntt_butterfly(
    uint64_t* __restrict__ d_data,   // SoA: 4*n u64s
    const uint64_t* __restrict__ d_tw, // SoA twiddles: 4*(n/2) u64s
    uint32_t half_len,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_butterflies = n / 2;
    if (tid >= total_butterflies) return;

    // Determine which butterfly this thread handles.
    // For half_len h: group index g = tid / h, offset within group j = tid % h
    // k = g * 2h + j
    uint32_t g = tid / half_len;
    uint32_t j = tid % half_len;
    uint32_t k = g * (2 * half_len) + j;

    // Twiddle index: for a full-size NTT of size n, the twiddle for position j in a
    // group of half_len is twiddle_table[j * (n/2 / half_len)] = twiddle_table[j * stride_tw]
    // where stride_tw = (n/2) / half_len
    uint32_t stride_tw = (n / 2) / half_len;
    uint32_t tw_idx = j * stride_tw;

    Fp252 u   = soa_load(d_data, k,             n);
    Fp252 vm  = soa_load(d_data, k + half_len,  n);
    Fp252 tw  = soa_load(d_tw,   tw_idx,        n / 2);

    Fp252 v = fp_mul(vm, tw);
    soa_store(d_data, k,            n, fp_add(u, v));
    soa_store(d_data, k + half_len, n, fp_sub(u, v));
}

// ─────────────────────────────────────────────────────────────────
// Bit-reversal permutation kernel
//
// Swaps element i with element bit_rev(i, log_n) for all i < bit_rev(i).
// ─────────────────────────────────────────────────────────────────

__device__ __forceinline__ uint32_t bit_rev(uint32_t x, uint32_t bits) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < bits; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

__global__ void stark252_bit_reverse(
    uint64_t* __restrict__ d_data,
    uint32_t log_n,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t j = bit_rev(i, log_n);
    if (i < j) {
        // Swap limbs for element i and j (SoA layout: offset k*n+idx)
        for (int k = 0; k < 4; k++) {
            uint64_t tmp = d_data[k*n + i];
            d_data[k*n + i] = d_data[k*n + j];
            d_data[k*n + j] = tmp;
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Scalar multiply by 1/N (for INTT normalization)
// ─────────────────────────────────────────────────────────────────

__global__ void stark252_scale(
    uint64_t* __restrict__ d_data,
    const uint64_t* __restrict__ d_inv_n, // 4 u64s (single Fp252 in standard form)
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Fp252 x      = soa_load(d_data, i, n);
    Fp252 inv_n  = {{d_inv_n[0], d_inv_n[1], d_inv_n[2], d_inv_n[3]}};
    Fp252 result = fp_mul(x, inv_n);
    soa_store(d_data, i, n, result);
}

// ─────────────────────────────────────────────────────────────────
// Host-callable entry points (extern "C" for Rust FFI)
// ─────────────────────────────────────────────────────────────────

static const int BLOCK = 256;

// Forward NTT (no normalization).
// d_data: device pointer to 4*n u64s (SoA).
// d_tw:   device pointer to 4*(n/2) u64s (SoA forward twiddles: ω_N^j for j=0..n/2-1).
// log_n:  log2(n).
extern "C" void cuda_stark252_ntt_forward(
    uint64_t* d_data,
    const uint64_t* d_tw,
    uint32_t log_n
) {
    uint32_t n = 1u << log_n;

    // Bit-reverse permutation
    stark252_bit_reverse<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(d_data, log_n, n);

    // DIT butterfly stages: half_len = 1, 2, 4, ..., n/2
    uint32_t half_len = 1;
    for (uint32_t s = 0; s < log_n; s++) {
        uint32_t total_butterflies = n / 2;
        stark252_ntt_butterfly<<<(total_butterflies + BLOCK - 1) / BLOCK, BLOCK>>>(
            d_data, d_tw, half_len, n
        );
        half_len <<= 1;
    }
}

// Inverse NTT (includes 1/N scaling).
// d_data: device pointer to 4*n u64s (SoA).
// d_tw:   device pointer to 4*(n/2) u64s (SoA inverse twiddles: ω_N^{-j}).
// log_n:  log2(n).
// d_inv_n: device pointer to 4 u64s (1/N in standard form).
extern "C" void cuda_stark252_ntt_inverse(
    uint64_t* d_data,
    const uint64_t* d_tw,
    uint32_t log_n,
    const uint64_t* d_inv_n
) {
    uint32_t n = 1u << log_n;

    // Bit-reverse permutation (same structure as forward)
    stark252_bit_reverse<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(d_data, log_n, n);

    // DIT butterfly stages with inverse twiddles
    uint32_t half_len = 1;
    for (uint32_t s = 0; s < log_n; s++) {
        uint32_t total_butterflies = n / 2;
        stark252_ntt_butterfly<<<(total_butterflies + BLOCK - 1) / BLOCK, BLOCK>>>(
            d_data, d_tw, half_len, n
        );
        half_len <<= 1;
    }

    // Scale by 1/N
    stark252_scale<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(d_data, d_inv_n, n);
}
