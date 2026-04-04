// GPU PoW nonce grinding for Blake2s channel with IS_M31_OUTPUT=true.
//
// Identical to grind.cu but after computing the Blake2s hash, each
// output word h[i] is reduced mod M31_P before checking trailing zeros.
//
// From stwo: reduce_to_m31(hash) applies M31::reduce(word as u64) to each
// 4-byte word, then checks trailing_zeros of the first 16 bytes.
//
// M31::reduce(x: u64) = ((x & M31_P) + (x >> 31)) & M31_P clamped to [0, P)
// For a u32 input x, reduce(x as u64) = (x & P) + (x >> 31), possible subtract P.

#include "include/blake2s.cuh"
#include "include/m31.cuh"

__global__ void grind_pow_m31_kernel(
    const uint32_t* __restrict__ prefixed_digest,  // [8] words
    uint64_t* __restrict__ result,
    uint32_t pow_bits,
    uint64_t batch_offset,
    uint32_t n_threads_total
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads_total) return;

    if (*((volatile uint64_t*)result) != UINT64_MAX) return;

    uint64_t nonce = (uint64_t)tid + batch_offset;

    uint32_t m0 = prefixed_digest[0];
    uint32_t m1 = prefixed_digest[1];
    uint32_t m2 = prefixed_digest[2];
    uint32_t m3 = prefixed_digest[3];
    uint32_t m4 = prefixed_digest[4];
    uint32_t m5 = prefixed_digest[5];
    uint32_t m6 = prefixed_digest[6];
    uint32_t m7 = prefixed_digest[7];
    uint32_t m8 = (uint32_t)(nonce);
    uint32_t m9 = (uint32_t)(nonce >> 32);

    uint32_t h0 = IV0 ^ 0x01010020u, h1 = IV1, h2 = IV2, h3 = IV3;
    uint32_t h4 = IV4, h5 = IV5, h6 = IV6, h7 = IV7;

    blake2s_compress(h0, h1, h2, h3, h4, h5, h6, h7,
                     m0, m1, m2, m3, m4, m5, m6, m7,
                     m8, m9, 0, 0, 0, 0, 0, 0,
                     40, 0xFFFFFFFF);

    // Apply M31 reduction to each output word
    // m31_reduce(x as u64): for u32 x, reduce = (x & P) + (x >> 31), clamp
    uint32_t r0 = m31_reduce((uint64_t)h0);
    uint32_t r1 = m31_reduce((uint64_t)h1);
    uint32_t r2 = m31_reduce((uint64_t)h2);
    uint32_t r3 = m31_reduce((uint64_t)h3);

    // Check trailing zeros of first 16 bytes (4 reduced M31 words as u128 LE)
    uint32_t tz;
    if (r0 != 0) {
        tz = (uint32_t)(__ffs((int)r0) - 1);
    } else if (r1 != 0) {
        tz = 32u + (uint32_t)(__ffs((int)r1) - 1);
    } else if (r2 != 0) {
        tz = 64u + (uint32_t)(__ffs((int)r2) - 1);
    } else if (r3 != 0) {
        tz = 96u + (uint32_t)(__ffs((int)r3) - 1);
    } else {
        tz = 128u;
    }

    if (tz >= pow_bits) {
        atomicMin((unsigned long long*)result, (unsigned long long)nonce);
    }
}

extern "C" {

void cuda_grind_pow_m31_output(
    const uint32_t* prefixed_digest,
    uint64_t* result,
    uint32_t pow_bits,
    uint64_t batch_offset,
    uint32_t n_threads
) {
    uint32_t threads = 256;
    uint32_t blocks = (n_threads + threads - 1) / threads;
    grind_pow_m31_kernel<<<blocks, threads>>>(
        prefixed_digest, result, pow_bits, batch_offset, n_threads);
}

} // extern "C"
