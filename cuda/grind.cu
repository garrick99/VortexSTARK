// GPU PoW nonce grinding for Blake2s channel.
//
// The PoW check for stwo's Blake2sChannel:
//   prefixed_digest = Blake2s(POW_PREFIX_LE || [0u8;12] || channel_digest || pow_bits_LE)
//   result = Blake2s(prefixed_digest || nonce_LE)
//   valid iff trailing_zeros(result[0..16] as u128_le) >= pow_bits
//
// The prefixed_digest is constant for all nonces, so we precompute it on CPU
// and upload it. Each GPU thread tries one nonce and does a single Blake2s
// compression (40 bytes input < 64 byte block = 1 block).

#include "include/blake2s.cuh"

// Each thread tries nonce = thread_global_id + batch_offset.
// If found, atomically writes the nonce to result[0] (initially ~0ULL).
__global__ void grind_pow_kernel(
    const uint32_t* __restrict__ prefixed_digest, // [8] words = 32 bytes
    uint64_t* __restrict__ result,                 // [1] atomic result (init to UINT64_MAX)
    uint32_t pow_bits,
    uint64_t batch_offset,
    uint32_t n_threads_total
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads_total) return;

    // Early exit: someone already found a nonce
    // Use volatile read to avoid caching stale values
    if (*((volatile uint64_t*)result) != UINT64_MAX) return;

    uint64_t nonce = (uint64_t)tid + batch_offset;

    // Build 64-byte message block:
    //   m[0..7] = prefixed_digest (32 bytes)
    //   m[8..9] = nonce_le (8 bytes)
    //   m[10..15] = 0 (padding)
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

    // Blake2s init: h = IV ^ param_block (hash_len=32, key_len=0 => 0x01010020)
    uint32_t h0 = IV0 ^ 0x01010020u, h1 = IV1, h2 = IV2, h3 = IV3;
    uint32_t h4 = IV4, h5 = IV5, h6 = IV6, h7 = IV7;

    // Single compression: t = 40 (input length), f = 0xFFFFFFFF (final block)
    blake2s_compress(h0, h1, h2, h3, h4, h5, h6, h7,
                     m0, m1, m2, m3, m4, m5, m6, m7,
                     m8, m9, 0, 0, 0, 0, 0, 0,
                     40, 0xFFFFFFFF);

    // Check trailing zeros of first 16 bytes (u128 little-endian)
    // h0..h3 are the first 16 bytes in LE order
    uint32_t tz;
    if (h0 != 0) {
        tz = __ffs(h0) - 1; // __ffs returns 1-indexed position of least significant set bit
    } else if (h1 != 0) {
        tz = 32 + __ffs(h1) - 1;
    } else if (h2 != 0) {
        tz = 64 + __ffs(h2) - 1;
    } else if (h3 != 0) {
        tz = 96 + __ffs(h3) - 1;
    } else {
        tz = 128;
    }

    if (tz >= pow_bits) {
        // Found! Atomically store the minimum nonce
        atomicMin((unsigned long long*)result, (unsigned long long)nonce);
    }
}

extern "C" {

// Launch GPU grind kernel.
// prefixed_digest: device pointer to 8 x uint32 (the precomputed Blake2s of prefix||digest||pow_bits)
// result: device pointer to 1 x uint64 (initialized to UINT64_MAX by caller)
// Returns immediately (async). Caller must cudaDeviceSynchronize and read result.
void cuda_grind_pow(
    const uint32_t* prefixed_digest,
    uint64_t* result,
    uint32_t pow_bits,
    uint64_t batch_offset,
    uint32_t n_threads
) {
    uint32_t threads_per_block = 256;
    uint32_t blocks = (n_threads + threads_per_block - 1) / threads_per_block;
    grind_pow_kernel<<<blocks, threads_per_block>>>(
        prefixed_digest, result, pow_bits, batch_offset, n_threads
    );
}

} // extern "C"
