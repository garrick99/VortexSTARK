// GPU PoW nonce grinding for Poseidon252 channel.
//
// stwo Poseidon252Channel::verify_pow_nonce:
//   prefixed_digest = poseidon_hash_many([POW_PREFIX, channel.digest, n_bits])
//   hash            = poseidon_hash(prefixed_digest, nonce)
//   n_zeros         = trailing_zeros of u128::from_be_bytes(hash.to_bytes_be()[16..])
//   valid iff n_zeros >= pow_bits
//
// poseidon_hash(x, y) = poseidon_permute([x, y, 2_mont])[0]
// All values in Montgomery form (Fp252 = 4 × u64 little-endian limbs).
//
// hash.to_bytes_be() converts from Montgomery form then encodes big-endian.
// The last 16 bytes correspond to the two least-significant limbs: v[1] and v[0].
//
// trailing_zeros of u128::from_be_bytes([v1_be, v0_be])
//   = trailing_zeros of (v1 << 64 | v0) as u128
//   = if v0 != 0: ctz(v0)
//     elif v1 != 0: 64 + ctz(v1)
//     else: 128
//
// To convert nonce (u64) to Montgomery form: nonce_mont = fp_mont_mul({nonce, 0, 0, 0}, r2)
// where r2 = R^2 mod p (constant available via fp252.cuh `to_mont`).

#include "include/poseidon252.cuh"

__global__ void grind_poseidon_kernel(
    const uint64_t* __restrict__ prefixed_digest_mont,  // 4 u64s, Montgomery form
    uint64_t* __restrict__ result,
    uint32_t pow_bits,
    uint64_t batch_offset,
    uint32_t n_threads_total
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads_total) return;

    if (*((volatile uint64_t*)result) != UINT64_MAX) return;

    uint64_t nonce = (uint64_t)tid + batch_offset;

    // Load prefixed_digest in Montgomery form
    Fp252 pdig;
    pdig.v[0] = prefixed_digest_mont[0];
    pdig.v[1] = prefixed_digest_mont[1];
    pdig.v[2] = prefixed_digest_mont[2];
    pdig.v[3] = prefixed_digest_mont[3];

    // Convert nonce to Montgomery form
    Fp252 nonce_std = {{nonce, 0, 0, 0}};
    Fp252 nonce_mont = to_mont(nonce_std);

    // Poseidon state: [pdig, nonce_mont, MONT_TWO]
    Fp252 state[3];
    state[0] = pdig;
    state[1] = nonce_mont;
    state[2] = mont_two();

    poseidon_permute_mont(state);

    // Convert hash result from Montgomery to standard form
    Fp252 hash_std = from_mont(state[0]);

    // trailing_zeros of u128 = (v1 << 64 | v0) via big-endian last 16 bytes
    uint64_t v0 = hash_std.v[0];
    uint64_t v1 = hash_std.v[1];

    uint32_t tz;
    if (v0 != 0ULL) {
        tz = (uint32_t)(__ffsll((long long)v0) - 1);
    } else if (v1 != 0ULL) {
        tz = 64u + (uint32_t)(__ffsll((long long)v1) - 1);
    } else {
        tz = 128u;
    }

    if (tz >= pow_bits) {
        atomicMin((unsigned long long*)result, (unsigned long long)nonce);
    }
}

extern "C" {

void cuda_grind_pow_poseidon(
    const uint64_t* prefixed_digest_mont,
    uint64_t* result,
    uint32_t pow_bits,
    uint64_t batch_offset,
    uint32_t n_threads
) {
    uint32_t threads = 256;
    uint32_t blocks = (n_threads + threads - 1) / threads;
    grind_poseidon_kernel<<<blocks, threads>>>(
        prefixed_digest_mont, result, pow_bits, batch_offset, n_threads);
}

} // extern "C"
