// GPU Poseidon252 permutation over Stark252 field — Montgomery form throughout.
//
// All values (state, round constants, inputs, outputs) are in Montgomery form:
//   a_mont = a * R mod p   where R = 2^256, p = 2^251 + 17*2^192 + 1
//
// This matches starknet-crypto's poseidon_permute_comp which operates on
// FieldElement values stored in Montgomery form by arkworks Fp256.
//
// Round structure (compressed, matching poseidon_permute_comp):
//   Full round:    state[i] += rc[idx+i]; state[i] = cube(state[i]) for all i; mix
//   Partial round: state[2] += rc[idx];   state[2] = cube(state[2]);            mix
//
// poseidon_hash(x, y) = permute([x, y, 2_mont])[0]

#pragma once
#include "fp252.cuh"
#include "poseidon_round_keys.cuh"

// Montgomery forms of small constants
#define MONT_ONE_0 0xffffffffffffffe1ULL
#define MONT_ONE_1 0xffffffffffffffffULL
#define MONT_ONE_2 0xffffffffffffffffULL
#define MONT_ONE_3 0x07fffffffffffdf0ULL

#define MONT_TWO_0 0xffffffffffffffc1ULL
#define MONT_TWO_1 0xffffffffffffffffULL
#define MONT_TWO_2 0xffffffffffffffffULL
#define MONT_TWO_3 0x07fffffffffffbd0ULL

__device__ __forceinline__ Fp252 mont_one()  { return {{MONT_ONE_0, MONT_ONE_1, MONT_ONE_2, MONT_ONE_3}}; }
__device__ __forceinline__ Fp252 mont_two()  { return {{MONT_TWO_0, MONT_TWO_1, MONT_TWO_2, MONT_TWO_3}}; }
__device__ __forceinline__ Fp252 mont_zero() { return {{0, 0, 0, 0}}; }

// x^3 in Montgomery form: two Montgomery multiplications
__device__ __forceinline__ Fp252 fp_cube_mont(Fp252 x) {
    Fp252 x2 = fp_mont_mul(x, x);
    return fp_mont_mul(x2, x);
}

// MDS matrix: t = s0+s1+s2; s0=t+2*s0; s1=t-2*s1; s2=t-3*s2
// Addition/subtraction is form-agnostic (same for Montgomery and normal)
__device__ __forceinline__ void poseidon_mix(Fp252 s[3]) {
    Fp252 t = fp_add(fp_add(s[0], s[1]), s[2]);
    s[0] = fp_add(t, fp_add(s[0], s[0]));
    s[1] = fp_sub(t, fp_add(s[1], s[1]));
    Fp252 s2x3 = fp_add(fp_add(s[2], s[2]), s[2]);
    s[2] = fp_sub(t, s2x3);
}

// Full round: 3 constants, cube all 3, mix
__device__ __forceinline__ int poseidon_full_round(Fp252 s[3], int idx) {
    s[0] = fp_cube_mont(fp_add(s[0], POSEIDON_COMP_CONSTS_GPU[idx]));
    s[1] = fp_cube_mont(fp_add(s[1], POSEIDON_COMP_CONSTS_GPU[idx + 1]));
    s[2] = fp_cube_mont(fp_add(s[2], POSEIDON_COMP_CONSTS_GPU[idx + 2]));
    poseidon_mix(s);
    return idx + 3;
}

// Partial round (compressed): 1 constant on state[2], cube state[2] only, mix
__device__ __forceinline__ int poseidon_partial_round(Fp252 s[3], int idx) {
    s[2] = fp_cube_mont(fp_add(s[2], POSEIDON_COMP_CONSTS_GPU[idx]));
    poseidon_mix(s);
    return idx + 1;
}

// Full permutation: 4 full + 83 partial + 4 full rounds
__device__ __forceinline__ void poseidon_permute_mont(Fp252 s[3]) {
    int idx = 0;
    idx = poseidon_full_round(s, idx);
    idx = poseidon_full_round(s, idx);
    idx = poseidon_full_round(s, idx);
    idx = poseidon_full_round(s, idx);
    #pragma unroll 4
    for (int r = 0; r < POSEIDON_PARTIAL_ROUNDS; r++) {
        idx = poseidon_partial_round(s, idx);
    }
    idx = poseidon_full_round(s, idx);
    idx = poseidon_full_round(s, idx);
    idx = poseidon_full_round(s, idx);
    idx = poseidon_full_round(s, idx);
}

// poseidon_hash(x, y): state = [x, y, 2_mont], permute, return state[0]
// x and y must be in Montgomery form (FieldElement252 / Fp252 Montgomery).
__device__ __forceinline__ Fp252 poseidon_hash_pair_mont(Fp252 x, Fp252 y) {
    Fp252 s[3] = {x, y, mont_two()};
    poseidon_permute_mont(s);
    return s[0];
}

// ── M31 packing ─────────────────────────────────────────────────────────────

// Shift a 4-limb little-endian bigint left by 31 bits in place.
__device__ __forceinline__ void shift_left_31_inplace(uint64_t limbs[4]) {
    limbs[3] = (limbs[3] << 31) | (limbs[2] >> 33);
    limbs[2] = (limbs[2] << 31) | (limbs[1] >> 33);
    limbs[1] = (limbs[1] << 31) | (limbs[0] >> 33);
    limbs[0] = limbs[0] << 31;
}

// Pack up to 8 M31 values into a Fp252 in NORMAL form.
// Packing: value = v[0]*2^(31*7) + v[1]*2^(31*6) + ... + v[n-1]*2^(31*(8-n))
// For n < 8: adds length padding n * 2^248 (bits 248-250 = top 3 bits of limb 3)
__device__ __forceinline__ Fp252 pack_m31s_normal(const uint32_t* vals, int n) {
    uint64_t limbs[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; i++) {
        shift_left_31_inplace(limbs);
        limbs[0] |= (uint64_t)vals[i];
    }
    if (n < 8) {
        // add_length_padding: += n * 2^248
        // 2^248 = bit 248 = limb 3 bit (248-192)=56
        limbs[3] += (uint64_t)n << 56;
    }
    Fp252 r = {{limbs[0], limbs[1], limbs[2], limbs[3]}};
    return r;
}

// Pack M31 values and convert to Montgomery form (ready for sponge absorption)
__device__ __forceinline__ Fp252 pack_m31s_mont(const uint32_t* vals, int n) {
    return to_mont(pack_m31s_normal(vals, n));
}
