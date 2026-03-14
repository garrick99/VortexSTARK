// 252-bit field arithmetic for STARK curve (p = 2^251 + 17*2^192 + 1).
// 4 x u64 limbs, little-endian.

#pragma once
#include <stdint.h>

// STARK prime limbs
#define FP_P0 0x0000000000000001ULL
#define FP_P1 0x0000000000000000ULL
#define FP_P2 0x0000000000000000ULL
#define FP_P3 0x0800000000000011ULL

struct Fp252 {
    uint64_t v[4];
};

__device__ __forceinline__ Fp252 fp_zero() { return {{0, 0, 0, 0}}; }
__device__ __forceinline__ Fp252 fp_one()  { return {{1, 0, 0, 0}}; }

__device__ __forceinline__ bool fp_is_zero(Fp252 a) {
    return a.v[0] == 0 && a.v[1] == 0 && a.v[2] == 0 && a.v[3] == 0;
}

__device__ __forceinline__ bool fp_ge(Fp252 a, Fp252 b) {
    for (int i = 3; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true; // equal
}

// 64-bit add with carry
__device__ __forceinline__ uint64_t add64(uint64_t a, uint64_t b, uint32_t* carry) {
    uint64_t r = a + b;
    *carry = (r < a) ? 1 : 0;
    return r;
}

__device__ __forceinline__ uint64_t adc64(uint64_t a, uint64_t b, uint32_t carry_in, uint32_t* carry_out) {
    uint64_t r = a + b;
    uint32_t c1 = (r < a) ? 1 : 0;
    uint64_t r2 = r + carry_in;
    uint32_t c2 = (r2 < r) ? 1 : 0;
    *carry_out = c1 + c2;
    return r2;
}

// 64x64 -> 128 multiply using PTX
__device__ __forceinline__ void mul64(uint64_t a, uint64_t b, uint64_t* lo, uint64_t* hi) {
    *lo = a * b;
    *hi = __umul64hi(a, b);
}

__device__ __forceinline__ Fp252 fp_add(Fp252 a, Fp252 b) {
    Fp252 r;
    uint32_t c = 0;
    r.v[0] = adc64(a.v[0], b.v[0], 0, &c);
    r.v[1] = adc64(a.v[1], b.v[1], c, &c);
    r.v[2] = adc64(a.v[2], b.v[2], c, &c);
    r.v[3] = adc64(a.v[3], b.v[3], c, &c);

    Fp252 p = {{FP_P0, FP_P1, FP_P2, FP_P3}};
    if (c || fp_ge(r, p)) {
        // Subtract p with proper borrow chain
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t pi_plus_borrow = p.v[i] + borrow;
            uint64_t overflow = (pi_plus_borrow < borrow) ? 1 : 0; // p.v[i] + borrow overflow
            borrow = (r.v[i] < pi_plus_borrow) ? 1 : 0;
            borrow += overflow;
            r.v[i] -= pi_plus_borrow;
        }
    }
    return r;
}

__device__ __forceinline__ Fp252 fp_sub(Fp252 a, Fp252 b) {
    if (fp_ge(a, b)) {
        Fp252 r;
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t bi_plus_borrow = b.v[i] + borrow;
            uint64_t overflow = (bi_plus_borrow < borrow) ? 1ULL : 0ULL;
            borrow = (a.v[i] < bi_plus_borrow) ? 1ULL : 0ULL;
            borrow += overflow;
            r.v[i] = a.v[i] - bi_plus_borrow;
        }
        return r;
    } else {
        // a < b: compute p + a - b (raw add, no reduce)
        uint32_t c = 0;
        Fp252 raw;
        raw.v[0] = adc64(FP_P0, a.v[0], 0, &c);
        raw.v[1] = adc64(FP_P1, a.v[1], c, &c);
        raw.v[2] = adc64(FP_P2, a.v[2], c, &c);
        raw.v[3] = adc64(FP_P3, a.v[3], c, &c);
        // Subtract b
        Fp252 r;
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t bi_plus_borrow = b.v[i] + borrow;
            uint64_t overflow = (bi_plus_borrow < borrow) ? 1ULL : 0ULL;
            borrow = (raw.v[i] < bi_plus_borrow) ? 1ULL : 0ULL;
            borrow += overflow;
            r.v[i] = raw.v[i] - bi_plus_borrow;
        }
        return r;
    }
}

__device__ __forceinline__ Fp252 fp_neg(Fp252 a) {
    if (fp_is_zero(a)) return a;
    Fp252 p = {{FP_P0, FP_P1, FP_P2, FP_P3}};
    return fp_sub(p, a);
}

// Helper: schoolbook 4x4 multiply into an 8-limb accumulator
__device__ static void schoolbook_4x4(const Fp252& a, const Fp252& b, uint64_t out[8]) {
    for (int i = 0; i < 8; i++) out[i] = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t plo = a.v[i] * b.v[j];
            uint64_t phi = __umul64hi(a.v[i], b.v[j]);

            // out[i+j] += plo + carry (with full overflow tracking)
            uint64_t old = out[i+j];
            uint64_t s1 = old + plo;
            uint64_t c1 = (s1 < old) ? 1ULL : 0ULL;
            uint64_t s2 = s1 + carry;
            uint64_t c2 = (s2 < s1) ? 1ULL : 0ULL;
            out[i+j] = s2;

            // carry for next iteration = phi + c1 + c2 (safe: phi < 2^64, c1+c2 <= 2)
            carry = phi + c1 + c2;
        }
        // Propagate final carry into out[i+4]
        if (i + 4 < 8) {
            uint64_t old = out[i+4];
            out[i+4] = old + carry;
            // If this overflows, propagate further
            if (out[i+4] < old && i + 5 < 8) out[i+5]++;
        }
    }
}

// Schoolbook 4x4 multiply with 64-bit limbs → 8-limb product.
// Then reduce via recursive lo + hi * R mod p.
__device__ static Fp252 fp_mul(Fp252 a, Fp252 b) {
    uint64_t full[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Schoolbook 4x4 multiply. Accumulate products into 8 u64 limbs.
    // Use __umul64hi for the high 64 bits of a 64x64 product.
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            // product = a.v[i] * b.v[j] (128-bit)
            uint64_t plo = a.v[i] * b.v[j];
            uint64_t phi = __umul64hi(a.v[i], b.v[j]);

            // Accumulate: full[i+j] += plo + carry
            uint64_t sum = full[i+j] + plo;
            uint64_t c1 = (sum < full[i+j]) ? 1ULL : 0ULL;
            uint64_t sum2 = sum + carry;
            uint64_t c2 = (sum2 < sum) ? 1ULL : 0ULL;
            full[i+j] = sum2;

            carry = phi + c1 + c2; // carry for next limb
        }
        if (i + 4 < 8) {
            full[i + 4] += carry;
        }
    }

    // Non-recursive reduction: compute hi * R as another 4x4 schoolbook mul,
    // then add to lo. Repeat until the high part is zero.
    // R = 2^256 mod p = {0xFFFFFFFFFFFFFFE1, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x07FFFFFFFFFFFDF0}
    Fp252 R = {{0xFFFFFFFFFFFFFFE1ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x07FFFFFFFFFFFDF0ULL}};
    Fp252 p = {{FP_P0, FP_P1, FP_P2, FP_P3}};

    // Iterative fold: keep folding hi*R + lo until hi is zero.
    // Each fold reduces bit-width by ~4 bits. 512 bits → 252 in ~65 iterations max,
    // but in practice converges in 3-4 iterations.
    uint64_t f[8] = {full[0], full[1], full[2], full[3], full[4], full[5], full[6], full[7]};

    // Just 3 iterations needed with the recursive multiply approach,
    // but we use iterative to avoid CUDA stack overflow.
    // Each iteration reduces by ~4 bits. For 252-bit inputs, product < 504 bits,
    // hi < 248 bits. After fold: hi*R product high < 244 bits. Need ~62 iterations worst case.
    for (int iter = 0; iter < 70; iter++) {
        Fp252 lo_i = {{f[0], f[1], f[2], f[3]}};
        Fp252 hi_i = {{f[4], f[5], f[6], f[7]}};

        if (fp_is_zero(hi_i)) {
            while (fp_ge(lo_i, p)) lo_i = fp_sub(lo_i, p);
            return lo_i;
        }

        // f = lo + hi * R
        uint64_t hr[8];
        schoolbook_4x4(hi_i, R, hr);

        // Add lo to hr
        uint32_t cc = 0;
        hr[0] = adc64(hr[0], lo_i.v[0], 0, &cc);
        hr[1] = adc64(hr[1], lo_i.v[1], cc, &cc);
        hr[2] = adc64(hr[2], lo_i.v[2], cc, &cc);
        hr[3] = adc64(hr[3], lo_i.v[3], cc, &cc);
        if (cc) {
            hr[4] += 1;
            if (hr[4] == 0) { hr[5] += 1; if (hr[5] == 0) hr[6] += 1; }
        }

        for (int i = 0; i < 8; i++) f[i] = hr[i];
    }

    // Should have converged
    Fp252 result = {{f[0], f[1], f[2], f[3]}};
    while (fp_ge(result, p)) result = fp_sub(result, p);
    return result;
}

// Projective point on STARK curve
struct ProjPoint {
    Fp252 x, y, z;
};

__device__ __forceinline__ ProjPoint proj_infinity() {
    ProjPoint p;
    p.x = fp_zero(); p.y = fp_one(); p.z = fp_zero();
    return p;
}

__device__ __forceinline__ ProjPoint proj_from_affine(Fp252 x, Fp252 y) {
    ProjPoint p;
    p.x = x; p.y = y; p.z = fp_one();
    return p;
}

// Jacobian point doubling: ~6M + 4S
__device__ static ProjPoint proj_double(ProjPoint p) {
    if (fp_is_zero(p.z)) return p;

    Fp252 xx = fp_mul(p.x, p.x);
    Fp252 yy = fp_mul(p.y, p.y);
    Fp252 zz = fp_mul(p.z, p.z);
    Fp252 zzzz = fp_mul(zz, zz);

    // S = 4*X*Y²
    Fp252 xy2 = fp_mul(p.x, yy);
    Fp252 s = fp_add(fp_add(xy2, xy2), fp_add(xy2, xy2));

    // M = 3X² + a*Z⁴ (a=1)
    Fp252 m = fp_add(fp_add(xx, fp_add(xx, xx)), zzzz);

    // X' = M² - 2S
    Fp252 x3 = fp_sub(fp_mul(m, m), fp_add(s, s));

    // Y' = M(S-X') - 8Y⁴
    Fp252 yyyy = fp_mul(yy, yy);
    Fp252 y4x8 = fp_add(fp_add(fp_add(yyyy, yyyy), fp_add(yyyy, yyyy)),
                         fp_add(fp_add(yyyy, yyyy), fp_add(yyyy, yyyy)));
    Fp252 y3 = fp_sub(fp_mul(m, fp_sub(s, x3)), y4x8);

    // Z' = 2YZ
    Fp252 z3 = fp_mul(fp_add(p.y, p.y), p.z);

    ProjPoint r;
    r.x = x3; r.y = y3; r.z = z3;
    return r;
}

// Jacobian point addition: ~12M + 4S
__device__ static ProjPoint proj_add(ProjPoint p1, ProjPoint p2) {
    if (fp_is_zero(p1.z)) return p2;
    if (fp_is_zero(p2.z)) return p1;

    Fp252 z1z1 = fp_mul(p1.z, p1.z);
    Fp252 z2z2 = fp_mul(p2.z, p2.z);
    Fp252 u1 = fp_mul(p1.x, z2z2);
    Fp252 u2 = fp_mul(p2.x, z1z1);
    Fp252 s1 = fp_mul(fp_mul(p1.y, z2z2), p2.z);
    Fp252 s2 = fp_mul(fp_mul(p2.y, z1z1), p1.z);

    Fp252 h = fp_sub(u2, u1);
    Fp252 r = fp_sub(s2, s1);

    if (fp_is_zero(h)) {
        if (fp_is_zero(r)) return proj_double(p1);
        return proj_infinity();
    }

    Fp252 hh = fp_mul(h, h);
    Fp252 hhh = fp_mul(hh, h);
    Fp252 u1hh = fp_mul(u1, hh);

    Fp252 x3 = fp_sub(fp_sub(fp_mul(r, r), hhh), fp_add(u1hh, u1hh));
    Fp252 y3 = fp_sub(fp_mul(r, fp_sub(u1hh, x3)), fp_mul(s1, hhh));
    Fp252 z3 = fp_mul(fp_mul(p1.z, p2.z), h);

    ProjPoint res;
    res.x = x3; res.y = y3; res.z = z3;
    return res;
}

// Scalar multiplication: double-and-add in projective, single inversion at end
__device__ static ProjPoint proj_scalar_mul(ProjPoint base, Fp252 scalar) {
    ProjPoint result = proj_infinity();
    ProjPoint current = base;

    for (int limb = 0; limb < 4; limb++) {
        uint64_t s = scalar.v[limb];
        int bits = (limb == 3) ? 60 : 64;
        for (int b = 0; b < bits; b++) {
            if (s & 1) {
                result = proj_add(result, current);
            }
            current = proj_double(current);
            s >>= 1;
        }
    }
    return result;
}

// Convert projective to affine x-coordinate (requires one fp_inverse)
__device__ static Fp252 proj_to_affine_x(ProjPoint p) {
    if (fp_is_zero(p.z)) return fp_zero();
    // For Jacobian: x = X/Z²
    Fp252 zz = fp_mul(p.z, p.z);
    Fp252 zz_inv = fp_one(); // TODO: need fp_inverse on GPU

    // Fermat inverse: zz^(p-2). Too expensive per-thread.
    // Use batch inverse across all threads instead.
    // For now, compute inline (slow but correct).

    // p-2 = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0800000000000010}
    Fp252 exp = {{0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x0800000000000010ULL}};
    Fp252 base_val = zz;
    Fp252 result = fp_one();

    // MSB-first square-and-multiply
    // Collect bits
    int n_bits = 252;
    for (int i = n_bits - 1; i >= 0; i--) {
        result = fp_mul(result, result);
        int limb = i / 64;
        int bit = i % 64;
        if (exp.v[limb] & (1ULL << bit)) {
            result = fp_mul(result, base_val);
        }
    }

    Fp252 x_affine = fp_mul(p.x, result);
    return x_affine;
}
