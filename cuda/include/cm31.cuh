#pragma once
#include "m31.cuh"

// CM31: Complex extension of M31. Elements are a + bi where i^2 = -1.
// Stored as {a, b} (two M31 values).
struct CM31 {
    uint32_t a;  // real part
    uint32_t b;  // imaginary part
};

__device__ __forceinline__ CM31 cm31_add(CM31 x, CM31 y) {
    return {m31_add(x.a, y.a), m31_add(x.b, y.b)};
}

__device__ __forceinline__ CM31 cm31_sub(CM31 x, CM31 y) {
    return {m31_sub(x.a, y.a), m31_sub(x.b, y.b)};
}

// (a+bi)(c+di) = (ac-bd) + (ad+bc)i
__device__ __forceinline__ CM31 cm31_mul(CM31 x, CM31 y) {
    uint32_t ac = m31_mul(x.a, y.a);
    uint32_t bd = m31_mul(x.b, y.b);
    uint32_t ad = m31_mul(x.a, y.b);
    uint32_t bc = m31_mul(x.b, y.a);
    return {m31_sub(ac, bd), m31_add(ad, bc)};
}

__device__ __forceinline__ CM31 cm31_neg(CM31 x) {
    return {m31_neg(x.a), m31_neg(x.b)};
}

// CM31 multiply by M31 scalar
__device__ __forceinline__ CM31 cm31_mul_m31(CM31 x, uint32_t s) {
    return {m31_mul(x.a, s), m31_mul(x.b, s)};
}

// CM31 conjugate: conj(a+bi) = a-bi
__device__ __forceinline__ CM31 cm31_conj(CM31 x) {
    return {x.a, m31_neg(x.b)};
}

// CM31 norm: |a+bi|^2 = a^2 + b^2 (in M31, since i^2 = -1)
__device__ __forceinline__ uint32_t cm31_norm(CM31 x) {
    return m31_add(m31_sqr(x.a), m31_sqr(x.b));
}

// CM31 inverse: 1/(a+bi) = conj/(norm) = (a-bi)/(a^2+b^2)
__device__ __forceinline__ CM31 cm31_inv(CM31 x) {
    uint32_t inv_norm = m31_inv(cm31_norm(x));
    CM31 c = cm31_conj(x);
    return cm31_mul_m31(c, inv_norm);
}

__device__ __forceinline__ CM31 cm31_sqr(CM31 x) {
    return cm31_mul(x, x);
}
