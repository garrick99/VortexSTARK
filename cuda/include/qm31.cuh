#pragma once
#include "cm31.cuh"

// QM31: Secure field extension. Elements are a + bu where u^2 = 2 + i.
// Stored as {v[0], v[1], v[2], v[3]} = CM31(v[0], v[1]) + CM31(v[2], v[3]) * u
// i.e., a = CM31(v[0], v[1]), b = CM31(v[2], v[3])
struct QM31 {
    uint32_t v[4];
};

__device__ __forceinline__ CM31 qm31_a(QM31 x) { return {x.v[0], x.v[1]}; }
__device__ __forceinline__ CM31 qm31_b(QM31 x) { return {x.v[2], x.v[3]}; }

__device__ __forceinline__ QM31 qm31_from(CM31 a, CM31 b) {
    return {{a.a, a.b, b.a, b.b}};
}

__device__ __forceinline__ QM31 qm31_add(QM31 x, QM31 y) {
    CM31 a = cm31_add(qm31_a(x), qm31_a(y));
    CM31 b = cm31_add(qm31_b(x), qm31_b(y));
    return qm31_from(a, b);
}

__device__ __forceinline__ QM31 qm31_sub(QM31 x, QM31 y) {
    CM31 a = cm31_sub(qm31_a(x), qm31_a(y));
    CM31 b = cm31_sub(qm31_b(x), qm31_b(y));
    return qm31_from(a, b);
}

// (a + bu)(c + du) = (ac + bd*(2+i)) + (ad + bc)u
// where u^2 = 2+i
__device__ __forceinline__ QM31 qm31_mul(QM31 x, QM31 y) {
    CM31 xa = qm31_a(x), xb = qm31_b(x);
    CM31 ya = qm31_a(y), yb = qm31_b(y);

    CM31 ac = cm31_mul(xa, ya);
    CM31 bd = cm31_mul(xb, yb);
    CM31 ad = cm31_mul(xa, yb);
    CM31 bc = cm31_mul(xb, ya);

    // bd * (2+i) = bd*2 + bd*i
    CM31 bd_times_u2;
    // (2+i) as CM31 = {2, 1}
    bd_times_u2.a = m31_sub(m31_add(m31_mul(bd.a, 2), m31_mul(bd.a, 0)), bd.b);  // 2*re - im
    // Actually: (p+qi)(2+i) = (2p - q) + (p + 2q)i
    bd_times_u2.a = m31_sub(m31_mul(bd.a, 2), bd.b);
    bd_times_u2.b = m31_add(bd.a, m31_mul(bd.b, 2));

    CM31 ra = cm31_add(ac, bd_times_u2);
    CM31 rb = cm31_add(ad, bc);
    return qm31_from(ra, rb);
}

__device__ __forceinline__ QM31 qm31_neg(QM31 x) {
    return {{m31_neg(x.v[0]), m31_neg(x.v[1]), m31_neg(x.v[2]), m31_neg(x.v[3])}};
}

// QM31 * M31 scalar
__device__ __forceinline__ QM31 qm31_mul_m31(QM31 x, uint32_t s) {
    return {{m31_mul(x.v[0], s), m31_mul(x.v[1], s), m31_mul(x.v[2], s), m31_mul(x.v[3], s)}};
}

// QM31 conjugate: conj(a + bu) = a - bu
__device__ __forceinline__ QM31 qm31_conj(QM31 x) {
    CM31 a = qm31_a(x);
    CM31 b = cm31_neg(qm31_b(x));
    return qm31_from(a, b);
}

// QM31 norm: a^2 - b^2*(2+i)
__device__ __forceinline__ CM31 qm31_norm(QM31 x) {
    CM31 a = qm31_a(x), b = qm31_b(x);
    CM31 a2 = cm31_sqr(a);
    CM31 b2 = cm31_sqr(b);
    // b2 * (2+i)
    CM31 b2u2;
    b2u2.a = m31_sub(m31_mul(b2.a, 2), b2.b);
    b2u2.b = m31_add(b2.a, m31_mul(b2.b, 2));
    return cm31_sub(a2, b2u2);
}

// QM31 inverse: conj / norm, then CM31 inverse
__device__ __forceinline__ QM31 qm31_inv(QM31 x) {
    CM31 norm = qm31_norm(x);
    CM31 inv_norm = cm31_inv(norm);
    QM31 conj = qm31_conj(x);
    // conj * inv_norm (scalar multiply by CM31)
    CM31 ca = qm31_a(conj), cb = qm31_b(conj);
    CM31 ra = cm31_mul(ca, inv_norm);
    CM31 rb = cm31_mul(cb, inv_norm);
    return qm31_from(ra, rb);
}

// QM31 zero/one
__device__ __forceinline__ QM31 qm31_zero() { return {{0, 0, 0, 0}}; }
__device__ __forceinline__ QM31 qm31_one()  { return {{1, 0, 0, 0}}; }

// Embed M31 scalar into QM31
__device__ __forceinline__ QM31 qm31_from_m31(uint32_t x) { return {{x, 0, 0, 0}}; }
