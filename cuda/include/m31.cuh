#pragma once
#include <cstdint>

// M31: Mersenne-31 prime field (p = 2^31 - 1)
#define M31_P 0x7FFFFFFFu
#define M31_ZERO 0u
#define M31_ONE 1u

__device__ __forceinline__ uint32_t m31_reduce(uint64_t x) {
    uint32_t lo = (uint32_t)(x & M31_P);
    uint32_t hi = (uint32_t)(x >> 31);
    uint32_t r = lo + hi;
    return r >= M31_P ? r - M31_P : r;
}

__device__ __forceinline__ uint32_t m31_add(uint32_t a, uint32_t b) {
    uint32_t r = a + b;
    return r >= M31_P ? r - M31_P : r;
}

__device__ __forceinline__ uint32_t m31_sub(uint32_t a, uint32_t b) {
    uint32_t r = a - b;
    // If underflow, wrap around
    return (a >= b) ? r : r + M31_P;
}

__device__ __forceinline__ uint32_t m31_mul(uint32_t a, uint32_t b) {
    return m31_reduce((uint64_t)a * b);
}

__device__ __forceinline__ uint32_t m31_neg(uint32_t a) {
    return a == 0 ? 0 : M31_P - a;
}

// Fermat's little theorem: a^(-1) = a^(p-2) mod p
__device__ __forceinline__ uint32_t m31_inv(uint32_t a) {
    // p-2 = 2^31 - 3 = 0x7FFFFFFD
    // Square-and-multiply
    uint32_t result = 1;
    uint32_t base = a;
    uint32_t exp = M31_P - 2;
    while (exp > 0) {
        if (exp & 1) result = m31_mul(result, base);
        base = m31_mul(base, base);
        exp >>= 1;
    }
    return result;
}

// Square
__device__ __forceinline__ uint32_t m31_sqr(uint32_t a) {
    return m31_reduce((uint64_t)a * a);
}
