// Blake2s compression function — shared header for GPU kernels.
#pragma once
#include <stdint.h>

#define IV0 0x6A09E667u
#define IV1 0xBB67AE85u
#define IV2 0x3C6EF372u
#define IV3 0xA54FF53Au
#define IV4 0x510E527Fu
#define IV5 0x9B05688Cu
#define IV6 0x1F83D9ABu
#define IV7 0x5BE0CD19u

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return __funnelshift_r(x, x, n);
}

#define G(va, vb, vc, vd, mx, my) do { \
    va = va + vb + mx; \
    vd = rotr32(vd ^ va, 16); \
    vc = vc + vd; \
    vb = rotr32(vb ^ vc, 12); \
    va = va + vb + my; \
    vd = rotr32(vd ^ va, 8); \
    vc = vc + vd; \
    vb = rotr32(vb ^ vc, 7); \
} while(0)

__device__ __forceinline__ void blake2s_compress(
    uint32_t &h0, uint32_t &h1, uint32_t &h2, uint32_t &h3,
    uint32_t &h4, uint32_t &h5, uint32_t &h6, uint32_t &h7,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3,
    uint32_t m4, uint32_t m5, uint32_t m6, uint32_t m7,
    uint32_t m8, uint32_t m9, uint32_t m10, uint32_t m11,
    uint32_t m12, uint32_t m13, uint32_t m14, uint32_t m15,
    uint32_t t, uint32_t f
) {
    uint32_t v0=h0, v1=h1, v2=h2, v3=h3, v4=h4, v5=h5, v6=h6, v7=h7;
    uint32_t v8=IV0, v9=IV1, v10=IV2, v11=IV3;
    uint32_t v12=IV4^t, v13=IV5, v14=IV6^f, v15=IV7;

    // Round 0
    G(v0,v4,v8,v12,m0,m1); G(v1,v5,v9,v13,m2,m3);
    G(v2,v6,v10,v14,m4,m5); G(v3,v7,v11,v15,m6,m7);
    G(v0,v5,v10,v15,m8,m9); G(v1,v6,v11,v12,m10,m11);
    G(v2,v7,v8,v13,m12,m13); G(v3,v4,v9,v14,m14,m15);
    // Round 1
    G(v0,v4,v8,v12,m14,m10); G(v1,v5,v9,v13,m4,m8);
    G(v2,v6,v10,v14,m9,m15); G(v3,v7,v11,v15,m13,m6);
    G(v0,v5,v10,v15,m1,m12); G(v1,v6,v11,v12,m0,m2);
    G(v2,v7,v8,v13,m11,m7); G(v3,v4,v9,v14,m5,m3);
    // Round 2
    G(v0,v4,v8,v12,m11,m8); G(v1,v5,v9,v13,m12,m0);
    G(v2,v6,v10,v14,m5,m2); G(v3,v7,v11,v15,m15,m13);
    G(v0,v5,v10,v15,m10,m14); G(v1,v6,v11,v12,m3,m6);
    G(v2,v7,v8,v13,m7,m1); G(v3,v4,v9,v14,m9,m4);
    // Round 3
    G(v0,v4,v8,v12,m7,m9); G(v1,v5,v9,v13,m3,m1);
    G(v2,v6,v10,v14,m13,m12); G(v3,v7,v11,v15,m11,m14);
    G(v0,v5,v10,v15,m2,m6); G(v1,v6,v11,v12,m5,m10);
    G(v2,v7,v8,v13,m4,m0); G(v3,v4,v9,v14,m15,m8);
    // Round 4
    G(v0,v4,v8,v12,m9,m0); G(v1,v5,v9,v13,m5,m7);
    G(v2,v6,v10,v14,m2,m4); G(v3,v7,v11,v15,m10,m15);
    G(v0,v5,v10,v15,m14,m1); G(v1,v6,v11,v12,m11,m12);
    G(v2,v7,v8,v13,m6,m8); G(v3,v4,v9,v14,m3,m13);
    // Round 5
    G(v0,v4,v8,v12,m2,m12); G(v1,v5,v9,v13,m6,m10);
    G(v2,v6,v10,v14,m0,m11); G(v3,v7,v11,v15,m8,m3);
    G(v0,v5,v10,v15,m4,m13); G(v1,v6,v11,v12,m7,m5);
    G(v2,v7,v8,v13,m15,m14); G(v3,v4,v9,v14,m1,m9);
    // Round 6
    G(v0,v4,v8,v12,m12,m5); G(v1,v5,v9,v13,m1,m15);
    G(v2,v6,v10,v14,m14,m13); G(v3,v7,v11,v15,m4,m10);
    G(v0,v5,v10,v15,m0,m7); G(v1,v6,v11,v12,m6,m3);
    G(v2,v7,v8,v13,m9,m2); G(v3,v4,v9,v14,m8,m11);
    // Round 7
    G(v0,v4,v8,v12,m13,m11); G(v1,v5,v9,v13,m7,m14);
    G(v2,v6,v10,v14,m12,m1); G(v3,v7,v11,v15,m3,m9);
    G(v0,v5,v10,v15,m5,m0); G(v1,v6,v11,v12,m15,m4);
    G(v2,v7,v8,v13,m8,m6); G(v3,v4,v9,v14,m2,m10);
    // Round 8
    G(v0,v4,v8,v12,m6,m15); G(v1,v5,v9,v13,m14,m9);
    G(v2,v6,v10,v14,m11,m3); G(v3,v7,v11,v15,m0,m8);
    G(v0,v5,v10,v15,m12,m2); G(v1,v6,v11,v12,m13,m7);
    G(v2,v7,v8,v13,m1,m4); G(v3,v4,v9,v14,m10,m5);
    // Round 9
    G(v0,v4,v8,v12,m10,m2); G(v1,v5,v9,v13,m8,m4);
    G(v2,v6,v10,v14,m7,m6); G(v3,v7,v11,v15,m1,m5);
    G(v0,v5,v10,v15,m15,m11); G(v1,v6,v11,v12,m9,m14);
    G(v2,v7,v8,v13,m3,m12); G(v3,v4,v9,v14,m13,m0);

    h0 ^= v0 ^ v8; h1 ^= v1 ^ v9; h2 ^= v2 ^ v10; h3 ^= v3 ^ v11;
    h4 ^= v4 ^ v12; h5 ^= v5 ^ v13; h6 ^= v6 ^ v14; h7 ^= v7 ^ v15;
}
