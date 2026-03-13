// Blake2s hashing for Merkle tree commitments.
// Fully unrolled rounds with hardcoded sigma for maximum GPU throughput.

#include "include/m31.cuh"

// Blake2s IV constants
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

// Inline G function operating directly on named variables
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

// Full Blake2s compression with fully unrolled rounds.
// Input: 16 message words m[0..15], initial state h[0..7], counter t, last flag f.
// Output: new state in h[0..7].
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

    // Round 0: sigma[0] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
    G(v0,v4,v8, v12, m0, m1);  G(v1,v5,v9, v13, m2, m3);
    G(v2,v6,v10,v14, m4, m5);  G(v3,v7,v11,v15, m6, m7);
    G(v0,v5,v10,v15, m8, m9);  G(v1,v6,v11,v12, m10,m11);
    G(v2,v7,v8, v13, m12,m13); G(v3,v4,v9, v14, m14,m15);

    // Round 1: sigma[1] = {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3}
    G(v0,v4,v8, v12, m14,m10); G(v1,v5,v9, v13, m4, m8);
    G(v2,v6,v10,v14, m9, m15); G(v3,v7,v11,v15, m13,m6);
    G(v0,v5,v10,v15, m1, m12); G(v1,v6,v11,v12, m0, m2);
    G(v2,v7,v8, v13, m11,m7);  G(v3,v4,v9, v14, m5, m3);

    // Round 2: sigma[2] = {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4}
    G(v0,v4,v8, v12, m11,m8);  G(v1,v5,v9, v13, m12,m0);
    G(v2,v6,v10,v14, m5, m2);  G(v3,v7,v11,v15, m15,m13);
    G(v0,v5,v10,v15, m10,m14); G(v1,v6,v11,v12, m3, m6);
    G(v2,v7,v8, v13, m7, m1);  G(v3,v4,v9, v14, m9, m4);

    // Round 3: sigma[3] = {7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8}
    G(v0,v4,v8, v12, m7, m9);  G(v1,v5,v9, v13, m3, m1);
    G(v2,v6,v10,v14, m13,m12); G(v3,v7,v11,v15, m11,m14);
    G(v0,v5,v10,v15, m2, m6);  G(v1,v6,v11,v12, m5, m10);
    G(v2,v7,v8, v13, m4, m0);  G(v3,v4,v9, v14, m15,m8);

    // Round 4: sigma[4] = {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13}
    G(v0,v4,v8, v12, m9, m0);  G(v1,v5,v9, v13, m5, m7);
    G(v2,v6,v10,v14, m2, m4);  G(v3,v7,v11,v15, m10,m15);
    G(v0,v5,v10,v15, m14,m1);  G(v1,v6,v11,v12, m11,m12);
    G(v2,v7,v8, v13, m6, m8);  G(v3,v4,v9, v14, m3, m13);

    // Round 5: sigma[5] = {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9}
    G(v0,v4,v8, v12, m2, m12); G(v1,v5,v9, v13, m6, m10);
    G(v2,v6,v10,v14, m0, m11); G(v3,v7,v11,v15, m8, m3);
    G(v0,v5,v10,v15, m4, m13); G(v1,v6,v11,v12, m7, m5);
    G(v2,v7,v8, v13, m15,m14); G(v3,v4,v9, v14, m1, m9);

    // Round 6: sigma[6] = {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11}
    G(v0,v4,v8, v12, m12,m5);  G(v1,v5,v9, v13, m1, m15);
    G(v2,v6,v10,v14, m14,m13); G(v3,v7,v11,v15, m4, m10);
    G(v0,v5,v10,v15, m0, m7);  G(v1,v6,v11,v12, m6, m3);
    G(v2,v7,v8, v13, m9, m2);  G(v3,v4,v9, v14, m8, m11);

    // Round 7: sigma[7] = {13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10}
    G(v0,v4,v8, v12, m13,m11); G(v1,v5,v9, v13, m7, m14);
    G(v2,v6,v10,v14, m12,m1);  G(v3,v7,v11,v15, m3, m9);
    G(v0,v5,v10,v15, m5, m0);  G(v1,v6,v11,v12, m15,m4);
    G(v2,v7,v8, v13, m8, m6);  G(v3,v4,v9, v14, m2, m10);

    // Round 8: sigma[8] = {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5}
    G(v0,v4,v8, v12, m6, m15); G(v1,v5,v9, v13, m14,m9);
    G(v2,v6,v10,v14, m11,m3);  G(v3,v7,v11,v15, m0, m8);
    G(v0,v5,v10,v15, m12,m2);  G(v1,v6,v11,v12, m13,m7);
    G(v2,v7,v8, v13, m1, m4);  G(v3,v4,v9, v14, m10,m5);

    // Round 9: sigma[9] = {10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0}
    G(v0,v4,v8, v12, m10,m2);  G(v1,v5,v9, v13, m8, m4);
    G(v2,v6,v10,v14, m7, m6);  G(v3,v7,v11,v15, m1, m5);
    G(v0,v5,v10,v15, m15,m11); G(v1,v6,v11,v12, m9, m14);
    G(v2,v7,v8, v13, m3, m12); G(v3,v4,v9, v14, m13,m0);

    // Finalize
    h0 ^= v0 ^ v8;  h1 ^= v1 ^ v9;  h2 ^= v2 ^ v10; h3 ^= v3 ^ v11;
    h4 ^= v4 ^ v12; h5 ^= v5 ^ v13; h6 ^= v6 ^ v14; h7 ^= v7 ^ v15;
}

// Hash leaf: n_cols M31 values → 32-byte Blake2s hash
__global__ void merkle_hash_leaves_kernel(
    const uint32_t* const* __restrict__ columns,
    uint32_t* __restrict__ hashes,
    uint32_t n_cols,
    uint32_t n_leaves
) {
    uint32_t leaf = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf >= n_leaves) return;

    // Load message words from columns
    uint32_t m0=0,m1=0,m2=0,m3=0,m4=0,m5=0,m6=0,m7=0;
    uint32_t m8=0,m9=0,m10=0,m11=0,m12=0,m13=0,m14=0,m15=0;
    switch (n_cols) {
        default:
        case 16: m15 = columns[15][leaf]; // fallthrough
        case 15: m14 = columns[14][leaf];
        case 14: m13 = columns[13][leaf];
        case 13: m12 = columns[12][leaf];
        case 12: m11 = columns[11][leaf];
        case 11: m10 = columns[10][leaf];
        case 10: m9  = columns[9][leaf];
        case 9:  m8  = columns[8][leaf];
        case 8:  m7  = columns[7][leaf];
        case 7:  m6  = columns[6][leaf];
        case 6:  m5  = columns[5][leaf];
        case 5:  m4  = columns[4][leaf];
        case 4:  m3  = columns[3][leaf];
        case 3:  m2  = columns[2][leaf];
        case 2:  m1  = columns[1][leaf];
        case 1:  m0  = columns[0][leaf];
        case 0:  break;
    }

    // Init state: personalization = 0x01010020 (hash length 32, no key)
    uint32_t h0=IV0^0x01010020, h1=IV1, h2=IV2, h3=IV3;
    uint32_t h4=IV4, h5=IV5, h6=IV6, h7=IV7;

    blake2s_compress(h0,h1,h2,h3,h4,h5,h6,h7,
                     m0,m1,m2,m3,m4,m5,m6,m7,
                     m8,m9,m10,m11,m12,m13,m14,m15,
                     n_cols * 4, 0xFFFFFFFF);

    uint32_t* out = &hashes[leaf * 8];
    out[0]=h0; out[1]=h1; out[2]=h2; out[3]=h3;
    out[4]=h4; out[5]=h5; out[6]=h6; out[7]=h7;
}

// Hash internal node: two 32-byte children → one 32-byte parent
__global__ void merkle_hash_nodes_kernel(
    const uint32_t* __restrict__ children,
    uint32_t* __restrict__ parents,
    uint32_t n_parents
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_parents) return;

    const uint32_t* left = &children[i * 16];
    const uint32_t* right = &children[i * 16 + 8];

    uint32_t h0=IV0^0x01010020, h1=IV1, h2=IV2, h3=IV3;
    uint32_t h4=IV4, h5=IV5, h6=IV6, h7=IV7;

    blake2s_compress(h0,h1,h2,h3,h4,h5,h6,h7,
                     left[0],left[1],left[2],left[3],
                     left[4],left[5],left[6],left[7],
                     right[0],right[1],right[2],right[3],
                     right[4],right[5],right[6],right[7],
                     64, 0xFFFFFFFF);

    uint32_t* out = &parents[i * 8];
    out[0]=h0; out[1]=h1; out[2]=h2; out[3]=h3;
    out[4]=h4; out[5]=h5; out[6]=h6; out[7]=h7;
}

extern "C" {

void cuda_merkle_hash_leaves(
    const uint32_t* const* columns,
    uint32_t* hashes,
    uint32_t n_cols,
    uint32_t n_leaves
) {
    uint32_t threads = 256;
    uint32_t blocks = (n_leaves + threads - 1) / threads;
    merkle_hash_leaves_kernel<<<blocks, threads>>>(columns, hashes, n_cols, n_leaves);
}

void cuda_merkle_hash_nodes(
    const uint32_t* children,
    uint32_t* parents,
    uint32_t n_parents
) {
    uint32_t threads = 256;
    uint32_t blocks = (n_parents + threads - 1) / threads;
    merkle_hash_nodes_kernel<<<blocks, threads>>>(children, parents, n_parents);
}

} // extern "C"
