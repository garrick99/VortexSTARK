// Blake2s hashing for Merkle tree commitments.
// Leaf hashing (column values → hash) and node hashing (children → parent).

#include "include/m31.cuh"

// Blake2s constants
static __constant__ uint32_t BLAKE2S_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static __constant__ uint8_t BLAKE2S_SIGMA[10][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    {14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3},
    {11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4},
    { 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8},
    { 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13},
    { 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9},
    {12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11},
    {13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10},
    { 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5},
    {10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0},
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void blake2s_g(uint32_t* v, int a, int b, int c, int d, uint32_t x, uint32_t y) {
    v[a] = v[a] + v[b] + x;
    v[d] = rotr(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + y;
    v[d] = rotr(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];
    v[b] = rotr(v[b] ^ v[c], 7);
}

// Hash leaf: n_cols M31 values → 32-byte Blake2s hash
__global__ void merkle_hash_leaves_kernel(
    const uint32_t* const* __restrict__ columns,
    uint32_t* __restrict__ hashes,  // 8 words per leaf
    uint32_t n_cols,
    uint32_t n_leaves
) {
    uint32_t leaf = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf >= n_leaves) return;

    uint32_t v[16];
    uint32_t m[16] = {0};

    for (uint32_t c = 0; c < n_cols && c < 16; c++) {
        m[c] = columns[c][leaf];
    }

    for (int i = 0; i < 8; i++) v[i] = BLAKE2S_IV[i];
    v[0] ^= 0x01010020;
    for (int i = 0; i < 8; i++) v[i + 8] = BLAKE2S_IV[i];

    uint32_t msg_len = n_cols * 4;
    v[12] ^= msg_len;
    v[14] ^= 0xFFFFFFFF;

    for (int r = 0; r < 10; r++) {
        blake2s_g(v, 0, 4,  8, 12, m[BLAKE2S_SIGMA[r][ 0]], m[BLAKE2S_SIGMA[r][ 1]]);
        blake2s_g(v, 1, 5,  9, 13, m[BLAKE2S_SIGMA[r][ 2]], m[BLAKE2S_SIGMA[r][ 3]]);
        blake2s_g(v, 2, 6, 10, 14, m[BLAKE2S_SIGMA[r][ 4]], m[BLAKE2S_SIGMA[r][ 5]]);
        blake2s_g(v, 3, 7, 11, 15, m[BLAKE2S_SIGMA[r][ 6]], m[BLAKE2S_SIGMA[r][ 7]]);
        blake2s_g(v, 0, 5, 10, 15, m[BLAKE2S_SIGMA[r][ 8]], m[BLAKE2S_SIGMA[r][ 9]]);
        blake2s_g(v, 1, 6, 11, 12, m[BLAKE2S_SIGMA[r][10]], m[BLAKE2S_SIGMA[r][11]]);
        blake2s_g(v, 2, 7,  8, 13, m[BLAKE2S_SIGMA[r][12]], m[BLAKE2S_SIGMA[r][13]]);
        blake2s_g(v, 3, 4,  9, 14, m[BLAKE2S_SIGMA[r][14]], m[BLAKE2S_SIGMA[r][15]]);
    }

    uint32_t* out = &hashes[leaf * 8];
    for (int i = 0; i < 8; i++) {
        out[i] = BLAKE2S_IV[i] ^ v[i] ^ v[i + 8];
    }
    out[0] ^= 0x01010020;
}

// Hash internal node: two 32-byte children → one 32-byte parent
__global__ void merkle_hash_nodes_kernel(
    const uint32_t* __restrict__ children,  // 16 words per pair
    uint32_t* __restrict__ parents,         // 8 words per parent
    uint32_t n_parents
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_parents) return;

    const uint32_t* left = &children[i * 16];
    const uint32_t* right = &children[i * 16 + 8];
    uint32_t* out = &parents[i * 8];

    uint32_t v[16];
    uint32_t m[16];

    for (int j = 0; j < 8; j++) {
        m[j] = left[j];
        m[j + 8] = right[j];
    }

    for (int j = 0; j < 8; j++) v[j] = BLAKE2S_IV[j];
    v[0] ^= 0x01010020;
    for (int j = 0; j < 8; j++) v[j + 8] = BLAKE2S_IV[j];
    v[12] ^= 64;
    v[14] ^= 0xFFFFFFFF;

    for (int r = 0; r < 10; r++) {
        blake2s_g(v, 0, 4,  8, 12, m[BLAKE2S_SIGMA[r][ 0]], m[BLAKE2S_SIGMA[r][ 1]]);
        blake2s_g(v, 1, 5,  9, 13, m[BLAKE2S_SIGMA[r][ 2]], m[BLAKE2S_SIGMA[r][ 3]]);
        blake2s_g(v, 2, 6, 10, 14, m[BLAKE2S_SIGMA[r][ 4]], m[BLAKE2S_SIGMA[r][ 5]]);
        blake2s_g(v, 3, 7, 11, 15, m[BLAKE2S_SIGMA[r][ 6]], m[BLAKE2S_SIGMA[r][ 7]]);
        blake2s_g(v, 0, 5, 10, 15, m[BLAKE2S_SIGMA[r][ 8]], m[BLAKE2S_SIGMA[r][ 9]]);
        blake2s_g(v, 1, 6, 11, 12, m[BLAKE2S_SIGMA[r][10]], m[BLAKE2S_SIGMA[r][11]]);
        blake2s_g(v, 2, 7,  8, 13, m[BLAKE2S_SIGMA[r][12]], m[BLAKE2S_SIGMA[r][13]]);
        blake2s_g(v, 3, 4,  9, 14, m[BLAKE2S_SIGMA[r][14]], m[BLAKE2S_SIGMA[r][15]]);
    }

    for (int j = 0; j < 8; j++) {
        out[j] = BLAKE2S_IV[j] ^ v[j] ^ v[j + 8];
    }
    out[0] ^= 0x01010020;
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
