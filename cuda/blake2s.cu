// Blake2s hashing for Merkle tree commitments.
// Uses shared blake2s_compress from include/blake2s.cuh.

#include "include/blake2s.cuh"

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

// Single-kernel Merkle commit for small 4-column SoA trees (≤ 2048 leaves).
// Does fused leaf hash + full tree reduction in shared memory — ONE kernel launch.
// Each block handles the entire tree. Only block 0 writes the root.
__global__ void merkle_commit_small_soa4_kernel(
    const uint32_t* __restrict__ col0,
    const uint32_t* __restrict__ col1,
    const uint32_t* __restrict__ col2,
    const uint32_t* __restrict__ col3,
    uint32_t* __restrict__ root_out, // [8] — only written by thread 0
    uint32_t n_leaves
) {
    // Shared memory: enough for n_leaves/2 hashes (after first merge)
    // Max 1024 hashes * 8 words = 8192 words = 32KB
    extern __shared__ uint32_t smem[];

    uint32_t tid = threadIdx.x;
    uint32_t n_pairs = n_leaves / 2;

    // Step 1: Each thread hashes a pair of leaves and merges them
    if (tid < n_pairs) {
        uint32_t left_idx = tid * 2;
        uint32_t right_idx = tid * 2 + 1;

        // Hash left leaf
        uint32_t lh0=IV0^0x01010020, lh1=IV1, lh2=IV2, lh3=IV3;
        uint32_t lh4=IV4, lh5=IV5, lh6=IV6, lh7=IV7;
        blake2s_compress(lh0,lh1,lh2,lh3,lh4,lh5,lh6,lh7,
                         col0[left_idx], col1[left_idx], col2[left_idx], col3[left_idx],
                         0,0,0,0, 0,0,0,0, 0,0,0,0,
                         16, 0xFFFFFFFF);

        // Hash right leaf
        uint32_t rh0=IV0^0x01010020, rh1=IV1, rh2=IV2, rh3=IV3;
        uint32_t rh4=IV4, rh5=IV5, rh6=IV6, rh7=IV7;
        blake2s_compress(rh0,rh1,rh2,rh3,rh4,rh5,rh6,rh7,
                         col0[right_idx], col1[right_idx], col2[right_idx], col3[right_idx],
                         0,0,0,0, 0,0,0,0, 0,0,0,0,
                         16, 0xFFFFFFFF);

        // Merge: hash the two leaf hashes
        uint32_t ph0=IV0^0x01010020, ph1=IV1, ph2=IV2, ph3=IV3;
        uint32_t ph4=IV4, ph5=IV5, ph6=IV6, ph7=IV7;
        blake2s_compress(ph0,ph1,ph2,ph3,ph4,ph5,ph6,ph7,
                         lh0,lh1,lh2,lh3,lh4,lh5,lh6,lh7,
                         rh0,rh1,rh2,rh3,rh4,rh5,rh6,rh7,
                         64, 0xFFFFFFFF);

        // Store in shared memory
        uint32_t* dst = &smem[tid * 8];
        dst[0]=ph0; dst[1]=ph1; dst[2]=ph2; dst[3]=ph3;
        dst[4]=ph4; dst[5]=ph5; dst[6]=ph6; dst[7]=ph7;
    }
    __syncthreads();

    // Step 2: Tree reduction in shared memory
    uint32_t level_size = n_pairs;
    while (level_size > 1) {
        uint32_t half = level_size / 2;
        if (tid < half) {
            const uint32_t* left = &smem[tid * 16];
            const uint32_t* right = &smem[tid * 16 + 8];

            uint32_t h0=IV0^0x01010020, h1=IV1, h2=IV2, h3=IV3;
            uint32_t h4=IV4, h5=IV5, h6=IV6, h7=IV7;
            blake2s_compress(h0,h1,h2,h3,h4,h5,h6,h7,
                             left[0],left[1],left[2],left[3],
                             left[4],left[5],left[6],left[7],
                             right[0],right[1],right[2],right[3],
                             right[4],right[5],right[6],right[7],
                             64, 0xFFFFFFFF);

            uint32_t* dst = &smem[tid * 8];
            dst[0]=h0; dst[1]=h1; dst[2]=h2; dst[3]=h3;
            dst[4]=h4; dst[5]=h5; dst[6]=h6; dst[7]=h7;
        }
        __syncthreads();
        level_size = half;
    }

    // Write root from thread 0
    if (tid == 0) {
        root_out[0]=smem[0]; root_out[1]=smem[1]; root_out[2]=smem[2]; root_out[3]=smem[3];
        root_out[4]=smem[4]; root_out[5]=smem[5]; root_out[6]=smem[6]; root_out[7]=smem[7];
    }
}

// Fused leaf hash + first node level for 4-column SoA (SecureColumn / QM31).
// Each thread processes a PAIR of leaves: hashes both leaves, then hashes
// the two leaf hashes together to produce one parent node.
// Eliminates the leaf hash buffer and one kernel launch.
__global__ void merkle_hash_leaves_and_merge_soa4_kernel(
    const uint32_t* __restrict__ col0,
    const uint32_t* __restrict__ col1,
    const uint32_t* __restrict__ col2,
    const uint32_t* __restrict__ col3,
    uint32_t* __restrict__ parents,
    uint32_t n_pairs // = n_leaves / 2
) {
    uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= n_pairs) return;

    uint32_t left_idx = pair * 2;
    uint32_t right_idx = pair * 2 + 1;

    // Hash left leaf (4 words input)
    uint32_t lh0=IV0^0x01010020, lh1=IV1, lh2=IV2, lh3=IV3;
    uint32_t lh4=IV4, lh5=IV5, lh6=IV6, lh7=IV7;
    blake2s_compress(lh0,lh1,lh2,lh3,lh4,lh5,lh6,lh7,
                     col0[left_idx], col1[left_idx], col2[left_idx], col3[left_idx],
                     0,0,0,0, 0,0,0,0, 0,0,0,0,
                     16, 0xFFFFFFFF);

    // Hash right leaf (4 words input)
    uint32_t rh0=IV0^0x01010020, rh1=IV1, rh2=IV2, rh3=IV3;
    uint32_t rh4=IV4, rh5=IV5, rh6=IV6, rh7=IV7;
    blake2s_compress(rh0,rh1,rh2,rh3,rh4,rh5,rh6,rh7,
                     col0[right_idx], col1[right_idx], col2[right_idx], col3[right_idx],
                     0,0,0,0, 0,0,0,0, 0,0,0,0,
                     16, 0xFFFFFFFF);

    // Hash the two leaf hashes together (64 bytes = 16 words input)
    uint32_t ph0=IV0^0x01010020, ph1=IV1, ph2=IV2, ph3=IV3;
    uint32_t ph4=IV4, ph5=IV5, ph6=IV6, ph7=IV7;
    blake2s_compress(ph0,ph1,ph2,ph3,ph4,ph5,ph6,ph7,
                     lh0,lh1,lh2,lh3,lh4,lh5,lh6,lh7,
                     rh0,rh1,rh2,rh3,rh4,rh5,rh6,rh7,
                     64, 0xFFFFFFFF);

    uint32_t* out = &parents[pair * 8];
    out[0]=ph0; out[1]=ph1; out[2]=ph2; out[3]=ph3;
    out[4]=ph4; out[5]=ph5; out[6]=ph6; out[7]=ph7;
}

// Reduce a small hash array (≤ 1024 nodes) to a single root in shared memory.
// Input: n_nodes hashes at nodes[], output: 8-word root at root_out[].
__global__ void merkle_reduce_to_root_kernel(
    const uint32_t* __restrict__ nodes,
    uint32_t* __restrict__ root_out,
    uint32_t n_nodes
) {
    extern __shared__ uint32_t smem[];
    uint32_t tid = threadIdx.x;

    // Load nodes into shared memory
    if (tid < n_nodes) {
        const uint32_t* src = &nodes[tid * 8];
        uint32_t* dst = &smem[tid * 8];
        dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3];
        dst[4]=src[4]; dst[5]=src[5]; dst[6]=src[6]; dst[7]=src[7];
    }
    __syncthreads();

    // Tree reduction
    uint32_t level_size = n_nodes;
    while (level_size > 1) {
        uint32_t half = level_size / 2;
        if (tid < half) {
            const uint32_t* left = &smem[tid * 16];
            const uint32_t* right = &smem[tid * 16 + 8];

            uint32_t h0=IV0^0x01010020, h1=IV1, h2=IV2, h3=IV3;
            uint32_t h4=IV4, h5=IV5, h6=IV6, h7=IV7;
            blake2s_compress(h0,h1,h2,h3,h4,h5,h6,h7,
                             left[0],left[1],left[2],left[3],
                             left[4],left[5],left[6],left[7],
                             right[0],right[1],right[2],right[3],
                             right[4],right[5],right[6],right[7],
                             64, 0xFFFFFFFF);

            uint32_t* dst = &smem[tid * 8];
            dst[0]=h0; dst[1]=h1; dst[2]=h2; dst[3]=h3;
            dst[4]=h4; dst[5]=h5; dst[6]=h6; dst[7]=h7;
        }
        __syncthreads();
        level_size = half;
    }

    if (tid == 0) {
        root_out[0]=smem[0]; root_out[1]=smem[1]; root_out[2]=smem[2]; root_out[3]=smem[3];
        root_out[4]=smem[4]; root_out[5]=smem[5]; root_out[6]=smem[6]; root_out[7]=smem[7];
    }
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

// Reduce node hash array (≤ 1024 nodes) to root in a single kernel.
void cuda_merkle_reduce_to_root(
    const uint32_t* nodes,
    uint32_t* root_out,
    uint32_t n_nodes
) {
    uint32_t smem_bytes = n_nodes * 8 * sizeof(uint32_t);
    merkle_reduce_to_root_kernel<<<1, n_nodes, smem_bytes>>>(nodes, root_out, n_nodes);
}

// Single-kernel small Merkle commit (≤ 2048 leaves).
// Writes 8-word root directly to device memory pointed by root_out.
void cuda_merkle_commit_small_soa4(
    const uint32_t* col0, const uint32_t* col1,
    const uint32_t* col2, const uint32_t* col3,
    uint32_t* root_out,
    uint32_t n_leaves
) {
    uint32_t n_pairs = n_leaves / 2;
    uint32_t smem_bytes = n_pairs * 8 * sizeof(uint32_t);
    merkle_commit_small_soa4_kernel<<<1, n_pairs, smem_bytes>>>(
        col0, col1, col2, col3, root_out, n_leaves
    );
}

// Fused leaf hash + first node merge for 4-column SoA data.
// Produces n_leaves/2 parent hashes (skipping the leaf hash buffer entirely).
void cuda_merkle_hash_leaves_merge_soa4(
    const uint32_t* col0, const uint32_t* col1,
    const uint32_t* col2, const uint32_t* col3,
    uint32_t* parents,
    uint32_t n_leaves
) {
    uint32_t n_pairs = n_leaves / 2;
    uint32_t threads = 256;
    uint32_t blocks = (n_pairs + threads - 1) / threads;
    merkle_hash_leaves_and_merge_soa4_kernel<<<blocks, threads>>>(
        col0, col1, col2, col3, parents, n_pairs
    );
}

} // extern "C"
