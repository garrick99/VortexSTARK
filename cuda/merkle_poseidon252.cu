// GPU Poseidon252 Merkle tree for lifted column hashing.
//
// Matches starknet-crypto's poseidon_hash_many sponge exactly:
//   state = [0, 0, 0]
//   for each pair (fe0, fe1) of packed FieldElement252s:
//       state[0] += fe0; state[1] += fe1; permute
//   if odd remainder re:
//       state[0] += re; state[1] += 1; permute
//   else:
//       state[0] += 1; permute
//   return state[0]
//
// Column values (M31) are packed into FieldElement252s via construct_felt252_from_m31s:
//   8 M31s → 1 FE252:  v[0]*2^(31*7) + ... + v[7]  (big-endian, no length padding)
//   n<8 M31s → 1 FE252: same packing + n*2^248  (length padding in top bits)
//
// All state/FE252 values are in Montgomery form throughout.
//
// Entry points (extern "C"):
//   build_leaves_poseidon252(col_ptrs, col_log_sizes, n_cols, lifting_log_size,
//                            output_hashes, n_leaves)
//   build_next_layer_poseidon252(prev_layer, output, n_parents)

#include "include/poseidon252.cuh"

// ── Lifted row index ─────────────────────────────────────────────────────────

__device__ __forceinline__
uint32_t poseidon_lifted_row(uint32_t leaf_idx, uint32_t lifting_log, uint32_t col_log) {
    if (col_log == lifting_log) return leaf_idx;
    uint32_t shift = lifting_log - col_log + 1;
    return ((leaf_idx >> shift) << 1) | (leaf_idx & 1);
}

// ── Leaf hashing kernel ──────────────────────────────────────────────────────

__global__ void build_leaves_poseidon252_kernel(
    const uint32_t* const* __restrict__ col_ptrs,
    const uint32_t*        __restrict__ col_log_sizes,
    uint32_t n_cols,
    uint32_t lifting_log_size,
    uint64_t* __restrict__ output_hashes,   // n_leaves × 4 uint64_t
    uint32_t n_leaves
) {
    uint32_t leaf = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf >= n_leaves) return;

    // Sponge state (zero in any form = zero)
    Fp252 s[3] = {mont_zero(), mont_zero(), mont_zero()};

    // M31 buffer: one block of up to 8 values
    uint32_t m31_buf[8];
    int m31_count = 0;

    // Pending first-of-pair FieldElement252
    bool fe0_ready = false;
    Fp252 fe0;

    for (uint32_t ci = 0; ci < n_cols; ci++) {
        uint32_t col_log = col_log_sizes[ci];
        uint32_t row = poseidon_lifted_row(leaf, lifting_log_size, col_log);
        m31_buf[m31_count++] = col_ptrs[ci][row];

        if (m31_count == 8) {
            // Full block: pack into Montgomery-form FE252 (no length padding when n==8)
            Fp252 fe = pack_m31s_mont(m31_buf, 8);
            if (!fe0_ready) {
                fe0 = fe;
                fe0_ready = true;
            } else {
                // Have a complete pair — absorb
                s[0] = fp_add(s[0], fe0);
                s[1] = fp_add(s[1], fe);
                poseidon_permute_mont(s);
                fe0_ready = false;
            }
            m31_count = 0;
        }
    }

    // Finalize: matches poseidon_hash_many's r = iter.remainder() logic.
    //   r.len()==0 (even total FEs): state[0] += ONE; permute
    //   r.len()==1 (odd total FEs):  state[0] += r[0]; state[1] += ONE; permute
    if (m31_count > 0) {
        // Partial block with length padding
        Fp252 fe_partial = pack_m31s_mont(m31_buf, m31_count);
        if (fe0_ready) {
            // fe0 + fe_partial form a pair (even total)
            s[0] = fp_add(s[0], fe0);
            s[1] = fp_add(s[1], fe_partial);
            poseidon_permute_mont(s);
            // Final even padding
            s[0] = fp_add(s[0], mont_one());
            poseidon_permute_mont(s);
        } else {
            // fe_partial alone (odd total, r.len()==1)
            s[0] = fp_add(s[0], fe_partial);
            s[1] = fp_add(s[1], mont_one());
            poseidon_permute_mont(s);
        }
    } else if (fe0_ready) {
        // fe0 alone (odd total, r.len()==1)
        s[0] = fp_add(s[0], fe0);
        s[1] = fp_add(s[1], mont_one());
        poseidon_permute_mont(s);
    } else {
        // No remainder (even total, including zero)
        s[0] = fp_add(s[0], mont_one());
        poseidon_permute_mont(s);
    }

    // Write result (state[0] in Montgomery form = FieldElement252 internal repr)
    uint32_t out_base = leaf * 4;
    output_hashes[out_base + 0] = s[0].v[0];
    output_hashes[out_base + 1] = s[0].v[1];
    output_hashes[out_base + 2] = s[0].v[2];
    output_hashes[out_base + 3] = s[0].v[3];
}

// ── Upper layer (parent node) hashing ───────────────────────────────────────

__global__ void build_next_layer_poseidon252_kernel(
    const uint64_t* __restrict__ prev_layer,   // 2*n_parents × 4 uint64_t
    uint64_t*       __restrict__ output,       // n_parents × 4 uint64_t
    uint32_t n_parents
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_parents) return;

    uint32_t left_base  = (2 * i)     * 4;
    uint32_t right_base = (2 * i + 1) * 4;

    Fp252 left  = {{prev_layer[left_base],   prev_layer[left_base+1],
                    prev_layer[left_base+2],  prev_layer[left_base+3]}};
    Fp252 right = {{prev_layer[right_base],  prev_layer[right_base+1],
                    prev_layer[right_base+2], prev_layer[right_base+3]}};

    Fp252 parent = poseidon_hash_pair_mont(left, right);

    uint32_t out_base = i * 4;
    output[out_base + 0] = parent.v[0];
    output[out_base + 1] = parent.v[1];
    output[out_base + 2] = parent.v[2];
    output[out_base + 3] = parent.v[3];
}

// ── Host-callable entry points ───────────────────────────────────────────────

extern "C" {

void build_leaves_poseidon252(
    const uint32_t** col_ptrs_dev,
    const uint32_t*  col_log_sizes_dev,
    uint32_t n_cols,
    uint32_t lifting_log_size,
    uint64_t* output_hashes_dev,
    uint32_t n_leaves
) {
    int threads = 256;
    int blocks  = (n_leaves + threads - 1) / threads;
    build_leaves_poseidon252_kernel<<<blocks, threads>>>(
        col_ptrs_dev, col_log_sizes_dev, n_cols, lifting_log_size,
        output_hashes_dev, n_leaves);
}

void build_next_layer_poseidon252(
    const uint64_t* prev_layer_dev,
    uint64_t*       output_dev,
    uint32_t n_parents
) {
    int threads = 256;
    int blocks  = (n_parents + threads - 1) / threads;
    build_next_layer_poseidon252_kernel<<<blocks, threads>>>(
        prev_layer_dev, output_dev, n_parents);
}

} // extern "C"
