// GPU lifted Merkle leaf hashing for Blake2s.
//
// Replaces the CPU fallback in build_leaves. Each thread hashes one leaf
// by iterating through column groups, computing the lifted row index,
// loading values, and feeding them into Blake2s.
//
// The lifting index formula: for a column of log_size L in a tree of
// lifting_log_size S, leaf `idx` reads from row:
//   row = (idx >> (S - L + 1) << 1) | (idx & 1)
//
// This composes across any number of intermediate expansion stages,
// so the kernel doesn't need to know about column grouping — just
// each column's log_size.

#include "include/blake2s.cuh"

// A chunk of up to 16 columns at the same log_size.
// 16 columns = 64 bytes = one Blake2s compression block.
struct LeafHashChunk {
    uint32_t col_indices[16]; // indices into col_ptrs array
    uint32_t n_cols;          // number of valid columns (1-16)
    uint32_t log_size;        // log2(column length) for all columns in this chunk
};

// Compute the lifted row index for a leaf.
__device__ __forceinline__
uint32_t lifted_row(uint32_t leaf_idx, uint32_t lifting_log, uint32_t col_log) {
    if (col_log == lifting_log) return leaf_idx;
    uint32_t shift = lifting_log - col_log + 1;
    return ((leaf_idx >> shift) << 1) | (leaf_idx & 1);
}

__global__ void build_leaves_lifted_kernel(
    const uint32_t* const* __restrict__ col_ptrs,
    const LeafHashChunk* __restrict__ schedule,
    uint32_t n_chunks,
    uint32_t lifting_log_size,
    uint32_t* __restrict__ output_hashes,
    uint32_t n_leaves
) {
    uint32_t leaf = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf >= n_leaves) return;

    // Init Blake2s state: personalization = 0x01010020 (hash length 32, no key)
    uint32_t h0 = IV0 ^ 0x01010020, h1 = IV1, h2 = IV2, h3 = IV3;
    uint32_t h4 = IV4, h5 = IV5, h6 = IV6, h7 = IV7;
    uint32_t total_bytes = 0;

    // Accumulate values across chunks into a 16-word message buffer.
    // Only compress when we have a full 64-byte block (16 words) or at the end.
    // This matches the blake2 crate's buffering semantics.
    uint32_t buf[16] = {0};
    uint32_t buf_pos = 0;  // words in buffer (0-16)

    for (uint32_t chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++) {
        LeafHashChunk chunk = schedule[chunk_idx];
        uint32_t row = lifted_row(leaf, lifting_log_size, chunk.log_size);

        for (uint32_t c = 0; c < chunk.n_cols; c++) {
            buf[buf_pos++] = col_ptrs[chunk.col_indices[c]][row];

            if (buf_pos == 16) {
                // Full block — compress (not final).
                total_bytes += 64;
                blake2s_compress(h0, h1, h2, h3, h4, h5, h6, h7,
                                 buf[0], buf[1], buf[2], buf[3],
                                 buf[4], buf[5], buf[6], buf[7],
                                 buf[8], buf[9], buf[10], buf[11],
                                 buf[12], buf[13], buf[14], buf[15],
                                 total_bytes, 0);
                buf_pos = 0;
                // Clear buffer for next block
                for (int j = 0; j < 16; j++) buf[j] = 0;
            }
        }
    }

    // Final compression: remaining buffered data (padded with zeros).
    total_bytes += buf_pos * 4;
    blake2s_compress(h0, h1, h2, h3, h4, h5, h6, h7,
                     buf[0], buf[1], buf[2], buf[3],
                     buf[4], buf[5], buf[6], buf[7],
                     buf[8], buf[9], buf[10], buf[11],
                     buf[12], buf[13], buf[14], buf[15],
                     total_bytes, 0xFFFFFFFF);

    // Write the 8-word hash to output.
    uint32_t* out = &output_hashes[leaf * 8];
    out[0] = h0; out[1] = h1; out[2] = h2; out[3] = h3;
    out[4] = h4; out[5] = h5; out[6] = h6; out[7] = h7;
}

// ─── C wrapper for Rust FFI ─────────────────────────────────────────────

extern "C" {

void cuda_build_leaves_lifted(
    const uint32_t* const* col_ptrs,
    const void* schedule,       // LeafHashChunk array
    uint32_t n_chunks,
    uint32_t lifting_log_size,
    uint32_t* output_hashes,
    uint32_t n_leaves
) {
    uint32_t threads = 256;
    uint32_t blocks = (n_leaves + threads - 1) / threads;
    build_leaves_lifted_kernel<<<blocks, threads>>>(
        col_ptrs,
        (const LeafHashChunk*)schedule,
        n_chunks, lifting_log_size,
        output_hashes, n_leaves
    );
}

} // extern "C"
