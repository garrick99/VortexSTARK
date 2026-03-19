// Circle NTT for stwo twiddle format.
// Accepts the flat twiddle buffer produced by stwo's slow_precompute_twiddles.
//
// Twiddle buffer layout (size = coset.size()):
//   [layer_0: n/2 values] [layer_1: n/4 values] ... [layer_{k-1}: 1 value] [pad: 1]
// Each layer's values are x-coordinates of the first half of the coset at that
// doubling level, stored in bit-reversed order.
//
// The circle twiddles (y-coordinates for layer 0) are derived from line_twiddles[0]:
//   For each pair (x, y) in the first line layer:
//     circle_twiddles = [y, -y, -x, x, ...]

#include "include/m31.cuh"

// Butterfly: v0' = v0 + v1*t, v1' = v0 - v1*t
__device__ __forceinline__ void butterfly(uint32_t& v0, uint32_t& v1, uint32_t t) {
    uint32_t tmp = m31_mul(v1, t);
    v1 = m31_sub(v0, tmp);
    v0 = m31_add(v0, tmp);
}

// Inverse butterfly: v0' = v0 + v1, v1' = (v0 - v1)*t
__device__ __forceinline__ void ibutterfly(uint32_t& v0, uint32_t& v1, uint32_t t) {
    uint32_t tmp = v0;
    v0 = m31_add(tmp, v1);
    v1 = m31_mul(m31_sub(tmp, v1), t);
}

// Generic layer kernel for stwo twiddle format (unchanged, used for large-stride layers).
__global__ void stwo_ntt_layer_kernel(
    uint32_t* __restrict__ data,
    const uint32_t* __restrict__ twiddle_ptr,
    uint32_t layer_idx,
    uint32_t half_n,
    int forward
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half_n) return;

    uint32_t stride = 1u << layer_idx;
    uint32_t h = tid >> layer_idx;
    uint32_t l = tid & (stride - 1);
    uint32_t idx0 = (h << (layer_idx + 1)) + l;
    uint32_t idx1 = idx0 + stride;

    uint32_t v0 = data[idx0];
    uint32_t v1 = data[idx1];
    uint32_t t = twiddle_ptr[h];

    if (forward) {
        butterfly(v0, v1, t);
    } else {
        ibutterfly(v0, v1, t);
    }

    data[idx0] = v0;
    data[idx1] = v1;
}

// ============================================================================
// Fused shared-memory NTT kernel: processes multiple butterfly layers in-place
// in shared memory, eliminating global memory round-trips between layers.
//
// Each block processes a tile of `tile_size = 2 << max_layer_idx` elements.
// All butterfly layers from first_layer_idx to last_layer_idx are done in
// shared memory before writing back to global memory once.
//
// For forward NTT: layers are applied from last_layer_idx down to first_layer_idx.
// For inverse NTT: layers are applied from first_layer_idx up to last_layer_idx.
// ============================================================================
__global__ void stwo_ntt_fused_kernel(
    uint32_t* __restrict__ data,
    const uint32_t* __restrict__ d_twiddles,
    uint32_t half_n,
    uint32_t n_line_layers,
    int forward,
    uint32_t first_layer_idx,
    uint32_t last_layer_idx
) {
    extern __shared__ uint32_t smem[];

    const uint32_t tile_size = 2u << last_layer_idx;
    const uint32_t n = half_n * 2;
    const uint32_t tile_base = blockIdx.x * tile_size;
    const uint32_t n_bflies = tile_size / 2;

    // Coalesced load: tile from global to shared memory
    for (uint32_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
        uint32_t gi = tile_base + i;
        smem[i] = (gi < n) ? data[gi] : 0u;
    }
    __syncthreads();

    if (forward) {
        // Forward: large strides first → small strides last
        for (uint32_t layer_idx = last_layer_idx; ; layer_idx--) {
            uint32_t stride = 1u << layer_idx;
            // Twiddle pointer for this layer: k = layer_idx - 1
            uint32_t k = layer_idx - 1;
            const uint32_t* twid_ptr = d_twiddles + (half_n - (1u << (n_line_layers - k)));
            // Global h offset: tile is aligned to 2*stride, so base_h = tile_base / (2*stride)
            uint32_t base_h = tile_base >> (layer_idx + 1);

            for (uint32_t i = threadIdx.x; i < n_bflies; i += blockDim.x) {
                uint32_t h = i >> layer_idx;
                uint32_t l = i & (stride - 1);
                uint32_t idx0 = (h << (layer_idx + 1)) + l;
                uint32_t idx1 = idx0 + stride;
                uint32_t t = twid_ptr[base_h + h];
                butterfly(smem[idx0], smem[idx1], t);
            }
            __syncthreads();
            if (layer_idx == first_layer_idx) break;
        }
    } else {
        // Inverse: small strides first → large strides last
        for (uint32_t layer_idx = first_layer_idx; layer_idx <= last_layer_idx; layer_idx++) {
            uint32_t stride = 1u << layer_idx;
            uint32_t k = layer_idx - 1;
            const uint32_t* twid_ptr = d_twiddles + (half_n - (1u << (n_line_layers - k)));
            uint32_t base_h = tile_base >> (layer_idx + 1);

            for (uint32_t i = threadIdx.x; i < n_bflies; i += blockDim.x) {
                uint32_t h = i >> layer_idx;
                uint32_t l = i & (stride - 1);
                uint32_t idx0 = (h << (layer_idx + 1)) + l;
                uint32_t idx1 = idx0 + stride;
                uint32_t t = twid_ptr[base_h + h];
                ibutterfly(smem[idx0], smem[idx1], t);
            }
            __syncthreads();
        }
    }

    // Coalesced write: shared memory back to global
    for (uint32_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
        uint32_t gi = tile_base + i;
        if (gi < n) data[gi] = smem[i];
    }
}

// ============================================================================
// Radix-4 NTT kernel: fuses two consecutive butterfly layers into one pass.
// Each thread processes 4 elements, doing both layer L (stride S) and
// layer L-1 (stride S/2) without writing intermediate results to global memory.
// Halves global memory traffic for the bandwidth-bound large-stride layers.
// ============================================================================
__global__ void stwo_ntt_radix4_kernel(
    uint32_t* __restrict__ data,
    const uint32_t* __restrict__ twid_hi,    // twiddles for layer L (higher stride)
    const uint32_t* __restrict__ twid_lo,    // twiddles for layer L-1 (lower stride)
    uint32_t layer_idx_hi,                    // L
    uint32_t n_groups,                        // total radix-4 groups = n/4
    int forward
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_groups) return;

    uint32_t S = 1u << layer_idx_hi;          // stride for layer L
    uint32_t S2 = S >> 1;                      // stride for layer L-1

    // Map gid to 4-element group position
    // Super-groups of 2S elements contain S/2 radix-4 groups each
    uint32_t super_idx = gid / S2;
    uint32_t intra = gid % S2;
    uint32_t base = super_idx * (2u * S) + intra;

    // Load 4 elements
    uint32_t v0 = data[base];
    uint32_t v1 = data[base + S2];
    uint32_t v2 = data[base + S];
    uint32_t v3 = data[base + S + S2];

    // Twiddles
    uint32_t t_hi = twid_hi[super_idx];
    uint32_t t_lo0 = twid_lo[super_idx * 2];
    uint32_t t_lo1 = twid_lo[super_idx * 2 + 1];

    if (forward) {
        // Forward: layer L first (larger stride), then layer L-1
        butterfly(v0, v2, t_hi);
        butterfly(v1, v3, t_hi);
        butterfly(v0, v1, t_lo0);
        butterfly(v2, v3, t_lo1);
    } else {
        // Inverse: layer L-1 first, then layer L
        ibutterfly(v0, v1, t_lo0);
        ibutterfly(v2, v3, t_lo1);
        ibutterfly(v0, v2, t_hi);
        ibutterfly(v1, v3, t_hi);
    }

    // Store 4 elements
    data[base] = v0;
    data[base + S2] = v1;
    data[base + S] = v2;
    data[base + S + S2] = v3;
}

// Circle layer kernel with warp-shuffle twiddle dedup.
// 4 consecutive threads share the same (x,y) pair — load once, broadcast via __shfl_sync.
__global__ void stwo_circle_layer_kernel(
    uint32_t* __restrict__ data,
    const uint32_t* __restrict__ first_line_layer,
    uint32_t half_n,
    int forward
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half_n) return;

    uint32_t pair_idx = tid / 4;
    uint32_t sub_idx = tid % 4;
    uint32_t lane = threadIdx.x & 31;

    // Only the first thread of each 4-group loads from global memory
    uint32_t x, y;
    if (sub_idx == 0) {
        x = first_line_layer[pair_idx * 2];
        y = first_line_layer[pair_idx * 2 + 1];
    }
    // Broadcast (x, y) from the group leader to all 4 threads
    uint32_t src_lane = lane & ~3u;
    x = __shfl_sync(0xFFFFFFFF, x, src_lane);
    y = __shfl_sync(0xFFFFFFFF, y, src_lane);

    uint32_t t;
    switch (sub_idx) {
        case 0: t = y; break;
        case 1: t = m31_neg(y); break;
        case 2: t = m31_neg(x); break;
        case 3: t = x; break;
    }

    uint32_t idx0 = tid * 2;
    uint32_t idx1 = idx0 + 1;

    uint32_t v0 = data[idx0];
    uint32_t v1 = data[idx1];

    if (forward) {
        butterfly(v0, v1, t);
    } else {
        ibutterfly(v0, v1, t);
    }

    data[idx0] = v0;
    data[idx1] = v1;
}

// Scale kernel
__global__ void stwo_scale_kernel(uint32_t* data, uint32_t scale, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    data[tid] = m31_mul(data[tid], scale);
}

extern "C" {

// Number of layers to fuse in shared memory. Tile size = 2^(FUSED+1) elements.
// 10 layers → tile of 2048 elements = 8KB shared memory.
// 10 fused layers → tile of 2048 elements = 8KB shared memory.
// Allows 4 blocks/SM at full occupancy (2048 threads/SM).
#define NTT_FUSED_LAYERS 10
#define NTT_FUSED_TILE (2u << NTT_FUSED_LAYERS)   // 2048
#define NTT_FUSED_THREADS 512

// Forward NTT (evaluate): coefficients -> values in bit-reversed order.
void cuda_stwo_ntt_evaluate(
    uint32_t* d_data,
    const uint32_t* d_twiddles,
    uint32_t n
) {
    uint32_t half_n = n / 2;
    uint32_t threads = 256;
    uint32_t blocks = (half_n + threads - 1) / threads;
    uint32_t log_n = 0;
    for (uint32_t tmp = n; tmp > 1; tmp >>= 1) log_n++;
    uint32_t n_line_layers = log_n - 1;

    // Forward: line layers from highest k to lowest k.
    // Large-stride layers (k >= NTT_FUSED_LAYERS) use radix-4 (pairs of 2 layers).
    {
        int k = (int)n_line_layers - 1;
        // If odd number of unfused layers, handle the first one solo
        int n_unfused = k - (int)NTT_FUSED_LAYERS + 1;
        if (n_unfused % 2 == 1) {
            uint32_t twid_offset = half_n - (1u << (n_line_layers - k));
            stwo_ntt_layer_kernel<<<blocks, threads>>>(
                d_data, d_twiddles + twid_offset, (uint32_t)(k + 1), half_n, 1
            );
            k--;
        }
        // Radix-4: process pairs (k, k-1) → (layer_idx=k+1, layer_idx=k)
        uint32_t n_groups = half_n / 2; // n/4 radix-4 groups
        uint32_t r4_blocks = (n_groups + threads - 1) / threads;
        while (k >= (int)NTT_FUSED_LAYERS + 1) {
            uint32_t li_hi = (uint32_t)(k + 1);     // layer_idx for layer k
            uint32_t li_lo = (uint32_t)k;            // layer_idx for layer k-1
            const uint32_t* twid_hi = d_twiddles + (half_n - (1u << (n_line_layers - k)));
            const uint32_t* twid_lo = d_twiddles + (half_n - (1u << (n_line_layers - (k - 1))));
            stwo_ntt_radix4_kernel<<<r4_blocks, threads>>>(
                d_data, twid_hi, twid_lo, li_hi, n_groups, 1
            );
            k -= 2;
        }
        // Handle any remaining single layer
        if (k >= (int)NTT_FUSED_LAYERS) {
            uint32_t twid_offset = half_n - (1u << (n_line_layers - k));
            stwo_ntt_layer_kernel<<<blocks, threads>>>(
                d_data, d_twiddles + twid_offset, (uint32_t)(k + 1), half_n, 1
            );
        }
    }

    // Fused kernel for the last NTT_FUSED_LAYERS layers (small strides, k = FUSED-1 down to 0).
    // These correspond to layer_idx = NTT_FUSED_LAYERS down to 1.
    if (n_line_layers > 0 && NTT_FUSED_LAYERS > 0) {
        uint32_t fused_count = (n_line_layers < NTT_FUSED_LAYERS) ? n_line_layers : NTT_FUSED_LAYERS;
        uint32_t last_li = fused_count;     // largest layer_idx in fused range
        uint32_t first_li = 1;              // smallest layer_idx
        uint32_t tile_size = 2u << last_li;
        uint32_t fused_blocks = n / tile_size;
        uint32_t smem_bytes = tile_size * sizeof(uint32_t);

        stwo_ntt_fused_kernel<<<fused_blocks, NTT_FUSED_THREADS, smem_bytes>>>(
            d_data, d_twiddles, half_n, n_line_layers, 1, first_li, last_li
        );
    }

    // Circle layer
    stwo_circle_layer_kernel<<<blocks, threads>>>(
        d_data, d_twiddles, half_n, 1
    );

    cudaDeviceSynchronize();
}

// Inverse NTT (interpolate): values -> coefficients, with 1/n scaling.
void cuda_stwo_ntt_interpolate(
    uint32_t* d_data,
    const uint32_t* d_itwiddles,
    uint32_t n
) {
    uint32_t half_n = n / 2;
    uint32_t threads = 256;
    uint32_t blocks = (half_n + threads - 1) / threads;
    uint32_t log_n = 0;
    for (uint32_t tmp = n; tmp > 1; tmp >>= 1) log_n++;
    uint32_t n_line_layers = log_n - 1;

    // Circle layer first
    stwo_circle_layer_kernel<<<blocks, threads>>>(
        d_data, d_itwiddles, half_n, 0
    );

    // Fused kernel for the first NTT_FUSED_LAYERS layers (small strides, k = 0 to FUSED-1).
    if (n_line_layers > 0 && NTT_FUSED_LAYERS > 0) {
        uint32_t fused_count = (n_line_layers < NTT_FUSED_LAYERS) ? n_line_layers : NTT_FUSED_LAYERS;
        uint32_t last_li = fused_count;
        uint32_t first_li = 1;
        uint32_t tile_size = 2u << last_li;
        uint32_t fused_blocks = n / tile_size;
        uint32_t smem_bytes = tile_size * sizeof(uint32_t);

        stwo_ntt_fused_kernel<<<fused_blocks, NTT_FUSED_THREADS, smem_bytes>>>(
            d_data, d_itwiddles, half_n, n_line_layers, 0, first_li, last_li
        );
    }

    // Remaining large-stride layers: radix-4 pairs ascending
    {
        uint32_t k = NTT_FUSED_LAYERS;
        uint32_t n_groups = half_n / 2;
        uint32_t r4_blocks = (n_groups + threads - 1) / threads;
        // Radix-4: process pairs (k, k+1) → (layer_idx=k+1, layer_idx=k+2)
        while (k + 1 < n_line_layers) {
            uint32_t li_lo = k + 1;     // layer_idx for layer k (lower stride)
            uint32_t li_hi = k + 2;     // layer_idx for layer k+1 (higher stride)
            const uint32_t* twid_lo = d_itwiddles + (half_n - (1u << (n_line_layers - k)));
            const uint32_t* twid_hi = d_itwiddles + (half_n - (1u << (n_line_layers - (k + 1))));
            stwo_ntt_radix4_kernel<<<r4_blocks, threads>>>(
                d_data, twid_hi, twid_lo, li_hi, n_groups, 0
            );
            k += 2;
        }
        // Handle remaining single layer if odd count
        if (k < n_line_layers) {
            uint32_t twid_offset = half_n - (1u << (n_line_layers - k));
            stwo_ntt_layer_kernel<<<blocks, threads>>>(
                d_data, d_itwiddles + twid_offset, k + 1, half_n, 0
            );
        }
    }

    // Scale by 1/n
    uint32_t exp = (30u * log_n) % 31u;
    uint32_t inv_n = (exp == 0) ? 1u : (1u << exp);
    uint32_t scale_blocks = (n + threads - 1) / threads;
    stwo_scale_kernel<<<scale_blocks, threads>>>(d_data, inv_n, n);

    cudaDeviceSynchronize();
}

} // extern "C"
