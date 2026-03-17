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

// Generic layer kernel for stwo twiddle format.
// layer_idx: butterfly stride = 2^layer_idx
// twiddle_ptr: pointer to the start of this layer's twiddles
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

// Circle layer kernel: derives y-twiddles from the first line layer.
// line_twiddles[0] stores pairs [x0, y0, x1, y1, ...] in the twiddle buffer.
// The circle twiddles for index h are: [y, -y, -x, x] cycling every 4.
__global__ void stwo_circle_layer_kernel(
    uint32_t* __restrict__ data,
    const uint32_t* __restrict__ first_line_layer, // line_twiddles[0], n/4 values
    uint32_t half_n,
    int forward
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half_n) return;

    // Derive circle twiddle from first line layer.
    // first_line_layer has pairs: [x0, y0, x1, y1, ...]
    // circle_twiddles = [y0, -y0, -x0, x0, y1, -y1, -x1, x1, ...]
    uint32_t pair_idx = tid / 4;
    uint32_t sub_idx = tid % 4;

    uint32_t x = first_line_layer[pair_idx * 2];
    uint32_t y = first_line_layer[pair_idx * 2 + 1];

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

// Forward NTT (evaluate): coefficients -> values in bit-reversed order.
// d_twiddles: flat stwo twiddle buffer (n values)
// n: domain size (must be power of 2, >= 8)
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

    // Twiddle buffer layout:
    // [layer_0: n/2] [layer_1: n/4] ... [layer_{k-1}: 1] [pad: 1]
    // Layer i starts at offset: n/2 + n/4 + ... + n/2^(i+1) = n - n/2^i
    // Wait, let me recalculate:
    // layer_0 occupies indices [0..n/2)
    // layer_1 occupies indices [n/2..n/2+n/4)
    // layer_k occupies indices [n-2^(log_n-k)..n-2^(log_n-k-1))
    // Actually: total = sum_{i=0}^{log_n-1} 2^{log_n-1-i} = n-1, plus 1 pad = n
    //
    // For domain_line_twiddles_from_tree, layer ordering is:
    //   line_twiddles[0] = largest layer (n/2 values at buffer start)
    //   line_twiddles[k] = layer at buffer[n-2^(k+1)..n-2^k]
    //
    // But domain_line_twiddles_from_tree reverses, so:
    //   line_twiddles[0] = &buffer[0..n/2]     (largest, n/2 values)
    //   line_twiddles[1] = &buffer[n/2..3n/4]  (n/4 values)
    //   line_twiddles[k] = offset_of_layer(k)

    // Forward evaluate: line layers highest to lowest, then circle layer.
    // line_twiddles[k] has 2^(log_n-1-k) values
    // For the stwo buffer: layer k occupies indices [cumsum..cumsum+size)
    // where cumsum = sum_{j=0}^{k-1} 2^{log_n-1-j}

    uint32_t n_line_layers = log_n - 1;

    // Forward evaluate: line layers from highest (smallest, 1 twiddle) to
    // lowest (largest, n/2 twiddles), then circle layer.
    //
    // line_twiddles[k] = &buffer[n - 2^(k+1) .. n - 2^k]
    // (from domain_line_twiddles_from_tree with .rev())
    // Twiddle buffer has half_n values (= n/2).
    // line_twiddles[k] = buf[half_n - 2^(k+1) .. half_n - 2^k]
    for (int k = (int)n_line_layers - 1; k >= 0; k--) {
        uint32_t twid_offset = half_n - (2u << k); // half_n - 2^(k+1)
        stwo_ntt_layer_kernel<<<blocks, threads>>>(
            d_data, d_twiddles + twid_offset, (uint32_t)(k + 1), half_n, 1
        );
    }

    // Circle layer: derived from line_twiddles[0] = buf[0..half_n/2]
    // (first half_n/2 values, which are n/4 values = half of the largest line layer)
    // Wait: line_twiddles[0] has half_n/2 values at buf[0..half_n/2].
    // But circle_twiddles_from_line_twiddles takes ALL of line_twiddles[0].
    // line_twiddles[0] = buf[half_n - 2^(n_line_layers)..half_n - 2^(n_line_layers-1)]
    // = buf[0..half_n/2] for n_line_layers = log_n - 1
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

    // Interpolate: line layers from lowest (largest) to highest (smallest).
    for (uint32_t k = 0; k < n_line_layers; k++) {
        uint32_t twid_offset = half_n - (2u << k); // half_n - 2^(k+1)
        stwo_ntt_layer_kernel<<<blocks, threads>>>(
            d_data, d_itwiddles + twid_offset, k + 1, half_n, 0
        );
    }

    // Scale by 1/n
    uint32_t exp = (30u * log_n) % 31u;
    uint32_t inv_n = (exp == 0) ? 1u : (1u << exp);
    uint32_t scale_blocks = (n + threads - 1) / threads;
    stwo_scale_kernel<<<scale_blocks, threads>>>(d_data, inv_n, n);

    cudaDeviceSynchronize();
}

} // extern "C"
