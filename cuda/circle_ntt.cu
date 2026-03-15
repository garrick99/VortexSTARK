// M31 Circle NTT for VortexSTARK.
// Forward (evaluate) and inverse (interpolate), single and batched.
// Butterfly structure matches Circle STARK NTT exactly.

#include "include/qm31.cuh"

// Forward butterfly: v0' = v0 + v1*t, v1' = v0 - v1*t
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

// Single-column NTT layer kernel
__global__ void circle_ntt_layer_kernel(
    uint32_t* __restrict__ data,
    const uint32_t* __restrict__ twiddles,
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
    uint32_t t = twiddles[h];

    if (forward) {
        butterfly(v0, v1, t);
    } else {
        ibutterfly(v0, v1, t);
    }

    data[idx0] = v0;
    data[idx1] = v1;
}

// Batched NTT layer kernel: processes multiple columns in one launch.
__global__ void circle_ntt_batch_layer_kernel(
    uint32_t** __restrict__ columns,
    const uint32_t* __restrict__ twiddles,
    uint32_t layer_idx,
    uint32_t half_n,
    uint32_t n_cols,
    int forward
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = half_n * n_cols;
    if (tid >= total) return;

    uint32_t col_idx = tid / half_n;
    uint32_t pair_idx = tid % half_n;

    uint32_t stride = 1u << layer_idx;
    uint32_t h = pair_idx >> layer_idx;
    uint32_t l = pair_idx & (stride - 1);
    uint32_t idx0 = (h << (layer_idx + 1)) + l;
    uint32_t idx1 = idx0 + stride;

    uint32_t* data = columns[col_idx];
    uint32_t v0 = data[idx0];
    uint32_t v1 = data[idx1];
    uint32_t t = twiddles[h];

    if (forward) {
        butterfly(v0, v1, t);
    } else {
        ibutterfly(v0, v1, t);
    }

    data[idx0] = v0;
    data[idx1] = v1;
}

// Scale all elements by a constant (single column)
__global__ void m31_scale_kernel(
    uint32_t* __restrict__ data,
    uint32_t scale,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    data[tid] = m31_mul(data[tid], scale);
}

// Batched scale kernel
__global__ void m31_batch_scale_kernel(
    uint32_t** __restrict__ columns,
    uint32_t scale,
    uint32_t n,
    uint32_t n_cols
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n * n_cols;
    if (tid >= total) return;

    uint32_t col_idx = tid / n;
    uint32_t elem_idx = tid % n;
    columns[col_idx][elem_idx] = m31_mul(columns[col_idx][elem_idx], scale);
}

// ---- GPU polynomial evaluation at a SecureField point ----

// First-level fold: M31 coefficients -> QM31
__global__ void fold_first_level_kernel(
    const uint32_t* __restrict__ coeffs,
    uint32_t* __restrict__ out,
    const uint32_t* __restrict__ factor,
    uint32_t half_n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half_n) return;

    uint32_t a = coeffs[tid];
    uint32_t b = coeffs[tid + half_n];

    QM31 f = {{factor[0], factor[1], factor[2], factor[3]}};
    QM31 bf = qm31_mul_m31(f, b);
    QM31 result = {{m31_add(a, bf.v[0]), bf.v[1], bf.v[2], bf.v[3]}};

    out[tid * 4 + 0] = result.v[0];
    out[tid * 4 + 1] = result.v[1];
    out[tid * 4 + 2] = result.v[2];
    out[tid * 4 + 3] = result.v[3];
}

// Subsequent fold levels: QM31 -> QM31
__global__ void fold_level_kernel(
    const uint32_t* __restrict__ in_data,
    uint32_t* __restrict__ out_data,
    const uint32_t* __restrict__ factor,
    uint32_t half_n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half_n) return;

    uint32_t ai = tid * 4;
    uint32_t bi = (tid + half_n) * 4;

    QM31 a = {{in_data[ai], in_data[ai+1], in_data[ai+2], in_data[ai+3]}};
    QM31 b = {{in_data[bi], in_data[bi+1], in_data[bi+2], in_data[bi+3]}};
    QM31 f = {{factor[0], factor[1], factor[2], factor[3]}};

    QM31 result = qm31_add(a, qm31_mul(b, f));

    out_data[tid * 4 + 0] = result.v[0];
    out_data[tid * 4 + 1] = result.v[1];
    out_data[tid * 4 + 2] = result.v[2];
    out_data[tid * 4 + 3] = result.v[3];
}

// Bit-reverse permutation
__global__ void bit_reverse_m31_kernel(uint32_t* data, uint32_t log_n, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t j = __brev(i) >> (32 - log_n);
    if (i < j) {
        uint32_t tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

extern "C" {

void cuda_circle_ntt_evaluate(
    uint32_t* d_data,
    const uint32_t* d_twiddles,
    const uint32_t* d_circle_twids,
    const uint32_t* h_layer_offsets,
    const uint32_t* h_layer_sizes,
    uint32_t n_line_layers,
    uint32_t n
) {
    uint32_t half_n = n / 2;
    uint32_t threads = 256;
    uint32_t blocks = (half_n + threads - 1) / threads;

    // Line layers (highest to lowest)
    for (int layer = (int)n_line_layers - 1; layer >= 0; layer--) {
        circle_ntt_layer_kernel<<<blocks, threads>>>(
            d_data,
            d_twiddles + h_layer_offsets[layer],
            (uint32_t)(layer + 1),
            half_n, 1
        );
    }

    // Circle layer (layer 0)
    circle_ntt_layer_kernel<<<blocks, threads>>>(
        d_data, d_circle_twids, 0, half_n, 1
    );

    cudaDeviceSynchronize();
}

void cuda_circle_ntt_interpolate(
    uint32_t* d_data,
    const uint32_t* d_itwiddles,
    const uint32_t* d_circle_itwids,
    const uint32_t* h_layer_offsets,
    const uint32_t* h_layer_sizes,
    uint32_t n_line_layers,
    uint32_t n
) {
    uint32_t half_n = n / 2;
    uint32_t threads = 256;
    uint32_t blocks = (half_n + threads - 1) / threads;

    // Circle layer first
    circle_ntt_layer_kernel<<<blocks, threads>>>(
        d_data, d_circle_itwids, 0, half_n, 0
    );

    // Line layers (lowest to highest)
    for (uint32_t layer = 0; layer < n_line_layers; layer++) {
        circle_ntt_layer_kernel<<<blocks, threads>>>(
            d_data,
            d_itwiddles + h_layer_offsets[layer],
            layer + 1, half_n, 0
        );
    }

    // Scale by 1/n: inv_n = 2^(30*log_n mod 31) in M31
    uint32_t log_n = 0;
    for (uint32_t tmp = n; tmp > 1; tmp >>= 1) log_n++;
    uint32_t exp = (30u * log_n) % 31u;
    uint32_t inv_n = (exp == 0) ? 1u : (1u << exp);

    uint32_t scale_blocks = (n + threads - 1) / threads;
    m31_scale_kernel<<<scale_blocks, threads>>>(d_data, inv_n, n);

    cudaDeviceSynchronize();
}

void cuda_circle_ntt_evaluate_batch(
    uint32_t** d_columns,
    const uint32_t* d_twiddles,
    const uint32_t* d_circle_twids,
    const uint32_t* h_layer_offsets,
    const uint32_t* h_layer_sizes,
    uint32_t n_line_layers,
    uint32_t n,
    uint32_t n_cols
) {
    uint32_t half_n = n / 2;
    uint32_t total = half_n * n_cols;
    uint32_t threads = 256;
    uint32_t blocks = (total + threads - 1) / threads;

    for (int layer = (int)n_line_layers - 1; layer >= 0; layer--) {
        circle_ntt_batch_layer_kernel<<<blocks, threads>>>(
            d_columns,
            d_twiddles + h_layer_offsets[layer],
            (uint32_t)(layer + 1),
            half_n, n_cols, 1
        );
    }

    circle_ntt_batch_layer_kernel<<<blocks, threads>>>(
        d_columns, d_circle_twids, 0, half_n, n_cols, 1
    );

    cudaDeviceSynchronize();
}

void cuda_circle_ntt_interpolate_batch(
    uint32_t** d_columns,
    const uint32_t* d_itwiddles,
    const uint32_t* d_circle_itwids,
    const uint32_t* h_layer_offsets,
    const uint32_t* h_layer_sizes,
    uint32_t n_line_layers,
    uint32_t n,
    uint32_t n_cols
) {
    uint32_t half_n = n / 2;
    uint32_t total = half_n * n_cols;
    uint32_t threads = 256;
    uint32_t blocks = (total + threads - 1) / threads;

    circle_ntt_batch_layer_kernel<<<blocks, threads>>>(
        d_columns, d_circle_itwids, 0, half_n, n_cols, 0
    );

    for (uint32_t layer = 0; layer < n_line_layers; layer++) {
        circle_ntt_batch_layer_kernel<<<blocks, threads>>>(
            d_columns,
            d_itwiddles + h_layer_offsets[layer],
            layer + 1, half_n, n_cols, 0
        );
    }

    uint32_t log_n = 0;
    for (uint32_t tmp = n; tmp > 1; tmp >>= 1) log_n++;
    uint32_t exp = (30u * log_n) % 31u;
    uint32_t inv_n = (exp == 0) ? 1u : (1u << exp);

    uint32_t scale_total = n * n_cols;
    uint32_t scale_blocks = (scale_total + threads - 1) / threads;
    m31_batch_scale_kernel<<<scale_blocks, threads>>>(d_columns, inv_n, n, n_cols);

    cudaDeviceSynchronize();
}

// Apply a single NTT butterfly layer (forward or inverse).
// layer_idx: butterfly stride = 2^layer_idx. For circle layer use layer_idx=0.
void cuda_circle_ntt_layer(
    uint32_t* d_data,
    const uint32_t* d_twiddles,
    uint32_t layer_idx,
    uint32_t n,
    int forward
) {
    uint32_t half_n = n / 2;
    uint32_t threads = 256;
    uint32_t blocks = (half_n + threads - 1) / threads;
    circle_ntt_layer_kernel<<<blocks, threads>>>(
        d_data, d_twiddles, layer_idx, half_n, forward
    );
}

void cuda_bit_reverse_m31(uint32_t* data, uint32_t log_n) {
    uint32_t n = 1u << log_n;
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    bit_reverse_m31_kernel<<<blocks, threads>>>(data, log_n, n);
}

void cuda_eval_at_point(
    const uint32_t* d_coeffs,
    const uint32_t* d_folding_factors,
    uint32_t* h_result,
    uint32_t n,
    uint32_t* d_scratch1,
    uint32_t* d_scratch2
) {
    uint32_t threads = 256;
    uint32_t log_n = 0;
    for (uint32_t tmp = n; tmp > 1; tmp >>= 1) log_n++;

    uint32_t half_n = n / 2;
    uint32_t blocks = (half_n + threads - 1) / threads;
    fold_first_level_kernel<<<blocks, threads>>>(
        d_coeffs, d_scratch1,
        d_folding_factors + 0,
        half_n
    );

    uint32_t* in_buf = d_scratch1;
    uint32_t* out_buf = d_scratch2;
    uint32_t cur_n = half_n;

    for (uint32_t level = 1; level < log_n; level++) {
        half_n = cur_n / 2;
        blocks = (half_n + threads - 1) / threads;
        fold_level_kernel<<<blocks, threads>>>(
            in_buf, out_buf,
            d_folding_factors + level * 4,
            half_n
        );
        uint32_t* tmp_ptr = in_buf;
        in_buf = out_buf;
        out_buf = tmp_ptr;
        cur_n = half_n;
    }

    cudaMemcpy(h_result, in_buf, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

} // extern "C"
