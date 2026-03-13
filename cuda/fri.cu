// FRI fold kernels (SoA layout) for kraken-stark.
// fold_line: halves a line evaluation using alpha.
// fold_circle_into_line: folds a circle evaluation into a line evaluation.

#include "include/qm31.cuh"

// FRI fold_line (SoA layout): folds a line evaluation by alpha.
// Pairs consecutive elements: (values[2*i], values[2*i+1])
// result[i] = (f0 + f1) + alpha * twiddle[i] * (f0 - f1)
__global__ void fold_line_soa_kernel(
    const uint32_t* __restrict__ in0, const uint32_t* __restrict__ in1,
    const uint32_t* __restrict__ in2, const uint32_t* __restrict__ in3,
    const uint32_t* __restrict__ twiddles,
    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2, uint32_t* __restrict__ out3,
    uint32_t alpha_a, uint32_t alpha_b, uint32_t alpha_c, uint32_t alpha_d,
    uint32_t half_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_n) return;

    QM31 alpha = {{alpha_a, alpha_b, alpha_c, alpha_d}};

    uint32_t idx0 = 2 * i;
    uint32_t idx1 = 2 * i + 1;

    QM31 f0 = {{in0[idx0], in1[idx0], in2[idx0], in3[idx0]}};
    QM31 f1 = {{in0[idx1], in1[idx1], in2[idx1], in3[idx1]}};

    QM31 sum = qm31_add(f0, f1);
    QM31 diff = qm31_sub(f0, f1);
    QM31 tw_diff = qm31_mul_m31(diff, twiddles[i]);
    QM31 result = qm31_add(sum, qm31_mul(alpha, tw_diff));

    out0[i] = result.v[0];
    out1[i] = result.v[1];
    out2[i] = result.v[2];
    out3[i] = result.v[3];
}

// FRI fold_circle_into_line (SoA layout):
// f_prime = (f0 + f1) + alpha * twiddle[i] * (f0 - f1)
// dst[i] = dst[i] * alpha_sq + f_prime
__global__ void fold_circle_into_line_soa_kernel(
    uint32_t* __restrict__ dst0, uint32_t* __restrict__ dst1,
    uint32_t* __restrict__ dst2, uint32_t* __restrict__ dst3,
    const uint32_t* __restrict__ src0, const uint32_t* __restrict__ src1,
    const uint32_t* __restrict__ src2, const uint32_t* __restrict__ src3,
    const uint32_t* __restrict__ twiddles,
    uint32_t alpha_a, uint32_t alpha_b, uint32_t alpha_c, uint32_t alpha_d,
    uint32_t alpha_sq_a, uint32_t alpha_sq_b, uint32_t alpha_sq_c, uint32_t alpha_sq_d,
    uint32_t half_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_n) return;

    QM31 alpha = {{alpha_a, alpha_b, alpha_c, alpha_d}};
    QM31 alpha_sq = {{alpha_sq_a, alpha_sq_b, alpha_sq_c, alpha_sq_d}};

    uint32_t idx0 = 2 * i;
    uint32_t idx1 = 2 * i + 1;

    QM31 f0 = {{src0[idx0], src1[idx0], src2[idx0], src3[idx0]}};
    QM31 f1 = {{src0[idx1], src1[idx1], src2[idx1], src3[idx1]}};

    QM31 sum = qm31_add(f0, f1);
    QM31 diff = qm31_sub(f0, f1);
    QM31 tw_diff = qm31_mul_m31(diff, twiddles[i]);
    QM31 f_prime = qm31_add(sum, qm31_mul(alpha, tw_diff));

    QM31 prev = {{dst0[i], dst1[i], dst2[i], dst3[i]}};
    QM31 result = qm31_add(qm31_mul(prev, alpha_sq), f_prime);

    dst0[i] = result.v[0];
    dst1[i] = result.v[1];
    dst2[i] = result.v[2];
    dst3[i] = result.v[3];
}

extern "C" {

void cuda_fold_line_soa(
    const uint32_t* in0, const uint32_t* in1,
    const uint32_t* in2, const uint32_t* in3,
    const uint32_t* twiddles,
    uint32_t* out0, uint32_t* out1,
    uint32_t* out2, uint32_t* out3,
    const uint32_t* alpha,  // [4] on host
    uint32_t half_n
) {
    uint32_t threads = 256;
    uint32_t blocks = (half_n + threads - 1) / threads;
    fold_line_soa_kernel<<<blocks, threads>>>(
        in0, in1, in2, in3, twiddles,
        out0, out1, out2, out3,
        alpha[0], alpha[1], alpha[2], alpha[3],
        half_n
    );
}

void cuda_fold_circle_into_line_soa(
    uint32_t* dst0, uint32_t* dst1,
    uint32_t* dst2, uint32_t* dst3,
    const uint32_t* src0, const uint32_t* src1,
    const uint32_t* src2, const uint32_t* src3,
    const uint32_t* twiddles,
    const uint32_t* alpha,     // [4] on host
    const uint32_t* alpha_sq,  // [4] on host
    uint32_t half_n
) {
    uint32_t threads = 256;
    uint32_t blocks = (half_n + threads - 1) / threads;
    fold_circle_into_line_soa_kernel<<<blocks, threads>>>(
        dst0, dst1, dst2, dst3,
        src0, src1, src2, src3,
        twiddles,
        alpha[0], alpha[1], alpha[2], alpha[3],
        alpha_sq[0], alpha_sq[1], alpha_sq[2], alpha_sq[3],
        half_n
    );
}

} // extern "C"
