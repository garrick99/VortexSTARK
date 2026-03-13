// GPU constraint evaluation for Fibonacci AIR.
// Computes transition constraint: t[i+2] - t[i+1] - t[i]
// and multiplies by alpha to produce QM31 quotient column.

#include "include/qm31.cuh"

// Fibonacci transition constraint kernel.
// For each i, computes: alpha * (trace[(i+2)%n] - trace[(i+1)%n] - trace[i])
// Output is 4 separate M31 columns (SoA QM31).
__global__ void fibonacci_quotient_kernel(
    const uint32_t* __restrict__ trace,  // M31 evaluation values
    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2, uint32_t* __restrict__ out3,
    uint32_t alpha_a, uint32_t alpha_b, uint32_t alpha_c, uint32_t alpha_d,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    QM31 alpha = {{alpha_a, alpha_b, alpha_c, alpha_d}};

    uint32_t t_i  = trace[i];
    uint32_t t_i1 = trace[(i + 1) % n];
    uint32_t t_i2 = trace[(i + 2) % n];

    // constraint = t[i+2] - t[i+1] - t[i]
    uint32_t constraint = m31_sub(t_i2, m31_add(t_i1, t_i));

    // result = alpha * constraint (QM31 * M31)
    QM31 result = qm31_mul_m31(alpha, constraint);

    out0[i] = result.v[0];
    out1[i] = result.v[1];
    out2[i] = result.v[2];
    out3[i] = result.v[3];
}

// Zero-pad kernel: copies n elements to dst, zeros the rest up to dst_n
__global__ void zero_pad_kernel(
    const uint32_t* __restrict__ src,
    uint32_t* __restrict__ dst,
    uint32_t src_n,
    uint32_t dst_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dst_n) return;
    dst[i] = (i < src_n) ? src[i] : 0;
}

extern "C" {

void cuda_fibonacci_quotient(
    const uint32_t* trace,
    uint32_t* out0, uint32_t* out1, uint32_t* out2, uint32_t* out3,
    const uint32_t* alpha, // [4] on host
    uint32_t n
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    fibonacci_quotient_kernel<<<blocks, threads>>>(
        trace, out0, out1, out2, out3,
        alpha[0], alpha[1], alpha[2], alpha[3], n
    );
}

void cuda_zero_pad(
    const uint32_t* src,
    uint32_t* dst,
    uint32_t src_n,
    uint32_t dst_n
) {
    uint32_t threads = 256;
    uint32_t blocks = (dst_n + threads - 1) / threads;
    zero_pad_kernel<<<blocks, threads>>>(src, dst, src_n, dst_n);
}

} // extern "C"
