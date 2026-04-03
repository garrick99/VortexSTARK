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

// Chunked quotient kernel: processes [offset, offset+chunk_n) of the global domain.
// Reads from full trace buffer (global_n elements), writes to chunk-sized output.
__global__ void fibonacci_quotient_chunk_kernel(
    const uint32_t* __restrict__ trace,
    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2, uint32_t* __restrict__ out3,
    uint32_t alpha_a, uint32_t alpha_b, uint32_t alpha_c, uint32_t alpha_d,
    uint32_t offset, uint32_t chunk_n, uint32_t global_n
) {
    uint32_t local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i >= chunk_n) return;

    uint32_t i = offset + local_i;
    QM31 alpha = {{alpha_a, alpha_b, alpha_c, alpha_d}};

    uint32_t t_i  = trace[i];
    uint32_t t_i1 = trace[(i + 1) % global_n];
    uint32_t t_i2 = trace[(i + 2) % global_n];

    uint32_t constraint = m31_sub(t_i2, m31_add(t_i1, t_i));
    QM31 result = qm31_mul_m31(alpha, constraint);

    out0[local_i] = result.v[0];
    out1[local_i] = result.v[1];
    out2[local_i] = result.v[2];
    out3[local_i] = result.v[3];
}

// Interleave two half-buffers: output[2i] = even[i], output[2i+1] = odd[i]
// Maps the twin-coset evaluations into the full circle group's natural ordering.
__global__ void interleave_u32_kernel(
    const uint32_t* __restrict__ even,
    const uint32_t* __restrict__ odd,
    uint32_t* __restrict__ output,
    uint32_t half_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_n) return;
    output[2 * i]     = even[i];
    output[2 * i + 1] = odd[i];
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

// Chunked quotient: compute constraint for positions [offset, offset+chunk_n) within
// a global domain of size global_n. Reads trace from full buffer, writes to chunk-sized outputs.
void cuda_fibonacci_quotient_chunk(
    const uint32_t* trace,  // full eval buffer, global_n elements on GPU
    uint32_t* out0, uint32_t* out1, uint32_t* out2, uint32_t* out3,
    const uint32_t* alpha,  // [4] on host
    uint32_t offset,        // start position in global domain
    uint32_t chunk_n,       // elements in this chunk
    uint32_t global_n       // total domain size (for modular access)
) {
    uint32_t threads = 256;
    uint32_t blocks = (chunk_n + threads - 1) / threads;
    fibonacci_quotient_chunk_kernel<<<blocks, threads>>>(
        trace, out0, out1, out2, out3,
        alpha[0], alpha[1], alpha[2], alpha[3],
        offset, chunk_n, global_n
    );
}

// Stream-aware chunked quotient
void cuda_fibonacci_quotient_chunk_stream(
    const uint32_t* trace,
    uint32_t* out0, uint32_t* out1, uint32_t* out2, uint32_t* out3,
    const uint32_t* alpha,
    uint32_t offset, uint32_t chunk_n, uint32_t global_n,
    cudaStream_t stream
) {
    uint32_t threads = 256;
    uint32_t blocks = (chunk_n + threads - 1) / threads;
    fibonacci_quotient_chunk_kernel<<<blocks, threads, 0, stream>>>(
        trace, out0, out1, out2, out3,
        alpha[0], alpha[1], alpha[2], alpha[3],
        offset, chunk_n, global_n
    );
}

// Interleave two half-buffers into one: output[2i] = even[i], output[2i+1] = odd[i]
void cuda_interleave_u32(
    const uint32_t* even,   // subgroup(30) values, half_n elements
    const uint32_t* odd,    // half_coset(30) values, half_n elements
    uint32_t* output,       // interleaved output, 2*half_n elements
    uint32_t half_n
) {
    uint32_t threads = 256;
    uint32_t blocks = (half_n + threads - 1) / threads;
    interleave_u32_kernel<<<blocks, threads>>>(even, odd, output, half_n);
}

} // extern "C"
