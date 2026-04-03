// GPU batch inverse for M31 using Montgomery's trick.
// Each thread handles CHUNK_SIZE elements with prefix products + unwind.

#include "include/m31.cuh"

static constexpr int CHUNK_SIZE = 64;

__global__ void batch_inverse_m31_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = tid * CHUNK_SIZE;
    if (start >= n) return;
    uint32_t end = min(start + (uint32_t)CHUNK_SIZE, n);
    uint32_t count = end - start;

    // Prefix products in registers
    uint32_t prefix[CHUNK_SIZE];
    prefix[0] = input[start];
    for (uint32_t i = 1; i < count; i++) {
        uint32_t val = input[start + i];
        prefix[i] = (val == 0) ? prefix[i-1] : m31_mul(prefix[i-1], val);
    }

    // Invert the total product (Fermat's little theorem)
    uint32_t inv = m31_inv(prefix[count - 1]);

    // Propagate inverses backward
    for (uint32_t i = count - 1; i > 0; i--) {
        uint32_t val = input[start + i];
        if (val == 0) {
            output[start + i] = 0;
        } else {
            output[start + i] = m31_mul(inv, prefix[i-1]);
            inv = m31_mul(inv, val);
        }
    }
    output[start] = (input[start] == 0) ? 0 : inv;
}

extern "C" {

void cuda_batch_inverse_m31(const uint32_t* input, uint32_t* output, uint32_t n) {
    uint32_t n_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    uint32_t threads = 256;
    uint32_t blocks = (n_chunks + threads - 1) / threads;
    batch_inverse_m31_kernel<<<blocks, threads>>>(input, output, n);
}

void cuda_batch_inverse_m31_stream(const uint32_t* input, uint32_t* output, uint32_t n, cudaStream_t stream) {
    uint32_t n_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    uint32_t threads = 256;
    uint32_t blocks = (n_chunks + threads - 1) / threads;
    batch_inverse_m31_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

} // extern "C"
