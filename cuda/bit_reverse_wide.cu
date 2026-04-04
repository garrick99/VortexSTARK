// GPU bit-reverse for wide (QM31 / 4-wide) columns.
//
// SecureField columns store QM31 elements as 4 consecutive u32s.
// This kernel permutes the elements (not the individual u32s) in bit-reversed order.
//
// Out-of-place: reads from `in`, writes to `out` (both n*4 u32s).
// Rust caller allocates temp and swaps the buffer.

#include <stdint.h>

__device__ __forceinline__ uint32_t bit_reverse_u32(uint32_t v, uint32_t log_n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log_n; i++) {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    return r;
}

// Each thread i reads element i from `in` and writes it to `out[bit_reverse(i)]`.
__global__ void bit_reverse_qm31_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out,
    uint32_t n,
    uint32_t log_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t j = bit_reverse_u32(i, log_n);

    out[4*j + 0] = in[4*i + 0];
    out[4*j + 1] = in[4*i + 1];
    out[4*j + 2] = in[4*i + 2];
    out[4*j + 3] = in[4*i + 3];
}

extern "C" {

// in-place: pass the same pointer for `in` and `out` is unsafe (concurrent read/write).
// The caller always passes separate buffers.
void cuda_bit_reverse_qm31(
    const uint32_t* in,
    uint32_t* out,
    uint32_t n,
    uint32_t log_n
) {
    uint32_t blocks = (n + 255) / 256;
    bit_reverse_qm31_kernel<<<blocks, 256>>>(in, out, n, log_n);
}

} // extern "C"
