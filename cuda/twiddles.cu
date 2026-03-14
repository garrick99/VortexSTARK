// GPU twiddle factor computation for Circle NTT and FRI.
// Computes circle domain points and their inverses entirely on GPU.

#include "include/m31.cuh"

// Circle point: (x, y) where x^2 + y^2 = 1 mod P
struct CirclePt {
    uint32_t x;
    uint32_t y;
};

// Circle group operation: (x1,y1) * (x2,y2) = (x1*x2 - y1*y2, x1*y2 + y1*x2)
__device__ __forceinline__ CirclePt circle_mul(CirclePt a, CirclePt b) {
    return {
        m31_sub(m31_mul(a.x, b.x), m31_mul(a.y, b.y)),
        m31_add(m31_mul(a.x, b.y), m31_mul(a.y, b.x))
    };
}

// Compute coset points: initial * step^i for i = 0..n-1
// Extract either x or y coordinate, bit-reversed, for pairs (2i)
// Output: half_n values ready for batch inverse
__global__ void compute_fold_twiddle_sources_kernel(
    uint32_t initial_x, uint32_t initial_y,
    uint32_t step_x, uint32_t step_y,
    uint32_t* __restrict__ output,
    uint32_t n,
    uint32_t log_n,
    int extract_y  // 0 = x coordinate, 1 = y coordinate
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half_n = n / 2;
    if (i >= half_n) return;

    // We need point at bit_reverse(i << 1, log_n)
    uint32_t idx = __brev(i << 1) >> (32 - log_n);

    // Compute step^idx by repeated squaring
    CirclePt result = {1, 0}; // identity
    CirclePt base = {step_x, step_y};
    uint32_t exp = idx;
    while (exp > 0) {
        if (exp & 1) result = circle_mul(result, base);
        base = circle_mul(base, base);
        exp >>= 1;
    }

    // Multiply by initial
    CirclePt point = circle_mul({initial_x, initial_y}, result);

    output[i] = extract_y ? point.y : point.x;
}

// Montgomery batch inverse: product tree + unwind
// Step 1: compute prefix products
__global__ void batch_inv_prefix_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ prefix,
    uint32_t n,
    uint32_t stride,
    uint32_t block_size
) {
    // Each thread handles one block of sequential prefix products
    uint32_t block_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id * block_size >= n) return;

    uint32_t start = block_id * block_size;
    uint32_t end = min(start + block_size, n);

    uint32_t acc = input[start];
    prefix[start] = acc;
    for (uint32_t i = start + 1; i < end; i++) {
        acc = m31_mul(acc, input[i]);
        prefix[i] = acc;
    }
}

// Compute all coset points: output_x[i] = (initial * step^i).x, output_y[i] = .y
// Each thread computes one point via repeated squaring of step.
__global__ void compute_coset_points_kernel(
    uint32_t initial_x, uint32_t initial_y,
    uint32_t step_x, uint32_t step_y,
    uint32_t* __restrict__ output_x,
    uint32_t* __restrict__ output_y,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Compute step^i by repeated squaring
    CirclePt result = {1, 0}; // identity
    CirclePt base = {step_x, step_y};
    uint32_t exp = i;
    while (exp > 0) {
        if (exp & 1) result = circle_mul(result, base);
        base = circle_mul(base, base);
        exp >>= 1;
    }

    // Multiply by initial
    CirclePt point = circle_mul({initial_x, initial_y}, result);

    output_x[i] = point.x;
    output_y[i] = point.y;
}

// Squash kernel: x' = 2*x^2 - 1, taking even-indexed elements
// input[2*i] → output[i] = 2*input[2*i]^2 - 1
__global__ void squash_x_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t out_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_n) return;

    uint32_t x = input[2 * i];
    // 2*x^2 - 1 mod P
    uint32_t x2 = m31_mul(x, x);
    uint32_t two_x2 = m31_add(x2, x2);
    output[i] = m31_sub(two_x2, 1); // 1 is already reduced
}

// Extract even-indexed elements and squash in one kernel:
// twiddle_out[i] = input[2*i]  (copy even elements to output)
// squash_out[i] = 2*input[2*i]^2 - 1  (squashed for next layer)
__global__ void extract_and_squash_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ twiddle_out,
    uint32_t* __restrict__ squash_out,
    uint32_t half_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_n) return;

    uint32_t x = input[2 * i];
    twiddle_out[i] = x;

    uint32_t x2 = m31_mul(x, x);
    uint32_t two_x2 = m31_add(x2, x2);
    squash_out[i] = m31_sub(two_x2, 1);
}

extern "C" {

void cuda_compute_fold_twiddle_sources(
    uint32_t initial_x, uint32_t initial_y,
    uint32_t step_x, uint32_t step_y,
    uint32_t* output,
    uint32_t n,
    uint32_t log_n,
    int extract_y
) {
    uint32_t half_n = n / 2;
    uint32_t threads = 256;
    uint32_t blocks = (half_n + threads - 1) / threads;
    compute_fold_twiddle_sources_kernel<<<blocks, threads>>>(
        initial_x, initial_y, step_x, step_y,
        output, n, log_n, extract_y
    );
}

// Stream-aware variant for overlapped execution
void cuda_compute_fold_twiddle_sources_stream(
    uint32_t initial_x, uint32_t initial_y,
    uint32_t step_x, uint32_t step_y,
    uint32_t* output,
    uint32_t n,
    uint32_t log_n,
    int extract_y,
    cudaStream_t stream
) {
    uint32_t half_n = n / 2;
    uint32_t threads = 256;
    uint32_t blocks = (half_n + threads - 1) / threads;
    compute_fold_twiddle_sources_kernel<<<blocks, threads, 0, stream>>>(
        initial_x, initial_y, step_x, step_y,
        output, n, log_n, extract_y
    );
}

void cuda_compute_coset_points(
    uint32_t initial_x, uint32_t initial_y,
    uint32_t step_x, uint32_t step_y,
    uint32_t* output_x, uint32_t* output_y,
    uint32_t n
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    compute_coset_points_kernel<<<blocks, threads>>>(
        initial_x, initial_y, step_x, step_y,
        output_x, output_y, n
    );
}

void cuda_squash_x(
    const uint32_t* input, uint32_t* output, uint32_t out_n
) {
    uint32_t threads = 256;
    uint32_t blocks = (out_n + threads - 1) / threads;
    squash_x_kernel<<<blocks, threads>>>(input, output, out_n);
}

void cuda_extract_and_squash(
    const uint32_t* input,
    uint32_t* twiddle_out,
    uint32_t* squash_out,
    uint32_t half_n
) {
    uint32_t threads = 256;
    uint32_t blocks = (half_n + threads - 1) / threads;
    extract_and_squash_kernel<<<blocks, threads>>>(input, twiddle_out, squash_out, half_n);
}

} // extern "C"
