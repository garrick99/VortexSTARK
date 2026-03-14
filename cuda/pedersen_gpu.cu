// GPU Pedersen hash: parallel computation of H(a,b) on the STARK curve.
// Each thread computes one complete Pedersen hash using projective coordinates.
// Precomputed constant points in __constant__ memory.

#include "include/fp252.cuh"

// The 5 Pedersen constant points (P₀..P₄) in __constant__ memory
__constant__ Fp252 PEDERSEN_PX[5]; // x-coordinates
__constant__ Fp252 PEDERSEN_PY[5]; // y-coordinates

// Mask lowest n_bits of a field element
__device__ Fp252 fp_mask(Fp252 val, int n_bits) {
    Fp252 r = val;
    int full_limbs = n_bits / 64;
    int remaining = n_bits % 64;
    for (int i = full_limbs + 1; i < 4; i++) r.v[i] = 0;
    if (full_limbs < 4 && remaining > 0) {
        r.v[full_limbs] &= (1ULL << remaining) - 1;
    } else if (full_limbs < 4) {
        r.v[full_limbs] = 0;
    }
    return r;
}

// Shift right by n_bits
__device__ Fp252 fp_shr(Fp252 val, int n_bits) {
    int limb_shift = n_bits / 64;
    int bit_shift = n_bits % 64;
    Fp252 r = fp_zero();
    for (int i = 0; i < 4; i++) {
        int src = i + limb_shift;
        if (src < 4) {
            r.v[i] = val.v[src] >> bit_shift;
            if (bit_shift > 0 && src + 1 < 4) {
                r.v[i] |= val.v[src + 1] << (64 - bit_shift);
            }
        }
    }
    return r;
}

// Compute Pedersen hash: H(a,b) = [P₀ + a_low·P₁ + a_high·P₂ + b_low·P₃ + b_high·P₄]_x
__device__ Fp252 pedersen_hash_device(Fp252 a, Fp252 b) {
    // Decompose inputs
    Fp252 a_low  = fp_mask(a, 248);
    Fp252 a_high = fp_shr(a, 248);
    Fp252 b_low  = fp_mask(b, 248);
    Fp252 b_high = fp_shr(b, 248);

    // Start with P₀
    ProjPoint result = proj_from_affine(PEDERSEN_PX[0], PEDERSEN_PY[0]);

    // Add a_low · P₁
    ProjPoint p1 = proj_from_affine(PEDERSEN_PX[1], PEDERSEN_PY[1]);
    result = proj_add(result, proj_scalar_mul(p1, a_low));

    // Add a_high · P₂
    ProjPoint p2 = proj_from_affine(PEDERSEN_PX[2], PEDERSEN_PY[2]);
    result = proj_add(result, proj_scalar_mul(p2, a_high));

    // Add b_low · P₃
    ProjPoint p3 = proj_from_affine(PEDERSEN_PX[3], PEDERSEN_PY[3]);
    result = proj_add(result, proj_scalar_mul(p3, b_low));

    // Add b_high · P₄
    ProjPoint p4 = proj_from_affine(PEDERSEN_PX[4], PEDERSEN_PY[4]);
    result = proj_add(result, proj_scalar_mul(p4, b_high));

    // Convert to affine x-coordinate
    return proj_to_affine_x(result);
}

// Each thread computes one Pedersen hash
__global__ void pedersen_batch_kernel(
    const uint64_t* __restrict__ inputs_a,  // [n * 4] — n Fp252 values for 'a'
    const uint64_t* __restrict__ inputs_b,  // [n * 4] — n Fp252 values for 'b'
    uint64_t* __restrict__ outputs,         // [n * 4] — n Fp252 hash results
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Fp252 a, b;
    a.v[0] = inputs_a[i*4+0]; a.v[1] = inputs_a[i*4+1];
    a.v[2] = inputs_a[i*4+2]; a.v[3] = inputs_a[i*4+3];
    b.v[0] = inputs_b[i*4+0]; b.v[1] = inputs_b[i*4+1];
    b.v[2] = inputs_b[i*4+2]; b.v[3] = inputs_b[i*4+3];

    Fp252 hash = pedersen_hash_device(a, b);

    outputs[i*4+0] = hash.v[0]; outputs[i*4+1] = hash.v[1];
    outputs[i*4+2] = hash.v[2]; outputs[i*4+3] = hash.v[3];
}

extern "C" {

// Upload the 5 Pedersen constant points to GPU constant memory
void cuda_pedersen_upload_points(const uint64_t* px, const uint64_t* py) {
    // px: [5 * 4] u64 — x-coordinates of P₀..P₄
    // py: [5 * 4] u64 — y-coordinates
    cudaMemcpyToSymbol(PEDERSEN_PX, px, 5 * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(PEDERSEN_PY, py, 5 * 4 * sizeof(uint64_t));
}

// Batch Pedersen hash on GPU
void cuda_pedersen_hash_batch(
    const uint64_t* inputs_a,
    const uint64_t* inputs_b,
    uint64_t* outputs,
    uint32_t n
) {
    // Use fewer threads per block — each thread does heavy EC work
    uint32_t threads = 64;
    uint32_t blocks = (n + threads - 1) / threads;
    pedersen_batch_kernel<<<blocks, threads>>>(inputs_a, inputs_b, outputs, n);
}

} // extern "C"
