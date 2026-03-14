// GPU Pedersen hash: parallel computation of H(a,b) on the STARK curve.
// Each thread computes one complete Pedersen hash using projective coordinates.
// Precomputed constant points in __constant__ memory.

#include "include/fp252.cuh"

// The 5 Pedersen constant points — defined here, extern'd by test
__constant__ Fp252 PEDERSEN_PX[5];
__constant__ Fp252 PEDERSEN_PY[5];

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

// Compute Pedersen hash in projective coordinates.
// Returns the projective point (X, Y, Z) — caller converts to affine.
__device__ ProjPoint pedersen_hash_proj(Fp252 a, Fp252 b) {
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

    return result;
}

// Each thread computes one Pedersen hash in projective coordinates.
// Outputs X and Z²; caller computes x = X * inv(Z²) on CPU via batch inverse.
__global__ void pedersen_batch_kernel(
    const uint64_t* __restrict__ inputs_a,
    const uint64_t* __restrict__ inputs_b,
    uint64_t* __restrict__ out_x,   // [n * 4] — projective X
    uint64_t* __restrict__ out_zz,  // [n * 4] — Z² for batch inverse
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Fp252 a, b;
    a.v[0] = inputs_a[i*4+0]; a.v[1] = inputs_a[i*4+1];
    a.v[2] = inputs_a[i*4+2]; a.v[3] = inputs_a[i*4+3];
    b.v[0] = inputs_b[i*4+0]; b.v[1] = inputs_b[i*4+1];
    b.v[2] = inputs_b[i*4+2]; b.v[3] = inputs_b[i*4+3];

    ProjPoint result = pedersen_hash_proj(a, b);

    // Output X and Z²
    Fp252 zz = fp_mul(result.z, result.z);
    out_x[i*4+0] = result.x.v[0]; out_x[i*4+1] = result.x.v[1];
    out_x[i*4+2] = result.x.v[2]; out_x[i*4+3] = result.x.v[3];
    out_zz[i*4+0] = zz.v[0]; out_zz[i*4+1] = zz.v[1];
    out_zz[i*4+2] = zz.v[2]; out_zz[i*4+3] = zz.v[3];
}

extern "C" {

// Upload the 5 Pedersen constant points to GPU constant memory
void cuda_pedersen_upload_points(const uint64_t* px, const uint64_t* py) {
    // px: [5 * 4] u64 — x-coordinates of P₀..P₄
    // py: [5 * 4] u64 — y-coordinates
    cudaMemcpyToSymbol(PEDERSEN_PX, px, 5 * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(PEDERSEN_PY, py, 5 * 4 * sizeof(uint64_t));
}

// Batch Pedersen hash on GPU — outputs projective (X, Z²) for CPU batch inverse
void cuda_pedersen_hash_batch(
    const uint64_t* inputs_a,
    const uint64_t* inputs_b,
    uint64_t* out_x,
    uint64_t* out_zz,
    uint32_t n
) {
    cudaDeviceSetLimit(cudaLimitStackSize, 65536);
    uint32_t threads = 64;
    uint32_t blocks = (n + threads - 1) / threads;
    pedersen_batch_kernel<<<blocks, threads>>>(inputs_a, inputs_b, out_x, out_zz, n);
}

} // extern "C"
