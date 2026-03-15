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

// ====== Montgomery-accelerated EC operations ======

// Point doubling with Montgomery mul (2 schoolbook muls per fp_mul instead of 71)
__device__ ProjPoint mont_double(ProjPoint p) {
    if (fp_is_zero(p.z)) return p;

    Fp252 xx = fp_mont_mul(p.x, p.x);
    Fp252 yy = fp_mont_mul(p.y, p.y);
    Fp252 zz = fp_mont_mul(p.z, p.z);
    Fp252 zzzz = fp_mont_mul(zz, zz);

    Fp252 xy2 = fp_mont_mul(p.x, yy);
    Fp252 s = fp_add(fp_add(xy2, xy2), fp_add(xy2, xy2));

    // a=1 in Montgomery form = to_mont(1) = R mod p
    Fp252 a_mont = {{0xFFFFFFFFFFFFFFE1ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x07FFFFFFFFFFFDF0ULL}};
    Fp252 m = fp_add(fp_add(xx, fp_add(xx, xx)), fp_mont_mul(a_mont, zzzz));

    Fp252 x3 = fp_sub(fp_mont_mul(m, m), fp_add(s, s));

    Fp252 yyyy = fp_mont_mul(yy, yy);
    Fp252 y4x8 = fp_add(fp_add(fp_add(yyyy, yyyy), fp_add(yyyy, yyyy)),
                         fp_add(fp_add(yyyy, yyyy), fp_add(yyyy, yyyy)));
    Fp252 y3 = fp_sub(fp_mont_mul(m, fp_sub(s, x3)), y4x8);

    Fp252 z3 = fp_mont_mul(fp_add(p.y, p.y), p.z);

    ProjPoint r; r.x = x3; r.y = y3; r.z = z3;
    return r;
}

// Point addition with Montgomery mul
__device__ ProjPoint mont_add(ProjPoint p1, ProjPoint p2) {
    if (fp_is_zero(p1.z)) return p2;
    if (fp_is_zero(p2.z)) return p1;

    Fp252 z1z1 = fp_mont_mul(p1.z, p1.z);
    Fp252 z2z2 = fp_mont_mul(p2.z, p2.z);
    Fp252 u1 = fp_mont_mul(p1.x, z2z2);
    Fp252 u2 = fp_mont_mul(p2.x, z1z1);
    Fp252 s1 = fp_mont_mul(fp_mont_mul(p1.y, z2z2), p2.z);
    Fp252 s2 = fp_mont_mul(fp_mont_mul(p2.y, z1z1), p1.z);

    Fp252 h = fp_sub(u2, u1);
    Fp252 rr = fp_sub(s2, s1);

    if (fp_is_zero(h)) {
        if (fp_is_zero(rr)) return mont_double(p1);
        return proj_infinity();
    }

    Fp252 hh = fp_mont_mul(h, h);
    Fp252 hhh = fp_mont_mul(hh, h);
    Fp252 u1hh = fp_mont_mul(u1, hh);

    Fp252 x3 = fp_sub(fp_sub(fp_mont_mul(rr, rr), hhh), fp_add(u1hh, u1hh));
    Fp252 y3 = fp_sub(fp_mont_mul(rr, fp_sub(u1hh, x3)), fp_mont_mul(s1, hhh));
    Fp252 z3 = fp_mont_mul(fp_mont_mul(p1.z, p2.z), h);

    ProjPoint res; res.x = x3; res.y = y3; res.z = z3;
    return res;
}

// Scalar mul with Montgomery EC ops
__device__ ProjPoint mont_scalar_mul(ProjPoint base, Fp252 scalar) {
    ProjPoint result = proj_infinity();
    ProjPoint current = base;

    for (int limb = 0; limb < 4; limb++) {
        uint64_t s = scalar.v[limb];
        int bits = (limb == 3) ? 60 : 64;
        for (int b = 0; b < bits; b++) {
            if (s & 1) {
                result = mont_add(result, current);
            }
            current = mont_double(current);
            s >>= 1;
        }
    }
    return result;
}

// Compute Pedersen hash using Montgomery-accelerated EC.
__device__ ProjPoint pedersen_hash_proj(Fp252 a, Fp252 b) {
    // Decompose inputs
    Fp252 a_low  = fp_mask(a, 248);
    Fp252 a_high = fp_shr(a, 248);
    Fp252 b_low  = fp_mask(b, 248);
    Fp252 b_high = fp_shr(b, 248);

    // Convert constant points to Montgomery form
    // (ideally precomputed, but for correctness first)
    auto mont_point = [](Fp252 px, Fp252 py) -> ProjPoint {
        ProjPoint p;
        p.x = to_mont(px);
        p.y = to_mont(py);
        p.z = to_mont(fp_one());
        return p;
    };

    ProjPoint result = mont_point(PEDERSEN_PX[0], PEDERSEN_PY[0]);

    ProjPoint p1 = mont_point(PEDERSEN_PX[1], PEDERSEN_PY[1]);
    result = mont_add(result, mont_scalar_mul(p1, a_low));

    ProjPoint p2 = mont_point(PEDERSEN_PX[2], PEDERSEN_PY[2]);
    result = mont_add(result, mont_scalar_mul(p2, a_high));

    ProjPoint p3 = mont_point(PEDERSEN_PX[3], PEDERSEN_PY[3]);
    result = mont_add(result, mont_scalar_mul(p3, b_low));

    ProjPoint p4 = mont_point(PEDERSEN_PX[4], PEDERSEN_PY[4]);
    result = mont_add(result, mont_scalar_mul(p4, b_high));

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

    // Output X and Z² (converted from Montgomery form)
    Fp252 rx = from_mont(result.x);
    Fp252 rz = from_mont(result.z);
    Fp252 zz = fp_mul(rz, rz); // standard mul for final output (not hot path)
    out_x[i*4+0] = rx.v[0]; out_x[i*4+1] = rx.v[1];
    out_x[i*4+2] = rx.v[2]; out_x[i*4+3] = rx.v[3];
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

// Debug kernel: test EC point doubling of a known affine point
void cuda_pedersen_test_double(
    const uint64_t* px, const uint64_t* py, // one affine point (4 u64 each)
    uint64_t* out_x, uint64_t* out_y, uint64_t* out_z // projective result
);

__global__ void pedersen_test_double_kernel(
    const uint64_t* px, const uint64_t* py,
    uint64_t* out_x, uint64_t* out_y, uint64_t* out_z
) {
    Fp252 x = {{px[0], px[1], px[2], px[3]}};
    Fp252 y = {{py[0], py[1], py[2], py[3]}};

    // Manual doubling with intermediate output for debugging
    // For Z=1 input: xx=x^2, yy=y^2, s=4*x*yy, m=3*xx+1
    Fp252 xx = fp_mul(x, x);
    Fp252 yy = fp_mul(y, y);

    // Store xx in out_z for debugging (we'll check against CPU x^2)
    out_z[0] = xx.v[0]; out_z[1] = xx.v[1]; out_z[2] = xx.v[2]; out_z[3] = xx.v[3];

    // Full doubling
    ProjPoint p = proj_from_affine(x, y);
    ProjPoint d = mont_proj_double(p);

    out_x[0] = d.x.v[0]; out_x[1] = d.x.v[1]; out_x[2] = d.x.v[2]; out_x[3] = d.x.v[3];
    out_y[0] = d.y.v[0]; out_y[1] = d.y.v[1]; out_y[2] = d.y.v[2]; out_y[3] = d.y.v[3];
    // out_z now holds xx (x squared) for debugging
}

void cuda_pedersen_test_double(
    const uint64_t* px, const uint64_t* py,
    uint64_t* out_x, uint64_t* out_y, uint64_t* out_z
) {
    pedersen_test_double_kernel<<<1, 1>>>(px, py, out_x, out_y, out_z);
}

} // extern "C"
