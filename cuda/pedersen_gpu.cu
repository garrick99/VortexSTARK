// GPU Pedersen hash: parallel computation of H(a,b) on the STARK curve.
// Each thread computes one complete Pedersen hash using projective coordinates.
// Windowed 4-bit scalar mul with precomputed tables in global memory (L1-cached).

#include "include/fp252.cuh"

// The 5 Pedersen constant points (affine, standard form)
__constant__ Fp252 PEDERSEN_PX[5];
__constant__ Fp252 PEDERSEN_PY[5];

// Windowed tables in global memory for L1-cached divergent access.
// __constant__ memory serializes when threads access different indices;
// global memory with L1 cache handles divergent reads natively.
// Table layout: [4 points (P1..P4)][16 multiples (0..15)] in Montgomery Jacobian.
// Total: 4 * 16 * 3 coords * 32 bytes = 6144 bytes. Fits in 128KB L1 cache.
__device__ Fp252 G_WTABLE_X[4][16];
__device__ Fp252 G_WTABLE_Y[4][16];
__device__ Fp252 G_WTABLE_Z[4][16];
__device__ Fp252 G_P0_MONT_X;
__device__ Fp252 G_P0_MONT_Y;
__device__ Fp252 G_P0_MONT_Z;

// Constant memory upload targets (cudaMemcpyToSymbol), copied to globals at init.
__constant__ Fp252 WTABLE_X[4][16];
__constant__ Fp252 WTABLE_Y[4][16];
__constant__ Fp252 WTABLE_Z[4][16];
__constant__ Fp252 P0_MONT_X;
__constant__ Fp252 P0_MONT_Y;
__constant__ Fp252 P0_MONT_Z;

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

// ====== Montgomery field inverse (Fermat's little theorem) ======

// Compute a^(-1) in Montgomery domain via a^(p-2) mod p.
// p = 2^251 + 17*2^192 + 1
// p-2 in binary (252 bits, MSB first): 1 [54 zeros] 1 [4 zeros] [192 ones]
// Total: 251 squarings + 193 multiplies = 444 fp_mont_mul.
// All threads process the same fixed exponent — zero warp divergence.
__device__ Fp252 fp_mont_inverse(Fp252 a) {
    Fp252 result = a; // Initialize with a (MSB = bit 251 = 1)

    // Bits 250..197: 54 squarings (all zero bits)
    for (int i = 0; i < 54; i++)
        result = fp_mont_mul(result, result);

    // Bit 196: square + multiply (1 bit)
    result = fp_mont_mul(result, result);
    result = fp_mont_mul(result, a);

    // Bits 195..192: 4 squarings (all zero bits)
    for (int i = 0; i < 4; i++)
        result = fp_mont_mul(result, result);

    // Bits 191..0: 192 square-and-multiply (all one bits)
    for (int i = 0; i < 192; i++) {
        result = fp_mont_mul(result, result);
        result = fp_mont_mul(result, a);
    }

    return result;
}

// ====== Montgomery-accelerated EC operations ======

// Point doubling with Montgomery mul
__device__ ProjPoint mont_double(ProjPoint p) {
    if (fp_is_zero(p.z)) return p;

    Fp252 xx = fp_mont_mul(p.x, p.x);
    Fp252 yy = fp_mont_mul(p.y, p.y);
    Fp252 zz = fp_mont_mul(p.z, p.z);
    Fp252 zzzz = fp_mont_mul(zz, zz);

    Fp252 xy2 = fp_mont_mul(p.x, yy);
    Fp252 s = fp_add(fp_add(xy2, xy2), fp_add(xy2, xy2));

    // m = 3*xx + a*zzzz. For STARK curve a=1, so a_mont = R mod p,
    // and fp_mont_mul(R, zzzz) = R * zzzz * R^{-1} = zzzz. Skip the mul.
    Fp252 m = fp_add(fp_add(xx, fp_add(xx, xx)), zzzz);

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

// Mixed addition: p1 (Jacobian) + p2 (affine, Z=R i.e. Montgomery 1).
// Saves 5 fp_mont_mul vs full Jacobian add (11 vs 16).
// Precondition: p2.z must be R mod p (Montgomery representation of 1).
__device__ ProjPoint mont_add_mixed(ProjPoint p1, ProjPoint p2) {
    if (fp_is_zero(p1.z)) return p2;
    // p2 is affine (Z=R, never infinity if from table with nibble != 0)

    Fp252 z1z1 = fp_mont_mul(p1.z, p1.z);
    // u1 = p1.x (since z2z2 = R*R*R^{-1} = R, and x1*R*R^{-1} = x1)
    Fp252 u2 = fp_mont_mul(p2.x, z1z1);
    // s1 = p1.y (since z2z2*z2 all cancel to identity in Montgomery)
    Fp252 s2 = fp_mont_mul(fp_mont_mul(p2.y, z1z1), p1.z);

    Fp252 h = fp_sub(u2, p1.x);
    Fp252 rr = fp_sub(s2, p1.y);

    if (fp_is_zero(h)) {
        if (fp_is_zero(rr)) return mont_double(p1);
        return proj_infinity();
    }

    Fp252 hh = fp_mont_mul(h, h);
    Fp252 hhh = fp_mont_mul(hh, h);
    Fp252 u1hh = fp_mont_mul(p1.x, hh);

    Fp252 x3 = fp_sub(fp_sub(fp_mont_mul(rr, rr), hhh), fp_add(u1hh, u1hh));
    Fp252 y3 = fp_sub(fp_mont_mul(rr, fp_sub(u1hh, x3)), fp_mont_mul(p1.y, hhh));
    Fp252 z3 = fp_mont_mul(p1.z, h); // skip z2 mul since z2=R is identity

    ProjPoint res; res.x = x3; res.y = y3; res.z = z3;
    return res;
}

// Scalar mul with Montgomery EC ops (bit-by-bit, kept for reference/debug)
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

// Fixed-base windowed scalar mul using L1-cached global memory tables.
// Processes scalar MSB-to-LSB in 4-bit windows.
// 248-bit scalar: 62 windows x (4 doublings + 1 table-add) = 248 doublings + 62 additions.
// 4-bit scalar: 1 table lookup, zero doublings.
__device__ ProjPoint windowed_scalar_mul(int point_idx, Fp252 scalar, int n_bits) {
    int n_windows = (n_bits + 3) / 4;
    ProjPoint result = proj_infinity();

    for (int w = n_windows - 1; w >= 0; w--) {
        if (w < n_windows - 1) {
            result = mont_double(result);
            result = mont_double(result);
            result = mont_double(result);
            result = mont_double(result);
        }

        int bit_offset = w * 4;
        int limb_idx = bit_offset / 64;
        int bit_in_limb = bit_offset % 64;
        uint32_t nibble;
        if (limb_idx < 4) {
            nibble = (uint32_t)((scalar.v[limb_idx] >> bit_in_limb) & 0xF);
            if (bit_in_limb > 60 && limb_idx + 1 < 4) {
                nibble |= (uint32_t)((scalar.v[limb_idx + 1] << (64 - bit_in_limb)) & 0xF);
            }
        } else {
            nibble = 0;
        }

        if (nibble != 0) {
            ProjPoint table_pt;
            table_pt.x = G_WTABLE_X[point_idx][nibble];
            table_pt.y = G_WTABLE_Y[point_idx][nibble];
            table_pt.z = G_WTABLE_Z[point_idx][nibble];
            // Table points have Z=R (Montgomery 1) — use mixed addition (11 vs 16 muls)
            result = mont_add_mixed(result, table_pt);
        }
    }

    return result;
}

// Compute Pedersen hash using windowed fixed-base scalar multiplication.
// H(a,b) = P0 + a_low*P1 + a_high*P2 + b_low*P3 + b_high*P4
__device__ ProjPoint pedersen_hash_proj(Fp252 a, Fp252 b) {
    Fp252 a_low  = fp_mask(a, 248);
    Fp252 a_high = fp_shr(a, 248);
    Fp252 b_low  = fp_mask(b, 248);
    Fp252 b_high = fp_shr(b, 248);

    // Start with P0 (precomputed in Montgomery Jacobian, no scalar mul needed)
    ProjPoint result;
    result.x = G_P0_MONT_X;
    result.y = G_P0_MONT_Y;
    result.z = G_P0_MONT_Z;

    // a_low * P1: 248-bit scalar, 62 windows
    result = mont_add(result, windowed_scalar_mul(0, a_low, 248));

    // a_high * P2: 4-bit scalar — single table lookup, result has Z=R (affine)
    {
        uint32_t nibble = (uint32_t)(a_high.v[0] & 0xF);
        if (nibble != 0) {
            ProjPoint tp; tp.x = G_WTABLE_X[1][nibble]; tp.y = G_WTABLE_Y[1][nibble]; tp.z = G_WTABLE_Z[1][nibble];
            result = mont_add_mixed(result, tp);
        }
    }

    // b_low * P3: 248-bit scalar
    result = mont_add(result, windowed_scalar_mul(2, b_low, 248));

    // b_high * P4: 4-bit scalar — single table lookup, result has Z=R (affine)
    {
        uint32_t nibble = (uint32_t)(b_high.v[0] & 0xF);
        if (nibble != 0) {
            ProjPoint tp; tp.x = G_WTABLE_X[3][nibble]; tp.y = G_WTABLE_Y[3][nibble]; tp.z = G_WTABLE_Z[3][nibble];
            result = mont_add_mixed(result, tp);
        }
    }

    return result;
}

// =====================================================================
// EC Trace generation kernel: outputs intermediate Jacobian points
// at each doubling/addition step for constraint verification.
// Each thread processes one hash and writes 622 rows to the trace.
// =====================================================================

#define ROWS_PER_HASH 622
#define EC_TRACE_VALS_PER_ROW 13  // X(4) + Y(4) + Z(4) + op_type(1)
#define OP_INIT 2
#define OP_DOUBLE_TRACE 0
#define OP_ADD_TRACE 1

__device__ void write_ec_row(
    uint64_t* trace,       // [n_hashes * ROWS_PER_HASH * 12] — X,Y,Z per step
    uint32_t* ops,         // [n_hashes * ROWS_PER_HASH] — op type
    uint32_t row,
    ProjPoint pt,
    uint32_t op
) {
    // Write from_mont coordinates
    Fp252 x = from_mont(pt.x);
    Fp252 y = from_mont(pt.y);
    Fp252 z = from_mont(pt.z);
    uint32_t base = row * 12;
    for (int j = 0; j < 4; j++) {
        trace[base + j] = x.v[j];
        trace[base + 4 + j] = y.v[j];
        trace[base + 8 + j] = z.v[j];
    }
    ops[row] = op;
}

__device__ uint32_t extract_nibble(Fp252 scalar, int w) {
    int bit_offset = w * 4;
    int limb_idx = bit_offset / 64;
    int bit_in_limb = bit_offset % 64;
    uint32_t nibble = 0;
    if (limb_idx < 4) {
        nibble = (uint32_t)((scalar.v[limb_idx] >> bit_in_limb) & 0xF);
        if (bit_in_limb > 60 && limb_idx + 1 < 4) {
            nibble |= (uint32_t)((scalar.v[limb_idx + 1] << (64 - bit_in_limb)) & 0xF);
        }
    }
    return nibble;
}

__global__ void pedersen_ec_trace_kernel(
    const uint64_t* __restrict__ inputs_a,
    const uint64_t* __restrict__ inputs_b,
    uint64_t* __restrict__ ec_trace,  // [n * ROWS_PER_HASH * 12] — X,Y,Z per step
    uint32_t* __restrict__ ec_ops,    // [n * ROWS_PER_HASH] — op type
    uint32_t n
) {
    uint32_t hash_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hash_idx >= n) return;

    Fp252 a, b;
    a.v[0] = inputs_a[hash_idx*4+0]; a.v[1] = inputs_a[hash_idx*4+1];
    a.v[2] = inputs_a[hash_idx*4+2]; a.v[3] = inputs_a[hash_idx*4+3];
    b.v[0] = inputs_b[hash_idx*4+0]; b.v[1] = inputs_b[hash_idx*4+1];
    b.v[2] = inputs_b[hash_idx*4+2]; b.v[3] = inputs_b[hash_idx*4+3];

    Fp252 a_low  = fp_mask(a, 248);
    Fp252 a_high = fp_shr(a, 248);
    Fp252 b_low  = fp_mask(b, 248);
    Fp252 b_high = fp_shr(b, 248);

    uint32_t row = hash_idx * ROWS_PER_HASH;

    // Start with P0
    ProjPoint result;
    result.x = G_P0_MONT_X;
    result.y = G_P0_MONT_Y;
    result.z = G_P0_MONT_Z;
    write_ec_row(ec_trace, ec_ops, row++, result, OP_INIT);

    // Process 2 big scalars (a_low*P1, b_low*P3)
    Fp252 big_scalars[2] = {a_low, b_low};
    int big_point_idx[2] = {0, 2};  // P1, P3

    for (int si = 0; si < 2; si++) {
        Fp252 scalar = big_scalars[si];
        int pi = big_point_idx[si];

        for (int w = 61; w >= 0; w--) {
            // 4 doublings
            for (int d = 0; d < 4; d++) {
                write_ec_row(ec_trace, ec_ops, row++, result, OP_DOUBLE_TRACE);
                result = mont_double(result);
            }
            // Table add
            uint32_t nibble = extract_nibble(scalar, w);
            write_ec_row(ec_trace, ec_ops, row, result, OP_ADD_TRACE);
            if (nibble != 0) {
                ProjPoint tp;
                tp.x = G_WTABLE_X[pi][nibble];
                tp.y = G_WTABLE_Y[pi][nibble];
                tp.z = G_WTABLE_Z[pi][nibble];
                result = mont_add_mixed(result, tp);
            }
            row++;
        }
    }

    // 2 small scalars (a_high*P2, b_high*P4)
    uint32_t small_nibbles[2] = {(uint32_t)(a_high.v[0] & 0xF), (uint32_t)(b_high.v[0] & 0xF)};
    int small_point_idx[2] = {1, 3};

    for (int si = 0; si < 2; si++) {
        uint32_t nibble = small_nibbles[si];
        write_ec_row(ec_trace, ec_ops, row, result, OP_ADD_TRACE);
        if (nibble != 0) {
            ProjPoint tp;
            tp.x = G_WTABLE_X[small_point_idx[si]][nibble];
            tp.y = G_WTABLE_Y[small_point_idx[si]][nibble];
            tp.z = G_WTABLE_Z[small_point_idx[si]][nibble];
            result = mont_add_mixed(result, tp);
        }
        row++;
    }
}

// Each thread computes one Pedersen hash and converts to affine x on-GPU.
// Inline Fermat inverse eliminates CPU batch inverse (was 81.6% of pipeline).
// Outputs affine x directly — no Z^2 transfer, no CPU post-processing.
__global__ void pedersen_batch_kernel(
    const uint64_t* __restrict__ inputs_a,
    const uint64_t* __restrict__ inputs_b,
    uint64_t* __restrict__ out_x,   // [n * 4] — affine x (standard form)
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

    // Compute affine x = X * Z^{-2} entirely on GPU (all in Montgomery domain)
    Fp252 zz_mont = fp_mont_mul(result.z, result.z);
    Fp252 zz_inv_mont = fp_mont_inverse(zz_mont);
    Fp252 ax_mont = fp_mont_mul(result.x, zz_inv_mont);
    Fp252 ax = from_mont(ax_mont);

    out_x[i*4+0] = ax.v[0]; out_x[i*4+1] = ax.v[1];
    out_x[i*4+2] = ax.v[2]; out_x[i*4+3] = ax.v[3];
}

// Decompose Fp252 (4×u64, 252 bits) into 9 × 31-bit M31 limbs.
// Writes each limb to a separate column buffer (SoA layout for NTT).
__device__ void fp252_to_m31_columns(
    Fp252 val,
    uint32_t** cols,  // [9] pointers to column buffers
    uint32_t row      // row index
) {
    // Extract 31-bit limbs from the 256-bit value (only 252 bits are meaningful)
    uint32_t limbs[9];
    // Limb 0: bits 0..30
    limbs[0] = (uint32_t)(val.v[0] & 0x7FFFFFFFULL);
    // Limb 1: bits 31..61
    limbs[1] = (uint32_t)((val.v[0] >> 31) & 0x7FFFFFFFULL);
    // Limb 2: bits 62..92 (spans v[0] and v[1])
    limbs[2] = (uint32_t)(((val.v[0] >> 62) | (val.v[1] << 2)) & 0x7FFFFFFFULL);
    // Limb 3: bits 93..123
    limbs[3] = (uint32_t)((val.v[1] >> 29) & 0x7FFFFFFFULL);
    // Limb 4: bits 124..154 (spans v[1] and v[2])
    limbs[4] = (uint32_t)(((val.v[1] >> 60) | (val.v[2] << 4)) & 0x7FFFFFFFULL);
    // Limb 5: bits 155..185
    limbs[5] = (uint32_t)((val.v[2] >> 27) & 0x7FFFFFFFULL);
    // Limb 6: bits 186..216 (spans v[2] and v[3])
    limbs[6] = (uint32_t)(((val.v[2] >> 58) | (val.v[3] << 6)) & 0x7FFFFFFFULL);
    // Limb 7: bits 217..247
    limbs[7] = (uint32_t)((val.v[3] >> 25) & 0x7FFFFFFFULL);
    // Limb 8: bits 248..251 (top 4 bits)
    limbs[8] = (uint32_t)((val.v[3] >> 56) & 0xFULL);

    for (int j = 0; j < 9; j++) {
        cols[j][row] = limbs[j];
    }
}

// Fused Pedersen hash + trace column generation.
// Each thread: hash one (a, b) pair → decompose a, b, output into 27 M31 columns.
// Results stay entirely on GPU — no host round-trip.
// Column layout: [a_limbs(9), b_limbs(9), output_limbs(9)] = 27 columns.
__global__ void pedersen_trace_kernel(
    const uint64_t* __restrict__ inputs_a,  // [n * 4] Fp values
    const uint64_t* __restrict__ inputs_b,  // [n * 4] Fp values
    uint32_t** __restrict__ trace_cols,      // [27] device pointers to column buffers
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Load inputs
    Fp252 a, b;
    a.v[0] = inputs_a[i*4+0]; a.v[1] = inputs_a[i*4+1];
    a.v[2] = inputs_a[i*4+2]; a.v[3] = inputs_a[i*4+3];
    b.v[0] = inputs_b[i*4+0]; b.v[1] = inputs_b[i*4+1];
    b.v[2] = inputs_b[i*4+2]; b.v[3] = inputs_b[i*4+3];

    // Compute Pedersen hash (windowed scalar mul + inline Fermat inverse)
    ProjPoint result = pedersen_hash_proj(a, b);
    Fp252 zz_mont = fp_mont_mul(result.z, result.z);
    Fp252 zz_inv_mont = fp_mont_inverse(zz_mont);
    Fp252 ax_mont = fp_mont_mul(result.x, zz_inv_mont);
    Fp252 output = from_mont(ax_mont);

    // Decompose all three Fp values into 9 M31 limbs each → 27 trace columns
    fp252_to_m31_columns(a, &trace_cols[0], i);       // cols 0..8: input a
    fp252_to_m31_columns(b, &trace_cols[9], i);       // cols 9..17: input b
    fp252_to_m31_columns(output, &trace_cols[18], i); // cols 18..26: output
}

// Decompose-only kernel: takes pre-computed Fp252 values (a, b, output)
// and writes 27 M31 limb columns. No EC computation — just bit extraction.
// Used when hashes were already computed (during VM execution).
__global__ void fp252_decompose_kernel(
    const uint64_t* __restrict__ vals_a,     // [n * 4] Fp values
    const uint64_t* __restrict__ vals_b,     // [n * 4] Fp values
    const uint64_t* __restrict__ vals_out,   // [n * 4] Fp values
    uint32_t** __restrict__ trace_cols,       // [27] device pointers
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Fp252 a, b, out;
    a.v[0] = vals_a[i*4+0]; a.v[1] = vals_a[i*4+1];
    a.v[2] = vals_a[i*4+2]; a.v[3] = vals_a[i*4+3];
    b.v[0] = vals_b[i*4+0]; b.v[1] = vals_b[i*4+1];
    b.v[2] = vals_b[i*4+2]; b.v[3] = vals_b[i*4+3];
    out.v[0] = vals_out[i*4+0]; out.v[1] = vals_out[i*4+1];
    out.v[2] = vals_out[i*4+2]; out.v[3] = vals_out[i*4+3];

    fp252_to_m31_columns(a, &trace_cols[0], i);
    fp252_to_m31_columns(b, &trace_cols[9], i);
    fp252_to_m31_columns(out, &trace_cols[18], i);
}

extern "C" {

// Launch decompose-only kernel (no hashing, just Fp → M31 limbs).
void cuda_pedersen_decompose(
    const uint64_t* vals_a, const uint64_t* vals_b, const uint64_t* vals_out,
    uint32_t** trace_cols, uint32_t n, cudaStream_t stream
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    fp252_decompose_kernel<<<blocks, threads, 0, stream>>>(
        vals_a, vals_b, vals_out, trace_cols, n);
}

// Launch EC trace generation kernel.
void cuda_pedersen_ec_trace(
    const uint64_t* inputs_a, const uint64_t* inputs_b,
    uint64_t* ec_trace, uint32_t* ec_ops,
    uint32_t n, cudaStream_t stream
) {
    uint32_t threads = 32; // fewer threads — each writes 622 rows
    uint32_t blocks = (n + threads - 1) / threads;
    pedersen_ec_trace_kernel<<<blocks, threads, 0, stream>>>(
        inputs_a, inputs_b, ec_trace, ec_ops, n);
}

// Decompose raw Jacobian trace (u64) into M31 SoA columns.
// Input: ec_trace[n_rows * 12] (X,Y,Z as 4×u64 each), ec_ops[n_rows]
// Output: 28 trace columns in SoA format (9 limbs for X, 9 for Y, 9 for Z, 1 op_type)
__global__ void ec_trace_decompose_kernel(
    const uint64_t* __restrict__ ec_trace,
    const uint32_t* __restrict__ ec_ops,
    uint32_t** __restrict__ trace_cols,  // [28] column pointers
    uint32_t n_rows
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;

    // Read X, Y, Z (standard form, 4×u64 each)
    Fp252 x, y, z;
    uint32_t base = i * 12;
    for (int j = 0; j < 4; j++) {
        x.v[j] = ec_trace[base + j];
        y.v[j] = ec_trace[base + 4 + j];
        z.v[j] = ec_trace[base + 8 + j];
    }

    // Decompose each Fp252 into 9 M31 limbs (same as fp252_to_m31_columns)
    // X limbs → cols 0..8
    uint32_t lx[9], ly[9], lz[9];
    lx[0] = (uint32_t)(x.v[0] & 0x7FFFFFFFULL);
    lx[1] = (uint32_t)((x.v[0] >> 31) & 0x7FFFFFFFULL);
    lx[2] = (uint32_t)(((x.v[0] >> 62) | (x.v[1] << 2)) & 0x7FFFFFFFULL);
    lx[3] = (uint32_t)((x.v[1] >> 29) & 0x7FFFFFFFULL);
    lx[4] = (uint32_t)(((x.v[1] >> 60) | (x.v[2] << 4)) & 0x7FFFFFFFULL);
    lx[5] = (uint32_t)((x.v[2] >> 27) & 0x7FFFFFFFULL);
    lx[6] = (uint32_t)(((x.v[2] >> 58) | (x.v[3] << 6)) & 0x7FFFFFFFULL);
    lx[7] = (uint32_t)((x.v[3] >> 25) & 0x7FFFFFFFULL);
    lx[8] = (uint32_t)((x.v[3] >> 56) & 0xFULL);

    ly[0] = (uint32_t)(y.v[0] & 0x7FFFFFFFULL);
    ly[1] = (uint32_t)((y.v[0] >> 31) & 0x7FFFFFFFULL);
    ly[2] = (uint32_t)(((y.v[0] >> 62) | (y.v[1] << 2)) & 0x7FFFFFFFULL);
    ly[3] = (uint32_t)((y.v[1] >> 29) & 0x7FFFFFFFULL);
    ly[4] = (uint32_t)(((y.v[1] >> 60) | (y.v[2] << 4)) & 0x7FFFFFFFULL);
    ly[5] = (uint32_t)((y.v[2] >> 27) & 0x7FFFFFFFULL);
    ly[6] = (uint32_t)(((y.v[2] >> 58) | (y.v[3] << 6)) & 0x7FFFFFFFULL);
    ly[7] = (uint32_t)((y.v[3] >> 25) & 0x7FFFFFFFULL);
    ly[8] = (uint32_t)((y.v[3] >> 56) & 0xFULL);

    lz[0] = (uint32_t)(z.v[0] & 0x7FFFFFFFULL);
    lz[1] = (uint32_t)((z.v[0] >> 31) & 0x7FFFFFFFULL);
    lz[2] = (uint32_t)(((z.v[0] >> 62) | (z.v[1] << 2)) & 0x7FFFFFFFULL);
    lz[3] = (uint32_t)((z.v[1] >> 29) & 0x7FFFFFFFULL);
    lz[4] = (uint32_t)(((z.v[1] >> 60) | (z.v[2] << 4)) & 0x7FFFFFFFULL);
    lz[5] = (uint32_t)((z.v[2] >> 27) & 0x7FFFFFFFULL);
    lz[6] = (uint32_t)(((z.v[2] >> 58) | (z.v[3] << 6)) & 0x7FFFFFFFULL);
    lz[7] = (uint32_t)((z.v[3] >> 25) & 0x7FFFFFFFULL);
    lz[8] = (uint32_t)((z.v[3] >> 56) & 0xFULL);

    for (int j = 0; j < 9; j++) {
        trace_cols[j][i] = lx[j];      // X limbs: cols 0..8
        trace_cols[9+j][i] = ly[j];     // Y limbs: cols 9..17
        trace_cols[18+j][i] = lz[j];    // Z limbs: cols 18..26
    }
    trace_cols[27][i] = ec_ops[i];       // op_type: col 27
}

void cuda_ec_trace_decompose(
    const uint64_t* ec_trace, const uint32_t* ec_ops,
    uint32_t** trace_cols, uint32_t n_rows, cudaStream_t stream
) {
    uint32_t threads = 256;
    uint32_t blocks = (n_rows + threads - 1) / threads;
    ec_trace_decompose_kernel<<<blocks, threads, 0, stream>>>(
        ec_trace, ec_ops, trace_cols, n_rows);
}

// Launch fused Pedersen hash + trace generation on a stream.
// trace_cols: device array of 27 device pointers (uint32_t*), one per column.
void cuda_pedersen_trace(
    const uint64_t* inputs_a,
    const uint64_t* inputs_b,
    uint32_t** trace_cols,
    uint32_t n,
    cudaStream_t stream
) {
    uint32_t threads = 128;
    uint32_t blocks = (n + threads - 1) / threads;
    pedersen_trace_kernel<<<blocks, threads, 0, stream>>>(
        inputs_a, inputs_b, trace_cols, n);
}

// Kernel to copy constant memory tables to global __device__ memory.
// Called once at init time after cudaMemcpyToSymbol uploads.
__global__ void copy_tables_to_global() {
    for (int p = 0; p < 4; p++) {
        for (int k = 0; k < 16; k++) {
            G_WTABLE_X[p][k] = WTABLE_X[p][k];
            G_WTABLE_Y[p][k] = WTABLE_Y[p][k];
            G_WTABLE_Z[p][k] = WTABLE_Z[p][k];
        }
    }
    G_P0_MONT_X = P0_MONT_X;
    G_P0_MONT_Y = P0_MONT_Y;
    G_P0_MONT_Z = P0_MONT_Z;
}

// Upload precomputed windowed tables + P0 in Montgomery form
void cuda_pedersen_upload_tables(
    const uint64_t* table_x, const uint64_t* table_y, const uint64_t* table_z,
    const uint64_t* p0_x, const uint64_t* p0_y, const uint64_t* p0_z
) {
    cudaMemcpyToSymbol(WTABLE_X, table_x, 4 * 16 * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(WTABLE_Y, table_y, 4 * 16 * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(WTABLE_Z, table_z, 4 * 16 * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(P0_MONT_X, p0_x, 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(P0_MONT_Y, p0_y, 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(P0_MONT_Z, p0_z, 4 * sizeof(uint64_t));
    copy_tables_to_global<<<1, 1>>>();
    cudaDeviceSynchronize();
}

// Upload the 5 Pedersen constant points to GPU constant memory
void cuda_pedersen_upload_points(const uint64_t* px, const uint64_t* py) {
    cudaMemcpyToSymbol(PEDERSEN_PX, px, 5 * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(PEDERSEN_PY, py, 5 * 4 * sizeof(uint64_t));
}

// Batch Pedersen hash on GPU — outputs affine x directly (inline Fermat inverse)
void cuda_pedersen_hash_batch(
    const uint64_t* inputs_a,
    const uint64_t* inputs_b,
    uint64_t* out_x,
    uint64_t* out_zz, // UNUSED — kept for FFI compat, will be removed
    uint32_t n
) {
    uint32_t threads = 128;
    uint32_t blocks = (n + threads - 1) / threads;
    pedersen_batch_kernel<<<blocks, threads>>>(inputs_a, inputs_b, out_x, n);
}

// Batch Pedersen hash on a CUDA stream (async, for overlapped transfers)
void cuda_pedersen_hash_batch_stream(
    const uint64_t* inputs_a,
    const uint64_t* inputs_b,
    uint64_t* out_x,
    uint64_t* out_zz, // unused
    uint32_t n,
    cudaStream_t stream
) {
    uint32_t threads = 128;
    uint32_t blocks = (n + threads - 1) / threads;
    pedersen_batch_kernel<<<blocks, threads, 0, stream>>>(inputs_a, inputs_b, out_x, n);
}

// Debug kernel: test EC point doubling of a known affine point
void cuda_pedersen_test_double(
    const uint64_t* px, const uint64_t* py,
    uint64_t* out_x, uint64_t* out_y, uint64_t* out_z
);

__global__ void pedersen_test_double_kernel(
    const uint64_t* px, const uint64_t* py,
    uint64_t* out_x, uint64_t* out_y, uint64_t* out_z
) {
    Fp252 x = {{px[0], px[1], px[2], px[3]}};
    Fp252 y = {{py[0], py[1], py[2], py[3]}};

    Fp252 xx = fp_mul(x, x);
    Fp252 yy = fp_mul(y, y);

    out_z[0] = xx.v[0]; out_z[1] = xx.v[1]; out_z[2] = xx.v[2]; out_z[3] = xx.v[3];

    ProjPoint p = proj_from_affine(x, y);
    ProjPoint d = mont_proj_double(p);

    out_x[0] = d.x.v[0]; out_x[1] = d.x.v[1]; out_x[2] = d.x.v[2]; out_x[3] = d.x.v[3];
    out_y[0] = d.y.v[0]; out_y[1] = d.y.v[1]; out_y[2] = d.y.v[2]; out_y[3] = d.y.v[3];
}

void cuda_pedersen_test_double(
    const uint64_t* px, const uint64_t* py,
    uint64_t* out_x, uint64_t* out_y, uint64_t* out_z
) {
    pedersen_test_double_kernel<<<1, 1>>>(px, py, out_x, out_y, out_z);
}

} // extern "C"
