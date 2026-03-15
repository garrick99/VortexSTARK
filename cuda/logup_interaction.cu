// GPU LogUp interaction trace generation.
//
// Phase 2 of the Cairo STARK: after committing the execution trace (27 columns),
// draw random challenges z, alpha, z_rc, then compute:
// 1. Memory LogUp: 4 reciprocals per row → running sum
// 2. Range check LogUp: 3 reciprocals per row → running sum
//
// Uses batch inverse (Montgomery's trick) to avoid per-element divisions,
// then parallel prefix sum (scan) for the running sum.

#include "include/qm31.cuh"

// ============================================================================
// Memory LogUp: compute 1/(z - (addr + alpha * value)) for 4 accesses per row
// ============================================================================

// Compute the 4 memory LogUp denominators for each row.
// denom[4*i + j] = z - (addr_j[i] + alpha * val_j[i])
// where j indexes the 4 memory accesses: (pc, inst), (dst_addr, dst), (op0_addr, op0), (op1_addr, op1)
__global__ void logup_memory_denoms_kernel(
    const uint32_t* __restrict__ col_pc,        // column 0
    const uint32_t* __restrict__ col_inst_lo,   // column 3
    const uint32_t* __restrict__ col_dst_addr,  // column 20
    const uint32_t* __restrict__ col_dst,        // column 21
    const uint32_t* __restrict__ col_op0_addr,  // column 22
    const uint32_t* __restrict__ col_op0,        // column 23
    const uint32_t* __restrict__ col_op1_addr,  // column 24
    const uint32_t* __restrict__ col_op1,        // column 25
    uint32_t* __restrict__ denom0, uint32_t* __restrict__ denom1,
    uint32_t* __restrict__ denom2, uint32_t* __restrict__ denom3,  // QM31 SoA output
    uint32_t z0, uint32_t z1, uint32_t z2, uint32_t z3,           // z challenge
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,           // alpha challenge
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    QM31 z = {{z0, z1, z2, z3}};
    QM31 alpha = {{a0, a1, a2, a3}};

    // 4 memory accesses: (addr, value) pairs
    uint32_t addrs[4] = { col_pc[i], col_dst_addr[i], col_op0_addr[i], col_op1_addr[i] };
    uint32_t vals[4]  = { col_inst_lo[i], col_dst[i], col_op0[i], col_op1[i] };

    // Sum of 4 denominators: accumulate z - (addr + alpha * val) for each access
    // We compute the PRODUCT of all 4 denominators and the sum of reciprocals
    // using the identity: sum(1/d_j) = sum(prod(d_k, k!=j)) / prod(d_j)
    // This way we only need ONE batch inverse per row instead of 4.

    // Actually, for the running sum we need the SUM of 1/d_j per row.
    // Compute each d_j, then batch-inverse all 4*n denominators,
    // then sum the 4 reciprocals per row.

    // For now: store all 4 denominators separately, batch inverse externally.
    // Each denom is: z - (addr_qm31 + alpha * val_qm31) = z - addr - alpha*val
    // Since addr and val are M31, the QM31 arithmetic simplifies:
    // z - addr - alpha*val = (z.a.a - addr - alpha.a.a*val, z.a.b - alpha.a.b*val,
    //                         z.b.a - alpha.b.a*val, z.b.b - alpha.b.b*val)

    // We store the sum of all 4 denominators' contributions.
    // But to batch-inverse, we need individual denoms. Let's output per-row sums
    // using the product-of-denoms trick for 4 elements:

    // d0, d1, d2, d3 are the 4 denominators
    QM31 d[4];
    for (int j = 0; j < 4; j++) {
        // d_j = z - addr_j - alpha * val_j (all in QM31)
        QM31 entry = qm31_mul_m31(alpha, vals[j]);
        entry.v[0] = m31_add(entry.v[0], addrs[j]); // add addr to component 0
        d[j] = qm31_sub(z, entry);
    }

    // Compute sum of reciprocals without division:
    // 1/d0 + 1/d1 + 1/d2 + 1/d3
    // = (d1*d2*d3 + d0*d2*d3 + d0*d1*d3 + d0*d1*d2) / (d0*d1*d2*d3)
    //
    // Numerator terms:
    QM31 p01 = qm31_mul(d[0], d[1]);
    QM31 p23 = qm31_mul(d[2], d[3]);
    QM31 p0123 = qm31_mul(p01, p23); // full product

    QM31 p12 = qm31_mul(d[1], d[2]);
    QM31 p03 = qm31_mul(d[0], d[3]);

    // sum = d1*d2*d3 + d0*d2*d3 + d0*d1*d3 + d0*d1*d2
    QM31 num = qm31_add(
        qm31_add(
            qm31_mul(d[1], p23),      // d1*d2*d3
            qm31_mul(d[0], p23)       // d0*d2*d3
        ),
        qm31_add(
            qm31_mul(p01, d[3]),      // d0*d1*d3
            qm31_mul(p01, d[2])       // d0*d1*d2
        )
    );

    // Store numerator and denominator (product) for external batch inverse
    // Output: num / p0123 = sum of reciprocals
    // We output p0123 (the denominator to invert) and num (the numerator to multiply)
    // After batch inverse of p0123: result = num * inv(p0123)

    // For simplicity, output the product (to be batch-inverted) in denom0-3
    // and the numerator in a separate buffer... but we don't have one.
    // Alternative: output the product, batch inverse it, then multiply by num in a second kernel.

    // Let's just output the PRODUCT for batch inverse. The numerator will be recomputed.
    denom0[i] = p0123.v[0];
    denom1[i] = p0123.v[1];
    denom2[i] = p0123.v[2];
    denom3[i] = p0123.v[3];
}

// After batch inverse of p0123, compute the per-row sum of reciprocals
// and accumulate into a running sum via inclusive scan.
// This kernel computes: result[i] = num[i] * inv_p0123[i]
// where num is recomputed from the original trace columns.
__global__ void logup_memory_combine_kernel(
    const uint32_t* __restrict__ col_pc,
    const uint32_t* __restrict__ col_inst_lo,
    const uint32_t* __restrict__ col_dst_addr,
    const uint32_t* __restrict__ col_dst,
    const uint32_t* __restrict__ col_op0_addr,
    const uint32_t* __restrict__ col_op0,
    const uint32_t* __restrict__ col_op1_addr,
    const uint32_t* __restrict__ col_op1,
    const uint32_t* __restrict__ inv0, const uint32_t* __restrict__ inv1,
    const uint32_t* __restrict__ inv2, const uint32_t* __restrict__ inv3,
    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2, uint32_t* __restrict__ out3,
    uint32_t z0, uint32_t z1, uint32_t z2, uint32_t z3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    QM31 z = {{z0, z1, z2, z3}};
    QM31 alpha = {{a0, a1, a2, a3}};

    uint32_t addrs[4] = { col_pc[i], col_dst_addr[i], col_op0_addr[i], col_op1_addr[i] };
    uint32_t vals[4]  = { col_inst_lo[i], col_dst[i], col_op0[i], col_op1[i] };

    QM31 d[4];
    for (int j = 0; j < 4; j++) {
        QM31 entry = qm31_mul_m31(alpha, vals[j]);
        entry.v[0] = m31_add(entry.v[0], addrs[j]);
        d[j] = qm31_sub(z, entry);
    }

    QM31 p01 = qm31_mul(d[0], d[1]);
    QM31 p23 = qm31_mul(d[2], d[3]);

    QM31 num = qm31_add(
        qm31_add(qm31_mul(d[1], p23), qm31_mul(d[0], p23)),
        qm31_add(qm31_mul(p01, d[3]), qm31_mul(p01, d[2]))
    );

    // Multiply numerator by the batch-inverted product
    QM31 inv_prod = {{inv0[i], inv1[i], inv2[i], inv3[i]}};
    QM31 result = qm31_mul(num, inv_prod);

    out0[i] = result.v[0];
    out1[i] = result.v[1];
    out2[i] = result.v[2];
    out3[i] = result.v[3];
}

// ============================================================================
// FUSED LogUp: denoms + inverse + combine in one kernel (zero intermediate storage)
// ============================================================================

// Fused kernel: computes per-row sum of 4 memory reciprocals with inline QM31 inverse.
// Eliminates 2 global memory round-trips (denoms→inverse→combine → single pass).
// Each thread: compute 4 denoms, product, inline inverse, numerator, multiply → output.
__global__ void logup_memory_fused_kernel(
    const uint32_t* __restrict__ col_pc,
    const uint32_t* __restrict__ col_inst_lo,
    const uint32_t* __restrict__ col_dst_addr,
    const uint32_t* __restrict__ col_dst,
    const uint32_t* __restrict__ col_op0_addr,
    const uint32_t* __restrict__ col_op0,
    const uint32_t* __restrict__ col_op1_addr,
    const uint32_t* __restrict__ col_op1,
    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2, uint32_t* __restrict__ out3,
    uint32_t z0, uint32_t z1, uint32_t z2, uint32_t z3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    QM31 z = {{z0, z1, z2, z3}};
    QM31 alpha = {{a0, a1, a2, a3}};

    uint32_t addrs[4] = { col_pc[i], col_dst_addr[i], col_op0_addr[i], col_op1_addr[i] };
    uint32_t vals[4]  = { col_inst_lo[i], col_dst[i], col_op0[i], col_op1[i] };

    // Compute 4 denominators
    QM31 d[4];
    for (int j = 0; j < 4; j++) {
        QM31 entry = qm31_mul_m31(alpha, vals[j]);
        entry.v[0] = m31_add(entry.v[0], addrs[j]);
        d[j] = qm31_sub(z, entry);
    }

    // Product of all 4 denominators
    QM31 p01 = qm31_mul(d[0], d[1]);
    QM31 p23 = qm31_mul(d[2], d[3]);
    QM31 p0123 = qm31_mul(p01, p23);

    // Numerator = sum of co-factor products
    QM31 num = qm31_add(
        qm31_add(qm31_mul(d[1], p23), qm31_mul(d[0], p23)),
        qm31_add(qm31_mul(p01, d[3]), qm31_mul(p01, d[2]))
    );

    // Inline QM31 inverse of p0123 (no global memory round-trip)
    QM31 inv_prod = qm31_inv(p0123);

    // Result = numerator * inv(product)
    QM31 result = qm31_mul(num, inv_prod);

    out0[i] = result.v[0];
    out1[i] = result.v[1];
    out2[i] = result.v[2];
    out3[i] = result.v[3];
}

// ============================================================================
// CHUNKED LogUp: process one (addr, value) pair at a time.
// Accumulates 1/(z - addr - alpha*val) into running output.
// Called 4 times (once per memory access) with only 2 columns in VRAM.
// Enables log_n=28+ by never holding more than 2 eval columns.
// ============================================================================

__global__ void logup_accumulate_pair_kernel(
    const uint32_t* __restrict__ col_addr,
    const uint32_t* __restrict__ col_val,
    uint32_t* __restrict__ acc0, uint32_t* __restrict__ acc1,
    uint32_t* __restrict__ acc2, uint32_t* __restrict__ acc3,
    uint32_t z0, uint32_t z1, uint32_t z2, uint32_t z3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t n,
    uint32_t is_first  // if 1, write instead of accumulate
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    QM31 z = {{z0, z1, z2, z3}};
    QM31 alpha = {{a0, a1, a2, a3}};

    // denom = z - (addr + alpha * val)
    QM31 entry = qm31_mul_m31(alpha, col_val[i]);
    entry.v[0] = m31_add(entry.v[0], col_addr[i]);
    QM31 denom = qm31_sub(z, entry);

    // 1/denom via inline QM31 inverse
    QM31 inv = qm31_inv(denom);

    if (is_first) {
        acc0[i] = inv.v[0]; acc1[i] = inv.v[1];
        acc2[i] = inv.v[2]; acc3[i] = inv.v[3];
    } else {
        // Accumulate: acc += 1/denom
        QM31 prev = {{acc0[i], acc1[i], acc2[i], acc3[i]}};
        QM31 sum = qm31_add(prev, inv);
        acc0[i] = sum.v[0]; acc1[i] = sum.v[1];
        acc2[i] = sum.v[2]; acc3[i] = sum.v[3];
    }
}

// ============================================================================
// QM31 Parallel Prefix Sum (Inclusive Scan)
// ============================================================================

// Simple block-level inclusive scan for QM31 values (stored as 4 M31 columns).
// Each block processes BLOCK_SIZE elements. Inter-block reduction done on host.
// For n <= 256 * BLOCK_SIZE, a single pass suffices.
__global__ void qm31_block_scan_kernel(
    uint32_t* __restrict__ c0, uint32_t* __restrict__ c1,
    uint32_t* __restrict__ c2, uint32_t* __restrict__ c3,
    uint32_t* __restrict__ block_sums0, uint32_t* __restrict__ block_sums1,
    uint32_t* __restrict__ block_sums2, uint32_t* __restrict__ block_sums3,
    uint32_t n
) {
    extern __shared__ uint32_t smem[]; // 4 * blockDim.x

    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;

    // Load into shared memory
    uint32_t v[4] = {0, 0, 0, 0};
    if (gid < n) {
        v[0] = c0[gid]; v[1] = c1[gid]; v[2] = c2[gid]; v[3] = c3[gid];
    }
    for (int k = 0; k < 4; k++) smem[k * blockDim.x + tid] = v[k];
    __syncthreads();

    // Hillis-Steele inclusive scan within block
    for (uint32_t stride = 1; stride < blockDim.x; stride <<= 1) {
        uint32_t prev[4] = {0, 0, 0, 0};
        if (tid >= stride) {
            for (int k = 0; k < 4; k++)
                prev[k] = smem[k * blockDim.x + tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            for (int k = 0; k < 4; k++) {
                // QM31 addition (component-wise M31 add)
                smem[k * blockDim.x + tid] = m31_add(smem[k * blockDim.x + tid], prev[k]);
            }
        }
        __syncthreads();
    }

    // Write scanned values back
    if (gid < n) {
        c0[gid] = smem[0 * blockDim.x + tid];
        c1[gid] = smem[1 * blockDim.x + tid];
        c2[gid] = smem[2 * blockDim.x + tid];
        c3[gid] = smem[3 * blockDim.x + tid];
    }

    // Write block sum (last element of each block)
    if (tid == blockDim.x - 1 && block_sums0 != nullptr) {
        block_sums0[blockIdx.x] = smem[0 * blockDim.x + tid];
        block_sums1[blockIdx.x] = smem[1 * blockDim.x + tid];
        block_sums2[blockIdx.x] = smem[2 * blockDim.x + tid];
        block_sums3[blockIdx.x] = smem[3 * blockDim.x + tid];
    }
}

// Add block prefix to all elements in subsequent blocks
__global__ void qm31_add_block_prefix_kernel(
    uint32_t* __restrict__ c0, uint32_t* __restrict__ c1,
    uint32_t* __restrict__ c2, uint32_t* __restrict__ c3,
    const uint32_t* __restrict__ prefix0, const uint32_t* __restrict__ prefix1,
    const uint32_t* __restrict__ prefix2, const uint32_t* __restrict__ prefix3,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n || blockIdx.x == 0) return; // block 0 has no prefix to add

    uint32_t block_prefix_idx = blockIdx.x - 1;
    if (gid < n) {
        c0[gid] = m31_add(c0[gid], prefix0[block_prefix_idx]);
        c1[gid] = m31_add(c1[gid], prefix1[block_prefix_idx]);
        c2[gid] = m31_add(c2[gid], prefix2[block_prefix_idx]);
        c3[gid] = m31_add(c3[gid], prefix3[block_prefix_idx]);
    }
}

// ============================================================================
// QM31 Batch Inverse for SoA layout
// ============================================================================

// Batch inverse for QM31 values stored in SoA (4 M31 columns).
// Uses Montgomery's trick but operates on QM31 elements.
// This is more complex than M31 batch inverse because QM31 multiplication
// is non-trivial. For now, use per-element inverse via Fermat/conjugate.
__global__ void qm31_inverse_kernel(
    const uint32_t* __restrict__ in0, const uint32_t* __restrict__ in1,
    const uint32_t* __restrict__ in2, const uint32_t* __restrict__ in3,
    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2, uint32_t* __restrict__ out3,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    QM31 val = {{in0[i], in1[i], in2[i], in3[i]}};

    // QM31 inverse via conjugate and norm:
    // inv(a + bu) = conj(a + bu) / norm(a + bu)
    // conj(a + bu) = a - bu
    // norm(a + bu) = a*a - b*b*(2+i) ∈ CM31
    // Then inv_norm = CM31 inverse, result = conj * inv_norm

    QM31 conj = {{val.v[0], val.v[1], m31_neg(val.v[2]), m31_neg(val.v[3])}};

    // norm = val * conj = (a + bu)(a - bu) = a^2 - b^2 * (2+i)
    // a = (val.v[0], val.v[1]) as CM31
    // b = (val.v[2], val.v[3]) as CM31 (before negation)
    // a^2: CM31 mul
    uint32_t a_r = val.v[0], a_i = val.v[1];
    uint32_t b_r = val.v[2], b_i = val.v[3];

    // a^2 = (a_r^2 - a_i^2, 2*a_r*a_i)
    uint32_t a2_r = m31_sub(m31_mul(a_r, a_r), m31_mul(a_i, a_i));
    uint32_t a2_i = m31_add(m31_mul(a_r, a_i), m31_mul(a_r, a_i));

    // b^2 = (b_r^2 - b_i^2, 2*b_r*b_i)
    uint32_t b2_r = m31_sub(m31_mul(b_r, b_r), m31_mul(b_i, b_i));
    uint32_t b2_i = m31_add(m31_mul(b_r, b_i), m31_mul(b_r, b_i));

    // b^2 * (2+i) = (2*b2_r - b2_i, 2*b2_i + b2_r)
    uint32_t b2u_r = m31_sub(m31_add(b2_r, b2_r), b2_i);
    uint32_t b2u_i = m31_add(m31_add(b2_i, b2_i), b2_r);

    // norm = a^2 - b^2*(2+i) ∈ CM31
    uint32_t norm_r = m31_sub(a2_r, b2u_r);
    uint32_t norm_i = m31_sub(a2_i, b2u_i);

    // CM31 inverse: (r - ii) / (r^2 + i^2)
    uint32_t norm_sq = m31_add(m31_mul(norm_r, norm_r), m31_mul(norm_i, norm_i));
    uint32_t norm_sq_inv = m31_inv(norm_sq);
    uint32_t inv_norm_r = m31_mul(norm_r, norm_sq_inv);
    uint32_t inv_norm_i = m31_neg(m31_mul(norm_i, norm_sq_inv));

    // result = conj * inv_norm (QM31 * CM31 = QM31)
    // (conj.a + conj.b * u) * inv_norm
    // conj.a * inv_norm + conj.b * inv_norm * u
    // where conj.a = (val.v[0], val.v[1]), conj.b = (-val.v[2], -val.v[3])
    // CM31 mul: (a_r, a_i) * (b_r, b_i) = (a_r*b_r - a_i*b_i, a_r*b_i + a_i*b_r)

    // result.a = conj.a * inv_norm
    uint32_t ra_r = m31_sub(m31_mul(conj.v[0], inv_norm_r), m31_mul(conj.v[1], inv_norm_i));
    uint32_t ra_i = m31_add(m31_mul(conj.v[0], inv_norm_i), m31_mul(conj.v[1], inv_norm_r));

    // result.b = conj.b * inv_norm
    uint32_t rb_r = m31_sub(m31_mul(conj.v[2], inv_norm_r), m31_mul(conj.v[3], inv_norm_i));
    uint32_t rb_i = m31_add(m31_mul(conj.v[2], inv_norm_i), m31_mul(conj.v[3], inv_norm_r));

    out0[i] = ra_r; out1[i] = ra_i; out2[i] = rb_r; out3[i] = rb_i;
}

extern "C" {

void cuda_logup_accumulate_pair(
    const uint32_t* col_addr, const uint32_t* col_val,
    uint32_t* acc0, uint32_t* acc1, uint32_t* acc2, uint32_t* acc3,
    const uint32_t* z, const uint32_t* alpha,
    uint32_t n, uint32_t is_first
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    logup_accumulate_pair_kernel<<<blocks, threads>>>(
        col_addr, col_val, acc0, acc1, acc2, acc3,
        z[0], z[1], z[2], z[3], alpha[0], alpha[1], alpha[2], alpha[3],
        n, is_first
    );
}

void cuda_logup_memory_fused(
    const uint32_t* col_pc, const uint32_t* col_inst_lo,
    const uint32_t* col_dst_addr, const uint32_t* col_dst,
    const uint32_t* col_op0_addr, const uint32_t* col_op0,
    const uint32_t* col_op1_addr, const uint32_t* col_op1,
    uint32_t* out0, uint32_t* out1, uint32_t* out2, uint32_t* out3,
    const uint32_t* z, const uint32_t* alpha,
    uint32_t n
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    logup_memory_fused_kernel<<<blocks, threads>>>(
        col_pc, col_inst_lo, col_dst_addr, col_dst,
        col_op0_addr, col_op0, col_op1_addr, col_op1,
        out0, out1, out2, out3,
        z[0], z[1], z[2], z[3], alpha[0], alpha[1], alpha[2], alpha[3], n
    );
}

void cuda_logup_memory_denoms(
    const uint32_t* col_pc, const uint32_t* col_inst_lo,
    const uint32_t* col_dst_addr, const uint32_t* col_dst,
    const uint32_t* col_op0_addr, const uint32_t* col_op0,
    const uint32_t* col_op1_addr, const uint32_t* col_op1,
    uint32_t* denom0, uint32_t* denom1, uint32_t* denom2, uint32_t* denom3,
    const uint32_t* z, const uint32_t* alpha,
    uint32_t n
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    logup_memory_denoms_kernel<<<blocks, threads>>>(
        col_pc, col_inst_lo, col_dst_addr, col_dst,
        col_op0_addr, col_op0, col_op1_addr, col_op1,
        denom0, denom1, denom2, denom3,
        z[0], z[1], z[2], z[3], alpha[0], alpha[1], alpha[2], alpha[3], n
    );
}

void cuda_logup_memory_combine(
    const uint32_t* col_pc, const uint32_t* col_inst_lo,
    const uint32_t* col_dst_addr, const uint32_t* col_dst,
    const uint32_t* col_op0_addr, const uint32_t* col_op0,
    const uint32_t* col_op1_addr, const uint32_t* col_op1,
    const uint32_t* inv0, const uint32_t* inv1,
    const uint32_t* inv2, const uint32_t* inv3,
    uint32_t* out0, uint32_t* out1, uint32_t* out2, uint32_t* out3,
    const uint32_t* z, const uint32_t* alpha,
    uint32_t n
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    logup_memory_combine_kernel<<<blocks, threads>>>(
        col_pc, col_inst_lo, col_dst_addr, col_dst,
        col_op0_addr, col_op0, col_op1_addr, col_op1,
        inv0, inv1, inv2, inv3,
        out0, out1, out2, out3,
        z[0], z[1], z[2], z[3], alpha[0], alpha[1], alpha[2], alpha[3], n
    );
}

void cuda_qm31_inverse(
    const uint32_t* in0, const uint32_t* in1,
    const uint32_t* in2, const uint32_t* in3,
    uint32_t* out0, uint32_t* out1, uint32_t* out2, uint32_t* out3,
    uint32_t n
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    qm31_inverse_kernel<<<blocks, threads>>>(
        in0, in1, in2, in3, out0, out1, out2, out3, n
    );
}

void cuda_qm31_block_scan(
    uint32_t* c0, uint32_t* c1, uint32_t* c2, uint32_t* c3,
    uint32_t* block_sums0, uint32_t* block_sums1,
    uint32_t* block_sums2, uint32_t* block_sums3,
    uint32_t n, uint32_t block_size
) {
    uint32_t blocks = (n + block_size - 1) / block_size;
    uint32_t smem = 4 * block_size * sizeof(uint32_t);
    qm31_block_scan_kernel<<<blocks, block_size, smem>>>(
        c0, c1, c2, c3,
        block_sums0, block_sums1, block_sums2, block_sums3, n
    );
}

void cuda_qm31_add_block_prefix(
    uint32_t* c0, uint32_t* c1, uint32_t* c2, uint32_t* c3,
    const uint32_t* prefix0, const uint32_t* prefix1,
    const uint32_t* prefix2, const uint32_t* prefix3,
    uint32_t n, uint32_t block_size
) {
    uint32_t blocks = (n + block_size - 1) / block_size;
    qm31_add_block_prefix_kernel<<<blocks, block_size>>>(
        c0, c1, c2, c3, prefix0, prefix1, prefix2, prefix3, n
    );
}

} // extern "C"
