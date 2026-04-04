// GPU kernels for GKR sum-check protocol.
//
// Implements:
//   fix_first_variable (BaseField→SecureField, SecureField→SecureField)
//   gen_eq_evals       (log_n sequential doubling passes)
//   next_layer         (GrandProduct, LogUpGeneric, LogUpMultiplicities, LogUpSingles)
//   sum_as_poly        (parallel reduction → partial sums for CPU assembly)
//
// Memory layout (same as all CudaColumn<SecureField>):
//   QM31 element i occupies u32 offsets [4i, 4i+1, 4i+2, 4i+3]
//   fold_mle_evals(r, a, b) = r*(b-a)+a  (see stwo/prover/lookups/utils.rs)

#include "include/qm31.cuh"
#include "include/m31.cuh"

// ─── fix_first_variable: BaseField (M31) → SecureField (QM31) ───────────────
//
// in:  n M31 values (n u32s)
// out: n/2 QM31 values (2n u32s)
// out[i] = r * (in[i + n/2] - in[i]) + in[i]
__global__ void gkr_fix_first_variable_base_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out,
    const uint32_t* __restrict__ r_ptr,  // QM31 assignment [4 u32s]
    uint32_t half_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_n) return;

    QM31 r = {{r_ptr[0], r_ptr[1], r_ptr[2], r_ptr[3]}};

    uint32_t a = in[i];
    uint32_t b = in[i + half_n];
    uint32_t diff = m31_sub(b, a);

    // out[i] = r * diff + a  (r is QM31, diff is M31 scalar)
    QM31 result = qm31_mul_m31(r, diff);
    result.v[0] = m31_add(result.v[0], a);

    out[4*i + 0] = result.v[0];
    out[4*i + 1] = result.v[1];
    out[4*i + 2] = result.v[2];
    out[4*i + 3] = result.v[3];
}

// ─── fix_first_variable: SecureField (QM31) → SecureField (QM31) ─────────────
//
// in:  n QM31 values (4n u32s)
// out: n/2 QM31 values (2n u32s), overwritten in first half of in[]
// out[i] = r * (in[i + n/2] - in[i]) + in[i]
__global__ void gkr_fix_first_variable_secure_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out,
    const uint32_t* __restrict__ r_ptr,
    uint32_t half_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_n) return;

    QM31 r = {{r_ptr[0], r_ptr[1], r_ptr[2], r_ptr[3]}};

    QM31 a = {{in[4*i], in[4*i+1], in[4*i+2], in[4*i+3]}};
    QM31 b = {{in[4*(i+half_n)], in[4*(i+half_n)+1], in[4*(i+half_n)+2], in[4*(i+half_n)+3]}};

    // out[i] = r * (b - a) + a
    QM31 diff = qm31_sub(b, a);
    QM31 result = qm31_add(qm31_mul(r, diff), a);

    out[4*i + 0] = result.v[0];
    out[4*i + 1] = result.v[1];
    out[4*i + 2] = result.v[2];
    out[4*i + 3] = result.v[3];
}

// ─── gen_eq_evals: init ───────────────────────────────────────────────────────
// Sets buf[0..3] = v (QM31 initial value)
__global__ void gkr_gen_eq_evals_init_kernel(uint32_t* buf, const uint32_t* v_ptr) {
    buf[0] = v_ptr[0]; buf[1] = v_ptr[1];
    buf[2] = v_ptr[2]; buf[3] = v_ptr[3];
}

// ─── gen_eq_evals: doubling pass ──────────────────────────────────────────────
// One pass of the "build EQ evaluations" algorithm (stwo/prover/backend/cpu/lookups/gkr.rs):
//   for j in 0..cur_size:
//       tmp      = buf[j] * y_i
//       buf[cur_size + j] = tmp
//       buf[j]            = buf[j] - tmp
//
// cur_size must be a power of 2, cur_size threads launched.
__global__ void gkr_gen_eq_evals_pass_kernel(
    uint32_t* buf,
    const uint32_t* y_i_ptr,  // QM31 challenge [4 u32s]
    uint32_t cur_size
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cur_size) return;

    QM31 y_i = {{y_i_ptr[0], y_i_ptr[1], y_i_ptr[2], y_i_ptr[3]}};

    QM31 val = {{buf[4*j], buf[4*j+1], buf[4*j+2], buf[4*j+3]}};
    QM31 tmp = qm31_mul(val, y_i);

    // buf[cur_size + j] = tmp
    uint32_t hi = cur_size + j;
    buf[4*hi + 0] = tmp.v[0];
    buf[4*hi + 1] = tmp.v[1];
    buf[4*hi + 2] = tmp.v[2];
    buf[4*hi + 3] = tmp.v[3];

    // buf[j] -= tmp
    QM31 sub = qm31_sub(val, tmp);
    buf[4*j + 0] = sub.v[0];
    buf[4*j + 1] = sub.v[1];
    buf[4*j + 2] = sub.v[2];
    buf[4*j + 3] = sub.v[3];
}

// ─── next_layer: GrandProduct ─────────────────────────────────────────────────
// out[i] = in[2i] * in[2i+1]  (QM31 multiplication)
__global__ void gkr_next_layer_grand_product_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out,
    uint32_t half_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_n) return;

    QM31 a = {{in[8*i],   in[8*i+1], in[8*i+2], in[8*i+3]}};
    QM31 b = {{in[8*i+4], in[8*i+5], in[8*i+6], in[8*i+7]}};
    QM31 r = qm31_mul(a, b);

    out[4*i+0] = r.v[0]; out[4*i+1] = r.v[1];
    out[4*i+2] = r.v[2]; out[4*i+3] = r.v[3];
}

// ─── next_layer: LogUpGeneric (QM31 numerators) ───────────────────────────────
// Fraction(num[2i], den[2i]) + Fraction(num[2i+1], den[2i+1])
//   = Fraction(num[2i]*den[2i+1] + num[2i+1]*den[2i], den[2i]*den[2i+1])
__global__ void gkr_next_layer_logup_generic_kernel(
    const uint32_t* __restrict__ in_num,
    const uint32_t* __restrict__ in_den,
    uint32_t* __restrict__ out_num,
    uint32_t* __restrict__ out_den,
    uint32_t half_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_n) return;

    QM31 n0 = {{in_num[8*i],   in_num[8*i+1], in_num[8*i+2], in_num[8*i+3]}};
    QM31 n1 = {{in_num[8*i+4], in_num[8*i+5], in_num[8*i+6], in_num[8*i+7]}};
    QM31 d0 = {{in_den[8*i],   in_den[8*i+1], in_den[8*i+2], in_den[8*i+3]}};
    QM31 d1 = {{in_den[8*i+4], in_den[8*i+5], in_den[8*i+6], in_den[8*i+7]}};

    QM31 rn = qm31_add(qm31_mul(n0, d1), qm31_mul(n1, d0));
    QM31 rd = qm31_mul(d0, d1);

    out_num[4*i+0] = rn.v[0]; out_num[4*i+1] = rn.v[1];
    out_num[4*i+2] = rn.v[2]; out_num[4*i+3] = rn.v[3];
    out_den[4*i+0] = rd.v[0]; out_den[4*i+1] = rd.v[1];
    out_den[4*i+2] = rd.v[2]; out_den[4*i+3] = rd.v[3];
}

// ─── next_layer: LogUpMultiplicities (M31 numerators) ────────────────────────
// Fraction(m31_as_qm31(num[2i]), den[2i]) + Fraction(m31_as_qm31(num[2i+1]), den[2i+1])
__global__ void gkr_next_layer_logup_mult_kernel(
    const uint32_t* __restrict__ in_num,   // M31 (1 u32 per element)
    const uint32_t* __restrict__ in_den,   // QM31 (4 u32 per element)
    uint32_t* __restrict__ out_num,        // QM31 output
    uint32_t* __restrict__ out_den,        // QM31 output
    uint32_t half_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_n) return;

    // n0, n1 as QM31 (embed M31 scalar)
    QM31 n0 = {{in_num[2*i],   0, 0, 0}};
    QM31 n1 = {{in_num[2*i+1], 0, 0, 0}};
    QM31 d0 = {{in_den[8*i],   in_den[8*i+1], in_den[8*i+2], in_den[8*i+3]}};
    QM31 d1 = {{in_den[8*i+4], in_den[8*i+5], in_den[8*i+6], in_den[8*i+7]}};

    // n0*d1 = qm31_mul_m31(d1, n0.v[0])
    QM31 rn = qm31_add(qm31_mul_m31(d1, n0.v[0]), qm31_mul_m31(d0, n1.v[0]));
    QM31 rd = qm31_mul(d0, d1);

    out_num[4*i+0] = rn.v[0]; out_num[4*i+1] = rn.v[1];
    out_num[4*i+2] = rn.v[2]; out_num[4*i+3] = rn.v[3];
    out_den[4*i+0] = rd.v[0]; out_den[4*i+1] = rd.v[1];
    out_den[4*i+2] = rd.v[2]; out_den[4*i+3] = rd.v[3];
}

// ─── next_layer: LogUpSingles (numerators = 1) ───────────────────────────────
// Fraction(1, den[2i]) + Fraction(1, den[2i+1])
//   = Fraction(den[2i] + den[2i+1], den[2i]*den[2i+1])
__global__ void gkr_next_layer_logup_singles_kernel(
    const uint32_t* __restrict__ in_den,
    uint32_t* __restrict__ out_num,
    uint32_t* __restrict__ out_den,
    uint32_t half_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_n) return;

    QM31 d0 = {{in_den[8*i],   in_den[8*i+1], in_den[8*i+2], in_den[8*i+3]}};
    QM31 d1 = {{in_den[8*i+4], in_den[8*i+5], in_den[8*i+6], in_den[8*i+7]}};

    QM31 rn = qm31_add(d0, d1);
    QM31 rd = qm31_mul(d0, d1);

    out_num[4*i+0] = rn.v[0]; out_num[4*i+1] = rn.v[1];
    out_num[4*i+2] = rn.v[2]; out_num[4*i+3] = rn.v[3];
    out_den[4*i+0] = rd.v[0]; out_den[4*i+1] = rd.v[1];
    out_den[4*i+2] = rd.v[2]; out_den[4*i+3] = rd.v[3];
}

// ─── sum_as_poly helpers ──────────────────────────────────────────────────────
// All sum_as_poly kernels produce partial sums: 2 QM31 per block = 8 u32 per block.
// Layout: [eval_at_0 (4 u32), eval_at_2 (4 u32)] per block.
//
// CPU accumulates the partial sums then calls correct_sum_as_poly_in_first_variable.

#define SUM_BLOCK 256

__device__ __forceinline__ void qm31_add_to(volatile uint32_t* acc, QM31 v) {
    // Non-atomic accumulation into shared memory (single-writer per slot)
    acc[0] = m31_add(acc[0], v.v[0]);
    acc[1] = m31_add(acc[1], v.v[1]);
    acc[2] = m31_add(acc[2], v.v[2]);
    acc[3] = m31_add(acc[3], v.v[3]);
}

// Parallel block reduction of two QM31 accumulators stored in shared memory.
// sdata layout: sdata[2 * SUM_BLOCK][4] (interleaved: [eval0_for_tid, eval2_for_tid])
__device__ __forceinline__ void block_reduce_2xqm31(
    volatile uint32_t sdata0[][4],  // eval_at_0 for each thread
    volatile uint32_t sdata2[][4],  // eval_at_2 for each thread
    uint32_t tid
) {
    __syncthreads();
    for (uint32_t s = SUM_BLOCK/2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < 4; c++) {
                sdata0[tid][c] = m31_add(sdata0[tid][c], sdata0[tid + s][c]);
                sdata2[tid][c] = m31_add(sdata2[tid][c], sdata2[tid + s][c]);
            }
        }
        __syncthreads();
    }
}

// ─── sum_as_poly: GrandProduct ────────────────────────────────────────────────
// For each i in 0..n_terms:
//   r0i0 = layer[2i],   r0i1 = layer[2i+1]
//   r1i0 = layer[(n_terms+i)*2], r1i1 = layer[(n_terms+i)*2+1]
//   r2i0 = 2*r1i0 - r0i0, r2i1 = 2*r1i1 - r0i1
//   prod0 = r0i0 * r0i1, prod2 = r2i0 * r2i1
//   eval_at_0 += eq[i] * prod0
//   eval_at_2 += eq[i] * prod2
__global__ void gkr_sum_poly_grand_product_kernel(
    const uint32_t* __restrict__ eq_evals,   // n_terms QM31 (4*n_terms u32)
    const uint32_t* __restrict__ layer,      // 4*n_terms QM31 (16*n_terms u32)
    uint32_t* __restrict__ partial_sums,     // num_blocks * 8 u32
    uint32_t n_terms
) {
    __shared__ uint32_t sdata0[SUM_BLOCK][4];
    __shared__ uint32_t sdata2[SUM_BLOCK][4];

    uint32_t tid = threadIdx.x;
    uint32_t i   = blockIdx.x * SUM_BLOCK + tid;

    // Initialize accumulators to zero
    for (int c = 0; c < 4; c++) { sdata0[tid][c] = 0; sdata2[tid][c] = 0; }

    if (i < n_terms) {
        QM31 eq_i = {{eq_evals[4*i], eq_evals[4*i+1], eq_evals[4*i+2], eq_evals[4*i+3]}};

        uint32_t base0 = 8*i;                // layer[2i] at offset 4*(2i) = 8i
        uint32_t base1 = 8*(n_terms + i);    // layer[(n_terms+i)*2] at offset 8*(n_terms+i)

        QM31 r0i0 = {{layer[base0+0], layer[base0+1], layer[base0+2], layer[base0+3]}};
        QM31 r0i1 = {{layer[base0+4], layer[base0+5], layer[base0+6], layer[base0+7]}};
        QM31 r1i0 = {{layer[base1+0], layer[base1+1], layer[base1+2], layer[base1+3]}};
        QM31 r1i1 = {{layer[base1+4], layer[base1+5], layer[base1+6], layer[base1+7]}};

        // r2 = 2*r1 - r0 (QM31)
        QM31 two_r1i0 = qm31_add(r1i0, r1i0);
        QM31 two_r1i1 = qm31_add(r1i1, r1i1);
        QM31 r2i0 = qm31_sub(two_r1i0, r0i0);
        QM31 r2i1 = qm31_sub(two_r1i1, r0i1);

        QM31 prod0 = qm31_mul(r0i0, r0i1);
        QM31 prod2 = qm31_mul(r2i0, r2i1);

        QM31 term0 = qm31_mul(eq_i, prod0);
        QM31 term2 = qm31_mul(eq_i, prod2);

        for (int c = 0; c < 4; c++) { sdata0[tid][c] = term0.v[c]; sdata2[tid][c] = term2.v[c]; }
    }

    block_reduce_2xqm31(sdata0, sdata2, tid);

    if (tid == 0) {
        uint32_t base = blockIdx.x * 8;
        for (int c = 0; c < 4; c++) {
            partial_sums[base + c]     = sdata0[0][c];
            partial_sums[base + 4 + c] = sdata2[0][c];
        }
    }
}

// ─── sum_as_poly: LogUpGeneric (QM31 numerators) ─────────────────────────────
__global__ void gkr_sum_poly_logup_generic_kernel(
    const uint32_t* __restrict__ eq_evals,
    const uint32_t* __restrict__ in_num,
    const uint32_t* __restrict__ in_den,
    const uint32_t* __restrict__ lambda_ptr,  // QM31 lambda [4 u32]
    uint32_t* __restrict__ partial_sums,
    uint32_t n_terms
) {
    __shared__ uint32_t sdata0[SUM_BLOCK][4];
    __shared__ uint32_t sdata2[SUM_BLOCK][4];

    uint32_t tid = threadIdx.x;
    uint32_t i   = blockIdx.x * SUM_BLOCK + tid;

    for (int c = 0; c < 4; c++) { sdata0[tid][c] = 0; sdata2[tid][c] = 0; }

    if (i < n_terms) {
        QM31 eq_i  = {{eq_evals[4*i], eq_evals[4*i+1], eq_evals[4*i+2], eq_evals[4*i+3]}};
        QM31 lam   = {{lambda_ptr[0], lambda_ptr[1], lambda_ptr[2], lambda_ptr[3]}};

        uint32_t base0 = 8*i;
        uint32_t base1 = 8*(n_terms + i);

        QM31 n_r0i0 = {{in_num[base0+0], in_num[base0+1], in_num[base0+2], in_num[base0+3]}};
        QM31 n_r0i1 = {{in_num[base0+4], in_num[base0+5], in_num[base0+6], in_num[base0+7]}};
        QM31 d_r0i0 = {{in_den[base0+0], in_den[base0+1], in_den[base0+2], in_den[base0+3]}};
        QM31 d_r0i1 = {{in_den[base0+4], in_den[base0+5], in_den[base0+6], in_den[base0+7]}};
        QM31 n_r1i0 = {{in_num[base1+0], in_num[base1+1], in_num[base1+2], in_num[base1+3]}};
        QM31 n_r1i1 = {{in_num[base1+4], in_num[base1+5], in_num[base1+6], in_num[base1+7]}};
        QM31 d_r1i0 = {{in_den[base1+0], in_den[base1+1], in_den[base1+2], in_den[base1+3]}};
        QM31 d_r1i1 = {{in_den[base1+4], in_den[base1+5], in_den[base1+6], in_den[base1+7]}};

        // r2 values
        QM31 n_r2i0 = qm31_sub(qm31_add(n_r1i0, n_r1i0), n_r0i0);
        QM31 n_r2i1 = qm31_sub(qm31_add(n_r1i1, n_r1i1), n_r0i1);
        QM31 d_r2i0 = qm31_sub(qm31_add(d_r1i0, d_r1i0), d_r0i0);
        QM31 d_r2i1 = qm31_sub(qm31_add(d_r1i1, d_r1i1), d_r0i1);

        // Fraction addition at t=0: num0*den1 + num1*den0, den0*den1
        QM31 frac0_num = qm31_add(qm31_mul(n_r0i0, d_r0i1), qm31_mul(n_r0i1, d_r0i0));
        QM31 frac0_den = qm31_mul(d_r0i0, d_r0i1);

        // Fraction addition at t=2
        QM31 frac2_num = qm31_add(qm31_mul(n_r2i0, d_r2i1), qm31_mul(n_r2i1, d_r2i0));
        QM31 frac2_den = qm31_mul(d_r2i0, d_r2i1);

        // combined: frac_num + lambda * frac_den
        QM31 combined0 = qm31_add(frac0_num, qm31_mul(lam, frac0_den));
        QM31 combined2 = qm31_add(frac2_num, qm31_mul(lam, frac2_den));

        QM31 term0 = qm31_mul(eq_i, combined0);
        QM31 term2 = qm31_mul(eq_i, combined2);

        for (int c = 0; c < 4; c++) { sdata0[tid][c] = term0.v[c]; sdata2[tid][c] = term2.v[c]; }
    }

    block_reduce_2xqm31(sdata0, sdata2, tid);

    if (tid == 0) {
        uint32_t base = blockIdx.x * 8;
        for (int c = 0; c < 4; c++) {
            partial_sums[base + c]     = sdata0[0][c];
            partial_sums[base + 4 + c] = sdata2[0][c];
        }
    }
}

// ─── sum_as_poly: LogUpMultiplicities (M31 numerators) ───────────────────────
__global__ void gkr_sum_poly_logup_mult_kernel(
    const uint32_t* __restrict__ eq_evals,
    const uint32_t* __restrict__ in_num,    // M31 (1 u32 per element)
    const uint32_t* __restrict__ in_den,    // QM31 (4 u32 per element)
    const uint32_t* __restrict__ lambda_ptr,
    uint32_t* __restrict__ partial_sums,
    uint32_t n_terms
) {
    __shared__ uint32_t sdata0[SUM_BLOCK][4];
    __shared__ uint32_t sdata2[SUM_BLOCK][4];

    uint32_t tid = threadIdx.x;
    uint32_t i   = blockIdx.x * SUM_BLOCK + tid;

    for (int c = 0; c < 4; c++) { sdata0[tid][c] = 0; sdata2[tid][c] = 0; }

    if (i < n_terms) {
        QM31 eq_i = {{eq_evals[4*i], eq_evals[4*i+1], eq_evals[4*i+2], eq_evals[4*i+3]}};
        QM31 lam  = {{lambda_ptr[0], lambda_ptr[1], lambda_ptr[2], lambda_ptr[3]}};

        uint32_t base0 = 8*i;
        uint32_t base1 = 8*(n_terms + i);

        // Numerators are M31
        uint32_t n_r0i0_m31 = in_num[2*i];
        uint32_t n_r0i1_m31 = in_num[2*i + 1];
        uint32_t n_r1i0_m31 = in_num[2*(n_terms + i)];
        uint32_t n_r1i1_m31 = in_num[2*(n_terms + i) + 1];

        QM31 d_r0i0 = {{in_den[base0+0], in_den[base0+1], in_den[base0+2], in_den[base0+3]}};
        QM31 d_r0i1 = {{in_den[base0+4], in_den[base0+5], in_den[base0+6], in_den[base0+7]}};
        QM31 d_r1i0 = {{in_den[base1+0], in_den[base1+1], in_den[base1+2], in_den[base1+3]}};
        QM31 d_r1i1 = {{in_den[base1+4], in_den[base1+5], in_den[base1+6], in_den[base1+7]}};

        // r2 for M31 numerators: 2*r1 - r0 (mod M31)
        uint32_t n_r2i0_m31 = m31_sub(m31_add(n_r1i0_m31, n_r1i0_m31), n_r0i0_m31);
        uint32_t n_r2i1_m31 = m31_sub(m31_add(n_r1i1_m31, n_r1i1_m31), n_r0i1_m31);
        QM31 d_r2i0 = qm31_sub(qm31_add(d_r1i0, d_r1i0), d_r0i0);
        QM31 d_r2i1 = qm31_sub(qm31_add(d_r1i1, d_r1i1), d_r0i1);

        // Fraction at t=0: n_r0i0_m31 * d_r0i1 + n_r0i1_m31 * d_r0i0, d_r0i0 * d_r0i1
        QM31 frac0_num = qm31_add(
            qm31_mul_m31(d_r0i1, n_r0i0_m31),
            qm31_mul_m31(d_r0i0, n_r0i1_m31)
        );
        QM31 frac0_den = qm31_mul(d_r0i0, d_r0i1);

        // Fraction at t=2
        QM31 frac2_num = qm31_add(
            qm31_mul_m31(d_r2i1, n_r2i0_m31),
            qm31_mul_m31(d_r2i0, n_r2i1_m31)
        );
        QM31 frac2_den = qm31_mul(d_r2i0, d_r2i1);

        QM31 combined0 = qm31_add(frac0_num, qm31_mul(lam, frac0_den));
        QM31 combined2 = qm31_add(frac2_num, qm31_mul(lam, frac2_den));

        QM31 term0 = qm31_mul(eq_i, combined0);
        QM31 term2 = qm31_mul(eq_i, combined2);

        for (int c = 0; c < 4; c++) { sdata0[tid][c] = term0.v[c]; sdata2[tid][c] = term2.v[c]; }
    }

    block_reduce_2xqm31(sdata0, sdata2, tid);

    if (tid == 0) {
        uint32_t base = blockIdx.x * 8;
        for (int c = 0; c < 4; c++) {
            partial_sums[base + c]     = sdata0[0][c];
            partial_sums[base + 4 + c] = sdata2[0][c];
        }
    }
}

// ─── sum_as_poly: LogUpSingles (numerators = 1) ───────────────────────────────
__global__ void gkr_sum_poly_logup_singles_kernel(
    const uint32_t* __restrict__ eq_evals,
    const uint32_t* __restrict__ in_den,
    const uint32_t* __restrict__ lambda_ptr,
    uint32_t* __restrict__ partial_sums,
    uint32_t n_terms
) {
    __shared__ uint32_t sdata0[SUM_BLOCK][4];
    __shared__ uint32_t sdata2[SUM_BLOCK][4];

    uint32_t tid = threadIdx.x;
    uint32_t i   = blockIdx.x * SUM_BLOCK + tid;

    for (int c = 0; c < 4; c++) { sdata0[tid][c] = 0; sdata2[tid][c] = 0; }

    if (i < n_terms) {
        QM31 eq_i = {{eq_evals[4*i], eq_evals[4*i+1], eq_evals[4*i+2], eq_evals[4*i+3]}};
        QM31 lam  = {{lambda_ptr[0], lambda_ptr[1], lambda_ptr[2], lambda_ptr[3]}};

        uint32_t base0 = 8*i;
        uint32_t base1 = 8*(n_terms + i);

        QM31 d_r0i0 = {{in_den[base0+0], in_den[base0+1], in_den[base0+2], in_den[base0+3]}};
        QM31 d_r0i1 = {{in_den[base0+4], in_den[base0+5], in_den[base0+6], in_den[base0+7]}};
        QM31 d_r1i0 = {{in_den[base1+0], in_den[base1+1], in_den[base1+2], in_den[base1+3]}};
        QM31 d_r1i1 = {{in_den[base1+4], in_den[base1+5], in_den[base1+6], in_den[base1+7]}};

        QM31 d_r2i0 = qm31_sub(qm31_add(d_r1i0, d_r1i0), d_r0i0);
        QM31 d_r2i1 = qm31_sub(qm31_add(d_r1i1, d_r1i1), d_r0i1);

        // Numerator = d0 + d1 (since both fractions have numerator 1)
        QM31 frac0_num = qm31_add(d_r0i0, d_r0i1);
        QM31 frac0_den = qm31_mul(d_r0i0, d_r0i1);
        QM31 frac2_num = qm31_add(d_r2i0, d_r2i1);
        QM31 frac2_den = qm31_mul(d_r2i0, d_r2i1);

        QM31 combined0 = qm31_add(frac0_num, qm31_mul(lam, frac0_den));
        QM31 combined2 = qm31_add(frac2_num, qm31_mul(lam, frac2_den));

        QM31 term0 = qm31_mul(eq_i, combined0);
        QM31 term2 = qm31_mul(eq_i, combined2);

        for (int c = 0; c < 4; c++) { sdata0[tid][c] = term0.v[c]; sdata2[tid][c] = term2.v[c]; }
    }

    block_reduce_2xqm31(sdata0, sdata2, tid);

    if (tid == 0) {
        uint32_t base = blockIdx.x * 8;
        for (int c = 0; c < 4; c++) {
            partial_sums[base + c]     = sdata0[0][c];
            partial_sums[base + 4 + c] = sdata2[0][c];
        }
    }
}

// ─── C entry points ───────────────────────────────────────────────────────────
extern "C" {

void cuda_gkr_fix_first_variable_base(
    const uint32_t* in, uint32_t* out, const uint32_t* r_ptr, uint32_t n
) {
    uint32_t half_n = n / 2;
    uint32_t blocks = (half_n + 255) / 256;
    gkr_fix_first_variable_base_kernel<<<blocks, 256>>>(in, out, r_ptr, half_n);
}

void cuda_gkr_fix_first_variable_secure(
    const uint32_t* in, uint32_t* out, const uint32_t* r_ptr, uint32_t n
) {
    uint32_t half_n = n / 2;
    uint32_t blocks = (half_n + 255) / 256;
    gkr_fix_first_variable_secure_kernel<<<blocks, 256>>>(in, out, r_ptr, half_n);
}

void cuda_gkr_gen_eq_evals_init(uint32_t* buf, const uint32_t* v_ptr) {
    gkr_gen_eq_evals_init_kernel<<<1, 1>>>(buf, v_ptr);
}

void cuda_gkr_gen_eq_evals_pass(uint32_t* buf, const uint32_t* y_i_ptr, uint32_t cur_size) {
    uint32_t blocks = (cur_size + 255) / 256;
    gkr_gen_eq_evals_pass_kernel<<<blocks, 256>>>(buf, y_i_ptr, cur_size);
}

void cuda_gkr_next_layer_grand_product(
    const uint32_t* in, uint32_t* out, uint32_t n
) {
    uint32_t half_n = n / 2;
    uint32_t blocks = (half_n + 255) / 256;
    gkr_next_layer_grand_product_kernel<<<blocks, 256>>>(in, out, half_n);
}

void cuda_gkr_next_layer_logup_generic(
    const uint32_t* in_num, const uint32_t* in_den,
    uint32_t* out_num, uint32_t* out_den, uint32_t n
) {
    uint32_t half_n = n / 2;
    uint32_t blocks = (half_n + 255) / 256;
    gkr_next_layer_logup_generic_kernel<<<blocks, 256>>>(in_num, in_den, out_num, out_den, half_n);
}

void cuda_gkr_next_layer_logup_mult(
    const uint32_t* in_num, const uint32_t* in_den,
    uint32_t* out_num, uint32_t* out_den, uint32_t n
) {
    uint32_t half_n = n / 2;
    uint32_t blocks = (half_n + 255) / 256;
    gkr_next_layer_logup_mult_kernel<<<blocks, 256>>>(in_num, in_den, out_num, out_den, half_n);
}

void cuda_gkr_next_layer_logup_singles(
    const uint32_t* in_den, uint32_t* out_num, uint32_t* out_den, uint32_t n
) {
    uint32_t half_n = n / 2;
    uint32_t blocks = (half_n + 255) / 256;
    gkr_next_layer_logup_singles_kernel<<<blocks, 256>>>(in_den, out_num, out_den, half_n);
}

// Returns the number of partial sum blocks (each block produces 8 u32s).
uint32_t cuda_gkr_sum_poly_grand_product(
    const uint32_t* eq_evals, const uint32_t* layer,
    uint32_t* partial_sums, uint32_t n_terms
) {
    uint32_t n_blocks = (n_terms + SUM_BLOCK - 1) / SUM_BLOCK;
    gkr_sum_poly_grand_product_kernel<<<n_blocks, SUM_BLOCK>>>(
        eq_evals, layer, partial_sums, n_terms);
    return n_blocks;
}

uint32_t cuda_gkr_sum_poly_logup_generic(
    const uint32_t* eq_evals, const uint32_t* in_num, const uint32_t* in_den,
    const uint32_t* lambda_ptr, uint32_t* partial_sums, uint32_t n_terms
) {
    uint32_t n_blocks = (n_terms + SUM_BLOCK - 1) / SUM_BLOCK;
    gkr_sum_poly_logup_generic_kernel<<<n_blocks, SUM_BLOCK>>>(
        eq_evals, in_num, in_den, lambda_ptr, partial_sums, n_terms);
    return n_blocks;
}

uint32_t cuda_gkr_sum_poly_logup_mult(
    const uint32_t* eq_evals, const uint32_t* in_num, const uint32_t* in_den,
    const uint32_t* lambda_ptr, uint32_t* partial_sums, uint32_t n_terms
) {
    uint32_t n_blocks = (n_terms + SUM_BLOCK - 1) / SUM_BLOCK;
    gkr_sum_poly_logup_mult_kernel<<<n_blocks, SUM_BLOCK>>>(
        eq_evals, in_num, in_den, lambda_ptr, partial_sums, n_terms);
    return n_blocks;
}

uint32_t cuda_gkr_sum_poly_logup_singles(
    const uint32_t* eq_evals, const uint32_t* in_den,
    const uint32_t* lambda_ptr, uint32_t* partial_sums, uint32_t n_terms
) {
    uint32_t n_blocks = (n_terms + SUM_BLOCK - 1) / SUM_BLOCK;
    gkr_sum_poly_logup_singles_kernel<<<n_blocks, SUM_BLOCK>>>(
        eq_evals, in_den, lambda_ptr, partial_sums, n_terms);
    return n_blocks;
}

} // extern "C"
