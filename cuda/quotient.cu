#include "include/qm31.cuh"

// ═══════════════════════════════════════════════════════════════════════════
//  accumulate_numerators kernel
// ═══════════════════════════════════════════════════════════════════════════
//
// For a single sample batch, compute per-row:
//   result[row] = sum_i (c_i * f_i[row] - b_i)
// where:
//   - (b_i, c_i) are QM31 line coefficients for the i-th column in this batch
//   - f_i[row] is the M31 column value at `row`
//
// The result is stored as SoA QM31 (4 separate M31 arrays).
// All columns have the same size.
//
// Parameters:
//   col_ptrs:    device array of pointers to M31 column data
//   col_indices: which columns from col_ptrs to use (length = n_batch_cols)
//   b_coeffs:    QM31 b coefficients [n_batch_cols * 4] (flat M31 limbs)
//   c_coeffs:    QM31 c coefficients [n_batch_cols * 4] (flat M31 limbs)
//   n_batch_cols: number of columns in this batch
//   n_rows:      number of rows
//   out0..3:     output SoA QM31 accumulator

__global__ void accumulate_numerators_kernel(
    const uint32_t* const* __restrict__ col_ptrs,
    const uint32_t* __restrict__ col_indices,
    const uint32_t* __restrict__ b_coeffs,   // [n_batch_cols * 4]
    const uint32_t* __restrict__ c_coeffs,   // [n_batch_cols * 4]
    uint32_t n_batch_cols,
    uint32_t n_rows,
    uint32_t* __restrict__ out0,
    uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2,
    uint32_t* __restrict__ out3
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    QM31 acc = qm31_zero();
    for (uint32_t i = 0; i < n_batch_cols; i++) {
        uint32_t col_idx = col_indices[i];
        uint32_t f_val = col_ptrs[col_idx][row];

        // Load c coefficient (QM31)
        QM31 c;
        c.v[0] = c_coeffs[i * 4 + 0];
        c.v[1] = c_coeffs[i * 4 + 1];
        c.v[2] = c_coeffs[i * 4 + 2];
        c.v[3] = c_coeffs[i * 4 + 3];

        // Load b coefficient (QM31)
        QM31 b;
        b.v[0] = b_coeffs[i * 4 + 0];
        b.v[1] = b_coeffs[i * 4 + 1];
        b.v[2] = b_coeffs[i * 4 + 2];
        b.v[3] = b_coeffs[i * 4 + 3];

        // c * f_val - b
        QM31 term = qm31_sub(qm31_mul_m31(c, f_val), b);
        acc = qm31_add(acc, term);
    }

    out0[row] = acc.v[0];
    out1[row] = acc.v[1];
    out2[row] = acc.v[2];
    out3[row] = acc.v[3];
}

// ═══════════════════════════════════════════════════════════════════════════
//  compute_quotients_and_combine kernel
// ═══════════════════════════════════════════════════════════════════════════
//
// For each row in the lifting domain:
//   domain_point = domain.at(bit_reverse_index(row, log_size))
//   quotient[row] = sum_j (numer_j[lifted_idx] - a_acc_j * y) * den_inv_j
//
// where den_inv_j is computed from (sample_point_j, domain_point).
//
// Domain points on the circle: initial_x, initial_y define the canonic coset,
// and we bit-reverse-index into it.

// Helper: bit-reverse an index
__device__ __forceinline__ uint32_t bit_reverse(uint32_t val, uint32_t log_n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log_n; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

// Circle domain point computation:
// CanonicCoset::new(log_size).circle_domain() generates points on the circle.
// The i-th point (natural order) of circle_domain is:
//   G^(2*i+1) where G is the circle group generator of order 2^(log_size+1)
// For bit-reversed order, we bit-reverse the index first.
//
// But the caller precomputes and passes the domain x,y coordinates.

__global__ void compute_quotients_combine_kernel(
    // Per-accumulation data (flattened, n_accs accumulations)
    const uint32_t* __restrict__ sample_points_x,  // [n_accs * 4] QM31
    const uint32_t* __restrict__ sample_points_y,  // [n_accs * 4] QM31
    const uint32_t* __restrict__ first_linear_acc,  // [n_accs * 4] QM31
    // Partial numerators for each acc: SoA, each of size = (1 << acc_log_sizes[j])
    const uint32_t* const* __restrict__ numer_ptrs0,  // [n_accs] pointers
    const uint32_t* const* __restrict__ numer_ptrs1,
    const uint32_t* const* __restrict__ numer_ptrs2,
    const uint32_t* const* __restrict__ numer_ptrs3,
    const uint32_t* __restrict__ acc_log_sizes,  // [n_accs] log2 of each acc's size
    uint32_t n_accs,
    // Domain info
    const uint32_t* __restrict__ domain_xs,  // [n_rows] M31 x-coordinates
    const uint32_t* __restrict__ domain_ys,  // [n_rows] M31 y-coordinates
    uint32_t lifting_log_size,
    uint32_t n_rows,
    // Output SoA QM31
    uint32_t* __restrict__ out0,
    uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2,
    uint32_t* __restrict__ out3
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    uint32_t dx = domain_xs[row];
    uint32_t dy = domain_ys[row];

    QM31 quotient = qm31_zero();

    for (uint32_t j = 0; j < n_accs; j++) {
        // Load sample point
        QM31 sp_x, sp_y;
        sp_x.v[0] = sample_points_x[j * 4 + 0];
        sp_x.v[1] = sample_points_x[j * 4 + 1];
        sp_x.v[2] = sample_points_x[j * 4 + 2];
        sp_x.v[3] = sample_points_x[j * 4 + 3];
        sp_y.v[0] = sample_points_y[j * 4 + 0];
        sp_y.v[1] = sample_points_y[j * 4 + 1];
        sp_y.v[2] = sample_points_y[j * 4 + 2];
        sp_y.v[3] = sample_points_y[j * 4 + 3];

        // Denominator: (Re(px) - dx)*Im(py) - (Re(py) - dy)*Im(px)
        // where Re = CM31.a (.v[0]) and Im = CM31.b (.v[1])
        // sp_x = CM31(v[0], v[1]) + CM31(v[2], v[3])*u
        // Re(sp_x) = CM31(v[0], v[1]), Im(sp_x) = CM31(v[2], v[3])
        CM31 prx = {sp_x.v[0], sp_x.v[1]};  // Re(sp_x)
        CM31 pry = {sp_y.v[0], sp_y.v[1]};  // Re(sp_y)
        CM31 pix = {sp_x.v[2], sp_x.v[3]};  // Im(sp_x)
        CM31 piy = {sp_y.v[2], sp_y.v[3]};  // Im(sp_y)

        // (prx - dx) * piy - (pry - dy) * pix
        CM31 term1 = cm31_mul(cm31_sub(prx, {dx, 0}), piy);
        CM31 term2 = cm31_mul(cm31_sub(pry, {dy, 0}), pix);
        CM31 denom = cm31_sub(term1, term2);
        CM31 den_inv = cm31_inv(denom);

        // Lifted index
        uint32_t acc_log_sz = acc_log_sizes[j];
        uint32_t log_ratio = lifting_log_size - acc_log_sz;
        uint32_t lifted_idx = (row >> (log_ratio + 1) << 1) + (row & 1);

        // Load partial numerator at lifted_idx
        QM31 partial_numer;
        partial_numer.v[0] = numer_ptrs0[j][lifted_idx];
        partial_numer.v[1] = numer_ptrs1[j][lifted_idx];
        partial_numer.v[2] = numer_ptrs2[j][lifted_idx];
        partial_numer.v[3] = numer_ptrs3[j][lifted_idx];

        // Load first_linear_term_acc
        QM31 a_acc;
        a_acc.v[0] = first_linear_acc[j * 4 + 0];
        a_acc.v[1] = first_linear_acc[j * 4 + 1];
        a_acc.v[2] = first_linear_acc[j * 4 + 2];
        a_acc.v[3] = first_linear_acc[j * 4 + 3];

        // full_numerator = partial_numer - a_acc * domain_point.y
        // domain_point.y is M31
        QM31 full_numer = qm31_sub(partial_numer, qm31_mul_m31(a_acc, dy));

        // Multiply by den_inv (CM31): QM31 * CM31
        // (a + bu) * c = ac + bc*u
        CM31 fa = qm31_a(full_numer);
        CM31 fb = qm31_b(full_numer);
        CM31 ra = cm31_mul(fa, den_inv);
        CM31 rb = cm31_mul(fb, den_inv);
        QM31 term_result = qm31_from(ra, rb);

        quotient = qm31_add(quotient, term_result);
    }

    out0[row] = quotient.v[0];
    out1[row] = quotient.v[1];
    out2[row] = quotient.v[2];
    out3[row] = quotient.v[3];
}

extern "C" {

void cuda_accumulate_numerators(
    const uint32_t* const* col_ptrs,
    const uint32_t* col_indices,
    const uint32_t* b_coeffs,
    const uint32_t* c_coeffs,
    uint32_t n_batch_cols,
    uint32_t n_rows,
    uint32_t* out0,
    uint32_t* out1,
    uint32_t* out2,
    uint32_t* out3
) {
    if (n_rows == 0) return;
    uint32_t threads = 256;
    uint32_t blocks = (n_rows + threads - 1) / threads;
    accumulate_numerators_kernel<<<blocks, threads>>>(
        col_ptrs, col_indices, b_coeffs, c_coeffs,
        n_batch_cols, n_rows, out0, out1, out2, out3
    );
}

void cuda_compute_quotients_combine(
    const uint32_t* sample_points_x,
    const uint32_t* sample_points_y,
    const uint32_t* first_linear_acc,
    const uint32_t* const* numer_ptrs0,
    const uint32_t* const* numer_ptrs1,
    const uint32_t* const* numer_ptrs2,
    const uint32_t* const* numer_ptrs3,
    const uint32_t* acc_log_sizes,
    uint32_t n_accs,
    const uint32_t* domain_xs,
    const uint32_t* domain_ys,
    uint32_t lifting_log_size,
    uint32_t n_rows,
    uint32_t* out0,
    uint32_t* out1,
    uint32_t* out2,
    uint32_t* out3
) {
    if (n_rows == 0) return;
    uint32_t threads = 256;
    uint32_t blocks = (n_rows + threads - 1) / threads;
    compute_quotients_combine_kernel<<<blocks, threads>>>(
        sample_points_x, sample_points_y, first_linear_acc,
        numer_ptrs0, numer_ptrs1, numer_ptrs2, numer_ptrs3,
        acc_log_sizes, n_accs,
        domain_xs, domain_ys, lifting_log_size, n_rows,
        out0, out1, out2, out3
    );
}

} // extern "C"
