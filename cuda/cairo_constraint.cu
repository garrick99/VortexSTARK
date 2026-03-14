// GPU constraint evaluation for Cairo VM AIR.
//
// 27 trace columns, 20 transition constraints.
// Each thread evaluates all constraints for one row, combining with
// alpha coefficients into a single QM31 quotient value.
//
// Column layout:
//  0:  pc          1:  ap          2:  fp
//  3:  inst_lo     4:  inst_hi
//  5-19: flags (15 binary: dst_reg, op0_reg, op1_imm, op1_fp, op1_ap,
//         res_add, res_mul, pc_jump_abs, pc_jump_rel, pc_jnz,
//         ap_add, ap_add1, opcode_call, opcode_ret, opcode_assert)
// 20: dst_addr   21: dst    22: op0_addr   23: op0
// 24: op1_addr   25: op1    26: res

#include "include/qm31.cuh"

#define CAIRO_N_COLS 27
#define CAIRO_N_FLAGS 15
#define CAIRO_N_CONSTRAINTS 20
#define COL_PC 0
#define COL_AP 1
#define COL_FP 2
#define COL_DST 21
#define COL_OP0 23
#define COL_OP1 25
#define COL_RES 26
#define COL_FLAGS 5

__global__ void cairo_quotient_kernel(
    const uint32_t* const* __restrict__ trace_cols,  // [27] column pointers
    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2, uint32_t* __restrict__ out3,
    const uint32_t* __restrict__ alpha_coeffs,  // [N_CONSTRAINTS * 4] QM31 coefficients
    uint32_t n  // eval domain size
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t next_i = (i + 1) % n;

    // Load current row values
    uint32_t pc  = trace_cols[COL_PC][i];
    uint32_t ap  = trace_cols[COL_AP][i];
    uint32_t fp  = trace_cols[COL_FP][i];
    uint32_t dst = trace_cols[COL_DST][i];
    uint32_t op0 = trace_cols[COL_OP0][i];
    uint32_t op1 = trace_cols[COL_OP1][i];
    uint32_t res = trace_cols[COL_RES][i];

    // Load next row registers
    uint32_t next_pc = trace_cols[COL_PC][next_i];
    uint32_t next_ap = trace_cols[COL_AP][next_i];
    uint32_t next_fp = trace_cols[COL_FP][next_i];

    // Load flags
    uint32_t flags[CAIRO_N_FLAGS];
    for (int j = 0; j < CAIRO_N_FLAGS; j++) {
        flags[j] = trace_cols[COL_FLAGS + j][i];
    }

    // Flag aliases
    uint32_t f_op1_imm     = flags[2];
    uint32_t f_res_add     = flags[5];
    uint32_t f_res_mul     = flags[6];
    uint32_t f_pc_jump_abs = flags[7];
    uint32_t f_pc_jump_rel = flags[8];
    uint32_t f_pc_jnz      = flags[9];
    uint32_t f_ap_add      = flags[10];
    uint32_t f_ap_add1     = flags[11];
    uint32_t f_call        = flags[12];
    uint32_t f_ret         = flags[13];
    uint32_t f_assert      = flags[14];

    // Accumulate constraints
    QM31 quotient = {{0, 0, 0, 0}};
    int ci = 0;  // constraint index

    // --- Constraint 0-14: Flag binary (flag * (1 - flag) = 0) ---
    for (int j = 0; j < CAIRO_N_FLAGS; j++) {
        uint32_t c = m31_mul(flags[j], m31_sub(1, flags[j]));
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1],
                       alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c));
        ci++;
    }

    // --- Constraint 15: Result computation ---
    // (1 - pc_jnz) * (res - expected_res) = 0
    // expected_res = (1 - res_add - res_mul) * op1 + res_add * (op0 + op1) + res_mul * (op0 * op1)
    {
        uint32_t one = 1;
        uint32_t coeff_default = m31_sub(m31_sub(one, f_res_add), f_res_mul);
        uint32_t expected = m31_add(
            m31_add(
                m31_mul(coeff_default, op1),
                m31_mul(f_res_add, m31_add(op0, op1))
            ),
            m31_mul(f_res_mul, m31_mul(op0, op1))
        );
        uint32_t c = m31_mul(m31_sub(one, f_pc_jnz), m31_sub(res, expected));
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1],
                       alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c));
        ci++;
    }

    // --- Constraint 16: PC update ---
    // Split into non-jnz and jnz parts:
    // (1-jnz) * (next_pc - regular - abs - rel) + jnz * dst * (next_pc - pc - op1) = 0
    {
        uint32_t one = 1;
        uint32_t inst_size = m31_add(one, f_op1_imm);
        uint32_t pc_default = m31_add(pc, inst_size);
        uint32_t not_jump = m31_sub(m31_sub(m31_sub(one, f_pc_jump_abs), f_pc_jump_rel), f_pc_jnz);
        uint32_t regular_part = m31_mul(not_jump, pc_default);
        uint32_t abs_part = m31_mul(f_pc_jump_abs, res);
        uint32_t rel_part = m31_mul(f_pc_jump_rel, m31_add(pc, res));

        uint32_t non_jnz = m31_mul(
            m31_sub(one, f_pc_jnz),
            m31_sub(next_pc, m31_add(m31_add(regular_part, abs_part), rel_part))
        );
        uint32_t jnz_part = m31_mul(
            f_pc_jnz,
            m31_mul(dst, m31_sub(next_pc, m31_add(pc, op1)))
        );
        uint32_t c = m31_add(non_jnz, jnz_part);
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1],
                       alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c));
        ci++;
    }

    // --- Constraint 17: AP update ---
    // next_ap - (ap + ap_add * res + ap_add1 + call * 2) = 0
    {
        uint32_t expected_ap = m31_add(
            m31_add(
                m31_add(ap, m31_mul(f_ap_add, res)),
                f_ap_add1
            ),
            m31_mul(f_call, 2)
        );
        uint32_t c = m31_sub(next_ap, expected_ap);
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1],
                       alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c));
        ci++;
    }

    // --- Constraint 18: FP update ---
    // next_fp - ((1 - call - ret) * fp + call * (ap + 2) + ret * dst) = 0
    {
        uint32_t one = 1;
        uint32_t keep = m31_sub(m31_sub(one, f_call), f_ret);
        uint32_t expected_fp = m31_add(
            m31_add(
                m31_mul(keep, fp),
                m31_mul(f_call, m31_add(ap, 2))
            ),
            m31_mul(f_ret, dst)
        );
        uint32_t c = m31_sub(next_fp, expected_fp);
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1],
                       alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c));
        ci++;
    }

    // --- Constraint 19: Assert_eq (dst = res when assert flag set) ---
    {
        uint32_t c = m31_mul(f_assert, m31_sub(dst, res));
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1],
                       alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c));
        ci++;
    }

    out0[i] = quotient.v[0];
    out1[i] = quotient.v[1];
    out2[i] = quotient.v[2];
    out3[i] = quotient.v[3];
}

// Chunked variant for streaming
__global__ void cairo_quotient_chunk_kernel(
    const uint32_t* const* __restrict__ trace_cols,
    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2, uint32_t* __restrict__ out3,
    const uint32_t* __restrict__ alpha_coeffs,
    uint32_t offset, uint32_t chunk_n, uint32_t global_n
) {
    uint32_t local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i >= chunk_n) return;

    uint32_t i = offset + local_i;
    uint32_t next_i = (i + 1) % global_n;

    // Same constraint logic as above, reading from global indices
    uint32_t pc  = trace_cols[COL_PC][i];
    uint32_t ap  = trace_cols[COL_AP][i];
    uint32_t fp  = trace_cols[COL_FP][i];
    uint32_t dst = trace_cols[COL_DST][i];
    uint32_t op0 = trace_cols[COL_OP0][i];
    uint32_t op1 = trace_cols[COL_OP1][i];
    uint32_t res = trace_cols[COL_RES][i];

    uint32_t next_pc = trace_cols[COL_PC][next_i];
    uint32_t next_ap = trace_cols[COL_AP][next_i];
    uint32_t next_fp = trace_cols[COL_FP][next_i];

    uint32_t flags[CAIRO_N_FLAGS];
    for (int j = 0; j < CAIRO_N_FLAGS; j++) {
        flags[j] = trace_cols[COL_FLAGS + j][i];
    }

    uint32_t f_op1_imm = flags[2], f_res_add = flags[5], f_res_mul = flags[6];
    uint32_t f_pc_jump_abs = flags[7], f_pc_jump_rel = flags[8], f_pc_jnz = flags[9];
    uint32_t f_ap_add = flags[10], f_ap_add1 = flags[11];
    uint32_t f_call = flags[12], f_ret = flags[13], f_assert = flags[14];

    QM31 quotient = {{0, 0, 0, 0}};
    int ci = 0;

    for (int j = 0; j < CAIRO_N_FLAGS; j++) {
        uint32_t c = m31_mul(flags[j], m31_sub(1, flags[j]));
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c));
        ci++;
    }

    {
        uint32_t coeff = m31_sub(m31_sub(1, f_res_add), f_res_mul);
        uint32_t expected = m31_add(m31_add(m31_mul(coeff, op1), m31_mul(f_res_add, m31_add(op0, op1))), m31_mul(f_res_mul, m31_mul(op0, op1)));
        uint32_t c = m31_mul(m31_sub(1, f_pc_jnz), m31_sub(res, expected));
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }
    {
        uint32_t inst_size = m31_add(1, f_op1_imm);
        uint32_t not_jump = m31_sub(m31_sub(m31_sub(1, f_pc_jump_abs), f_pc_jump_rel), f_pc_jnz);
        uint32_t regular = m31_mul(not_jump, m31_add(pc, inst_size));
        uint32_t expected_pc = m31_add(m31_add(regular, m31_mul(f_pc_jump_abs, res)), m31_mul(f_pc_jump_rel, m31_add(pc, res)));
        uint32_t non_jnz = m31_mul(m31_sub(1, f_pc_jnz), m31_sub(next_pc, expected_pc));
        uint32_t jnz = m31_mul(f_pc_jnz, m31_mul(dst, m31_sub(next_pc, m31_add(pc, op1))));
        uint32_t c = m31_add(non_jnz, jnz);
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }
    {
        uint32_t expected_ap = m31_add(m31_add(m31_add(ap, m31_mul(f_ap_add, res)), f_ap_add1), m31_mul(f_call, 2));
        uint32_t c = m31_sub(next_ap, expected_ap);
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }
    {
        uint32_t keep = m31_sub(m31_sub(1, f_call), f_ret);
        uint32_t expected_fp = m31_add(m31_add(m31_mul(keep, fp), m31_mul(f_call, m31_add(ap, 2))), m31_mul(f_ret, dst));
        uint32_t c = m31_sub(next_fp, expected_fp);
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }
    {
        uint32_t c = m31_mul(f_assert, m31_sub(dst, res));
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }

    out0[local_i] = quotient.v[0];
    out1[local_i] = quotient.v[1];
    out2[local_i] = quotient.v[2];
    out3[local_i] = quotient.v[3];
}

extern "C" {

void cuda_cairo_quotient(
    const uint32_t* const* trace_cols,
    uint32_t* out0, uint32_t* out1, uint32_t* out2, uint32_t* out3,
    const uint32_t* alpha_coeffs,
    uint32_t n
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    cairo_quotient_kernel<<<blocks, threads>>>(
        trace_cols, out0, out1, out2, out3, alpha_coeffs, n
    );
}

void cuda_cairo_quotient_chunk(
    const uint32_t* const* trace_cols,
    uint32_t* out0, uint32_t* out1, uint32_t* out2, uint32_t* out3,
    const uint32_t* alpha_coeffs,
    uint32_t offset, uint32_t chunk_n, uint32_t global_n
) {
    uint32_t threads = 256;
    uint32_t blocks = (chunk_n + threads - 1) / threads;
    cairo_quotient_chunk_kernel<<<blocks, threads>>>(
        trace_cols, out0, out1, out2, out3, alpha_coeffs,
        offset, chunk_n, global_n
    );
}

} // extern "C"
