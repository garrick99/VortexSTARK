// GPU constraint evaluation for Cairo VM AIR.
//
// 31 trace columns, 31 transition constraints.
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

#define CAIRO_N_COLS 34
#define CAIRO_N_FLAGS 15
#define CAIRO_N_CONSTRAINTS 35
#define COL_INST_LO 3
#define COL_INST_HI 4
#define COL_PC 0
#define COL_AP 1
#define COL_FP 2
#define COL_DST 21
#define COL_OP0 23
#define COL_OP1 25
#define COL_RES 26
#define COL_FLAGS 5
#define COL_OFF0 27
#define COL_OFF1 28
#define COL_OFF2 29
#define COL_DST_INV 30
#define COL_DST_ADDR 20
#define COL_OP0_ADDR 22
#define COL_OP1_ADDR 24
// Dict linkage columns (GAP-1 closure)
#define COL_DICT_KEY    31
#define COL_DICT_NEW    32
#define COL_DICT_ACTIVE 33

// ── Vanishing polynomial helpers ───────────────────────────────────────────

// Bit-reverse an index of log_n bits.
__device__ __forceinline__ uint32_t cairo_bit_reverse(uint32_t val, uint32_t log_n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log_n; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

// For trace half_coset(log_n), the vanishing polynomial is:
//   Z_H(x) = f_{log_n}(x) + 1,  where f_0(x) = x, f_{i+1}(x) = 2x^2 - 1.
// Z_H(x) = 0 iff x is the x-coordinate of a point in the trace domain.
__device__ __forceinline__ uint32_t vanishing_poly(uint32_t x, uint32_t log_n) {
    uint32_t v = x;
    for (uint32_t k = 0; k < log_n; k++) {
        // v = 2v^2 - 1  (circle group doubling x-coordinate)
        v = m31_sub(m31_add(m31_mul(v, v), m31_mul(v, v)), 1u);
    }
    return m31_add(v, 1u);  // Z_H = f_{log_n}(x) + 1
}

// Compute 1/Z_H for every NTT position in the eval domain.
// The eval domain is half_coset(log_eval): initial*(step^j) at natural index j.
// NTT output index i corresponds to natural index j = bit_reverse(i, log_eval).
//
// Per-thread: compute step^j via square-and-multiply, multiply by initial,
// extract x, apply Z_H formula, compute Fermat inverse.
__global__ void compute_vanishing_inv_kernel(
    uint32_t initial_x, uint32_t initial_y,
    uint32_t step_x,    uint32_t step_y,
    uint32_t* __restrict__ out_vh_inv,
    uint32_t log_eval,  // log2(eval domain size)
    uint32_t log_n      // log2(trace domain size) — doubling count for Z_H
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t eval_n = 1u << log_eval;
    if (i >= eval_n) return;

    // NTT index → natural coset index
    uint32_t j = cairo_bit_reverse(i, log_eval);

    // Compute step^j in the circle group via square-and-multiply
    // Identity = (1, 0)
    uint32_t rx = 1u, ry = 0u;
    uint32_t bx = step_x, by = step_y;
    uint32_t k = j;
    for (uint32_t bit = 0; bit < log_eval; bit++) {
        if (k & 1u) {
            // r = r * b  (circle multiplication)
            uint32_t nx = m31_sub(m31_mul(rx, bx), m31_mul(ry, by));
            uint32_t ny = m31_add(m31_mul(rx, by), m31_mul(ry, bx));
            rx = nx; ry = ny;
        }
        // b = b^2  (circle doubling: x' = 2x^2-1, y' = 2xy)
        uint32_t nx = m31_sub(m31_add(m31_mul(bx, bx), m31_mul(bx, bx)), 1u);
        uint32_t ny = m31_add(m31_mul(bx, by), m31_mul(bx, by));
        bx = nx; by = ny;
        k >>= 1;
    }

    // point = initial * step^j
    uint32_t px = m31_sub(m31_mul(initial_x, rx), m31_mul(initial_y, ry));
    // (y-coordinate not needed)

    // Z_H(px) and its inverse
    uint32_t zh = vanishing_poly(px, log_n);
    out_vh_inv[i] = m31_inv(zh);
}

// Helper: compute LogUp step delta for one row.
// access0 uses extended denom: z - (pc + alpha*inst_lo + alpha_sq*inst_hi)
// accesses 1-3 use simple denom: z - (addr + alpha*val)
__device__ __forceinline__ QM31 logup_step_delta(
    uint32_t pc, uint32_t inst_lo, uint32_t inst_hi,
    uint32_t dst_addr, uint32_t dst,
    uint32_t op0_addr, uint32_t op0,
    uint32_t op1_addr, uint32_t op1,
    QM31 z_mem, QM31 alpha_mem, QM31 alpha_mem_sq
) {
    // Access 0: instruction fetch (extended denom with inst_hi)
    QM31 e0 = qm31_add(qm31_from_m31(pc),
               qm31_add(qm31_mul(alpha_mem, qm31_from_m31(inst_lo)),
                        qm31_mul(alpha_mem_sq, qm31_from_m31(inst_hi))));
    QM31 d0 = qm31_sub(z_mem, e0);
    // Access 1-3: data accesses
    QM31 d1 = qm31_sub(z_mem, qm31_add(qm31_from_m31(dst_addr),  qm31_mul(alpha_mem, qm31_from_m31(dst))));
    QM31 d2 = qm31_sub(z_mem, qm31_add(qm31_from_m31(op0_addr),  qm31_mul(alpha_mem, qm31_from_m31(op0))));
    QM31 d3 = qm31_sub(z_mem, qm31_add(qm31_from_m31(op1_addr),  qm31_mul(alpha_mem, qm31_from_m31(op1))));
    return qm31_add(qm31_add(qm31_add(qm31_inv(d0), qm31_inv(d1)), qm31_inv(d2)), qm31_inv(d3));
}

// Helper: compute RC step delta for one row.
__device__ __forceinline__ QM31 rc_step_delta(
    uint32_t off0, uint32_t off1, uint32_t off2, QM31 z_rc
) {
    QM31 r0 = qm31_sub(z_rc, qm31_from_m31(off0));
    QM31 r1 = qm31_sub(z_rc, qm31_from_m31(off1));
    QM31 r2 = qm31_sub(z_rc, qm31_from_m31(off2));
    return qm31_add(qm31_add(qm31_inv(r0), qm31_inv(r1)), qm31_inv(r2));
}

__global__ void cairo_quotient_kernel(
    const uint32_t* const* __restrict__ trace_cols,  // [34] column pointers
    // LogUp interaction trace (QM31 stored as 4 M31 cols, eval-domain order)
    const uint32_t* __restrict__ s_logup0, const uint32_t* __restrict__ s_logup1,
    const uint32_t* __restrict__ s_logup2, const uint32_t* __restrict__ s_logup3,
    // RC interaction trace
    const uint32_t* __restrict__ s_rc0, const uint32_t* __restrict__ s_rc1,
    const uint32_t* __restrict__ s_rc2, const uint32_t* __restrict__ s_rc3,
    // Dict step-transition interaction trace (QM31 stored as 4 M31 cols)
    const uint32_t* __restrict__ s_dict0, const uint32_t* __restrict__ s_dict1,
    const uint32_t* __restrict__ s_dict2, const uint32_t* __restrict__ s_dict3,
    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2, uint32_t* __restrict__ out3,
    const uint32_t* __restrict__ alpha_coeffs,  // [N_CONSTRAINTS * 4] QM31 coefficients
    const uint32_t* __restrict__ vh_inv,        // [n] 1/Z_H at each eval point (NTT order)
    // QM31 challenges: [z_mem(4), alpha_mem(4), alpha_mem_sq(4), z_rc(4), z_dict_link(4), alpha_dict_link(4)]
    const uint32_t* __restrict__ challenges,
    uint32_t n  // eval domain size
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t next_i = (i + 1) % n;

    // Unpack QM31 challenges
    QM31 z_mem          = {{challenges[0],  challenges[1],  challenges[2],  challenges[3]}};
    QM31 alpha_mem      = {{challenges[4],  challenges[5],  challenges[6],  challenges[7]}};
    QM31 alpha_mem_sq   = {{challenges[8],  challenges[9],  challenges[10], challenges[11]}};
    QM31 z_rc           = {{challenges[12], challenges[13], challenges[14], challenges[15]}};
    QM31 z_dict_link    = {{challenges[16], challenges[17], challenges[18], challenges[19]}};
    QM31 alpha_dict_link = {{challenges[20], challenges[21], challenges[22], challenges[23]}};

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

    // --- New soundness constraints 20-29 ---
    // Load new columns
    uint32_t off0 = trace_cols[COL_OFF0][i];
    uint32_t off1 = trace_cols[COL_OFF1][i];
    uint32_t off2 = trace_cols[COL_OFF2][i];
    uint32_t dst_inv = trace_cols[COL_DST_INV][i];
    uint32_t dst_addr = trace_cols[COL_DST_ADDR][i];
    uint32_t op0_addr = trace_cols[COL_OP0_ADDR][i];
    uint32_t op1_addr = trace_cols[COL_OP1_ADDR][i];
    uint32_t f_dst_reg = flags[0];
    uint32_t f_op0_reg = flags[1];
    uint32_t f_op1_fp = flags[3];
    uint32_t f_op1_ap = flags[4];

    uint32_t BIAS = 0x8000;

    // Constraint 20: dst_addr verification
    {
        uint32_t base = m31_add(m31_mul(m31_sub(1, f_dst_reg), ap), m31_mul(f_dst_reg, fp));
        uint32_t expected = m31_sub(m31_add(base, off0), BIAS);
        uint32_t c = m31_sub(dst_addr, expected);
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }

    // Constraint 21: op0_addr verification
    {
        uint32_t base = m31_add(m31_mul(m31_sub(1, f_op0_reg), ap), m31_mul(f_op0_reg, fp));
        uint32_t expected = m31_sub(m31_add(base, off1), BIAS);
        uint32_t c = m31_sub(op0_addr, expected);
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }

    // Constraint 22: op1_addr verification
    {
        uint32_t op1_default = m31_sub(m31_sub(m31_sub(1, f_op1_imm), f_op1_fp), f_op1_ap);
        uint32_t base = m31_add(m31_add(m31_add(
            m31_mul(f_op1_imm, pc), m31_mul(f_op1_fp, fp)),
            m31_mul(f_op1_ap, ap)), m31_mul(op1_default, op0));
        uint32_t expected = m31_sub(m31_add(base, off2), BIAS);
        uint32_t c = m31_sub(op1_addr, expected);
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }

    // Constraint 23: JNZ fall-through
    {
        uint32_t inst_size_c = m31_add(1, f_op1_imm);
        uint32_t dst_x_inv = m31_mul(dst, dst_inv);
        uint32_t c = m31_mul(f_pc_jnz, m31_mul(m31_sub(1, dst_x_inv), m31_sub(next_pc, m31_add(pc, inst_size_c))));
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }

    // Constraint 24: JNZ inverse consistency
    {
        uint32_t c = m31_mul(f_pc_jnz, m31_mul(dst, m31_sub(1, m31_mul(dst, dst_inv))));
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }

    // Constraint 25-27: Op1 source exclusivity
    {
        uint32_t c = m31_mul(f_op1_imm, f_op1_fp);
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }
    {
        uint32_t c = m31_mul(f_op1_imm, f_op1_ap);
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }
    {
        uint32_t c = m31_mul(f_op1_fp, f_op1_ap);
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }

    // Constraint 28: PC update exclusivity
    {
        uint32_t c = m31_add(m31_add(
            m31_mul(f_pc_jump_abs, f_pc_jump_rel),
            m31_mul(f_pc_jump_abs, f_pc_jnz)),
            m31_mul(f_pc_jump_rel, f_pc_jnz));
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }

    // Constraint 29: Opcode exclusivity
    {
        uint32_t c = m31_add(m31_add(
            m31_mul(f_call, f_ret),
            m31_mul(f_call, f_assert)),
            m31_mul(f_ret, f_assert));
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }

    // Constraint 30: Instruction decomposition
    // inst_lo + inst_hi * 2^31 = off0 + off1 * 2^16 + off2 * 2^32 + sum(flag_i * 2^(48+i))
    // In M31: 2^31 ≡ 1, 2^32 ≡ 2, 2^(48+i) ≡ 2^(17+i), 2^62 ≡ 1
    {
        uint32_t inst_lo_v = trace_cols[COL_INST_LO][i];
        uint32_t inst_hi_v = trace_cols[COL_INST_HI][i];
        // RHS = off0 + off1 * 2^16 + off2 * 2
        uint32_t rhs = m31_add(m31_add(off0, m31_mul(off1, (1u << 16))), m31_mul(off2, 2));
        // Add flag contributions: flag_i * 2^(17+i) for i=0..13
        for (int fi = 0; fi < 14; fi++) {
            rhs = m31_add(rhs, m31_mul(flags[fi], (1u << (17 + fi))));
        }
        // flag_14 * 2^62 ≡ flag_14 * 1
        rhs = m31_add(rhs, flags[14]);
        // LHS = inst_lo + inst_hi (since 2^31 ≡ 1)
        uint32_t lhs = m31_add(inst_lo_v, inst_hi_v);
        uint32_t c = m31_sub(lhs, rhs);
        QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
    }

    // --- Constraint 31: LogUp step transition (QM31 constraint) ---
    // S_logup[i+1] - S_logup[i] - logup_delta(row_i) = 0
    {
        QM31 s_curr = {{s_logup0[i],      s_logup1[i],      s_logup2[i],      s_logup3[i]}};
        QM31 s_next = {{s_logup0[next_i], s_logup1[next_i], s_logup2[next_i], s_logup3[next_i]}};
        uint32_t dst_addr_v = trace_cols[COL_DST_ADDR][i];
        uint32_t op0_addr_v = trace_cols[COL_OP0_ADDR][i];
        uint32_t op1_addr_v = trace_cols[COL_OP1_ADDR][i];
        uint32_t inst_lo_v  = trace_cols[COL_INST_LO][i];
        uint32_t inst_hi_v  = trace_cols[COL_INST_HI][i];
        QM31 delta = logup_step_delta(pc, inst_lo_v, inst_hi_v,
                                       dst_addr_v, dst, op0_addr_v, op0, op1_addr_v, op1,
                                       z_mem, alpha_mem, alpha_mem_sq);
        QM31 c31 = qm31_sub(qm31_sub(s_next, s_curr), delta);
        QM31 alpha31 = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul(alpha31, c31));
        ci++;
    }

    // --- Constraint 32: RC step transition (QM31 constraint) ---
    // S_rc[i+1] - S_rc[i] - rc_delta(row_i) = 0
    {
        QM31 s_curr_rc = {{s_rc0[i],      s_rc1[i],      s_rc2[i],      s_rc3[i]}};
        QM31 s_next_rc = {{s_rc0[next_i], s_rc1[next_i], s_rc2[next_i], s_rc3[next_i]}};
        QM31 delta_rc = rc_step_delta(off0, off1, off2, z_rc);
        QM31 c32 = qm31_sub(qm31_sub(s_next_rc, s_curr_rc), delta_rc);
        QM31 alpha32 = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul(alpha32, c32));
        ci++;
    }

    // --- Constraint 33: dict_active binary (dict_active * (1 - dict_active) = 0) ---
    {
        uint32_t dict_active_v = trace_cols[COL_DICT_ACTIVE][i];
        uint32_t c = m31_mul(dict_active_v, m31_sub(1, dict_active_v));
        QM31 alpha33 = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul_m31(alpha33, c));
        ci++;
    }

    // --- Constraint 34: S_dict step-transition (QM31 constraint) ---
    // S_dict[i+1] - S_dict[i] - dict_active[i] * inv(z_dict_link - (dict_key[i] + alpha_dict_link * dict_new[i])) = 0
    // When dict_active[i] = 0: S_dict is constant at this row (no dict access).
    // When dict_active[i] = 1: S_dict advances by one LogUp term.
    {
        QM31 s_curr_d = {{s_dict0[i],      s_dict1[i],      s_dict2[i],      s_dict3[i]}};
        QM31 s_next_d = {{s_dict0[next_i], s_dict1[next_i], s_dict2[next_i], s_dict3[next_i]}};
        uint32_t dict_active_v = trace_cols[COL_DICT_ACTIVE][i];
        QM31 dict_delta;
        if (dict_active_v == 0) {
            dict_delta.v[0] = 0; dict_delta.v[1] = 0;
            dict_delta.v[2] = 0; dict_delta.v[3] = 0;
        } else {
            uint32_t dict_key_v = trace_cols[COL_DICT_KEY][i];
            uint32_t dict_new_v = trace_cols[COL_DICT_NEW][i];
            QM31 entry = qm31_add(qm31_from_m31(dict_key_v),
                         qm31_mul(alpha_dict_link, qm31_from_m31(dict_new_v)));
            QM31 denom = qm31_sub(z_dict_link, entry);
            dict_delta = qm31_inv(denom);
        }
        QM31 c34 = qm31_sub(qm31_sub(s_next_d, s_curr_d), dict_delta);
        QM31 alpha34 = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
        quotient = qm31_add(quotient, qm31_mul(alpha34, c34));
        ci++;
    }

    // Divide by vanishing polynomial: Q(x) = C(x) / Z_H(x)
    quotient = qm31_mul_m31(quotient, vh_inv[i]);

    out0[i] = quotient.v[0];
    out1[i] = quotient.v[1];
    out2[i] = quotient.v[2];
    out3[i] = quotient.v[3];
}

// Chunked variant for streaming
__global__ void cairo_quotient_chunk_kernel(
    const uint32_t* const* __restrict__ trace_cols,
    const uint32_t* __restrict__ s_logup0, const uint32_t* __restrict__ s_logup1,
    const uint32_t* __restrict__ s_logup2, const uint32_t* __restrict__ s_logup3,
    const uint32_t* __restrict__ s_rc0, const uint32_t* __restrict__ s_rc1,
    const uint32_t* __restrict__ s_rc2, const uint32_t* __restrict__ s_rc3,
    const uint32_t* __restrict__ s_dict0, const uint32_t* __restrict__ s_dict1,
    const uint32_t* __restrict__ s_dict2, const uint32_t* __restrict__ s_dict3,
    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2, uint32_t* __restrict__ out3,
    const uint32_t* __restrict__ alpha_coeffs,
    const uint32_t* __restrict__ vh_inv,
    const uint32_t* __restrict__ challenges,
    uint32_t offset, uint32_t chunk_n, uint32_t global_n
) {
    uint32_t local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i >= chunk_n) return;

    uint32_t i = offset + local_i;
    uint32_t next_i = (i + 1) % global_n;

    QM31 z_mem           = {{challenges[0],  challenges[1],  challenges[2],  challenges[3]}};
    QM31 alpha_mem       = {{challenges[4],  challenges[5],  challenges[6],  challenges[7]}};
    QM31 alpha_mem_sq    = {{challenges[8],  challenges[9],  challenges[10], challenges[11]}};
    QM31 z_rc            = {{challenges[12], challenges[13], challenges[14], challenges[15]}};
    QM31 z_dict_link     = {{challenges[16], challenges[17], challenges[18], challenges[19]}};
    QM31 alpha_dict_link = {{challenges[20], challenges[21], challenges[22], challenges[23]}};

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

    // --- New soundness constraints 20-29 (chunk kernel) ---
    {
        uint32_t off0_v = trace_cols[COL_OFF0][i];
        uint32_t off1_v = trace_cols[COL_OFF1][i];
        uint32_t off2_v = trace_cols[COL_OFF2][i];
        uint32_t dst_inv_v = trace_cols[COL_DST_INV][i];
        uint32_t dst_addr_v = trace_cols[COL_DST_ADDR][i];
        uint32_t op0_addr_v = trace_cols[COL_OP0_ADDR][i];
        uint32_t op1_addr_v = trace_cols[COL_OP1_ADDR][i];
        uint32_t f_dst_reg = flags[0];
        uint32_t f_op0_reg = flags[1];
        uint32_t f_op1_fp_v = flags[3];
        uint32_t f_op1_ap_v = flags[4];
        uint32_t BIAS = 0x8000;

        // Constraint 20: dst_addr
        {
            uint32_t base = m31_add(m31_mul(m31_sub(1, f_dst_reg), ap), m31_mul(f_dst_reg, fp));
            uint32_t expected = m31_sub(m31_add(base, off0_v), BIAS);
            uint32_t c = m31_sub(dst_addr_v, expected);
            QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
        }
        // Constraint 21: op0_addr
        {
            uint32_t base = m31_add(m31_mul(m31_sub(1, f_op0_reg), ap), m31_mul(f_op0_reg, fp));
            uint32_t expected = m31_sub(m31_add(base, off1_v), BIAS);
            uint32_t c = m31_sub(op0_addr_v, expected);
            QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
        }
        // Constraint 22: op1_addr
        {
            uint32_t op1_default = m31_sub(m31_sub(m31_sub(1, f_op1_imm), f_op1_fp_v), f_op1_ap_v);
            uint32_t base = m31_add(m31_add(m31_add(
                m31_mul(f_op1_imm, pc), m31_mul(f_op1_fp_v, fp)),
                m31_mul(f_op1_ap_v, ap)), m31_mul(op1_default, op0));
            uint32_t expected = m31_sub(m31_add(base, off2_v), BIAS);
            uint32_t c = m31_sub(op1_addr_v, expected);
            QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
        }
        // Constraint 23: JNZ fall-through
        {
            uint32_t inst_size_c = m31_add(1, f_op1_imm);
            uint32_t dst_x_inv = m31_mul(dst, dst_inv_v);
            uint32_t c = m31_mul(f_pc_jnz, m31_mul(m31_sub(1, dst_x_inv), m31_sub(next_pc, m31_add(pc, inst_size_c))));
            QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
        }
        // Constraint 24: JNZ inverse consistency
        {
            uint32_t c = m31_mul(f_pc_jnz, m31_mul(dst, m31_sub(1, m31_mul(dst, dst_inv_v))));
            QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
        }
        // Constraint 25-27: Op1 source exclusivity
        {
            uint32_t c = m31_mul(f_op1_imm, f_op1_fp_v);
            QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
        }
        {
            uint32_t c = m31_mul(f_op1_imm, f_op1_ap_v);
            QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
        }
        {
            uint32_t c = m31_mul(f_op1_fp_v, f_op1_ap_v);
            QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
        }
        // Constraint 28: PC update exclusivity
        {
            uint32_t c = m31_add(m31_add(
                m31_mul(f_pc_jump_abs, f_pc_jump_rel),
                m31_mul(f_pc_jump_abs, f_pc_jnz)),
                m31_mul(f_pc_jump_rel, f_pc_jnz));
            QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
        }
        // Constraint 29: Opcode exclusivity
        {
            uint32_t c = m31_add(m31_add(
                m31_mul(f_call, f_ret),
                m31_mul(f_call, f_assert)),
                m31_mul(f_ret, f_assert));
            QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
        }

        // Constraint 30: Instruction decomposition
        {
            uint32_t inst_lo_v = trace_cols[COL_INST_LO][i];
            uint32_t inst_hi_v = trace_cols[COL_INST_HI][i];
            uint32_t rhs = m31_add(m31_add(off0_v, m31_mul(off1_v, (1u << 16))), m31_mul(off2_v, 2));
            for (int fi = 0; fi < 14; fi++) {
                rhs = m31_add(rhs, m31_mul(flags[fi], (1u << (17 + fi))));
            }
            rhs = m31_add(rhs, flags[14]);
            uint32_t lhs = m31_add(inst_lo_v, inst_hi_v);
            uint32_t c = m31_sub(lhs, rhs);
            QM31 alpha = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul_m31(alpha, c)); ci++;
        }

        // Constraint 31: LogUp step transition
        {
            QM31 s_curr = {{s_logup0[i],      s_logup1[i],      s_logup2[i],      s_logup3[i]}};
            QM31 s_next = {{s_logup0[next_i], s_logup1[next_i], s_logup2[next_i], s_logup3[next_i]}};
            uint32_t inst_lo_v  = trace_cols[COL_INST_LO][i];
            uint32_t inst_hi_v  = trace_cols[COL_INST_HI][i];
            uint32_t dst_addr_v = trace_cols[COL_DST_ADDR][i];
            uint32_t op0_addr_v = trace_cols[COL_OP0_ADDR][i];
            uint32_t op1_addr_v = trace_cols[COL_OP1_ADDR][i];
            QM31 delta = logup_step_delta(pc, inst_lo_v, inst_hi_v,
                                           dst_addr_v, dst, op0_addr_v, op0, op1_addr_v, op1,
                                           z_mem, alpha_mem, alpha_mem_sq);
            QM31 c31 = qm31_sub(qm31_sub(s_next, s_curr), delta);
            QM31 alpha31 = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul(alpha31, c31)); ci++;
        }

        // Constraint 32: RC step transition
        {
            QM31 s_curr_rc = {{s_rc0[i],      s_rc1[i],      s_rc2[i],      s_rc3[i]}};
            QM31 s_next_rc = {{s_rc0[next_i], s_rc1[next_i], s_rc2[next_i], s_rc3[next_i]}};
            QM31 delta_rc = rc_step_delta(off0_v, off1_v, off2_v, z_rc);
            QM31 c32 = qm31_sub(qm31_sub(s_next_rc, s_curr_rc), delta_rc);
            QM31 alpha32 = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul(alpha32, c32)); ci++;
        }

        // Constraint 33: dict_active binary
        {
            uint32_t dict_active_v = trace_cols[COL_DICT_ACTIVE][i];
            uint32_t c = m31_mul(dict_active_v, m31_sub(1, dict_active_v));
            QM31 alpha33 = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul_m31(alpha33, c)); ci++;
        }

        // Constraint 34: S_dict step-transition
        {
            QM31 s_curr_d = {{s_dict0[i],      s_dict1[i],      s_dict2[i],      s_dict3[i]}};
            QM31 s_next_d = {{s_dict0[next_i], s_dict1[next_i], s_dict2[next_i], s_dict3[next_i]}};
            uint32_t dict_active_v = trace_cols[COL_DICT_ACTIVE][i];
            QM31 dict_delta;
            if (dict_active_v == 0) {
                dict_delta.v[0] = 0; dict_delta.v[1] = 0;
                dict_delta.v[2] = 0; dict_delta.v[3] = 0;
            } else {
                uint32_t dict_key_v = trace_cols[COL_DICT_KEY][i];
                uint32_t dict_new_v = trace_cols[COL_DICT_NEW][i];
                QM31 entry = qm31_add(qm31_from_m31(dict_key_v),
                             qm31_mul(alpha_dict_link, qm31_from_m31(dict_new_v)));
                QM31 denom = qm31_sub(z_dict_link, entry);
                dict_delta = qm31_inv(denom);
            }
            QM31 c34 = qm31_sub(qm31_sub(s_next_d, s_curr_d), dict_delta);
            QM31 alpha34 = {{alpha_coeffs[ci*4], alpha_coeffs[ci*4+1], alpha_coeffs[ci*4+2], alpha_coeffs[ci*4+3]}};
            quotient = qm31_add(quotient, qm31_mul(alpha34, c34)); ci++;
        }
    }

    // Divide by vanishing polynomial: Q(x) = C(x) / Z_H(x)
    quotient = qm31_mul_m31(quotient, vh_inv[i]);

    out0[local_i] = quotient.v[0];
    out1[local_i] = quotient.v[1];
    out2[local_i] = quotient.v[2];
    out3[local_i] = quotient.v[3];
}

extern "C" {

void cuda_compute_vanishing_inv(
    uint32_t initial_x, uint32_t initial_y,
    uint32_t step_x, uint32_t step_y,
    uint32_t* out_vh_inv,
    uint32_t log_eval,
    uint32_t log_n
) {
    uint32_t eval_n = 1u << log_eval;
    uint32_t threads = 256;
    uint32_t blocks = (eval_n + threads - 1) / threads;
    compute_vanishing_inv_kernel<<<blocks, threads>>>(
        initial_x, initial_y, step_x, step_y,
        out_vh_inv, log_eval, log_n
    );
}

void cuda_cairo_quotient(
    const uint32_t* const* trace_cols,
    const uint32_t* s_logup0, const uint32_t* s_logup1,
    const uint32_t* s_logup2, const uint32_t* s_logup3,
    const uint32_t* s_rc0, const uint32_t* s_rc1,
    const uint32_t* s_rc2, const uint32_t* s_rc3,
    const uint32_t* s_dict0, const uint32_t* s_dict1,
    const uint32_t* s_dict2, const uint32_t* s_dict3,
    uint32_t* out0, uint32_t* out1, uint32_t* out2, uint32_t* out3,
    const uint32_t* alpha_coeffs,
    const uint32_t* vh_inv,
    const uint32_t* challenges,
    uint32_t n
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    cairo_quotient_kernel<<<blocks, threads>>>(
        trace_cols,
        s_logup0, s_logup1, s_logup2, s_logup3,
        s_rc0, s_rc1, s_rc2, s_rc3,
        s_dict0, s_dict1, s_dict2, s_dict3,
        out0, out1, out2, out3,
        alpha_coeffs, vh_inv, challenges, n
    );
}

void cuda_cairo_quotient_chunk(
    const uint32_t* const* trace_cols,
    const uint32_t* s_logup0, const uint32_t* s_logup1,
    const uint32_t* s_logup2, const uint32_t* s_logup3,
    const uint32_t* s_rc0, const uint32_t* s_rc1,
    const uint32_t* s_rc2, const uint32_t* s_rc3,
    const uint32_t* s_dict0, const uint32_t* s_dict1,
    const uint32_t* s_dict2, const uint32_t* s_dict3,
    uint32_t* out0, uint32_t* out1, uint32_t* out2, uint32_t* out3,
    const uint32_t* alpha_coeffs,
    const uint32_t* vh_inv,
    const uint32_t* challenges,
    uint32_t offset, uint32_t chunk_n, uint32_t global_n
) {
    uint32_t threads = 256;
    uint32_t blocks = (chunk_n + threads - 1) / threads;
    cairo_quotient_chunk_kernel<<<blocks, threads>>>(
        trace_cols,
        s_logup0, s_logup1, s_logup2, s_logup3,
        s_rc0, s_rc1, s_rc2, s_rc3,
        s_dict0, s_dict1, s_dict2, s_dict3,
        out0, out1, out2, out3,
        alpha_coeffs, vh_inv, challenges,
        offset, chunk_n, global_n
    );
}

// ── ZK Blinding helpers ───────────────────────────────────────────────────
//
// cuda_compute_vanishing: compute Z_H at each eval-domain position (not its
//   inverse).  Z_H(x) = f_{log_n}(x)+1; same formula as vanishing_poly().
//   Used to blind trace columns: column[i] += r * Z_H[i] before commitment.
//
// cuda_axpy_m31: fused multiply-add: y[i] = y[i] + scalar * x[i]  (mod P).

__global__ void compute_vanishing_kernel(
    uint32_t initial_x, uint32_t initial_y,
    uint32_t step_x,    uint32_t step_y,
    uint32_t* __restrict__ out_zh,
    uint32_t log_eval,
    uint32_t log_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t eval_n = 1u << log_eval;
    if (i >= eval_n) return;

    uint32_t j = cairo_bit_reverse(i, log_eval);

    uint32_t rx = 1u, ry = 0u;
    uint32_t bx = step_x, by = step_y;
    uint32_t k = j;
    for (uint32_t bit = 0; bit < log_eval; bit++) {
        if (k & 1u) {
            uint32_t nx = m31_sub(m31_mul(rx, bx), m31_mul(ry, by));
            uint32_t ny = m31_add(m31_mul(rx, by), m31_mul(ry, bx));
            rx = nx; ry = ny;
        }
        uint32_t nx = m31_sub(m31_add(m31_mul(bx, bx), m31_mul(bx, bx)), 1u);
        uint32_t ny = m31_add(m31_mul(bx, by), m31_mul(bx, by));
        bx = nx; by = ny;
        k >>= 1;
    }
    uint32_t px = m31_sub(m31_mul(initial_x, rx), m31_mul(initial_y, ry));
    out_zh[i] = vanishing_poly(px, log_n);   // Z_H, NOT its inverse
}

void cuda_compute_vanishing(
    uint32_t initial_x, uint32_t initial_y,
    uint32_t step_x, uint32_t step_y,
    uint32_t* out_zh,
    uint32_t log_eval,
    uint32_t log_n
) {
    uint32_t eval_n = 1u << log_eval;
    uint32_t threads = 256;
    uint32_t blocks = (eval_n + threads - 1) / threads;
    compute_vanishing_kernel<<<blocks, threads>>>(
        initial_x, initial_y, step_x, step_y,
        out_zh, log_eval, log_n
    );
}

__global__ void axpy_m31_kernel(
    uint32_t scalar,
    const uint32_t* __restrict__ x,
    uint32_t* __restrict__ y,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = m31_add(y[i], m31_mul(scalar, x[i]));
}

void cuda_axpy_m31(
    uint32_t scalar,
    const uint32_t* x,
    uint32_t* y,
    uint32_t n
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    axpy_m31_kernel<<<blocks, threads>>>(scalar, x, y, n);
}

} // extern "C"
