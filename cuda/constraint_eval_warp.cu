// Warp-cooperative GPU bytecode interpreter for constraint evaluation.
//
// Instead of one thread per row (which needs 90KB+ register files for large
// components), this kernel uses one WARP (32 threads) per row. The register
// file is distributed across warp lanes:
//   - Thread `lane` owns registers where (reg_idx % 32 == lane)
//   - Each thread holds at most ceil(n_registers / 32) QM31 values
//   - For 5600 registers: 175 per thread = 2.8KB (fits in actual GPU registers!)
//
// Operations use __shfl_sync to broadcast source operand values from the
// owning lane. This costs ~1 cycle per shuffle vs hundreds of cycles for
// local memory spills.

#include <cstdio>
#include "include/m31.cuh"
#include "include/cm31.cuh"
#include "include/qm31.cuh"

// ─── Opcodes (must match bytecode.rs) ──────────────────────────────────

#define OP_LOAD_CONST           0x01
#define OP_LOAD_SECURE_CONST    0x02
#define OP_LOAD_TRACE           0x03

#define OP_ADD                  0x10
#define OP_SUB                  0x11
#define OP_MUL                  0x12
#define OP_NEG                  0x13
#define OP_ADD_CONST            0x14
#define OP_MUL_CONST            0x15

#define OP_WIDE_ADD             0x20
#define OP_WIDE_SUB             0x21
#define OP_WIDE_MUL             0x22
#define OP_WIDE_NEG             0x23
#define OP_WIDE_ADD_CONST       0x24
#define OP_WIDE_MUL_CONST       0x25

#define OP_WIDE_ADD_BASE        0x28
#define OP_WIDE_MUL_BASE        0x29
#define OP_BASE_ADD_SECURE_CONST 0x2A
#define OP_BASE_MUL_SECURE_CONST 0x2B

#define OP_WIDEN                0x30
#define OP_COMBINE_EF           0x31

#define OP_ADD_CONSTRAINT       0x40

// Max registers per thread (= ceil(total_regs / 32)).
// 256 per thread * 32 lanes = 8192 total registers max.
#define REGS_PER_LANE 256

// ─── Bit-reversed circle domain offset ─────────────────────────────────

__device__ __forceinline__
uint32_t bit_reverse(uint32_t v, uint32_t log_n) {
    return __brev(v) >> (32 - log_n);
}

__device__ __forceinline__
uint32_t offset_bit_reversed_circle_domain_index(
    uint32_t i, uint32_t domain_log_size, uint32_t eval_log_size, int32_t offset
) {
    uint32_t prev_index = bit_reverse(i, eval_log_size);
    uint32_t half_size = 1u << (eval_log_size - 1);
    int32_t step_size = offset * (int32_t)(1u << (eval_log_size - domain_log_size - 1));
    if (prev_index < half_size) {
        int32_t idx = ((int32_t)prev_index + step_size) % (int32_t)half_size;
        if (idx < 0) idx += (int32_t)half_size;
        prev_index = (uint32_t)idx;
    } else {
        int32_t idx = ((int32_t)(prev_index - half_size) - step_size) % (int32_t)half_size;
        if (idx < 0) idx += (int32_t)half_size;
        prev_index = (uint32_t)idx + half_size;
    }
    return bit_reverse(prev_index, eval_log_size);
}

// ─── Warp-cooperative register access ──────────────────────────────────

// Read a QM31 value from the distributed register file.
// The owning lane broadcasts all 4 components via __shfl_sync.
__device__ __forceinline__
QM31 warp_reg_read(const QM31* local_regs, uint32_t reg_idx, uint32_t lane) {
    uint32_t owner = reg_idx & 31;        // which lane owns this register
    uint32_t slot  = reg_idx >> 5;         // index within that lane's local array
    // Each lane reads from its own local_regs[slot] — only the owner has the real value.
    // We broadcast from the owner lane.
    QM31 val;
    if (slot < REGS_PER_LANE) {
        QM31 my_val = local_regs[slot];
        val.v[0] = __shfl_sync(0xFFFFFFFF, my_val.v[0], owner);
        val.v[1] = __shfl_sync(0xFFFFFFFF, my_val.v[1], owner);
        val.v[2] = __shfl_sync(0xFFFFFFFF, my_val.v[2], owner);
        val.v[3] = __shfl_sync(0xFFFFFFFF, my_val.v[3], owner);
    } else {
        val = qm31_zero();
    }
    return val;
}

// Write a QM31 value to the distributed register file.
// Only the owning lane performs the write.
__device__ __forceinline__
void warp_reg_write(QM31* local_regs, uint32_t reg_idx, QM31 val, uint32_t lane) {
    uint32_t owner = reg_idx & 31;
    uint32_t slot  = reg_idx >> 5;
    if (lane == owner && slot < REGS_PER_LANE) {
        local_regs[slot] = val;
    }
}

// ─── Kernel ────────────────────────────────────────────────────────────

__global__ void warp_bytecode_constraint_eval_kernel(
    const uint32_t* __restrict__ bytecode,
    uint32_t n_words,
    const uint32_t* const* __restrict__ trace_cols,
    const uint32_t* __restrict__ trace_col_sizes,
    uint32_t n_trace_cols,
    uint32_t n_rows,
    uint32_t trace_n_rows,
    const uint32_t* __restrict__ random_coeff_powers,
    const uint32_t* __restrict__ denom_inv,
    uint32_t log_expand,
    uint32_t* __restrict__ accum0,
    uint32_t* __restrict__ accum1,
    uint32_t* __restrict__ accum2,
    uint32_t* __restrict__ accum3
) {
    // One warp per row. Global warp index = row.
    uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    uint32_t lane = threadIdx.x & 31;
    uint32_t row = warp_id;
    if (row >= n_rows) return;

    // Distributed register file: each lane holds REGS_PER_LANE QM31 values.
    QM31 regs[REGS_PER_LANE];

    QM31 row_accum = qm31_zero();
    int constraint_idx = 0;

    uint32_t pc = 0;
    while (pc < n_words) {
        // All lanes decode the same instruction word (SIMT — free).
        uint32_t word = bytecode[pc++];
        uint32_t opcode = word >> 24;

        switch (opcode) {

        // ── Register loads ──────────────────────────────────────────

        case OP_LOAD_CONST: {
            uint32_t dst = word & 0xFFFF;
            uint32_t val = bytecode[pc++];
            warp_reg_write(regs, dst, {{val, 0, 0, 0}}, lane);
            break;
        }

        case OP_LOAD_SECURE_CONST: {
            uint32_t dst = word & 0xFFFF;
            uint32_t a = bytecode[pc++];
            uint32_t b = bytecode[pc++];
            uint32_t c = bytecode[pc++];
            uint32_t d = bytecode[pc++];
            warp_reg_write(regs, dst, {{a, b, c, d}}, lane);
            break;
        }

        case OP_LOAD_TRACE: {
            uint32_t dst      = word & 0xFFFF;
            uint32_t flat_col = bytecode[pc++];
            uint32_t ext      = bytecode[pc++];
            uint32_t sign     = ext >> 31;
            uint32_t abs_off  = ext & 0x7FFFFFFF;
            int32_t  offset   = sign ? -(int32_t)abs_off : (int32_t)abs_off;

            if (flat_col >= n_trace_cols) { return; }

            uint32_t effective_row;
            if (offset == 0) {
                effective_row = row;
            } else {
                uint32_t eval_log   = 31 - __clz(n_rows);
                uint32_t domain_log = 31 - __clz(trace_n_rows);
                effective_row = offset_bit_reversed_circle_domain_index(
                    row, domain_log, eval_log, offset);
            }

            const uint32_t* col_ptr = trace_cols[flat_col];
            uint32_t val = (col_ptr != nullptr) ? col_ptr[effective_row] : 0;
            warp_reg_write(regs, dst, {{val, 0, 0, 0}}, lane);
            break;
        }

        // ── M31 arithmetic (3-register) ─────────────────────────────

        case OP_ADD: {
            uint32_t dst  = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t src1 = word2 >> 16;
            uint32_t src2 = word2 & 0xFFFF;
            QM31 a = warp_reg_read(regs, src1, lane);
            QM31 b = warp_reg_read(regs, src2, lane);
            warp_reg_write(regs, dst, {{m31_add(a.v[0], b.v[0]), 0, 0, 0}}, lane);
            break;
        }

        case OP_SUB: {
            uint32_t dst  = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t src1 = word2 >> 16;
            uint32_t src2 = word2 & 0xFFFF;
            QM31 a = warp_reg_read(regs, src1, lane);
            QM31 b = warp_reg_read(regs, src2, lane);
            warp_reg_write(regs, dst, {{m31_sub(a.v[0], b.v[0]), 0, 0, 0}}, lane);
            break;
        }

        case OP_MUL: {
            uint32_t dst  = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t src1 = word2 >> 16;
            uint32_t src2 = word2 & 0xFFFF;
            QM31 a = warp_reg_read(regs, src1, lane);
            QM31 b = warp_reg_read(regs, src2, lane);
            warp_reg_write(regs, dst, {{m31_mul(a.v[0], b.v[0]), 0, 0, 0}}, lane);
            break;
        }

        case OP_NEG: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            QM31 a = warp_reg_read(regs, src, lane);
            warp_reg_write(regs, dst, {{m31_neg(a.v[0]), 0, 0, 0}}, lane);
            break;
        }

        case OP_ADD_CONST: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            uint32_t val = bytecode[pc++];
            QM31 a = warp_reg_read(regs, src, lane);
            warp_reg_write(regs, dst, {{m31_add(a.v[0], val), 0, 0, 0}}, lane);
            break;
        }

        case OP_MUL_CONST: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            uint32_t val = bytecode[pc++];
            QM31 a = warp_reg_read(regs, src, lane);
            warp_reg_write(regs, dst, {{m31_mul(a.v[0], val), 0, 0, 0}}, lane);
            break;
        }

        // ── QM31 arithmetic ────────────────────────────────────────

        case OP_WIDE_ADD: {
            uint32_t dst  = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t src1 = word2 >> 16;
            uint32_t src2 = word2 & 0xFFFF;
            QM31 a = warp_reg_read(regs, src1, lane);
            QM31 b = warp_reg_read(regs, src2, lane);
            warp_reg_write(regs, dst, qm31_add(a, b), lane);
            break;
        }

        case OP_WIDE_SUB: {
            uint32_t dst  = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t src1 = word2 >> 16;
            uint32_t src2 = word2 & 0xFFFF;
            QM31 a = warp_reg_read(regs, src1, lane);
            QM31 b = warp_reg_read(regs, src2, lane);
            warp_reg_write(regs, dst, qm31_sub(a, b), lane);
            break;
        }

        case OP_WIDE_MUL: {
            uint32_t dst  = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t src1 = word2 >> 16;
            uint32_t src2 = word2 & 0xFFFF;
            QM31 a = warp_reg_read(regs, src1, lane);
            QM31 b = warp_reg_read(regs, src2, lane);
            warp_reg_write(regs, dst, qm31_mul(a, b), lane);
            break;
        }

        case OP_WIDE_NEG: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            QM31 a = warp_reg_read(regs, src, lane);
            warp_reg_write(regs, dst, qm31_neg(a), lane);
            break;
        }

        case OP_WIDE_ADD_CONST: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            QM31 a = warp_reg_read(regs, src, lane);
            warp_reg_write(regs, dst, qm31_add(a, c), lane);
            break;
        }

        case OP_WIDE_MUL_CONST: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            QM31 a = warp_reg_read(regs, src, lane);
            warp_reg_write(regs, dst, qm31_mul(a, c), lane);
            break;
        }

        // ── Mixed-width arithmetic ──────────────────────────────────

        case OP_WIDE_ADD_BASE: {
            uint32_t dst   = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t wide  = word2 >> 16;
            uint32_t base  = word2 & 0xFFFF;
            QM31 a = warp_reg_read(regs, wide, lane);
            QM31 b = warp_reg_read(regs, base, lane);
            a.v[0] = m31_add(a.v[0], b.v[0]);
            warp_reg_write(regs, dst, a, lane);
            break;
        }

        case OP_WIDE_MUL_BASE: {
            uint32_t dst   = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t wide  = word2 >> 16;
            uint32_t base  = word2 & 0xFFFF;
            QM31 a = warp_reg_read(regs, wide, lane);
            QM31 b = warp_reg_read(regs, base, lane);
            warp_reg_write(regs, dst, qm31_mul_m31(a, b.v[0]), lane);
            break;
        }

        case OP_BASE_ADD_SECURE_CONST: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            QM31 a = warp_reg_read(regs, src, lane);
            QM31 base_as_qm31 = {{a.v[0], 0, 0, 0}};
            warp_reg_write(regs, dst, qm31_add(base_as_qm31, c), lane);
            break;
        }

        case OP_BASE_MUL_SECURE_CONST: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            QM31 a = warp_reg_read(regs, src, lane);
            warp_reg_write(regs, dst, qm31_mul_m31(c, a.v[0]), lane);
            break;
        }

        // ── Widening ────────────────────────────────────────────────

        case OP_WIDEN: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            QM31 a = warp_reg_read(regs, src, lane);
            warp_reg_write(regs, dst, {{a.v[0], 0, 0, 0}}, lane);
            break;
        }

        case OP_COMBINE_EF: {
            uint32_t dst   = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t word3 = bytecode[pc++];
            uint32_t src0  = word2 >> 16;
            uint32_t src1  = word2 & 0xFFFF;
            uint32_t src2  = word3 >> 16;
            uint32_t src3  = word3 & 0xFFFF;
            QM31 a = warp_reg_read(regs, src0, lane);
            QM31 b = warp_reg_read(regs, src1, lane);
            QM31 c = warp_reg_read(regs, src2, lane);
            QM31 d = warp_reg_read(regs, src3, lane);
            warp_reg_write(regs, dst, {{a.v[0], b.v[0], c.v[0], d.v[0]}}, lane);
            break;
        }

        // ── Constraint accumulation ─────────────────────────────────

        case OP_ADD_CONSTRAINT: {
            uint32_t src = word & 0xFFFF;
            QM31 val = warp_reg_read(regs, src, lane);
            QM31 coeff = {{
                random_coeff_powers[constraint_idx * 4 + 0],
                random_coeff_powers[constraint_idx * 4 + 1],
                random_coeff_powers[constraint_idx * 4 + 2],
                random_coeff_powers[constraint_idx * 4 + 3]
            }};
            row_accum = qm31_add(row_accum, qm31_mul(val, coeff));
            constraint_idx++;
            break;
        }

        default:
            break;
        }
    }

    // Only lane 0 writes the final result (all lanes computed the same row_accum).
    if (lane == 0) {
        uint32_t trace_log_size = 31 - __clz(trace_n_rows);
        uint32_t denom = denom_inv[row >> trace_log_size];
        QM31 result = qm31_mul_m31(row_accum, denom);

        accum0[row] = m31_add(accum0[row], result.v[0]);
        accum1[row] = m31_add(accum1[row], result.v[1]);
        accum2[row] = m31_add(accum2[row], result.v[2]);
        accum3[row] = m31_add(accum3[row], result.v[3]);
    }
}

// ─── C wrapper for Rust FFI ─────────────────────────────────────────────

extern "C" {

void cuda_warp_bytecode_constraint_eval(
    const uint32_t* bytecode,
    uint32_t n_words,
    const uint32_t* const* trace_cols,
    const uint32_t* trace_col_sizes,
    uint32_t n_trace_cols,
    uint32_t n_rows,
    uint32_t trace_n_rows,
    const uint32_t* random_coeff_powers,
    const uint32_t* denom_inv,
    uint32_t log_expand,
    uint32_t* accum0,
    uint32_t* accum1,
    uint32_t* accum2,
    uint32_t* accum3,
    uint32_t n_registers
) {
    // Each warp (32 threads) handles one row.
    // Block size should be a multiple of 32. Use 128 threads = 4 warps per block.
    uint32_t threads_per_block = 128;
    uint32_t warps_per_block = threads_per_block / 32;
    uint32_t blocks = (n_rows + warps_per_block - 1) / warps_per_block;

    warp_bytecode_constraint_eval_kernel<<<blocks, threads_per_block>>>(
        bytecode, n_words,
        trace_cols, trace_col_sizes, n_trace_cols,
        n_rows, trace_n_rows,
        random_coeff_powers, denom_inv, log_expand,
        accum0, accum1, accum2, accum3
    );
}

} // extern "C"
