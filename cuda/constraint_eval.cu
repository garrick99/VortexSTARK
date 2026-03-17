// GPU register-based bytecode interpreter for generic constraint evaluation.
//
// Replays a flat bytecode program (recorded by TracingEvalAtRow) per row in parallel.
// Each thread evaluates all constraints for one row of the evaluation domain.
//
// The bytecode is the same for every row — only trace column lookups differ.
//
// Register-based design: each value lives in a virtual register. Clone just copies
// the register index — no duplication, no stack management, no underflow.
// This handles ALL stwo-cairo components including the 32 that used Clone.

#include "include/m31.cuh"
#include "include/cm31.cuh"
#include "include/qm31.cuh"

// ─── Opcodes (must match BytecodeOp::opcode_id() in bytecode.rs) ────────

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

// Maximum number of virtual registers per thread.
// Logup-heavy components with many fraction cross-multiplications and
// Clone-based value reuse can use many registers. 256 covers the
// largest stwo-cairo components.
#define MAX_REGS 256

// ─── Kernel ─────────────────────────────────────────────────────────────

__global__ void bytecode_constraint_eval_kernel(
    const uint32_t* __restrict__ bytecode,      // encoded instruction stream
    uint32_t n_words,                            // total words in bytecode
    const uint32_t* const* __restrict__ trace_cols, // array of column device pointers
    const uint32_t* __restrict__ trace_col_sizes,   // reserved (all cols same size)
    uint32_t n_trace_cols,                       // total number of columns in trace_cols
    uint32_t n_rows,                             // rows in evaluation domain
    uint32_t trace_n_rows,                       // rows in trace domain (for offset wrapping)
    const uint32_t* __restrict__ random_coeff_powers, // [n_constraints * 4] QM31 values (reversed)
    const uint32_t* __restrict__ denom_inv,      // [1 << log_expand] M31 values (bit-reversed)
    uint32_t log_expand,                         // eval_log_size - trace_log_size
    uint32_t* __restrict__ accum0,               // output accumulator columns (SoA QM31)
    uint32_t* __restrict__ accum1,
    uint32_t* __restrict__ accum2,
    uint32_t* __restrict__ accum3
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Register file. Each register holds a QM31 value.
    // M31 values are stored as (val, 0, 0, 0).
    QM31 regs[MAX_REGS];

    QM31 row_accum = qm31_zero();
    int constraint_idx = 0;

    uint32_t pc = 0;
    while (pc < n_words) {
        uint32_t word = bytecode[pc++];
        uint32_t opcode = word >> 24;

        switch (opcode) {

        // ── Register loads ──────────────────────────────────────────

        case OP_LOAD_CONST: {
            // [opcode:8 | dst:8 | value:16]
            // If value == 0xFFFF, next word has full value.
            uint32_t dst = (word >> 16) & 0xFF;
            uint32_t val = word & 0xFFFF;
            if (val == 0xFFFF) {
                val = bytecode[pc++];
            }
            if (dst < MAX_REGS) {
                regs[dst] = {{val, 0, 0, 0}};
            }
            break;
        }

        case OP_LOAD_SECURE_CONST: {
            // [opcode:8 | dst:8 | 0:16] + 4 data words
            uint32_t dst = (word >> 16) & 0xFF;
            uint32_t a = bytecode[pc++];
            uint32_t b = bytecode[pc++];
            uint32_t c = bytecode[pc++];
            uint32_t d = bytecode[pc++];
            if (dst < MAX_REGS) {
                regs[dst] = {{a, b, c, d}};
            }
            break;
        }

        case OP_LOAD_TRACE: {
            // [opcode:8 | dst:8 | operand:16]
            // After Rust remapping: operand = flat_col:14 | sign:1 | abs_offset:1
            // If both low bits are 1 (marker = 0x3), extended format: next word has offset
            uint32_t dst = (word >> 16) & 0xFF;
            uint32_t operand = word & 0xFFFF;

            uint32_t flat_col;
            int32_t offset;

            if ((operand & 0x3) == 0x3) {
                // Extended: next word has (sign:1 | abs_offset:31)
                // Operand has flat_col in upper bits
                flat_col = (operand >> 2) & 0x3FFF;
                uint32_t ext = bytecode[pc++];
                uint32_t sign = ext >> 31;
                uint32_t abs_off = ext & 0x7FFFFFFF;
                offset = sign ? -(int32_t)abs_off : (int32_t)abs_off;
            } else {
                flat_col = (operand >> 2) & 0x3FFF;
                uint32_t sign = (operand >> 1) & 1;
                uint32_t abs_off = operand & 1;
                offset = sign ? -(int32_t)abs_off : (int32_t)abs_off;
            }

            if (flat_col >= n_trace_cols || dst >= MAX_REGS) {
                return;
            }

            uint32_t effective_row;
            if (offset == 0) {
                effective_row = row;
            } else {
                int32_t r = (int32_t)row + offset;
                effective_row = (uint32_t)(r & (int32_t)(n_rows - 1));
            }

            const uint32_t* col_ptr = trace_cols[flat_col];
            if (col_ptr == nullptr) {
                regs[dst] = {{0, 0, 0, 0}};
                break;
            }
            uint32_t val = col_ptr[effective_row];
            regs[dst] = {{val, 0, 0, 0}};
            break;
        }

        // ── M31 arithmetic (3-register format) ──────────────────────
        // [opcode:8 | dst:8 | src1:8 | src2:8]

        case OP_ADD: {
            uint32_t dst  = (word >> 16) & 0xFF;
            uint32_t src1 = (word >> 8) & 0xFF;
            uint32_t src2 = word & 0xFF;
            if (dst < MAX_REGS) {
                regs[dst] = {{m31_add(regs[src1].v[0], regs[src2].v[0]), 0, 0, 0}};
            }
            break;
        }

        case OP_SUB: {
            uint32_t dst  = (word >> 16) & 0xFF;
            uint32_t src1 = (word >> 8) & 0xFF;
            uint32_t src2 = word & 0xFF;
            if (dst < MAX_REGS) {
                regs[dst] = {{m31_sub(regs[src1].v[0], regs[src2].v[0]), 0, 0, 0}};
            }
            break;
        }

        case OP_MUL: {
            uint32_t dst  = (word >> 16) & 0xFF;
            uint32_t src1 = (word >> 8) & 0xFF;
            uint32_t src2 = word & 0xFF;
            if (dst < MAX_REGS) {
                regs[dst] = {{m31_mul(regs[src1].v[0], regs[src2].v[0]), 0, 0, 0}};
            }
            break;
        }

        case OP_NEG: {
            // [opcode:8 | dst:8 | src:8 | 0:8]
            uint32_t dst = (word >> 16) & 0xFF;
            uint32_t src = (word >> 8) & 0xFF;
            if (dst < MAX_REGS) {
                regs[dst] = {{m31_neg(regs[src].v[0]), 0, 0, 0}};
            }
            break;
        }

        case OP_ADD_CONST: {
            // [opcode:8 | dst:8 | src:8 | 0:8] + value word
            uint32_t dst = (word >> 16) & 0xFF;
            uint32_t src = (word >> 8) & 0xFF;
            uint32_t val = bytecode[pc++];
            if (dst < MAX_REGS) {
                regs[dst] = {{m31_add(regs[src].v[0], val), 0, 0, 0}};
            }
            break;
        }

        case OP_MUL_CONST: {
            // [opcode:8 | dst:8 | src:8 | 0:8] + value word
            uint32_t dst = (word >> 16) & 0xFF;
            uint32_t src = (word >> 8) & 0xFF;
            uint32_t val = bytecode[pc++];
            if (dst < MAX_REGS) {
                regs[dst] = {{m31_mul(regs[src].v[0], val), 0, 0, 0}};
            }
            break;
        }

        // ── QM31 arithmetic ────────────────────────────────────────

        case OP_WIDE_ADD: {
            uint32_t dst  = (word >> 16) & 0xFF;
            uint32_t src1 = (word >> 8) & 0xFF;
            uint32_t src2 = word & 0xFF;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_add(regs[src1], regs[src2]);
            }
            break;
        }

        case OP_WIDE_SUB: {
            uint32_t dst  = (word >> 16) & 0xFF;
            uint32_t src1 = (word >> 8) & 0xFF;
            uint32_t src2 = word & 0xFF;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_sub(regs[src1], regs[src2]);
            }
            break;
        }

        case OP_WIDE_MUL: {
            uint32_t dst  = (word >> 16) & 0xFF;
            uint32_t src1 = (word >> 8) & 0xFF;
            uint32_t src2 = word & 0xFF;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_mul(regs[src1], regs[src2]);
            }
            break;
        }

        case OP_WIDE_NEG: {
            uint32_t dst = (word >> 16) & 0xFF;
            uint32_t src = (word >> 8) & 0xFF;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_neg(regs[src]);
            }
            break;
        }

        case OP_WIDE_ADD_CONST: {
            // [opcode:8 | dst:8 | src:8 | 0:8] + 4 data words
            uint32_t dst = (word >> 16) & 0xFF;
            uint32_t src = (word >> 8) & 0xFF;
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_add(regs[src], c);
            }
            break;
        }

        case OP_WIDE_MUL_CONST: {
            uint32_t dst = (word >> 16) & 0xFF;
            uint32_t src = (word >> 8) & 0xFF;
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_mul(regs[src], c);
            }
            break;
        }

        // ── Mixed-width arithmetic ──────────────────────────────────

        case OP_WIDE_ADD_BASE: {
            // [opcode:8 | dst:8 | wide:8 | base:8]
            uint32_t dst  = (word >> 16) & 0xFF;
            uint32_t wide = (word >> 8) & 0xFF;
            uint32_t base = word & 0xFF;
            if (dst < MAX_REGS) {
                QM31 a = regs[wide];
                a.v[0] = m31_add(a.v[0], regs[base].v[0]);
                regs[dst] = a;
            }
            break;
        }

        case OP_WIDE_MUL_BASE: {
            uint32_t dst  = (word >> 16) & 0xFF;
            uint32_t wide = (word >> 8) & 0xFF;
            uint32_t base = word & 0xFF;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_mul_m31(regs[wide], regs[base].v[0]);
            }
            break;
        }

        case OP_BASE_ADD_SECURE_CONST: {
            // [opcode:8 | dst:8 | src:8 | 0:8] + 4 data words
            uint32_t dst = (word >> 16) & 0xFF;
            uint32_t src = (word >> 8) & 0xFF;
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            if (dst < MAX_REGS) {
                QM31 base_as_qm31 = {{regs[src].v[0], 0, 0, 0}};
                regs[dst] = qm31_add(base_as_qm31, c);
            }
            break;
        }

        case OP_BASE_MUL_SECURE_CONST: {
            uint32_t dst = (word >> 16) & 0xFF;
            uint32_t src = (word >> 8) & 0xFF;
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            if (dst < MAX_REGS) {
                uint32_t m = regs[src].v[0];
                regs[dst] = qm31_mul_m31(c, m);
            }
            break;
        }

        // ── Widening ────────────────────────────────────────────────

        case OP_WIDEN: {
            // [opcode:8 | dst:8 | src:8 | 0:8]
            uint32_t dst = (word >> 16) & 0xFF;
            uint32_t src = (word >> 8) & 0xFF;
            if (dst < MAX_REGS) {
                regs[dst] = {{regs[src].v[0], 0, 0, 0}};
            }
            break;
        }

        case OP_COMBINE_EF: {
            // word1: [opcode:8 | dst:8 | src0:8 | src1:8]
            // word2: [src2:8 | src3:8 | 0:16]
            uint32_t dst  = (word >> 16) & 0xFF;
            uint32_t src0 = (word >> 8) & 0xFF;
            uint32_t src1 = word & 0xFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t src2 = (word2 >> 24) & 0xFF;
            uint32_t src3 = (word2 >> 16) & 0xFF;
            if (dst < MAX_REGS) {
                regs[dst] = {{regs[src0].v[0], regs[src1].v[0], regs[src2].v[0], regs[src3].v[0]}};
            }
            break;
        }

        // ── Constraint accumulation ─────────────────────────────────

        case OP_ADD_CONSTRAINT: {
            // [opcode:8 | src:8 | 0:16]
            uint32_t src = (word >> 16) & 0xFF;
            QM31 val = regs[src];
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

    // Multiply accumulated constraint value by denom_inv.
    uint32_t trace_log_size = 31 - __clz(trace_n_rows);
    uint32_t denom = denom_inv[row >> trace_log_size];
    QM31 result = qm31_mul_m31(row_accum, denom);

    // Accumulate into output (add to existing values).
    accum0[row] = m31_add(accum0[row], result.v[0]);
    accum1[row] = m31_add(accum1[row], result.v[1]);
    accum2[row] = m31_add(accum2[row], result.v[2]);
    accum3[row] = m31_add(accum3[row], result.v[3]);
}

// ─── C wrapper for Rust FFI ─────────────────────────────────────────────

extern "C" {

void cuda_bytecode_constraint_eval(
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
    uint32_t* accum3
) {
    uint32_t threads = 256;
    uint32_t blocks = (n_rows + threads - 1) / threads;
    bytecode_constraint_eval_kernel<<<blocks, threads>>>(
        bytecode, n_words,
        trace_cols, trace_col_sizes, n_trace_cols,
        n_rows, trace_n_rows,
        random_coeff_powers, denom_inv, log_expand,
        accum0, accum1, accum2, accum3
    );
}

} // extern "C"
