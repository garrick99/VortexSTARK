// GPU bytecode interpreter for generic constraint evaluation.
//
// Replays a flat bytecode program (recorded by TracingEvalAtRow) per row in parallel.
// Each thread evaluates all constraints for one row of the evaluation domain.
//
// The bytecode is the same for every row — only trace column lookups differ.
// This is a "virtual machine" approach: no per-AIR kernel compilation needed.

#include "include/m31.cuh"
#include "include/cm31.cuh"
#include "include/qm31.cuh"

// ─── Opcodes (must match BytecodeOp::opcode_id() in bytecode.rs) ────────

#define OP_PUSH_BASE_FIELD      0x01
#define OP_PUSH_SECURE_FIELD    0x02
#define OP_PUSH_TRACE_VAL       0x03

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

// Extended operand flag: bit 23 set in the operand field means next word has full value
#define EXTENDED_FLAG           (1u << 23)

// Maximum stack depth. Logup-heavy components with many fraction
// cross-multiplications can reach deep stacks. 256 covers the
// largest stwo-cairo components.
#define MAX_STACK_DEPTH 256

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

    // QM31 stack. M31 values stored as (val, 0, 0, 0).
    QM31 stack[MAX_STACK_DEPTH];
    int sp = 0;

    QM31 row_accum = qm31_zero();
    int constraint_idx = 0;

    uint32_t pc = 0;
    while (pc < n_words) {
        uint32_t word = bytecode[pc++];
        uint32_t opcode = word >> 24;
        uint32_t operand = word & 0xFFFFFF;

        // Stack overflow/underflow protection
        if (sp >= MAX_STACK_DEPTH - 4 || sp < 0) return;

        switch (opcode) {

        // ── Stack loads ──────────────────────────────────────────────

        case OP_PUSH_BASE_FIELD: {
            uint32_t val;
            if (operand & EXTENDED_FLAG) {
                val = bytecode[pc++];
            } else {
                val = operand;
            }
            stack[sp++] = {{val, 0, 0, 0}};
            break;
        }

        case OP_PUSH_SECURE_FIELD: {
            uint32_t a = bytecode[pc++];
            uint32_t b = bytecode[pc++];
            uint32_t c = bytecode[pc++];
            uint32_t d = bytecode[pc++];
            stack[sp++] = {{a, b, c, d}};
            break;
        }

        case OP_PUSH_TRACE_VAL: {
            // After Rust remapping: operand = flat_col:14 | sign:1 | abs_offset:9
            // Supports up to 16383 columns and offsets -511..+511.
            uint32_t flat_col = (operand >> 10) & 0x3FFF;
            uint32_t sign = (operand >> 9) & 1;
            uint32_t abs_offset = operand & 0x1FF;
            int32_t offset = sign ? -(int32_t)abs_offset : (int32_t)abs_offset;

            if (flat_col >= n_trace_cols) {
                // Bounds error — return to prevent crash.
                return;
            }

            uint32_t effective_row;
            if (offset == 0) {
                effective_row = row;
            } else {
                // Wrap within evaluation domain size (power of 2)
                int32_t r = (int32_t)row + offset;
                effective_row = (uint32_t)(r & (int32_t)(n_rows - 1));
            }

            const uint32_t* col_ptr = trace_cols[flat_col];
            if (col_ptr == nullptr) {
                stack[sp++] = {{0, 0, 0, 0}};
                break;
            }
            uint32_t val = col_ptr[effective_row];
            stack[sp++] = {{val, 0, 0, 0}};
            break;
        }

        // ── M31 arithmetic ───────────────────────────────────────────

        case OP_ADD: {
            QM31 b = stack[--sp];
            QM31 a = stack[--sp];
            stack[sp++] = {{m31_add(a.v[0], b.v[0]), 0, 0, 0}};
            break;
        }

        case OP_SUB: {
            QM31 b = stack[--sp];
            QM31 a = stack[--sp];
            stack[sp++] = {{m31_sub(a.v[0], b.v[0]), 0, 0, 0}};
            break;
        }

        case OP_MUL: {
            QM31 b = stack[--sp];
            QM31 a = stack[--sp];
            stack[sp++] = {{m31_mul(a.v[0], b.v[0]), 0, 0, 0}};
            break;
        }

        case OP_NEG: {
            stack[sp-1].v[0] = m31_neg(stack[sp-1].v[0]);
            break;
        }

        case OP_ADD_CONST: {
            uint32_t val;
            if (operand & EXTENDED_FLAG) {
                val = bytecode[pc++];
            } else {
                val = operand;
            }
            stack[sp-1].v[0] = m31_add(stack[sp-1].v[0], val);
            break;
        }

        case OP_MUL_CONST: {
            uint32_t val;
            if (operand & EXTENDED_FLAG) {
                val = bytecode[pc++];
            } else {
                val = operand;
            }
            stack[sp-1].v[0] = m31_mul(stack[sp-1].v[0], val);
            break;
        }

        // ── QM31 arithmetic ─────────────────────────────────────────

        case OP_WIDE_ADD: {
            QM31 b = stack[--sp];
            QM31 a = stack[--sp];
            stack[sp++] = qm31_add(a, b);
            break;
        }

        case OP_WIDE_SUB: {
            QM31 b = stack[--sp];
            QM31 a = stack[--sp];
            stack[sp++] = qm31_sub(a, b);
            break;
        }

        case OP_WIDE_MUL: {
            QM31 b = stack[--sp];
            QM31 a = stack[--sp];
            stack[sp++] = qm31_mul(a, b);
            break;
        }

        case OP_WIDE_NEG: {
            stack[sp-1] = qm31_neg(stack[sp-1]);
            break;
        }

        case OP_WIDE_ADD_CONST: {
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            stack[sp-1] = qm31_add(stack[sp-1], c);
            break;
        }

        case OP_WIDE_MUL_CONST: {
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            stack[sp-1] = qm31_mul(stack[sp-1], c);
            break;
        }

        // ── Mixed-width arithmetic ───────────────────────────────────

        case OP_WIDE_ADD_BASE: {
            // Stack: [..., QM31_a, M31_b] -> [..., QM31_a + widen(M31_b)]
            QM31 b = stack[--sp];
            QM31 a = stack[--sp];
            a.v[0] = m31_add(a.v[0], b.v[0]);
            stack[sp++] = a;
            break;
        }

        case OP_WIDE_MUL_BASE: {
            // Stack: [..., QM31_a, M31_b] -> [..., QM31_a * M31_b]
            QM31 b = stack[--sp];
            QM31 a = stack[--sp];
            stack[sp++] = qm31_mul_m31(a, b.v[0]);
            break;
        }

        case OP_BASE_ADD_SECURE_CONST: {
            // Pop M31, add QM31 constant, push QM31
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            QM31 base_as_qm31 = {{stack[sp-1].v[0], 0, 0, 0}};
            stack[sp-1] = qm31_add(base_as_qm31, c);
            break;
        }

        case OP_BASE_MUL_SECURE_CONST: {
            // Pop M31, multiply by QM31 constant, push QM31
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            uint32_t m = stack[sp-1].v[0];
            stack[sp-1] = qm31_mul_m31(c, m);
            break;
        }

        // ── Widening ─────────────────────────────────────────────────

        case OP_WIDEN: {
            // M31 -> QM31: already stored as (v, 0, 0, 0), no-op on unified stack.
            break;
        }

        case OP_COMBINE_EF: {
            // Pop 4 M31 values, combine into 1 QM31.
            // Stack order: [v0, v1, v2, v3] with v3 on top.
            QM31 v3 = stack[--sp];
            QM31 v2 = stack[--sp];
            QM31 v1 = stack[--sp];
            QM31 v0 = stack[--sp];
            stack[sp++] = {{v0.v[0], v1.v[0], v2.v[0], v3.v[0]}};
            break;
        }

        // ── Constraint accumulation ──────────────────────────────────

        case OP_ADD_CONSTRAINT: {
            if (sp <= 0) return;
            QM31 val = stack[--sp];
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
    // denom_inv is indexed by row >> trace_log_size (bit-reversed).
    // Compute trace_log_size from trace_n_rows (always a power of 2).
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
