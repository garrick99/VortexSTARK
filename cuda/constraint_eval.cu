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

#include <cstdio>
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
// 16-bit register indices: theoretical max 65536.
// 1024 covers most stwo-cairo components. Components with >1024 registers
// use the warp-cooperative kernel (constraint_eval_warp.cu) instead.
#define MAX_REGS 1024

// ─── Bit-reversed circle domain offset ───────────────────────────────
//
// Implements stwo's offset_bit_reversed_circle_domain_index.
// Given a row index in a bit-reversed CircleEvaluation, returns the
// index of the row that is `offset` steps away in the circle domain
// (before bit-reversal). The eval domain may be larger than the trace
// domain — `step_size` accounts for the expansion factor.
//
// This is necessary because trace data is stored in bit-reversed order.
// A simple `(row + offset) % n_rows` does NOT give the correct neighbor
// in the circle domain.
__device__ __forceinline__
uint32_t bit_reverse(uint32_t v, uint32_t log_n) {
    return __brev(v) >> (32 - log_n);
}

__device__ __forceinline__
uint32_t offset_bit_reversed_circle_domain_index(
    uint32_t i,
    uint32_t domain_log_size,   // trace log size
    uint32_t eval_log_size,     // evaluation domain log size
    int32_t offset
) {
    uint32_t prev_index = bit_reverse(i, eval_log_size);
    uint32_t half_size = 1u << (eval_log_size - 1);
    int32_t step_size = offset * (int32_t)(1u << (eval_log_size - domain_log_size - 1));

    if (prev_index < half_size) {
        // First half: add step_size mod half_size
        int32_t idx = ((int32_t)prev_index + step_size) % (int32_t)half_size;
        if (idx < 0) idx += (int32_t)half_size;
        prev_index = (uint32_t)idx;
    } else {
        // Second half: subtract step_size mod half_size, then add half_size
        int32_t idx = ((int32_t)(prev_index - half_size) - step_size) % (int32_t)half_size;
        if (idx < 0) idx += (int32_t)half_size;
        prev_index = (uint32_t)idx + half_size;
    }

    return bit_reverse(prev_index, eval_log_size);
}

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
        // All instructions: dst (or src for AddConstraint) is in low 16 bits of word1.

        switch (opcode) {

        // ── Register loads ──────────────────────────────────────────

        case OP_LOAD_CONST: {
            // word1: [opcode:8 | 0:8 | dst:16]
            // word2: [value:32]
            uint32_t dst = word & 0xFFFF;
            uint32_t val = bytecode[pc++];
            if (dst < MAX_REGS) {
                regs[dst] = {{val, 0, 0, 0}};
            }
            break;
        }

        case OP_LOAD_SECURE_CONST: {
            // word1: [opcode:8 | 0:8 | dst:16]
            // words 2-5: value[0..3]
            uint32_t dst = word & 0xFFFF;
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
            // word1: [opcode:8 | 0:8 | dst:16]
            // word2: [flat_col:32]  (after Rust remap from interaction/col_idx)
            // word3: [sign:1 | abs_offset:31]
            uint32_t dst      = word & 0xFFFF;
            uint32_t flat_col = bytecode[pc++];
            uint32_t ext      = bytecode[pc++];
            uint32_t sign     = ext >> 31;
            uint32_t abs_off  = ext & 0x7FFFFFFF;
            int32_t  offset   = sign ? -(int32_t)abs_off : (int32_t)abs_off;

            if (flat_col >= n_trace_cols || dst >= MAX_REGS) {
                return;
            }

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
            if (col_ptr == nullptr) {
                regs[dst] = {{0, 0, 0, 0}};
                break;
            }
            uint32_t val = col_ptr[effective_row];
            regs[dst] = {{val, 0, 0, 0}};
            break;
        }

        // ── M31 arithmetic (3-register format) ──────────────────────
        // word1: [opcode:8 | 0:8 | dst:16]
        // word2: [src1:16 | src2:16]

        case OP_ADD: {
            uint32_t dst  = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t src1 = word2 >> 16;
            uint32_t src2 = word2 & 0xFFFF;
            if (dst < MAX_REGS) {
                regs[dst] = {{m31_add(regs[src1].v[0], regs[src2].v[0]), 0, 0, 0}};
            }
            break;
        }

        case OP_SUB: {
            uint32_t dst  = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t src1 = word2 >> 16;
            uint32_t src2 = word2 & 0xFFFF;
            if (dst < MAX_REGS) {
                regs[dst] = {{m31_sub(regs[src1].v[0], regs[src2].v[0]), 0, 0, 0}};
            }
            break;
        }

        case OP_MUL: {
            uint32_t dst  = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t src1 = word2 >> 16;
            uint32_t src2 = word2 & 0xFFFF;
            if (dst < MAX_REGS) {
                regs[dst] = {{m31_mul(regs[src1].v[0], regs[src2].v[0]), 0, 0, 0}};
            }
            break;
        }

        case OP_NEG: {
            // word1: [opcode:8 | 0:8 | dst:16]
            // word2: [src:16 | 0:16]
            uint32_t dst  = word & 0xFFFF;
            uint32_t src  = bytecode[pc++] >> 16;
            if (dst < MAX_REGS) {
                regs[dst] = {{m31_neg(regs[src].v[0]), 0, 0, 0}};
            }
            break;
        }

        case OP_ADD_CONST: {
            // word1: [opcode:8 | 0:8 | dst:16]
            // word2: [src:16 | 0:16]
            // word3: [value:32]
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            uint32_t val = bytecode[pc++];
            if (dst < MAX_REGS) {
                regs[dst] = {{m31_add(regs[src].v[0], val), 0, 0, 0}};
            }
            break;
        }

        case OP_MUL_CONST: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            uint32_t val = bytecode[pc++];
            if (dst < MAX_REGS) {
                regs[dst] = {{m31_mul(regs[src].v[0], val), 0, 0, 0}};
            }
            break;
        }

        // ── QM31 arithmetic ────────────────────────────────────────

        case OP_WIDE_ADD: {
            uint32_t dst  = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t src1 = word2 >> 16;
            uint32_t src2 = word2 & 0xFFFF;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_add(regs[src1], regs[src2]);
            }
            break;
        }

        case OP_WIDE_SUB: {
            uint32_t dst  = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t src1 = word2 >> 16;
            uint32_t src2 = word2 & 0xFFFF;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_sub(regs[src1], regs[src2]);
            }
            break;
        }

        case OP_WIDE_MUL: {
            uint32_t dst  = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t src1 = word2 >> 16;
            uint32_t src2 = word2 & 0xFFFF;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_mul(regs[src1], regs[src2]);
            }
            break;
        }

        case OP_WIDE_NEG: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_neg(regs[src]);
            }
            break;
        }

        case OP_WIDE_ADD_CONST: {
            // word1: header, word2: [src:16|0:16], words 3-6: value
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_add(regs[src], c);
            }
            break;
        }

        case OP_WIDE_MUL_CONST: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_mul(regs[src], c);
            }
            break;
        }

        // ── Mixed-width arithmetic ──────────────────────────────────

        case OP_WIDE_ADD_BASE: {
            // word1: [opcode:8 | 0:8 | dst:16]
            // word2: [wide:16 | base:16]
            uint32_t dst   = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t wide  = word2 >> 16;
            uint32_t base  = word2 & 0xFFFF;
            if (dst < MAX_REGS) {
                QM31 a = regs[wide];
                a.v[0] = m31_add(a.v[0], regs[base].v[0]);
                regs[dst] = a;
            }
            break;
        }

        case OP_WIDE_MUL_BASE: {
            uint32_t dst   = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t wide  = word2 >> 16;
            uint32_t base  = word2 & 0xFFFF;
            if (dst < MAX_REGS) {
                regs[dst] = qm31_mul_m31(regs[wide], regs[base].v[0]);
            }
            break;
        }

        case OP_BASE_ADD_SECURE_CONST: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            QM31 c = {{bytecode[pc], bytecode[pc+1], bytecode[pc+2], bytecode[pc+3]}};
            pc += 4;
            if (dst < MAX_REGS) {
                QM31 base_as_qm31 = {{regs[src].v[0], 0, 0, 0}};
                regs[dst] = qm31_add(base_as_qm31, c);
            }
            break;
        }

        case OP_BASE_MUL_SECURE_CONST: {
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
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
            // word1: [opcode:8 | 0:8 | dst:16]
            // word2: [src:16 | 0:16]
            uint32_t dst = word & 0xFFFF;
            uint32_t src = bytecode[pc++] >> 16;
            if (dst < MAX_REGS) {
                regs[dst] = {{regs[src].v[0], 0, 0, 0}};
            }
            break;
        }

        case OP_COMBINE_EF: {
            // word1: [opcode:8 | 0:8 | dst:16]
            // word2: [src0:16 | src1:16]
            // word3: [src2:16 | src3:16]
            uint32_t dst   = word & 0xFFFF;
            uint32_t word2 = bytecode[pc++];
            uint32_t word3 = bytecode[pc++];
            uint32_t src0  = word2 >> 16;
            uint32_t src1  = word2 & 0xFFFF;
            uint32_t src2  = word3 >> 16;
            uint32_t src3  = word3 & 0xFFFF;
            if (dst < MAX_REGS) {
                regs[dst] = {{regs[src0].v[0], regs[src1].v[0], regs[src2].v[0], regs[src3].v[0]}};
            }
            break;
        }

        // ── Constraint accumulation ─────────────────────────────────

        case OP_ADD_CONSTRAINT: {
            // word1: [opcode:8 | 0:8 | src:16]
            uint32_t src = word & 0xFFFF;
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
    uint32_t* accum3,
    uint32_t n_registers
) {
    // Adaptive block size: fewer threads for high-register components
    // to reduce local memory spilling. Each register = 16 bytes (QM31).
    // RTX 5090: 256KB register file per SM, ~128KB L1 for local mem.
    uint32_t threads;
    if (n_registers <= 256)       threads = 256;  // 4KB/thread → fits in registers
    else if (n_registers <= 512)  threads = 128;  // 8KB/thread
    else if (n_registers <= 1024) threads = 64;   // 16KB/thread
    else if (n_registers <= 2048) threads = 32;   // 32KB/thread
    else                          threads = 32;   // >32KB → heavy spill, minimize threads

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
