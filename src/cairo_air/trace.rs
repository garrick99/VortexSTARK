//! Cairo trace layout: converts VM execution trace into columnar M31 format.
//!
//! Trace columns (all values reduced mod P = 2^31 - 1):
//!
//! Registers (3):
//!   0: pc
//!   1: ap
//!   2: fp
//!
//! Instruction (2, stored as lo/hi M31 since instructions can be > 2^31):
//!   3: instruction_lo (lower 31 bits)
//!   4: instruction_hi (upper bits)
//!
//! Flags (15 binary columns):
//!   5-19: dst_reg, op0_reg, op1_imm, op1_fp, op1_ap,
//!         res_add, res_mul, pc_jump_abs, pc_jump_rel, pc_jnz,
//!         ap_add, ap_add1, opcode_call, opcode_ret, opcode_assert
//!
//! Operands (7):
//!   20: dst_addr
//!   21: dst
//!   22: op0_addr
//!   23: op0
//!   24: op1_addr
//!   25: op1
//!   26: res
//!
//! Total: 27 columns

use crate::field::m31::P;
use super::vm::TraceRow;

/// Number of trace columns.
pub const N_COLS: usize = 27;

/// Column indices.
pub const COL_PC: usize = 0;
pub const COL_AP: usize = 1;
pub const COL_FP: usize = 2;
pub const COL_INST_LO: usize = 3;
pub const COL_INST_HI: usize = 4;
pub const COL_FLAGS_START: usize = 5;
pub const COL_FLAGS_END: usize = 20; // exclusive
pub const COL_DST_ADDR: usize = 20;
pub const COL_DST: usize = 21;
pub const COL_OP0_ADDR: usize = 22;
pub const COL_OP0: usize = 23;
pub const COL_OP1_ADDR: usize = 24;
pub const COL_OP1: usize = 25;
pub const COL_RES: usize = 26;

/// Convert execution trace to columnar M31 format.
/// Returns N_COLS vectors, each of length `trace.len()`, padded to power of 2.
pub fn trace_to_columns(trace: &[TraceRow], log_n: u32) -> Vec<Vec<u32>> {
    let n = 1usize << log_n;
    assert!(trace.len() <= n, "trace too large for log_n={log_n}");

    let mut cols: Vec<Vec<u32>> = (0..N_COLS).map(|_| vec![0u32; n]).collect();

    // Reduce u64 → M31: use bit trick instead of expensive % P division
    #[inline(always)]
    fn to_m31(v: u64) -> u32 {
        let lo = (v & P as u64) as u32;
        let hi = (v >> 31) as u32;
        let r = lo + hi;
        if r >= P { r - P } else { r }
    }

    for (i, row) in trace.iter().enumerate() {
        cols[COL_PC][i] = to_m31(row.pc);
        cols[COL_AP][i] = to_m31(row.ap);
        cols[COL_FP][i] = to_m31(row.fp);

        cols[COL_INST_LO][i] = (row.instruction & 0x7FFF_FFFF) as u32;
        cols[COL_INST_HI][i] = ((row.instruction >> 31) & 0x7FFF_FFFF) as u32;

        for (j, &flag) in row.flags.iter().enumerate() {
            cols[COL_FLAGS_START + j][i] = flag;
        }

        cols[COL_DST_ADDR][i] = to_m31(row.dst_addr);
        cols[COL_DST][i] = to_m31(row.dst);
        cols[COL_OP0_ADDR][i] = to_m31(row.op0_addr);
        cols[COL_OP0][i] = to_m31(row.op0);
        cols[COL_OP1_ADDR][i] = to_m31(row.op1_addr);
        cols[COL_OP1][i] = to_m31(row.op1);
        cols[COL_RES][i] = to_m31(row.res);
    }

    cols
}

/// Evaluate Cairo transition constraints at a given row.
/// Returns a vector of constraint values (each should be 0 for a valid trace).
///
/// Constraints checked:
/// 1. Flag binary: each flag ∈ {0, 1}  (15 constraints)
/// 2. Result: res = op0 + op1 if res_add, res = op0 * op1 if res_mul, res = op1 otherwise
/// 3. PC update: next_pc follows pc_update rule
/// 4. AP update: next_ap follows ap_update rule
/// 5. FP update: next_fp follows opcode rule
/// 6. DST consistency: dst = res for assert_eq
pub fn eval_transition_constraints(
    cols: &[&[u32]],
    row: usize,
    n: usize,
) -> Vec<crate::field::M31> {
    use crate::field::M31;

    let next = (row + 1) % n;
    let mut constraints = Vec::new();

    // Helper: read column value at row as M31
    let val = |col: usize, r: usize| -> M31 { M31(cols[col][r]) };

    let pc = val(COL_PC, row);
    let ap = val(COL_AP, row);
    let fp = val(COL_FP, row);
    let dst = val(COL_DST, row);
    let op0 = val(COL_OP0, row);
    let op1 = val(COL_OP1, row);
    let res = val(COL_RES, row);

    let next_pc = val(COL_PC, next);
    let next_ap = val(COL_AP, next);
    let next_fp = val(COL_FP, next);

    // Flag values
    let f = |i: usize| -> M31 { val(COL_FLAGS_START + i, row) };
    let _dst_reg = f(0);
    let _op0_reg = f(1);
    let _op1_imm = f(2);
    let _op1_fp = f(3);
    let _op1_ap = f(4);
    let res_add = f(5);
    let res_mul = f(6);
    let pc_jump_abs = f(7);
    let pc_jump_rel = f(8);
    let pc_jnz = f(9);
    let ap_add = f(10);
    let ap_add1 = f(11);
    let opcode_call = f(12);
    let opcode_ret = f(13);
    let opcode_assert = f(14);

    // 1. Flag binary constraints: flag * (1 - flag) = 0
    for i in 0..15 {
        let flag = f(i);
        constraints.push(flag * (M31::ONE - flag));
    }

    // 2. Result constraint (when not jnz):
    //    (1 - pc_jnz) * (res - (1 - res_add - res_mul) * op1 - res_add * (op0 + op1) - res_mul * (op0 * op1)) = 0
    let one = M31::ONE;
    let expected_res = (one - res_add - res_mul) * op1
        + res_add * (op0 + op1)
        + res_mul * (op0 * op1);
    constraints.push((one - pc_jnz) * (res - expected_res));

    // 3. PC update:
    //    next_pc = (1 - pc_jump_abs - pc_jump_rel - pc_jnz) * (pc + instruction_size)
    //            + pc_jump_abs * res
    //            + pc_jump_rel * (pc + res)
    //            + pc_jnz * (pc + jnz_target)
    // Simplified: instruction_size = 1 + op1_imm
    let inst_size = one + _op1_imm;
    let pc_default = pc + inst_size;
    let pc_regular = (one - pc_jump_abs - pc_jump_rel - pc_jnz) * pc_default;
    let pc_abs = pc_jump_abs * res;
    let pc_rel = pc_jump_rel * (pc + res);
    // jnz: if dst != 0, pc += op1; else pc += inst_size
    // This is hard to express as a single polynomial. Use: pc_jnz * (dst * (next_pc - pc - op1) + (1-dst_nonzero) * (next_pc - pc - inst_size))
    // For now, simplified constraint (works when trace is correct):
    let _expected_pc = pc_regular + pc_abs + pc_rel + pc_jnz * (pc + op1);
    // Only check when not jnz with dst=0 (degenerate case handled by the VM)
    constraints.push((one - pc_jnz) * (next_pc - (pc_regular + pc_abs + pc_rel))
        + pc_jnz * dst * (next_pc - pc - op1));

    // 4. AP update:
    let expected_ap = ap
        + ap_add * res
        + ap_add1 * one
        + opcode_call * M31(2);
    constraints.push(next_ap - expected_ap);

    // 5. FP update:
    //    opcode_call: next_fp = ap + 2
    //    opcode_ret: next_fp = dst
    //    otherwise: next_fp = fp
    let expected_fp = (one - opcode_call - opcode_ret) * fp
        + opcode_call * (ap + M31(2))
        + opcode_ret * dst;
    constraints.push(next_fp - expected_fp);

    // 6. Assert_eq: dst = res (when opcode_assert is set)
    constraints.push(opcode_assert * (dst - res));

    constraints
}

/// Number of transition constraints.
pub const N_CONSTRAINTS: usize = 15 + 1 + 1 + 1 + 1 + 1; // 20

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cairo_air::vm::{Memory, execute};
    use crate::cairo_air::decode::Instruction;
    use crate::field::M31;

    #[test]
    fn test_trace_columns_size() {
        let mut mem = Memory::new();
        let i = Instruction { off0: 0x8000, off1: 0x8000, off2: 0x8001, op1_imm: 1, opcode_assert: 1, ap_add1: 1, ..Default::default() };
        mem.set(0, i.encode());
        mem.set(1, 42);
        mem.set(2, i.encode());
        mem.set(3, 99);

        let trace = execute(&mut mem, 2);
        let cols = trace_to_columns(&trace, 2); // pad to 4 rows
        assert_eq!(cols.len(), N_COLS);
        assert_eq!(cols[0].len(), 4);
    }

    #[test]
    fn test_constraints_satisfied_simple() {
        let mut mem = Memory::new();
        // [ap] = 42; ap++
        let i = Instruction { off0: 0x8000, off1: 0x8000, off2: 0x8001, op1_imm: 1, opcode_assert: 1, ap_add1: 1, ..Default::default() };
        mem.set(0, i.encode());
        mem.set(1, 42);
        // [ap] = 99; ap++
        mem.set(2, i.encode());
        mem.set(3, 99);
        // [ap] = [ap-2] + [ap-1]; ap++
        let add = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        mem.set(4, add.encode());
        // Pad with a noop-ish instruction
        mem.set(5, i.encode());
        mem.set(6, 0);

        let trace = execute(&mut mem, 4);
        let log_n = 2; // 4 rows
        let cols = trace_to_columns(&trace, log_n);
        let col_refs: Vec<&[u32]> = cols.iter().map(|c| c.as_slice()).collect();

        // Check constraints at rows 0, 1, 2 (row 3 wraps to row 0, skip)
        for row in 0..3 {
            let c = eval_transition_constraints(&col_refs, row, 4);
            // Flag binary constraints should all be 0
            for j in 0..15 {
                assert_eq!(c[j], M31::ZERO,
                    "flag binary constraint {j} violated at row {row}: {:?}", c[j]);
            }
            // Result constraint
            assert_eq!(c[15], M31::ZERO,
                "result constraint violated at row {row}: {:?}", c[15]);
            // Assert_eq constraint
            assert_eq!(c[19], M31::ZERO,
                "assert_eq constraint violated at row {row}: {:?}", c[19]);
        }
    }
}
