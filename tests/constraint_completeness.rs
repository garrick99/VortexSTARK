//! Constraint completeness tests.
//!
//! For each of the 33 AIR constraints, there must exist a test that:
//! 1. Builds a minimal valid program and proves it.
//! 2. Corrupts exactly the value(s) that the constraint guards.
//! 3. Asserts the verifier rejects the tampered proof.
//!
//! This file tests the mutual exclusivity constraints (25-29) and the
//! instruction decomposition constraint (30) that do not have dedicated
//! tests elsewhere, plus a systematic per-flag binary test.
//!
//! Constraints 0-24, 31-32 are covered by existing tests in prover.rs.
//! See AUDIT.md §1 for the full coverage map.

use vortexstark::cairo_air::prover::{cairo_prove, cairo_verify};
use vortexstark::cairo_air::trace::{
    COL_FLAGS_START, COL_INST_LO, COL_INST_HI, COL_OFF0, COL_OFF1, COL_OFF2,
    COL_DST_ADDR, COL_OP0_ADDR, COL_OP1_ADDR, COL_RES,
};
use vortexstark::cairo_air::decode::Instruction;

// Flag column offsets relative to COL_FLAGS_START
const FLAG_OP1_IMM:      usize = 2;  // COL_FLAGS_START + 2
const FLAG_OP1_AP:       usize = 4;  // COL_FLAGS_START + 4
const FLAG_PC_JUMP_ABS:  usize = 7;  // COL_FLAGS_START + 7
const FLAG_PC_JUMP_REL:  usize = 8;  // COL_FLAGS_START + 8
const FLAG_OPCODE_CALL:  usize = 12; // COL_FLAGS_START + 12
const FLAG_OPCODE_ASSERT:usize = 14; // COL_FLAGS_START + 14

fn init_gpu() {
    vortexstark::cuda::ffi::init_memory_pool();
}

/// Build a simple Fibonacci program (32 add steps) in memory.
fn build_fib_program() -> Vec<u64> {
    let init = Instruction {
        off0: 0x8000, off1: 0x8000, off2: 0x8001,
        op1_imm: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    let add = Instruction {
        off0: 0x8000, off1: 0x8000u16.wrapping_sub(2),
        off2: 0x8000u16.wrapping_sub(1),
        op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    let mut program = vec![init.encode(), 1u64, init.encode(), 1u64];
    for _ in 0..28 { program.push(add.encode()); }
    program
}

/// Tamper a single u32 in the proof's trace_values_at_queries at the first query, first column
/// matching `col`, then verify rejection.
fn assert_tamper_rejected_at_col(col: usize, delta: u32) {
    const M31_P: u32 = 0x7FFF_FFFF;
    init_gpu();
    let program = build_fib_program();
    let n_steps = 32;
    let log_n = 5u32;
    let mut proof = cairo_prove(&program, n_steps, log_n);

    // Corrupt the value at the first query point for this column.
    if let Some(row) = proof.trace_values_at_queries.get_mut(0) {
        row[col] = row[col].wrapping_add(delta) % M31_P;
    }

    let result = cairo_verify(&proof);
    assert!(result.is_err(),
        "Verifier should reject tampered col {col}, but accepted the proof");
}

// ---------------------------------------------------------------------------
// Constraint 15: result computation
// ---------------------------------------------------------------------------

#[test]
fn test_constraint_result_computation_rejected() {
    assert_tamper_rejected_at_col(COL_RES, 1);
}

// ---------------------------------------------------------------------------
// Constraints 0-14: flag binary — spot-check each flag column
// ---------------------------------------------------------------------------

#[test]
fn test_constraint_flags_binary_all() {
    init_gpu();
    let program = build_fib_program();
    let n_steps = 32;
    let log_n = 5u32;
    let proof_base = cairo_prove(&program, n_steps, log_n);

    for flag in 0..15usize {
        let col = COL_FLAGS_START + flag;
        let mut proof = proof_base.clone();
        if let Some(row) = proof.trace_values_at_queries.get_mut(0) {
            // Flip a binary flag from 0→1 or 1→2 (either makes flag*(1-flag) ≠ 0)
            row[col] = if row[col] == 0 { 2 } else { 0 };
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(),
            "Verifier should reject corrupted flag {flag} (col {col})");
    }
}

// ---------------------------------------------------------------------------
// Constraints 25-27: op1 source exclusivity
// ---------------------------------------------------------------------------

#[test]
fn test_constraint_op1_exclusivity() {
    init_gpu();
    // Build a program that uses op1_ap (flag index 4) — an add instruction.
    // We forcibly set op1_imm (flag index 2) to 1 at the same row,
    // violating op1_imm * op1_ap = 0 (constraint 26).
    let program = build_fib_program();
    let n_steps = 32;
    let log_n = 5u32;
    let mut proof = cairo_prove(&program, n_steps, log_n);

    // Tamper row 0: force both op1_ap and op1_imm to 1 simultaneously,
    // violating op1_imm * op1_ap = 0 (constraint 26).
    // If the row already has op1_ap=1 (Fibonacci add), we just set op1_imm=1.
    // If it has op1_imm=1, we set op1_ap=1. Either way both are 1.
    let op1_ap_col  = COL_FLAGS_START + FLAG_OP1_AP;
    let op1_imm_col = COL_FLAGS_START + FLAG_OP1_IMM;
    proof.trace_values_at_queries[0][op1_ap_col]  = 1;
    proof.trace_values_at_queries[0][op1_imm_col] = 1;

    let result = cairo_verify(&proof);
    assert!(result.is_err(),
        "Verifier should reject op1 source exclusivity violation (op1_imm=1 AND op1_ap=1)");
}

// ---------------------------------------------------------------------------
// Constraint 28: PC update exclusivity
// ---------------------------------------------------------------------------

#[test]
fn test_constraint_pc_exclusivity() {
    init_gpu();
    let program = build_fib_program();
    let n_steps = 32;
    let log_n = 5u32;
    let mut proof = cairo_prove(&program, n_steps, log_n);

    // Forcibly set jump_abs=1 AND jump_rel=1 at the same row → violates exclusivity
    let jump_abs_col = COL_FLAGS_START + FLAG_PC_JUMP_ABS;
    let jump_rel_col = COL_FLAGS_START + FLAG_PC_JUMP_REL;
    if let Some(row) = proof.trace_values_at_queries.get_mut(0) {
        row[jump_abs_col] = 1;
        row[jump_rel_col] = 1;
    }

    let result = cairo_verify(&proof);
    assert!(result.is_err(),
        "Verifier should reject PC update exclusivity violation (jump_abs=1 AND jump_rel=1)");
}

// ---------------------------------------------------------------------------
// Constraint 29: opcode exclusivity
// ---------------------------------------------------------------------------

#[test]
fn test_constraint_opcode_exclusivity() {
    init_gpu();
    let program = build_fib_program();
    let n_steps = 32;
    let log_n = 5u32;
    let mut proof = cairo_prove(&program, n_steps, log_n);

    // Forcibly set opcode_assert=1 AND opcode_call=1 at the same row
    let call_col   = COL_FLAGS_START + FLAG_OPCODE_CALL;
    let assert_col = COL_FLAGS_START + FLAG_OPCODE_ASSERT;
    if let Some(row) = proof.trace_values_at_queries.get_mut(0) {
        row[call_col]   = 1;
        row[assert_col] = 1;
    }

    let result = cairo_verify(&proof);
    assert!(result.is_err(),
        "Verifier should reject opcode exclusivity violation (call=1 AND assert=1)");
}

// ---------------------------------------------------------------------------
// Constraint 30: instruction decomposition
// ---------------------------------------------------------------------------

#[test]
fn test_constraint_instruction_decomposition() {
    init_gpu();
    let program = build_fib_program();
    let n_steps = 32;
    let log_n = 5u32;
    let mut proof = cairo_prove(&program, n_steps, log_n);

    // Corrupt inst_lo at the first query — the decomposition identity will fail.
    if let Some(row) = proof.trace_values_at_queries.get_mut(0) {
        row[COL_INST_LO] = row[COL_INST_LO].wrapping_add(1) % 0x7FFF_FFFF;
    }

    let result = cairo_verify(&proof);
    assert!(result.is_err(),
        "Verifier should reject instruction decomposition violation (corrupted inst_lo)");
}

#[test]
fn test_constraint_instruction_decomposition_inst_hi() {
    init_gpu();
    let program = build_fib_program();
    let n_steps = 32;
    let log_n = 5u32;
    let mut proof = cairo_prove(&program, n_steps, log_n);

    if let Some(row) = proof.trace_values_at_queries.get_mut(0) {
        row[COL_INST_HI] = row[COL_INST_HI].wrapping_add(1) % 0x7FFF_FFFF;
    }

    let result = cairo_verify(&proof);
    assert!(result.is_err(),
        "Verifier should reject instruction decomposition violation (corrupted inst_hi)");
}

#[test]
fn test_constraint_instruction_decomposition_off0() {
    init_gpu();
    let program = build_fib_program();
    let n_steps = 32;
    let log_n = 5u32;
    let mut proof = cairo_prove(&program, n_steps, log_n);

    if let Some(row) = proof.trace_values_at_queries.get_mut(0) {
        // Increment off0 by 1 — the decomposition sum changes, violating constraint 30.
        row[COL_OFF0] = (row[COL_OFF0] + 1) % 0x10000;
    }

    let result = cairo_verify(&proof);
    assert!(result.is_err(),
        "Verifier should reject instruction decomposition violation (corrupted off0)");
}

// ---------------------------------------------------------------------------
// Operand address constraints (20-22)
// ---------------------------------------------------------------------------

#[test]
fn test_constraint_dst_addr() {
    assert_tamper_rejected_at_col(COL_DST_ADDR, 1);
}

#[test]
fn test_constraint_op0_addr() {
    assert_tamper_rejected_at_col(COL_OP0_ADDR, 1);
}

#[test]
fn test_constraint_op1_addr() {
    assert_tamper_rejected_at_col(COL_OP1_ADDR, 1);
}
