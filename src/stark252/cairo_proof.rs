//! Integrated Cairo STARK proof over Stark252.
//!
//! Combines:
//! 1. Execution STARK (`prove_multi`) — transition + instruction constraints
//! 2. Memory LogUp — proves all (addr, value) reads are consistent
//! 3. Range-check LogUp — proves all instruction offsets ∈ [0, 2^16)
//!
//! # Fiat-Shamir continuity
//!
//! The three proofs share a single transcript channel:
//!
//! ```text
//! channel = Channel252::new()
//! mix(public_inputs)
//! mix(col_roots)                  ← execution trace commitment
//! draw(alphas)                    ← execution constraint combination
//! mix(quotient_root)              ← execution quotient
//! [FRI for execution quotient]
//! draw(query_indices_exec)
//!                                 ── after execution proof ──
//! draw(z_mem, alpha_mem)          ← memory LogUp challenges
//! mix(s_mem_root)
//! mix(q_mem_root)
//! [FRI for memory quotient]
//! draw(query_indices_mem)
//!                                 ── after memory proof ──
//! draw(z_rc)                      ← range-check challenge
//! mix(s_rc_root)
//! mix(q_rc_root)
//! [FRI for range-check quotient]
//! draw(query_indices_rc)
//! ```
//!
//! This transcript structure ensures that each sub-proof's challenges are bound
//! to all prior commitments, preventing cross-proof malleability.

use serde::{Serialize, Deserialize};
use super::field::{Fp, fp_to_u32x8, fp_from_u32x8, Channel252};
use super::multi_stark::{MultiProof, prove_multi, verify_multi};
use super::logup::{MemoryLogupProof, prove_memory_logup, verify_memory_logup};
use super::range_check::{RangeCheckProof, prove_range_check, verify_range_check};
use super::cairo_air::CairoAir252;

// ─────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────

/// A complete Cairo STARK proof over Stark252.
///
/// Combines execution constraints, memory consistency, and range checks
/// into a single verifiable proof object.
#[derive(Clone, Serialize, Deserialize)]
pub struct CairoProof {
    /// Execution constraint proof (multi-column STARK).
    pub exec_proof:   MultiProof,
    /// Memory consistency proof (LogUp argument).
    pub memory_proof: MemoryLogupProof,
    /// Range-check proof (offsets ∈ [0, 2^16)).
    pub rc_proof:     RangeCheckProof,
    /// log₂ of the trace length.
    pub log_n:        u32,
}

// ─────────────────────────────────────────────
// Prover
// ─────────────────────────────────────────────

/// Prove a Cairo program execution end-to-end.
///
/// `air`   — CairoAir252 carrying the execution trace (built via `from_vm`).
/// `log_n` — log₂ of the number of execution steps (must equal `air.trace.n_rows`).
///
/// Returns a `CairoProof` containing all three sub-proofs on a shared channel.
pub fn prove_cairo(air: &CairoAir252, log_n: u32) -> CairoProof {
    // ── Round 1: Execution STARK ──────────────
    let exec_proof = prove_multi(air, log_n);

    // ── Reconstruct channel state ─────────────
    // The execution STARK prover consumed the channel up to the query-index draw.
    // We replay it here so the memory and range-check draws are properly chained.
    let mut channel = Channel252::new();
    replay_exec_channel(&exec_proof, air, log_n, &mut channel);

    // ── Round 2: Memory LogUp ─────────────────
    let trace_cols = air.trace.columns.clone();
    let memory_proof = prove_memory_logup(&trace_cols, log_n, &mut channel);

    // ── Round 3: Range-check LogUp ────────────
    let rc_proof = prove_range_check(&trace_cols, log_n, &mut channel);

    CairoProof { exec_proof, memory_proof, rc_proof, log_n }
}

// ─────────────────────────────────────────────
// Verifier
// ─────────────────────────────────────────────

/// Verify a `CairoProof`.
///
/// Checks all three sub-proofs on the shared channel transcript.
pub fn verify_cairo(proof: &CairoProof, air: &CairoAir252) -> Result<(), String> {
    let log_n = proof.log_n;

    // ── Round 1: Execution STARK ──────────────
    verify_multi(&proof.exec_proof, air)?;

    // ── Reconstruct channel state ─────────────
    let mut channel = Channel252::new();
    replay_exec_channel(&proof.exec_proof, air, log_n, &mut channel);

    // ── Round 2: Memory LogUp ─────────────────
    verify_memory_logup(&proof.memory_proof, log_n, &mut channel)?;

    // ── Round 3: Range-check ──────────────────
    // For range-check, we can supply the instruction column root from the exec proof
    // to enable full auth-path verification.
    let inst_col_idx = 3; // INST column index in Cairo trace
    let inst_root    = &proof.exec_proof.col_roots[inst_col_idx];
    verify_range_check(&proof.rc_proof, log_n, Some(inst_root), &mut channel)?;

    Ok(())
}

// ─────────────────────────────────────────────
// Channel replay helper
// ─────────────────────────────────────────────

/// Advance `channel` through all operations the execution STARK consumed,
/// leaving it in the state just before the memory LogUp draws.
fn replay_exec_channel(
    exec_proof: &MultiProof,
    air:        &CairoAir252,
    _log_n:     u32,
    channel:    &mut Channel252,
) {
    use super::multi_stark::{LOG_BLOWUP, N_QUERIES};

    let eval_n = 1usize << (exec_proof.log_n + LOG_BLOWUP);

    // Mix public inputs
    for pi in &exec_proof.public_inputs {
        channel.mix_fp(&fp_from_u32x8(pi));
    }
    // Mix col roots
    for root in &exec_proof.col_roots {
        channel.mix_digest(root);
    }
    // Draw exec alphas
    for _ in 0..air.n_constraints() {
        let _ = channel.draw_fp();
    }
    // Mix quotient root
    channel.mix_digest(&exec_proof.quotient_root);
    // Advance through FRI
    let _ = channel.draw_fp(); // alpha_fri0
    for root in &exec_proof.fri_proof.inner_roots {
        channel.mix_digest(root);
        let _ = channel.draw_fp();
    }
    for v in &exec_proof.fri_proof.last_layer_evals {
        channel.mix_fp(&fp_from_u32x8(v));
    }
    // Consume query indices
    for _ in 0..N_QUERIES {
        let _ = channel.draw_number(eval_n);
    }
}

// We need the MultiColumnAir trait to call n_constraints().
use super::multi_stark::MultiColumnAir;

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cairo_air::{Instruction, CairoVm};

    fn make_test_air(log_n: u32) -> CairoAir252 {
        let inst = Instruction {
            off_dst:   0i16,
            off_op0:  -1i16,
            off_op1:  -1i16,
            dst_reg: 0, op0_reg: 1,
            op1_imm: 0, op1_fp: 1, op1_ap: 0,
            res_add: 1, res_mul: 0,
            pc_jump_abs: 0, pc_jump_rel: 0, pc_jnz: 0,
            ap_add: 0, ap_add1: 1,
            opcode_call: 0, opcode_ret: 0, opcode_assert_eq: 1,
        };
        let word = inst.encode();
        let n = 1usize << log_n;
        let pc0: u64 = 1000;
        let ap0: u64 = 2000;
        let fp0: u64 = 2000;
        let mut vm = CairoVm::new(pc0, ap0, fp0);
        for i in 0u64..n as u64 {
            vm.write(pc0 + i, Fp { v: [word, 0, 0, 0] });
        }
        vm.write(fp0 - 1, Fp::from_u64(5));
        for i in 0u64..n as u64 {
            vm.write(ap0 + i, Fp::from_u64(10));
        }
        CairoAir252::from_vm(&mut vm, n)
    }

    /// Full end-to-end Cairo proof: execution + memory + range-check.
    #[test]
    fn test_cairo_proof_e2e() {
        let log_n: u32 = 3;
        let air = make_test_air(log_n);

        let proof = prove_cairo(&air, log_n);
        verify_cairo(&proof, &air).expect("Full Cairo proof should verify");
    }

    /// Tamper the memory proof — should be caught.
    #[test]
    fn test_cairo_proof_tamper_memory() {
        let log_n: u32 = 3;
        let air = make_test_air(log_n);

        let mut proof = prove_cairo(&air, log_n);
        // Corrupt the exec_sum in the memory proof.
        proof.memory_proof.exec_sum[0] ^= 1;

        let result = verify_cairo(&proof, &air);
        assert!(result.is_err(), "Tampered memory proof should fail");
    }

    /// Tamper the range-check mults — should be caught.
    #[test]
    fn test_cairo_proof_tamper_rc() {
        let log_n: u32 = 3;
        let air = make_test_air(log_n);

        let mut proof = prove_cairo(&air, log_n);
        // Shift a multiplicity.
        let len = proof.rc_proof.mults.len();
        let pos = proof.rc_proof.mults.iter().position(|&m| m > 0).unwrap_or(0);
        let next = (pos + 1) % len;
        proof.rc_proof.mults[pos] -= 1;
        proof.rc_proof.mults[next] += 1;

        let result = verify_cairo(&proof, &air);
        assert!(result.is_err(), "Tampered RC proof should fail");
    }
}
