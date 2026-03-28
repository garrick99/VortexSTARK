//! Range-check argument over Stark252 (LogUp variant).
//!
//! Proves that each Cairo instruction offset (off0, off1, off2) lies in [0, 2^16).
//!
//! In the biased Cairo encoding: stored_off = actual_off + 2^15, so stored_off ∈ [0, 2^16).
//! The three 16-bit windows of the instruction word low 48 bits are the witnesses.
//!
//! # Protocol
//!
//! Draw challenge z from channel (after execution trace commitment).
//!
//! Execution side:
//!   S_exec = Σ_{i=0}^{N-1} Σ_{j∈{0,1,2}} 1/(z − off_j(row_i))
//!
//! Table: fixed lookup table T = {0, 1, …, 65535}.
//! Prover provides multiplicities mult[v] = #{(i,j) : off_j(row_i) = v}.
//!   S_table = Σ_{v=0}^{65535} −mult[v] / (z − v)
//!
//! Claim: S_exec + S_table = 0  (LogUp identity, holds iff all offsets ∈ T).
//!
//! The exec sum is certified by an interaction STARK: running sum S[k], step constraint
//!   S_next − S_cur = Σ_j 1/(z − off_j(next_row))
//! FRI certifies degree of the step quotient.

use serde::{Serialize, Deserialize};
use super::field::{Fp, fp_to_u32x8, fp_from_u32x8, ntt_root_of_unity, batch_inverse, Channel252};
use super::ntt::lde_cpu;
use super::merkle::{MerkleTree252, Digest, verify_auth_path};
use super::fri::{FriProof, fri_commit, fri_build_proof, fri_verify};
use super::multi_stark::{LOG_BLOWUP, BLOWUP, N_QUERIES};

/// Number of 16-bit offsets per instruction (off0, off1, off2).
pub const OFFSETS_PER_ROW: usize = 3;
/// Range of the lookup table: [0, RC_TABLE_SIZE).
pub const RC_TABLE_SIZE: usize = 1 << 16;

/// Cairo instruction column index.
const INST: usize = 3;

// ─────────────────────────────────────────────
// Offset extraction
// ─────────────────────────────────────────────

/// Extract the three 16-bit offsets from an instruction field element.
///
/// The instruction word's low 48 bits encode:
///   bits 0-15:  off0 (biased dst offset)
///   bits 16-31: off1 (biased op0 offset)
///   bits 32-47: off2 (biased op1 offset)
///
/// Returns [off0, off1, off2] as u16 values (range [0, 2^16)).
#[inline]
pub fn offsets_from_inst(inst: Fp) -> [u16; OFFSETS_PER_ROW] {
    let w = inst.v[0];
    [
        (w & 0xFFFF)        as u16,
        ((w >> 16) & 0xFFFF) as u16,
        ((w >> 32) & 0xFFFF) as u16,
    ]
}

/// Compute the LogUp contribution for one row's offsets:
/// Σ_{j=0..2} 1/(z − off_j).
///
/// `inst_lde_val` is the LDE value of the instruction column at this position.
/// Uses batch_inverse for efficiency.
pub fn rc_row_contribution(z: Fp, inst_val: Fp) -> Fp {
    let offs = offsets_from_inst(inst_val);
    let denoms: Vec<Fp> = offs.iter()
        .map(|&o| z.sub(Fp::from_u64(o as u64)))
        .collect();
    let inv = batch_inverse(&denoms);
    inv.iter().fold(Fp::ZERO, |acc, &v| acc.add(v))
}

// ─────────────────────────────────────────────
// Table sum
// ─────────────────────────────────────────────

/// Compute multiplicities and table sum.
///
/// mult[v] = number of offsets in the trace equal to v.
/// table_sum = Σ_{v=0}^{65535} −mult[v] / (z − v).
///
/// Returns (mults: [u32; 65536], table_sum).
pub fn compute_rc_table_sum(trace_inst_col: &[Fp], z: Fp) -> ([u32; RC_TABLE_SIZE], Fp) {
    let mut mults = [0u32; RC_TABLE_SIZE];
    for &inst in trace_inst_col {
        let offs = offsets_from_inst(inst);
        for o in offs {
            mults[o as usize] += 1;
        }
    }

    // table_sum = Σ −mult[v] / (z − v), only over non-zero mults for efficiency.
    let nonzero: Vec<(usize, u32)> = mults.iter().enumerate()
        .filter(|(_, m)| **m > 0)
        .map(|(v, m)| (v, *m))
        .collect();
    let denoms: Vec<Fp> = nonzero.iter()
        .map(|&(v, _)| z.sub(Fp::from_u64(v as u64)))
        .collect();
    let inv = batch_inverse(&denoms);
    let table_sum = nonzero.iter().zip(inv.iter())
        .fold(Fp::ZERO, |acc, (&(_, m), &d_inv)| {
            acc.sub(Fp::from_u64(m as u64).mul(d_inv))
        });

    (mults, table_sum)
}

// ─────────────────────────────────────────────
// Interaction column
// ─────────────────────────────────────────────

/// Compute the running sum column S[0..N] for range-check LogUp.
/// S[k] = Σ_{i=0}^{k} rc_row_contribution(z, inst[i]).
pub fn compute_rc_interaction_column(inst_col: &[Fp], z: Fp) -> Vec<Fp> {
    let mut s = Vec::with_capacity(inst_col.len());
    let mut running = Fp::ZERO;
    for &inst in inst_col {
        running = running.add(rc_row_contribution(z, inst));
        s.push(running);
    }
    s
}

// ─────────────────────────────────────────────
// Step quotient
// ─────────────────────────────────────────────

/// Compute the range-check step quotient:
///   Q_rc[i] = (S_lde[i+BLOWUP] − S_lde[i] − rc_contrib(inst_lde[i+BLOWUP])) / Z(x_i)
fn compute_rc_quotient(
    s_lde:    &[Fp],
    inst_lde: &[Fp],
    z:        Fp,
    log_n:    u32,
    log_eval: u32,
    d:        usize,
) -> Vec<Fp> {
    let n      = 1usize << log_n;
    let eval_n = 1usize << log_eval;

    let omega_n    = ntt_root_of_unity(log_n);
    let omega_eval = ntt_root_of_unity(log_eval);
    let excluded: Vec<Fp> = (n - d..n)
        .map(|k| fp_pow_u64(omega_n, k as u64))
        .collect();

    let mut z_raw = Vec::with_capacity(eval_n);
    let mut c_raw = Vec::with_capacity(eval_n);

    let mut xi = Fp::ONE;
    for i in 0..eval_n {
        z_raw.push(z_eval_at(xi, n, &excluded));

        let ni     = (i + BLOWUP) % eval_n;
        let s_diff = s_lde[ni].sub(s_lde[i]);
        let contrib = rc_row_contribution(z, inst_lde[ni]);
        c_raw.push(s_diff.sub(contrib));

        xi = xi.mul(omega_eval);
    }

    let nonzero_idx: Vec<usize> = (0..eval_n).filter(|&i| z_raw[i] != Fp::ZERO).collect();
    let nonzero_z:   Vec<Fp>    = nonzero_idx.iter().map(|&i| z_raw[i]).collect();
    let nonzero_inv              = batch_inverse(&nonzero_z);

    let mut z_inv = vec![Fp::ZERO; eval_n];
    for (k, &i) in nonzero_idx.iter().enumerate() {
        z_inv[i] = nonzero_inv[k];
    }

    (0..eval_n).map(|i| {
        if z_raw[i] == Fp::ZERO { Fp::ZERO }
        else { c_raw[i].mul(z_inv[i]) }
    }).collect()
}

// ─────────────────────────────────────────────
// Proof types
// ─────────────────────────────────────────────

/// Per-query decommitment for the range-check STARK verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RcDecommit {
    /// S(x_q)
    pub s_cur:       [u32; 8],
    /// S(x_{q+BLOWUP})
    pub s_next:      [u32; 8],
    pub cur_path:    Vec<Digest>,
    pub next_path:   Vec<Digest>,
    /// inst column value at x_{q+BLOWUP} (next row's instruction).
    pub inst_next:   [u32; 8],
    /// Auth path for inst_next.
    pub inst_path:   Vec<Digest>,
}

/// Proof of the range-check LogUp argument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeCheckProof {
    /// Commitment to LDE of running sum S.
    pub s_root:  Digest,
    /// Commitment to LDE of step quotient Q_rc.
    pub q_root:  Digest,
    /// Final exec sum S[N-1].
    pub exec_sum: [u32; 8],
    /// Multiplicities for each value in [0, 2^16).
    pub mults: Vec<u32>,
    /// Per-query decommits.
    pub decommits: Vec<RcDecommit>,
    /// FRI proof for Q_rc.
    pub fri_proof: FriProof,
    /// Query indices.
    pub query_indices: Vec<usize>,
}

// ─────────────────────────────────────────────
// Prover
// ─────────────────────────────────────────────

/// Prove the range-check LogUp argument.
///
/// `trace_cols` — the Cairo execution trace (31 columns, each length N=2^log_n).
/// `channel`    — Fiat-Shamir channel, already advanced past the execution proof.
pub fn prove_range_check(
    trace_cols: &[Vec<Fp>],
    log_n:      u32,
    channel:    &mut Channel252,
) -> RangeCheckProof {
    let n        = 1usize << log_n;
    let log_eval = log_n + LOG_BLOWUP;
    let eval_n   = 1usize << log_eval;
    let d = 1usize;

    let inst_col = &trace_cols[INST];
    assert_eq!(inst_col.len(), n);

    // ── Draw challenge ────────────────────────
    let z = channel.draw_fp();

    // ── Interaction column S ──────────────────
    let s_trace  = compute_rc_interaction_column(inst_col, z);
    let exec_sum = *s_trace.last().unwrap();

    let s_lde   = lde_cpu(&s_trace, log_n, LOG_BLOWUP);
    let s_tree  = MerkleTree252::commit(&s_lde);
    let s_root  = s_tree.root();
    channel.mix_digest(&s_root);

    // ── Table sum ─────────────────────────────
    let (mults, table_sum) = compute_rc_table_sum(inst_col, z);
    let check = exec_sum.add(table_sum);
    assert!(check == Fp::ZERO,
        "Range-check LogUp: exec_sum + table_sum ≠ 0 — offset out of range");

    // ── Step quotient ─────────────────────────
    let inst_lde = lde_cpu(inst_col, log_n, LOG_BLOWUP);
    let q_lde    = compute_rc_quotient(&s_lde, &inst_lde, z, log_n, log_eval, d);

    let q_tree  = MerkleTree252::commit(&q_lde);
    let q_root  = q_tree.root();
    channel.mix_digest(&q_root);

    // ── FRI ───────────────────────────────────
    let fri_witness = fri_commit(q_lde.clone(), &q_root, log_eval, channel);

    // ── Draw queries ──────────────────────────
    let query_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_n))
        .collect();

    let fri_proof = fri_build_proof(&fri_witness, &q_tree, &query_indices);

    // ── Commit inst LDE tree (for auth paths) ─
    let inst_tree = MerkleTree252::commit(&inst_lde);

    // ── Decommit ──────────────────────────────
    let decommits: Vec<RcDecommit> = query_indices.iter().map(|&q| {
        let q_next = (q + BLOWUP) % eval_n;
        RcDecommit {
            s_cur:     fp_to_u32x8(&s_lde[q]),
            s_next:    fp_to_u32x8(&s_lde[q_next]),
            cur_path:  s_tree.auth_path(q),
            next_path: s_tree.auth_path(q_next),
            inst_next: fp_to_u32x8(&inst_lde[q_next]),
            inst_path: inst_tree.auth_path(q_next),
        }
    }).collect();

    RangeCheckProof {
        s_root,
        q_root,
        exec_sum: fp_to_u32x8(&exec_sum),
        mults: mults.to_vec(),
        decommits,
        fri_proof,
        query_indices,
    }
}

// ─────────────────────────────────────────────
// Verifier
// ─────────────────────────────────────────────

/// Verify the range-check LogUp proof.
///
/// `inst_root` — the Merkle root of the LDE of the instruction column
///              (from the execution proof's col_roots). Pass `None` to skip
///              auth-path verification of the inst column (standalone mode).
pub fn verify_range_check(
    proof:     &RangeCheckProof,
    log_n:     u32,
    inst_root: Option<&Digest>,
    channel:   &mut Channel252,
) -> Result<(), String> {
    let n        = 1usize << log_n;
    let log_eval = log_n + LOG_BLOWUP;
    let eval_n   = 1usize << log_eval;
    let d = 1usize;

    if proof.mults.len() != RC_TABLE_SIZE {
        return Err(format!("Expected {} mults, got {}", RC_TABLE_SIZE, proof.mults.len()));
    }

    // ── Replay challenge ──────────────────────
    let z = channel.draw_fp();
    channel.mix_digest(&proof.s_root);

    // ── Check 1: exec_sum + table_sum == 0 ────
    let exec_sum = fp_from_u32x8(&proof.exec_sum);
    let nonzero: Vec<(usize, u32)> = proof.mults.iter().enumerate()
        .filter(|(_, m)| **m > 0)
        .map(|(v, m)| (v, *m))
        .collect();
    let denoms: Vec<Fp> = nonzero.iter()
        .map(|&(v, _)| z.sub(Fp::from_u64(v as u64)))
        .collect();
    let inv = batch_inverse(&denoms);
    let table_sum = nonzero.iter().zip(inv.iter())
        .fold(Fp::ZERO, |acc, (&(_, m), &d_inv)| {
            acc.sub(Fp::from_u64(m as u64).mul(d_inv))
        });

    if exec_sum.add(table_sum) != Fp::ZERO {
        return Err("Range-check identity failed: exec_sum + table_sum ≠ 0".into());
    }

    // ── Check 2: multiplicity sanity ──────────
    let total_mults: u64 = proof.mults.iter().map(|&m| m as u64).sum();
    let expected: u64    = (n * OFFSETS_PER_ROW) as u64;
    if total_mults != expected {
        return Err(format!(
            "Multiplicity total mismatch: expected {expected} (N={n}×3), got {total_mults}"
        ));
    }

    channel.mix_digest(&proof.q_root);

    // Clone channel here — same state the prover was in when fri_commit was called.
    let mut fri_ch = channel.clone();

    // ── Advance main channel through FRI ops (for query-index replay) ──
    let _alpha0 = channel.draw_fp();
    for root in &proof.fri_proof.inner_roots {
        channel.mix_digest(root);
        let _ = channel.draw_fp();
    }
    for v in &proof.fri_proof.last_layer_evals {
        channel.mix_fp(&fp_from_u32x8(v));
    }
    let expected_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_n))
        .collect();
    if proof.query_indices != expected_indices {
        return Err("Range-check: query indices do not match Fiat-Shamir".into());
    }

    // ── Vanishing polynomial ──────────────────
    let omega_n    = ntt_root_of_unity(log_n);
    let omega_eval = ntt_root_of_unity(log_eval);
    let excluded: Vec<Fp> = (n - d..n)
        .map(|k| fp_pow_u64(omega_n, k as u64))
        .collect();

    // ── Per-query checks ─────────────────────
    for (qi, (&q, dc)) in proof.query_indices.iter()
        .zip(proof.decommits.iter())
        .enumerate()
    {
        let q_next = (q + BLOWUP) % eval_n;

        // Verify S auth paths.
        let s_cur  = fp_from_u32x8(&dc.s_cur);
        let s_next = fp_from_u32x8(&dc.s_next);
        if !verify_auth_path(&s_cur,  &dc.cur_path,  &proof.s_root, q,      log_eval) {
            return Err(format!("Query {qi}: RC S cur path failed (q={q})"));
        }
        if !verify_auth_path(&s_next, &dc.next_path, &proof.s_root, q_next, log_eval) {
            return Err(format!("Query {qi}: RC S next path failed (q_next={q_next})"));
        }

        // Optionally verify inst auth path.
        let inst_next = fp_from_u32x8(&dc.inst_next);
        if let Some(ir) = inst_root {
            if !verify_auth_path(&inst_next, &dc.inst_path, ir, q_next, log_eval) {
                return Err(format!("Query {qi}: RC inst next path failed (q_next={q_next})"));
            }
        }

        // Compute contrib from decommitted inst value.
        let contrib = rc_row_contribution(z, inst_next);

        // Get Q_rc(x_q) from FRI layer-0.
        let fri_ld = &proof.fri_proof.query_decommits[qi].layers[0];
        let half   = eval_n / 2;
        let low    = q % half;
        let q_val  = if q == low {
            fp_from_u32x8(&fri_ld.f_lo)
        } else {
            fp_from_u32x8(&fri_ld.f_hi)
        };

        // Check: Q_rc(x_q) · Z(x_q) == S_next − S_cur − contrib
        let x_q  = fp_pow_u64(omega_eval, q as u64);
        let z_q  = z_eval_at(x_q, n, &excluded);
        let lhs  = q_val.mul(z_q);
        let rhs  = s_next.sub(s_cur).sub(contrib);

        if lhs != rhs {
            return Err(format!(
                "Query {qi}: RC Q·Z ≠ S_next−S_cur−contrib at q={q}: lhs={:?}, rhs={:?}",
                fp_to_u32x8(&lhs), fp_to_u32x8(&rhs)
            ));
        }
    }

    // ── FRI verify ────────────────────────────
    fri_verify(
        &proof.fri_proof,
        &proof.q_root,
        &proof.query_indices,
        log_eval,
        &mut fri_ch,
    )?;

    Ok(())
}

// ─────────────────────────────────────────────
// Internal helpers (shared pattern)
// ─────────────────────────────────────────────

fn z_eval_at(x: Fp, n: usize, excluded: &[Fp]) -> Fp {
    let x_pow_n   = fp_pow_u64(x, n as u64);
    let numerator = x_pow_n.sub(Fp::ONE);

    let mut denom_prod = Fp::ONE;
    let mut zero_idx: Option<usize> = None;
    for (k, &e) in excluded.iter().enumerate() {
        let d = x.sub(e);
        if d == Fp::ZERO {
            zero_idx = Some(k);
        } else {
            denom_prod = denom_prod.mul(d);
        }
    }

    if let Some(k) = zero_idx {
        let e         = excluded[k];
        let e_pow_nm1 = fp_pow_u64(e, (n - 1) as u64);
        let n_fp      = Fp::from_u64(n as u64);
        n_fp.mul(e_pow_nm1).mul(denom_prod.inverse())
    } else {
        numerator.mul(denom_prod.inverse())
    }
}

fn fp_pow_u64(base: Fp, exp: u64) -> Fp {
    base.pow_fp(crate::cairo_air::stark252_field::Fp { v: [exp, 0, 0, 0] })
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cairo_air::{Instruction, CairoVm, CairoAir252};
    use super::super::multi_stark::prove_multi;

    fn make_test_vm(log_n: u32) -> (CairoAir252, Vec<Vec<Fp>>) {
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
        let air = CairoAir252::from_vm(&mut vm, n);
        let trace_cols = air.trace.columns.clone();
        (air, trace_cols)
    }

    /// Test that exec_sum + table_sum == 0 for valid offsets.
    #[test]
    fn test_rc_sum_identity() {
        let (_, trace_cols) = make_test_vm(3);
        let mut ch = Channel252::new();
        let z = ch.draw_fp();

        let inst_col = &trace_cols[INST];
        let s = compute_rc_interaction_column(inst_col, z);
        let exec_sum = *s.last().unwrap();
        let (_, table_sum) = compute_rc_table_sum(inst_col, z);
        assert_eq!(exec_sum.add(table_sum), Fp::ZERO);
    }

    /// Full prove + verify roundtrip.
    #[test]
    fn test_rc_prove_verify() {
        let log_n: u32 = 3;
        let (_, trace_cols) = make_test_vm(log_n);

        let mut channel = Channel252::new();
        let proof = prove_range_check(&trace_cols, log_n, &mut channel);

        let mut verify_channel = Channel252::new();
        verify_range_check(&proof, log_n, None, &mut verify_channel)
            .expect("Range-check proof should verify");
    }

    /// Tamper: flip a multiplicity — should be caught.
    #[test]
    fn test_rc_tamper_mults() {
        let log_n: u32 = 3;
        let (_, trace_cols) = make_test_vm(log_n);

        let mut channel = Channel252::new();
        let mut proof = prove_range_check(&trace_cols, log_n, &mut channel);

        // Move a count from one value to another (keeps total, changes sum).
        if proof.mults[0] > 0 {
            proof.mults[0] -= 1;
            proof.mults[1] += 1;
        } else {
            proof.mults[32768] -= 1;
            proof.mults[32769] += 1;
        }

        let mut verify_channel = Channel252::new();
        let result = verify_range_check(&proof, log_n, None, &mut verify_channel);
        assert!(result.is_err(), "Tampered mults should fail verification");
    }

    /// Combined execution STARK + range check (end-to-end).
    #[test]
    fn test_cairo_with_range_check() {
        let log_n: u32 = 3;
        let (air, trace_cols) = make_test_vm(log_n);

        let exec_proof = prove_multi(&air, log_n);
        super::super::multi_stark::verify_multi(&exec_proof, &air)
            .expect("Execution STARK should verify");

        let mut rc_channel = Channel252::new();
        let rc_proof = prove_range_check(&trace_cols, log_n, &mut rc_channel);

        let mut verify_channel = Channel252::new();
        verify_range_check(&rc_proof, log_n, None, &mut verify_channel)
            .expect("Range-check proof should verify");
    }
}
