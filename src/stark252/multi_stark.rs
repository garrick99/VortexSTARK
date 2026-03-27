//! Multi-column STARK prover over Stark252 using standard FRI.
//!
//! # Protocol
//!
//! Given an AIR with `n_cols` columns and `n_constraints` constraints:
//!
//! 1. Generate trace T[col][row] (length N = 2^log_n per column).
//! 2. LDE each column to size 4N (blowup 4).
//! 3. Commit each column → col_roots[0..n_cols].
//! 4. Fiat-Shamir: mix all col_roots → draw α_0..α_{n_constraints-1}.
//! 5. Compute combined quotient:
//!       Q(x) = Σ_j α_j · C_j(cur_x, next_x) / Z(x)
//!    where Z(x) = (x^N − 1) / ∏_{k=N-d..N-1}(x − ω_N^k)
//!    and (cur_x, next_x) are the column values at evaluation domain point x
//!    and the next step point (ω^blowup · x).
//! 6. Commit Q → quotient_root. Mix into channel.
//! 7. FRI commit on Q.
//! 8. Draw N_QUERIES query indices.
//! 9. For each query q:
//!    - Decommit all columns at q and q+BLOWUP (Merkle paths).
//!    - Decommit Q(x_q) via FRI layer-0 auth path.
//!    - Check Q(x_q) · Z(x_q) == Σ_j α_j · C_j.
//! 10. Provide FRI fold proofs.
//!
//! # Degree analysis
//! For a degree-2 constraint: deg(C_j) ≤ 2(N−1), deg(Z) = N−1, deg(Q) ≤ N−1.
//! FRI certifies Q has degree < N on a 4N domain. ✓

use serde::{Serialize, Deserialize};
use super::field::{Fp, fp_to_u32x8, fp_from_u32x8, ntt_root_of_unity, batch_inverse, Channel252};
use super::ntt::lde_cpu;
use super::merkle::{MerkleTree252, Digest, verify_auth_path};
use super::fri::{FriProof, fri_commit, fri_build_proof, fri_verify};

pub const LOG_BLOWUP: u32 = 2;
pub const BLOWUP: usize   = 1 << LOG_BLOWUP;
pub const N_QUERIES: usize = 40;

// ─────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────

/// A multi-column AIR (Algebraic Intermediate Representation) over Stark252.
pub trait MultiColumnAir {
    /// Number of trace columns.
    fn n_cols(&self) -> usize;

    /// Generate the execution trace.
    ///
    /// Returns `columns[col][row]` with `col ∈ [0, n_cols)` and `row ∈ [0, 2^log_n)`.
    fn generate_trace(&self, log_n: u32) -> Vec<Vec<Fp>>;

    /// Evaluate all constraints at a single evaluation domain point.
    ///
    /// `cur[c]`  = column c value at current position.
    /// `next[c]` = column c value at next trace step (current position + BLOWUP in eval domain).
    ///
    /// Returns a `Vec<Fp>` of length `n_constraints()`, each entry = C_j(cur, next).
    /// Every entry should be 0 when evaluated at trace domain points k < N − n_boundary_rows.
    fn eval_constraints(&self, cur: &[Fp], next: &[Fp]) -> Vec<Fp>;

    /// Number of constraints.
    fn n_constraints(&self) -> usize;

    /// Number of boundary (unconstrained) rows at the end of the trace.
    /// Typically 1: the last row has no "next" row.
    fn n_boundary_rows(&self) -> usize;

    /// Public inputs (mixed into Fiat-Shamir transcript for binding).
    fn public_inputs(&self) -> Vec<Fp>;
}

// ─────────────────────────────────────────────
// Proof types
// ─────────────────────────────────────────────

/// Per-query decommitment for all trace columns.
#[derive(Clone, Serialize, Deserialize)]
pub struct MultiColDecommit {
    /// Column values at evaluation domain index q.  Length = n_cols.
    pub cur_vals: Vec<[u32; 8]>,
    /// Column values at evaluation domain index (q + BLOWUP) % eval_n.
    pub next_vals: Vec<[u32; 8]>,
    /// Merkle auth paths for cur_vals (one per column).
    pub cur_paths: Vec<Vec<[u32; 8]>>,
    /// Merkle auth paths for next_vals.
    pub next_paths: Vec<Vec<[u32; 8]>>,
}

/// A multi-column STARK proof over Stark252.
#[derive(Clone, Serialize, Deserialize)]
pub struct MultiProof {
    pub log_n: u32,
    /// Public inputs (as u32x8).
    pub public_inputs: Vec<[u32; 8]>,
    /// One Merkle root per column.
    pub col_roots: Vec<Digest>,
    /// Merkle root of the combined quotient polynomial Q.
    pub quotient_root: Digest,
    /// Query indices in [0, eval_n).
    pub query_indices: Vec<usize>,
    /// Per-query trace decommitments.
    pub trace_decommits: Vec<MultiColDecommit>,
    /// FRI proof for Q.
    pub fri_proof: FriProof,
}

// ─────────────────────────────────────────────
// Prove
// ─────────────────────────────────────────────

pub fn prove_multi<A: MultiColumnAir>(air: &A, log_n: u32) -> MultiProof {
    let n        = 1usize << log_n;
    let log_eval = log_n + LOG_BLOWUP;
    let eval_n   = 1usize << log_eval;

    // ── Step 1: Trace + LDE ──────────────────
    let trace = air.generate_trace(log_n);
    assert_eq!(trace.len(), air.n_cols(), "generate_trace returned wrong number of columns");
    for col in &trace {
        assert_eq!(col.len(), n, "trace column has wrong length");
    }

    let t_eval: Vec<Vec<Fp>> = trace.iter()
        .map(|col| lde_cpu(col, log_n, LOG_BLOWUP))
        .collect();

    // ── Step 2: Commit columns ───────────────
    let col_trees: Vec<MerkleTree252> = t_eval.iter()
        .map(|col| MerkleTree252::commit(col))
        .collect();
    let col_roots: Vec<Digest> = col_trees.iter().map(|t| t.root()).collect();

    // ── Step 3: Fiat-Shamir → alphas ─────────
    let mut channel = Channel252::new();
    for pi in air.public_inputs() {
        channel.mix_fp(&pi);
    }
    for root in &col_roots {
        channel.mix_digest(root);
    }
    let alphas: Vec<Fp> = (0..air.n_constraints())
        .map(|_| channel.draw_fp())
        .collect();

    // ── Step 4: Compute combined quotient ────
    let d     = air.n_boundary_rows();
    let q_eval = compute_combined_quotient(air, &t_eval, &alphas, n, log_n, log_eval, d);

    // ── Step 5: Commit quotient ──────────────
    let quotient_tree = MerkleTree252::commit(&q_eval);
    let quotient_root = quotient_tree.root();
    channel.mix_digest(&quotient_root);

    // ── Step 6: FRI ──────────────────────────
    let witness = fri_commit(q_eval.clone(), &quotient_root, log_eval, &mut channel);

    // ── Step 7: Draw queries ─────────────────
    let query_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_n))
        .collect();

    // ── Step 8: Decommit trace ───────────────
    let n_cols = air.n_cols();
    let trace_decommits: Vec<MultiColDecommit> = query_indices.iter().map(|&q| {
        let q_next = (q + BLOWUP) % eval_n;
        MultiColDecommit {
            cur_vals:   (0..n_cols).map(|c| fp_to_u32x8(&t_eval[c][q])).collect(),
            next_vals:  (0..n_cols).map(|c| fp_to_u32x8(&t_eval[c][q_next])).collect(),
            cur_paths:  (0..n_cols).map(|c| col_trees[c].auth_path(q)).collect(),
            next_paths: (0..n_cols).map(|c| col_trees[c].auth_path(q_next)).collect(),
        }
    }).collect();

    // ── Step 9: FRI proof ────────────────────
    let fri_proof = fri_build_proof(&witness, &quotient_tree, &query_indices);

    MultiProof {
        log_n,
        public_inputs: air.public_inputs().iter().map(fp_to_u32x8).collect(),
        col_roots,
        quotient_root,
        query_indices,
        trace_decommits,
        fri_proof,
    }
}

// ─────────────────────────────────────────────
// Verify
// ─────────────────────────────────────────────

pub fn verify_multi<A: MultiColumnAir>(proof: &MultiProof, air: &A) -> Result<(), String> {
    let log_n    = proof.log_n;
    let log_eval = log_n + LOG_BLOWUP;
    let eval_n   = 1usize << log_eval;
    let n        = 1usize << log_n;
    let d        = air.n_boundary_rows();
    let n_cols   = air.n_cols();

    // Public input binding
    let expected_pubs: Vec<[u32; 8]> = air.public_inputs().iter().map(fp_to_u32x8).collect();
    if proof.public_inputs != expected_pubs {
        return Err("Public inputs mismatch".into());
    }
    if proof.col_roots.len() != n_cols {
        return Err(format!("Expected {n_cols} col roots, got {}", proof.col_roots.len()));
    }
    if proof.trace_decommits.len() != N_QUERIES {
        return Err(format!("Expected {N_QUERIES} trace decommits"));
    }

    // ── Replay Fiat-Shamir ───────────────────
    let mut channel = Channel252::new();
    for pi in &proof.public_inputs {
        channel.mix_fp(&fp_from_u32x8(pi));
    }
    for root in &proof.col_roots {
        channel.mix_digest(root);
    }
    let alphas: Vec<Fp> = (0..air.n_constraints())
        .map(|_| channel.draw_fp())
        .collect();
    channel.mix_digest(&proof.quotient_root);

    // Advance through FRI commit to reach query index draw point
    let _alpha_fri0 = channel.draw_fp();
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
        return Err("Query indices do not match Fiat-Shamir transcript".into());
    }

    // ── Precompute vanishing polynomial roots ──
    let omega_n    = ntt_root_of_unity(log_n);
    let omega_eval = ntt_root_of_unity(log_eval);
    let excluded: Vec<Fp> = (n - d..n)
        .map(|k| fp_pow_u64(omega_n, k as u64))
        .collect();

    // ── Per-query checks ─────────────────────
    for (qi, (&q, td)) in proof.query_indices.iter()
        .zip(proof.trace_decommits.iter())
        .enumerate()
    {
        if td.cur_vals.len() != n_cols || td.next_vals.len() != n_cols {
            return Err(format!("Query {qi}: wrong column count in decommit"));
        }

        let q_next = (q + BLOWUP) % eval_n;

        // 1. Verify Merkle auth paths for all columns
        for c in 0..n_cols {
            let cur_val  = fp_from_u32x8(&td.cur_vals[c]);
            let next_val = fp_from_u32x8(&td.next_vals[c]);
            if !verify_auth_path(&cur_val,  &td.cur_paths[c],  &proof.col_roots[c], q,      log_eval) {
                return Err(format!("Query {qi} col {c}: cur auth path failed (q={q})"));
            }
            if !verify_auth_path(&next_val, &td.next_paths[c], &proof.col_roots[c], q_next, log_eval) {
                return Err(format!("Query {qi} col {c}: next auth path failed (q_next={q_next})"));
            }
        }

        // 2. Compute combined constraint C = Σ α_j · C_j(cur, next)
        let cur:  Vec<Fp> = td.cur_vals.iter().map(fp_from_u32x8).collect();
        let next: Vec<Fp> = td.next_vals.iter().map(fp_from_u32x8).collect();
        let cs     = air.eval_constraints(&cur, &next);
        let c_q    = alphas.iter().zip(cs.iter())
            .fold(Fp::ZERO, |acc, (&a, &c)| acc.add(a.mul(c)));

        // 3. Get Q(x_q) from FRI layer-0 decommit
        let fri_ld = &proof.fri_proof.query_decommits[qi].layers[0];
        let half   = eval_n / 2;
        let low    = q % half;
        let q_val  = if q == low {
            fp_from_u32x8(&fri_ld.f_lo)
        } else {
            fp_from_u32x8(&fri_ld.f_hi)
        };

        // 4. Constraint check: Q(x_q) · Z(x_q) == C(x_q)
        let x_q = fp_pow_u64(omega_eval, q as u64);
        let z_q = z_eval_at(x_q, n, &excluded);
        let lhs = q_val.mul(z_q);
        if lhs != c_q {
            return Err(format!(
                "Query {qi}: Q·Z ≠ C at q={q}: Q·Z={:?}, C={:?}",
                fp_to_u32x8(&lhs), fp_to_u32x8(&c_q)
            ));
        }
    }

    // ── FRI verify ───────────────────────────
    // Rebuild channel state at the point fri_verify expects it
    // (f0_root = quotient_root already mixed in).
    let mut fri_channel = Channel252::new();
    for pi in &proof.public_inputs {
        fri_channel.mix_fp(&fp_from_u32x8(pi));
    }
    for root in &proof.col_roots {
        fri_channel.mix_digest(root);
    }
    for _ in 0..air.n_constraints() {
        let _ = fri_channel.draw_fp();
    }
    fri_channel.mix_digest(&proof.quotient_root);

    fri_verify(
        &proof.fri_proof,
        &proof.quotient_root,
        &proof.query_indices,
        log_eval,
        &mut fri_channel,
    )?;

    Ok(())
}

// ─────────────────────────────────────────────
// Internal: combined quotient computation
// ─────────────────────────────────────────────

fn compute_combined_quotient<A: MultiColumnAir>(
    air: &A,
    t_eval: &[Vec<Fp>],
    alphas: &[Fp],
    n: usize,
    log_n: u32,
    log_eval: u32,
    d: usize,
) -> Vec<Fp> {
    let eval_n     = t_eval[0].len();
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

        let cur:  Vec<Fp> = t_eval.iter().map(|col| col[i]).collect();
        let next: Vec<Fp> = t_eval.iter().map(|col| col[(i + BLOWUP) % eval_n]).collect();
        let cs: Vec<Fp>   = air.eval_constraints(&cur, &next);
        let c = alphas.iter().zip(cs.iter())
            .fold(Fp::ZERO, |acc, (&a, &cv)| acc.add(a.mul(cv)));
        c_raw.push(c);
        xi = xi.mul(omega_eval);
    }

    // Batch invert: skip zeros (trace domain points where Z=0)
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
// Vanishing polynomial
// ─────────────────────────────────────────────

/// Z(x) = (x^N − 1) / ∏_{e ∈ excluded}(x − e).
/// Handles x = e via L'Hôpital: Z(e) = N · e^{N−1} / ∏_{f≠e}(e − f).
fn z_eval_at(x: Fp, n: usize, excluded: &[Fp]) -> Fp {
    let x_pow_n   = fp_pow_u64(x, n as u64);
    let numerator = x_pow_n.sub(Fp::ONE);

    let mut denom = Fp::ONE;
    let mut zero_idx: Option<usize> = None;
    for (k, &e) in excluded.iter().enumerate() {
        let d = x.sub(e);
        if d == Fp::ZERO {
            zero_idx = Some(k);
        } else {
            denom = denom.mul(d);
        }
    }

    if let Some(k) = zero_idx {
        let e          = excluded[k];
        let e_pow_nm1  = fp_pow_u64(e, (n - 1) as u64);
        let n_fp       = Fp::from_u64(n as u64);
        n_fp.mul(e_pow_nm1).mul(denom.inverse())
    } else {
        numerator.mul(denom.inverse())
    }
}

fn fp_pow_u64(base: Fp, exp: u64) -> Fp {
    base.pow_fp(crate::cairo_air::stark252_field::Fp { v: [exp, 0, 0, 0] })
}


// ─────────────────────────────────────────────
// Example AIR: 2-column Fibonacci
// ─────────────────────────────────────────────

/// A 2-column Fibonacci AIR.
///
/// Columns: [a, b].
/// Constraints:
///   C0: a_next − b_cur = 0
///   C1: b_next − a_cur − b_cur = 0
///
/// This proves that (a_k, b_k) follows the Fibonacci recurrence.
pub struct FibonacciTwoColAir {
    pub a0: Fp,
    pub b0: Fp,
}

impl MultiColumnAir for FibonacciTwoColAir {
    fn n_cols(&self) -> usize { 2 }

    fn generate_trace(&self, log_n: u32) -> Vec<Vec<Fp>> {
        let n = 1usize << log_n;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        a.push(self.a0);
        b.push(self.b0);
        for _ in 1..n {
            let (ai, bi) = (*a.last().unwrap(), *b.last().unwrap());
            a.push(bi);
            b.push(ai.add(bi));
        }
        vec![a, b]
    }

    fn eval_constraints(&self, cur: &[Fp], next: &[Fp]) -> Vec<Fp> {
        // C0: a_next = b_cur
        let c0 = next[0].sub(cur[1]);
        // C1: b_next = a_cur + b_cur
        let c1 = next[1].sub(cur[0]).sub(cur[1]);
        vec![c0, c1]
    }

    fn n_constraints(&self) -> usize { 2 }
    fn n_boundary_rows(&self) -> usize { 1 }
    fn public_inputs(&self) -> Vec<Fp> { vec![self.a0, self.b0] }
}

/// A 3-column Tribonacci AIR.
///
/// Columns: [a, b, c].
/// Constraint: c_next = a_cur + b_cur + c_cur  (with a_next = b_cur, b_next = c_cur).
pub struct TribonacciAir {
    pub a0: Fp,
    pub b0: Fp,
    pub c0: Fp,
}

impl MultiColumnAir for TribonacciAir {
    fn n_cols(&self) -> usize { 3 }

    fn generate_trace(&self, log_n: u32) -> Vec<Vec<Fp>> {
        let n = 1usize << log_n;
        let mut a = vec![self.a0];
        let mut b = vec![self.b0];
        let mut c = vec![self.c0];
        for _ in 1..n {
            let (ai, bi, ci) = (*a.last().unwrap(), *b.last().unwrap(), *c.last().unwrap());
            a.push(bi);
            b.push(ci);
            c.push(ai.add(bi).add(ci));
        }
        vec![a, b, c]
    }

    fn eval_constraints(&self, cur: &[Fp], next: &[Fp]) -> Vec<Fp> {
        vec![
            next[0].sub(cur[1]),             // a_next = b_cur
            next[1].sub(cur[2]),             // b_next = c_cur
            next[2].sub(cur[0]).sub(cur[1]).sub(cur[2]), // c_next = a+b+c
        ]
    }

    fn n_constraints(&self) -> usize { 3 }
    fn n_boundary_rows(&self) -> usize { 1 }
    fn public_inputs(&self) -> Vec<Fp> { vec![self.a0, self.b0, self.c0] }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_two_col_prove_verify() {
        let air = FibonacciTwoColAir {
            a0: Fp::from_u64(1),
            b0: Fp::from_u64(1),
        };
        let proof = prove_multi(&air, 5);
        verify_multi(&proof, &air).expect("2-column Fibonacci STARK should verify");
    }

    #[test]
    fn test_tribonacci_prove_verify() {
        let air = TribonacciAir {
            a0: Fp::from_u64(1),
            b0: Fp::from_u64(1),
            c0: Fp::from_u64(2),
        };
        let proof = prove_multi(&air, 5);
        verify_multi(&proof, &air).expect("Tribonacci STARK should verify");
    }

    #[test]
    fn test_tampered_trace_rejected() {
        let air = FibonacciTwoColAir {
            a0: Fp::from_u64(3),
            b0: Fp::from_u64(5),
        };
        let mut proof = prove_multi(&air, 5);
        proof.trace_decommits[0].cur_vals[0][0] ^= 1;
        assert!(
            verify_multi(&proof, &air).is_err(),
            "Tampered trace should be rejected"
        );
    }

    #[test]
    fn test_wrong_public_inputs_rejected() {
        let air_prover = FibonacciTwoColAir { a0: Fp::from_u64(1), b0: Fp::from_u64(1) };
        let proof = prove_multi(&air_prover, 5);

        let air_verifier = FibonacciTwoColAir { a0: Fp::from_u64(2), b0: Fp::from_u64(1) };
        assert!(
            verify_multi(&proof, &air_verifier).is_err(),
            "Wrong public input should be rejected"
        );
    }

    #[test]
    fn test_larger_trace() {
        // log_n=7 → N=128, domain=512; tests more FRI fold rounds
        let air = FibonacciTwoColAir {
            a0: Fp::from_u64(7),
            b0: Fp::from_u64(11),
        };
        let proof = prove_multi(&air, 7);
        verify_multi(&proof, &air).expect("Larger trace should verify");
    }
}
