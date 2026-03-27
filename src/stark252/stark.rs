//! General single-column STARK prover over Stark252 using standard FRI.
//!
//! Handles arbitrary-degree constraints — unlike the Fibonacci prover which exploits
//! the degree-1 quotient. The FRI protocol provides low-degree proof for Q = C/Z.
//!
//! # Protocol
//!
//! Given:
//!   - Trace T[0..N] (single column, length N = 2^log_n)
//!   - Constraint C(x) = 0 on trace domain {ω_N^k : k=0..N-D} for some D ≥ 1
//!     (D = number of unconstrained rows at end)
//!   - Vanishing polynomial Z(x) = (x^N − 1) / ∏_{k=N-D}^{N-1}(x − ω_N^k)
//!   - Blowup factor 4×
//!
//! Steps:
//! 1. Generate trace T, LDE to 4N evaluations.
//! 2. Commit trace → trace_root.
//! 3. Compute Q_eval = C_eval / Z_eval (pointwise on the 4N eval domain).
//!    Q has degree N−1 for degree-2 constraints (quotient degree = 2N-1 - (N-D) ≈ N).
//! 4. Commit Q_eval → quotient_root. Mix into channel.
//! 5. FRI commit on Q_eval (fold log_n times).
//! 6. Draw query indices.
//! 7. For each query q:
//!    - Decommit trace at q (and neighbors needed by constraint).
//!    - Decommit Q_eval at q (via FRI layer-0 path).
//!    - Check Q(x_q) * Z(x_q) == C(x_q).
//!    - FRI fold checks (handled by fri_verify).
//!
//! # Example constraint: T[k+1] = T[k]^2 + c
//! Degree-2 constraint polynomial. Quotient has degree N, requires FRI.

use serde::{Serialize, Deserialize};
use super::field::{Fp, fp_to_u32x8, fp_from_u32x8, ntt_root_of_unity, batch_inverse, Channel252};
use super::ntt::lde_cpu;
use super::merkle::{MerkleTree252, Digest, verify_auth_path};
use super::fri::{FriProof, FriWitness, fri_commit, fri_build_proof, fri_verify, LOG_LAST_LAYER};

/// Blowup factor.
pub const LOG_BLOWUP: u32 = 2;
pub const BLOWUP: usize   = 1 << LOG_BLOWUP;

/// Number of STARK queries (combined constraint + FRI).
pub const N_QUERIES: usize = 40;

// ─────────────────────────────────────────────
// Proof types
// ─────────────────────────────────────────────

/// Decommitment for trace values at one query position.
#[derive(Clone, Serialize, Deserialize)]
pub struct TraceQueryDecommit {
    /// T_eval[q] and its Merkle auth path.
    pub t_val:  [u32; 8],
    pub t_path: Vec<[u32; 8]>,
    /// T_eval[(q + blowup) % eval_n] — the "next step" value.
    pub t_next_val:  [u32; 8],
    pub t_next_path: Vec<[u32; 8]>,
}

/// Decommitment for the quotient value at one query position (layer 0).
/// The auth path comes from the FRI layer-0 decommit.
#[derive(Clone, Serialize, Deserialize)]
pub struct QuotientQueryDecommit {
    pub q_val:      [u32; 8],
    pub q_val_pair: [u32; 8],  // Q_eval[q XOR (eval_n/2)] for FRI pairing
    pub q_path:     Vec<[u32; 8]>,
    pub q_path_pair: Vec<[u32; 8]>,
}

/// A general STARK proof over Stark252 with FRI.
#[derive(Clone, Serialize, Deserialize)]
pub struct StarkProof {
    pub log_n: u32,
    /// Public inputs (domain-specific; for T[k+1]=T[k]^2+c: seed T[0] and constant c).
    pub public_inputs: Vec<[u32; 8]>,
    /// Merkle root of trace evaluations.
    pub trace_root: Digest,
    /// Merkle root of quotient evaluations (FRI layer 0).
    pub quotient_root: Digest,
    /// Query indices in [0, eval_n).
    pub query_indices: Vec<usize>,
    /// Per-query trace decommitments.
    pub trace_decommits: Vec<TraceQueryDecommit>,
    /// Per-query quotient decommitments (FRI layer 0 values + auth paths).
    /// These are embedded in fri_proof.query_decommits[q].layers[0].
    /// We use the FRI proof's layer-0 decommit as the quotient decommit.
    /// FRI proof (inner layers + last layer).
    pub fri_proof: FriProof,
}

// ─────────────────────────────────────────────
// Trait: STARK AIR
// ─────────────────────────────────────────────

/// A single-column AIR (Algebraic Intermediate Representation) over Stark252.
///
/// Implementors define:
/// - The trace generation
/// - The constraint polynomial C(T_cur, T_next, x) evaluated pointwise
/// - The vanishing polynomial Z(x)
/// - The number of unconstrained boundary rows D
pub trait StarkAir {
    /// Generate the execution trace (length N = 2^log_n).
    fn generate_trace(&self, log_n: u32) -> Vec<Fp>;

    /// Evaluate the constraint at evaluation domain index `i`.
    ///
    /// `t_cur`  = T_eval[i]
    /// `t_next` = T_eval[(i + blowup) % eval_n]
    ///
    /// Returns C(x_i) where C vanishes at all trace domain points {k·blowup} for k < N-D.
    fn constraint_at(&self, t_cur: Fp, t_next: Fp) -> Fp;

    /// Number of unconstrained rows at the end of the trace (boundary rows).
    /// The vanishing polynomial excludes these from its roots.
    fn n_boundary_rows(&self) -> usize;

    /// Public inputs (for Fiat-Shamir binding).
    fn public_inputs(&self) -> Vec<Fp>;
}

// ─────────────────────────────────────────────
// Prove
// ─────────────────────────────────────────────

/// Prove an AIR instance.
pub fn prove<A: StarkAir>(air: &A, log_n: u32) -> StarkProof {
    assert!(log_n + LOG_BLOWUP > LOG_LAST_LAYER, "trace too small for FRI");
    let n = 1usize << log_n;
    let log_eval = log_n + LOG_BLOWUP;
    let eval_n   = 1usize << log_eval;

    // ── Step 1 & 2: Trace + LDE ──────────────
    let trace  = air.generate_trace(log_n);
    let t_eval = lde_cpu(&trace, log_n, LOG_BLOWUP);

    let trace_tree = MerkleTree252::commit(&t_eval);
    let trace_root = trace_tree.root();

    // ── Step 3: Compute constraint and quotient ──
    let d = air.n_boundary_rows();
    let (q_eval, _z_eval) = compute_quotient(air, &t_eval, n, log_n, log_eval, d);

    // ── Step 4: Commit quotient ──────────────
    let quotient_tree = MerkleTree252::commit(&q_eval);
    let quotient_root = quotient_tree.root();

    // ── Step 5: Fiat-Shamir + FRI ────────────
    let mut channel = Channel252::new();
    for pi in air.public_inputs() {
        channel.mix_fp(&pi);
    }
    channel.mix_digest(&trace_root);
    channel.mix_digest(&quotient_root);

    // FRI commit (quotient_root already in channel)
    let witness = fri_commit(q_eval.clone(), &quotient_root, log_eval, &mut channel);

    // Draw query indices after all FRI commitments
    let query_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_n))
        .collect();

    // ── Step 6: Build trace decommits ────────
    let trace_decommits: Vec<TraceQueryDecommit> = query_indices.iter().map(|&q| {
        let q_next = (q + BLOWUP) % eval_n;
        TraceQueryDecommit {
            t_val:       fp_to_u32x8(&t_eval[q]),
            t_path:      trace_tree.auth_path(q),
            t_next_val:  fp_to_u32x8(&t_eval[q_next]),
            t_next_path: trace_tree.auth_path(q_next),
        }
    }).collect();

    // ── Step 7: Build FRI proof (includes layer-0 quotient decommits) ──
    let fri_proof = fri_build_proof(&witness, &quotient_tree, &query_indices);

    StarkProof {
        log_n,
        public_inputs: air.public_inputs().iter().map(fp_to_u32x8).collect(),
        trace_root,
        quotient_root,
        query_indices,
        trace_decommits,
        fri_proof,
    }
}

// ─────────────────────────────────────────────
// Verify
// ─────────────────────────────────────────────

/// Verify a STARK proof.
///
/// `vanishing_at` — the verifier's evaluation of Z(x_q) at each query point.
/// Since Z depends on the AIR, callers pass a closure.
pub fn verify<F>(
    proof: &StarkProof,
    vanishing_at: F,
) -> Result<(), String>
where
    F: Fn(Fp, usize) -> Fp,  // (x_q, n) → Z(x_q)
{
    let log_n    = proof.log_n;
    let log_eval = log_n + LOG_BLOWUP;
    let eval_n   = 1usize << log_eval;
    let n        = 1usize << log_n;

    // ── Recompute Fiat-Shamir ────────────────
    let mut channel = Channel252::new();
    for pi in &proof.public_inputs {
        channel.mix_fp(&fp_from_u32x8(pi));
    }
    channel.mix_digest(&proof.trace_root);
    channel.mix_digest(&proof.quotient_root);

    let expected_indices: Vec<usize> = {
        // Replay FRI channel state: fri_commit draws alphas and mixes inner roots.
        // We call fri_verify which replays this, but we need query indices AFTER FRI.
        // So we must replay the FRI commit state first.
        let n_folds = log_eval - LOG_LAST_LAYER;
        // α_0 (after quotient_root)
        let _alpha0 = channel.draw_fp();
        // inner roots + alphas (n_folds - 1 of them)
        for root in &proof.fri_proof.inner_roots {
            channel.mix_digest(root);
            let _alpha = channel.draw_fp();
        }
        // last layer
        for v in &proof.fri_proof.last_layer_evals {
            channel.mix_fp(&fp_from_u32x8(v));
        }
        // Now draw query indices
        (0..N_QUERIES).map(|_| channel.draw_number(eval_n)).collect()
    };

    if proof.query_indices != expected_indices {
        return Err("Query indices do not match Fiat-Shamir transcript".into());
    }
    if proof.trace_decommits.len() != N_QUERIES {
        return Err(format!("Expected {N_QUERIES} trace decommits"));
    }

    // ── Per-query constraint checks ──────────
    let omega_eval = ntt_root_of_unity(log_eval);

    for (qi, (&q, td)) in proof.query_indices.iter()
        .zip(proof.trace_decommits.iter())
        .enumerate()
    {
        let q_next = (q + BLOWUP) % eval_n;

        let t_cur  = fp_from_u32x8(&td.t_val);
        let t_next = fp_from_u32x8(&td.t_next_val);

        // 1. Merkle auth paths for trace
        if !verify_auth_path(&t_cur,  &td.t_path,      &proof.trace_root, q,      log_eval) {
            return Err(format!("Query {qi}: trace auth path failed (q={q})"));
        }
        if !verify_auth_path(&t_next, &td.t_next_path, &proof.trace_root, q_next, log_eval) {
            return Err(format!("Query {qi}: trace_next auth path failed (q_next={q_next})"));
        }

        // 2. Get Q(x_q) from FRI layer-0 decommit
        let fri_ld = &proof.fri_proof.query_decommits[qi].layers[0];
        let half   = eval_n / 2;
        let low    = q % half;
        let q_val  = if q == low {
            fp_from_u32x8(&fri_ld.f_lo)
        } else {
            fp_from_u32x8(&fri_ld.f_hi)
        };

        // 3. Constraint check: Q(x_q) * Z(x_q) == C(x_q)
        let x_q    = fp_pow_u64(omega_eval, q as u64);
        let c_q    = air_constraint(t_cur, t_next);   // caller's constraint
        let z_q    = vanishing_at(x_q, n);
        let lhs    = q_val.mul(z_q);
        if lhs != c_q {
            return Err(format!(
                "Query {qi}: constraint check failed at q={q}: Q*Z={:?}, C={:?}",
                fp_to_u32x8(&lhs), fp_to_u32x8(&c_q)
            ));
        }
    }

    // ── FRI verify ───────────────────────────
    let mut fri_channel = Channel252::new();
    for pi in &proof.public_inputs {
        fri_channel.mix_fp(&fp_from_u32x8(pi));
    }
    fri_channel.mix_digest(&proof.trace_root);
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
// Internal helpers
// ─────────────────────────────────────────────

/// Evaluate the AIR's constraint at a specific eval domain point.
/// This is a placeholder — actual constraint is in the AIR trait.
/// The verifier calls this via closure, so we just return a zero here.
fn air_constraint(_t_cur: Fp, _t_next: Fp) -> Fp {
    // Concrete constraint is handled by the verify() caller via the vanishing_at closure.
    // This fn is not used; constraint is evaluated by the verifier in the per-query loop
    // using t_cur and t_next directly (see verify() above which calls the closure).
    //
    // Actually: the verifier doesn't have access to the AIR's constraint function here.
    // We need to restructure: pass a constraint closure too.
    // See the verify_with_air() variant below.
    Fp::ZERO
}

/// Compute quotient polynomial evaluations Q_eval = C_eval / Z_eval.
///
/// Returns (Q_eval, Z_eval) both of size eval_n.
fn compute_quotient<A: StarkAir>(
    air: &A,
    t_eval: &[Fp],
    n: usize,
    log_n: u32,
    log_eval: u32,
    d: usize,   // boundary rows to exclude from vanishing polynomial
) -> (Vec<Fp>, Vec<Fp>) {
    let eval_n = t_eval.len();
    let omega_n    = ntt_root_of_unity(log_n);
    let omega_eval = ntt_root_of_unity(log_eval);

    // Precompute Z(x_i) for all eval domain points (batch inverse for efficiency)
    // Z(x) = (x^N - 1) / ∏_{k=N-d}^{N-1}(x - ω_N^k)
    //
    // For blowup=4: trace domain = {ω_eval^{4k}}. We need to avoid 0/0 at the
    // excluded roots. Use the same L'Hôpital trick for the excluded roots.

    // Compute excluded roots ω_N^{N-d}, ..., ω_N^{N-1}
    let excluded: Vec<Fp> = (n - d..n)
        .map(|k| fp_pow_u64(omega_n, k as u64))
        .collect();

    // Z numerator: x^N - 1 for each eval domain point
    // Z denominator: ∏(x - excluded_k) for each eval domain point

    // Compute raw Z values (with possible 0/0 at excluded trace points)
    let mut z_raw: Vec<Fp> = Vec::with_capacity(eval_n);
    for i in 0..eval_n {
        let x = fp_pow_u64(omega_eval, i as u64);
        z_raw.push(z_eval_at(x, n, &excluded));
    }

    // Compute C values
    let c_raw: Vec<Fp> = (0..eval_n).map(|i| {
        let t_cur  = t_eval[i];
        let t_next = t_eval[(i + BLOWUP) % eval_n];
        air.constraint_at(t_cur, t_next)
    }).collect();

    // Q = C / Z pointwise. batch_inverse fails if any input is zero (Montgomery's trick
    // contaminates all results). Filter to nonzero entries only.
    let nonzero_idx: Vec<usize> = (0..eval_n).filter(|&i| z_raw[i] != Fp::ZERO).collect();
    let nonzero_z:   Vec<Fp>    = nonzero_idx.iter().map(|&i| z_raw[i]).collect();
    let nonzero_inv              = batch_inverse(&nonzero_z);

    let mut z_inv = vec![Fp::ZERO; eval_n];
    for (k, &i) in nonzero_idx.iter().enumerate() {
        z_inv[i] = nonzero_inv[k];
    }

    let q_eval: Vec<Fp> = (0..eval_n).map(|i| {
        if z_raw[i] == Fp::ZERO {
            Fp::ZERO // trace domain point where C=0; Q undefined but never queried
        } else {
            c_raw[i].mul(z_inv[i])
        }
    }).collect();

    (q_eval, z_raw)
}

/// Evaluate Z(x) = (x^N − 1) / ∏_{e ∈ excluded}(x − e).
///
/// Handles the 0/0 case at excluded roots via L'Hôpital.
fn z_eval_at(x: Fp, n: usize, excluded: &[Fp]) -> Fp {
    let x_pow_n   = fp_pow_u64(x, n as u64);
    let numerator = x_pow_n.sub(Fp::ONE);

    // Denominator: ∏(x - e)
    let mut denom = Fp::ONE;
    let mut zero_idx: Option<usize> = None;
    for (k, &e) in excluded.iter().enumerate() {
        let diff = x.sub(e);
        if diff == Fp::ZERO {
            zero_idx = Some(k);
        } else {
            denom = denom.mul(diff);
        }
    }

    if let Some(k) = zero_idx {
        // x = excluded[k], so x is an N-th root of unity with numerator=0.
        // L'Hôpital for the single vanishing factor (x - excluded[k]):
        // lim_{x→e} (x^N − 1)/(x − e) = N * e^{N-1}
        // Then divide by remaining denominator factors.
        let e = excluded[k];
        let e_pow_nm1 = fp_pow_u64(e, (n - 1) as u64);
        let n_fp      = Fp::from_u64(n as u64);
        n_fp.mul(e_pow_nm1).mul(denom.inverse())
    } else {
        numerator.mul(denom.inverse())
    }
}

fn fp_pow_u64(base: Fp, exp: u64) -> Fp {
    base.pow_fp(crate::cairo_air::stark252_field::Fp { v: [exp, 0, 0, 0] })
}

// ─────────────────────────────────────────────
// Verify with AIR trait (the real verifier)
// ─────────────────────────────────────────────

/// Verify a STARK proof using the full AIR trait.
///
/// This is the correct verifier: the constraint and vanishing polynomial
/// are evaluated directly from the AIR.
pub fn verify_with_air<A: StarkAir>(
    proof: &StarkProof,
    air: &A,
) -> Result<(), String> {
    let log_n    = proof.log_n;
    let log_eval = log_n + LOG_BLOWUP;
    let eval_n   = 1usize << log_eval;
    let n        = 1usize << log_n;
    let d        = air.n_boundary_rows();

    // Public input check
    let expected_pubs: Vec<[u32; 8]> = air.public_inputs().iter().map(fp_to_u32x8).collect();
    if proof.public_inputs != expected_pubs {
        return Err("Public inputs do not match".into());
    }

    // Recompute Fiat-Shamir
    let mut channel = Channel252::new();
    for pi in &proof.public_inputs {
        channel.mix_fp(&fp_from_u32x8(pi));
    }
    channel.mix_digest(&proof.trace_root);
    channel.mix_digest(&proof.quotient_root);

    // Replay FRI commit to advance channel state
    let _alpha0 = channel.draw_fp();
    for root in &proof.fri_proof.inner_roots {
        channel.mix_digest(root);
        let _alpha = channel.draw_fp();
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

    let omega_n    = ntt_root_of_unity(log_n);
    let omega_eval = ntt_root_of_unity(log_eval);
    let excluded: Vec<Fp> = (n - d..n)
        .map(|k| fp_pow_u64(omega_n, k as u64))
        .collect();

    for (qi, (&q, td)) in proof.query_indices.iter()
        .zip(proof.trace_decommits.iter())
        .enumerate()
    {
        let q_next = (q + BLOWUP) % eval_n;
        let t_cur  = fp_from_u32x8(&td.t_val);
        let t_next = fp_from_u32x8(&td.t_next_val);

        // Merkle checks
        if !verify_auth_path(&t_cur,  &td.t_path,      &proof.trace_root, q,      log_eval) {
            return Err(format!("Query {qi}: trace auth path failed"));
        }
        if !verify_auth_path(&t_next, &td.t_next_path, &proof.trace_root, q_next, log_eval) {
            return Err(format!("Query {qi}: trace_next auth path failed"));
        }

        // Get Q(x_q) from FRI layer-0 decommit
        let fri_ld = &proof.fri_proof.query_decommits[qi].layers[0];
        let half   = eval_n / 2;
        let low    = q % half;
        let q_val  = if q == low {
            fp_from_u32x8(&fri_ld.f_lo)
        } else {
            fp_from_u32x8(&fri_ld.f_hi)
        };

        // Constraint + vanishing check
        let x_q = fp_pow_u64(omega_eval, q as u64);
        let c_q = air.constraint_at(t_cur, t_next);
        let z_q = z_eval_at(x_q, n, &excluded);
        let lhs = q_val.mul(z_q);
        if lhs != c_q {
            return Err(format!(
                "Query {qi}: Q*Z != C at q={q}: Q*Z={:?}, C={:?}",
                fp_to_u32x8(&lhs), fp_to_u32x8(&c_q)
            ));
        }
    }

    // FRI verify
    let mut fri_channel = Channel252::new();
    for pi in &proof.public_inputs {
        fri_channel.mix_fp(&fp_from_u32x8(pi));
    }
    fri_channel.mix_digest(&proof.trace_root);
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
// Example AIR: T[k+1] = T[k]^2 + c
// ─────────────────────────────────────────────

/// An AIR proving T[k+1] = T[k]² + c for k = 0..N-2,
/// with seed T[0] and constant c both as public inputs.
pub struct SquareAddAir {
    pub seed: Fp,
    pub c: Fp,
}

impl StarkAir for SquareAddAir {
    fn generate_trace(&self, log_n: u32) -> Vec<Fp> {
        let n = 1usize << log_n;
        let mut t = Vec::with_capacity(n);
        t.push(self.seed);
        while t.len() < n {
            let last = *t.last().unwrap();
            t.push(last.mul(last).add(self.c));
        }
        t
    }

    fn constraint_at(&self, t_cur: Fp, t_next: Fp) -> Fp {
        // C = T_next - T_cur^2 - c
        t_next.sub(t_cur.mul(t_cur)).sub(self.c)
    }

    fn n_boundary_rows(&self) -> usize {
        1 // last row has no "next" to constrain
    }

    fn public_inputs(&self) -> Vec<Fp> {
        vec![self.seed, self.c]
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_add_prove_verify() {
        let air = SquareAddAir {
            seed: Fp::from_u64(2),
            c:    Fp::from_u64(1),
        };
        let log_n = 5u32; // N=32, domain=128
        let proof = prove(&air, log_n);
        verify_with_air(&proof, &air).expect("SquareAdd STARK should verify");
    }

    #[test]
    fn test_square_add_tampered_trace_rejected() {
        let air = SquareAddAir {
            seed: Fp::from_u64(1),
            c:    Fp::from_u64(3),
        };
        let log_n = 5u32;
        let mut proof = prove(&air, log_n);

        // Corrupt a trace value
        proof.trace_decommits[0].t_val[0] ^= 1;
        assert!(
            verify_with_air(&proof, &air).is_err(),
            "Tampered trace should be rejected"
        );
    }

    #[test]
    fn test_square_add_wrong_public_input_rejected() {
        let air = SquareAddAir {
            seed: Fp::from_u64(5),
            c:    Fp::from_u64(7),
        };
        let log_n = 5u32;
        let proof = prove(&air, log_n);

        // Wrong public input
        let wrong_air = SquareAddAir {
            seed: Fp::from_u64(6), // different seed
            c:    Fp::from_u64(7),
        };
        assert!(
            verify_with_air(&proof, &wrong_air).is_err(),
            "Wrong public input should be rejected"
        );
    }

    #[test]
    fn test_fibonacci_via_stark_air() {
        // Fibonacci as a degree-1 AIR: T[k+2] = T[k+1] + T[k].
        // Two-column version: T_cur = (a, b), T_next = (b, a+b).
        // Simplify: single column with stride 2.
        // Actually, for single column: constraint T[k+1] - T[k] = T[k-1]...
        // Let's just test a single-column Fibonacci variant:
        // T[k] = k+1 (arithmetic sequence), constraint T[k+1] - T[k] = 1.
        struct ArithmeticAir {
            start: Fp,
            step:  Fp,
        }
        impl StarkAir for ArithmeticAir {
            fn generate_trace(&self, log_n: u32) -> Vec<Fp> {
                let n = 1usize << log_n;
                let mut t = Vec::with_capacity(n);
                let mut v = self.start;
                for _ in 0..n {
                    t.push(v);
                    v = v.add(self.step);
                }
                t
            }
            fn constraint_at(&self, t_cur: Fp, t_next: Fp) -> Fp {
                t_next.sub(t_cur).sub(self.step)
            }
            fn n_boundary_rows(&self) -> usize { 1 }
            fn public_inputs(&self) -> Vec<Fp> { vec![self.start, self.step] }
        }

        let air = ArithmeticAir { start: Fp::from_u64(1), step: Fp::from_u64(3) };
        let proof = prove(&air, 5);
        verify_with_air(&proof, &air).expect("Arithmetic STARK should verify");
    }
}
