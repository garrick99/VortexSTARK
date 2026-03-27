//! Fibonacci STARK prover over Stark252.
//!
//! # Protocol
//!
//! Given public inputs (a, b) and trace length N = 2^log_n, proves that the sequence
//! T[0]=a, T[1]=b, T[k+2]=T[k+1]+T[k] holds for N steps.
//!
//! ## Steps
//! 1. Generate Fibonacci trace T[0..N].
//! 2. Low-degree extension: extend T to evaluation domain D_eval = {ω_{4N}^i} via NTT.
//! 3. Commit trace → trace_root (Merkle tree over T_eval).
//! 4. Compute constraint polynomial C[i] = T_eval[(i+8)%4N] − T_eval[(i+4)%4N] − T_eval[i].
//! 5. Compute quotient Q(x) = C(x) / Z_{N-2}(x) where
//!       Z_{N-2}(x) = (x^N − 1) / ((x − e1)(x − e2))
//!    e1 = ω_N^{N-2}, e2 = ω_N^{N-1}.
//!    Since Q has degree 1 for a Fibonacci STARK, compute Q analytically from two safe points
//!    and extend Q_eval[i] = a_q * ω_{4N}^i + b_q over the full evaluation domain.
//! 6. Commit quotient → quotient_root.
//! 7. Fiat-Shamir: mix roots → draw query indices.
//! 8. Decommit T_eval and Q_eval at each query.

use super::field::{Fp, fp_to_u32x8, fp_from_u32x8, ntt_root_of_unity, batch_inverse, Channel252};
use super::ntt::lde_cpu;
use super::merkle::{MerkleTree252, Digest, verify_auth_path};

use serde::{Serialize, Deserialize};

/// Blowup factor for the evaluation domain (4×).
pub const LOG_BLOWUP: u32 = 2;
pub const BLOWUP: usize   = 1 << LOG_BLOWUP; // 4

/// Number of FRI queries.
pub const N_QUERIES: usize = 40;

/// ─────────────────────────────────────────────
/// Proof types
/// ─────────────────────────────────────────────

/// Decommitment for three trace evaluations at indices (q, q+blowup, q+2*blowup).
#[derive(Clone, Serialize, Deserialize)]
pub struct TraceDecommit {
    pub t0: [u32; 8],               // T_eval[q]
    pub t1: [u32; 8],               // T_eval[(q + blowup) % eval_n]
    pub t2: [u32; 8],               // T_eval[(q + 2*blowup) % eval_n]
    pub auth0: Vec<[u32; 8]>,
    pub auth1: Vec<[u32; 8]>,
    pub auth2: Vec<[u32; 8]>,
}

/// Decommitment for the quotient at index q.
#[derive(Clone, Serialize, Deserialize)]
pub struct QuotientDecommit {
    pub q: [u32; 8],
    pub auth: Vec<[u32; 8]>,
}

/// A Fibonacci STARK proof over Stark252.
#[derive(Clone, Serialize, Deserialize)]
pub struct Stark252Proof {
    /// log2 of trace length.
    pub log_n: u32,
    /// Public inputs.
    pub public_a: [u32; 8],
    pub public_b: [u32; 8],
    /// Commitments.
    pub trace_root:    Digest,
    pub quotient_root: Digest,
    /// Quotient polynomial Q(x) = q_a * x + q_b.
    pub q_a: [u32; 8],
    pub q_b: [u32; 8],
    /// Query indices in [0, eval_n).
    pub query_indices: Vec<usize>,
    /// Per-query decommitments.
    pub trace_decommits:    Vec<TraceDecommit>,
    pub quotient_decommits: Vec<QuotientDecommit>,
}

/// ─────────────────────────────────────────────
/// Prove
/// ─────────────────────────────────────────────

/// Prove the Fibonacci STARK: T[0]=a, T[1]=b, T[k+2]=T[k+1]+T[k], length N=2^log_n.
pub fn prove(a: Fp, b: Fp, log_n: u32) -> Stark252Proof {
    let n = 1usize << log_n;
    assert!(n >= 4, "trace must have at least 4 rows");

    let log_eval = log_n + LOG_BLOWUP;
    let eval_n   = 1usize << log_eval;

    // ── Step 1: Generate trace ──────────────────
    let trace = generate_trace(a, b, n);

    // ── Step 2: LDE ─────────────────────────────
    let t_eval = lde_cpu(&trace, log_n, LOG_BLOWUP);

    // ── Step 3: Commit trace ────────────────────
    let trace_tree = MerkleTree252::commit(&t_eval);
    let trace_root = trace_tree.root();

    // ── Step 4 & 5: Compute quotient ────────────
    let (q_a, q_b) = compute_quotient_polynomial(&t_eval, n, log_n, log_eval);

    // Build Q evaluation table: Q_eval[i] = q_a * ω_{eval}^i + q_b
    let omega_eval = ntt_root_of_unity(log_eval);
    let q_eval: Vec<Fp> = {
        let mut vals = Vec::with_capacity(eval_n);
        let mut xi = Fp::ONE; // ω_{eval}^0
        for _ in 0..eval_n {
            vals.push(q_a.mul(xi).add(q_b));
            xi = xi.mul(omega_eval);
        }
        vals
    };

    // ── Step 6: Commit quotient ─────────────────
    let quotient_tree = MerkleTree252::commit(&q_eval);
    let quotient_root = quotient_tree.root();

    // ── Step 7: Fiat-Shamir ─────────────────────
    let mut channel = Channel252::new();
    channel.mix_fp(&a);
    channel.mix_fp(&b);
    channel.mix_digest(&trace_root);
    channel.mix_digest(&quotient_root);
    channel.mix_fp(&q_a);
    channel.mix_fp(&q_b);

    let query_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_n))
        .collect();

    // ── Step 8: Decommit ────────────────────────
    let trace_decommits = query_indices.iter().map(|&q| {
        let q1 = (q + BLOWUP)     % eval_n;
        let q2 = (q + 2 * BLOWUP) % eval_n;
        TraceDecommit {
            t0: fp_to_u32x8(&t_eval[q]),
            t1: fp_to_u32x8(&t_eval[q1]),
            t2: fp_to_u32x8(&t_eval[q2]),
            auth0: trace_tree.auth_path(q),
            auth1: trace_tree.auth_path(q1),
            auth2: trace_tree.auth_path(q2),
        }
    }).collect();

    let quotient_decommits = query_indices.iter().map(|&q| {
        QuotientDecommit {
            q:    fp_to_u32x8(&q_eval[q]),
            auth: quotient_tree.auth_path(q),
        }
    }).collect();

    Stark252Proof {
        log_n,
        public_a: fp_to_u32x8(&a),
        public_b: fp_to_u32x8(&b),
        trace_root,
        quotient_root,
        q_a: fp_to_u32x8(&q_a),
        q_b: fp_to_u32x8(&q_b),
        query_indices,
        trace_decommits,
        quotient_decommits,
    }
}

/// ─────────────────────────────────────────────
/// Internal helpers
/// ─────────────────────────────────────────────

/// Generate Fibonacci trace: T[0]=a, T[1]=b, T[k+2]=T[k+1]+T[k].
fn generate_trace(a: Fp, b: Fp, n: usize) -> Vec<Fp> {
    let mut t = Vec::with_capacity(n);
    t.push(a);
    t.push(b);
    while t.len() < n {
        let l = t.len();
        t.push(t[l - 1].add(t[l - 2]));
    }
    t
}

/// Compute the degree-1 quotient polynomial Q(x) = a_q * x + b_q.
///
/// Approach: evaluate C(x)/Z_{N-2}(x) at two safe non-trace evaluation points
/// (indices 1 and 3 in D_eval, which are never in the trace domain since blowup=4)
/// then fit a line.
fn compute_quotient_polynomial(
    t_eval: &[Fp],
    n: usize,
    log_n: u32,
    log_eval: u32,
) -> (Fp, Fp) {
    let eval_n = t_eval.len();
    let step = BLOWUP;

    let omega_eval = ntt_root_of_unity(log_eval);
    let omega_n    = ntt_root_of_unity(log_n);
    let e1 = fp_pow_u64(omega_n, (n - 2) as u64); // ω_N^{N-2}
    let e2 = fp_pow_u64(omega_n, (n - 1) as u64); // ω_N^{N-1}
    let omega_4 = ntt_root_of_unity(2); // primitive 4th root of unity

    // Evaluate at index i=1 (x = ω_{4N}^1)
    let x1 = omega_eval; // ω_{4N}^1
    let c1  = constraint_at(t_eval, 1, step, eval_n);
    let z1  = vanishing_at(x1, n, e1, e2, omega_4);
    let q1  = c1.mul(z1.inverse());

    // Evaluate at index i=3 (x = ω_{4N}^3)
    let x3 = fp_pow_u64(omega_eval, 3); // ω_{4N}^3
    let c3  = constraint_at(t_eval, 3, step, eval_n);
    let z3  = vanishing_at(x3, n, e1, e2, omega_4);
    let q3  = c3.mul(z3.inverse());

    // Fit Q(x) = a_q * x + b_q through (x1, q1) and (x3, q3)
    let dx = x3.sub(x1);
    let dq = q3.sub(q1);
    let a_q = dq.mul(dx.inverse());
    let b_q = q1.sub(a_q.mul(x1));

    (a_q, b_q)
}

/// Compute C(x) = T(ω²x) − T(ωx) − T(x) at evaluation domain index i.
fn constraint_at(t_eval: &[Fp], i: usize, step: usize, eval_n: usize) -> Fp {
    let t0 = t_eval[i];
    let t1 = t_eval[(i + step) % eval_n];
    let t2 = t_eval[(i + 2 * step) % eval_n];
    t2.sub(t1).sub(t0)
}

/// Compute Z_{N-2}(x) = (x^N − 1) / ((x − e1)(x − e2)).
///
/// For non-trace-domain points, x^N − 1 ≠ 0 and the denominators are nonzero.
/// omega_4_pow_i = x^N (which cycles as ω_4^{i mod 4} for i in the eval domain).
fn vanishing_at(x: Fp, n: usize, e1: Fp, e2: Fp, omega_4: Fp) -> Fp {
    // x^N for x in D_eval:
    // x = ω_{4N}^i → x^N = ω_4^i (since ω_{4N}^N = ω_4)
    // But we have x directly, not i. So compute x^N via exponentiation.
    let x_pow_n = fp_pow_u64(x, n as u64);
    let numerator   = x_pow_n.sub(Fp::ONE);
    let d1          = x.sub(e1);
    let d2          = x.sub(e2);
    let denominator = d1.mul(d2);
    numerator.mul(denominator.inverse())
}

/// Compute base^exp for a u64 exponent.
fn fp_pow_u64(base: Fp, exp: u64) -> Fp {
    base.pow_fp(crate::cairo_air::stark252_field::Fp { v: [exp, 0, 0, 0] })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_generation() {
        let a = Fp::from_u64(1);
        let b = Fp::from_u64(1);
        let trace = generate_trace(a, b, 8);
        // Verify Fibonacci relation
        for i in 0..trace.len() - 2 {
            assert_eq!(
                trace[i].add(trace[i + 1]),
                trace[i + 2],
                "Fibonacci failed at i={i}"
            );
        }
    }

    #[test]
    fn test_constraint_zero_at_trace_points() {
        let log_n = 4u32;
        let log_eval = log_n + LOG_BLOWUP as u32;
        let n = 1usize << log_n;
        let eval_n = 1usize << log_eval;
        let a = Fp::from_u64(1);
        let b = Fp::from_u64(1);
        let trace = generate_trace(a, b, n);
        let t_eval = lde_cpu(&trace, log_n, LOG_BLOWUP);

        // At trace domain points (indices 0, 4, 8, ..., 4*(N-3)), C should be 0
        for k in 0..(n - 2) {
            let i = k * BLOWUP;
            let c = constraint_at(&t_eval, i, BLOWUP, eval_n);
            assert_eq!(c, Fp::ZERO, "Constraint nonzero at trace point k={k}");
        }
    }

    #[test]
    fn test_prove_small() {
        let a = Fp::from_u64(1);
        let b = Fp::from_u64(1);
        let log_n = 4u32;
        let proof = prove(a, b, log_n);
        // Smoke test: verify the proof using the verifier
        let result = super::super::verifier::verify(&proof);
        assert!(result.is_ok(), "Proof failed to verify: {:?}", result);
    }
}
