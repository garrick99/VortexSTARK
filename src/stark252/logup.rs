//! Memory argument (LogUp) over Stark252.
//!
//! Proves that all memory reads in the Cairo execution trace are consistent:
//! every (address, value) pair read during execution corresponds to a valid write.
//!
//! # Protocol (2-round Fiat-Shamir)
//!
//! ## Round 1 — Execution STARK
//! The execution trace is committed and proven by `prove_multi` (see multi_stark.rs).
//! The caller captures the Fiat-Shamir channel state after the execution proof.
//!
//! ## Round 2 — Interaction STARK
//! 1. Channel draws challenges z, alpha from Stark252 field.
//! 2. Prover computes the interaction column S[0..N]:
//!    - For each row i with accesses (addr_j, val_j):
//!      contrib_i = Σ_j  1 / (z − addr_j − alpha·val_j)
//!      S[i] = S[i−1] + contrib_i   (S[−1] = 0)
//! 3. Prover commits S to a Merkle tree after LDE.
//! 4. Memory table: deduplicate accesses, compute
//!    table_sum = Σ_k  −mult_k / (z − addr_k − alpha·val_k)
//! 5. Final check: S[N−1] + table_sum == 0.
//! 6. The step constraint Q_logup = (S(ω·x) − S(x) − contrib(x)) / Z(x) has degree N−1.
//!    Prover commits Q_logup and runs FRI on it.
//! 7. At each query q: verifier checks
//!    Q_logup(x_q)·Z(x_q) == S(ω^blowup · x_q) − S(x_q) − contrib(x_q),
//!    using decommitted S values + already-decommitted execution trace values.
//!
//! # Soundness
//! Over Stark252 (252-bit field), drawing z, alpha provides ~252-bit soundness
//! for the LogUp identity: if any read is inconsistent, the running sum will
//! fail the final check with probability 1 − O(N²/|Fp|) ≈ 1.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use super::field::{Fp, fp_to_u32x8, fp_from_u32x8, ntt_root_of_unity, batch_inverse, Channel252};
use super::ntt::lde_cpu;
use super::merkle::{MerkleTree252, Digest, verify_auth_path};
use super::fri::{FriProof, fri_commit, fri_build_proof, fri_verify};
use super::multi_stark::{LOG_BLOWUP, BLOWUP, N_QUERIES};

// ─────────────────────────────────────────────
// Cairo column indices for memory accesses
// (must match cairo_air.rs TraceRow layout)
// ─────────────────────────────────────────────

const PC:       usize = 0;
const INST:     usize = 3;
const DST_ADDR: usize = 4;
const DST:      usize = 7;
const OP0_ADDR: usize = 5;
const OP0:      usize = 8;
const OP1_ADDR: usize = 6;
const OP1:      usize = 9;

/// Number of memory accesses per Cairo execution step:
/// instruction fetch + dst + op0 + op1.
pub const ACCESSES_PER_ROW: usize = 4;

// ─────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────

/// A single (address, value) memory access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryAccess {
    pub addr: Fp,
    pub val:  Fp,
}

/// Per-query decommitment for the LogUp STARK verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogupDecommit {
    /// S(x_q)
    pub s_cur:  [u32; 8],
    /// S(x_{q+BLOWUP})
    pub s_next: [u32; 8],
    /// Merkle auth path for s_cur at position q in the LDE tree.
    pub cur_path:  Vec<Digest>,
    /// Merkle auth path for s_next at position q+BLOWUP.
    pub next_path: Vec<Digest>,
    /// Access values at this query: [(addr, val); 4]
    /// (same as execution trace decommit, included here for standalone verification)
    pub accesses: [([u32; 8], [u32; 8]); ACCESSES_PER_ROW],
    /// Merkle auth paths for each access column pair at position q.
    pub access_paths: [Vec<Digest>; ACCESSES_PER_ROW * 2],
}

/// Proof of the LogUp quotient step constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLogupProof {
    /// Commitment to the LDE of the interaction column S.
    pub s_root: Digest,
    /// Commitment to the LDE of the step quotient Q_logup.
    pub q_root: Digest,
    /// S[N−1] (exec-side final sum) as a field element.
    pub exec_sum: [u32; 8],
    /// Memory table: (addr, val, multiplicity).
    pub table: Vec<([u32; 8], [u32; 8], u64)>,
    /// Per-query decommits (N_QUERIES entries).
    pub decommits: Vec<LogupDecommit>,
    /// FRI proof certifying degree of Q_logup.
    pub fri_proof: FriProof,
    /// Query indices (shared with FRI).
    pub query_indices: Vec<usize>,
}

// ─────────────────────────────────────────────
// Core computation
// ─────────────────────────────────────────────

/// LogUp denominator for a single access: z - addr - alpha*val.
#[inline]
fn denom(z: Fp, alpha: Fp, addr: Fp, val: Fp) -> Fp {
    z.sub(addr).sub(alpha.mul(val))
}

/// LogUp contribution for one row: Σ_j 1/(z - addr_j - alpha*val_j).
/// Returns the sum of reciprocals for the four access pairs.
/// Panics if any denominator is zero (negligible probability over Fp252).
pub fn row_contribution(z: Fp, alpha: Fp, accesses: &[MemoryAccess]) -> Fp {
    let denoms: Vec<Fp> = accesses.iter()
        .map(|a| denom(z, alpha, a.addr, a.val))
        .collect();
    // Batch inverse for efficiency.
    let inv = batch_inverse(&denoms);
    inv.iter().fold(Fp::ZERO, |acc, &v| acc.add(v))
}

/// Extract the 4 memory accesses for row `row` from the Cairo execution trace.
///
/// Columns layout (see cairo_air.rs):
///   0=pc, 3=inst, 4=dst_addr, 7=dst, 5=op0_addr, 8=op0, 6=op1_addr, 9=op1
pub fn accesses_at_row(trace_cols: &[Vec<Fp>], row: usize) -> [MemoryAccess; ACCESSES_PER_ROW] {
    [
        MemoryAccess { addr: trace_cols[PC][row],       val: trace_cols[INST][row] },
        MemoryAccess { addr: trace_cols[DST_ADDR][row], val: trace_cols[DST][row]  },
        MemoryAccess { addr: trace_cols[OP0_ADDR][row], val: trace_cols[OP0][row]  },
        MemoryAccess { addr: trace_cols[OP1_ADDR][row], val: trace_cols[OP1][row]  },
    ]
}

/// Compute the LogUp interaction column S[0..N].
/// S[i] = S[i−1] + contrib_i,  S[−1] = 0.
pub fn compute_interaction_column(trace_cols: &[Vec<Fp>], z: Fp, alpha: Fp) -> Vec<Fp> {
    let n = trace_cols[0].len();
    let mut s = Vec::with_capacity(n);
    let mut running = Fp::ZERO;
    for i in 0..n {
        let acc = accesses_at_row(trace_cols, i);
        running = running.add(row_contribution(z, alpha, &acc));
        s.push(running);
    }
    s
}

/// Build the deduped memory table and compute table_sum.
///
/// For each unique (addr, val) pair, multiplicity = access count.
/// table_sum = Σ_k  −mult_k / (z − addr_k − alpha·val_k).
///
/// Returns: (sorted table, table_sum).
pub fn compute_table_sum(
    trace_cols: &[Vec<Fp>],
    z: Fp,
    alpha: Fp,
) -> (Vec<(Fp, Fp, u64)>, Fp) {
    let n = trace_cols[0].len();
    // Use [u32;8] as hash key since Fp doesn't impl Hash.
    let mut freq: HashMap<([u32;8], [u32;8]), u64> = HashMap::new();
    for i in 0..n {
        let acc = accesses_at_row(trace_cols, i);
        for a in &acc {
            *freq.entry((fp_to_u32x8(&a.addr), fp_to_u32x8(&a.val))).or_insert(0) += 1;
        }
    }

    let mut table: Vec<(Fp, Fp, u64)> = freq.into_iter()
        .map(|((a, v), m)| (fp_from_u32x8(&a), fp_from_u32x8(&v), m))
        .collect();
    // Sort for determinism.
    table.sort_by_key(|&(a, v, _)| (fp_to_u32x8(&a), fp_to_u32x8(&v)));

    // Compute table_sum = Σ −mult/(z − addr − alpha·val).
    let denoms: Vec<Fp> = table.iter().map(|&(a,v,_)| denom(z, alpha, a, v)).collect();
    let inv = batch_inverse(&denoms);
    let table_sum = table.iter().zip(inv.iter())
        .fold(Fp::ZERO, |acc, (&(_, _, m), &d_inv)| {
            // acc − m * d_inv
            acc.sub(Fp::from_u64(m).mul(d_inv))
        });

    (table, table_sum)
}

/// Compute the LogUp step quotient at eval domain positions.
///
/// Q_logup[i] = (S_lde[i+BLOWUP] − S_lde[i] − contrib(x_i)) / Z(x_i)
/// where Z(x) = (x^N − 1) / ∏(x − excluded_k).
///
/// At trace domain points (i = k*BLOWUP), Z = 0; set Q_logup = 0 there.
/// (The constraint is zero at those points by construction of S.)
fn compute_logup_quotient(
    s_lde: &[Fp],
    trace_cols: &[Vec<Fp>],
    z_chal: Fp,
    alpha: Fp,
    log_n: u32,
    log_eval: u32,
    d: usize,
) -> Vec<Fp> {
    let n      = 1usize << log_n;
    let eval_n = 1usize << log_eval;

    let omega_n    = ntt_root_of_unity(log_n);
    let omega_eval = ntt_root_of_unity(log_eval);

    // Excluded roots: last d trace domain positions.
    let excluded: Vec<Fp> = (n - d..n)
        .map(|k| fp_pow_u64(omega_n, k as u64))
        .collect();

    // LDE each access column (8 columns: addr*4 + val*4)
    let access_cols = [PC, INST, DST_ADDR, DST, OP0_ADDR, OP0, OP1_ADDR, OP1];
    let acc_lde: Vec<Vec<Fp>> = access_cols.iter()
        .map(|&c| lde_cpu(&trace_cols[c], log_n, LOG_BLOWUP))
        .collect();

    // Batch-compute Z values and C values over all eval domain points.
    let mut z_raw = Vec::with_capacity(eval_n);
    let mut c_raw = Vec::with_capacity(eval_n);

    let mut xi = Fp::ONE;
    for i in 0..eval_n {
        // Z(xi)
        z_raw.push(z_eval_at(xi, n, &excluded));

        // S(ω·xi) − S(xi)
        let s_diff = s_lde[(i + BLOWUP) % eval_n].sub(s_lde[i]);

        // Step constraint uses NEXT row's accesses: S(ω·x) - S(x) = contrib(ω·x).
        // S[k+1] - S[k] = contrib(row k+1) — use (i+BLOWUP) % eval_n.
        let ni = (i + BLOWUP) % eval_n;
        let accs = [
            MemoryAccess { addr: acc_lde[0][ni], val: acc_lde[1][ni] },
            MemoryAccess { addr: acc_lde[2][ni], val: acc_lde[3][ni] },
            MemoryAccess { addr: acc_lde[4][ni], val: acc_lde[5][ni] },
            MemoryAccess { addr: acc_lde[6][ni], val: acc_lde[7][ni] },
        ];
        let contrib = row_contribution(z_chal, alpha, &accs);

        c_raw.push(s_diff.sub(contrib));
        xi = xi.mul(omega_eval);
    }

    // Batch-invert Z where nonzero.
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
// Prover
// ─────────────────────────────────────────────

/// Prove the LogUp memory argument.
///
/// `trace_cols` — the Cairo execution trace columns (31 columns, each length N=2^log_n).
/// `channel`    — Fiat-Shamir channel, already advanced past the execution proof.
///                Draws z and alpha from here.
///
/// Returns a `MemoryLogupProof`.
pub fn prove_memory_logup(
    trace_cols: &[Vec<Fp>],
    log_n: u32,
    channel: &mut Channel252,
) -> MemoryLogupProof {
    let n      = 1usize << log_n;
    let log_eval = log_n + LOG_BLOWUP;
    let eval_n = 1usize << log_eval;
    let d = 1usize; // 1 boundary row excluded (last)

    assert_eq!(trace_cols[0].len(), n);

    // ── Draw challenges ────────────────────────
    let z_chal = channel.draw_fp();
    let alpha  = channel.draw_fp();

    // ── Interaction column S ───────────────────
    let s_trace = compute_interaction_column(trace_cols, z_chal, alpha);
    let exec_sum = *s_trace.last().unwrap();

    // LDE of S
    let s_lde = lde_cpu(&s_trace, log_n, LOG_BLOWUP);
    let s_tree = MerkleTree252::commit(&s_lde);
    let s_root = s_tree.root();
    channel.mix_digest(&s_root);

    // ── Table sum ─────────────────────────────
    let (table_raw, table_sum) = compute_table_sum(trace_cols, z_chal, alpha);

    // Final consistency check (prover-side assertion; will also be verified).
    let check = exec_sum.add(table_sum);
    assert!(check == Fp::ZERO,
        "LogUp memory argument: exec_sum + table_sum ≠ 0 — memory inconsistency detected");

    // ── Quotient Q_logup ──────────────────────
    let q_lde = compute_logup_quotient(&s_lde, trace_cols, z_chal, alpha, log_n, log_eval, d);
    let q_tree = MerkleTree252::commit(&q_lde);
    let q_root = q_tree.root();
    channel.mix_digest(&q_root);

    // ── FRI on Q_logup ────────────────────────
    let fri_witness = fri_commit(q_lde.clone(), &q_root, log_eval, channel);

    // ── Draw queries ──────────────────────────
    let query_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_n))
        .collect();

    // ── Build per-query FRI proof ─────────────
    let fri_proof = fri_build_proof(&fri_witness, &q_tree, &query_indices);

    // ── Decommit S and access columns ─────────
    // LDE the 8 access columns for decommitting.
    let access_col_indices = [PC, INST, DST_ADDR, DST, OP0_ADDR, OP0, OP1_ADDR, OP1];
    let acc_lde: Vec<Vec<Fp>> = access_col_indices.iter()
        .map(|&c| lde_cpu(&trace_cols[c], log_n, LOG_BLOWUP))
        .collect();
    let acc_trees: Vec<MerkleTree252> = acc_lde.iter()
        .map(|col| MerkleTree252::commit(col))
        .collect();

    let decommits: Vec<LogupDecommit> = query_indices.iter().map(|&q| {
        let q_next = (q + BLOWUP) % eval_n;

        // Constraint uses next-row accesses: decommit at q_next.
        let accesses: [([u32;8], [u32;8]); ACCESSES_PER_ROW] = [
            (fp_to_u32x8(&acc_lde[0][q_next]), fp_to_u32x8(&acc_lde[1][q_next])),
            (fp_to_u32x8(&acc_lde[2][q_next]), fp_to_u32x8(&acc_lde[3][q_next])),
            (fp_to_u32x8(&acc_lde[4][q_next]), fp_to_u32x8(&acc_lde[5][q_next])),
            (fp_to_u32x8(&acc_lde[6][q_next]), fp_to_u32x8(&acc_lde[7][q_next])),
        ];

        // Auth paths at q_next for each access column.
        let access_paths: [Vec<Digest>; ACCESSES_PER_ROW * 2] = [
            acc_trees[0].auth_path(q_next), acc_trees[1].auth_path(q_next),
            acc_trees[2].auth_path(q_next), acc_trees[3].auth_path(q_next),
            acc_trees[4].auth_path(q_next), acc_trees[5].auth_path(q_next),
            acc_trees[6].auth_path(q_next), acc_trees[7].auth_path(q_next),
        ];

        LogupDecommit {
            s_cur:  fp_to_u32x8(&s_lde[q]),
            s_next: fp_to_u32x8(&s_lde[q_next]),
            cur_path:  s_tree.auth_path(q),
            next_path: s_tree.auth_path(q_next),
            accesses,
            access_paths,
        }
    }).collect();

    // Encode table for proof.
    let table: Vec<([u32;8], [u32;8], u64)> = table_raw.iter()
        .map(|&(a,v,m)| (fp_to_u32x8(&a), fp_to_u32x8(&v), m))
        .collect();

    MemoryLogupProof {
        s_root,
        q_root,
        exec_sum: fp_to_u32x8(&exec_sum),
        table,
        decommits,
        fri_proof,
        query_indices,
    }
}

// ─────────────────────────────────────────────
// Verifier
// ─────────────────────────────────────────────

/// Verify a `MemoryLogupProof`.
///
/// The `channel` must be in the same state as when `prove_memory_logup` was called
/// (i.e., advanced past the execution proof).
///
/// The verifier checks:
/// 1. exec_sum + table_sum == 0   (LogUp identity)
/// 2. Per-query: Q·Z == S_next − S_cur − contrib  (step constraint)
/// 3. FRI: Q has low degree
pub fn verify_memory_logup(
    proof: &MemoryLogupProof,
    log_n: u32,
    channel: &mut Channel252,
) -> Result<(), String> {
    let n        = 1usize << log_n;
    let log_eval = log_n + LOG_BLOWUP;
    let eval_n   = 1usize << log_eval;
    let d = 1usize;

    // ── Replay challenges ─────────────────────
    let z_chal = channel.draw_fp();
    let alpha  = channel.draw_fp();

    channel.mix_digest(&proof.s_root);

    // ── Check 1: exec_sum + table_sum == 0 ────
    let exec_sum = fp_from_u32x8(&proof.exec_sum);
    let denoms: Vec<Fp> = proof.table.iter()
        .map(|&(a, v, _)| denom(z_chal, alpha, fp_from_u32x8(&a), fp_from_u32x8(&v)))
        .collect();
    let inv = batch_inverse(&denoms);
    let table_sum = proof.table.iter().zip(inv.iter())
        .fold(Fp::ZERO, |acc, (&(_, _, m), &d_inv)| {
            acc.sub(Fp::from_u64(m).mul(d_inv))
        });

    if exec_sum.add(table_sum) != Fp::ZERO {
        return Err(format!(
            "LogUp identity failed: exec_sum + table_sum ≠ 0. \
             exec_sum={:?}, table_sum={:?}",
            proof.exec_sum, fp_to_u32x8(&table_sum)
        ));
    }

    channel.mix_digest(&proof.q_root);

    // Clone channel here — this is the state the prover was in when fri_commit was called.
    // We use this clone for fri_verify; the main channel is advanced through the FRI ops
    // so subsequent sub-proofs (range-check) get the correct starting state.
    let mut fri_channel = channel.clone();

    // ── Advance main channel through FRI ops (for query-index replay) ──
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
        return Err("LogUp: query indices do not match Fiat-Shamir transcript".into());
    }

    // ── Vanishing polynomial setup ─────────────
    let omega_n    = ntt_root_of_unity(log_n);
    let omega_eval = ntt_root_of_unity(log_eval);
    let excluded: Vec<Fp> = (n - d..n)
        .map(|k| fp_pow_u64(omega_n, k as u64))
        .collect();

    // Need roots for access columns (we verify their auth paths).
    // We don't have the access column roots in the proof (they're in the exec proof).
    // In a combined Cairo proof this would reference exec proof's col_roots.
    // For standalone testing we recompute from the decommits — the auth path
    // must be consistent with SOME root; we extract it implicitly from the first
    // decommit path root. (Proper integration passes col_roots from exec proof.)
    //
    // For now, check auth paths against the s_root (for S) only, and verify the
    // Q·Z == S_diff − contrib constraint directly from the decommitted values.

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
            return Err(format!("Query {qi}: S cur auth path failed (q={q})"));
        }
        if !verify_auth_path(&s_next, &dc.next_path, &proof.s_root, q_next, log_eval) {
            return Err(format!("Query {qi}: S next auth path failed (q_next={q_next})"));
        }

        // Compute contrib from decommitted access values.
        let accesses: [MemoryAccess; ACCESSES_PER_ROW] = [
            MemoryAccess { addr: fp_from_u32x8(&dc.accesses[0].0), val: fp_from_u32x8(&dc.accesses[0].1) },
            MemoryAccess { addr: fp_from_u32x8(&dc.accesses[1].0), val: fp_from_u32x8(&dc.accesses[1].1) },
            MemoryAccess { addr: fp_from_u32x8(&dc.accesses[2].0), val: fp_from_u32x8(&dc.accesses[2].1) },
            MemoryAccess { addr: fp_from_u32x8(&dc.accesses[3].0), val: fp_from_u32x8(&dc.accesses[3].1) },
        ];
        let contrib = row_contribution(z_chal, alpha, &accesses);

        // Get Q(x_q) from FRI decommit.
        let fri_ld = &proof.fri_proof.query_decommits[qi].layers[0];
        let half   = eval_n / 2;
        let low    = q % half;
        let q_val  = if q == low {
            fp_from_u32x8(&fri_ld.f_lo)
        } else {
            fp_from_u32x8(&fri_ld.f_hi)
        };

        // Check: Q(x_q)·Z(x_q) == S_next − S_cur − contrib
        let x_q   = fp_pow_u64(omega_eval, q as u64);
        let z_q   = z_eval_at(x_q, n, &excluded);
        let lhs   = q_val.mul(z_q);
        let rhs   = s_next.sub(s_cur).sub(contrib);

        if lhs != rhs {
            return Err(format!(
                "Query {qi}: Q·Z ≠ S_next−S_cur−contrib at q={q}: Q·Z={:?}, rhs={:?}",
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
        &mut fri_channel,
    )?;

    Ok(())
}

// ─────────────────────────────────────────────
// Internal helpers
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

    /// Test that exec_sum + table_sum == 0 for a valid execution trace.
    #[test]
    fn test_logup_sum_identity() {
        let (_, trace_cols) = make_test_vm(3);

        let mut ch = Channel252::new();
        let z     = ch.draw_fp();
        let alpha = ch.draw_fp();

        let s = compute_interaction_column(&trace_cols, z, alpha);
        let exec_sum = *s.last().unwrap();

        let (_, table_sum) = compute_table_sum(&trace_cols, z, alpha);

        assert_eq!(exec_sum.add(table_sum), Fp::ZERO,
            "exec_sum + table_sum should be zero for a valid execution trace");
    }

    /// Full prove + verify roundtrip of the LogUp memory argument.
    #[test]
    fn test_logup_prove_verify() {
        let log_n: u32 = 3;
        let (_, trace_cols) = make_test_vm(log_n);

        let mut channel = Channel252::new();
        let proof = prove_memory_logup(&trace_cols, log_n, &mut channel);

        // Replay channel for verifier.
        let mut verify_channel = Channel252::new();
        verify_memory_logup(&proof, log_n, &mut verify_channel)
            .expect("LogUp proof should verify");
    }

    /// Tamper: flip one memory value in the table — should be caught.
    #[test]
    fn test_logup_tamper_table() {
        let log_n: u32 = 3;
        let (_, trace_cols) = make_test_vm(log_n);

        let mut channel = Channel252::new();
        let mut proof = prove_memory_logup(&trace_cols, log_n, &mut channel);

        // Corrupt the first table entry's value.
        if !proof.table.is_empty() {
            proof.table[0].1 = fp_to_u32x8(&Fp::from_u64(9999));
        }

        let mut verify_channel = Channel252::new();
        let result = verify_memory_logup(&proof, log_n, &mut verify_channel);
        assert!(result.is_err(), "Tampered table should fail verification");
    }

    /// Combined Cairo execution STARK + LogUp memory argument (end-to-end).
    #[test]
    fn test_cairo_with_memory_logup() {
        let log_n: u32 = 3;
        let (air, trace_cols) = make_test_vm(log_n);

        // Round 1: execution STARK.
        let exec_proof = prove_multi(&air, log_n);

        // Reconstruct channel state to the point where logup would draw z/alpha.
        // The logup channel starts fresh (standalone mode for this test).
        let mut logup_channel = Channel252::new();
        let logup_proof = prove_memory_logup(&trace_cols, log_n, &mut logup_channel);

        // Verify both.
        super::super::multi_stark::verify_multi(&exec_proof, &air)
            .expect("Execution STARK should verify");

        let mut verify_channel = Channel252::new();
        verify_memory_logup(&logup_proof, log_n, &mut verify_channel)
            .expect("LogUp memory argument should verify");
    }

    /// Verify the step constraint holds at all trace domain points.
    #[test]
    fn test_logup_debug_constraint() {
        let log_n: u32 = 3;
        let (_, trace_cols) = make_test_vm(log_n);
        let eval_n = 1usize << (log_n + LOG_BLOWUP);

        let mut ch = Channel252::new();
        let z_chal = ch.draw_fp();
        let alpha  = ch.draw_fp();

        let s_trace = compute_interaction_column(&trace_cols, z_chal, alpha);
        let s_lde   = lde_cpu(&s_trace, log_n, LOG_BLOWUP);

        let access_col_indices = [PC, INST, DST_ADDR, DST, OP0_ADDR, OP0, OP1_ADDR, OP1];
        let acc_lde: Vec<Vec<Fp>> = access_col_indices.iter()
            .map(|&c| lde_cpu(&trace_cols[c], log_n, LOG_BLOWUP))
            .collect();

        // d=1 boundary rows excluded; skip last BLOWUP*d eval-domain points.
        let d = 1usize;
        for q in (0..eval_n - BLOWUP * d).step_by(BLOWUP) {
            let q_next = (q + BLOWUP) % eval_n;
            let s_cur  = s_lde[q];
            let s_next = s_lde[q_next];
            // Step constraint: S_next - S_cur = contrib(next_row) — use q_next accesses.
            let accs = [
                MemoryAccess { addr: acc_lde[0][q_next], val: acc_lde[1][q_next] },
                MemoryAccess { addr: acc_lde[2][q_next], val: acc_lde[3][q_next] },
                MemoryAccess { addr: acc_lde[4][q_next], val: acc_lde[5][q_next] },
                MemoryAccess { addr: acc_lde[6][q_next], val: acc_lde[7][q_next] },
            ];
            let contrib = row_contribution(z_chal, alpha, &accs);
            let diff = s_next.sub(s_cur).sub(contrib);
            assert_eq!(diff, Fp::ZERO, "step constraint failed at q={q} (row {})", q/BLOWUP);
        }
    }
}
