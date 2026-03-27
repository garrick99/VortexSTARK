//! Standard FRI (Fast Reed-Solomon IOP of Proximity) over Stark252.
//!
//! # Protocol
//!
//! Given a polynomial f of degree < D committed on evaluation domain D_0 = {ω_M^i}
//! (where M = 4D, blowup=4):
//!
//! **Commit phase** (log_n fold rounds):
//! 1. Mix layer-0 root (caller's commitment) into channel → draw α_0.
//! 2. Fold: g[i] = (f[i] + f[i+M/2])/2 + α_0·(f[i] − f[i+M/2])/(2·ω_M^i)
//!    Result: g on domain of size M/2.
//! 3. Commit g → inner_roots[0]. Mix into channel → α_1. Repeat.
//! 4. Stop when domain size = 2^LOG_LAST_LAYER. Send final evaluations.
//!
//! **Query phase**:
//! For each query q ∈ [0, M):
//!   At layer l: low = q mod (M_l/2), high = low + M_l/2.
//!   Decommit f_l[low] and f_l[high].
//!   Verify fold: expected_next = (f_lo + f_hi)/2 + α_l·(f_lo − f_hi)/(2·ω_{M_l}^low)
//!   equals f_{l+1}[low] (decommitted from next layer).
//!
//! **Soundness**: each query adds log|D| bits of security over the alphabet.
//! 40 queries + 4× blowup → ~80 bits of combined soundness.

use serde::{Serialize, Deserialize};
use super::field::{Fp, fp_to_u32x8, fp_from_u32x8, ntt_root_of_unity, batch_inverse, Channel252};
use super::merkle::{MerkleTree252, Digest, verify_auth_path};

/// Stop folding when layer size reaches 2^LOG_LAST_LAYER.
pub const LOG_LAST_LAYER: u32 = 4; // last layer size = 16

// ─────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────

/// Decommitment for one query at one fold layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriLayerDecommit {
    /// f_l[low] where low = query_idx mod (M_l / 2).
    pub f_lo: [u32; 8],
    /// f_l[low + M_l/2].
    pub f_hi: [u32; 8],
    /// Auth path for f_lo into the layer's Merkle tree.
    pub path_lo: Vec<[u32; 8]>,
    /// Auth path for f_hi into the layer's Merkle tree.
    pub path_hi: Vec<[u32; 8]>,
}

/// All layer decommitments for one query index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriQueryDecommit {
    /// layers[0] is the decommitment into the caller's f0 tree.
    /// layers[1..] are decommitments into inner FRI trees.
    pub layers: Vec<FriLayerDecommit>,
}

/// The serializable FRI proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriProof {
    /// Merkle roots for FRI layers 1..n_folds.
    /// Layer 0 root is the caller's commitment (not repeated here).
    pub inner_roots: Vec<Digest>,
    /// Final layer evaluations (size 2^LOG_LAST_LAYER).
    pub last_layer_evals: Vec<[u32; 8]>,
    /// Per-query decommitments.
    pub query_decommits: Vec<FriQueryDecommit>,
}

/// Intermediate FRI commit data (not serialized; used to build the proof).
pub struct FriWitness {
    pub f0_evals: Vec<Fp>,
    /// inner_evals[l] = evaluations at FRI layer l+1 (l=0 means first fold).
    pub inner_evals: Vec<Vec<Fp>>,
    pub last_layer_evals: Vec<Fp>,
    pub inner_trees: Vec<MerkleTree252>,
    /// fold_alphas[l] = α_l used to fold layer l → l+1.
    pub fold_alphas: Vec<Fp>,
    pub log_m0: u32,
}

// ─────────────────────────────────────────────
// Fold one layer
// ─────────────────────────────────────────────

/// Fold evaluations of size M → M/2.
///
/// g[i] = (f[i] + f[i+M/2]) / 2 + alpha * (f[i] - f[i+M/2]) / (2 * ω_M^i)
/// for i = 0..M/2 − 1.
pub fn fold_layer(f: &[Fp], log_m: u32, alpha: Fp) -> Vec<Fp> {
    let m = f.len();
    debug_assert_eq!(m, 1usize << log_m);
    let half = m / 2;

    let omega    = ntt_root_of_unity(log_m);
    let two      = Fp::from_u64(2);
    let two_inv  = two.inverse();

    // Precompute 2·ω_M^i for batch inversion
    let mut xi = Fp::ONE;
    let two_xi: Vec<Fp> = (0..half).map(|_| {
        let val = two.mul(xi);
        xi = xi.mul(omega);
        val
    }).collect();
    let inv_two_xi = batch_inverse(&two_xi);

    (0..half).map(|i| {
        let f_lo = f[i];
        let f_hi = f[i + half];
        let sum  = f_lo.add(f_hi);
        let diff = f_lo.sub(f_hi);
        // (f_lo + f_hi) / 2  +  alpha * (f_lo - f_hi) / (2 * ω^i)
        sum.mul(two_inv).add(alpha.mul(diff.mul(inv_two_xi[i])))
    }).collect()
}

// ─────────────────────────────────────────────
// Commit phase
// ─────────────────────────────────────────────

/// Run the FRI commit phase.
///
/// `f0_evals` — evaluations of the polynomial to test (layer 0).
/// `f0_root`  — the already-committed root for f0 (mixed into channel before calling).
/// `log_m0`   — log2 of the domain size (= log_n + log_blowup).
///
/// Returns a `FriWitness` containing all layer data needed to build query proofs.
pub fn fri_commit(
    f0_evals: Vec<Fp>,
    f0_root: &Digest,
    log_m0: u32,
    channel: &mut Channel252,
) -> FriWitness {
    assert!(log_m0 > LOG_LAST_LAYER, "domain must be larger than last layer");

    // f0 root already mixed into channel by caller; draw alpha_0
    let n_folds = (log_m0 - LOG_LAST_LAYER) as usize;

    let mut inner_evals: Vec<Vec<Fp>> = Vec::with_capacity(n_folds);
    let mut inner_trees: Vec<MerkleTree252> = Vec::with_capacity(n_folds);
    let mut fold_alphas: Vec<Fp> = Vec::with_capacity(n_folds);

    // α_0 is drawn after f0_root is mixed — caller must have already done that
    let alpha0 = channel.draw_fp();
    fold_alphas.push(alpha0);

    let mut current = fold_layer(&f0_evals, log_m0, alpha0);
    let mut log_m = log_m0 - 1;

    for fold_idx in 1..n_folds {
        let tree = MerkleTree252::commit(&current);
        let root = tree.root();
        channel.mix_digest(&root);
        let alpha = channel.draw_fp();

        inner_evals.push(current.clone());
        inner_trees.push(tree);
        fold_alphas.push(alpha);

        current = fold_layer(&current, log_m, alpha);
        log_m -= 1;
    }

    // Last layer (size 2^LOG_LAST_LAYER): commit as raw evals
    let last_layer_evals = current;

    // Mix last layer into channel
    for v in &last_layer_evals {
        channel.mix_fp(v);
    }

    FriWitness {
        f0_evals,
        inner_evals,
        last_layer_evals,
        inner_trees,
        fold_alphas,
        log_m0,
    }
}

// ─────────────────────────────────────────────
// Query / decommit phase
// ─────────────────────────────────────────────

/// Build query decommitments for a set of query indices.
///
/// `f0_tree` — the caller's Merkle tree for layer 0 evaluations.
pub fn fri_build_proof(
    witness: &FriWitness,
    f0_tree: &MerkleTree252,
    query_indices: &[usize],
) -> FriProof {
    let log_m0 = witness.log_m0;
    let n_folds = witness.fold_alphas.len();

    let inner_roots: Vec<Digest> = witness.inner_trees.iter().map(|t| t.root()).collect();

    let query_decommits: Vec<FriQueryDecommit> = query_indices.iter().map(|&q0| {
        let mut layers = Vec::with_capacity(n_folds);
        let mut m_l = 1usize << log_m0;
        let mut q_l = q0;

        for fold_idx in 0..n_folds {
            let half = m_l / 2;
            let low  = q_l % half;
            let high = low + half;

            let (f_lo_val, f_hi_val, path_lo, path_hi) = if fold_idx == 0 {
                let lo = fp_to_u32x8(&witness.f0_evals[low]);
                let hi = fp_to_u32x8(&witness.f0_evals[high]);
                let plo = f0_tree.auth_path(low);
                let phi = f0_tree.auth_path(high);
                (lo, hi, plo, phi)
            } else {
                let layer_evals = &witness.inner_evals[fold_idx - 1];
                let tree = &witness.inner_trees[fold_idx - 1];
                let lo = fp_to_u32x8(&layer_evals[low]);
                let hi = fp_to_u32x8(&layer_evals[high]);
                let plo = tree.auth_path(low);
                let phi = tree.auth_path(high);
                (lo, hi, plo, phi)
            };

            layers.push(FriLayerDecommit {
                f_lo: f_lo_val,
                f_hi: f_hi_val,
                path_lo,
                path_hi,
            });

            // Advance to next layer
            q_l = low;
            m_l = half;
        }

        FriQueryDecommit { layers }
    }).collect();

    FriProof {
        inner_roots,
        last_layer_evals: witness.last_layer_evals.iter().map(fp_to_u32x8).collect(),
        query_decommits,
    }
}

// ─────────────────────────────────────────────
// Verify
// ─────────────────────────────────────────────

/// Verify a FRI proof.
///
/// `f0_root`       — Merkle root of layer 0 (from the caller's commitment).
/// `query_indices` — query indices (must match what was used to build the proof).
/// `log_m0`        — log2 of the initial domain size.
/// `channel`       — Fiat-Shamir channel; must be in the same state as when
///                   fri_commit drew alphas (i.e., f0_root already mixed in).
pub fn fri_verify(
    proof: &FriProof,
    f0_root: &Digest,
    query_indices: &[usize],
    log_m0: u32,
    channel: &mut Channel252,
) -> Result<(), String> {
    let n_folds = log_m0 - LOG_LAST_LAYER;
    if proof.inner_roots.len() != (n_folds as usize) - 1 {
        return Err(format!(
            "Expected {} inner roots, got {}",
            n_folds - 1,
            proof.inner_roots.len()
        ));
    }
    if proof.last_layer_evals.len() != (1 << LOG_LAST_LAYER) {
        return Err(format!(
            "Expected {} last-layer evals, got {}",
            1 << LOG_LAST_LAYER,
            proof.last_layer_evals.len()
        ));
    }

    // ── Replay commit phase to recover alphas ────────────────
    // f0_root already mixed by caller before this function is called.
    let alpha0 = channel.draw_fp();
    let mut alphas = vec![alpha0];

    for root in &proof.inner_roots {
        channel.mix_digest(root);
        alphas.push(channel.draw_fp());
    }
    // Mix last layer into channel (same as prover)
    let last_layer: Vec<Fp> = proof.last_layer_evals.iter().map(fp_from_u32x8).collect();
    for v in &last_layer {
        channel.mix_fp(v);
    }

    // ── Per-query checks ─────────────────────────────────────
    let two = Fp::from_u64(2);
    let two_inv = two.inverse();

    // Build layer roots array (layer 0 = f0_root, layers 1..n_folds-1 = inner_roots)
    let mut layer_roots: Vec<&Digest> = Vec::with_capacity(n_folds as usize);
    layer_roots.push(f0_root);
    for r in &proof.inner_roots {
        layer_roots.push(r);
    }

    for (qi, (&q0, decommit)) in query_indices.iter()
        .zip(proof.query_decommits.iter())
        .enumerate()
    {
        if decommit.layers.len() != n_folds as usize {
            return Err(format!("Query {qi}: expected {} layers, got {}", n_folds, decommit.layers.len()));
        }

        let mut m_l = 1usize << log_m0;
        let mut q_l = q0;
        let mut log_m_l = log_m0;

        for fold_idx in 0..n_folds as usize {
            let half = m_l / 2;
            let low  = q_l % half;
            let high = low + half;

            let ld = &decommit.layers[fold_idx];
            let f_lo  = fp_from_u32x8(&ld.f_lo);
            let f_hi  = fp_from_u32x8(&ld.f_hi);
            let root  = layer_roots[fold_idx];
            let log_layer = log_m_l;

            // 1. Verify Merkle auth paths
            if !verify_auth_path(&f_lo, &ld.path_lo, root, low,  log_layer) {
                return Err(format!("Query {qi} layer {fold_idx}: f_lo auth path failed (low={low})"));
            }
            if !verify_auth_path(&f_hi, &ld.path_hi, root, high, log_layer) {
                return Err(format!("Query {qi} layer {fold_idx}: f_hi auth path failed (high={high})"));
            }

            // 2. Compute expected fold value
            let omega    = ntt_root_of_unity(log_m_l);
            let x_low    = fp_pow_u64(omega, low as u64);
            let two_x    = two.mul(x_low);
            let alpha    = alphas[fold_idx];
            let expected = f_lo.add(f_hi).mul(two_inv)
                          .add(alpha.mul(f_lo.sub(f_hi).mul(two_x.inverse())));

            // 3. Verify fold == the value at position `low` in the next layer.
            //
            // The next layer's decommit provides f_lo at index (low % next_half)
            // and f_hi at index (low % next_half + next_half).
            //  - If low < next_half:  f_{l+1}[low] == f_lo_next
            //  - If low >= next_half: f_{l+1}[low] == f_hi_next
            if fold_idx + 1 < n_folds as usize {
                let next_ld   = &decommit.layers[fold_idx + 1];
                let next_half = half / 2; // = size of next layer / 2
                let next_val  = if low < next_half {
                    fp_from_u32x8(&next_ld.f_lo)
                } else {
                    fp_from_u32x8(&next_ld.f_hi)
                };
                if expected != next_val {
                    return Err(format!(
                        "Query {qi} layer {fold_idx}: fold mismatch at low={low}: \
                         expected {:?}, got {:?}",
                        fp_to_u32x8(&expected), fp_to_u32x8(&next_val)
                    ));
                }
            } else {
                // Last fold: output has size `half`; check against last_layer_evals[low].
                if expected != last_layer[low] {
                    return Err(format!(
                        "Query {qi}: final fold mismatch at last_layer[{low}]: \
                         expected {:?}, got {:?}",
                        fp_to_u32x8(&expected), fp_to_u32x8(&last_layer[low])
                    ));
                }
            }

            q_l = low;
            m_l = half;
            log_m_l -= 1;
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────

fn fp_pow_u64(base: Fp, exp: u64) -> Fp {
    base.pow_fp(crate::cairo_air::stark252_field::Fp { v: [exp, 0, 0, 0] })
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ntt::lde_cpu;
    use super::super::merkle::MerkleTree252;

    /// Build a random-ish polynomial of degree < N, extend to 4N, commit + prove + verify FRI.
    #[test]
    fn test_fri_roundtrip() {
        let log_n = 6u32; // N = 64, domain = 256
        let log_blowup = 2u32;
        let log_m0 = log_n + log_blowup; // 8

        // Build a polynomial of degree N-1: coefficients c[0..N]
        let n = 1usize << log_n;
        let coeffs: Vec<Fp> = (0..n as u64).map(|i| Fp::from_u64(i * 7 + 3)).collect();
        let f0_evals = lde_cpu(&coeffs, log_n, log_blowup);
        assert_eq!(f0_evals.len(), 1 << log_m0);

        // Commit layer 0
        let f0_tree = MerkleTree252::commit(&f0_evals);
        let f0_root = f0_tree.root();

        // FRI commit phase
        let mut channel = Channel252::new();
        channel.mix_digest(&f0_root);
        let witness = fri_commit(f0_evals.clone(), &f0_root, log_m0, &mut channel);

        // Draw query indices
        let m0 = 1usize << log_m0;
        let query_indices: Vec<usize> = (0..40).map(|_| channel.draw_number(m0)).collect();

        // Build proof
        let proof = fri_build_proof(&witness, &f0_tree, &query_indices);

        // Verify
        let mut vch = Channel252::new();
        vch.mix_digest(&f0_root);
        fri_verify(&proof, &f0_root, &query_indices, log_m0, &mut vch)
            .expect("FRI proof should verify");
    }

    /// A tampered last layer should be rejected.
    #[test]
    fn test_fri_tamper_last_layer() {
        let log_n = 5u32;
        let log_blowup = 2u32;
        let log_m0 = log_n + log_blowup;
        let n = 1usize << log_n;

        let coeffs: Vec<Fp> = (0..n as u64).map(|i| Fp::from_u64(i + 1)).collect();
        let f0_evals = lde_cpu(&coeffs, log_n, log_blowup);
        let f0_tree = MerkleTree252::commit(&f0_evals);
        let f0_root = f0_tree.root();

        let mut channel = Channel252::new();
        channel.mix_digest(&f0_root);
        let witness = fri_commit(f0_evals.clone(), &f0_root, log_m0, &mut channel);

        let m0 = 1usize << log_m0;
        let query_indices: Vec<usize> = (0..40).map(|_| channel.draw_number(m0)).collect();

        let mut proof = fri_build_proof(&witness, &f0_tree, &query_indices);

        // Tamper: corrupt all last-layer evals so at least one queried position is affected.
        for e in &mut proof.last_layer_evals {
            e[0] ^= 1;
        }

        let mut vch = Channel252::new();
        vch.mix_digest(&f0_root);
        assert!(
            fri_verify(&proof, &f0_root, &query_indices, log_m0, &mut vch).is_err(),
            "Tampered last layer should be rejected"
        );
    }

    /// Tamper a decommitted value — auth path check should fail.
    #[test]
    fn test_fri_tamper_decommit() {
        let log_n = 5u32;
        let log_blowup = 2u32;
        let log_m0 = log_n + log_blowup;
        let n = 1usize << log_n;

        let coeffs: Vec<Fp> = (0..n as u64).map(|i| Fp::from_u64(i * 3 + 2)).collect();
        let f0_evals = lde_cpu(&coeffs, log_n, log_blowup);
        let f0_tree = MerkleTree252::commit(&f0_evals);
        let f0_root = f0_tree.root();

        let mut channel = Channel252::new();
        channel.mix_digest(&f0_root);
        let witness = fri_commit(f0_evals.clone(), &f0_root, log_m0, &mut channel);

        let m0 = 1usize << log_m0;
        let query_indices: Vec<usize> = (0..40).map(|_| channel.draw_number(m0)).collect();

        let mut proof = fri_build_proof(&witness, &f0_tree, &query_indices);

        // Tamper: flip a bit in the first query's first layer f_lo
        proof.query_decommits[0].layers[0].f_lo[0] ^= 1;

        let mut vch = Channel252::new();
        vch.mix_digest(&f0_root);
        assert!(
            fri_verify(&proof, &f0_root, &query_indices, log_m0, &mut vch).is_err(),
            "Tampered decommit should be rejected"
        );
    }
}
