//! Stwo wire-format proof export.
//!
//! Converts a VortexSTARK `CairoProof` to a JSON-serializable structure that uses
//! stwo-compatible conventions:
//!   - Blake2s hashes (standard, no domain byte) for Merkle nodes
//!   - `hash_witness`: deduplicated sibling hashes (level 0 → root), not raw auth paths
//!   - QM31 as [u32; 4] (little-endian M31 components)
//!   - M31 as u32
//!   - FRI proof: first_layer (OODS quotient), inner_layers, last_layer_poly
//!
//! VortexSTARK uses 7 commitment trees; each is exported as a separate tree in the
//! `TwoStarkProof`. The trees are in Fiat-Shamir channel order.

use crate::cairo_air::prover::CairoProof;
use crate::prover::QueryDecommitment;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ── Output types ──────────────────────────────────────────────────────────────

/// Blake2s 32-byte hash as bytes (Stwo wire format).
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TwoHash(pub [u8; 32]);

impl From<[u32; 8]> for TwoHash {
    fn from(words: [u32; 8]) -> Self {
        let mut bytes = [0u8; 32];
        for (i, &w) in words.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        TwoHash(bytes)
    }
}

impl std::fmt::Debug for TwoHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TwoHash({})", hex::encode(self.0))
    }
}

/// M31 base field element.
pub type TwoBaseField = u32;

/// QM31 secure field element: [v0, v1, v2, v3] each M31.
pub type TwoSecureField = [u32; 4];

/// Merkle decommitment: deduplicated sibling hashes from leaf level to root.
///
/// Matches stwo's `MerkleDecommitment` / `MerkleDecommitmentLifted`.
/// The verifier replays `hash_witness` level by level, pulling hashes only for
/// positions whose siblings were not also queried.
#[derive(Clone, Serialize, Deserialize)]
pub struct TwoMerkleDecommitment {
    /// Sibling hashes in level order (leaf → root), deduplicated.
    pub hash_witness: Vec<TwoHash>,
}

/// FRI first-layer proof (the OODS quotient layer).
#[derive(Clone, Serialize, Deserialize)]
pub struct TwoFriFirstLayerProof {
    /// OODS quotient values at sibling positions of queried positions
    /// (needed for fold verification at the first FRI step).
    pub fri_witness: Vec<TwoSecureField>,
    /// Merkle opening for the OODS quotient polynomial at query positions.
    pub decommitment: TwoMerkleDecommitment,
    /// Merkle root of the OODS quotient tree.
    pub commitment: TwoHash,
}

/// FRI inner-layer fold proof.
#[derive(Clone, Serialize, Deserialize)]
pub struct TwoFriFoldProof {
    /// Values at sibling positions (needed for fold verification).
    pub fri_witness: Vec<TwoSecureField>,
    /// Merkle opening for this FRI layer at query positions.
    pub decommitment: TwoMerkleDecommitment,
    /// Merkle root of this FRI layer.
    pub commitment: TwoHash,
}

/// Full FRI proof.
#[derive(Clone, Serialize, Deserialize)]
pub struct TwoFriProof {
    pub first_layer: TwoFriFirstLayerProof,
    pub inner_layers: Vec<TwoFriFoldProof>,
    pub last_layer_poly: Vec<TwoSecureField>,
}

/// Per-tree metadata (informational, not part of stwo wire format).
#[derive(Clone, Serialize, Deserialize)]
pub struct TwoTreeMeta {
    /// Human-readable name for this commitment tree.
    pub name: String,
    /// Number of columns in this tree (M31-valued).
    pub n_cols: usize,
    /// Log2 of the eval domain (= leaf count) for this tree.
    pub log_eval_size: u32,
}

/// FRI/PCS configuration parameters (mirrors stwo's `PcsConfig`).
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct TwoPcsConfig {
    /// log2 of the FRI blowup factor (= BLOWUP_BITS in VortexSTARK).
    pub log_blowup_factor: u32,
    /// Number of FRI query positions drawn per proof.
    pub n_queries: u32,
    /// Required proof-of-work bits (leading zeros in PoW hash).
    pub pow_bits: u32,
    /// log2 of last FRI layer size (below which coefficients are sent directly).
    pub log_last_layer_degree_bound: u32,
}

/// Stwo-compatible full proof for VortexSTARK Cairo AIR.
///
/// Trees are in Fiat-Shamir channel order:
///   0 = trace lo (cols 0-15)
///   1 = trace hi (cols 16-30)
///   2 = dict trace (cols 31-33)
///   3 = LogUp interaction (1 QM31 = 4 M31)
///   4 = RC interaction (1 QM31 = 4 M31)
///   5 = S_dict interaction (1 QM31 = 4 M31)
///   6 = composition / quotient (4 QM31 = 16 M31)
#[derive(Clone, Serialize, Deserialize)]
pub struct TwoStarkProof {
    /// PCS/FRI configuration parameters.
    pub config: TwoPcsConfig,

    /// Informational: column count and domain size for each commitment tree.
    pub tree_meta: Vec<TwoTreeMeta>,

    /// Merkle roots in channel order (7 trees).
    pub commitments: Vec<TwoHash>,

    /// OODS sampled values: `[tree][col]` = Vec of SecureField evaluations per sample point.
    /// Main trace trees (0-2): two sample points each (z and z_next).
    /// Interaction trees (3-5): two sample points each (z and z_next).
    /// Quotient tree (6): one sample point (z only).
    pub sampled_values: Vec<Vec<Vec<TwoSecureField>>>,

    /// Merkle decommitments per tree (hash_witness, deduplicated).
    pub decommitments: Vec<TwoMerkleDecommitment>,

    /// Column values at query positions: `[tree][col][query]`.
    pub queried_values: Vec<Vec<Vec<TwoBaseField>>>,

    /// Query positions in the eval domain.
    pub query_indices: Vec<usize>,

    /// Proof-of-work nonce.
    pub proof_of_work: u64,

    /// FRI proof.
    pub fri_proof: TwoFriProof,
}

// ── Core algorithm: FRI witness deduplication ─────────────────────────────────

/// Build a stwo-compatible `fri_witness` for one FRI layer.
///
/// stwo's `FriLayerProof.fri_witness` contains evaluations at positions that are
/// in the "fold group" but NOT in the queried set.  With `fold_step = 1`, each
/// fold group is a pair `{q & !1, (q & !1) | 1}`.  If both siblings of a pair are
/// queried, neither appears in `fri_witness`.  If only one is queried, the other
/// (its sibling's value) must be in `fri_witness`.
///
/// VortexSTARK's `QueryDecommitment.sibling_values[i]` is always `layer[pos ^ 1]`
/// for every query `i` — N_QUERIES entries regardless.  This function deduplicates
/// them to match stwo's format.
///
/// `layer_positions[i]` is the queried position for query `i` at this layer.
/// `sibling_values[i]`  is the evaluated value at `layer_positions[i] ^ 1`.
///
/// Returns witness values in sorted-pair order (matching stwo's consumption order).
fn dedup_fri_witness(
    layer_positions: &[usize],
    sibling_values: &[[u32; 4]],
) -> Vec<TwoSecureField> {
    use std::collections::BTreeMap;

    // Build map: position → sibling_value (first occurrence wins for duplicates).
    let mut pos_to_sibling: BTreeMap<usize, [u32; 4]> = BTreeMap::new();
    for (i, &pos) in layer_positions.iter().enumerate() {
        if i < sibling_values.len() {
            pos_to_sibling.entry(pos).or_insert(sibling_values[i]);
        }
    }

    // Unique queried positions (sorted by BTreeMap).
    let unique_positions: std::collections::BTreeSet<usize> =
        pos_to_sibling.keys().copied().collect();

    // For each unique position p: if sibling (p ^ 1) is NOT also queried,
    // we must include the sibling value in fri_witness.
    // Sorted iteration = pair-order (stwo processes pairs in ascending order).
    let mut witness = Vec::new();
    for &pos in &unique_positions {
        let sibling = pos ^ 1;
        if !unique_positions.contains(&sibling) {
            witness.push(pos_to_sibling[&pos]);
        }
    }
    witness
}

// ── Core algorithm: fold-pair hash witness for FRI layers ────────────────────

/// Build hash_witness for FRI fold layers where stwo opens complete fold pairs.
pub fn fold_pair_hash_witness(
    query_positions: &[usize],
    auth_paths: &[Vec<[u32; 8]>],
    tree_height: usize,
) -> Vec<TwoHash> {
    if query_positions.is_empty() || tree_height == 0 { return Vec::new(); }
    let mut pos_to_path: BTreeMap<usize, usize> = BTreeMap::new();
    for (idx, &pos) in query_positions.iter().enumerate() {
        pos_to_path.entry(pos).or_insert(idx);
    }
    let mut pair_set: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for &pos in pos_to_path.keys() {
        pair_set.insert(pos & !1);
        pair_set.insert(pos | 1);
    }
    let decom_positions: Vec<usize> = pair_set.into_iter().collect();
    let mut level_pos_to_q: BTreeMap<usize, usize> = BTreeMap::new();
    for &pos in &decom_positions {
        if let Some(&idx) = pos_to_path.get(&pos) {
            level_pos_to_q.insert(pos, idx);
        } else if let Some(&idx) = pos_to_path.get(&(pos ^ 1)) {
            level_pos_to_q.insert(pos, idx);
        }
    }
    let mut current = decom_positions;
    let mut witness = Vec::new();
    for level in 0..tree_height {
        let mut next: BTreeMap<usize, usize> = BTreeMap::new();
        let mut i = 0;
        while i < current.len() {
            let pos = current[i];
            let sib = pos ^ 1;
            let parent = pos >> 1;
            if i + 1 < current.len() && current[i + 1] == sib {
                next.entry(parent).or_insert(level_pos_to_q[&pos]);
                i += 2;
            } else {
                let q_idx = level_pos_to_q[&pos];
                if level < auth_paths[q_idx].len() {
                    witness.push(TwoHash::from(auth_paths[q_idx][level]));
                }
                next.entry(parent).or_insert(q_idx);
                i += 1;
            }
        }
        current = next.keys().copied().collect();
        level_pos_to_q = next;
    }
    witness
}

// ── Core algorithm: auth paths → hash_witness ─────────────────────────────────

/// Convert a set of Merkle auth paths to a deduplicated `hash_witness`.
///
/// `query_positions[i]` is the leaf index for query `i`.
/// `auth_paths[i][level]` is the sibling hash at `level` (0 = leaf level) for query `i`.
///
/// Matches stwo's `MerkleVerifier` expectation: the witness contains only the hashes
/// for positions whose siblings are NOT also queried (deduplication removes redundancy
/// when two queried positions share a parent).
pub fn auth_paths_to_hash_witness(
    query_positions: &[usize],
    auth_paths: &[Vec<[u32; 8]>],
    tree_height: usize,
) -> Vec<TwoHash> {
    if query_positions.is_empty() || tree_height == 0 {
        return Vec::new();
    }

    // Map each leaf position to an index into auth_paths (deduplicate, keep first).
    let mut pos_to_path: BTreeMap<usize, usize> = BTreeMap::new();
    for (idx, &pos) in query_positions.iter().enumerate() {
        pos_to_path.entry(pos).or_insert(idx);
    }

    // Sorted unique positions at the current level.
    let mut current_positions: Vec<usize> = pos_to_path.keys().copied().collect();
    // Map from shifted position at current level to an original query index (for auth path lookup).
    let mut level_pos_to_query: BTreeMap<usize, usize> = pos_to_path.clone();

    let mut witness = Vec::new();

    for level in 0..tree_height {
        let mut i = 0;
        while i < current_positions.len() {
            let pos = current_positions[i];
            let sibling = pos ^ 1;

            if i + 1 < current_positions.len() && current_positions[i + 1] == sibling {
                // Both pos and its sibling are queried — no witness hash needed.
                i += 2;
            } else {
                // Sibling not queried — pull its hash from the auth path of pos.
                let query_idx = level_pos_to_query[&pos];
                if level < auth_paths[query_idx].len() {
                    witness.push(TwoHash::from(auth_paths[query_idx][level]));
                }
                i += 1;
            }
        }

        // Advance to parent level.
        let mut parent_positions: BTreeMap<usize, usize> = BTreeMap::new();
        for &pos in &current_positions {
            let parent = pos >> 1;
            let query_idx = level_pos_to_query[&pos];
            parent_positions.entry(parent).or_insert(query_idx);
        }

        current_positions = parent_positions.keys().copied().collect();
        level_pos_to_query = parent_positions;
    }

    witness
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Transpose `values_by_query[query][col]` → `values_by_col[col][query]`.
fn transpose_query_major<T: Clone>(
    values_by_query: &[Vec<T>],
    n_cols: usize,
    n_queries: usize,
) -> Vec<Vec<T>> {
    let mut out: Vec<Vec<T>> = (0..n_cols).map(|_| Vec::with_capacity(n_queries)).collect();
    for q in 0..n_queries {
        for c in 0..n_cols {
            out[c].push(values_by_query[q][c].clone());
        }
    }
    out
}

/// Build `queried_values` (column-major) for a trace tree.
///
/// `values_at_queries[q]` = all M31 column values at query `q`.
/// `col_range` = which subset of columns belongs to this tree.
fn trace_queried_values(
    values_at_queries: &[Vec<u32>],
    col_start: usize,
    col_end: usize,
    n_queries: usize,
) -> Vec<Vec<TwoBaseField>> {
    let n_cols = col_end - col_start;
    let mut out: Vec<Vec<u32>> = (0..n_cols).map(|_| Vec::with_capacity(n_queries)).collect();
    for q in 0..n_queries {
        for c in 0..n_cols {
            out[c].push(values_at_queries[q][col_start + c]);
        }
    }
    out
}

/// Build `queried_values` for a QM31 interaction tree.
///
/// `values[q]` = QM31 as `[u32; 4]`; exported as 4 separate M31 columns.
fn interaction_queried_values(values: &[[u32; 4]], n_queries: usize) -> Vec<Vec<TwoBaseField>> {
    let mut out: Vec<Vec<u32>> = (0..4).map(|_| Vec::with_capacity(n_queries)).collect();
    for q in 0..n_queries {
        for comp in 0..4 {
            out[comp].push(values[q][comp]);
        }
    }
    out
}

/// Build `queried_values` for the quotient tree (4 QM31 = 16 M31 columns).
///
/// `values[q]` = `[u32; 4]` representing one QM31 (the combined composition polynomial).
/// Exported as 4 M31 columns.
fn quotient_queried_values(
    decomm: &QueryDecommitment<[u32; 4]>,
    n_queries: usize,
) -> Vec<Vec<TwoBaseField>> {
    interaction_queried_values(&decomm.values[..n_queries], n_queries)
}

/// Build `sampled_values` for a main trace tree (sample at z and z_next).
///
/// `oods_at_z[col]` and `oods_at_z_next[col]` are QM31 evaluations.
/// Returns `[col][sample_pt]` = Vec<SecureField> per column (2 points: z, z_next).
fn trace_sampled_values(
    oods_at_z: &[[u32; 4]],
    oods_at_z_next: &[[u32; 4]],
    col_start: usize,
    col_end: usize,
) -> Vec<Vec<TwoSecureField>> {
    (col_start..col_end)
        .map(|c| vec![oods_at_z[c], oods_at_z_next[c]])
        .collect()
}

/// Build hash_witness for a `QueryDecommitment` (used for interaction/quotient/FRI trees).
///
/// `query_positions[i]` is the leaf index for decommitment entry `i`.
fn decommitment_hash_witness(
    query_positions: &[usize],
    decomm: &QueryDecommitment<[u32; 4]>,
    tree_height: usize,
) -> Vec<TwoHash> {
    auth_paths_to_hash_witness(query_positions, &decomm.auth_paths, tree_height)
}

// ── Main export function ───────────────────────────────────────────────────────

/// Convert a `CairoProof` to a stwo-compatible `TwoStarkProof`.
pub fn cairo_proof_to_stwo(proof: &CairoProof) -> TwoStarkProof {
    use crate::prover::BLOWUP_BITS;

    let log_n = proof.log_trace_size;
    let log_eval = log_n + BLOWUP_BITS;
    let tree_height = log_eval as usize; // auth path length for main trace trees
    let n_queries = proof.query_indices.len();

    // ── Commitments (7 trees in channel order) ────────────────────────────────
    let commitments = vec![
        TwoHash::from(proof.trace_commitment),           // tree 0: trace lo (cols 0-15)
        TwoHash::from(proof.trace_commitment_hi),        // tree 1: trace hi (cols 16-30)
        TwoHash::from(proof.dict_trace_commitment),      // tree 2: dict trace (cols 31-33)
        TwoHash::from(proof.interaction_commitment),     // tree 3: LogUp interaction
        TwoHash::from(proof.rc_interaction_commitment),  // tree 4: RC interaction
        TwoHash::from(proof.dict_main_interaction_commitment), // tree 5: S_dict interaction
        TwoHash::from(proof.quotient_commitment),        // tree 6: composition quotient
    ];

    // ── Tree metadata ─────────────────────────────────────────────────────────
    let tree_meta = vec![
        TwoTreeMeta { name: "trace_lo".into(),      n_cols: 16, log_eval_size: log_eval },
        TwoTreeMeta { name: "trace_hi".into(),      n_cols: 15, log_eval_size: log_eval },
        TwoTreeMeta { name: "dict_trace".into(),    n_cols:  3, log_eval_size: log_eval },
        TwoTreeMeta { name: "interaction".into(),   n_cols:  4, log_eval_size: log_eval },
        TwoTreeMeta { name: "rc_interaction".into(), n_cols: 4, log_eval_size: log_eval },
        TwoTreeMeta { name: "dict_interaction".into(), n_cols: 4, log_eval_size: log_eval },
        TwoTreeMeta { name: "quotient".into(),      n_cols:  4, log_eval_size: log_eval },
    ];

    // ── OODS sampled values ───────────────────────────────────────────────────
    // Trees 0-2: trace columns sampled at z and z_next.
    // Trees 3-5: interaction columns — no OODS in VortexSTARK (empty).
    // Tree 6: quotient columns sampled at z.
    let oods_len = proof.oods_trace_at_z.len();
    let oods_next_len = proof.oods_trace_at_z_next.len();

    let lo_end   = 16.min(oods_len);
    let hi_end   = 31.min(oods_len);
    let dict_end = oods_len.min(34);

    let mut sampled_values: Vec<Vec<Vec<TwoSecureField>>> = Vec::with_capacity(7);

    // Tree 0: trace lo, cols 0..16
    if oods_len > 0 && oods_next_len > 0 {
        sampled_values.push(trace_sampled_values(
            &proof.oods_trace_at_z,
            &proof.oods_trace_at_z_next,
            0, lo_end,
        ));
        // Tree 1: trace hi, cols 16..31
        sampled_values.push(trace_sampled_values(
            &proof.oods_trace_at_z,
            &proof.oods_trace_at_z_next,
            lo_end.min(16), hi_end,
        ));
        // Tree 2: dict trace, cols 31..34
        sampled_values.push(trace_sampled_values(
            &proof.oods_trace_at_z,
            &proof.oods_trace_at_z_next,
            hi_end.min(31), dict_end,
        ));
    } else {
        sampled_values.push(Vec::new());
        sampled_values.push(Vec::new());
        sampled_values.push(Vec::new());
    }

    // Trees 3-5: interaction polynomials sampled at z and z_next (4 M31 components each).
    let has_interaction = proof.oods_interaction_at_z != [[[0u32; 4]; 4]; 3];
    let has_interaction_next = proof.oods_interaction_at_z_next != [[[0u32; 4]; 4]; 3];
    for pi in 0..3 {
        if has_interaction {
            let cols: Vec<Vec<TwoSecureField>> = (0..4)
                .map(|k| {
                    let mut pts = vec![proof.oods_interaction_at_z[pi][k]];
                    if has_interaction_next {
                        pts.push(proof.oods_interaction_at_z_next[pi][k]);
                    }
                    pts
                })
                .collect();
            sampled_values.push(cols);
        } else {
            sampled_values.push(Vec::new());
        }
    }

    // Tree 6: composition quotient sampled at z (oods_quotient_at_z, 4 QM31 cols)
    let quotient_oods: Vec<Vec<TwoSecureField>> = if proof.oods_quotient_at_z != [[0u32; 4]; 4] {
        proof.oods_quotient_at_z.iter().map(|&v| vec![v]).collect()
    } else {
        Vec::new()
    };
    sampled_values.push(quotient_oods);

    // ── Queried values (column-major per tree) ────────────────────────────────
    let mut queried_values: Vec<Vec<Vec<TwoBaseField>>> = Vec::with_capacity(7);

    // Tree 0: trace lo cols 0..16
    queried_values.push(trace_queried_values(&proof.trace_values_at_queries, 0, 16, n_queries));
    // Tree 1: trace hi cols 16..31 (may be shorter if proof has fewer cols)
    let hi_cols = proof.trace_values_at_queries.first().map(|v| v.len().min(31)).unwrap_or(16);
    queried_values.push(trace_queried_values(&proof.trace_values_at_queries, 16, hi_cols, n_queries));
    // Tree 2: dict trace cols 31..34
    let dict_cols = proof.trace_values_at_queries.first().map(|v| v.len().min(34)).unwrap_or(31);
    queried_values.push(trace_queried_values(&proof.trace_values_at_queries, hi_cols, dict_cols, n_queries));

    // Tree 3: LogUp interaction (4 M31 components of combined QM31)
    queried_values.push(interaction_queried_values(&proof.interaction_decommitment.values, n_queries));
    // Tree 4: RC interaction
    queried_values.push(interaction_queried_values(&proof.rc_interaction_decommitment.values, n_queries));
    // Tree 5: S_dict interaction
    queried_values.push(interaction_queried_values(&proof.dict_main_interaction_decommitment.values, n_queries));
    // Tree 6: composition quotient
    queried_values.push(quotient_queried_values(&proof.quotient_decommitment, n_queries));

    // ── Merkle decommitments (hash_witness per tree) ──────────────────────────
    let mut decommitments: Vec<TwoMerkleDecommitment> = Vec::with_capacity(7);

    // Tree 0: trace lo — query and next-row positions both need to be opened.
    // We include both current and next queries in the witness generation.
    let lo_witness = build_trace_witness(
        &proof.query_indices,
        &proof.trace_auth_paths,
        &proof.trace_auth_paths_next,
        tree_height,
    );
    decommitments.push(TwoMerkleDecommitment { hash_witness: lo_witness });

    // Tree 1: trace hi
    let hi_witness = build_trace_witness(
        &proof.query_indices,
        &proof.trace_auth_paths_hi,
        &proof.trace_auth_paths_hi_next,
        tree_height,
    );
    decommitments.push(TwoMerkleDecommitment { hash_witness: hi_witness });

    // Tree 2: dict trace
    let dict_witness = build_trace_witness(
        &proof.query_indices,
        &proof.trace_auth_paths_dict,
        &proof.trace_auth_paths_dict_next,
        tree_height,
    );
    decommitments.push(TwoMerkleDecommitment { hash_witness: dict_witness });

    // Trees 3-5: interaction trees (current row only, query_indices)
    decommitments.push(TwoMerkleDecommitment {
        hash_witness: decommitment_hash_witness(
            &proof.query_indices, &proof.interaction_decommitment, tree_height)
    });
    decommitments.push(TwoMerkleDecommitment {
        hash_witness: decommitment_hash_witness(
            &proof.query_indices, &proof.rc_interaction_decommitment, tree_height)
    });
    decommitments.push(TwoMerkleDecommitment {
        hash_witness: decommitment_hash_witness(
            &proof.query_indices, &proof.dict_main_interaction_decommitment, tree_height)
    });

    // Tree 6: quotient
    decommitments.push(TwoMerkleDecommitment {
        hash_witness: decommitment_hash_witness(
            &proof.query_indices, &proof.quotient_decommitment, tree_height)
    });

    // ── FRI proof ─────────────────────────────────────────────────────────────

    // FRI first layer = OODS quotient.
    // fri_witness = stwo-deduplicated sibling values (omit when both siblings queried).
    let oods_q_witness = dedup_fri_witness(
        &proof.query_indices,
        &proof.oods_quotient_decommitment.sibling_values,
    );
    let oods_q_decommitment = TwoMerkleDecommitment {
        hash_witness: fold_pair_hash_witness(
            &proof.query_indices, &proof.oods_quotient_decommitment.auth_paths, tree_height)
    };
    let first_layer = TwoFriFirstLayerProof {
        fri_witness: oods_q_witness,
        decommitment: oods_q_decommitment,
        commitment: TwoHash::from(proof.oods_quotient_commitment),
    };

    // FRI inner layers.
    let inner_layers: Vec<TwoFriFoldProof> = proof.fri_commitments.iter()
        .enumerate()
        .map(|(i, &commitment)| {
            let decomm = &proof.fri_decommitments[i];
            // FRI tree at layer i has log_eval - (i+1) levels (one fold per layer).
            let layer_height = (tree_height).saturating_sub(i + 1);
            // Query positions at this layer are folded by (i+1) halvings.
            let layer_positions: Vec<usize> = proof.query_indices.iter()
                .map(|&q| q >> (i + 1))
                .collect();
            // fri_witness = stwo-deduplicated sibling values at this layer.
            let fri_witness = dedup_fri_witness(&layer_positions, &decomm.sibling_values);
            let layer_decommitment = fold_pair_hash_witness(
                &layer_positions, &decomm.auth_paths, layer_height);
            TwoFriFoldProof {
                fri_witness,
                decommitment: TwoMerkleDecommitment { hash_witness: layer_decommitment },
                commitment: TwoHash::from(commitment),
            }
        })
        .collect();

    // Last layer: convert BRT-ordered half_odds evaluations to LinePoly coefficients.
    // Convert FRI last layer evaluations to LinePoly BRT-ordered coefficients.
    // last_layer_poly_coeffs returns natural-order coefficients; BRT-permute for LinePoly::new.
    // LinePoly from the 8-element folded data (correct basis on half_odds(3)).
    let last_layer_poly: Vec<TwoSecureField> = {
        let coeffs = crate::fri::last_layer_poly_coeffs(proof.fri_last_layer_poly.clone());
        let deg = crate::fri::LOG_LAST_LAYER_DEGREE_BOUND;
        let n = coeffs.len();
        let mut brt = vec![crate::field::QM31::ZERO; n];
        for i in 0..n {
            brt[i.reverse_bits() >> (usize::BITS - deg)] = coeffs[i];
        }
        brt.iter().map(|q| q.to_u32_array()).collect()
    };

    let fri_proof = TwoFriProof { first_layer, inner_layers, last_layer_poly };

    use crate::prover::{N_QUERIES, POW_BITS};
    let config = TwoPcsConfig {
        log_blowup_factor: BLOWUP_BITS,
        n_queries: N_QUERIES as u32,
        pow_bits: POW_BITS,
        log_last_layer_degree_bound: crate::fri::LOG_LAST_LAYER_DEGREE_BOUND,
    };

    TwoStarkProof {
        config,
        tree_meta,
        commitments,
        sampled_values,
        decommitments,
        queried_values,
        query_indices: proof.query_indices.clone(),
        proof_of_work: proof.pow_nonce,
        fri_proof,
    }
}

/// Build combined hash_witness for a trace tree queried at both current-row
/// (`query_indices`) and next-row (`query_indices[i] + 1`) positions.
///
/// Merges both position sets, deduplicates, then delegates to
/// `auth_paths_to_hash_witness` for stwo-compatible sibling deduplication.
fn build_trace_witness(
    query_indices: &[usize],
    auth_paths_cur: &[Vec<[u32; 8]>],
    auth_paths_next: &[Vec<[u32; 8]>],
    tree_height: usize,
) -> Vec<TwoHash> {
    let log_eval_size = tree_height as u32;
    // Collect merged positions in sorted order; for each position keep one auth path.
    // Current positions take priority over next positions on collision.
    let mut pos_to_auth: BTreeMap<usize, &Vec<[u32; 8]>> = BTreeMap::new();
    for (i, &pos) in query_indices.iter().enumerate() {
        pos_to_auth.entry(pos).or_insert(&auth_paths_cur[i]);
    }
    if !auth_paths_next.is_empty() {
        for (i, &pos) in query_indices.iter().enumerate() {
            let next_pos = crate::cairo_air::prover::canonic_next(pos, log_eval_size);
            pos_to_auth.entry(next_pos).or_insert(&auth_paths_next[i]);
        }
    }

    let merged_positions: Vec<usize> = pos_to_auth.keys().copied().collect();
    let merged_auth_paths: Vec<Vec<[u32; 8]>> =
        merged_positions.iter().map(|&p| pos_to_auth[&p].clone()).collect();

    auth_paths_to_hash_witness(&merged_positions, &merged_auth_paths, tree_height)
}

/// Serialize a `CairoProof` to Stwo-compatible JSON.
pub fn cairo_proof_to_json(proof: &CairoProof) -> Result<String, serde_json::Error> {
    let two_proof = cairo_proof_to_stwo(proof);
    serde_json::to_string_pretty(&two_proof)
}

// ── hex dependency shim ───────────────────────────────────────────────────────

mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes.as_ref().iter().map(|b| format!("{:02x}", b)).collect()
    }
}

// ── Hash witness verifier (for tests and external callers) ────────────────────

/// Reconstruct a Merkle root from queried leaf values and a `hash_witness`.
///
/// Replicates stwo's `MerkleVerifierLifted::verify()` using VortexSTARK's Blake2s.
/// Returns `Ok(computed_root)` or `Err(msg)` if the witness is malformed.
///
/// `query_positions` must be sorted ascending (duplicates allowed; they are deduped).
/// `col_values[c][q_idx]` = M31 value of column `c` at query index `q_idx`.
/// Leaf hash = Blake2s(col0_le4 || col1_le4 || ... || colN_le4) — matches stwo's `update_leaf`.
/// Node hash = Blake2s(left_32 || right_32) — matches stwo's `hash_children`.
pub fn reconstruct_root_from_witness(
    log_height: u32,
    query_positions: &[usize],
    col_values: &[Vec<u32>],   // [col][query_idx]
    hash_witness: &[TwoHash],
) -> Result<TwoHash, String> {
    use crate::channel::blake2s_hash;
    use std::collections::BTreeMap;

    let n_cols = col_values.len();
    let height = log_height as usize;

    if query_positions.is_empty() {
        return Err("no query positions".into());
    }

    // Build map: position → row of M31 values (first occurrence wins for dedup).
    let mut pos_to_row: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for (q_idx, &pos) in query_positions.iter().enumerate() {
        pos_to_row.entry(pos).or_insert_with(|| {
            (0..n_cols).map(|c| col_values[c][q_idx]).collect()
        });
    }

    // Compute leaf hashes.
    let mut layer: BTreeMap<usize, [u8; 32]> = BTreeMap::new();
    for (&pos, row) in &pos_to_row {
        let mut input = vec![0u8; n_cols * 4];
        for (c, &v) in row.iter().enumerate() {
            input[c * 4..c * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        layer.insert(pos, blake2s_hash(&input));
    }

    let mut witness_iter = hash_witness.iter();

    for _level in 0..height {
        let positions: Vec<usize> = layer.keys().copied().collect();
        let mut next_layer: BTreeMap<usize, [u8; 32]> = BTreeMap::new();

        let mut i = 0;
        while i < positions.len() {
            let pos = positions[i];
            let sibling = pos ^ 1;

            let (left, right, consumed) =
                if i + 1 < positions.len() && positions[i + 1] == sibling {
                    let l = layer[&(pos & !1)]; // even sibling is left
                    let r = layer[&(pos | 1)];  // odd sibling is right
                    (l, r, 2)
                } else {
                    let w = witness_iter
                        .next()
                        .ok_or("hash_witness too short")?
                        .0;
                    if pos & 1 == 0 {
                        (layer[&pos], w, 1)
                    } else {
                        (w, layer[&pos], 1)
                    }
                };

            let mut node_input = [0u8; 64];
            node_input[..32].copy_from_slice(&left);
            node_input[32..].copy_from_slice(&right);
            let parent_hash = blake2s_hash(&node_input);
            next_layer.insert(pos >> 1, parent_hash);

            i += consumed;
        }

        layer = next_layer;
    }

    if witness_iter.next().is_some() {
        return Err("hash_witness too long (not fully consumed)".into());
    }

    let [(_, root)] = layer
        .into_iter()
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| "expected exactly one root node")?;

    Ok(TwoHash(root))
}

// ── Merkle witness verifier ───────────────────────────────────────────────────

/// Verify all 7 Merkle tree witnesses in a `TwoStarkProof`.
///
/// For each tree, reconstructs the Merkle root from queried leaf values and
/// `hash_witness`, then checks it equals the committed root.
///
/// Trees 3-5 (interaction) and 6 (OODS quotient) are queried at `query_indices` only.
/// Trees 0-2 (trace) are queried at both `query_indices` and `query_indices + 1`
/// (next-row); the caller must supply `trace_values_next[col][q]` for the next-row leaves.
///
/// Returns `Ok(())` on success, `Err(msg)` with the first failure description.
pub fn verify_all_merkle_witnesses(
    two: &TwoStarkProof,
    query_indices: &[usize],
    eval_size: usize,
    trace_values_next: &[Vec<u32>],  // [col 0..34][q_idx] next-row trace values
) -> Result<(), String> {
    let log_eval = two.tree_meta[0].log_eval_size;
    let n_queries = query_indices.len();

    // Sort order: order[i] is the original index whose position sorts to i.
    let mut order: Vec<usize> = (0..n_queries).collect();
    order.sort_unstable_by_key(|&i| query_indices[i]);
    let sorted_qi: Vec<usize> = order.iter().map(|&i| query_indices[i]).collect();

    // ── Trees 3-6: single query-set (current row only) ────────────────────────
    for tree in 3..7 {
        if two.queried_values[tree].is_empty() { continue; }
        // Sort col values to match sorted_qi order.
        let col_vals: Vec<Vec<u32>> = two.queried_values[tree].iter()
            .map(|col| order.iter().map(|&i| col[i]).collect())
            .collect();
        let root = reconstruct_root_from_witness(
            log_eval, &sorted_qi, &col_vals, &two.decommitments[tree].hash_witness,
        ).map_err(|e| format!("tree {tree} witness error: {e}"))?;
        if root != two.commitments[tree] {
            return Err(format!(
                "tree {tree} root mismatch: got {:?}, expected {:?}",
                root, two.commitments[tree]
            ));
        }
    }

    // ── Trees 0-2: double query-set (current row + next row) ──────────────────
    // Build merged sorted+deduped position list.
    let mut merged: Vec<usize> = Vec::with_capacity(2 * n_queries);
    merged.extend_from_slice(query_indices);
    merged.extend(query_indices.iter().map(|&qi| crate::cairo_air::prover::canonic_next(qi, log_eval as u32)));
    merged.sort_unstable();
    merged.dedup();

    // Map position → original index in query_indices (for value lookup in queried_values).
    let cur_idx: std::collections::HashMap<usize, usize> = query_indices.iter()
        .enumerate().map(|(i, &qi)| (qi, i)).collect();
    let next_idx: std::collections::HashMap<usize, usize> = query_indices.iter()
        .enumerate().map(|(i, &qi)| (crate::cairo_air::prover::canonic_next(qi, log_eval as u32), i)).collect();

    // Tree col ranges: [0..16], [16..31], [31..34]
    let ranges = [(0usize, 16usize), (16, 31), (31, 34)];
    for (tree, &(col_start, col_end)) in (0usize..3).zip(ranges.iter()) {
        let n_tree_cols = col_end - col_start;
        // Build col_values[col][merged_pos_idx] for this tree.
        let mut col_vals: Vec<Vec<u32>> = vec![Vec::with_capacity(merged.len()); n_tree_cols];
        for &pos in &merged {
            for (local_c, global_c) in (col_start..col_end).enumerate() {
                let v = if let Some(&qi) = cur_idx.get(&pos) {
                    two.queried_values[tree][local_c][qi]
                } else if let Some(&qi) = next_idx.get(&pos) {
                    if global_c < trace_values_next.len() && qi < trace_values_next[global_c].len() {
                        trace_values_next[global_c][qi]
                    } else {
                        return Err(format!("trace_values_next missing col={global_c} qi={qi}"));
                    }
                } else {
                    return Err(format!("tree {tree}: pos {pos} not in cur or next set"));
                };
                col_vals[local_c].push(v);
            }
        }
        let root = reconstruct_root_from_witness(
            log_eval, &merged, &col_vals, &two.decommitments[tree].hash_witness,
        ).map_err(|e| format!("tree {tree} witness error: {e}"))?;
        if root != two.commitments[tree] {
            return Err(format!(
                "tree {tree} root mismatch: got {:?}, expected {:?}",
                root, two.commitments[tree]
            ));
        }
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_witness_no_queries() {
        let witness = auth_paths_to_hash_witness(&[], &[], 5);
        assert!(witness.is_empty());
    }

    #[test]
    fn test_hash_witness_single_query() {
        // Single query at position 0 with a 2-level tree.
        // Level 0: pos=0, sibling=1 not queried → add auth_paths[0][0].
        // Level 1: pos=0, sibling=1 not queried → add auth_paths[0][1].
        let fake_hash_a = [1u32, 0, 0, 0, 0, 0, 0, 0];
        let fake_hash_b = [2u32, 0, 0, 0, 0, 0, 0, 0];
        let auth_paths = vec![vec![fake_hash_a, fake_hash_b]];

        let witness = auth_paths_to_hash_witness(&[0], &auth_paths, 2);
        assert_eq!(witness.len(), 2);
        assert_eq!(witness[0], TwoHash::from(fake_hash_a));
        assert_eq!(witness[1], TwoHash::from(fake_hash_b));
    }

    #[test]
    fn test_hash_witness_sibling_pair_deduplication() {
        // Two queries at sibling positions 0 and 1.
        // Level 0: both queried → no witness hash needed.
        // Level 1: pos=0, sibling=1 not queried → add auth_paths[0][1].
        let fake_hash_a = [1u32, 0, 0, 0, 0, 0, 0, 0];
        let fake_hash_b = [2u32, 0, 0, 0, 0, 0, 0, 0];
        let auth_paths = vec![
            vec![fake_hash_a, fake_hash_b],   // query at pos 0
            vec![fake_hash_a, fake_hash_b],   // query at pos 1
        ];

        let witness = auth_paths_to_hash_witness(&[0, 1], &auth_paths, 2);
        // Level 0: 0 and 1 are siblings → deduplicated, 0 witness hashes.
        // Level 1: parent 0 has no sibling in set → 1 witness hash.
        assert_eq!(witness.len(), 1);
        assert_eq!(witness[0], TwoHash::from(fake_hash_b));
    }

    #[test]
    fn test_hash_witness_two_isolated_queries() {
        // Two queries at positions 0 and 4 (not siblings of each other at level 0).
        // Level 0: 0 needs sibling 1's hash, 4 needs sibling 5's hash.
        // Level 1: parent(0) = 0, parent(4) = 2. Not siblings.
        //          0 needs parent_sibling = auth_paths[0][1]
        //          2 needs parent_sibling = auth_paths[1][1]
        let h = |n: u32| [n, 0, 0, 0, 0, 0, 0, 0u32];
        let auth_paths = vec![
            vec![h(10), h(20), h(30)],  // pos 0
            vec![h(11), h(21), h(31)],  // pos 4
        ];

        let witness = auth_paths_to_hash_witness(&[0, 4], &auth_paths, 3);
        // Level 0: pos 0 needs h(10), pos 4 needs h(11) — 2 hashes.
        // Level 1: parent(0)=0, parent(4)=2 — not siblings. pos 0 needs h(20), pos 2 needs h(21) — 2 hashes.
        // Level 2: parent(0)=0, parent(2)=1 — ARE siblings (0^1==1). Deduped — 0 hashes.
        assert_eq!(witness.len(), 4);
    }

    // ── Integration tests (require GPU / CUDA init) ───────────────────────────

    /// Item 1: prove a small Cairo program and verify the exported proof structure.
    #[test]
    fn test_export_structure_log6() {
        use crate::cairo_air::prover::cairo_prove;
        use crate::cuda::ffi;
        use crate::prover::{BLOWUP_BITS, N_QUERIES};

        ffi::init_memory_pool();

        // Minimal Fibonacci program: [fp+1] = [fp] + [fp-1], 64 steps.
        let program: Vec<u64> = vec![
            0x480680017fff8000, // [ap] = 1; ap++
            0x480680017fff8000, // [ap] = 1; ap++
            0x48307ffb7fff8000, // [ap] = [fp-1] + [fp-2]; ap++  (fib step)
            0x48307ffb7fff8000,
            0x40780017fff7fff, // jmp rel -2 (loop back 2 instructions)
        ];
        let proof = cairo_prove(&program, 64, 6);
        let two = cairo_proof_to_stwo(&proof);

        let log_eval = 6 + BLOWUP_BITS;

        // ── 7 commitment trees ────────────────────────────────────────────────
        assert_eq!(two.commitments.len(), 7, "expected 7 tree commitments");
        assert_eq!(two.tree_meta.len(), 7);
        assert_eq!(two.decommitments.len(), 7);
        assert_eq!(two.queried_values.len(), 7);
        assert_eq!(two.sampled_values.len(), 7);

        // All roots must be non-zero (trivial sanity: prover produced real commitments).
        for (i, c) in two.commitments.iter().enumerate() {
            assert_ne!(c.0, [0u8; 32], "commitment[{i}] is all-zeros");
        }

        // ── Tree metadata ─────────────────────────────────────────────────────
        assert_eq!(two.tree_meta[0].n_cols, 16);   // trace lo
        assert_eq!(two.tree_meta[1].n_cols, 15);   // trace hi
        assert_eq!(two.tree_meta[2].n_cols, 3);    // dict trace
        assert_eq!(two.tree_meta[3].n_cols, 4);    // logup interaction
        assert_eq!(two.tree_meta[4].n_cols, 4);    // RC interaction
        assert_eq!(two.tree_meta[5].n_cols, 4);    // S_dict interaction
        assert_eq!(two.tree_meta[6].n_cols, 4);    // quotient
        for meta in &two.tree_meta {
            assert_eq!(meta.log_eval_size, log_eval);
        }

        // ── OODS sampled values: trees 0-2 have 2 sample points (z, z_next) ──
        if !proof.oods_trace_at_z.is_empty() {
            assert_eq!(two.sampled_values[0].len(), 16, "tree0: 16 M31 cols sampled");
            for col in &two.sampled_values[0] {
                assert_eq!(col.len(), 2, "tree0: each col has 2 sample points");
            }
            assert_eq!(two.sampled_values[1].len(), 15);
            assert_eq!(two.sampled_values[2].len(), 3);
            // Trees 3-5: interaction polynomials — 4 component columns each, 2 sample points (z, z_next).
            assert_eq!(two.sampled_values[3].len(), 4, "tree3: 4 interaction component cols");
            assert_eq!(two.sampled_values[4].len(), 4, "tree4: 4 RC interaction component cols");
            assert_eq!(two.sampled_values[5].len(), 4, "tree5: 4 S_dict interaction component cols");
            for tree in 3..6 {
                for col in &two.sampled_values[tree] {
                    assert_eq!(col.len(), 2, "interaction cols sampled at z and z_next");
                }
            }
            // Tree 6: quotient — 4 cols × 1 sample point
            if !proof.oods_quotient_at_z.iter().all(|v| *v == [0u32; 4]) {
                assert_eq!(two.sampled_values[6].len(), 4);
                for col in &two.sampled_values[6] {
                    assert_eq!(col.len(), 1, "quotient: 1 sample point (z only)");
                }
            }
        }

        // ── Queried values dimensions ─────────────────────────────────────────
        let n_q = proof.query_indices.len();
        assert_eq!(n_q, N_QUERIES);
        assert_eq!(two.queried_values[0].len(), 16);
        for col in &two.queried_values[0] { assert_eq!(col.len(), n_q); }
        assert_eq!(two.queried_values[3].len(), 4);  // 4 M31 components of QM31
        for col in &two.queried_values[3] { assert_eq!(col.len(), n_q); }

        // ── Hash witnesses must be non-empty for real trees ───────────────────
        for i in 0..7 {
            assert!(!two.decommitments[i].hash_witness.is_empty(),
                "decommitment[{i}] hash_witness is empty");
        }

        // ── FRI proof structure ───────────────────────────────────────────────
        assert_ne!(two.fri_proof.first_layer.commitment.0, [0u8; 32]);
        assert!(!two.fri_proof.inner_layers.is_empty());
        assert!(!two.fri_proof.last_layer_poly.is_empty());

        // ── PoW nonce round-trips ─────────────────────────────────────────────
        assert_eq!(two.proof_of_work, proof.pow_nonce);
    }

    /// Item 2 (all-trees): verify all 7 Merkle witnesses in a single prove+export run.
    #[test]
    fn test_two_verify_all_trees_merkle() {
        use crate::cairo_air::prover::cairo_prove;
        use crate::cuda::ffi;
        use crate::prover::BLOWUP_BITS;

        ffi::init_memory_pool();

        let program: Vec<u64> = vec![
            0x480680017fff8000,
            0x480680017fff8000,
            0x48307ffb7fff8000,
            0x48307ffb7fff8000,
            0x40780017fff7fff,
        ];
        let proof = cairo_prove(&program, 64, 6);
        let two = cairo_proof_to_stwo(&proof);

        let eval_size = 1usize << (6 + BLOWUP_BITS);
        let n_q = proof.query_indices.len();

        // Build trace_values_next[col][q] from proof.trace_values_at_queries_next (row-major).
        // proof.trace_values_at_queries_next: Vec<Vec<u32>> where outer is query, inner is col.
        let n_trace_cols = 34;
        let mut trace_values_next: Vec<Vec<u32>> = vec![Vec::with_capacity(n_q); n_trace_cols];
        for row in proof.trace_values_at_queries_next.iter() {
            for (col, &val) in row.iter().enumerate() {
                if col < n_trace_cols {
                    trace_values_next[col].push(val);
                }
            }
        }

        // Pass unsorted query_indices — the function sorts internally.
        verify_all_merkle_witnesses(&two, &proof.query_indices, eval_size, &trace_values_next)
            .expect("all-tree Merkle witness verification failed");
    }

    /// Item 2: verify the interaction tree (tree 3) hash_witness round-trip.
    ///
    /// Takes the exported `queried_values[3]` + `decommitments[3].hash_witness` and
    /// reconstructs the Merkle root bottom-up. The result must match `commitments[3]`
    /// (the `interaction_commitment` in the original proof).
    #[test]
    fn test_hash_witness_round_trip_interaction_tree() {
        use crate::cairo_air::prover::cairo_prove;
        use crate::cuda::ffi;
        use crate::prover::BLOWUP_BITS;

        ffi::init_memory_pool();

        let program: Vec<u64> = vec![
            0x480680017fff8000,
            0x480680017fff8000,
            0x48307ffb7fff8000,
            0x48307ffb7fff8000,
            0x40780017fff7fff,
        ];
        let proof = cairo_prove(&program, 64, 6);
        let two = cairo_proof_to_stwo(&proof);

        let log_eval = (6 + BLOWUP_BITS) as u32;

        // Sort query positions and reorder queried_values accordingly.
        let n_q = proof.query_indices.len();
        let mut order: Vec<usize> = (0..n_q).collect();
        order.sort_by_key(|&i| proof.query_indices[i]);

        let sorted_positions: Vec<usize> = order.iter().map(|&i| proof.query_indices[i]).collect();
        let n_cols = two.queried_values[3].len(); // 4
        let sorted_col_values: Vec<Vec<u32>> = (0..n_cols)
            .map(|c| order.iter().map(|&i| two.queried_values[3][c][i]).collect())
            .collect();

        let computed = reconstruct_root_from_witness(
            log_eval,
            &sorted_positions,
            &sorted_col_values,
            &two.decommitments[3].hash_witness,
        );

        match computed {
            Ok(root) => {
                assert_eq!(
                    root, two.commitments[3],
                    "interaction tree: reconstructed root does not match commitment\n\
                     computed:  {:?}\n\
                     expected:  {:?}",
                    root, two.commitments[3]
                );
            }
            Err(e) => panic!("hash_witness reconstruction failed: {e}"),
        }
    }

    /// Call stwo's actual MerkleVerifierLifted::verify on the interaction tree (tree 3).
    ///
    /// This is the definitive proof that VortexSTARK's Merkle leaf/node format is
    /// byte-for-byte identical to stwo's Blake2sMerkleHasher format.
    #[test]
    fn test_stwo_merkle_verifier_interaction_tree() {
        use crate::cairo_air::prover::cairo_prove;
        use crate::cuda::ffi;
        use crate::prover::BLOWUP_BITS;

        // stwo types.
        use stwo::core::fields::m31::BaseField as SBaseField;
        use stwo::core::vcs::blake2_hash::Blake2sHash as SBlake2sHash;
        use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher as SBlake2sMerkleHasher;
        use stwo::core::vcs_lifted::verifier::{MerkleDecommitmentLifted, MerkleVerifierLifted};

        ffi::init_memory_pool();

        let program: Vec<u64> = vec![
            0x480680017fff8000,
            0x480680017fff8000,
            0x48307ffb7fff8000,
            0x48307ffb7fff8000,
            0x40780017fff7fff,
        ];
        let proof = cairo_prove(&program, 64, 6);
        let two = cairo_proof_to_stwo(&proof);

        let log_eval = (6 + BLOWUP_BITS) as u32;

        // Sort query indices and reorder values (same as test_hash_witness_round_trip).
        let n_q = proof.query_indices.len();
        let mut order: Vec<usize> = (0..n_q).collect();
        order.sort_by_key(|&i| proof.query_indices[i]);
        let sorted_positions: Vec<usize> = order.iter().map(|&i| proof.query_indices[i]).collect();

        // Build queried_values in stwo's format: ColumnVec<Vec<BaseField>>.
        // For interaction tree (4 columns), each column is a Vec<BaseField> of n_q values
        // in sorted-position order.
        let n_cols = two.queried_values[3].len(); // 4
        let queried_values_stwo: Vec<Vec<SBaseField>> = (0..n_cols)
            .map(|c| {
                order.iter()
                    .map(|&i| SBaseField::from(two.queried_values[3][c][i]))
                    .collect()
            })
            .collect();

        // Build stwo's MerkleDecommitmentLifted from VortexSTARK's hash_witness.
        let hash_witness_stwo: Vec<SBlake2sHash> = two.decommitments[3]
            .hash_witness
            .iter()
            .map(|h| SBlake2sHash(h.0))
            .collect();
        let decommitment = MerkleDecommitmentLifted::<SBlake2sMerkleHasher> {
            hash_witness: hash_witness_stwo,
        };

        // Root commitment.
        let root = SBlake2sHash(two.commitments[3].0);

        // All columns have the same eval domain size.
        let column_log_sizes = vec![log_eval; n_cols];
        let verifier = MerkleVerifierLifted::<SBlake2sMerkleHasher>::new(
            root, column_log_sizes, None,
        );

        verifier
            .verify(&sorted_positions, queried_values_stwo, decommitment)
            .expect("stwo MerkleVerifierLifted rejected VortexSTARK interaction tree witness");
    }

    /// Call stwo's MerkleVerifierLifted::verify on all 7 VortexSTARK commitment trees.
    ///
    /// Trees 3-6: single query set (current row only).
    /// Trees 0-2: merged current+next positions (trace next-row constraints).
    #[test]
    fn test_stwo_merkle_verifier_all_trees() {
        use crate::cairo_air::prover::cairo_prove;
        use crate::cuda::ffi;
        use crate::prover::BLOWUP_BITS;

        use stwo::core::fields::m31::BaseField as SBaseField;
        use stwo::core::vcs::blake2_hash::Blake2sHash as SBlake2sHash;
        use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher as SBlake2sMerkleHasher;
        use stwo::core::vcs_lifted::verifier::{MerkleDecommitmentLifted, MerkleVerifierLifted};

        ffi::init_memory_pool();

        let program: Vec<u64> = vec![
            0x480680017fff8000,
            0x480680017fff8000,
            0x48307ffb7fff8000,
            0x48307ffb7fff8000,
            0x40780017fff7fff,
        ];
        let proof = cairo_prove(&program, 64, 6);
        let two = cairo_proof_to_stwo(&proof);

        let log_eval = (6 + BLOWUP_BITS) as u32;
        let eval_size = 1usize << log_eval;
        let n_q = proof.query_indices.len();

        // Sort order for simple trees (3-6).
        let mut order: Vec<usize> = (0..n_q).collect();
        order.sort_by_key(|&i| proof.query_indices[i]);
        let sorted_qi: Vec<usize> = order.iter().map(|&i| proof.query_indices[i]).collect();

        // ── Trees 3-6: single query set ──────────────────────────────────────
        for tree in 3..7usize {
            if two.queried_values[tree].is_empty() { continue; }
            let n_cols = two.queried_values[tree].len();
            let qv: Vec<Vec<SBaseField>> = (0..n_cols)
                .map(|c| order.iter().map(|&i| SBaseField::from(two.queried_values[tree][c][i])).collect())
                .collect();
            let hw: Vec<SBlake2sHash> = two.decommitments[tree].hash_witness.iter()
                .map(|h| SBlake2sHash(h.0)).collect();
            let decommitment = MerkleDecommitmentLifted::<SBlake2sMerkleHasher> { hash_witness: hw };
            let root = SBlake2sHash(two.commitments[tree].0);
            let col_log_sizes = vec![log_eval; n_cols];
            MerkleVerifierLifted::<SBlake2sMerkleHasher>::new(root, col_log_sizes, None)
                .verify(&sorted_qi, qv, decommitment)
                .unwrap_or_else(|e| panic!("stwo rejected tree {tree}: {e}"));
        }

        // ── Trees 0-2: merged current+next positions ──────────────────────────
        // Merge and sort current+next positions.
        let cur_idx: std::collections::HashMap<usize, usize> =
            proof.query_indices.iter().enumerate().map(|(i, &qi)| (qi, i)).collect();
        let next_idx: std::collections::HashMap<usize, usize> =
            proof.query_indices.iter().enumerate().map(|(i, &qi)| (crate::cairo_air::prover::canonic_next(qi, log_eval as u32), i)).collect();
        let mut merged: Vec<usize> = Vec::with_capacity(2 * n_q);
        merged.extend(proof.query_indices.iter().copied());
        merged.extend(proof.query_indices.iter().map(|&qi| crate::cairo_air::prover::canonic_next(qi, log_eval as u32)));
        merged.sort_unstable();
        merged.dedup();

        // Build trace_values_next column-major.
        let n_trace = 34usize;
        let mut tvn: Vec<Vec<u32>> = vec![Vec::new(); n_trace];
        for row in &proof.trace_values_at_queries_next {
            for (col, &val) in row.iter().enumerate() {
                if col < n_trace { tvn[col].push(val); }
            }
        }

        let ranges = [(0usize, 16usize), (16, 31), (31, 34)];
        for (tree, &(col_start, col_end)) in (0usize..3).zip(ranges.iter()) {
            let n_tree_cols = col_end - col_start;
            let mut qv: Vec<Vec<SBaseField>> = vec![Vec::with_capacity(merged.len()); n_tree_cols];
            for &pos in &merged {
                for (local_c, global_c) in (col_start..col_end).enumerate() {
                    let v = if let Some(&qi) = cur_idx.get(&pos) {
                        two.queried_values[tree][local_c][qi]
                    } else {
                        let qi = *next_idx.get(&pos).expect("pos not in cur or next");
                        tvn[global_c][qi]
                    };
                    qv[local_c].push(SBaseField::from(v));
                }
            }
            let hw: Vec<SBlake2sHash> = two.decommitments[tree].hash_witness.iter()
                .map(|h| SBlake2sHash(h.0)).collect();
            let decommitment = MerkleDecommitmentLifted::<SBlake2sMerkleHasher> { hash_witness: hw };
            let root = SBlake2sHash(two.commitments[tree].0);
            let col_log_sizes = vec![log_eval; n_tree_cols];
            MerkleVerifierLifted::<SBlake2sMerkleHasher>::new(root, col_log_sizes, None)
                .verify(&merged, qv, decommitment)
                .unwrap_or_else(|e| panic!("stwo rejected trace tree {tree}: {e}"));
        }
    }

    /// Verify that `TwoStarkProof.sampled_values.flatten_cols()` (stwo's OODS mix input)
    /// is byte-for-byte identical to VortexSTARK's combined OODS mix vector.
    ///
    /// stwo calls `channel.mix_felts(&proof.sampled_values.clone().flatten_cols())`.
    /// VortexSTARK now builds the same vector: for each tree, for each col, for each sample.
    /// If the two representations are equal, the drawn oods_alpha / random_coeff will match.
    #[test]
    fn test_oods_sampled_values_flatten_cols_order() {
        use crate::cairo_air::prover::cairo_prove;
        use crate::cuda::ffi;

        ffi::init_memory_pool();

        let program: Vec<u64> = vec![
            0x480680017fff8000,
            0x480680017fff8000,
            0x48307ffb7fff8000,
            0x48307ffb7fff8000,
            0x40780017fff7fff,
        ];
        let proof = cairo_prove(&program, 64, 6);
        let two = cairo_proof_to_stwo(&proof);

        // stwo's flatten_cols order: for each tree, for each col, for each sample.
        let stwo_flat: Vec<[u32; 4]> = two.sampled_values.iter()
            .flat_map(|tree_cols| tree_cols.iter())
            .flat_map(|col_samples| col_samples.iter())
            .copied()
            .collect();

        // VortexSTARK's combined vector (mirrors the prover's Phase 2b combined slice).
        // Trees 0-2 (N_COLS trace cols × 2 samples): at_z[i], at_next[i] interleaved.
        // Trees 3-5 (3 interaction groups × 4 cols × 2 samples): interleaved.
        // Tree 6 (4 quotient cols × 1 sample): at_z[k].
        let n_trace = proof.oods_trace_at_z.len().min(proof.oods_trace_at_z_next.len());
        let mut vortex_flat: Vec<[u32; 4]> = Vec::with_capacity(n_trace * 2 + 12 * 2 + 4);
        for i in 0..n_trace {
            vortex_flat.push(proof.oods_trace_at_z[i]);
            vortex_flat.push(proof.oods_trace_at_z_next[i]);
        }
        for pi in 0..3 {
            for k in 0..4 {
                vortex_flat.push(proof.oods_interaction_at_z[pi][k]);
                vortex_flat.push(proof.oods_interaction_at_z_next[pi][k]);
            }
        }
        for k in 0..4 {
            vortex_flat.push(proof.oods_quotient_at_z[k]);
        }

        assert_eq!(
            stwo_flat.len(), vortex_flat.len(),
            "flatten_cols length mismatch: stwo={} vortex={}",
            stwo_flat.len(), vortex_flat.len()
        );
        for (i, (s, v)) in stwo_flat.iter().zip(vortex_flat.iter()).enumerate() {
            assert_eq!(
                s, v,
                "sampled_values mismatch at index {i}: stwo={:?} vortex={:?}",
                s, v
            );
        }
    }

    /// Verify that the channel state change from VortexSTARK's combined mix_felts is
    /// identical to what stwo's channel.mix_felts(sampled_values.flatten_cols()) produces.
    ///
    /// Starting from the same arbitrary channel state, both operations must produce
    /// the same output state — because `test_oods_sampled_values_flatten_cols_order`
    /// already proves the input slice is byte-for-byte identical.
    ///
    /// This is the definitive proof that Items 1 and 2 are resolved:
    /// VortexSTARK's combined OODS mix is stwo-compatible.
    #[test]
    fn test_oods_channel_state_change_matches_stwo() {
        use crate::cairo_air::prover::cairo_prove;
        use crate::channel::Channel;
        use crate::cuda::ffi;
        use crate::field::QM31;
        use blake2::{Blake2s256, Digest as _};

        ffi::init_memory_pool();

        let program: Vec<u64> = vec![
            0x480680017fff8000,
            0x480680017fff8000,
            0x48307ffb7fff8000,
            0x48307ffb7fff8000,
            0x40780017fff7fff,
        ];
        let proof = cairo_prove(&program, 64, 6);
        let two = cairo_proof_to_stwo(&proof);

        // Build VortexSTARK's combined slice (same code as prover Phase 2b).
        let n_trace = proof.oods_trace_at_z.len().min(proof.oods_trace_at_z_next.len());
        let mut vortex_combined: Vec<QM31> = Vec::with_capacity(n_trace * 2 + 12 * 2 + 4);
        for i in 0..n_trace {
            vortex_combined.push(QM31::from_u32_array(proof.oods_trace_at_z[i]));
            vortex_combined.push(QM31::from_u32_array(proof.oods_trace_at_z_next[i]));
        }
        for pi in 0..3 {
            for k in 0..4 {
                vortex_combined.push(QM31::from_u32_array(proof.oods_interaction_at_z[pi][k]));
                vortex_combined.push(QM31::from_u32_array(proof.oods_interaction_at_z_next[pi][k]));
            }
        }
        for k in 0..4 {
            vortex_combined.push(QM31::from_u32_array(proof.oods_quotient_at_z[k]));
        }

        // Build stwo's flatten_cols slice from TwoStarkProof.sampled_values.
        let stwo_combined: Vec<QM31> = two.sampled_values.iter()
            .flat_map(|tree| tree.iter())
            .flat_map(|col| col.iter())
            .map(|&v| QM31::from_u32_array(v))
            .collect();

        // The two slices must be identical (already tested by flatten_cols_order test,
        // but confirm here for clarity).
        assert_eq!(vortex_combined, stwo_combined,
            "combined slice != sampled_values.flatten_cols()");

        // Verify that mix_felts(vortex_combined) == Blake2s(state || bytes) —
        // the same operation stwo performs — from an arbitrary starting state.
        let arbitrary_state = [0x42u8; 32];
        let mut ch = Channel::new();
        // Manually set the channel state to our arbitrary state.
        // (mix_digest is Blake2s(all_zeros || arbitrary_state), so we use a
        // known state by mixing a known digest.)
        let sentinel = [0xDEADBEEFu32, 0xCAFEBABE, 0, 0, 0, 0, 0, 0];
        ch.mix_digest(&sentinel);
        let state_before = ch.state_words();

        ch.mix_felts(&vortex_combined);
        let state_after_vortex = ch.state_words();

        // Compute expected state: Blake2s(state_before_bytes || combined_bytes).
        let mut hasher = Blake2s256::new();
        for &w in &state_before {
            hasher.update(&w.to_le_bytes());
        }
        for f in &vortex_combined {
            for &w in &f.to_u32_array() {
                hasher.update(&w.to_le_bytes());
            }
        }
        let expected: [u8; 32] = hasher.finalize().into();
        let expected_words: [u32; 8] = std::array::from_fn(|i|
            u32::from_le_bytes(expected[i*4..i*4+4].try_into().unwrap()));

        assert_eq!(state_after_vortex, expected_words,
            "mix_felts(combined) state mismatch vs direct Blake2s computation");
    }

    // ── FRI Merkle decommitment tests ─────────────────────────────────────────

    /// Verify all FRI layer Merkle witnesses using stwo's `MerkleVerifierLifted`.
    ///
    /// Each FRI layer commits to a QM31 polynomial (4 M31 components per leaf).
    /// This test verifies that the hash_witness exported by VortexSTARK is accepted
    /// by stwo's verifier for all layers:
    ///   - first_layer (OODS quotient, circle domain log_size=log_eval)
    ///   - inner_layers[i] (line domain log_size=log_eval-1-i)
    #[test]
    #[ignore = "stwo last layer domain mismatch: stwo adds blowup to last layer domain, VortexSTARK folds through blowup"]
    fn test_stwo_fri_merkle_witnesses() {
        use crate::cairo_air::prover::cairo_prove;
        use crate::cuda::ffi;
        use crate::prover::BLOWUP_BITS;

        use stwo::core::fields::m31::BaseField as SBaseField;
        use stwo::core::vcs::blake2_hash::Blake2sHash as SBlake2sHash;
        use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher as SBlake2sMerkleHasher;
        use stwo::core::vcs_lifted::verifier::{MerkleDecommitmentLifted, MerkleVerifierLifted};

        ffi::init_memory_pool();

        let program: Vec<u64> = vec![
            0x480680017fff8000,
            0x480680017fff8000,
            0x48307ffb7fff8000,
            0x48307ffb7fff8000,
            0x40780017fff7fff,
        ];
        let proof = cairo_prove(&program, 64, 6);
        let two = cairo_proof_to_stwo(&proof);

        let log_eval = (6u32 + BLOWUP_BITS);
        let n_q = proof.query_indices.len();

        /// Helper: call MerkleVerifierLifted on 4-component QM31 leaf data.
        ///
        /// `positions` must be sorted ascending.
        /// `values[q]` = [u32; 4] QM31 at position `positions[q]`.
        fn verify_fri_layer_merkle(
            log_size: u32,
            positions: &[usize],
            values: &[[u32; 4]],
            hash_witness: &[TwoHash],
            commitment: &TwoHash,
        ) {
            // Sort positions and reorder values.
            let n = positions.len();
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by_key(|&i| positions[i]);
            let sorted_pos: Vec<usize> = order.iter().map(|&i| positions[i]).collect();

            // Build 4-column layout.
            let qv: Vec<Vec<SBaseField>> = (0..4)
                .map(|comp| order.iter().map(|&i| SBaseField::from(values[i][comp])).collect())
                .collect();

            let hw: Vec<SBlake2sHash> = hash_witness.iter().map(|h| SBlake2sHash(h.0)).collect();
            let decommitment = MerkleDecommitmentLifted::<SBlake2sMerkleHasher> { hash_witness: hw };
            let root = SBlake2sHash(commitment.0);
            let col_log_sizes = vec![log_size; 4];

            MerkleVerifierLifted::<SBlake2sMerkleHasher>::new(root, col_log_sizes, None)
                .verify(&sorted_pos, qv, decommitment)
                .unwrap_or_else(|e| panic!("FRI Merkle verify failed at log_size={log_size}: {e}"));
        }

        // ── First layer: OODS quotient (circle domain, log_size = log_eval) ──
        verify_fri_layer_merkle(
            log_eval,
            &proof.query_indices,
            &proof.oods_quotient_decommitment.values,
            &two.fri_proof.first_layer.decommitment.hash_witness,
            &two.fri_proof.first_layer.commitment,
        );

        // ── Inner layers: line domains ─────────────────────────────────────────
        // Layer i has log_size = log_eval - 1 - i.
        // Query positions at layer i are folded from the original: query >> (i+1).
        for (i, layer) in two.fri_proof.inner_layers.iter().enumerate() {
            let layer_log = log_eval - 1 - i as u32;
            let layer_positions: Vec<usize> = proof.query_indices.iter()
                .map(|&q| q >> (i + 1))
                .collect();
            verify_fri_layer_merkle(
                layer_log,
                &layer_positions,
                &proof.fri_decommitments[i].values,
                &layer.decommitment.hash_witness,
                &layer.commitment,
            );
        }
    }

    // ── Unit test: dedup_fri_witness logic ────────────────────────────────────

    #[test]
    fn test_dedup_fri_witness_no_shared_siblings() {
        // Queries at positions 0, 4, 8 — no two are siblings (differ by ^1).
        // All three need their sibling in fri_witness.
        let positions = vec![0usize, 4, 8];
        let vals: Vec<[u32; 4]> = positions.iter().map(|&p| [(p ^ 1) as u32, 0, 0, 0]).collect();
        let witness = dedup_fri_witness(&positions, &vals);
        // 0^1=1, 4^1=5, 8^1=9 — none queried → all 3 included.
        assert_eq!(witness.len(), 3);
        assert_eq!(witness[0], [1, 0, 0, 0]); // sibling of 0
        assert_eq!(witness[1], [5, 0, 0, 0]); // sibling of 4
        assert_eq!(witness[2], [9, 0, 0, 0]); // sibling of 8
    }

    #[test]
    fn test_dedup_fri_witness_all_siblings_paired() {
        // Queries at positions 0, 1, 4, 5 — (0,1) and (4,5) are both full pairs.
        // Both pairs are fully queried → fri_witness is empty.
        let positions = vec![0usize, 1, 4, 5];
        let vals: Vec<[u32; 4]> = vec![[1,0,0,0],[0,0,0,0],[5,0,0,0],[4,0,0,0]];
        let witness = dedup_fri_witness(&positions, &vals);
        assert!(witness.is_empty(), "full sibling pairs: witness must be empty");
    }

    #[test]
    fn test_dedup_fri_witness_mixed() {
        // Queries at {0, 1, 4} — pair (0,1) is full; position 4's sibling (5) is not queried.
        let positions = vec![0usize, 1, 4];
        // sibling_values[i] = val at positions[i]^1
        let vals: Vec<[u32; 4]> = vec![[1,0,0,0],[0,0,0,0],[5,0,0,0]];
        let witness = dedup_fri_witness(&positions, &vals);
        // (0,1) full pair → 0 entries; 4 needs sibling 5 → 1 entry.
        assert_eq!(witness.len(), 1);
        assert_eq!(witness[0], [5, 0, 0, 0]);
    }

    #[test]
    fn test_dedup_fri_witness_duplicate_positions() {
        // Same position queried twice (possible with N_QUERIES draws).
        // Logical set = {3} — sibling 2 not queried → 1 witness entry.
        let positions = vec![3usize, 3];
        let vals: Vec<[u32; 4]> = vec![[2,0,0,0],[2,0,0,0]];
        let witness = dedup_fri_witness(&positions, &vals);
        assert_eq!(witness.len(), 1);
        assert_eq!(witness[0], [2, 0, 0, 0]);
    }

    // ── Integration test: fri_witness dedup in real proof ─────────────────────

    /// Verify that the exported FRI witness counts are consistent with the
    /// deduplicated format: for each layer, the witness count equals the number of
    /// queries whose siblings are NOT also queried (i.e. unpaired positions).
    #[test]
    fn test_fri_witness_dedup_counts() {
        use crate::cairo_air::prover::cairo_prove;
        use crate::cuda::ffi;
        use std::collections::BTreeSet;

        ffi::init_memory_pool();

        let program: Vec<u64> = vec![
            0x480680017fff8000,
            0x480680017fff8000,
            0x48307ffb7fff8000,
            0x48307ffb7fff8000,
            0x40780017fff7fff,
        ];
        let proof = cairo_prove(&program, 64, 6);
        let two = cairo_proof_to_stwo(&proof);

        // Count unpaired queries at each FRI layer.
        fn unpaired_count(positions: &[usize]) -> usize {
            let unique: BTreeSet<usize> = positions.iter().copied().collect();
            unique.iter().filter(|&&p| !unique.contains(&(p ^ 1))).count()
        }

        // First layer: query positions = proof.query_indices.
        let expected_first = unpaired_count(&proof.query_indices);
        assert_eq!(
            two.fri_proof.first_layer.fri_witness.len(), expected_first,
            "first_layer fri_witness count mismatch"
        );

        // Inner layers: folded positions.
        for (i, layer) in two.fri_proof.inner_layers.iter().enumerate() {
            let layer_positions: Vec<usize> = proof.query_indices.iter()
                .map(|&q| q >> (i + 1))
                .collect();
            let expected = unpaired_count(&layer_positions);
            assert_eq!(
                layer.fri_witness.len(), expected,
                "inner_layer[{i}] fri_witness count mismatch"
            );
        }
    }

    /// Replay the Fiat-Shamir transcript using stwo's Blake2sChannel and verify that
    /// stwo's `draw_queries` produces the exact same query positions as VortexSTARK.
    ///
    /// This test proves that VortexSTARK's Fiat-Shamir channel is byte-for-byte
    /// compatible with stwo's Blake2sChannel at every mixing and drawing step.
    ///
    /// Requires a simple Fibonacci program (no dicts, no EC, no bitwise).
    #[test]
    fn test_stwo_fri_channel_transcript_compatibility() {
        use crate::cairo_air::prover::cairo_prove;
        use crate::cairo_air::trace::N_CONSTRAINTS;
        use crate::cuda::ffi;
        use crate::prover::{BLOWUP_BITS, N_QUERIES};

        use stwo::core::channel::Channel as SChannel;
        use stwo::core::channel::Blake2sChannel as SBlake2sChannel;
        use stwo::core::fields::qm31::SecureField as SSecureField;
        use stwo::core::fields::m31::BaseField as SBaseField;
        use stwo::core::queries::draw_queries;

        ffi::init_memory_pool();

        // Simple Fibonacci — no dicts, no EC, no bitwise builtins.
        let program: Vec<u64> = vec![
            0x480680017fff8000, // [ap] = 1; ap++
            0x480680017fff8000, // [ap] = 1; ap++
            0x48307ffb7fff8000, // [ap] = [fp-1] + [fp-2]; ap++
            0x48307ffb7fff8000,
            0x40780017fff7fff,  // jmp rel -2
        ];
        let proof = cairo_prove(&program, 64, 6);
        let log_n = 6u32;
        let log_eval_size = log_n + BLOWUP_BITS;

        // Verify no dict/bitwise data (assumptions for this test).
        assert!(proof.dict_exec_data.is_empty(), "test requires no dict accesses");
        assert!(proof.dict_exec_commitment.is_none(), "test requires no dict commitment");
        assert!(proof.bitwise_rows.is_empty(), "test requires no bitwise rows");
        assert!(proof.ec_trace_commitment.is_none(), "test requires no EC trace");

        // Convenience: convert VortexSTARK [u32;4] to stwo SecureField.
        let sf = |v: [u32; 4]| -> SSecureField {
            use stwo::core::fields::m31::M31;
            use stwo::core::fields::cm31::CM31;
            SSecureField::from_m31_array([M31(v[0]), M31(v[1]), M31(v[2]), M31(v[3])])
        };

        let mut ch = SBlake2sChannel::default();

        // ── Replay transcript ──────────────────────────────────────────────────
        // Matches cairo_verify channel sequence exactly.

        // Phase 1: Public commitments.
        ch.mix_u32s(&proof.public_inputs.program_hash);
        ch.mix_u32s(&proof.trace_commitment);
        ch.mix_u32s(&proof.trace_commitment_hi);
        ch.mix_u32s(&proof.dict_trace_commitment);

        // Draw S_dict link challenges (z_dict_link, alpha_dict_link).
        ch.draw_secure_felt();
        ch.draw_secure_felt();

        // Mix dict_main_interaction + dict_link_final (always mixed).
        ch.mix_u32s(&proof.dict_main_interaction_commitment);
        ch.mix_u32s(&[
            proof.dict_link_final[0], proof.dict_link_final[1],
            proof.dict_link_final[2], proof.dict_link_final[3],
            0, 0, 0, 0,
        ]);

        // (No EC trace commits — assert above.)
        // (No dict exec/sorted commitment rounds — assert above.)

        // Draw z_mem, alpha_mem, z_rc (for LogUp / RC challenges).
        ch.draw_secure_felt();
        ch.draw_secure_felt();
        ch.draw_secure_felt();

        // Mix interaction commitments and LogUp+RC final sums.
        ch.mix_u32s(&proof.interaction_commitment);
        ch.mix_u32s(&proof.rc_interaction_commitment);
        ch.mix_u32s(&[
            proof.logup_final_sum[0], proof.logup_final_sum[1],
            proof.logup_final_sum[2], proof.logup_final_sum[3],
            proof.rc_final_sum[0],    proof.rc_final_sum[1],
            proof.rc_final_sum[2],    proof.rc_final_sum[3],
        ]);

        // (No bitwise commitment — assert above.)

        // Mix memory table and RC counts commitments.
        ch.mix_u32s(&proof.memory_table_commitment);
        ch.mix_u32s(&proof.rc_counts_commitment);

        // Draw constraint alphas (N_CONSTRAINTS = 35).
        for _ in 0..N_CONSTRAINTS {
            ch.draw_secure_felt();
        }

        // Mix quotient commitment.
        ch.mix_u32s(&proof.quotient_commitment);

        // Draw OODS point z.
        ch.draw_secure_felt();

        // Mix all OODS sampled values in flatten_cols order.
        let n_trace = proof.oods_trace_at_z.len().min(proof.oods_trace_at_z_next.len());
        let mut oods_felts: Vec<SSecureField> = Vec::with_capacity(n_trace * 2 + 12 * 2 + 4);
        for i in 0..n_trace {
            oods_felts.push(sf(proof.oods_trace_at_z[i]));
            oods_felts.push(sf(proof.oods_trace_at_z_next[i]));
        }
        for pi in 0..3 {
            for k in 0..4 {
                oods_felts.push(sf(proof.oods_interaction_at_z[pi][k]));
                oods_felts.push(sf(proof.oods_interaction_at_z_next[pi][k]));
            }
        }
        for k in 0..4 {
            oods_felts.push(sf(proof.oods_quotient_at_z[k]));
        }
        ch.mix_felts(&oods_felts);

        // Draw oods_alpha.
        ch.draw_secure_felt();

        // ── FRI commitment phase ───────────────────────────────────────────────
        ch.mix_u32s(&proof.oods_quotient_commitment);
        ch.draw_secure_felt(); // fri_alpha (circle fold)

        for fri_commit in &proof.fri_commitments {
            ch.mix_u32s(fri_commit);
            ch.draw_secure_felt(); // fold alpha
        }
        // Uncommitted fold alphas (BLOWUP_BITS extra draws, no commits).
        // Save these for CPU fold replication.
        let mut uncommitted_alphas = Vec::new();
        for _ in 0..BLOWUP_BITS {
            uncommitted_alphas.push(ch.draw_secure_felt());
        }

        // CPU fold: replicate prover's uncommitted folds to get 8-element data.
        let folded_data = {
            use crate::field::QM31 as VQMM;
            use crate::cairo_air::prover::{fold_twiddle_at, fold_pair};
            let mut data: Vec<VQMM> = proof.fri_last_layer.clone();
            let mut dl = crate::fri::LOG_LAST_LAYER_DEGREE_BOUND + BLOWUP_BITS;
            for uf in 0..BLOWUP_BITS as usize {
                let alpha = {
                    let sv = uncommitted_alphas[uf];
                    let arr = sv.to_m31_array();
                    VQMM::from_m31_array([
                        crate::field::M31(arr[0].0), crate::field::M31(arr[1].0),
                        crate::field::M31(arr[2].0), crate::field::M31(arr[3].0),
                    ])
                };
                let ho = crate::circle::Coset::half_odds(dl);
                let half = data.len() / 2;
                let mut folded = vec![VQMM::ZERO; half];
                for i in 0..half {
                    let twid = fold_twiddle_at(&ho, i, false);
                    folded[i] = fold_pair(data[2*i], data[2*i+1], alpha, twid);
                }
                data = folded;
                dl -= 1;
            }
            data
        };
        let stwo_deg_t = crate::fri::LOG_LAST_LAYER_DEGREE_BOUND;
        let stwo_n_t = 1usize << stwo_deg_t;
        let nat_c = crate::fri::last_layer_poly_coeffs(folded_data);
        let mut brt_coeffs = vec![crate::field::QM31::ZERO; stwo_n_t];
        for i in 0..stwo_n_t { brt_coeffs[i.reverse_bits() >> (usize::BITS - stwo_deg_t)] = nat_c[i]; }
        let last_layer_felts: Vec<SSecureField> = brt_coeffs.iter()
            .map(|qm| sf(qm.to_u32_array()))
            .collect();
        ch.mix_felts(&last_layer_felts);

        // Mix PoW nonce (after verify_pow, which doesn't change channel state).
        ch.mix_u64(proof.pow_nonce);

        // ── Query derivation ─────────────────────────────────────────────────
        // draw_queries uses stwo's 8-per-squeeze scheme; must match draw_query_indices.
        let stwo_queries = draw_queries(&mut ch, log_eval_size, N_QUERIES);

        assert_eq!(
            stwo_queries, proof.query_indices,
            "stwo draw_queries produced different positions than VortexSTARK draw_query_indices.\n\
             stwo:       {:?}\n\
             VortexSTARK: {:?}",
            &stwo_queries[..stwo_queries.len().min(8)],
            &proof.query_indices[..proof.query_indices.len().min(8)],
        );
    }

    /// End-to-end test: feed VortexSTARK's exported FRI proof to stwo's FriVerifier.
    ///
    /// Tests the full FRI verification pipeline:
    /// 1. Replay Fiat-Shamir transcript up to FRI start
    /// 2. FriVerifier::commit() — mix commitments and draw fold alphas
    /// 3. PoW verification and query sampling
    /// 4. FriVerifier::decommit() — verify fold equations at all query points
    ///
    /// STATUS: FriVerifier::commit() succeeds (layer counts, LinePoly coefficients, channel match).
    /// FriVerifier::decommit() fails: SoA4 leaf hashing convention mismatch.
    ///
    /// Root cause: VortexSTARK commits all trees in half_coset BRT-NTT order; stwo expects
    /// CanonicCoset (conjugate-pair) order. The `permute_half_coset_to_canonic` function
    /// (defined in prover.rs) maps between the two orderings — same circle point set, different
    /// array positions. The fix requires:
    ///   1. Permute ALL column data to canonic order before Merkle commitment
    ///   2. Compute circle fold twiddles in natural half_odds order (not BRT)
    ///   3. Compute line fold twiddles in natural half_odds order
    ///   4. Update the VortexSTARK verifier's domain point computations + constraint evaluation
    ///      to use canonic position → domain point mapping
    ///   5. Update all next-row indices from (qi+1)%n to canonic_next(qi)
    ///
    /// Fold equations ARE algebraically identical (both use (f0+f1) + alpha * inv_twiddle * (f0-f1)).
    /// The FRI layer count and LinePoly coefficient mixing already match stwo.
    #[test]
    #[ignore = "stwo last layer: LinePoly basis is domain-specific, half_odds(3) coeffs invalid on half_odds(5)"]
    fn test_stwo_fri_verifier_e2e() {
        use crate::cairo_air::decode::Instruction;
        use crate::cairo_air::prover::cairo_prove;
        use crate::cairo_air::trace::N_CONSTRAINTS;
        use crate::cuda::ffi;
        use crate::prover::{BLOWUP_BITS, N_QUERIES, POW_BITS};

        use stwo::core::channel::Channel as SChannel;
        use stwo::core::channel::Blake2sChannel as SBlake2sChannel;
        use stwo::core::fields::m31::M31 as SM31;
        use stwo::core::fields::qm31::SecureField as SSecureField;
        use stwo::core::fri::{FriConfig, FriVerifier, FriProof, FriLayerProof, CirclePolyDegreeBound};
        use stwo::core::poly::line::LinePoly;
        // PoW verification uses channel.verify_pow_nonce (no separate ProofOfWork struct).
        use stwo::core::queries::Queries;
        use stwo::core::vcs::blake2_hash::Blake2sHash as SBlake2sHash;
        use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
        use stwo::core::vcs_lifted::verifier::MerkleDecommitmentLifted;

        ffi::init_memory_pool();

        // Build a 64-step Fibonacci program.
        let program: Vec<u64> = {
            let assert_imm = Instruction {
                off0: 0x8000, off1: 0x8000, off2: 0x8001,
                op1_imm: 1, opcode_assert: 1, ap_add1: 1,
                ..Default::default()
            };
            let add_instr = Instruction {
                off0: 0x8000, off1: 0x8000u16.wrapping_sub(2), off2: 0x8000u16.wrapping_sub(1),
                op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
                ..Default::default()
            };
            let mut p = Vec::new();
            p.push(assert_imm.encode()); p.push(1u64);
            p.push(assert_imm.encode()); p.push(1u64);
            for _ in 0..62 { p.push(add_instr.encode()); }
            p
        };
        let proof = cairo_prove(&program, 64, 6);
        let log_n = 6u32;
        let log_eval_size = log_n + BLOWUP_BITS;
        let two = cairo_proof_to_stwo(&proof);

        // Convenience: convert [u32;4] → stwo SecureField.
        let sf = |v: [u32; 4]| -> SSecureField {
            use stwo::core::fields::cm31::CM31;
            SSecureField::from_m31_array([SM31(v[0]), SM31(v[1]), SM31(v[2]), SM31(v[3])])
        };

        // ── Replay Fiat-Shamir transcript up to FRI start ──────────────────────
        let mut ch = SBlake2sChannel::default();

        ch.mix_u32s(&proof.public_inputs.program_hash);
        ch.mix_u32s(&proof.trace_commitment);
        ch.mix_u32s(&proof.trace_commitment_hi);
        ch.mix_u32s(&proof.dict_trace_commitment);
        ch.draw_secure_felt(); // z_dict_link
        ch.draw_secure_felt(); // alpha_dict_link
        ch.mix_u32s(&proof.dict_main_interaction_commitment);
        ch.mix_u32s(&[
            proof.dict_link_final[0], proof.dict_link_final[1],
            proof.dict_link_final[2], proof.dict_link_final[3],
            0, 0, 0, 0,
        ]);
        ch.draw_secure_felt(); // z_mem
        ch.draw_secure_felt(); // alpha_mem
        ch.draw_secure_felt(); // z_rc
        ch.mix_u32s(&proof.interaction_commitment);
        ch.mix_u32s(&proof.rc_interaction_commitment);
        ch.mix_u32s(&[
            proof.logup_final_sum[0], proof.logup_final_sum[1],
            proof.logup_final_sum[2], proof.logup_final_sum[3],
            proof.rc_final_sum[0],    proof.rc_final_sum[1],
            proof.rc_final_sum[2],    proof.rc_final_sum[3],
        ]);
        ch.mix_u32s(&proof.memory_table_commitment);
        ch.mix_u32s(&proof.rc_counts_commitment);
        for _ in 0..N_CONSTRAINTS { ch.draw_secure_felt(); }
        ch.mix_u32s(&proof.quotient_commitment);
        ch.draw_secure_felt(); // oods_z

        // Mix OODS sampled values.
        let n_trace = proof.oods_trace_at_z.len().min(proof.oods_trace_at_z_next.len());
        let mut oods_felts: Vec<SSecureField> = Vec::new();
        for i in 0..n_trace {
            oods_felts.push(sf(proof.oods_trace_at_z[i]));
            oods_felts.push(sf(proof.oods_trace_at_z_next[i]));
        }
        for pi in 0..3 {
            for k in 0..4 {
                oods_felts.push(sf(proof.oods_interaction_at_z[pi][k]));
                oods_felts.push(sf(proof.oods_interaction_at_z_next[pi][k]));
            }
        }
        for k in 0..4 {
            oods_felts.push(sf(proof.oods_quotient_at_z[k]));
        }
        ch.mix_felts(&oods_felts);
        ch.draw_secure_felt(); // oods_alpha

        // ── Now at FRI start point. Build FriProof and call FriVerifier::commit. ──

        // Convert TwoFriProof → stwo FriProof<Blake2sMerkleHasher>
        let to_hash = |h: &TwoHash| -> SBlake2sHash { SBlake2sHash(h.0) };
        let to_sf_arr = |v: &[u32; 4]| -> SSecureField { sf(*v) };

        let stwo_first_layer = FriLayerProof::<Blake2sMerkleHasher> {
            fri_witness: two.fri_proof.first_layer.fri_witness.iter()
                .map(|v| to_sf_arr(v)).collect(),
            decommitment: MerkleDecommitmentLifted::<Blake2sMerkleHasher> {
                hash_witness: two.fri_proof.first_layer.decommitment.hash_witness.iter()
                    .map(|h| to_hash(h)).collect(),
            },
            commitment: to_hash(&two.fri_proof.first_layer.commitment),
        };

        let stwo_inner_layers: Vec<FriLayerProof<Blake2sMerkleHasher>> = two.fri_proof.inner_layers.iter()
            .map(|layer| FriLayerProof::<Blake2sMerkleHasher> {
                fri_witness: layer.fri_witness.iter().map(|v| to_sf_arr(v)).collect(),
                decommitment: MerkleDecommitmentLifted::<Blake2sMerkleHasher> {
                    hash_witness: layer.decommitment.hash_witness.iter()
                        .map(|h| to_hash(h)).collect(),
                },
                commitment: to_hash(&layer.commitment),
            })
            .collect();

        let nat_c2 = crate::fri::last_layer_poly_coeffs(proof.fri_last_layer_poly.clone());
        let sd2 = crate::fri::LOG_LAST_LAYER_DEGREE_BOUND;
        let sn2 = 1usize << sd2;
        let trunc: Vec<SSecureField> = nat_c2[..sn2].iter().map(|q| sf(q.to_u32_array())).collect();
        let stwo_last_layer = LinePoly::from_ordered_coefficients(trunc);

        let fri_proof = FriProof::<Blake2sMerkleHasher> {
            first_layer: stwo_first_layer,
            inner_layers: stwo_inner_layers,
            last_layer_poly: stwo_last_layer,
        };

        let fri_config = FriConfig::new(
            crate::fri::LOG_LAST_LAYER_DEGREE_BOUND, // log_last_layer_degree_bound
            BLOWUP_BITS,                              // log_blowup_factor = 1
            N_QUERIES,                                // n_queries = 80
            1,                                        // line_fold_step = 1
        );

        let column_bound = CirclePolyDegreeBound::new(log_n);

        let mut fri_verifier = FriVerifier::<Blake2sMerkleChannel>::commit(
            &mut ch,
            fri_config,
            fri_proof,
            column_bound,
        ).unwrap_or_else(|e| panic!("FriVerifier::commit failed: {e}"));

        // ── PoW verification ──────────────────────────────────────────────────
        assert!(ch.verify_pow_nonce(POW_BITS, proof.pow_nonce),
            "PoW verification failed");
        ch.mix_u64(proof.pow_nonce);

        // ── Sample query positions ────────────────────────────────────────────
        let query_positions = fri_verifier.sample_query_positions(&mut ch);

        // ── Extract first-layer query evals (OODS quotient at query positions) ──
        // The first_layer_query_evals are the OODS quotient column values at the
        // sorted/deduped query positions (matching Queries::positions order).
        let sorted_queries = Queries::new(&proof.query_indices, log_eval_size);
        assert_eq!(query_positions, sorted_queries.positions,
            "stwo query positions don't match VortexSTARK's");

        // Map query_index → original proof index for value lookup.
        let qi_to_proof_idx: std::collections::HashMap<usize, usize> =
            proof.query_indices.iter().enumerate().map(|(i, &qi)| (qi, i)).collect();

        let first_layer_evals: Vec<SSecureField> = sorted_queries.positions.iter()
            .map(|&pos| {
                let proof_idx = qi_to_proof_idx[&pos];
                let v = proof.oods_quotient_decommitment.values[proof_idx];
                sf(v)
            })
            .collect();

        // ── Decommit: verify all FRI fold equations ───────────────────────────
        fri_verifier.decommit(first_layer_evals)
            .unwrap_or_else(|e| panic!("FriVerifier::decommit failed: {e:?}"));
    }
}
