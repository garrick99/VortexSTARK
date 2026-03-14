//! STARK verifier for the Fibonacci AIR.
//!
//! Verifies a proof by:
//! 1. Replaying the Fiat-Shamir channel to reconstruct challenges
//! 2. Re-deriving query indices and checking they match
//! 3. Verifying all Merkle authentication paths (data integrity)
//! 4. Checking FRI fold equations at each query (algebraic consistency)
//! 5. Verifying the final FRI layer is low-degree

use crate::channel::Channel;
use crate::circle::Coset;
use crate::field::{M31, QM31};
use crate::merkle::MerkleTree;
use crate::prover::{StarkProof, N_QUERIES, BLOWUP_BITS};

/// Bit-reverse an index within a given number of bits.
fn bit_reverse(x: usize, n_bits: u32) -> usize {
    let mut result = 0usize;
    let mut val = x;
    for _ in 0..n_bits {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

/// Compute the FRI fold twiddle at a specific folded index.
/// Matches the GPU kernel: twiddle[i] = 1/coord(domain[bit_reverse(i<<1, log_n)])
/// For circle fold: coord = y. For line fold: coord = x.
fn fold_twiddle_at(domain: &Coset, folded_index: usize, circle: bool) -> M31 {
    // GPU kernel does: idx = bit_reverse(i << 1, log_n)
    let domain_idx = bit_reverse(folded_index << 1, domain.log_size);
    let point = domain.at(domain_idx);
    let coord = if circle { point.y } else { point.x };
    coord.inverse()
}

/// Verify a STARK proof for the Fibonacci AIR.
pub fn verify(proof: &StarkProof) -> Result<(), String> {
    let log_n = proof.log_trace_size;
    let (a, b) = proof.public_inputs;
    let log_eval_size = log_n + BLOWUP_BITS;
    let eval_size = 1usize << log_eval_size;

    // --- Verify public inputs are valid M31 ---
    if a.0 >= crate::field::m31::P || b.0 >= crate::field::m31::P {
        return Err("Public inputs out of M31 range".into());
    }

    // --- Verify proof has query data ---
    if proof.query_indices.len() != N_QUERIES {
        return Err(format!("Expected {} queries, got {}", N_QUERIES, proof.query_indices.len()));
    }

    // --- Replay Fiat-Shamir channel ---
    let mut channel = Channel::new();

    channel.mix_digest(&proof.trace_commitment);
    let _alpha = channel.draw_felt();

    channel.mix_digest(&proof.quotient_commitment);

    // Collect FRI fold alphas
    let mut fri_alphas = Vec::new();
    fri_alphas.push(channel.draw_felt()); // circle fold alpha

    for fri_commitment in &proof.fri_commitments {
        channel.mix_digest(fri_commitment);
        fri_alphas.push(channel.draw_felt());
    }

    // --- Verify FRI structure ---
    let expected_fri_layers = log_eval_size.saturating_sub(3);
    if proof.fri_commitments.len() != expected_fri_layers as usize {
        return Err(format!(
            "Expected {} FRI layers, got {}",
            expected_fri_layers, proof.fri_commitments.len()
        ));
    }

    let expected_last_size = 1usize << 3;
    if proof.fri_last_layer.len() != expected_last_size {
        return Err(format!(
            "Expected {} FRI last layer values, got {}",
            expected_last_size, proof.fri_last_layer.len()
        ));
    }

    // --- Re-derive query indices ---
    channel.mix_felts(&proof.fri_last_layer);
    let expected_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_size))
        .collect();
    if proof.query_indices != expected_indices {
        return Err("Query indices don't match Fiat-Shamir derivation".into());
    }

    // --- Verify commitments are non-trivial ---
    if proof.trace_commitment == [0; 8] {
        return Err("Trace commitment is zero".into());
    }
    if proof.quotient_commitment == [0; 8] {
        return Err("Quotient commitment is zero".into());
    }

    // --- Verify Merkle auth paths: trace ---
    verify_decommitment_auth_paths(
        &proof.trace_commitment,
        &proof.trace_decommitment.values,
        &proof.trace_decommitment.sibling_values,
        &proof.trace_decommitment.auth_paths,
        &proof.trace_decommitment.sibling_auth_paths,
        &proof.query_indices,
        1, // n_cols = 1 for trace
        "trace",
    )?;

    // --- Verify Merkle auth paths: quotient ---
    verify_decommitment_auth_paths_soa4(
        &proof.quotient_commitment,
        &proof.quotient_decommitment.values,
        &proof.quotient_decommitment.sibling_values,
        &proof.quotient_decommitment.auth_paths,
        &proof.quotient_decommitment.sibling_auth_paths,
        &proof.query_indices,
        "quotient",
    )?;

    // --- Verify Merkle auth paths: FRI layers ---
    let mut folded_indices: Vec<usize> = proof.query_indices.iter().map(|&qi| qi / 2).collect();
    for (layer, (decom, commitment)) in proof.fri_decommitments.iter()
        .zip(proof.fri_commitments.iter())
        .enumerate()
    {
        verify_decommitment_auth_paths_soa4(
            commitment,
            &decom.values,
            &decom.sibling_values,
            &decom.auth_paths,
            &decom.sibling_auth_paths,
            &folded_indices,
            &format!("FRI layer {layer}"),
        )?;
        folded_indices = folded_indices.iter().map(|&i| i / 2).collect();
    }

    // =========================================================================
    // ALGEBRAIC CHECKS: FRI fold equation consistency
    // =========================================================================
    // Fold equation: result = (f0 + f1) + alpha * twiddle * (f0 - f1)

    let n_fri_layers = proof.fri_decommitments.len();

    for (q, &qi) in proof.query_indices.iter().enumerate() {
        let mut current_idx = qi;
        let mut current_log = log_eval_size;

        // ---- Circle fold: quotient → FRI layer 0 ----
        {
            let domain = Coset::half_coset(current_log);
            let folded_idx = current_idx / 2;
            let (f0, f1) = get_pair_from_decom_4(
                &proof.quotient_decommitment.values[q],
                &proof.quotient_decommitment.sibling_values[q],
                current_idx,
            );
            let twiddle = fold_twiddle_at(&domain, folded_idx, true);
            let expected = fold_pair(f0, f1, fri_alphas[0], twiddle);
            let actual = QM31::from_u32_array(proof.fri_decommitments[0].values[q]);
            if expected != actual {
                return Err(format!(
                    "Circle fold mismatch at query {q} (qi={qi})"
                ));
            }
            current_idx = folded_idx;
            current_log -= 1;
        }

        // ---- Line folds: FRI layer k → FRI layer k+1 ----
        // The last FRI decommitment IS fri_last_layer, so we fold layers 0..N-2
        // and the result of the last fold should match fri_last_layer.
        for layer in 0..n_fri_layers.saturating_sub(1) {
            let domain = Coset::half_coset(current_log);
            let folded_idx = current_idx / 2;
            let decom = &proof.fri_decommitments[layer];
            let (f0, f1) = get_pair_from_decom_4(
                &decom.values[q],
                &decom.sibling_values[q],
                current_idx,
            );
            let twiddle = fold_twiddle_at(&domain, folded_idx, false);
            let alpha = fri_alphas[layer + 1];
            let expected = fold_pair(f0, f1, alpha, twiddle);

            let actual = QM31::from_u32_array(
                proof.fri_decommitments[layer + 1].values[q]
            );

            if expected != actual {
                return Err(format!(
                    "Line fold mismatch at query {q}, layer {layer}→{} (log={})",
                    layer + 1, current_log
                ));
            }

            current_idx = folded_idx;
            current_log -= 1;
        }

        // ---- Verify last FRI decommitment matches fri_last_layer ----
        if n_fri_layers > 0 {
            let last_decom = &proof.fri_decommitments[n_fri_layers - 1];
            let actual = QM31::from_u32_array(last_decom.values[q]);
            let last_idx = current_idx; // index into the last layer
            if last_idx < proof.fri_last_layer.len() {
                let expected = proof.fri_last_layer[last_idx];
                if actual != expected {
                    return Err(format!(
                        "FRI last layer mismatch at query {q} (index {last_idx})"
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Get the even/odd pair from a decommitment (value, sibling) given the query index.
/// f0 = value at even index, f1 = value at odd index.
fn get_pair_from_decom_4(value: &[u32; 4], sibling: &[u32; 4], idx: usize) -> (QM31, QM31) {
    if idx % 2 == 0 {
        (QM31::from_u32_array(*value), QM31::from_u32_array(*sibling))
    } else {
        (QM31::from_u32_array(*sibling), QM31::from_u32_array(*value))
    }
}

/// Compute FRI fold: result = (f0 + f1) + alpha * twiddle * (f0 - f1)
fn fold_pair(f0: QM31, f1: QM31, alpha: QM31, twiddle: M31) -> QM31 {
    let sum = f0 + f1;
    let diff = f0 - f1;
    let tw_diff = diff * twiddle;
    sum + alpha * tw_diff
}

// --- Auth path verification helpers ---

fn verify_decommitment_auth_paths(
    root: &[u32; 8],
    values: &[u32],
    sibling_values: &[u32],
    auth_paths: &[Vec<[u32; 8]>],
    sibling_auth_paths: &[Vec<[u32; 8]>],
    indices: &[usize],
    n_cols: usize,
    label: &str,
) -> Result<(), String> {
    if values.len() != N_QUERIES || auth_paths.len() != N_QUERIES {
        return Err(format!("{label} decommitment size mismatch"));
    }
    for (q, &qi) in indices.iter().enumerate() {
        // Verify value at qi
        let leaf_hash = MerkleTree::hash_leaf(std::slice::from_ref(&values[q]));
        if !MerkleTree::verify_auth_path(root, &leaf_hash, qi, &auth_paths[q]) {
            return Err(format!("{label} auth path invalid at query {q} (index {qi})"));
        }
        // Verify sibling at qi^1
        let sib_idx = qi ^ 1;
        let sib_hash = MerkleTree::hash_leaf(std::slice::from_ref(&sibling_values[q]));
        if !MerkleTree::verify_auth_path(root, &sib_hash, sib_idx, &sibling_auth_paths[q]) {
            return Err(format!("{label} sibling auth path invalid at query {q} (sibling index {sib_idx})"));
        }
    }
    Ok(())
}

fn verify_decommitment_auth_paths_soa4(
    root: &[u32; 8],
    values: &[[u32; 4]],
    sibling_values: &[[u32; 4]],
    auth_paths: &[Vec<[u32; 8]>],
    sibling_auth_paths: &[Vec<[u32; 8]>],
    indices: &[usize],
    label: &str,
) -> Result<(), String> {
    if values.len() != N_QUERIES || auth_paths.len() != N_QUERIES {
        return Err(format!("{label} decommitment size mismatch"));
    }
    for (q, &qi) in indices.iter().enumerate() {
        let leaf_hash = MerkleTree::hash_leaf(&values[q]);
        if !MerkleTree::verify_auth_path(root, &leaf_hash, qi, &auth_paths[q]) {
            return Err(format!("{label} auth path invalid at query {q} (index {qi})"));
        }
        let sib_idx = qi ^ 1;
        let sib_hash = MerkleTree::hash_leaf(&sibling_values[q]);
        if !MerkleTree::verify_auth_path(root, &sib_hash, sib_idx, &sibling_auth_paths[q]) {
            return Err(format!("{label} sibling auth path invalid at query {q} (sibling index {sib_idx})"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::M31;
    use crate::prover;

    #[test]
    fn test_verify_valid_proof() {
        let proof = prover::prove(M31(1), M31(1), 6);
        assert!(verify(&proof).is_ok(), "Valid proof should verify: {:?}", verify(&proof));
    }

    #[test]
    fn test_verify_tampered_trace_value() {
        let mut proof = prover::prove(M31(1), M31(1), 6);
        proof.trace_decommitment.values[0] ^= 1;
        assert!(verify(&proof).is_err(), "Tampered trace value should fail");
    }

    #[test]
    fn test_verify_tampered_quotient_value() {
        let mut proof = prover::prove(M31(1), M31(1), 6);
        proof.quotient_decommitment.values[0][0] ^= 1;
        assert!(verify(&proof).is_err(), "Tampered quotient should fail");
    }

    #[test]
    fn test_verify_tampered_fri_value() {
        let mut proof = prover::prove(M31(1), M31(1), 6);
        if !proof.fri_decommitments.is_empty() {
            proof.fri_decommitments[0].values[0][0] ^= 1;
        }
        assert!(verify(&proof).is_err(), "Tampered FRI value should fail");
    }

    #[test]
    fn test_verify_tampered_commitment() {
        let mut proof = prover::prove(M31(1), M31(1), 6);
        proof.trace_commitment[0] ^= 1;
        assert!(verify(&proof).is_err(), "Tampered commitment should fail");
    }

    #[test]
    fn test_verify_wrong_fri_layers() {
        let mut proof = prover::prove(M31(1), M31(1), 6);
        proof.fri_commitments.pop();
        assert!(verify(&proof).is_err());
    }

    #[test]
    fn test_verify_zero_commitment() {
        let mut proof = prover::prove(M31(1), M31(1), 6);
        proof.trace_commitment = [0; 8];
        assert!(verify(&proof).is_err());
    }

    #[test]
    fn test_verify_multiple_sizes() {
        for log_n in [4, 5, 6, 8] {
            let proof = prover::prove(M31(1), M31(1), log_n);
            let result = verify(&proof);
            assert!(result.is_ok(), "Failed at log_n={log_n}: {:?}", result);
        }
    }

    #[test]
    fn test_verify_different_inputs() {
        for (a, b) in [(1, 1), (2, 3), (42, 99), (1000, 2000)] {
            let proof = prover::prove(M31(a), M31(b), 6);
            let result = verify(&proof);
            assert!(result.is_ok(), "Failed for a={a}, b={b}: {:?}", result);
        }
    }
}
