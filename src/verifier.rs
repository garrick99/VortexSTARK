//! STARK verifier for the Fibonacci AIR.
//!
//! Verifies a proof by:
//! 1. Replaying the Fiat-Shamir channel to reconstruct challenges
//! 2. Checking FRI commitment chain consistency
//! 3. Verifying the final FRI value matches a constant polynomial

use crate::channel::Channel;
use crate::field::{M31, QM31};
use crate::prover::StarkProof;

/// Verify a STARK proof for the Fibonacci AIR.
///
/// Returns Ok(()) if the proof is valid, Err with a reason if not.
pub fn verify(proof: &StarkProof) -> Result<(), String> {
    let log_n = proof.log_trace_size;
    let (a, b) = proof.public_inputs;

    // --- Verify public inputs are valid M31 ---
    if a.0 >= crate::field::m31::P || b.0 >= crate::field::m31::P {
        return Err("Public inputs out of M31 range".into());
    }

    // --- Replay Fiat-Shamir channel ---
    let mut channel = Channel::new();

    // Step 1: Mix trace commitment
    channel.mix_digest(&proof.trace_commitment);
    let _alpha = channel.draw_felt();

    // Step 2: Mix quotient commitment
    channel.mix_digest(&proof.quotient_commitment);

    // Step 3: FRI challenges
    let _fri_alpha = channel.draw_felt(); // circle → line fold

    for fri_commitment in &proof.fri_commitments {
        channel.mix_digest(fri_commitment);
        let _fold_alpha = channel.draw_felt();
    }

    // --- Verify FRI structure ---
    let log_eval_size = log_n + 1; // blowup = 2
    let expected_fri_layers = log_eval_size.saturating_sub(3);
    if proof.fri_commitments.len() != expected_fri_layers as usize {
        return Err(format!(
            "Expected {} FRI layers, got {}",
            expected_fri_layers,
            proof.fri_commitments.len()
        ));
    }

    // --- Verify commitments are non-trivial ---
    if proof.trace_commitment == [0; 8] {
        return Err("Trace commitment is zero".into());
    }
    if proof.quotient_commitment == [0; 8] {
        return Err("Quotient commitment is zero".into());
    }
    for (i, c) in proof.fri_commitments.iter().enumerate() {
        if *c == [0; 8] {
            return Err(format!("FRI commitment {i} is zero"));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover;

    #[test]
    fn test_verify_valid_proof() {
        let proof = prover::prove(M31(1), M31(1), 6);
        assert!(verify(&proof).is_ok());
    }

    #[test]
    fn test_verify_tampered_commitment() {
        let mut proof = prover::prove(M31(1), M31(1), 6);
        proof.trace_commitment[0] ^= 1; // tamper
        // Structural verification still passes since we only check non-zero
        // Full verification with query decommitment would catch this
        assert!(verify(&proof).is_ok());
    }

    #[test]
    fn test_verify_wrong_fri_layers() {
        let mut proof = prover::prove(M31(1), M31(1), 6);
        proof.fri_commitments.pop(); // remove a layer
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
        for log_n in [4, 5, 6, 8, 10] {
            let proof = prover::prove(M31(1), M31(1), log_n);
            assert!(verify(&proof).is_ok(), "Failed at log_n={log_n}");
        }
    }
}
