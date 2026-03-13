//! STARK prover for the Fibonacci AIR.
//!
//! Pipeline:
//! 1. Generate trace → upload to GPU
//! 2. NTT to get trace polynomial coefficients
//! 3. Evaluate on larger domain (blowup)
//! 4. Commit trace evaluations via Merkle tree
//! 5. Compute constraint quotient polynomials
//! 6. Commit quotient evaluations
//! 7. FRI protocol to prove low-degree
//! 8. Package proof

use crate::air;
use crate::channel::Channel;
use crate::circle::Coset;
use crate::device::DeviceBuffer;
use crate::field::{M31, QM31};
use crate::fri::{self, SecureColumn};
use crate::merkle::MerkleTree;
use crate::ntt::{self, TwiddleCache};

/// Blowup factor: evaluation domain is 2^BLOWUP_BITS times the trace domain.
const BLOWUP_BITS: u32 = 1; // blowup factor = 2

/// Number of FRI queries for security.
const N_QUERIES: usize = 20;

/// A STARK proof for the Fibonacci AIR.
pub struct StarkProof {
    /// Merkle root of the trace commitment.
    pub trace_commitment: [u32; 8],
    /// Merkle root of the quotient commitment.
    pub quotient_commitment: [u32; 8],
    /// FRI layer commitments.
    pub fri_commitments: Vec<[u32; 8]>,
    /// Final FRI value (constant polynomial).
    pub fri_final: QM31,
    /// Log size of the trace.
    pub log_trace_size: u32,
    /// Public inputs: (a, b) = first two Fibonacci values.
    pub public_inputs: (M31, M31),
}

/// Generate a STARK proof for the Fibonacci sequence.
///
/// Proves that the prover knows a Fibonacci trace of length 2^log_n
/// starting with (a, b).
pub fn prove(a: M31, b: M31, log_n: u32) -> StarkProof {
    assert!(log_n >= 4, "trace too small");
    let n = 1usize << log_n;
    let log_eval_size = log_n + BLOWUP_BITS;
    let eval_size = 1usize << log_eval_size;

    // --- Step 1: Generate trace ---
    let trace = air::fibonacci_trace(a, b, log_n);

    // --- Step 2: Interpolate (values → coefficients) ---
    let trace_domain = Coset::half_coset(log_n);
    let trace_cache = TwiddleCache::new(&trace_domain);

    let trace_raw: Vec<u32> = trace.iter().map(|v| v.0).collect();
    let mut d_trace = DeviceBuffer::from_host(&trace_raw);

    // Values are on the trace domain; interpolate to get coefficients
    ntt::interpolate(&mut d_trace, &trace_cache);
    // d_trace now holds coefficients

    // --- Step 3: Evaluate on larger domain (blowup) ---
    // Pad coefficients to eval_size by zero-extending
    let coeffs = d_trace.to_host();
    let mut padded = vec![0u32; eval_size];
    padded[..n].copy_from_slice(&coeffs);
    let mut d_eval = DeviceBuffer::from_host(&padded);

    let eval_domain = Coset::half_coset(log_eval_size);
    let eval_cache = TwiddleCache::new(&eval_domain);

    // Forward NTT on larger domain: coefficients → evaluation values
    ntt::evaluate(&mut d_eval, &eval_cache);

    // --- Step 4: Commit trace ---
    let trace_tree = MerkleTree::commit(&[d_eval], log_eval_size);
    let trace_commitment = trace_tree.root();

    // --- Step 5: Fiat-Shamir ---
    let mut channel = Channel::new();
    channel.mix_digest(&trace_commitment);

    // Draw random alpha for constraint combination
    let alpha = channel.draw_felt();

    // --- Step 6: Compute constraint quotient ---
    // For Fibonacci: quotient = transition_constraint / vanishing_polynomial
    // The transition constraint is: trace(x*g^2) - trace(x*g) - trace(x) = 0
    // evaluated over the trace domain.
    //
    // For the MVP, we compute this on CPU and upload.
    // In production, this would be a GPU kernel.

    let eval_values = {
        let mut d_copy = DeviceBuffer::from_host(&padded);
        ntt::evaluate(&mut d_copy, &eval_cache);
        d_copy.to_host()
    };

    // Compute quotient values on the evaluation domain
    // For each point in eval domain, the constraint polynomial evaluates to:
    //   C(x) = trace(x*g^2) - trace(x*g) - trace(x)
    // The quotient is C(x) / V(x) where V(x) vanishes on the constraint domain.
    //
    // Simplified for MVP: compute quotient as random linear combination
    // of constraint evaluations, producing a QM31 column.
    let quotient_values: Vec<QM31> = (0..eval_size)
        .map(|i| {
            let t_i = M31(eval_values[i]);
            let t_i1 = M31(eval_values[(i + 1) % eval_size]);
            let t_i2 = M31(eval_values[(i + 2) % eval_size]);
            let constraint = air::eval_transition(t_i, t_i1, t_i2);
            // Wrap into QM31 and apply alpha
            alpha * QM31::from_m31_array([constraint, M31::ZERO, M31::ZERO, M31::ZERO])
        })
        .collect();

    // Commit quotient as SecureColumn
    let quotient_col = SecureColumn::from_qm31(&quotient_values);
    let quotient_tree = MerkleTree::commit(&quotient_col.cols, log_eval_size);
    let quotient_commitment = quotient_tree.root();

    channel.mix_digest(&quotient_commitment);

    // --- Step 7: FRI ---
    // The FRI protocol proves that the quotient polynomial has low degree.
    // We fold repeatedly using random challenges from the channel.

    let mut fri_commitments = Vec::new();
    let mut current_eval = quotient_col;
    let mut current_log_size = log_eval_size;
    let mut current_domain = eval_domain;

    // First fold: circle → line
    let fri_alpha = channel.draw_felt();
    let mut line_eval = SecureColumn::zeros(current_eval.len / 2);
    fri::fold_circle_into_line(&mut line_eval, &current_eval, fri_alpha, &current_domain);
    current_log_size -= 1;

    // Commit first FRI layer
    let fri_tree = MerkleTree::commit(&line_eval.cols, current_log_size);
    let fri_root = fri_tree.root();
    fri_commitments.push(fri_root);
    channel.mix_digest(&fri_root);

    current_eval = line_eval;

    // Subsequent folds: line → line
    while current_log_size > 3 {
        let fold_alpha = channel.draw_felt();
        current_domain = Coset::half_coset(current_log_size);
        let folded = fri::fold_line(&current_eval, fold_alpha, &current_domain);
        current_log_size -= 1;

        let fri_tree = MerkleTree::commit(&folded.cols, current_log_size);
        let fri_root = fri_tree.root();
        fri_commitments.push(fri_root);
        channel.mix_digest(&fri_root);

        current_eval = folded;
    }

    // Final FRI value: download the remaining small evaluation
    let final_values = current_eval.to_qm31();
    let fri_final = final_values[0]; // constant polynomial

    StarkProof {
        trace_commitment,
        quotient_commitment,
        fri_commitments,
        fri_final,
        log_trace_size: log_n,
        public_inputs: (a, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prove_runs() {
        // Just verify the prover runs end-to-end without panicking
        let a = M31(1);
        let b = M31(1);
        let proof = prove(a, b, 6); // small trace for testing

        assert_ne!(proof.trace_commitment, [0; 8]);
        assert_ne!(proof.quotient_commitment, [0; 8]);
        assert!(!proof.fri_commitments.is_empty());
        assert_eq!(proof.log_trace_size, 6);
        assert_eq!(proof.public_inputs, (a, b));
    }

    #[test]
    fn test_prove_deterministic() {
        let a = M31(1);
        let b = M31(1);
        let proof1 = prove(a, b, 5);
        let proof2 = prove(a, b, 5);

        assert_eq!(proof1.trace_commitment, proof2.trace_commitment);
        assert_eq!(proof1.quotient_commitment, proof2.quotient_commitment);
        assert_eq!(proof1.fri_commitments, proof2.fri_commitments);
        assert_eq!(proof1.fri_final, proof2.fri_final);
    }

    #[test]
    fn test_prove_different_inputs() {
        let proof1 = prove(M31(1), M31(1), 5);
        let proof2 = prove(M31(2), M31(3), 5);

        assert_ne!(proof1.trace_commitment, proof2.trace_commitment);
    }
}
