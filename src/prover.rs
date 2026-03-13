//! STARK prover for the Fibonacci AIR.
//!
//! Pipeline (all GPU, minimal host transfers):
//! 1. Generate trace on CPU → upload once
//! 2. Interpolate on GPU (values → coefficients)
//! 3. Zero-pad on GPU → evaluate on blowup domain
//! 4. Commit trace via GPU Merkle tree
//! 5. Compute constraint quotient on GPU
//! 6. Commit quotient via GPU Merkle tree
//! 7. FRI protocol (all GPU folds + Merkle commits)
//! 8. Package proof

use crate::air;
use crate::channel::Channel;
use crate::circle::Coset;
use crate::cuda::ffi;
use crate::device::DeviceBuffer;
use crate::field::{M31, QM31};
use crate::fri::{self, SecureColumn};
use crate::merkle::MerkleTree;
use crate::ntt::{self, TwiddleCache};
use std::time::Instant;

/// Blowup factor: evaluation domain is 2^BLOWUP_BITS times the trace domain.
const BLOWUP_BITS: u32 = 1; // blowup factor = 2

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

/// Pre-computed caches that can be reused across multiple proofs of the same size.
pub struct ProverCache {
    pub trace_cache: TwiddleCache,
    pub eval_cache: TwiddleCache,
    pub fri_twiddles: Vec<DeviceBuffer<u32>>,
    pub log_n: u32,
    pub log_eval_size: u32,
}

impl ProverCache {
    /// Build caches for proving traces of size 2^log_n.
    pub fn new(log_n: u32) -> Self {
        let log_eval_size = log_n + BLOWUP_BITS;
        let trace_domain = Coset::half_coset(log_n);
        let eval_domain = Coset::half_coset(log_eval_size);
        let trace_cache = TwiddleCache::new(&trace_domain);
        let eval_cache = TwiddleCache::new(&eval_domain);
        let fri_twiddles = fri::precompute_fri_twiddles(log_eval_size, 3);
        ProverCache { trace_cache, eval_cache, fri_twiddles, log_n, log_eval_size }
    }
}

/// Generate a STARK proof for the Fibonacci sequence.
pub fn prove(a: M31, b: M31, log_n: u32) -> StarkProof {
    prove_inner(a, b, log_n, false)
}

/// Generate a STARK proof with optional timing output.
pub fn prove_timed(a: M31, b: M31, log_n: u32) -> StarkProof {
    prove_inner(a, b, log_n, true)
}

/// Generate a STARK proof using pre-computed caches (amortized setup).
pub fn prove_cached(a: M31, b: M31, cache: &ProverCache) -> StarkProof {
    prove_with_cache(a, b, cache, false)
}

fn prove_inner(a: M31, b: M31, log_n: u32, timed: bool) -> StarkProof {
    assert!(log_n >= 4, "trace too small");
    let n = 1usize << log_n;
    let log_eval_size = log_n + BLOWUP_BITS;
    let eval_size = 1usize << log_eval_size;

    let t_total = Instant::now();

    // --- Precompute twiddle caches (both domains) ---
    let t0 = Instant::now();
    let trace_domain = Coset::half_coset(log_n);
    let eval_domain = Coset::half_coset(log_eval_size);
    let trace_cache = TwiddleCache::new(&trace_domain);
    let eval_cache = TwiddleCache::new(&eval_domain);
    if timed {
        eprintln!("  twiddle_setup: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 1: Generate trace (CPU) → upload once ---
    let t0 = Instant::now();
    let trace_raw = air::fibonacci_trace_raw(a, b, log_n);
    let mut d_trace = DeviceBuffer::from_host(&trace_raw);
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  trace_gen+upload: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 2: Interpolate on GPU ---
    let t0 = Instant::now();
    ntt::interpolate(&mut d_trace, &trace_cache);
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  interpolate: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 3: Zero-pad on GPU → evaluate on blowup domain ---
    let t0 = Instant::now();
    let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
    unsafe {
        ffi::cuda_zero_pad(d_trace.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32);
    }
    drop(d_trace); // free trace coefficients early
    ntt::evaluate(&mut d_eval, &eval_cache);
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  blowup_eval: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 4: Commit trace (GPU Merkle, no clone — use eval directly) ---
    let t0 = Instant::now();
    let trace_ref = [&d_eval];
    let trace_commitment = MerkleTree::commit_root_only(&trace_ref, log_eval_size);
    if timed {
        eprintln!("  trace_commit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 5: Fiat-Shamir + constraint quotient on GPU ---
    let t0 = Instant::now();
    let mut channel = Channel::new();
    channel.mix_digest(&trace_commitment);
    let alpha = channel.draw_felt();
    let alpha_arr = alpha.to_u32_array();

    // Compute quotient entirely on GPU
    let mut q0 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut q1 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut q2 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut q3 = DeviceBuffer::<u32>::alloc(eval_size);
    unsafe {
        ffi::cuda_fibonacci_quotient(
            d_eval.as_ptr(),
            q0.as_mut_ptr(),
            q1.as_mut_ptr(),
            q2.as_mut_ptr(),
            q3.as_mut_ptr(),
            alpha_arr.as_ptr(),
            eval_size as u32,
        );
    }
    drop(d_eval); // free trace evaluation early

    let quotient_col = SecureColumn {
        cols: [q0, q1, q2, q3],
        len: eval_size,
    };
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  quotient_gpu: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 6: Commit quotient (GPU Merkle) ---
    let t0 = Instant::now();
    let quotient_commitment = MerkleTree::commit_root_only(&quotient_col.cols, log_eval_size);
    channel.mix_digest(&quotient_commitment);
    if timed {
        eprintln!("  quotient_commit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 7: FRI (all GPU) ---
    // Precompute all FRI twiddles upfront
    let t0 = Instant::now();
    let fri_twiddles = fri::precompute_fri_twiddles(log_eval_size, 3);
    if timed {
        eprintln!("  fri_twiddle_setup: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    let t0 = Instant::now();
    let mut fri_commitments = Vec::new();
    let mut current_eval = quotient_col;
    let mut current_log_size = log_eval_size;

    // First fold: circle → line (uses twiddles[0])
    let fri_alpha = channel.draw_felt();
    let mut line_eval = SecureColumn::zeros(current_eval.len / 2);
    fri::fold_circle_into_line_with_twiddles(
        &mut line_eval, &current_eval, fri_alpha, &fri_twiddles[0],
    );
    drop(current_eval); // free source data after fold
    current_log_size -= 1;

    let fri_root = MerkleTree::commit_root_only(&line_eval.cols, current_log_size);
    fri_commitments.push(fri_root);
    channel.mix_digest(&fri_root);
    current_eval = line_eval;

    // Subsequent folds: line → line (uses twiddles[1..])
    let mut twid_idx = 1;
    while current_log_size > 3 {
        let fold_alpha = channel.draw_felt();
        let folded = fri::fold_line_with_twiddles(
            &current_eval, fold_alpha, &fri_twiddles[twid_idx],
        );
        drop(current_eval); // free source data after fold
        twid_idx += 1;
        current_log_size -= 1;

        let fri_root = MerkleTree::commit_root_only(&folded.cols, current_log_size);
        fri_commitments.push(fri_root);
        channel.mix_digest(&fri_root);

        current_eval = folded;
    }

    let final_values = current_eval.to_qm31();
    let fri_final = final_values[0];
    if timed {
        eprintln!("  fri_fold+commit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    if timed {
        eprintln!("  TOTAL: {:.1}ms", t_total.elapsed().as_secs_f64() * 1000.0);
    }

    StarkProof {
        trace_commitment,
        quotient_commitment,
        fri_commitments,
        fri_final,
        log_trace_size: log_n,
        public_inputs: (a, b),
    }
}

fn prove_with_cache(a: M31, b: M31, cache: &ProverCache, timed: bool) -> StarkProof {
    let log_n = cache.log_n;
    assert!(log_n >= 4, "trace too small");
    let n = 1usize << log_n;
    let log_eval_size = cache.log_eval_size;
    let eval_size = 1usize << log_eval_size;

    let t_total = Instant::now();

    // --- Step 1: Generate trace (CPU) → upload ---
    let t0 = Instant::now();
    let trace_raw = air::fibonacci_trace_raw(a, b, log_n);
    let mut d_trace = DeviceBuffer::from_host(&trace_raw);
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  trace_gen+upload: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 2: Interpolate ---
    ntt::interpolate(&mut d_trace, &cache.trace_cache);

    // --- Step 3: Zero-pad + evaluate on blowup domain ---
    let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
    unsafe {
        ffi::cuda_zero_pad(d_trace.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32);
    }
    drop(d_trace);
    ntt::evaluate(&mut d_eval, &cache.eval_cache);

    // --- Step 4: Commit trace ---
    let trace_ref = [&d_eval];
    let trace_commitment = MerkleTree::commit_root_only(&trace_ref, log_eval_size);

    // --- Step 5: Quotient ---
    let mut channel = Channel::new();
    channel.mix_digest(&trace_commitment);
    let alpha = channel.draw_felt();
    let alpha_arr = alpha.to_u32_array();

    let mut q0 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut q1 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut q2 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut q3 = DeviceBuffer::<u32>::alloc(eval_size);
    unsafe {
        ffi::cuda_fibonacci_quotient(
            d_eval.as_ptr(),
            q0.as_mut_ptr(), q1.as_mut_ptr(), q2.as_mut_ptr(), q3.as_mut_ptr(),
            alpha_arr.as_ptr(),
            eval_size as u32,
        );
    }
    drop(d_eval);

    let quotient_col = SecureColumn {
        cols: [q0, q1, q2, q3],
        len: eval_size,
    };

    // --- Step 6: Commit quotient ---
    let quotient_commitment = MerkleTree::commit_root_only(&quotient_col.cols, log_eval_size);
    channel.mix_digest(&quotient_commitment);

    // --- Step 7: FRI ---
    let t0 = Instant::now();
    let mut fri_commitments = Vec::new();
    let mut current_eval = quotient_col;
    let mut current_log_size = log_eval_size;

    // Circle fold
    let fri_alpha = channel.draw_felt();
    let mut line_eval = SecureColumn::zeros(current_eval.len / 2);
    fri::fold_circle_into_line_with_twiddles(
        &mut line_eval, &current_eval, fri_alpha, &cache.fri_twiddles[0],
    );
    drop(current_eval);
    current_log_size -= 1;

    let fri_root = MerkleTree::commit_root_only(&line_eval.cols, current_log_size);
    fri_commitments.push(fri_root);
    channel.mix_digest(&fri_root);
    current_eval = line_eval;

    // Line folds
    let mut twid_idx = 1;
    while current_log_size > 3 {
        let fold_alpha = channel.draw_felt();
        let folded = fri::fold_line_with_twiddles(
            &current_eval, fold_alpha, &cache.fri_twiddles[twid_idx],
        );
        drop(current_eval);
        twid_idx += 1;
        current_log_size -= 1;

        let fri_root = MerkleTree::commit_root_only(&folded.cols, current_log_size);
        fri_commitments.push(fri_root);
        channel.mix_digest(&fri_root);
        current_eval = folded;
    }

    let final_values = current_eval.to_qm31();
    let fri_final = final_values[0];
    if timed {
        eprintln!("  prove_body: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
        eprintln!("  TOTAL: {:.1}ms", t_total.elapsed().as_secs_f64() * 1000.0);
    }

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
        let a = M31(1);
        let b = M31(1);
        let proof = prove(a, b, 6);

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
