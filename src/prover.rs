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
use std::sync::Once;
use std::time::Instant;

static POOL_INIT: Once = Once::new();
fn ensure_pool_init() {
    POOL_INIT.call_once(|| ffi::init_memory_pool());
}

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

/// Log size threshold below which FRI iterations run on CPU (avoids kernel launch overhead).
const FRI_CPU_TAIL_LOG: u32 = 10;

/// Pre-computed caches that can be reused across multiple proofs of the same size.
pub struct ProverCache {
    pub trace_cache: TwiddleCache,
    pub eval_cache: TwiddleCache,
    pub fri_twiddles: Vec<DeviceBuffer<u32>>,
    /// Host copies of FRI twiddles for CPU tail iterations.
    pub fri_twiddles_host: Vec<Vec<u32>>,
    pub log_n: u32,
    pub log_eval_size: u32,
    /// Pre-allocated pinned host memory for trace generation (avoids per-proof allocation).
    pinned_trace: *mut u32,
}

unsafe impl Send for ProverCache {}

impl ProverCache {
    /// Build caches for proving traces of size 2^log_n.
    pub fn new(log_n: u32) -> Self {
        ensure_pool_init();
        let log_eval_size = log_n + BLOWUP_BITS;
        let trace_domain = Coset::half_coset(log_n);
        let eval_domain = Coset::half_coset(log_eval_size);
        let trace_cache = TwiddleCache::new(&trace_domain);
        let eval_cache = TwiddleCache::new(&eval_domain);
        let fri_twiddles = fri::precompute_fri_twiddles(log_eval_size, 3);

        // Download host copies of FRI twiddles for CPU tail iterations
        let fri_twiddles_host: Vec<Vec<u32>> = fri_twiddles.iter().map(|d| d.to_host()).collect();

        let bytes = (1usize << log_n) * std::mem::size_of::<u32>();
        let mut pinned_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { ffi::cudaMallocHost(&mut pinned_ptr, bytes) };
        assert!(err == 0, "cudaMallocHost failed: {err}");

        ProverCache {
            trace_cache, eval_cache, fri_twiddles, fri_twiddles_host, log_n, log_eval_size,
            pinned_trace: pinned_ptr as *mut u32,
        }
    }
}

impl Drop for ProverCache {
    fn drop(&mut self) {
        if !self.pinned_trace.is_null() {
            unsafe { ffi::cudaFreeHost(self.pinned_trace as *mut std::ffi::c_void) };
        }
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

/// Generate a STARK proof using pre-computed caches with per-stage timing.
pub fn prove_cached_timed(a: M31, b: M31, cache: &ProverCache) -> StarkProof {
    prove_with_cache(a, b, cache, true)
}

fn prove_inner(a: M31, b: M31, log_n: u32, timed: bool) -> StarkProof {
    ensure_pool_init();
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

    // --- Step 1: Generate trace (CPU) → upload ---
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
    let quotient_commitment = MerkleTree::commit_root_soa4(
        &quotient_col.cols[0], &quotient_col.cols[1],
        &quotient_col.cols[2], &quotient_col.cols[3],
        log_eval_size,
    );
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

    let fri_root = MerkleTree::commit_root_soa4(
        &line_eval.cols[0], &line_eval.cols[1],
        &line_eval.cols[2], &line_eval.cols[3],
        current_log_size,
    );
    fri_commitments.push(fri_root);
    channel.mix_digest(&fri_root);
    current_eval = line_eval;

    // Subsequent folds: line → line (GPU path for large sizes)
    let mut twid_idx = 1;
    while current_log_size > FRI_CPU_TAIL_LOG.max(3) {
        let fold_alpha = channel.draw_felt();
        let folded = fri::fold_line_with_twiddles(
            &current_eval, fold_alpha, &fri_twiddles[twid_idx],
        );
        drop(current_eval);
        twid_idx += 1;
        current_log_size -= 1;

        let fri_root = MerkleTree::commit_root_soa4(
            &folded.cols[0], &folded.cols[1],
            &folded.cols[2], &folded.cols[3],
            current_log_size,
        );
        fri_commitments.push(fri_root);
        channel.mix_digest(&fri_root);
        current_eval = folded;
    }

    // CPU tail: download only the small tail twiddles (avoids downloading all twiddles)
    let mut cpu_eval = current_eval.to_qm31();
    drop(current_eval);

    while current_log_size > 3 {
        let fold_alpha = channel.draw_felt();
        let host_tw = fri_twiddles[twid_idx].to_host();
        cpu_eval = fri::fold_line_cpu(&cpu_eval, fold_alpha, &host_tw);
        twid_idx += 1;
        current_log_size -= 1;

        let fri_root = fri::merkle_root_cpu(&cpu_eval);
        fri_commitments.push(fri_root);
        channel.mix_digest(&fri_root);
    }

    let fri_final = cpu_eval[0];
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
    ensure_pool_init();
    let log_n = cache.log_n;
    assert!(log_n >= 4, "trace too small");
    let n = 1usize << log_n;
    let log_eval_size = cache.log_eval_size;
    let eval_size = 1usize << log_eval_size;

    let t_total = Instant::now();

    // --- Step 1: Generate trace (parallel) into pinned memory → fast upload ---
    let t0 = Instant::now();
    unsafe { air::fibonacci_trace_parallel(a, b, log_n, cache.pinned_trace) };
    if timed {
        eprintln!("  trace_gen (cpu): {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }
    let t0b = Instant::now();
    let mut d_trace = unsafe { DeviceBuffer::from_pinned(cache.pinned_trace as *const u32, n) };
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  trace_upload: {:.1}ms", t0b.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 2: Interpolate ---
    let t0 = Instant::now();
    ntt::interpolate(&mut d_trace, &cache.trace_cache);
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  interpolate: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 3: Zero-pad + evaluate on blowup domain ---
    let t0 = Instant::now();
    let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
    unsafe {
        ffi::cuda_zero_pad(d_trace.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32);
    }
    drop(d_trace);
    ntt::evaluate(&mut d_eval, &cache.eval_cache);
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  blowup_eval: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 4: Commit trace ---
    let t0 = Instant::now();
    let trace_ref = [&d_eval];
    let trace_commitment = MerkleTree::commit_root_only(&trace_ref, log_eval_size);
    if timed {
        eprintln!("  trace_commit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 5: Quotient ---
    let t0 = Instant::now();
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
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  quotient_gpu: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 6: Commit quotient ---
    let t0 = Instant::now();
    let quotient_commitment = MerkleTree::commit_root_soa4(
        &quotient_col.cols[0], &quotient_col.cols[1],
        &quotient_col.cols[2], &quotient_col.cols[3],
        log_eval_size,
    );
    channel.mix_digest(&quotient_commitment);
    if timed {
        eprintln!("  quotient_commit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

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

    let fri_root = MerkleTree::commit_root_soa4(
        &line_eval.cols[0], &line_eval.cols[1],
        &line_eval.cols[2], &line_eval.cols[3],
        current_log_size,
    );
    fri_commitments.push(fri_root);
    channel.mix_digest(&fri_root);
    current_eval = line_eval;
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("    fri[0] circle fold+commit (log={}): {:.3}ms", current_log_size, t0.elapsed().as_secs_f64() * 1000.0);
    }

    // Line folds (GPU path for large sizes)
    let mut twid_idx = 1;
    while current_log_size > FRI_CPU_TAIL_LOG.max(3) {
        let ti = Instant::now();
        let fold_alpha = channel.draw_felt();
        let folded = fri::fold_line_with_twiddles(
            &current_eval, fold_alpha, &cache.fri_twiddles[twid_idx],
        );
        drop(current_eval);
        twid_idx += 1;
        current_log_size -= 1;

        let fri_root = MerkleTree::commit_root_soa4(
            &folded.cols[0], &folded.cols[1],
            &folded.cols[2], &folded.cols[3],
            current_log_size,
        );
        fri_commitments.push(fri_root);
        channel.mix_digest(&fri_root);
        current_eval = folded;
        if timed {
            eprintln!("    fri[{}] gpu fold+commit (log={}): {:.3}ms", twid_idx, current_log_size, ti.elapsed().as_secs_f64() * 1000.0);
        }
    }

    // CPU tail: avoid kernel launch overhead for small data
    let mut cpu_eval = if current_log_size > 3 {
        let ti = Instant::now();
        let v = current_eval.to_qm31();
        drop(current_eval);
        if timed {
            eprintln!("    fri download for cpu tail: {:.3}ms", ti.elapsed().as_secs_f64() * 1000.0);
        }
        v
    } else {
        let v = current_eval.to_qm31();
        drop(current_eval);
        v
    };

    while current_log_size > 3 {
        let ti = Instant::now();
        let fold_alpha = channel.draw_felt();
        cpu_eval = fri::fold_line_cpu(
            &cpu_eval, fold_alpha, &cache.fri_twiddles_host[twid_idx],
        );
        twid_idx += 1;
        current_log_size -= 1;

        let fri_root = fri::merkle_root_cpu(&cpu_eval);
        fri_commitments.push(fri_root);
        channel.mix_digest(&fri_root);
        if timed {
            eprintln!("    fri[{}] cpu fold+commit (log={}): {:.3}ms", twid_idx, current_log_size, ti.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let fri_final = cpu_eval[0];
    if timed {
        eprintln!("  fri_fold+commit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
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
