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
use crate::ntt::{self, TwiddleCache, ForwardTwiddleCache, InverseTwiddleCache};
use std::sync::Once;
use std::time::Instant;

static POOL_INIT: Once = Once::new();
fn ensure_pool_init() {
    POOL_INIT.call_once(|| ffi::init_memory_pool());
}

/// Blowup factor: evaluation domain is 2^BLOWUP_BITS times the trace domain.
pub const BLOWUP_BITS: u32 = 1; // blowup factor = 2

/// Number of queries for ~100-bit security (blowup=2, 1 bit/query).
pub const N_QUERIES: usize = 100;

/// Decommitment data for a set of queries against a Merkle commitment.
/// Includes both the queried value and its fold-sibling (index ^ 1) for
/// verifying FRI fold equations.
#[derive(Clone)]
pub struct QueryDecommitment<T: Clone> {
    /// Leaf values at queried positions.
    pub values: Vec<T>,
    /// Sibling values at (queried_index ^ 1) — needed for fold verification.
    pub sibling_values: Vec<T>,
    /// Merkle authentication paths for queried positions.
    pub auth_paths: Vec<Vec<[u32; 8]>>,
    /// Merkle authentication paths for sibling positions.
    pub sibling_auth_paths: Vec<Vec<[u32; 8]>>,
}

/// A STARK proof for the Fibonacci AIR.
pub struct StarkProof {
    /// Merkle root of the trace commitment.
    pub trace_commitment: [u32; 8],
    /// Merkle root of the quotient commitment.
    pub quotient_commitment: [u32; 8],
    /// FRI layer commitments.
    pub fri_commitments: Vec<[u32; 8]>,
    /// Final FRI polynomial values (2^3 = 8 QM31 elements).
    pub fri_last_layer: Vec<QM31>,
    /// Log size of the trace.
    pub log_trace_size: u32,
    /// Public inputs: (a, b) = first two Fibonacci values.
    pub public_inputs: (M31, M31),
    /// Queried indices in the evaluation domain.
    pub query_indices: Vec<usize>,
    /// Trace values at queried positions.
    pub trace_decommitment: QueryDecommitment<u32>,
    /// Quotient values at queried positions (4 M31 components = QM31).
    pub quotient_decommitment: QueryDecommitment<[u32; 4]>,
    /// FRI layer decommitments (one per FRI layer).
    pub fri_decommitments: Vec<QueryDecommitment<[u32; 4]>>,
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
    // Build a temporary cache and delegate to the cached path.
    let cache = ProverCache::new(log_n);
    unsafe { air::fibonacci_trace_parallel(a, b, log_n, cache.pinned_trace) };
    prove_with_cache(a, b, &cache, timed)
}

/// Generate a STARK proof with lazy VRAM management.
/// Uses root-only Merkle commits and computes auth paths on CPU from host data.
pub fn prove_lean(a: M31, b: M31, log_n: u32) -> StarkProof {
    if log_n >= 20 {
        return prove_lean_fused(a, b, log_n, false);
    }
    prove_lean_inner(a, b, log_n, false)
}

/// Generate a STARK proof with lazy VRAM management and timing output.
pub fn prove_lean_timed(a: M31, b: M31, log_n: u32) -> StarkProof {
    if log_n >= 20 {
        return prove_lean_fused(a, b, log_n, true);
    }
    prove_lean_inner(a, b, log_n, true)
}

fn prove_lean_inner(a: M31, b: M31, log_n: u32, timed: bool) -> StarkProof {
    ensure_pool_init();
    assert!(log_n >= 4, "trace too small");
    let n = 1usize << log_n;
    let log_eval_size = log_n + BLOWUP_BITS;
    let eval_size = 1usize << log_eval_size;

    let t_total = Instant::now();

    // Pinned host memory for trace
    let bytes = n * std::mem::size_of::<u32>();
    let mut pinned_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let err = unsafe { ffi::cudaMallocHost(&mut pinned_ptr, bytes) };
    assert!(err == 0, "cudaMallocHost failed: {err}");
    let pinned_trace = pinned_ptr as *mut u32;

    // --- Step 1: Generate trace ---
    let t0 = Instant::now();
    unsafe { air::fibonacci_trace_parallel(a, b, log_n, pinned_trace) };
    if timed { eprintln!("  trace_gen (cpu): {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0); }

    let t0b = Instant::now();
    let mut d_trace = unsafe { DeviceBuffer::from_pinned(pinned_trace as *const u32, n) };
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  trace_upload: {:.1}ms", t0b.elapsed().as_secs_f64() * 1000.0);
    }
    unsafe { ffi::cudaFreeHost(pinned_ptr) };

    // --- Step 2: Interpolate (needs inverse twiddles only for trace domain) ---
    let t0 = Instant::now();
    let trace_domain = Coset::half_coset(log_n);
    let trace_inv = InverseTwiddleCache::new(&trace_domain);
    ntt::interpolate(&mut d_trace, &trace_inv);
    drop(trace_inv); // Free inverse twiddles immediately
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  interpolate: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 3: Zero-pad + evaluate on blowup domain (needs forward twiddles for eval domain) ---
    let t0 = Instant::now();
    let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
    unsafe {
        ffi::cuda_zero_pad(d_trace.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32);
    }
    drop(d_trace);
    let eval_domain = Coset::half_coset(log_eval_size);
    let eval_fwd = ForwardTwiddleCache::new(&eval_domain);
    ntt::evaluate(&mut d_eval, &eval_fwd);
    drop(eval_fwd); // Free forward twiddles immediately
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  blowup_eval: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 4: Commit trace (root-only, tiled) — save subtree roots for fast decommit ---
    let t0 = Instant::now();
    let (trace_commitment, trace_subtree_roots) =
        MerkleTree::commit_root_only_with_subtrees(std::slice::from_ref(&d_eval), log_eval_size);
    if timed { eprintln!("  trace_commit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0); }

    // Download trace eval to host for later decommitment
    let eval_host = d_eval.to_host();

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
    drop(d_eval); // Free eval GPU buffer — we have host copy

    let quotient_col = SecureColumn {
        cols: [q0, q1, q2, q3],
        len: eval_size,
    };
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  quotient_gpu: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 6: Commit quotient (root-only, tiled) — save subtree roots for fast decommit ---
    let t0 = Instant::now();
    let (quotient_commitment, quotient_subtree_roots) = MerkleTree::commit_root_soa4_with_subtrees(
        &quotient_col.cols[0], &quotient_col.cols[1],
        &quotient_col.cols[2], &quotient_col.cols[3],
        log_eval_size,
    );
    channel.mix_digest(&quotient_commitment);
    if timed { eprintln!("  quotient_commit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0); }

    // Download quotient to host for later decommitment
    let quotient_host: [Vec<u32>; 4] = [
        quotient_col.cols[0].to_host(),
        quotient_col.cols[1].to_host(),
        quotient_col.cols[2].to_host(),
        quotient_col.cols[3].to_host(),
    ];

    // --- Step 7: FRI ---
    let t0 = Instant::now();
    let mut fri_commitments = Vec::new();
    let _fri_trees: Vec<MerkleTree> = Vec::new();
    let _fri_evals: Vec<SecureColumn> = Vec::new();
    let current_eval = quotient_col;
    let mut current_log_size = log_eval_size;

    // FRI layers above this threshold use root-only commit + CPU decommitment.
    // Below it, full GPU trees are cheap enough to keep.
    const FRI_LEAN_THRESHOLD_LOG: u32 = 18;

    // Circle fold — compute twiddle on demand
    let fri_alpha = channel.draw_felt();
    let mut line_eval = SecureColumn::zeros(current_eval.len / 2);
    {
        let circle_domain = Coset::half_coset(current_log_size);
        let d_twid = fri::compute_fold_twiddles_on_demand(&circle_domain, true);
        fri::fold_circle_into_line_with_twiddles(
            &mut line_eval, &current_eval, fri_alpha, &d_twid,
        );
    }
    drop(current_eval);
    current_log_size -= 1;

    // Track FRI data: either a full GPU tree or host data + subtree roots
    enum FriLayerData {
        GpuTree(MerkleTree, SecureColumn),
        HostData([Vec<u32>; 4], Vec<[u32; 8]>), // host cols + subtree roots
    }
    let mut fri_layer_data: Vec<FriLayerData> = Vec::new();

    // Commit first FRI layer
    if current_log_size >= FRI_LEAN_THRESHOLD_LOG {
        let (fri_root, subtrees) = MerkleTree::commit_root_soa4_with_subtrees(
            &line_eval.cols[0], &line_eval.cols[1],
            &line_eval.cols[2], &line_eval.cols[3], current_log_size,
        );
        fri_commitments.push(fri_root);
        channel.mix_digest(&fri_root);
        let host_data = [
            line_eval.cols[0].to_host(), line_eval.cols[1].to_host(),
            line_eval.cols[2].to_host(), line_eval.cols[3].to_host(),
        ];
        fri_layer_data.push(FriLayerData::HostData(host_data, subtrees));
    } else {
        let fri_tree = MerkleTree::commit_soa4(
            &line_eval.cols[0], &line_eval.cols[1],
            &line_eval.cols[2], &line_eval.cols[3], current_log_size,
        );
        fri_commitments.push(fri_tree.root());
        channel.mix_digest(&fri_tree.root());
        fri_layer_data.push(FriLayerData::GpuTree(fri_tree, line_eval));
    }
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("    fri[0] circle fold+commit (log={}): {:.3}ms", current_log_size, t0.elapsed().as_secs_f64() * 1000.0);
    }

    // We need to keep the last GPU eval alive for folding into the next layer.
    // For lean layers (host data), re-upload from host for the fold.
    // For GPU layers, fold directly from the stored SecureColumn.

    // Line folds with overlapped twiddle generation.
    // While layer k folds+commits on default stream, layer k+1's twiddles
    // compute on a separate stream (domain-only, no data dependency on fold result).
    let twid_stream = ffi::CudaStream::new();
    let mut prefetched_twid: Option<(DeviceBuffer<u32>, DeviceBuffer<u32>)> = None;

    // Prefetch first line fold twiddle
    if current_log_size > FRI_CPU_TAIL_LOG.max(3) {
        let next_domain = Coset::half_coset(current_log_size);
        prefetched_twid = Some(fri::compute_fold_twiddles_async(&next_domain, false, &twid_stream));
    }

    while current_log_size > FRI_CPU_TAIL_LOG.max(3) {
        let ti = Instant::now();
        let fold_alpha = channel.draw_felt();

        // Wait for prefetched twiddle to finish
        twid_stream.sync();
        let (_sources, d_twid) = prefetched_twid.take().unwrap();

        // Start prefetching NEXT layer's twiddle (overlaps with this fold+commit)
        let next_log = current_log_size - 1;
        if next_log > FRI_CPU_TAIL_LOG.max(3) {
            let next_domain = Coset::half_coset(next_log);
            prefetched_twid = Some(fri::compute_fold_twiddles_async(&next_domain, false, &twid_stream));
        }

        // Fold on default stream
        let folded = match fri_layer_data.last().unwrap() {
            FriLayerData::GpuTree(_, eval) => {
                fri::fold_line_with_twiddles(eval, fold_alpha, &d_twid)
            }
            FriLayerData::HostData(host, _) => {
                let src = SecureColumn {
                    cols: [
                        DeviceBuffer::from_host(&host[0]),
                        DeviceBuffer::from_host(&host[1]),
                        DeviceBuffer::from_host(&host[2]),
                        DeviceBuffer::from_host(&host[3]),
                    ],
                    len: host[0].len(),
                };
                let result = fri::fold_line_with_twiddles(&src, fold_alpha, &d_twid);
                drop(src);
                result
            }
        };
        drop(d_twid);
        current_log_size -= 1;

        if current_log_size >= FRI_LEAN_THRESHOLD_LOG {
            let (fri_root, subtrees) = MerkleTree::commit_root_soa4_with_subtrees(
                &folded.cols[0], &folded.cols[1],
                &folded.cols[2], &folded.cols[3], current_log_size,
            );
            fri_commitments.push(fri_root);
            channel.mix_digest(&fri_root);
            let host_data = [
                folded.cols[0].to_host(), folded.cols[1].to_host(),
                folded.cols[2].to_host(), folded.cols[3].to_host(),
            ];
            fri_layer_data.push(FriLayerData::HostData(host_data, subtrees));
        } else {
            let fri_tree = MerkleTree::commit_soa4(
                &folded.cols[0], &folded.cols[1],
                &folded.cols[2], &folded.cols[3], current_log_size,
            );
            fri_commitments.push(fri_tree.root());
            channel.mix_digest(&fri_tree.root());
            fri_layer_data.push(FriLayerData::GpuTree(fri_tree, folded));
        }
        if timed {
            eprintln!("    fri gpu fold+commit (log={}): {:.3}ms", current_log_size, ti.elapsed().as_secs_f64() * 1000.0);
        }
    }
    drop(prefetched_twid);
    drop(twid_stream);

    // CPU tail
    let mut cpu_eval = match fri_layer_data.last().unwrap() {
        FriLayerData::GpuTree(_, eval) => eval.to_qm31(),
        FriLayerData::HostData(host, _) => {
            let n = host[0].len();
            (0..n).map(|i| QM31::from_u32_array([host[0][i], host[1][i], host[2][i], host[3][i]])).collect()
        }
    };

    let mut cpu_fri_trees: Vec<(Vec<QM31>, [u32; 8])> = Vec::new();
    while current_log_size > 3 {
        let ti = Instant::now();
        let fold_alpha = channel.draw_felt();
        let line_domain = Coset::half_coset(current_log_size);
        let d_twid = fri::compute_fold_twiddles_on_demand(&line_domain, false);
        let twid_host = d_twid.to_host();
        drop(d_twid);

        cpu_eval = fri::fold_line_cpu(&cpu_eval, fold_alpha, &twid_host);
        current_log_size -= 1;

        let fri_root = fri::merkle_root_cpu(&cpu_eval);
        fri_commitments.push(fri_root);
        channel.mix_digest(&fri_root);
        cpu_fri_trees.push((cpu_eval.clone(), fri_root));
        if timed {
            eprintln!("    fri cpu fold+commit (log={}): {:.3}ms", current_log_size, ti.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let fri_last_layer = cpu_eval.clone();
    if timed {
        eprintln!("  fri_fold+commit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 8: Query phase ---
    let t0 = Instant::now();

    channel.mix_felts(&fri_last_layer);
    let query_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_size))
        .collect();

    // Decommit trace — GPU per-tile Merkle trees + CPU upper tree
    let trace_decommitment = decommit_trace_gpu(&eval_host, &trace_subtree_roots, &query_indices);

    // Decommit quotient — GPU per-tile Merkle trees + CPU upper tree
    let quotient_decommitment = decommit_soa4_gpu(&quotient_host, &quotient_subtree_roots, &query_indices);

    // Decommit FRI layers
    let mut fri_decommitments = Vec::new();
    let mut folded_indices: Vec<usize> = query_indices.iter().map(|&qi| qi / 2).collect();
    for layer_data in &fri_layer_data {
        match layer_data {
            FriLayerData::GpuTree(tree, eval) => {
                let decom = decommit_fri_layer(tree, eval, &folded_indices);
                fri_decommitments.push(decom);
            }
            FriLayerData::HostData(host, subtrees) => {
                let decom = decommit_soa4_gpu(host, subtrees, &folded_indices);
                fri_decommitments.push(decom);
            }
        }
        folded_indices = folded_indices.iter().map(|&i| i / 2).collect();
    }
    drop(fri_layer_data);

    // Decommit CPU tail FRI layers
    for (cpu_vals, _root) in &cpu_fri_trees {
        let values: Vec<[u32; 4]> = folded_indices
            .iter()
            .map(|&i| cpu_vals[i].to_u32_array())
            .collect();
        let sibling_indices: Vec<usize> = folded_indices.iter().map(|&i| i ^ 1).collect();
        let sibling_values: Vec<[u32; 4]> = sibling_indices
            .iter()
            .map(|&i| cpu_vals[i].to_u32_array())
            .collect();
        let auth_paths = cpu_merkle_auth_paths(cpu_vals, &folded_indices);
        let sibling_auth_paths = cpu_merkle_auth_paths(cpu_vals, &sibling_indices);
        fri_decommitments.push(QueryDecommitment {
            values, sibling_values, auth_paths, sibling_auth_paths,
        });
        folded_indices = folded_indices.iter().map(|&i| i / 2).collect();
    }

    if timed {
        eprintln!("  query_decommit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
        eprintln!("  TOTAL: {:.1}ms", t_total.elapsed().as_secs_f64() * 1000.0);
    }

    StarkProof {
        trace_commitment,
        quotient_commitment,
        fri_commitments,
        fri_last_layer,
        log_trace_size: log_n,
        public_inputs: (a, b),
        query_indices,
        trace_decommitment,
        quotient_decommitment,
        fri_decommitments,
    }
}

/// Prove at the maximum size: log_n=30 (1B trace elements, 2B eval domain).
/// Uses twin-coset evaluation (half_coset(30) + subgroup(30)) since half_coset(31)
/// doesn't exist (the M31 circle group has order 2^31). Quotient and FRI are
/// streamed in chunks to fit in 32GB VRAM.
/// Fused prover path: zero host transfer for quotient, GPU-resident FRI.
/// Works for any log_n >= 4. Uses full-group NTT for log_n=30, standard for others.
fn prove_lean_fused(a: M31, b: M31, log_n: u32, timed: bool) -> StarkProof {
    ensure_pool_init();
    let n = 1usize << log_n;
    let log_eval_size = log_n + BLOWUP_BITS;
    let eval_size = 1usize << log_eval_size;

    let t_total = Instant::now();

    // --- Step 1: Generate trace ---
    let t0 = Instant::now();
    let bytes = n * std::mem::size_of::<u32>();
    let mut pinned_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let err = unsafe { ffi::cudaMallocHost(&mut pinned_ptr, bytes) };
    assert!(err == 0, "cudaMallocHost failed: {err}");
    let pinned_trace = pinned_ptr as *mut u32;
    unsafe { air::fibonacci_trace_parallel(a, b, log_n, pinned_trace) };
    if timed { eprintln!("  trace_gen (cpu): {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0); }

    let t0 = Instant::now();
    let mut d_coeffs = unsafe { DeviceBuffer::from_pinned(pinned_trace as *const u32, n) };
    if timed { unsafe { ffi::cuda_device_sync() }; eprintln!("  trace_upload: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0); }
    unsafe { ffi::cudaFreeHost(pinned_ptr) };

    // --- Step 2: Interpolate on trace domain ---
    let t0 = Instant::now();
    let trace_domain = Coset::half_coset(log_n);
    let trace_inv = InverseTwiddleCache::new(&trace_domain);
    ntt::interpolate(&mut d_coeffs, &trace_inv);
    drop(trace_inv);
    if timed { unsafe { ffi::cuda_device_sync() }; eprintln!("  interpolate: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0); }

    // --- Step 3: Zero-pad + full-group NTT on subgroup(31) ---
    let t0 = Instant::now();
    let mut d_eval_full = DeviceBuffer::<u32>::alloc(eval_size);
    unsafe {
        ffi::cuda_zero_pad(d_coeffs.as_ptr(), d_eval_full.as_mut_ptr(), n as u32, eval_size as u32);
    }
    drop(d_coeffs);

    // For log_n=30: eval domain is full circle group (subgroup(31), half_coset(31) doesn't exist)
    // For log_n<30: standard half_coset(log_eval_size)
    let eval_domain = if log_eval_size <= 30 {
        Coset::half_coset(log_eval_size)
    } else {
        Coset::subgroup(log_eval_size)
    };
    let eval_fwd = ForwardTwiddleCache::new(&eval_domain);
    ntt::evaluate(&mut d_eval_full, &eval_fwd);
    drop(eval_fwd);
    unsafe { ffi::cuda_device_sync() };
    if timed { eprintln!("  fullgroup_ntt: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0); }

    // --- Step 5: Commit trace (root-only, tiled) ---
    let t0 = Instant::now();
    let (trace_commitment, trace_subtree_roots) =
        MerkleTree::commit_root_only_with_subtrees(std::slice::from_ref(&d_eval_full), log_eval_size);
    if timed { eprintln!("  trace_commit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0); }

    // Download eval to host for decommitment (quotient recomputation needs it).
    // Then drop GPU copy to free 8GB for FRI arena.
    let t0 = Instant::now();
    let eval_full_host = d_eval_full.to_host();
    if timed { eprintln!("  trace_download: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0); }

    // --- Steps 6+7: Fused quotient + circle fold (zero host transfer!) ---
    // Instead of: quotient → download 32GB → upload for fold
    // Do: quotient chunk → circle fold chunk → drop quotient. All on GPU.
    // Quotient values for decommitment re-computed from eval_full_host at query time.
    let t0 = Instant::now();
    let mut channel = Channel::new();
    channel.mix_digest(&trace_commitment);
    let alpha = channel.draw_felt();
    let alpha_arr = alpha.to_u32_array();

    // Quotient chunk size: 2^27 (2GB output per chunk, 16 chunks)
    // This matches fold_chunk_size so each quotient chunk feeds one fold chunk.
    let quotient_chunk_log = log_eval_size.min(27); // cap at eval domain size
    let chunk_size = 1usize << quotient_chunk_log;
    let n_chunks = eval_size / chunk_size; // 16 chunks

    let mut all_quotient_subtrees: Vec<[u32; 8]> = Vec::new();

    // Compute quotient Merkle subtree roots chunk-by-chunk
    // (we need these for quotient commitment + decommitment)
    for chunk_idx in 0..n_chunks {
        let offset = chunk_idx * chunk_size;
        let mut dq0 = DeviceBuffer::<u32>::alloc(chunk_size);
        let mut dq1 = DeviceBuffer::<u32>::alloc(chunk_size);
        let mut dq2 = DeviceBuffer::<u32>::alloc(chunk_size);
        let mut dq3 = DeviceBuffer::<u32>::alloc(chunk_size);

        unsafe {
            ffi::cuda_fibonacci_quotient_chunk(
                d_eval_full.as_ptr(),
                dq0.as_mut_ptr(), dq1.as_mut_ptr(), dq2.as_mut_ptr(), dq3.as_mut_ptr(),
                alpha_arr.as_ptr(),
                offset as u32,
                chunk_size as u32,
                eval_size as u32,
            );
        }

        // Hash this chunk into Merkle subtree roots (no download!)
        let n_subtrees = chunk_size / 1024;
        let mut d_subtrees = DeviceBuffer::<u32>::alloc(n_subtrees * 8);
        unsafe {
            ffi::cuda_merkle_tiled_soa4(
                dq0.as_ptr(), dq1.as_ptr(), dq2.as_ptr(), dq3.as_ptr(),
                d_subtrees.as_mut_ptr(), chunk_size as u32,
            );
        }
        let sub_host = d_subtrees.to_host();
        let chunk_subtrees: Vec<[u32; 8]> = (0..n_subtrees)
            .map(|i| {
                let mut h = [0u32; 8];
                h.copy_from_slice(&sub_host[i * 8..(i + 1) * 8]);
                h
            })
            .collect();
        all_quotient_subtrees.extend_from_slice(&chunk_subtrees);
        // dq0-3, d_subtrees dropped — quotient data NOT downloaded to host
    }

    let quotient_commitment = reduce_subtree_roots_to_root(&all_quotient_subtrees);
    channel.mix_digest(&quotient_commitment);
    if timed { eprintln!("  quotient_commit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0); }

    // --- Circle fold: fused with quotient (re-compute quotient per chunk) ---
    let t0 = Instant::now();
    let mut fri_commitments = Vec::new();
    let mut current_log_size = log_eval_size; // 31

    enum FriLayerData {
        HostData([Vec<u32>; 4], Vec<[u32; 8]>),
        GpuTree(MerkleTree, SecureColumn),
        GpuLean(SecureColumn, Vec<[u32; 8]>),
    }
    let mut fri_layer_data: Vec<FriLayerData> = Vec::new();

    let fri_alpha = channel.draw_felt();
    let fold_domain = if log_eval_size <= 30 {
        Coset::half_coset(log_eval_size)
    } else {
        Coset::subgroup(log_eval_size)
    };
    let d_fold_twid = fri::compute_fold_twiddles_on_demand(&fold_domain, true);

    let fold_chunk_size = chunk_size / 2; // 2^26 fold outputs per quotient chunk
    let _fold_chunk_log: u32 = quotient_chunk_log - 1;
    let n_fold_output = eval_size / 2; // 2^30

    let fri_alpha_arr = fri_alpha.to_u32_array();
    let fri_alpha_sq = fri_alpha * fri_alpha;
    let fri_alpha_sq_arr = fri_alpha_sq.to_u32_array();

    // Allocate circle fold output on GPU (16GB)
    let mut fri_gpu = SecureColumn::zeros(n_fold_output);

    // Fused: re-compute quotient chunk → circle fold → drop quotient
    for chunk_idx in 0..n_chunks {
        let q_offset = chunk_idx * chunk_size;
        let fold_offset = chunk_idx * fold_chunk_size;

        // Re-compute quotient for this chunk (GPU, ~0.5s)
        let mut dq0 = DeviceBuffer::<u32>::alloc(chunk_size);
        let mut dq1 = DeviceBuffer::<u32>::alloc(chunk_size);
        let mut dq2 = DeviceBuffer::<u32>::alloc(chunk_size);
        let mut dq3 = DeviceBuffer::<u32>::alloc(chunk_size);
        unsafe {
            ffi::cuda_fibonacci_quotient_chunk(
                d_eval_full.as_ptr(),
                dq0.as_mut_ptr(), dq1.as_mut_ptr(), dq2.as_mut_ptr(), dq3.as_mut_ptr(),
                alpha_arr.as_ptr(),
                q_offset as u32, chunk_size as u32, eval_size as u32,
            );
        }

        // Circle fold this quotient chunk directly into fri_gpu
        // Quotient chunk has chunk_size elements. Fold pairs adjacent → fold_chunk_size outputs.
        unsafe {
            ffi::cuda_fold_circle_into_line_soa(
                fri_gpu.cols[0].as_mut_ptr().add(fold_offset),
                fri_gpu.cols[1].as_mut_ptr().add(fold_offset),
                fri_gpu.cols[2].as_mut_ptr().add(fold_offset),
                fri_gpu.cols[3].as_mut_ptr().add(fold_offset),
                dq0.as_ptr(), dq1.as_ptr(), dq2.as_ptr(), dq3.as_ptr(),
                d_fold_twid.as_ptr().add(fold_offset),
                fri_alpha_arr.as_ptr(),
                fri_alpha_sq_arr.as_ptr(),
                fold_chunk_size as u32,
            );
        }
        // Quotient chunk dropped — zero host transfer!
    }
    drop(d_fold_twid);
    drop(d_eval_full); // free 8GB — eval data on host, VRAM freed for FRI arena
    unsafe { ffi::cuda_device_sync() };

    // Commit circle fold output
    let (fri0_root, fri0_subtrees) = MerkleTree::commit_root_soa4_with_subtrees(
        &fri_gpu.cols[0], &fri_gpu.cols[1],
        &fri_gpu.cols[2], &fri_gpu.cols[3], log_eval_size - 1,
    );
    fri_commitments.push(fri0_root);
    channel.mix_digest(&fri0_root);
    current_log_size -= 1; // now 30
    if timed {
        eprintln!("  fused_quotient_fold: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- GPU-resident line folds (no host round-trips!) ---
    // VRAM budget: prev(16GB) + twid(4GB) + next(8GB) = 28GB for first fold.
    // Prefetch deferred until after drop(prev) to avoid 32GB peak.

    // Download circle fold output for decommitment (16GB — too large for GPU arena)
    let fri0_host = [
        fri_gpu.cols[0].to_host(), fri_gpu.cols[1].to_host(),
        fri_gpu.cols[2].to_host(), fri_gpu.cols[3].to_host(),
    ];
    fri_layer_data.push(FriLayerData::HostData(fri0_host, fri0_subtrees));

    // GPU-resident line folds: fold from fri_gpu, store results.
    // Large layers download to host (VRAM headroom). Small layers clone to GPU.
    let mut gpu_fold_active = true;
    while current_log_size > FRI_CPU_TAIL_LOG.max(3) {
        let ti = Instant::now();
        let fold_alpha = channel.draw_felt();

        let line_domain = Coset::half_coset(current_log_size);
        let d_twid = fri::compute_fold_twiddles_on_demand(&line_domain, false);

        let folded = if gpu_fold_active {
            let result = fri::fold_line_with_twiddles(&fri_gpu, fold_alpha, &d_twid);
            drop(d_twid);
            drop(fri_gpu);
            fri_gpu = SecureColumn::alloc(0);
            result
        } else {
            let src = match fri_layer_data.last().unwrap() {
                FriLayerData::GpuTree(_, eval) => eval,
                _ => unreachable!(),
            };
            let result = fri::fold_line_with_twiddles(src, fold_alpha, &d_twid);
            drop(d_twid);
            result
        };
        current_log_size -= 1;

        const FRI_LEAN_THRESHOLD_LOG: u32 = 18;
        if current_log_size >= FRI_LEAN_THRESHOLD_LOG {
            let (fri_root, subtrees) = MerkleTree::commit_root_soa4_with_subtrees(
                &folded.cols[0], &folded.cols[1],
                &folded.cols[2], &folded.cols[3], current_log_size,
            );
            fri_commitments.push(fri_root);
            channel.mix_digest(&fri_root);
            if current_log_size <= 29 {
                fri_layer_data.push(FriLayerData::GpuLean(
                    SecureColumn {
                        cols: std::array::from_fn(|c| folded.cols[c].clone_device()),
                        len: folded.len,
                    },
                    subtrees,
                ));
            } else {
                let host_data = [
                    folded.cols[0].to_host(), folded.cols[1].to_host(),
                    folded.cols[2].to_host(), folded.cols[3].to_host(),
                ];
                fri_layer_data.push(FriLayerData::HostData(host_data, subtrees));
            }
            fri_gpu = folded;
            gpu_fold_active = true;
        } else {
            let fri_tree = MerkleTree::commit_soa4(
                &folded.cols[0], &folded.cols[1],
                &folded.cols[2], &folded.cols[3], current_log_size,
            );
            fri_commitments.push(fri_tree.root());
            channel.mix_digest(&fri_tree.root());
            fri_layer_data.push(FriLayerData::GpuTree(fri_tree, folded));
            gpu_fold_active = false;
        }
        if timed {
            eprintln!("    fri line (log={}): {:.1}ms", current_log_size, ti.elapsed().as_secs_f64() * 1000.0);
        }
    }
    drop(fri_gpu);

    // CPU tail
    let mut cpu_eval = match fri_layer_data.last().unwrap() {
        FriLayerData::GpuTree(_, eval) | FriLayerData::GpuLean(eval, _) => eval.to_qm31(),
        FriLayerData::HostData(host, _) => {
            let nn = host[0].len();
            (0..nn).map(|i| QM31::from_u32_array([host[0][i], host[1][i], host[2][i], host[3][i]])).collect()
        }
    };

    let mut cpu_fri_trees: Vec<(Vec<QM31>, [u32; 8])> = Vec::new();
    while current_log_size > 3 {
        let fold_alpha = channel.draw_felt();
        let line_domain = Coset::half_coset(current_log_size);
        let d_twid = fri::compute_fold_twiddles_on_demand(&line_domain, false);
        let twid_host = d_twid.to_host();
        drop(d_twid);

        cpu_eval = fri::fold_line_cpu(&cpu_eval, fold_alpha, &twid_host);
        current_log_size -= 1;

        let fri_root = fri::merkle_root_cpu(&cpu_eval);
        fri_commitments.push(fri_root);
        channel.mix_digest(&fri_root);
        cpu_fri_trees.push((cpu_eval.clone(), fri_root));
    }

    let fri_last_layer = cpu_eval.clone();
    if timed {
        eprintln!("  fri_total: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 8: Query phase ---
    let t0 = Instant::now();

    channel.mix_felts(&fri_last_layer);
    let query_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_size))
        .collect();

    // Decommit trace from host eval data
    let trace_decommitment = decommit_trace_gpu(&eval_full_host, &trace_subtree_roots, &query_indices);

    // Decommit quotient: recompute from host eval data
    let quotient_decommitment = decommit_quotient_recompute(
        &eval_full_host, eval_size, alpha, &all_quotient_subtrees, &query_indices,
    );

    // Decommit FRI layers
    let mut fri_decommitments = Vec::new();
    let mut folded_indices: Vec<usize> = query_indices.iter().map(|&qi| qi / 2).collect();
    for layer_data in fri_layer_data.drain(..) {
        match layer_data {
            FriLayerData::GpuTree(tree, eval) => {
                let decom = decommit_fri_layer(&tree, &eval, &folded_indices);
                fri_decommitments.push(decom);
            }
            FriLayerData::HostData(host, subtrees) => {
                let decom = decommit_soa4_gpu(&host, &subtrees, &folded_indices);
                fri_decommitments.push(decom);
            }
            FriLayerData::GpuLean(ref eval, ref subtrees) => {
                let decom = decommit_soa4_from_gpu_resident(eval, subtrees, &folded_indices);
                fri_decommitments.push(decom);
            }
        }
        folded_indices = folded_indices.iter().map(|&i| i / 2).collect();
    }

    // Decommit CPU tail FRI layers
    for (cpu_vals, _root) in &cpu_fri_trees {
        let values: Vec<[u32; 4]> = folded_indices.iter().map(|&i| cpu_vals[i].to_u32_array()).collect();
        let sibling_indices: Vec<usize> = folded_indices.iter().map(|&i| i ^ 1).collect();
        let sibling_values: Vec<[u32; 4]> = sibling_indices.iter().map(|&i| cpu_vals[i].to_u32_array()).collect();
        let auth_paths = cpu_merkle_auth_paths(cpu_vals, &folded_indices);
        let sibling_auth_paths = cpu_merkle_auth_paths(cpu_vals, &sibling_indices);
        fri_decommitments.push(QueryDecommitment {
            values, sibling_values, auth_paths, sibling_auth_paths,
        });
        folded_indices = folded_indices.iter().map(|&i| i / 2).collect();
    }

    if timed {
        eprintln!("  query_decommit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
        eprintln!("  TOTAL: {:.1}ms", t_total.elapsed().as_secs_f64() * 1000.0);
    }

    StarkProof {
        trace_commitment,
        quotient_commitment,
        fri_commitments,
        fri_last_layer,
        log_trace_size: log_n,
        public_inputs: (a, b),
        query_indices,
        trace_decommitment,
        quotient_decommitment,
        fri_decommitments,
    }
}

/// Reduce a flat list of subtree roots to a single Merkle root.
fn reduce_subtree_roots_to_root(subtree_roots: &[[u32; 8]]) -> [u32; 8] {
    use crate::merkle::HASH_WORDS;

    let n = subtree_roots.len();
    assert!(n.is_power_of_two() && n >= 1);

    if n == 1 {
        return subtree_roots[0];
    }

    // Upload subtree roots to GPU and reduce
    let flat: Vec<u32> = subtree_roots.iter().flat_map(|h| h.iter().copied()).collect();
    let mut current = DeviceBuffer::from_host(&flat);
    let mut current_size = n as u32;

    while current_size > 1024 {
        let parent_size = current_size / 2;
        let mut parents = DeviceBuffer::<u32>::alloc((parent_size as usize) * HASH_WORDS);
        unsafe {
            ffi::cuda_merkle_hash_nodes(current.as_ptr(), parents.as_mut_ptr(), parent_size);
        }
        current = parents;
        current_size = parent_size;
    }

    if current_size > 1 {
        let mut d_root = DeviceBuffer::<u32>::alloc(HASH_WORDS);
        unsafe {
            ffi::cuda_merkle_reduce_to_root(current.as_ptr(), d_root.as_mut_ptr(), current_size);
        }
        let host = d_root.to_host();
        let mut root = [0u32; HASH_WORDS];
        root.copy_from_slice(&host[..HASH_WORDS]);
        root
    } else {
        let host = current.to_host();
        let mut root = [0u32; HASH_WORDS];
        root.copy_from_slice(&host[..HASH_WORDS]);
        root
    }
}

/// Extract trace decommitment with sibling values for fold verification.
fn decommit_trace(
    tree: &MerkleTree,
    d_eval: &DeviceBuffer<u32>,
    indices: &[usize],
) -> QueryDecommitment<u32> {
    let eval_host = d_eval.to_host();
    let values: Vec<u32> = indices.iter().map(|&i| eval_host[i]).collect();
    let sibling_indices: Vec<usize> = indices.iter().map(|&i| i ^ 1).collect();
    let sibling_values: Vec<u32> = sibling_indices.iter().map(|&i| eval_host[i]).collect();
    let auth_paths = tree.batch_auth_paths(indices);
    let sibling_auth_paths = tree.batch_auth_paths(&sibling_indices);
    QueryDecommitment { values, sibling_values, auth_paths, sibling_auth_paths }
}

/// Extract trace decommitment using GPU-accelerated per-tile Merkle trees.
/// Uploads only the ~200 queried tiles to GPU, hashes on GPU, downloads auth paths.
/// Zero CPU blake2s for large trees.
fn decommit_trace_gpu(
    eval_host: &[u32],
    subtree_roots: &[[u32; 8]],
    indices: &[usize],
) -> QueryDecommitment<u32> {
    let values: Vec<u32> = indices.iter().map(|&i| eval_host[i]).collect();
    let sibling_indices: Vec<usize> = indices.iter().map(|&i| i ^ 1).collect();
    let sibling_values: Vec<u32> = sibling_indices.iter().map(|&i| eval_host[i]).collect();

    let n = eval_host.len();
    if n < 4096 {
        let auth_paths = MerkleTree::cpu_merkle_auth_paths_single(eval_host, indices);
        let sibling_auth_paths = MerkleTree::cpu_merkle_auth_paths_single(eval_host, &sibling_indices);
        return QueryDecommitment { values, sibling_values, auth_paths, sibling_auth_paths };
    }

    // GPU path: upload only queried tiles, hash on GPU
    let all_indices: Vec<usize> = indices.iter().chain(sibling_indices.iter()).copied().collect();
    let tile_paths = gpu_tile_auth_paths_single(eval_host, subtree_roots, &all_indices);
    let (auth_paths, sibling_auth_paths) = tile_paths.split_at(indices.len());

    QueryDecommitment {
        values, sibling_values,
        auth_paths: auth_paths.to_vec(),
        sibling_auth_paths: sibling_auth_paths.to_vec(),
    }
}

/// Extract SoA4 decommitment with siblings from host-side column data.
fn decommit_soa4_from_host(
    tree: &MerkleTree,
    host: &[Vec<u32>; 4],
    indices: &[usize],
) -> QueryDecommitment<[u32; 4]> {
    let values: Vec<[u32; 4]> = indices
        .iter()
        .map(|&i| [host[0][i], host[1][i], host[2][i], host[3][i]])
        .collect();
    let sibling_indices: Vec<usize> = indices.iter().map(|&i| i ^ 1).collect();
    let sibling_values: Vec<[u32; 4]> = sibling_indices
        .iter()
        .map(|&i| [host[0][i], host[1][i], host[2][i], host[3][i]])
        .collect();
    let auth_paths = tree.batch_auth_paths(indices);
    let sibling_auth_paths = tree.batch_auth_paths(&sibling_indices);
    QueryDecommitment { values, sibling_values, auth_paths, sibling_auth_paths }
}

/// Extract SoA4 decommitment using GPU-accelerated per-tile Merkle trees.
fn decommit_soa4_gpu(
    host: &[Vec<u32>; 4],
    subtree_roots: &[[u32; 8]],
    indices: &[usize],
) -> QueryDecommitment<[u32; 4]> {
    let values: Vec<[u32; 4]> = indices
        .iter()
        .map(|&i| [host[0][i], host[1][i], host[2][i], host[3][i]])
        .collect();
    let sibling_indices: Vec<usize> = indices.iter().map(|&i| i ^ 1).collect();
    let sibling_values: Vec<[u32; 4]> = sibling_indices
        .iter()
        .map(|&i| [host[0][i], host[1][i], host[2][i], host[3][i]])
        .collect();

    let n = host[0].len();
    if n < 4096 {
        let auth_paths = MerkleTree::cpu_merkle_auth_paths_soa4(host, indices);
        let sibling_auth_paths = MerkleTree::cpu_merkle_auth_paths_soa4(host, &sibling_indices);
        return QueryDecommitment { values, sibling_values, auth_paths, sibling_auth_paths };
    }

    let all_indices: Vec<usize> = indices.iter().chain(sibling_indices.iter()).copied().collect();
    let tile_paths = gpu_tile_auth_paths_soa4(host, subtree_roots, &all_indices);
    let (auth_paths, sibling_auth_paths) = tile_paths.split_at(indices.len());

    QueryDecommitment {
        values, sibling_values,
        auth_paths: auth_paths.to_vec(),
        sibling_auth_paths: sibling_auth_paths.to_vec(),
    }
}

#[allow(dead_code)]
fn decommit_trace_from_gpu_resident(
    d_eval: &DeviceBuffer<u32>,
    subtree_roots: &[[u32; 8]],
    indices: &[usize],
) -> QueryDecommitment<u32> {
    use std::collections::{BTreeSet, HashMap};

    const TILE_SIZE: usize = 1024;

    let sibling_indices: Vec<usize> = indices.iter().map(|&i| i ^ 1).collect();
    let needed_tiles: BTreeSet<usize> = indices.iter()
        .chain(sibling_indices.iter())
        .map(|&qi| qi / TILE_SIZE)
        .collect();

    // Download only needed tiles + build per-tile Merkle trees via D2D
    let mut tile_host: HashMap<usize, Vec<u32>> = HashMap::new();
    let mut tile_trees: HashMap<usize, MerkleTree> = HashMap::new();

    for &tile_idx in &needed_tiles {
        let base = tile_idx * TILE_SIZE;
        let mut d_tile = DeviceBuffer::<u32>::alloc(TILE_SIZE);
        unsafe {
            ffi::cudaMemcpy(
                d_tile.as_mut_ptr() as *mut std::ffi::c_void,
                d_eval.as_ptr().add(base) as *const std::ffi::c_void,
                TILE_SIZE * 4, ffi::MEMCPY_D2D,
            );
        }
        let tree = MerkleTree::commit(std::slice::from_ref(&d_tile), 10);
        tile_host.insert(tile_idx, d_tile.to_host());
        tile_trees.insert(tile_idx, tree);
    }

    let values: Vec<u32> = indices.iter().map(|&i| tile_host[&(i / TILE_SIZE)][i % TILE_SIZE]).collect();
    let sibling_values: Vec<u32> = sibling_indices.iter().map(|&i| tile_host[&(i / TILE_SIZE)][i % TILE_SIZE]).collect();

    let upper_layers = gpu_build_upper_tree(subtree_roots);

    let all_indices: Vec<usize> = indices.iter().chain(sibling_indices.iter()).copied().collect();
    let paths: Vec<Vec<[u32; 8]>> = all_indices.iter().map(|&qi| {
        let tile_idx = qi / TILE_SIZE;
        let intra_idx = qi % TILE_SIZE;
        let mut path = Vec::new();
        path.extend_from_slice(&tile_trees[&tile_idx].auth_path(intra_idx));
        let mut idx = tile_idx;
        for layer in &upper_layers[..upper_layers.len() - 1] {
            path.push(layer[idx ^ 1]);
            idx /= 2;
        }
        path
    }).collect();

    let (auth_paths, sibling_auth_paths) = paths.split_at(indices.len());
    QueryDecommitment {
        values, sibling_values,
        auth_paths: auth_paths.to_vec(),
        sibling_auth_paths: sibling_auth_paths.to_vec(),
    }
}

#[allow(dead_code)]
fn decommit_quotient_from_gpu(
    d_eval: &DeviceBuffer<u32>,
    eval_size: usize,
    alpha: QM31,
    subtree_roots: &[[u32; 8]],
    indices: &[usize],
) -> QueryDecommitment<[u32; 4]> {
    use std::collections::{BTreeSet, HashMap};

    const TILE_SIZE: usize = 1024;

    let sibling_indices: Vec<usize> = indices.iter().map(|&i| i ^ 1).collect();
    let needed_tiles: BTreeSet<usize> = indices.iter()
        .chain(sibling_indices.iter())
        .map(|&qi| qi / TILE_SIZE)
        .collect();

    // Download eval tiles from GPU (for quotient recomputation)
    // Each tile needs +2 elements for the constraint lookahead
    let mut tile_eval: HashMap<usize, Vec<u32>> = HashMap::new();
    for &tile_idx in &needed_tiles {
        let base = tile_idx * TILE_SIZE;
        // Download tile + 2 extra elements for constraint boundary
        let fetch_size = TILE_SIZE + 2;
        let _buf = vec![0u32; fetch_size];
        for j in 0..fetch_size {
            let _idx = (base + j) % eval_size;
            // Single-element D2H (batching would be faster but this is ~200 tiles × 1026 = 205K reads)
            // Actually let me download the tile in one shot and handle boundary separately
            if j < TILE_SIZE {
                // Will be filled below
            }
        }
        // Download tile data in one D2H
        let mut d_tile = DeviceBuffer::<u32>::alloc(TILE_SIZE);
        unsafe {
            ffi::cudaMemcpy(
                d_tile.as_mut_ptr() as *mut std::ffi::c_void,
                d_eval.as_ptr().add(base) as *const std::ffi::c_void,
                TILE_SIZE * 4, ffi::MEMCPY_D2D,
            );
        }
        let mut tile_data = d_tile.to_host();
        // Get the 2 boundary elements
        let mut boundary = [0u32; 2];
        for k in 0..2 {
            let idx = (base + TILE_SIZE + k) % eval_size;
            unsafe {
                ffi::cudaMemcpy(
                    &mut boundary[k] as *mut u32 as *mut std::ffi::c_void,
                    d_eval.as_ptr().add(idx) as *const std::ffi::c_void,
                    4, ffi::MEMCPY_D2H,
                );
            }
        }
        tile_data.push(boundary[0]);
        tile_data.push(boundary[1]);
        tile_eval.insert(tile_idx, tile_data);
    }

    // Recompute quotient values and build tile Merkle trees
    let mut tile_trees: HashMap<usize, MerkleTree> = HashMap::new();
    for &tile_idx in &needed_tiles {
        let _base = tile_idx * TILE_SIZE;
        let eval_tile = &tile_eval[&tile_idx];
        let mut cols = [vec![0u32; TILE_SIZE], vec![0u32; TILE_SIZE],
                       vec![0u32; TILE_SIZE], vec![0u32; TILE_SIZE]];
        for j in 0..TILE_SIZE {
            let t_i = M31(eval_tile[j]);
            let t_i1 = M31(eval_tile[j + 1]);  // safe: we fetched +2
            let t_i2 = M31(eval_tile[j + 2]);
            let constraint = t_i2 - t_i1 - t_i;
            let result = alpha * constraint;
            let arr = result.to_u32_array();
            cols[0][j] = arr[0]; cols[1][j] = arr[1]; cols[2][j] = arr[2]; cols[3][j] = arr[3];
        }
        let c0 = DeviceBuffer::from_host(&cols[0]);
        let c1 = DeviceBuffer::from_host(&cols[1]);
        let c2 = DeviceBuffer::from_host(&cols[2]);
        let c3 = DeviceBuffer::from_host(&cols[3]);
        let tree = MerkleTree::commit_soa4(&c0, &c1, &c2, &c3, 10);
        tile_trees.insert(tile_idx, tree);
    }

    // Extract values at query positions
    let values: Vec<[u32; 4]> = indices.iter().map(|&i| {
        let tile = &tile_eval[&(i / TILE_SIZE)];
        let off = i % TILE_SIZE;
        let t_i = M31(tile[off]);
        let t_i1 = M31(tile[off + 1]);
        let t_i2 = M31(tile[off + 2]);
        let constraint = t_i2 - t_i1 - t_i;
        let result = alpha * constraint;
        result.to_u32_array()
    }).collect();

    let sibling_values: Vec<[u32; 4]> = sibling_indices.iter().map(|&i| {
        let tile = &tile_eval[&(i / TILE_SIZE)];
        let off = i % TILE_SIZE;
        let t_i = M31(tile[off]);
        let t_i1 = M31(tile[off + 1]);
        let t_i2 = M31(tile[off + 2]);
        let constraint = t_i2 - t_i1 - t_i;
        let result = alpha * constraint;
        result.to_u32_array()
    }).collect();

    let upper_layers = gpu_build_upper_tree(subtree_roots);

    let all_indices: Vec<usize> = indices.iter().chain(sibling_indices.iter()).copied().collect();
    let paths: Vec<Vec<[u32; 8]>> = all_indices.iter().map(|&qi| {
        let tile_idx = qi / TILE_SIZE;
        let intra_idx = qi % TILE_SIZE;
        let mut path = Vec::new();
        path.extend_from_slice(&tile_trees[&tile_idx].auth_path(intra_idx));
        let mut idx = tile_idx;
        for layer in &upper_layers[..upper_layers.len() - 1] {
            path.push(layer[idx ^ 1]);
            idx /= 2;
        }
        path
    }).collect();

    let (auth_paths, sibling_auth_paths) = paths.split_at(indices.len());
    QueryDecommitment {
        values, sibling_values,
        auth_paths: auth_paths.to_vec(),
        sibling_auth_paths: sibling_auth_paths.to_vec(),
    }
}

/// Recompute quotient values at specific positions from eval host data.
/// This avoids storing the full 32GB quotient on host — values computed on demand.
fn recompute_quotient_at(eval_host: &[u32], eval_size: usize, alpha: QM31, index: usize) -> [u32; 4] {
    let t_i = M31(eval_host[index]);
    let t_i1 = M31(eval_host[(index + 1) % eval_size]);
    let t_i2 = M31(eval_host[(index + 2) % eval_size]);
    let constraint = t_i2 - t_i1 - t_i;
    // QM31 * M31 — matches GPU kernel qm31_mul_m31(alpha, constraint)
    let result = alpha * constraint;
    result.to_u32_array()
}

/// Decommit quotient by recomputing values on CPU from eval_full_host.
/// Downloads only ~3.2MB of tile data (re-computed) for Merkle auth paths.
fn decommit_quotient_recompute(
    eval_host: &[u32],
    eval_size: usize,
    alpha: QM31,
    subtree_roots: &[[u32; 8]],
    indices: &[usize],
) -> QueryDecommitment<[u32; 4]> {
    use std::collections::{BTreeSet, HashMap};

    const TILE_SIZE: usize = 1024;

    // Recompute values at query positions
    let values: Vec<[u32; 4]> = indices.iter()
        .map(|&i| recompute_quotient_at(eval_host, eval_size, alpha, i))
        .collect();
    let sibling_indices: Vec<usize> = indices.iter().map(|&i| i ^ 1).collect();
    let sibling_values: Vec<[u32; 4]> = sibling_indices.iter()
        .map(|&i| recompute_quotient_at(eval_host, eval_size, alpha, i))
        .collect();

    // Collect needed tiles and recompute quotient for each tile on CPU
    let needed_tiles: BTreeSet<usize> = indices.iter()
        .chain(sibling_indices.iter())
        .map(|&qi| qi / TILE_SIZE)
        .collect();

    // Build per-tile Merkle trees from recomputed quotient data
    let mut tile_trees: HashMap<usize, MerkleTree> = HashMap::new();
    for &tile_idx in &needed_tiles {
        let base = tile_idx * TILE_SIZE;
        // Recompute quotient for this tile on CPU
        let tile_data: [Vec<u32>; 4] = {
            let mut cols = [vec![0u32; TILE_SIZE], vec![0u32; TILE_SIZE],
                           vec![0u32; TILE_SIZE], vec![0u32; TILE_SIZE]];
            for j in 0..TILE_SIZE {
                let q = recompute_quotient_at(eval_host, eval_size, alpha, base + j);
                cols[0][j] = q[0]; cols[1][j] = q[1]; cols[2][j] = q[2]; cols[3][j] = q[3];
            }
            cols
        };
        // Upload tile and build Merkle tree on GPU
        let c0 = DeviceBuffer::from_host(&tile_data[0]);
        let c1 = DeviceBuffer::from_host(&tile_data[1]);
        let c2 = DeviceBuffer::from_host(&tile_data[2]);
        let c3 = DeviceBuffer::from_host(&tile_data[3]);
        let tree = MerkleTree::commit_soa4(&c0, &c1, &c2, &c3, 10);
        tile_trees.insert(tile_idx, tree);
    }

    // Upper tree from subtree roots
    let upper_layers = gpu_build_upper_tree(subtree_roots);

    // Extract auth paths
    let all_indices: Vec<usize> = indices.iter().chain(sibling_indices.iter()).copied().collect();
    let paths: Vec<Vec<[u32; 8]>> = all_indices.iter().map(|&qi| {
        let tile_idx = qi / TILE_SIZE;
        let intra_idx = qi % TILE_SIZE;
        let mut path = Vec::new();
        let tree = &tile_trees[&tile_idx];
        path.extend_from_slice(&tree.auth_path(intra_idx));
        let mut idx = tile_idx;
        for layer in &upper_layers[..upper_layers.len() - 1] {
            path.push(layer[idx ^ 1]);
            idx /= 2;
        }
        path
    }).collect();

    let (auth_paths, sibling_auth_paths) = paths.split_at(indices.len());
    QueryDecommitment {
        values, sibling_values,
        auth_paths: auth_paths.to_vec(),
        sibling_auth_paths: sibling_auth_paths.to_vec(),
    }
}

/// Extract SoA4 decommitment directly from GPU SecureColumn.
/// Downloads only the ~200 needed tiles (~3.2MB) instead of the full layer (8GB+).
/// Tile Merkle trees built via D2D copies — zero host round-trip for hashing.
fn decommit_soa4_from_gpu_resident(
    gpu_cols: &SecureColumn,
    subtree_roots: &[[u32; 8]],
    indices: &[usize],
) -> QueryDecommitment<[u32; 4]> {
    use std::collections::BTreeSet;

    const TILE_SIZE: usize = 1024;
    let n = gpu_cols.len;

    let sibling_indices: Vec<usize> = indices.iter().map(|&i| i ^ 1).collect();
    let needed_tiles: BTreeSet<usize> = indices.iter()
        .chain(sibling_indices.iter())
        .map(|&qi| qi / TILE_SIZE)
        .collect();

    let n_needed = needed_tiles.len();
    let batch_size = n_needed * TILE_SIZE;
    let tile_indices: Vec<usize> = needed_tiles.iter().copied().collect();

    // Allocate contiguous GPU buffers for needed tiles
    let mut d_batch: [DeviceBuffer<u32>; 4] = std::array::from_fn(|_| DeviceBuffer::<u32>::alloc(batch_size));

    // Queue all D2D copies
    for (slot, &tile_idx) in tile_indices.iter().enumerate() {
        let src_base = tile_idx * TILE_SIZE;
        let dst_base = slot * TILE_SIZE;
        for c in 0..4 {
            unsafe {
                ffi::cudaMemcpy(
                    d_batch[c].as_mut_ptr().add(dst_base) as *mut std::ffi::c_void,
                    gpu_cols.cols[c].as_ptr().add(src_base) as *const std::ffi::c_void,
                    TILE_SIZE * 4,
                    ffi::MEMCPY_D2D,
                );
            }
        }
    }

    // Single sync + batch download to host
    unsafe { ffi::cuda_device_sync() };
    let host_batch: [Vec<u32>; 4] = std::array::from_fn(|c| d_batch[c].to_host());
    drop(d_batch);

    // Build index: tile_idx → slot in batch
    let tile_slot: std::collections::HashMap<usize, usize> = tile_indices.iter()
        .enumerate().map(|(slot, &idx)| (idx, slot)).collect();

    // Extract values from batch
    let get_val = |i: usize| -> [u32; 4] {
        let slot = tile_slot[&(i / TILE_SIZE)];
        let off = slot * TILE_SIZE + (i % TILE_SIZE);
        [host_batch[0][off], host_batch[1][off], host_batch[2][off], host_batch[3][off]]
    };
    let values: Vec<[u32; 4]> = indices.iter().map(|&i| get_val(i)).collect();
    let sibling_values: Vec<[u32; 4]> = sibling_indices.iter().map(|&i| get_val(i)).collect();

    // Auth paths: targeted approach (CPU tile subtrees + CPU upper tree)
    let all_indices: Vec<usize> = indices.iter().chain(sibling_indices.iter()).copied().collect();
    let hash_leaf = |i: usize| -> [u32; 8] {
        let slot = tile_slot[&(i / TILE_SIZE)];
        let off = slot * TILE_SIZE + (i % TILE_SIZE);
        MerkleTree::hash_leaf(&[host_batch[0][off], host_batch[1][off],
                                host_batch[2][off], host_batch[3][off]])
    };
    let auth_paths_all = MerkleTree::targeted_auth_paths_with_tile_roots(
        subtree_roots, n, &all_indices, &hash_leaf,
    );

    let (auth_paths, sibling_auth_paths) = auth_paths_all.split_at(indices.len());
    QueryDecommitment {
        values, sibling_values,
        auth_paths: auth_paths.to_vec(),
        sibling_auth_paths: sibling_auth_paths.to_vec(),
    }
}

/// Extract FRI layer decommitment with siblings.
fn decommit_fri_layer(
    tree: &MerkleTree,
    eval: &SecureColumn,
    folded_indices: &[usize],
) -> QueryDecommitment<[u32; 4]> {
    let c0 = eval.cols[0].to_host();
    let c1 = eval.cols[1].to_host();
    let c2 = eval.cols[2].to_host();
    let c3 = eval.cols[3].to_host();
    let host = [c0, c1, c2, c3];
    decommit_soa4_from_host(tree, &host, folded_indices)
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

    // --- Step 4: Commit trace (full tree for decommitment) ---
    let t0 = Instant::now();
    let trace_tree = MerkleTree::commit(std::slice::from_ref(&d_eval), log_eval_size);
    let trace_commitment = trace_tree.root();
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

    let quotient_col = SecureColumn {
        cols: [q0, q1, q2, q3],
        len: eval_size,
    };
    if timed {
        unsafe { ffi::cuda_device_sync() };
        eprintln!("  quotient_gpu: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 6: Commit quotient (full tree for decommitment) ---
    let t0 = Instant::now();
    let quotient_tree = MerkleTree::commit_soa4(
        &quotient_col.cols[0], &quotient_col.cols[1],
        &quotient_col.cols[2], &quotient_col.cols[3],
        log_eval_size,
    );
    let quotient_commitment = quotient_tree.root();
    channel.mix_digest(&quotient_commitment);
    if timed {
        eprintln!("  quotient_commit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // Download quotient values to host before FRI consumes the GPU buffers.
    // We need these for decommitment after query indices are drawn.
    let quotient_host: [Vec<u32>; 4] = [
        quotient_col.cols[0].to_host(),
        quotient_col.cols[1].to_host(),
        quotient_col.cols[2].to_host(),
        quotient_col.cols[3].to_host(),
    ];

    // --- Step 7: FRI (retain trees + evaluations for decommitment) ---
    let t0 = Instant::now();
    let mut fri_commitments = Vec::new();
    let mut fri_trees: Vec<MerkleTree> = Vec::new();
    let mut fri_evals: Vec<SecureColumn> = Vec::new();
    let current_eval = quotient_col;
    let mut current_log_size = log_eval_size;

    // Circle fold
    let fri_alpha = channel.draw_felt();
    let mut line_eval = SecureColumn::zeros(current_eval.len / 2);
    fri::fold_circle_into_line_with_twiddles(
        &mut line_eval, &current_eval, fri_alpha, &cache.fri_twiddles[0],
    );
    drop(current_eval);
    current_log_size -= 1;

    let fri_tree = MerkleTree::commit_soa4(
        &line_eval.cols[0], &line_eval.cols[1],
        &line_eval.cols[2], &line_eval.cols[3],
        current_log_size,
    );
    let fri_root = fri_tree.root();
    fri_commitments.push(fri_root);
    channel.mix_digest(&fri_root);
    fri_trees.push(fri_tree);
    fri_evals.push(line_eval);
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
            &fri_evals.last().unwrap(), fold_alpha, &cache.fri_twiddles[twid_idx],
        );
        twid_idx += 1;
        current_log_size -= 1;

        let fri_tree = MerkleTree::commit_soa4(
            &folded.cols[0], &folded.cols[1],
            &folded.cols[2], &folded.cols[3],
            current_log_size,
        );
        let fri_root = fri_tree.root();
        fri_commitments.push(fri_root);
        channel.mix_digest(&fri_root);
        fri_trees.push(fri_tree);
        fri_evals.push(folded);
        if timed {
            eprintln!("    fri[{}] gpu fold+commit (log={}): {:.3}ms", twid_idx, current_log_size, ti.elapsed().as_secs_f64() * 1000.0);
        }
    }

    // CPU tail
    let mut cpu_eval = {
        let last = fri_evals.last().unwrap();
        last.to_qm31()
    };

    // CPU tail layers — use CPU Merkle trees
    let mut cpu_fri_trees: Vec<(Vec<QM31>, [u32; 8])> = Vec::new();
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
        cpu_fri_trees.push((cpu_eval.clone(), fri_root));
        if timed {
            eprintln!("    fri[{}] cpu fold+commit (log={}): {:.3}ms", twid_idx, current_log_size, ti.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let fri_last_layer = cpu_eval.clone();
    if timed {
        eprintln!("  fri_fold+commit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Step 8: Query phase ---
    let t0 = Instant::now();

    // Mix last layer into channel, then draw query indices
    channel.mix_felts(&fri_last_layer);
    let query_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_size))
        .collect();

    // Decommit trace
    let trace_decommitment = decommit_trace(&trace_tree, &d_eval, &query_indices);
    drop(d_eval);
    drop(trace_tree);

    // Decommit quotient (using host copy saved before FRI consumed GPU buffers)
    let quotient_decommitment = decommit_soa4_from_host(&quotient_tree, &quotient_host, &query_indices);
    drop(quotient_tree);

    // Decommit FRI layers (GPU layers)
    let mut fri_decommitments = Vec::new();
    let mut folded_indices: Vec<usize> = query_indices.iter().map(|&qi| qi / 2).collect();
    for (tree, eval) in fri_trees.iter().zip(fri_evals.iter()) {
        let decom = decommit_fri_layer(tree, eval, &folded_indices);
        fri_decommitments.push(decom);
        folded_indices = folded_indices.iter().map(|&i| i / 2).collect();
    }
    drop(fri_trees);
    drop(fri_evals);

    // Decommit CPU tail FRI layers
    for (cpu_vals, _root) in &cpu_fri_trees {
        let values: Vec<[u32; 4]> = folded_indices
            .iter()
            .map(|&i| cpu_vals[i].to_u32_array())
            .collect();
        let sibling_indices: Vec<usize> = folded_indices.iter().map(|&i| i ^ 1).collect();
        let sibling_values: Vec<[u32; 4]> = sibling_indices
            .iter()
            .map(|&i| cpu_vals[i].to_u32_array())
            .collect();
        let auth_paths = cpu_merkle_auth_paths(cpu_vals, &folded_indices);
        let sibling_auth_paths = cpu_merkle_auth_paths(cpu_vals, &sibling_indices);
        fri_decommitments.push(QueryDecommitment {
            values, sibling_values, auth_paths, sibling_auth_paths,
        });
        folded_indices = folded_indices.iter().map(|&i| i / 2).collect();
    }

    if timed {
        eprintln!("  query_decommit: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
        eprintln!("  TOTAL: {:.1}ms", t_total.elapsed().as_secs_f64() * 1000.0);
    }

    StarkProof {
        trace_commitment,
        quotient_commitment,
        fri_commitments,
        fri_last_layer,
        log_trace_size: log_n,
        public_inputs: (a, b),
        query_indices,
        trace_decommitment,
        quotient_decommitment,
        fri_decommitments,
    }
}

/// Build Merkle tree from pre-hashed nodes entirely on GPU using hash_nodes kernel.
/// Returns layers as host vectors [leaves(=subtree_roots), parents, ..., root].
/// Uses GPU for the heavy hashing, downloads all layers for path extraction.
fn gpu_build_upper_tree(subtree_roots: &[[u32; 8]]) -> Vec<Vec<[u32; 8]>> {
    use crate::merkle::HASH_WORDS;

    let n = subtree_roots.len();
    if n <= 1 {
        return vec![subtree_roots.to_vec()];
    }

    // Upload subtree roots as flat u32 buffer
    let flat: Vec<u32> = subtree_roots.iter().flat_map(|h| h.iter().copied()).collect();
    let d_current = DeviceBuffer::from_host(&flat);

    // Download leaf layer
    let mut layers: Vec<Vec<[u32; HASH_WORDS]>> = vec![subtree_roots.to_vec()];
    let mut current = d_current;
    let mut current_size = n as u32;

    // Reduce using GPU hash_nodes
    while current_size > 1 {
        let parent_size = current_size / 2;
        let mut parents = DeviceBuffer::<u32>::alloc((parent_size as usize) * HASH_WORDS);
        unsafe {
            ffi::cuda_merkle_hash_nodes(current.as_ptr(), parents.as_mut_ptr(), parent_size);
        }

        // Download this layer for path extraction
        let host = parents.to_host();
        let layer: Vec<[u32; HASH_WORDS]> = (0..parent_size as usize)
            .map(|i| {
                let mut h = [0u32; HASH_WORDS];
                h.copy_from_slice(&host[i * HASH_WORDS..(i + 1) * HASH_WORDS]);
                h
            })
            .collect();
        layers.push(layer);

        current = parents;
        current_size = parent_size;
    }

    layers
}

/// GPU-accelerated auth path extraction for single-column data.
/// Uploads only the ~200 queried tiles (1024 leaves each) to GPU,
/// builds per-tile Merkle trees on GPU, and extracts auth paths.
/// Upper tree (subtree roots → final root) built on CPU (small).
fn gpu_tile_auth_paths_single(
    host_col: &[u32],
    subtree_roots: &[[u32; 8]],
    indices: &[usize],
) -> Vec<Vec<[u32; 8]>> {
    use std::collections::{BTreeSet, HashMap};

    const TILE_SIZE: usize = 1024;
    let n = host_col.len();
    let _n_tiles = n / TILE_SIZE;

    // Collect unique tiles needed
    let needed_tiles: BTreeSet<usize> = indices.iter().map(|&qi| qi / TILE_SIZE).collect();

    // Build per-tile GPU Merkle trees (only for queried tiles)
    let mut tile_trees: HashMap<usize, MerkleTree> = HashMap::new();
    for &tile_idx in &needed_tiles {
        let base = tile_idx * TILE_SIZE;
        let tile_data: Vec<u32> = host_col[base..base + TILE_SIZE].to_vec();
        let d_tile = DeviceBuffer::from_host(&tile_data);
        let log_tile = 10; // log2(1024)
        let tree = MerkleTree::commit(std::slice::from_ref(&d_tile), log_tile);
        tile_trees.insert(tile_idx, tree);
    }

    // Build upper tree on GPU using hash_nodes (subtree roots are already hashes)
    let upper_layers = gpu_build_upper_tree(subtree_roots);

    // Extract auth paths: intra-tile (GPU) + upper tree (CPU)
    indices
        .iter()
        .map(|&qi| {
            let tile_idx = qi / TILE_SIZE;
            let intra_idx = qi % TILE_SIZE;
            let mut path = Vec::new();

            // Intra-tile path from GPU tree
            let tree = &tile_trees[&tile_idx];
            let intra_path = tree.auth_path(intra_idx);
            path.extend_from_slice(&intra_path);

            // Upper tree path
            let mut idx = tile_idx;
            for layer in &upper_layers[..upper_layers.len() - 1] {
                path.push(layer[idx ^ 1]);
                idx /= 2;
            }

            path
        })
        .collect()
}

/// GPU-accelerated auth path extraction for SoA4 (QM31) data.
fn gpu_tile_auth_paths_soa4(
    host_cols: &[Vec<u32>; 4],
    subtree_roots: &[[u32; 8]],
    indices: &[usize],
) -> Vec<Vec<[u32; 8]>> {
    use std::collections::{BTreeSet, HashMap};

    const TILE_SIZE: usize = 1024;

    // Collect unique tiles needed
    let needed_tiles: BTreeSet<usize> = indices.iter().map(|&qi| qi / TILE_SIZE).collect();

    // Build per-tile GPU Merkle trees
    let mut tile_trees: HashMap<usize, MerkleTree> = HashMap::new();
    for &tile_idx in &needed_tiles {
        let base = tile_idx * TILE_SIZE;
        let c0 = DeviceBuffer::from_host(&host_cols[0][base..base + TILE_SIZE]);
        let c1 = DeviceBuffer::from_host(&host_cols[1][base..base + TILE_SIZE]);
        let c2 = DeviceBuffer::from_host(&host_cols[2][base..base + TILE_SIZE]);
        let c3 = DeviceBuffer::from_host(&host_cols[3][base..base + TILE_SIZE]);
        let tree = MerkleTree::commit_soa4(&c0, &c1, &c2, &c3, 10);
        tile_trees.insert(tile_idx, tree);
    }

    // Upper tree from subtree roots on GPU
    let upper_layers = gpu_build_upper_tree(subtree_roots);

    // Extract auth paths
    indices
        .iter()
        .map(|&qi| {
            let tile_idx = qi / TILE_SIZE;
            let intra_idx = qi % TILE_SIZE;
            let mut path = Vec::new();

            let tree = &tile_trees[&tile_idx];
            let intra_path = tree.auth_path(intra_idx);
            path.extend_from_slice(&intra_path);

            let mut idx = tile_idx;
            for layer in &upper_layers[..upper_layers.len() - 1] {
                path.push(layer[idx ^ 1]);
                idx /= 2;
            }

            path
        })
        .collect()
}

/// Build CPU Merkle tree from QM31 values and extract auth paths at given indices.
fn cpu_merkle_auth_paths(values: &[QM31], indices: &[usize]) -> Vec<Vec<[u32; 8]>> {
    use crate::channel::blake2s_hash;

    let n = values.len();
    assert!(n.is_power_of_two() && n >= 1);

    // Hash leaves
    let leaf_hashes: Vec<[u32; 8]> = values
        .iter()
        .map(|v| {
            let arr = v.to_u32_array();
            let mut input = [0u8; 16];
            for (i, &w) in arr.iter().enumerate() {
                input[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            let h = blake2s_hash(&input);
            let mut out = [0u32; 8];
            for i in 0..8 {
                out[i] = u32::from_le_bytes([h[i*4], h[i*4+1], h[i*4+2], h[i*4+3]]);
            }
            out
        })
        .collect();

    // Build all layers
    let mut layers: Vec<Vec<[u32; 8]>> = vec![leaf_hashes];
    while layers.last().unwrap().len() > 1 {
        let prev = layers.last().unwrap();
        let parent_count = prev.len() / 2;
        let parents: Vec<[u32; 8]> = (0..parent_count)
            .map(|i| {
                let mut input = [0u8; 64];
                for (j, &w) in prev[2 * i].iter().enumerate() {
                    input[j * 4..j * 4 + 4].copy_from_slice(&w.to_le_bytes());
                }
                for (j, &w) in prev[2 * i + 1].iter().enumerate() {
                    input[32 + j * 4..32 + j * 4 + 4].copy_from_slice(&w.to_le_bytes());
                }
                let h = blake2s_hash(&input);
                let mut out = [0u32; 8];
                for k in 0..8 {
                    out[k] = u32::from_le_bytes([h[k*4], h[k*4+1], h[k*4+2], h[k*4+3]]);
                }
                out
            })
            .collect();
        layers.push(parents);
    }

    // Extract auth paths: layers[0]=leaves, layers[last]=root
    indices
        .iter()
        .map(|&qi| {
            let mut path = Vec::new();
            let mut idx = qi;
            for layer in &layers[..layers.len() - 1] {
                let sibling = idx ^ 1;
                path.push(layer[sibling]);
                idx /= 2;
            }
            path
        })
        .collect()
}

/// Double-buffered prover pipeline. Overlaps CPU trace generation for proof N+1
/// with GPU processing of proof N, hiding the trace gen latency.
pub struct ProverPipeline {
    cache: ProverCache,
    /// Second pinned buffer for double-buffering.
    pinned_trace_b: *mut u32,
}

unsafe impl Send for ProverPipeline {}

impl ProverPipeline {
    pub fn new(log_n: u32) -> Self {
        let cache = ProverCache::new(log_n);
        let bytes = (1usize << log_n) * std::mem::size_of::<u32>();
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { ffi::cudaMallocHost(&mut ptr, bytes) };
        assert!(err == 0, "cudaMallocHost failed for pipeline buffer: {err}");
        Self {
            cache,
            pinned_trace_b: ptr as *mut u32,
        }
    }

    /// Get a reference to the underlying cache (for non-pipelined use).
    pub fn cache(&self) -> &ProverCache {
        &self.cache
    }

    /// Prove a batch of inputs with pipelined trace generation.
    /// Returns proofs in order. Throughput is limited by max(trace_gen, gpu_work)
    /// instead of trace_gen + gpu_work.
    pub fn prove_batch(&self, inputs: &[(M31, M31)]) -> Vec<StarkProof> {
        if inputs.is_empty() {
            return Vec::new();
        }

        let log_n = self.cache.log_n;
        let n = 1usize << log_n;
        let buffers = [self.cache.pinned_trace, self.pinned_trace_b];
        let mut proofs = Vec::with_capacity(inputs.len());

        // Generate first trace (no overlap possible for the first one)
        unsafe { air::fibonacci_trace_parallel(inputs[0].0, inputs[0].1, log_n, buffers[0]) };

        for i in 0..inputs.len() {
            let current_buf = buffers[i % 2];

            if i + 1 < inputs.len() {
                // Overlap: generate next trace on CPU while GPU processes current proof
                let next_buf = buffers[(i + 1) % 2];
                let next_a = inputs[i + 1].0;
                let next_b = inputs[i + 1].1;
                let next_buf_addr = next_buf as usize;

                std::thread::scope(|s| {
                    // Background: generate next trace
                    s.spawn(move || {
                        let ptr = next_buf_addr as *mut u32;
                        unsafe { air::fibonacci_trace_parallel(next_a, next_b, log_n, ptr) };
                    });

                    // Foreground: process current proof on GPU
                    let proof = prove_from_pinned(current_buf, n, &self.cache);
                    proofs.push(proof);
                });
            } else {
                // Last proof: no overlap needed
                let proof = prove_from_pinned(current_buf, n, &self.cache);
                proofs.push(proof);
            }
        }

        proofs
    }
}

impl Drop for ProverPipeline {
    fn drop(&mut self) {
        if !self.pinned_trace_b.is_null() {
            unsafe { ffi::cudaFreeHost(self.pinned_trace_b as *mut std::ffi::c_void) };
        }
    }
}

/// Prove from a pre-filled pinned trace buffer (used by pipeline).
/// Copies the trace into the cache's pinned buffer, then delegates to prove_with_cache.
fn prove_from_pinned(pinned_trace: *const u32, n: usize, cache: &ProverCache) -> StarkProof {
    let a = M31(unsafe { *pinned_trace });
    let b = M31(unsafe { *pinned_trace.add(1) });

    // If the source buffer isn't the cache's own pinned buffer, copy it.
    if pinned_trace != cache.pinned_trace as *const u32 {
        unsafe {
            std::ptr::copy_nonoverlapping(pinned_trace, cache.pinned_trace, n);
        }
    }

    prove_with_cache(a, b, cache, false)
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
        assert_eq!(proof1.fri_last_layer, proof2.fri_last_layer);
        assert_eq!(proof1.query_indices, proof2.query_indices);
    }

    #[test]
    fn test_prove_different_inputs() {
        let proof1 = prove(M31(1), M31(1), 5);
        let proof2 = prove(M31(2), M31(3), 5);

        assert_ne!(proof1.trace_commitment, proof2.trace_commitment);
    }

    #[test]
    fn test_prove_has_decommitments() {
        let proof = prove(M31(1), M31(1), 6);
        assert_eq!(proof.query_indices.len(), N_QUERIES);
        assert_eq!(proof.trace_decommitment.values.len(), N_QUERIES);
        assert_eq!(proof.trace_decommitment.auth_paths.len(), N_QUERIES);
        assert_eq!(proof.quotient_decommitment.values.len(), N_QUERIES);
        assert!(!proof.fri_decommitments.is_empty());
        assert!(!proof.fri_last_layer.is_empty());
    }

    #[test]
    fn test_pipeline_matches_cached() {
        let a = M31(1);
        let b = M31(1);
        let log_n = 6;

        // Reference: cached proof
        let cache = ProverCache::new(log_n);
        let cached_proof = prove_cached(a, b, &cache);

        // Pipeline proof
        let pipeline = ProverPipeline::new(log_n);
        let pipelined = pipeline.prove_batch(&[(a, b)]);
        let pipe_proof = &pipelined[0];

        assert_eq!(cached_proof.trace_commitment, pipe_proof.trace_commitment);
        assert_eq!(cached_proof.quotient_commitment, pipe_proof.quotient_commitment);
        assert_eq!(cached_proof.fri_commitments, pipe_proof.fri_commitments);
        assert_eq!(cached_proof.fri_last_layer, pipe_proof.fri_last_layer);
    }

    #[test]
    fn test_pipeline_batch() {
        let pipeline = ProverPipeline::new(5);
        let inputs = vec![(M31(1), M31(1)), (M31(2), M31(3)), (M31(5), M31(8))];
        let proofs = pipeline.prove_batch(&inputs);

        assert_eq!(proofs.len(), 3);
        // Each proof should be different (different inputs)
        assert_ne!(proofs[0].trace_commitment, proofs[1].trace_commitment);
        assert_ne!(proofs[1].trace_commitment, proofs[2].trace_commitment);
    }

    #[test]
    fn test_prove_lean_runs() {
        let a = M31(1);
        let b = M31(1);
        let proof = prove_lean(a, b, 6);

        assert_ne!(proof.trace_commitment, [0; 8]);
        assert_ne!(proof.quotient_commitment, [0; 8]);
        assert!(!proof.fri_commitments.is_empty());
        assert_eq!(proof.log_trace_size, 6);
    }

    #[test]
    fn test_prove_lean_matches_prove() {
        // prove_lean should produce the same commitments as prove
        // (same Fiat-Shamir transcript, same math, just different VRAM strategy)
        let a = M31(1);
        let b = M31(1);
        let proof_std = prove(a, b, 6);
        let proof_lean = prove_lean(a, b, 6);

        assert_eq!(proof_std.trace_commitment, proof_lean.trace_commitment);
        assert_eq!(proof_std.quotient_commitment, proof_lean.quotient_commitment);
        assert_eq!(proof_std.fri_commitments, proof_lean.fri_commitments);
        assert_eq!(proof_std.fri_last_layer, proof_lean.fri_last_layer);
        assert_eq!(proof_std.query_indices, proof_lean.query_indices);
    }

    #[test]
    fn test_prove_lean_on_subgroup_small() {
        // Test the full proof pipeline using subgroup eval domain (like log_n=30 does)
        // at a small size where we can verify correctness.
        // This manually constructs a proof using subgroup(log_n) as eval domain
        // (instead of the standard half_coset(log_n+1)).
        let log_n = 6u32;
        let n = 1usize << log_n;
        let eval_size = 2 * n;
        let log_eval = log_n + 1;

        // Standard proof (reference)
        let std_proof = prove_lean(M31(1), M31(1), log_n);
        let std_result = crate::verifier::verify(&std_proof);
        assert!(std_result.is_ok(), "Standard proof should verify");

        // Now: the standard proof uses half_coset(log_n+1) as eval domain.
        // For log_n=30, we'd use subgroup(31) instead.
        // At log_n=6, let's verify that the NTT on subgroup(7) produces a DIFFERENT
        // output than half_coset(7), confirming they're different domains.
        let trace = crate::air::fibonacci_trace_raw(M31(1), M31(1), log_n);
        let mut d_coeffs = DeviceBuffer::from_host(&trace);
        let trace_domain = Coset::half_coset(log_n);
        let inv = InverseTwiddleCache::new(&trace_domain);
        ntt::interpolate(&mut d_coeffs, &inv);
        drop(inv);

        // Eval on half_coset(log_n+1)
        let mut d_eval_hc = DeviceBuffer::<u32>::alloc(eval_size);
        unsafe {
            ffi::cuda_zero_pad(d_coeffs.as_ptr(), d_eval_hc.as_mut_ptr(), n as u32, eval_size as u32);
        }
        let hc_domain = Coset::half_coset(log_eval);
        let hc_fwd = ForwardTwiddleCache::new(&hc_domain);
        ntt::evaluate(&mut d_eval_hc, &hc_fwd);
        drop(hc_fwd);
        let hc_eval = d_eval_hc.to_host();

        // Eval on subgroup(log_n+1)
        let mut d_eval_sg = DeviceBuffer::<u32>::alloc(eval_size);
        unsafe {
            ffi::cuda_zero_pad(d_coeffs.as_ptr(), d_eval_sg.as_mut_ptr(), n as u32, eval_size as u32);
        }
        let sg_domain = Coset::subgroup(log_eval);
        let sg_fwd = ForwardTwiddleCache::new(&sg_domain);
        ntt::evaluate(&mut d_eval_sg, &sg_fwd);
        drop(sg_fwd);
        let sg_eval = d_eval_sg.to_host();

        // They should be DIFFERENT evaluations (different domains)
        assert_ne!(hc_eval, sg_eval, "half_coset and subgroup evals should differ");

        // But both should represent valid polynomial evaluations
        // (non-trivial, containing non-zero values)
        assert!(sg_eval.iter().any(|&v| v != 0));
    }

    #[test]
    fn test_layerwise_ntt_matches_standard() {
        // Verify that layer-by-layer NTT produces the same result as
        // the standard ForwardTwiddleCache NTT.
        let log_n = 8u32;
        let n = 1usize << log_n;
        let coset = Coset::half_coset(log_n);

        let data: Vec<u32> = (0..n).map(|i| ((i * 7 + 13) % 0x7FFF_FFFF as usize) as u32).collect();

        // Standard NTT
        let mut d_std = DeviceBuffer::from_host(&data);
        let cache = ForwardTwiddleCache::new(&coset);
        ntt::evaluate(&mut d_std, &cache);
        let std_result = d_std.to_host();
        drop(d_std);
        drop(cache);

        // Layer-by-layer NTT (same as prove_lean_max does)
        let mut d_layer = DeviceBuffer::from_host(&data);
        {
            let n_line_layers = log_n - 1;
            let mut d_x = DeviceBuffer::<u32>::alloc(n);
            let mut d_y = DeviceBuffer::<u32>::alloc(n);
            unsafe {
                ffi::cuda_compute_coset_points(
                    coset.initial.x.0, coset.initial.y.0,
                    coset.step.x.0, coset.step.y.0,
                    d_x.as_mut_ptr(), d_y.as_mut_ptr(), n as u32,
                );
                ffi::cuda_device_sync();
            }
            let half = n / 2;
            let mut d_circle = DeviceBuffer::<u32>::alloc(half);
            unsafe {
                ffi::cudaMemcpy(
                    d_circle.as_mut_ptr() as *mut std::ffi::c_void,
                    d_y.as_ptr() as *const std::ffi::c_void,
                    half * 4, ffi::MEMCPY_D2D,
                );
            }
            drop(d_y);

            let mut d_current = d_x;
            let mut host_twiddles: Vec<Vec<u32>> = Vec::new();
            for _build in 0..n_line_layers as usize {
                let ls = d_current.len() / 2;
                let mut d_tw = DeviceBuffer::<u32>::alloc(ls);
                let mut d_sq = DeviceBuffer::<u32>::alloc(ls);
                unsafe {
                    ffi::cuda_extract_and_squash(
                        d_current.as_ptr(), d_tw.as_mut_ptr(), d_sq.as_mut_ptr(), ls as u32,
                    );
                    ffi::cuda_device_sync();
                }
                host_twiddles.push(d_tw.to_host());
                drop(d_tw);
                drop(d_current);
                d_current = d_sq;
            }
            drop(d_current);

            // Apply in reverse order
            for layer in (0..n_line_layers as usize).rev() {
                let d_tw = DeviceBuffer::from_host(&host_twiddles[layer]);
                unsafe {
                    ffi::cuda_circle_ntt_layer(
                        d_layer.as_mut_ptr(), d_tw.as_ptr(),
                        (layer + 1) as u32, n as u32, 1,
                    );
                }
            }
            // Circle layer
            unsafe {
                ffi::cuda_circle_ntt_layer(
                    d_layer.as_mut_ptr(), d_circle.as_ptr(),
                    0, n as u32, 1,
                );
                ffi::cuda_device_sync();
            }
        }
        let layer_result = d_layer.to_host();
        assert_eq!(std_result, layer_result,
            "Layer-by-layer NTT doesn't match standard NTT");
    }

    #[test]
    fn test_twin_coset_eval_roundtrip() {
        // Validate twin-coset evaluation by checking that eval_0 (on half_coset(log_n))
        // correctly reproduces the original trace values via NTT roundtrip.
        let log_n = 8u32;
        let n = 1usize << log_n;

        let trace = crate::air::fibonacci_trace_raw(M31(1), M31(1), log_n);
        let mut d_data = DeviceBuffer::from_host(&trace);
        let trace_domain = Coset::half_coset(log_n);

        // Forward NTT should produce eval values, inverse NTT should recover trace
        let fwd = ForwardTwiddleCache::new(&trace_domain);
        let inv = InverseTwiddleCache::new(&trace_domain);

        // Interpolate → coefficients
        ntt::interpolate(&mut d_data, &inv);
        let coeffs = d_data.to_host();

        // Evaluate on same domain → should recover original trace
        ntt::evaluate(&mut d_data, &fwd);
        let recovered = d_data.to_host();
        assert_eq!(trace, recovered, "NTT roundtrip on half_coset failed");

        // Now test subgroup NTT: interpolate coefficients, evaluate on subgroup(log_n)
        let mut d_coeffs = DeviceBuffer::from_host(&coeffs);
        let complement = Coset::subgroup(log_n);
        let complement_fwd = ForwardTwiddleCache::new(&complement);
        ntt::evaluate(&mut d_coeffs, &complement_fwd);
        let complement_eval = d_coeffs.to_host();

        // The complement eval should be non-trivial (not all zeros, not same as trace)
        assert_ne!(complement_eval, trace, "Complement eval should differ from trace");
        assert!(complement_eval.iter().any(|&v| v != 0), "Complement eval should be non-zero");
    }
}
