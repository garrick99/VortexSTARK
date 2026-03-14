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
pub struct QueryDecommitment<T> {
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
    prove_lean_inner(a, b, log_n, false)
}

/// Generate a STARK proof with lazy VRAM management and timing output.
pub fn prove_lean_timed(a: M31, b: M31, log_n: u32) -> StarkProof {
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
    let mut fri_trees: Vec<MerkleTree> = Vec::new();
    let mut fri_evals: Vec<SecureColumn> = Vec::new();
    let mut current_eval = quotient_col;
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
    let n_tiles = n / TILE_SIZE;

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
}
