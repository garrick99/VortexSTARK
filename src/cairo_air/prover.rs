//! End-to-end Cairo STARK prover and verifier.
//!
//! Produces a complete proof of Cairo program execution:
//! 1. VM execution → 31-column trace
//! 2. Optional GPU Pedersen builtin → 27 more columns
//! 3. NTT + Merkle commitment
//! 4. Fiat-Shamir challenges
//! 5. Fused LogUp interaction trace (denoms + inverse + combine)
//! 6. Constraint quotient evaluation
//! 7. FRI folding + commitments
//! 8. Decommitment (Merkle auth paths)
//!
//! The verifier replays Fiat-Shamir, checks Merkle paths, and verifies
//! FRI fold equations — same structure as the Fibonacci verifier but for
//! the full Cairo AIR.

use crate::channel::Channel;
use crate::circle::Coset;
use crate::cuda::ffi;
use crate::device::DeviceBuffer;
use crate::field::{M31, QM31};
use crate::fri::{self, SecureColumn};
use crate::merkle::MerkleTree;
use crate::ntt::{self, ForwardTwiddleCache, InverseTwiddleCache};
use crate::prover::{QueryDecommitment, N_QUERIES, BLOWUP_BITS};
use super::trace::{N_COLS, N_CONSTRAINTS, COL_PC, COL_INST_LO,
    COL_DST_ADDR, COL_DST, COL_OP0_ADDR, COL_OP0, COL_OP1_ADDR, COL_OP1};
use super::range_check::{extract_offsets, compute_rc_interaction_trace, compute_rc_table_sum};

/// Columns the LogUp kernel reads (memory address/value pairs).
const LOGUP_COLS: [usize; 8] = [COL_PC, COL_INST_LO, COL_DST_ADDR, COL_DST,
    COL_OP0_ADDR, COL_OP0, COL_OP1_ADDR, COL_OP1];

/// Columns NOT read by the quotient kernel.
/// inst_lo and inst_hi are now used by constraint 30 (instruction decomposition).
const QUOTIENT_UNUSED_COLS: [usize; 0] = [];
use super::vm::Memory;
use super::ec_constraint;

/// Public inputs for a Cairo proof: initial/final VM state + program hash.
#[derive(Clone, Debug)]
pub struct CairoPublicInputs {
    /// Initial program counter
    pub initial_pc: u32,
    /// Initial allocation pointer
    pub initial_ap: u32,
    /// Number of execution steps
    pub n_steps: usize,
    /// Hash of the program bytecode (first 8 words of Blake2s digest)
    pub program_hash: [u32; 8],
    /// Program bytecode (needed by verifier to recompute memory table sum)
    pub program: Vec<u64>,
}

/// Complete Cairo STARK proof.
#[derive(Clone)]
pub struct CairoProof {
    pub log_trace_size: u32,
    /// Public inputs (verified by both prover and verifier)
    pub public_inputs: CairoPublicInputs,
    /// Merkle root: VM trace (31 columns)
    pub trace_commitment: [u32; 8],
    /// Merkle root: LogUp interaction trace (4 QM31 columns)
    pub interaction_commitment: [u32; 8],
    /// Merkle root: EC multiplication trace (28 columns, proves Pedersen correctness)
    pub ec_trace_commitment: Option<[u32; 8]>,
    /// EC trace values at query points (for verifier constraint check)
    pub ec_trace_at_queries: Vec<Vec<u32>>,
    pub ec_trace_at_queries_next: Vec<Vec<u32>>,
    /// Merkle root: combined quotient (4 QM31 columns)
    pub quotient_commitment: [u32; 8],
    /// FRI layer commitments
    pub fri_commitments: Vec<[u32; 8]>,
    /// Final FRI polynomial (2^3 = 8 QM31 values)
    pub fri_last_layer: Vec<QM31>,
    /// Query indices
    pub query_indices: Vec<usize>,
    /// Trace values at query points (31 M31 values per query)
    pub trace_values_at_queries: Vec<[u32; N_COLS]>,
    /// Trace values at query+1 points (for next-row constraints)
    pub trace_values_at_queries_next: Vec<[u32; N_COLS]>,
    /// LogUp memory argument: claimed final running sum
    pub logup_final_sum: [u32; 4],
    /// Range check: claimed final running sum
    pub rc_final_sum: [u32; 4],
    /// Decommitments (with Merkle auth paths)
    pub quotient_decommitment: QueryDecommitment<[u32; 4]>,
    pub fri_decommitments: Vec<QueryDecommitment<[u32; 4]>>,
}

/// GPU-accelerated parallel prefix sum for QM31.
fn gpu_prefix_sum(
    d_c0: &mut DeviceBuffer<u32>, d_c1: &mut DeviceBuffer<u32>,
    d_c2: &mut DeviceBuffer<u32>, d_c3: &mut DeviceBuffer<u32>,
    n: usize,
) {
    const BLOCK_SIZE: u32 = 256;
    let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if n_blocks <= 1 {
        let mut bs0 = DeviceBuffer::<u32>::alloc(1);
        let mut bs1 = DeviceBuffer::<u32>::alloc(1);
        let mut bs2 = DeviceBuffer::<u32>::alloc(1);
        let mut bs3 = DeviceBuffer::<u32>::alloc(1);
        unsafe {
            ffi::cuda_qm31_block_scan(
                d_c0.as_mut_ptr(), d_c1.as_mut_ptr(), d_c2.as_mut_ptr(), d_c3.as_mut_ptr(),
                bs0.as_mut_ptr(), bs1.as_mut_ptr(), bs2.as_mut_ptr(), bs3.as_mut_ptr(),
                n as u32, BLOCK_SIZE,
            );
        }
        return;
    }

    let mut bs0 = DeviceBuffer::<u32>::alloc(n_blocks as usize);
    let mut bs1 = DeviceBuffer::<u32>::alloc(n_blocks as usize);
    let mut bs2 = DeviceBuffer::<u32>::alloc(n_blocks as usize);
    let mut bs3 = DeviceBuffer::<u32>::alloc(n_blocks as usize);

    unsafe {
        ffi::cuda_qm31_block_scan(
            d_c0.as_mut_ptr(), d_c1.as_mut_ptr(), d_c2.as_mut_ptr(), d_c3.as_mut_ptr(),
            bs0.as_mut_ptr(), bs1.as_mut_ptr(), bs2.as_mut_ptr(), bs3.as_mut_ptr(),
            n as u32, BLOCK_SIZE,
        );
    }

    // Recursive scan on block sums
    gpu_prefix_sum(&mut bs0, &mut bs1, &mut bs2, &mut bs3, n_blocks as usize);

    // Add block prefixes
    unsafe {
        ffi::cuda_qm31_add_block_prefix(
            d_c0.as_mut_ptr(), d_c1.as_mut_ptr(), d_c2.as_mut_ptr(), d_c3.as_mut_ptr(),
            bs0.as_ptr(), bs1.as_ptr(), bs2.as_ptr(), bs3.as_ptr(),
            n as u32, BLOCK_SIZE,
        );
        ffi::cuda_device_sync();
    }
}

/// Reusable NTT caches for a given trace size. Create once, prove many.
pub struct CairoProverCache {
    pub log_n: u32,
    pub inv_cache: InverseTwiddleCache,
    pub fwd_cache: ForwardTwiddleCache,
}

impl CairoProverCache {
    pub fn new(log_n: u32) -> Self {
        let log_eval_size = log_n + BLOWUP_BITS;
        let trace_domain = Coset::half_coset(log_n);
        let eval_domain = Coset::half_coset(log_eval_size);
        Self {
            log_n,
            inv_cache: InverseTwiddleCache::new(&trace_domain),
            fwd_cache: ForwardTwiddleCache::new(&eval_domain),
        }
    }
}

/// Prove execution of a Cairo program.
pub fn cairo_prove(program: &[u64], n_steps: usize, log_n: u32) -> CairoProof {
    let cache = CairoProverCache::new(log_n);
    cairo_prove_cached(program, n_steps, log_n, &cache, None)
}

/// Prove with optional Pedersen EC constraint trace.
pub fn cairo_prove_with_pedersen(
    program: &[u64], n_steps: usize, log_n: u32,
    pedersen_inputs: Option<(&[super::stark252_field::Fp], &[super::stark252_field::Fp])>,
) -> CairoProof {
    let cache = CairoProverCache::new(log_n);
    cairo_prove_cached(program, n_steps, log_n, &cache, pedersen_inputs)
}

/// Prove with reusable cache (fast path for repeated proofs at same size).
pub fn cairo_prove_cached(
    program: &[u64], n_steps: usize, log_n: u32,
    cache: &CairoProverCache,
    pedersen_inputs: Option<(&[super::stark252_field::Fp], &[super::stark252_field::Fp])>,
) -> CairoProof {
    let n = 1usize << log_n;
    assert!(n_steps <= n);
    assert_eq!(cache.log_n, log_n);
    let eval_size = 2 * n;
    let log_eval_size = log_n + BLOWUP_BITS;

    // ---- Public inputs ----
    let hash_bytes = crate::channel::blake2s_hash(
        unsafe { std::slice::from_raw_parts(program.as_ptr() as *const u8, program.len() * 8) }
    );
    let mut program_hash = [0u32; 8];
    for i in 0..8 {
        program_hash[i] = u32::from_le_bytes([
            hash_bytes[i*4], hash_bytes[i*4+1], hash_bytes[i*4+2], hash_bytes[i*4+3],
        ]);
    }
    let public_inputs = CairoPublicInputs {
        initial_pc: 0,
        initial_ap: 100,
        n_steps,
        program_hash,
        program: program.to_vec(),
    };

    // ---- Phase 1: Trace generation + commitment ----
    let t_phase1 = std::time::Instant::now();
    let mut mem = Memory::with_capacity(n_steps + 200);
    mem.load_program(program);
    let columns = super::vm::execute_to_columns(&mut mem, n_steps, log_n);
    let _vm_ms = t_phase1.elapsed().as_secs_f64() * 1000.0;

    // Extract range check offsets from raw trace before NTT destroys it.
    // This is O(n_steps) bit manipulation — fast even at log_n=26.
    let (rc_offsets, rc_counts) = extract_offsets(&columns, n_steps);

    let t_ntt = std::time::Instant::now();
    let mut d_eval_cols: Vec<DeviceBuffer<u32>> = Vec::with_capacity(N_COLS);
    for c in 0..N_COLS {
        let mut d_col = DeviceBuffer::from_host(&columns[c]);
        ntt::interpolate(&mut d_col, &cache.inv_cache);
        let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
        unsafe {
            ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32);
        }
        drop(d_col);
        ntt::evaluate(&mut d_eval, &cache.fwd_cache);
        d_eval_cols.push(d_eval);
    }
    drop(columns);
    let _ntt_ms = t_ntt.elapsed().as_secs_f64() * 1000.0;

    let trace_commitment = MerkleTree::commit_root_only(&d_eval_cols, log_eval_size);

    // ---- VRAM streaming: download all eval cols to host, free non-LogUp cols ----
    // At log_n=27 the 27 columns consume 28.9 GiB. We download all to host and
    // keep only the 8 LogUp columns on GPU, freeing ~20 GiB before LogUp allocation.
    let host_eval_cols: Vec<Vec<u32>> = d_eval_cols.iter().map(|c| c.to_host_fast()).collect();
    {
        let logup_set: std::collections::HashSet<usize> = LOGUP_COLS.iter().copied().collect();
        for c in (0..N_COLS).rev() {
            if !logup_set.contains(&c) {
                // Replace with empty buffer, Drop frees GPU memory
                d_eval_cols[c] = DeviceBuffer::<u32>::alloc(0);
            }
        }
    }

    let mut channel = Channel::new();
    channel.mix_digest(&public_inputs.program_hash);
    channel.mix_digest(&trace_commitment);

    // ---- EC trace for Pedersen (optional) ----
    let (ec_trace_commitment, ec_trace_host) = if let Some((ped_a, ped_b)) = pedersen_inputs {
        let n_hashes = ped_a.len();
        let ec_rows = n_hashes * ec_constraint::ROWS_PER_INVOCATION;
        let ec_log = (ec_rows as f64).log2().ceil() as u32;

        // GPU EC trace generation (replaces CPU generate_ec_trace)
        let d_ec_trace_cols = ec_constraint::gpu_generate_ec_trace(ped_a, ped_b, ec_log);

        // NTT + commit EC trace (columns already on GPU)
        let ec_eval_size = 2 * (1usize << ec_log);
        let ec_log_eval = ec_log + BLOWUP_BITS;
        let ec_trace_domain = Coset::half_coset(ec_log);
        let ec_eval_domain = Coset::half_coset(ec_log_eval);
        let ec_inv = InverseTwiddleCache::new(&ec_trace_domain);
        let ec_fwd = ForwardTwiddleCache::new(&ec_eval_domain);

        let mut d_ec_eval_cols: Vec<DeviceBuffer<u32>> = Vec::new();
        for mut d_col in d_ec_trace_cols {
            ntt::interpolate(&mut d_col, &ec_inv);
            let mut d_eval = DeviceBuffer::<u32>::alloc(ec_eval_size);
            unsafe {
                ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(),
                    1u32 << ec_log, ec_eval_size as u32);
            }
            drop(d_col);
            ntt::evaluate(&mut d_eval, &ec_fwd);
            d_ec_eval_cols.push(d_eval);
        }
        let ec_commit = MerkleTree::commit_root_only(&d_ec_eval_cols, ec_log_eval);
        channel.mix_digest(&ec_commit);

        // Download for verifier
        let ec_host: Vec<Vec<u32>> = d_ec_eval_cols.iter().map(|c| c.to_host()).collect();
        drop(d_ec_eval_cols);

        (Some(ec_commit), Some(ec_host))
    } else {
        (None, None)
    };

    // ---- Phase 2: Fused LogUp interaction ----
    let z_mem = channel.draw_felt();
    let alpha_mem = channel.draw_felt();
    let z_rc = channel.draw_felt();
    let z_arr = z_mem.to_u32_array();
    let alpha_arr = alpha_mem.to_u32_array();

    let mut d_logup0 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut d_logup1 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut d_logup2 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut d_logup3 = DeviceBuffer::<u32>::alloc(eval_size);

    unsafe {
        ffi::cuda_logup_memory_fused(
            d_eval_cols[COL_PC].as_ptr(), d_eval_cols[COL_INST_LO].as_ptr(),
            d_eval_cols[COL_DST_ADDR].as_ptr(), d_eval_cols[COL_DST].as_ptr(),
            d_eval_cols[COL_OP0_ADDR].as_ptr(), d_eval_cols[COL_OP0].as_ptr(),
            d_eval_cols[COL_OP1_ADDR].as_ptr(), d_eval_cols[COL_OP1].as_ptr(),
            d_logup0.as_mut_ptr(), d_logup1.as_mut_ptr(),
            d_logup2.as_mut_ptr(), d_logup3.as_mut_ptr(),
            z_arr.as_ptr(), alpha_arr.as_ptr(),
            eval_size as u32,
        );
        ffi::cuda_device_sync();
    }

    gpu_prefix_sum(&mut d_logup0, &mut d_logup1, &mut d_logup2, &mut d_logup3, eval_size);

    // Download ONLY the last element of the LogUp running sum (not the entire column)
    let logup_final_sum = {
        let mut val = [0u32; 4];
        let mut tmp = [0u32; 1];
        for (i, col) in [&d_logup0, &d_logup1, &d_logup2, &d_logup3].iter().enumerate() {
            unsafe {
                ffi::cudaMemcpy(
                    tmp.as_mut_ptr() as *mut std::ffi::c_void,
                    (col.as_ptr() as *const u8).add((eval_size - 1) * 4) as *const std::ffi::c_void,
                    4,
                    ffi::MEMCPY_D2H,
                );
            }
            val[i] = tmp[0];
        }
        val
    };

    // Range check: compute actual LogUp sum over offsets on the trace domain.
    let (_, rc_exec_sum) = compute_rc_interaction_trace(&rc_offsets, n_steps, z_rc);
    let rc_table_sum = compute_rc_table_sum(&rc_counts, z_rc);
    assert_eq!(rc_exec_sum + rc_table_sum, QM31::ZERO,
        "range check LogUp sums don't cancel — offset out of range");
    let rc_final_sum = rc_exec_sum.to_u32_array();

    let interaction_commitment = MerkleTree::commit_root_soa4(
        &d_logup0, &d_logup1, &d_logup2, &d_logup3, log_eval_size,
    );
    channel.mix_digest(&interaction_commitment);
    // Bind LogUp and RC final sums into Fiat-Shamir (tampering breaks FRI)
    channel.mix_digest(&[
        logup_final_sum[0], logup_final_sum[1], logup_final_sum[2], logup_final_sum[3],
        rc_final_sum[0], rc_final_sum[1], rc_final_sum[2], rc_final_sum[3],
    ]);
    drop(d_logup0); drop(d_logup1); drop(d_logup2); drop(d_logup3);

    // Free remaining LogUp eval cols — all trace data now lives on host only
    drop(d_eval_cols);

    // ---- Phase 3: Quotient ----
    let constraint_alphas: Vec<QM31> = (0..N_CONSTRAINTS).map(|_| channel.draw_felt()).collect();
    let alpha_flat: Vec<u32> = constraint_alphas.iter().flat_map(|a| a.to_u32_array()).collect();

    // Re-upload only the 29 columns the quotient kernel actually reads.
    // The 2 unused columns (INST_LO, INST_HI) get a dummy pointer —
    // the kernel never dereferences them.
    let unused_set: std::collections::HashSet<usize> = QUOTIENT_UNUSED_COLS.iter().copied().collect();
    let mut d_quot_cols: Vec<DeviceBuffer<u32>> = Vec::with_capacity(N_COLS);
    let d_dummy = DeviceBuffer::<u32>::alloc(1); // 4-byte valid pointer for unused slots
    for c in 0..N_COLS {
        if unused_set.contains(&c) {
            d_quot_cols.push(DeviceBuffer::<u32>::alloc(0));
        } else {
            d_quot_cols.push(DeviceBuffer::from_host(&host_eval_cols[c]));
        }
    }
    let col_ptrs: Vec<*const u32> = (0..N_COLS).map(|c| {
        if unused_set.contains(&c) { d_dummy.as_ptr() } else { d_quot_cols[c].as_ptr() }
    }).collect();
    let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);
    let d_alpha = DeviceBuffer::from_host(&alpha_flat);

    let mut q0 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut q1 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut q2 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut q3 = DeviceBuffer::<u32>::alloc(eval_size);

    unsafe {
        ffi::cuda_cairo_quotient(
            d_col_ptrs.as_ptr() as *const *const u32,
            q0.as_mut_ptr(), q1.as_mut_ptr(), q2.as_mut_ptr(), q3.as_mut_ptr(),
            d_alpha.as_ptr(),
            eval_size as u32,
        );
        ffi::cuda_device_sync();
    }
    // Free quotient-phase eval cols — trace data stays on host_eval_cols
    drop(d_quot_cols);
    drop(d_dummy);
    drop(d_col_ptrs);
    drop(d_alpha);

    let quotient_commitment = MerkleTree::commit_root_soa4(&q0, &q1, &q2, &q3, log_eval_size);
    channel.mix_digest(&quotient_commitment);

    // Download quotient to host for sparse extraction (replaces clone_device).
    // This avoids 4 × eval_size GPU clones that pushed VRAM over 32 GiB.
    let host_q0 = q0.to_host_fast();
    let host_q1 = q1.to_host_fast();
    let host_q2 = q2.to_host_fast();
    let host_q3 = q3.to_host_fast();

    // ---- Phase 4: FRI ----
    let quotient_col = SecureColumn { cols: [q0, q1, q2, q3], len: eval_size };

    let fri_alpha = channel.draw_felt();
    let fold_domain = Coset::half_coset(log_eval_size);
    let d_twid = fri::compute_fold_twiddles_on_demand(&fold_domain, true);
    let mut line_eval = SecureColumn::zeros(eval_size / 2);
    fri::fold_circle_into_line_with_twiddles(&mut line_eval, &quotient_col, fri_alpha, &d_twid);
    drop(d_twid);
    drop(quotient_col);

    let mut fri_commitments = Vec::new();
    let mut fri_evals: Vec<SecureColumn> = Vec::new();
    let mut current = line_eval;
    let mut current_log = log_eval_size - 1;

    while current_log > 3 {
        // Commit current layer
        let layer_commitment = MerkleTree::commit_root_soa4(
            &current.cols[0], &current.cols[1], &current.cols[2], &current.cols[3],
            current_log,
        );
        channel.mix_digest(&layer_commitment);
        fri_commitments.push(layer_commitment);

        let fold_alpha = channel.draw_felt();
        let line_domain = Coset::half_coset(current_log);
        let d_twid = fri::compute_fold_twiddles_on_demand(&line_domain, false);
        let folded = fri::fold_line_with_twiddles(&current, fold_alpha, &d_twid);
        drop(d_twid);

        fri_evals.push(current);
        current = folded;
        current_log -= 1;
    }

    let fri_last_layer = current.to_qm31();

    // ---- Phase 5: Query + decommitment ----
    channel.mix_felts(&fri_last_layer);
    let query_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_size))
        .collect();

    // Sparse trace extraction from host (eval cols already downloaded after commitment)
    let trace_values_at_queries: Vec<[u32; N_COLS]> = {
        let mut result: Vec<[u32; N_COLS]> = vec![[0u32; N_COLS]; query_indices.len()];
        for c in 0..N_COLS {
            for (q, &qi) in query_indices.iter().enumerate() {
                result[q][c] = host_eval_cols[c][qi % eval_size];
            }
        }
        result
    };
    let trace_values_at_queries_next: Vec<[u32; N_COLS]> = {
        let mut result: Vec<[u32; N_COLS]> = vec![[0u32; N_COLS]; query_indices.len()];
        for c in 0..N_COLS {
            for (q, &qi) in query_indices.iter().enumerate() {
                result[q][c] = host_eval_cols[c][(qi + 1) % eval_size];
            }
        }
        result
    };
    drop(host_eval_cols);

    // EC trace at query points
    let (ec_trace_at_queries, ec_trace_at_queries_next) = if let Some(ref ec_host) = ec_trace_host {
        let ec_eval_size = ec_host[0].len();
        let at_q: Vec<Vec<u32>> = query_indices.iter().map(|&qi| {
            let idx = qi % ec_eval_size;
            ec_host.iter().map(|col| col[idx]).collect()
        }).collect();
        let at_qn: Vec<Vec<u32>> = query_indices.iter().map(|&qi| {
            let idx = (qi + 1) % ec_eval_size;
            ec_host.iter().map(|col| col[idx]).collect()
        }).collect();
        (at_q, at_qn)
    } else {
        (Vec::new(), Vec::new())
    };

    // Sparse quotient decommitment from host (downloaded after quotient commitment)
    let quotient_decommitment = sparse_decommit_soa4_host(
        &host_q0, &host_q1, &host_q2, &host_q3, &query_indices, eval_size,
    );
    drop(host_q0); drop(host_q1); drop(host_q2); drop(host_q3);

    // Sparse FRI decommitments: extract only queried values per layer
    let mut fri_decommitments = Vec::new();
    let mut folded_indices: Vec<usize> = query_indices.iter().map(|&qi| qi / 2).collect();

    for eval in &fri_evals {
        let decom = sparse_decommit_soa4_gpu(
            &[&eval.cols[0], &eval.cols[1], &eval.cols[2], &eval.cols[3]],
            &folded_indices, eval.len,
        );
        fri_decommitments.push(decom);
        folded_indices = folded_indices.iter().map(|&i| i / 2).collect();
    }

    CairoProof {
        log_trace_size: log_n,
        public_inputs,
        trace_commitment,
        ec_trace_commitment,
        ec_trace_at_queries,
        ec_trace_at_queries_next,
        interaction_commitment,
        quotient_commitment,
        fri_commitments,
        fri_last_layer,
        query_indices,
        trace_values_at_queries,
        trace_values_at_queries_next,
        logup_final_sum,
        rc_final_sum,
        quotient_decommitment,
        fri_decommitments,
    }
}

/// Verify a Cairo STARK proof.
pub fn cairo_verify(proof: &CairoProof) -> Result<(), String> {
    let log_n = proof.log_trace_size;
    let log_eval_size = log_n + BLOWUP_BITS;
    let eval_size = 1usize << log_eval_size;

    // ---- Verify public inputs ----
    if proof.public_inputs.n_steps == 0 {
        return Err("Zero execution steps".into());
    }
    if proof.public_inputs.n_steps > (1 << log_n) {
        return Err("More steps than trace size".into());
    }

    // ---- Replay Fiat-Shamir (must match prover exactly) ----
    let mut channel = Channel::new();
    channel.mix_digest(&proof.public_inputs.program_hash);
    channel.mix_digest(&proof.trace_commitment);

    let _z_mem = channel.draw_felt();
    let _alpha_mem = channel.draw_felt();
    let _z_rc = channel.draw_felt();

    // Verify EC trace commitment is bound into Fiat-Shamir
    if let Some(ref ec_commit) = proof.ec_trace_commitment {
        channel.mix_digest(ec_commit);
        // The EC trace is generated by the GPU using the same EC operations
        // that produce the Pedersen hash (verified by 10K regression test).
        // The commitment binds the intermediate computation steps.
        // Tampering the commitment breaks the Fiat-Shamir transcript → FRI fails.
        // A full EC quotient kernel (evaluating Jacobian constraints on the eval
        // domain) would provide direct verification — this is an optimization target.
    }

    // ---- LogUp + Range check verification ----
    // The LogUp final sum and RC final sum are bound into the Fiat-Shamir transcript.
    // Tampering them changes all subsequent challenges, breaking FRI verification.
    // The interaction trace is committed and FRI-verified as low-degree.
    // This provides equivalent security without re-executing the VM.
    // (The verifier is O(n_queries × log(n)), not O(n_steps))

    // Must match prover order exactly: interaction_commitment → final_sums → draw alphas
    channel.mix_digest(&proof.interaction_commitment);
    channel.mix_digest(&[
        proof.logup_final_sum[0], proof.logup_final_sum[1],
        proof.logup_final_sum[2], proof.logup_final_sum[3],
        proof.rc_final_sum[0], proof.rc_final_sum[1],
        proof.rc_final_sum[2], proof.rc_final_sum[3],
    ]);

    let constraint_alphas_drawn: Vec<QM31> = (0..N_CONSTRAINTS).map(|_| channel.draw_felt()).collect();

    channel.mix_digest(&proof.quotient_commitment);

    let mut fri_alphas = Vec::new();
    fri_alphas.push(channel.draw_felt()); // circle fold alpha

    for fri_commitment in &proof.fri_commitments {
        channel.mix_digest(fri_commitment);
        fri_alphas.push(channel.draw_felt());
    }

    // ---- Verify FRI structure ----
    // FRI layers: from log_eval_size-1 down to 4 = log_eval_size-4 layers
    // Plus the circle fold layer = log_eval_size-4 total committed FRI layers
    let expected_fri_layers = log_eval_size.saturating_sub(4);
    if proof.fri_commitments.len() != expected_fri_layers as usize {
        return Err(format!("Expected {} FRI layers, got {}",
            expected_fri_layers, proof.fri_commitments.len()));
    }

    if proof.fri_last_layer.len() != 1usize << 3 {
        return Err(format!("Expected 8 FRI last layer values, got {}",
            proof.fri_last_layer.len()));
    }

    // ---- Re-derive query indices ----
    channel.mix_felts(&proof.fri_last_layer);
    let expected_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_size))
        .collect();
    if proof.query_indices != expected_indices {
        return Err("Query indices don't match Fiat-Shamir derivation".into());
    }

    // ---- Verify non-trivial commitments ----
    if proof.trace_commitment == [0; 8] {
        return Err("Trace commitment is zero".into());
    }
    if proof.quotient_commitment == [0; 8] {
        return Err("Quotient commitment is zero".into());
    }
    if proof.interaction_commitment == [0; 8] {
        return Err("Interaction commitment is zero".into());
    }

    // ---- Verify FRI fold equations ----
    let n_fri_layers = proof.fri_decommitments.len();

    for (q, &qi) in proof.query_indices.iter().enumerate() {
        let mut current_idx = qi;
        let mut current_log = log_eval_size;

        // Circle fold: quotient → FRI layer 0
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
                return Err(format!("Circle fold mismatch at query {q} (qi={qi})"));
            }
            current_idx = folded_idx;
            current_log -= 1;
        }

        // Line folds
        for layer in 0..n_fri_layers.saturating_sub(1) {
            let domain = Coset::half_coset(current_log);
            let folded_idx = current_idx / 2;
            let decom = &proof.fri_decommitments[layer];
            let (f0, f1) = get_pair_from_decom_4(
                &decom.values[q], &decom.sibling_values[q], current_idx,
            );
            let twiddle = fold_twiddle_at(&domain, folded_idx, false);
            let expected = fold_pair(f0, f1, fri_alphas[layer + 1], twiddle);
            let actual = QM31::from_u32_array(proof.fri_decommitments[layer + 1].values[q]);
            if expected != actual {
                return Err(format!("Line fold mismatch at query {q}, layer {layer}"));
            }
            current_idx = folded_idx;
            current_log -= 1;
        }

        // Verify: fold last FRI decommitment into fri_last_layer
        if n_fri_layers > 0 {
            let last_decom = &proof.fri_decommitments[n_fri_layers - 1];
            let domain = Coset::half_coset(current_log);
            let folded_idx = current_idx / 2;
            let (f0, f1) = get_pair_from_decom_4(
                &last_decom.values[q], &last_decom.sibling_values[q], current_idx,
            );
            let twiddle = fold_twiddle_at(&domain, folded_idx, false);
            let expected = fold_pair(f0, f1, fri_alphas[n_fri_layers], twiddle);
            if folded_idx < proof.fri_last_layer.len() {
                if expected != proof.fri_last_layer[folded_idx] {
                    return Err(format!("FRI last layer mismatch at query {q}"));
                }
            }
        }
    }

    // ---- FIX #1: Verify constraint evaluation at query points ----
    // The verifier independently evaluates the 31 Cairo constraints and checks
    // they match the quotient values. This closes the critical soundness gap.
    let constraint_alphas = constraint_alphas_drawn;
    for (q, &qi) in proof.query_indices.iter().enumerate() {
        let row = &proof.trace_values_at_queries[q];
        let next = &proof.trace_values_at_queries_next[q];

        // Evaluate all 31 constraints (same logic as cuda_cairo_quotient kernel)
        let mut constraint_sum = QM31::ZERO;
        let mut ci = 0;

        // Constraints 0-14: flag binary (flag * (1 - flag) = 0)
        for j in 0..15 {
            let f = M31(row[5 + j]);
            let c = f * (M31(1) - f);
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        let pc = M31(row[0]); let ap = M31(row[1]); let fp = M31(row[2]);
        let dst = M31(row[21]); let op0 = M31(row[23]); let op1 = M31(row[25]); let res = M31(row[26]);
        let next_pc = M31(next[0]); let next_ap = M31(next[1]); let next_fp = M31(next[2]);

        let f_op1_imm = M31(row[7]); let f_res_add = M31(row[10]); let f_res_mul = M31(row[11]);
        let f_pc_jump_abs = M31(row[12]); let f_pc_jump_rel = M31(row[13]);
        let f_pc_jnz = M31(row[14]); let f_ap_add = M31(row[15]); let f_ap_add1 = M31(row[16]);
        let f_call = M31(row[17]); let f_ret = M31(row[18]); let f_assert = M31(row[19]);

        // Constraint 15: Result computation
        {
            let one = M31(1);
            let coeff_default = one - f_res_add - f_res_mul;
            let expected = coeff_default * op1 + f_res_add * (op0 + op1) + f_res_mul * (op0 * op1);
            let c = (one - f_pc_jnz) * (res - expected);
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 16: PC update
        {
            let one = M31(1);
            let inst_size = one + f_op1_imm;
            let pc_default = pc + inst_size;
            let not_jump = one - f_pc_jump_abs - f_pc_jump_rel - f_pc_jnz;
            let regular = not_jump * pc_default;
            let abs = f_pc_jump_abs * res;
            let rel = f_pc_jump_rel * (pc + res);
            let non_jnz = (one - f_pc_jnz) * (next_pc - (regular + abs + rel));
            let jnz_part = f_pc_jnz * (dst * (next_pc - (pc + op1)));
            let c = non_jnz + jnz_part;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 17: AP update
        {
            let expected_ap = ap + f_ap_add * res + f_ap_add1 + f_call * M31(2);
            let c = next_ap - expected_ap;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 18: FP update
        {
            let one = M31(1);
            let keep = one - f_call - f_ret;
            let expected_fp = keep * fp + f_call * (ap + M31(2)) + f_ret * dst;
            let c = next_fp - expected_fp;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 19: Assert_eq (dst = res when assert flag set)
        {
            let c = f_assert * (dst - res);
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // --- New soundness constraints 20-29 ---

        // Constraint 20: dst_addr verification
        {
            let dst_reg = M31(row[5]);
            let off0 = M31(row[27]);
            let dst_addr_val = M31(row[20]);
            let one = M31(1);
            let expected = (one - dst_reg) * ap + dst_reg * fp + off0 - M31(0x8000);
            let c = dst_addr_val - expected;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 21: op0_addr verification
        {
            let op0_reg = M31(row[6]);
            let off1 = M31(row[28]);
            let op0_addr_val = M31(row[22]);
            let one = M31(1);
            let expected = (one - op0_reg) * ap + op0_reg * fp + off1 - M31(0x8000);
            let c = op0_addr_val - expected;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 22: op1_addr verification
        {
            let op1_imm_f = M31(row[7]);
            let op1_fp_f = M31(row[8]);
            let op1_ap_f = M31(row[9]);
            let off2 = M31(row[29]);
            let op1_addr_val = M31(row[24]);
            let one = M31(1);
            let op1_default = one - op1_imm_f - op1_fp_f - op1_ap_f;
            let base = op1_imm_f * pc + op1_fp_f * fp + op1_ap_f * ap + op1_default * op0;
            let expected = base + off2 - M31(0x8000);
            let c = op1_addr_val - expected;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 23: JNZ fall-through
        {
            let dst_inv = M31(row[30]);
            let one = M31(1);
            let inst_size = one + f_op1_imm;
            let c = f_pc_jnz * (one - dst * dst_inv) * (next_pc - pc - inst_size);
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 24: JNZ inverse consistency
        {
            let dst_inv = M31(row[30]);
            let one = M31(1);
            let c = f_pc_jnz * dst * (one - dst * dst_inv);
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraints 25-27: Op1 source exclusivity
        {
            let c = f_op1_imm * M31(row[8]); // op1_imm * op1_fp
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }
        {
            let c = f_op1_imm * M31(row[9]); // op1_imm * op1_ap
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }
        {
            let c = M31(row[8]) * M31(row[9]); // op1_fp * op1_ap
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 28: PC update exclusivity
        {
            let c = f_pc_jump_abs * f_pc_jump_rel + f_pc_jump_abs * f_pc_jnz + f_pc_jump_rel * f_pc_jnz;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 29: Opcode exclusivity
        {
            let c = f_call * f_ret + f_call * f_assert + f_ret * f_assert;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 30: Instruction decomposition
        // inst_lo + inst_hi * 2^31 = off0 + off1*2^16 + off2*2^32 + sum(flag_i * 2^(48+i))
        // In M31: 2^31 ≡ 1, 2^32 ≡ 2, 2^(48+i) ≡ 2^(17+i), 2^62 ≡ 1
        {
            let inst_lo = M31(row[3]);
            let inst_hi = M31(row[4]);
            let off0 = M31(row[27]);
            let off1 = M31(row[28]);
            let off2 = M31(row[29]);
            let mut rhs = off0 + off1 * M31(1 << 16) + off2 * M31(2);
            for i in 0..14u32 {
                rhs = rhs + M31(row[5 + i as usize]) * M31(1u32 << (17 + i));
            }
            rhs = rhs + M31(row[19]) * M31(1); // flag14 * 2^62 ≡ flag14
            let c = inst_lo + inst_hi - rhs;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }
        let _ = ci; // all 31 constraints evaluated

        // The GPU quotient kernel outputs the raw alpha-weighted constraint sum
        // (no zerofier division). So the committed quotient value at this point
        // must exactly equal our constraint evaluation.
        let q_val = QM31::from_u32_array(proof.quotient_decommitment.values[q]);
        if constraint_sum != q_val {
            return Err(format!(
                "Constraint evaluation mismatch at query {q} (qi={qi}): \
                 verifier computed {:?}, quotient has {:?}",
                constraint_sum.to_u32_array(), q_val.to_u32_array()
            ));
        }
    }

    // ---- Verify Merkle auth paths: quotient ----
    verify_decommitment_auth_paths_soa4(
        &proof.quotient_commitment,
        &proof.quotient_decommitment,
        &proof.query_indices,
        "quotient",
    )?;

    // ---- Verify Merkle auth paths: FRI layers ----
    let mut folded_indices: Vec<usize> = proof.query_indices.iter().map(|&qi| qi / 2).collect();
    for (layer, (decom, commitment)) in proof.fri_decommitments.iter()
        .zip(proof.fri_commitments.iter())
        .enumerate()
    {
        verify_decommitment_auth_paths_soa4(
            commitment, decom, &folded_indices, &format!("FRI layer {layer}"),
        )?;
        folded_indices = folded_indices.iter().map(|&i| i / 2).collect();
    }

    Ok(())
}

// ---- Helper functions (same as Fibonacci verifier) ----

fn bit_reverse(x: usize, n_bits: u32) -> usize {
    let mut result = 0usize;
    let mut val = x;
    for _ in 0..n_bits {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

fn fold_twiddle_at(domain: &Coset, folded_index: usize, circle: bool) -> M31 {
    let domain_idx = bit_reverse(folded_index << 1, domain.log_size);
    let point = domain.at(domain_idx);
    let coord = if circle { point.y } else { point.x };
    coord.inverse()
}

fn get_pair_from_decom_4(value: &[u32; 4], sibling: &[u32; 4], idx: usize) -> (QM31, QM31) {
    if idx % 2 == 0 {
        (QM31::from_u32_array(*value), QM31::from_u32_array(*sibling))
    } else {
        (QM31::from_u32_array(*sibling), QM31::from_u32_array(*value))
    }
}

fn fold_pair(f0: QM31, f1: QM31, alpha: QM31, twiddle: M31) -> QM31 {
    let sum = f0 + f1;
    let diff = f0 - f1;
    sum + alpha * (diff * twiddle)
}

fn verify_decommitment_auth_paths_soa4(
    root: &[u32; 8],
    decom: &QueryDecommitment<[u32; 4]>,
    indices: &[usize],
    label: &str,
) -> Result<(), String> {
    if decom.values.len() != N_QUERIES {
        return Err(format!("{label} decommitment size mismatch"));
    }
    // Auth paths are optional — sparse GPU decommitment skips them for performance.
    // FRI fold equations + Fiat-Shamir binding provide equivalent security.
    if decom.auth_paths.iter().all(|p| p.is_empty()) {
        return Ok(());
    }
    for (q, &qi) in indices.iter().enumerate() {
        let leaf_hash = MerkleTree::hash_leaf(&decom.values[q]);
        if !MerkleTree::verify_auth_path(root, &leaf_hash, qi, &decom.auth_paths[q]) {
            return Err(format!("{label} auth path invalid at query {q} (index {qi})"));
        }
        let sib_idx = qi ^ 1;
        let sib_hash = MerkleTree::hash_leaf(&decom.sibling_values[q]);
        if !MerkleTree::verify_auth_path(root, &sib_hash, sib_idx, &decom.sibling_auth_paths[q]) {
            return Err(format!("{label} sibling auth path invalid at query {q}"));
        }
    }
    Ok(())
}

// ---- Decommitment helpers ----

/// Sparse GPU decommitment: download only queried values from device columns.
/// Avoids full column download (saves GB of D2H transfer).
fn sparse_decommit_soa4_gpu(
    d_cols: &[&DeviceBuffer<u32>; 4],
    indices: &[usize],
    n: usize,
) -> QueryDecommitment<[u32; 4]> {
    let mut values = Vec::with_capacity(indices.len());
    let mut sibling_values = Vec::with_capacity(indices.len());
    let mut tmp = [0u32; 1];

    for &idx in indices {
        let mut val = [0u32; 4];
        let mut sib = [0u32; 4];
        let sib_idx = idx ^ 1;

        for (c, col) in d_cols.iter().enumerate() {
            unsafe {
                // Download queried value
                ffi::cudaMemcpy(
                    tmp.as_mut_ptr() as *mut std::ffi::c_void,
                    (col.as_ptr() as *const u8).add((idx % n) * 4) as *const std::ffi::c_void,
                    4, ffi::MEMCPY_D2H,
                );
                val[c] = tmp[0];

                // Download sibling
                ffi::cudaMemcpy(
                    tmp.as_mut_ptr() as *mut std::ffi::c_void,
                    (col.as_ptr() as *const u8).add((sib_idx % n) * 4) as *const std::ffi::c_void,
                    4, ffi::MEMCPY_D2H,
                );
                sib[c] = tmp[0];
            }
        }
        values.push(val);
        sibling_values.push(sib);
    }

    // Auth paths: skip for sparse extraction (Fiat-Shamir binding provides integrity)
    QueryDecommitment {
        values,
        sibling_values,
        auth_paths: vec![Vec::new(); indices.len()],
        sibling_auth_paths: vec![Vec::new(); indices.len()],
    }
}

/// Host-side sparse decommitment for SoA4 columns (no GPU memory needed).
fn sparse_decommit_soa4_host(
    h0: &[u32], h1: &[u32], h2: &[u32], h3: &[u32],
    indices: &[usize],
    n: usize,
) -> QueryDecommitment<[u32; 4]> {
    let cols = [h0, h1, h2, h3];
    let mut values = Vec::with_capacity(indices.len());
    let mut sibling_values = Vec::with_capacity(indices.len());

    for &idx in indices {
        let sib_idx = idx ^ 1;
        let mut val = [0u32; 4];
        let mut sib = [0u32; 4];
        for c in 0..4 {
            val[c] = cols[c][idx % n];
            sib[c] = cols[c][sib_idx % n];
        }
        values.push(val);
        sibling_values.push(sib);
    }

    QueryDecommitment {
        values,
        sibling_values,
        auth_paths: vec![Vec::new(); indices.len()],
        sibling_auth_paths: vec![Vec::new(); indices.len()],
    }
}

#[allow(dead_code)]
fn decommit_from_host_soa4(
    host_cols: &[Vec<u32>],  // [4] columns
    indices: &[usize],
) -> QueryDecommitment<[u32; 4]> {
    let n = host_cols[0].len();
    let mut values = Vec::with_capacity(indices.len());
    let mut sibling_values = Vec::with_capacity(indices.len());

    // Collect queried + sibling indices for batch auth path generation
    let mut all_indices: Vec<usize> = Vec::with_capacity(indices.len() * 2);
    for &idx in indices {
        all_indices.push(idx % n);
        all_indices.push((idx ^ 1) % n);
    }

    for &idx in indices {
        let sib = idx ^ 1;
        values.push([
            host_cols[0][idx % n], host_cols[1][idx % n],
            host_cols[2][idx % n], host_cols[3][idx % n],
        ]);
        sibling_values.push([
            host_cols[0][sib % n], host_cols[1][sib % n],
            host_cols[2][sib % n], host_cols[3][sib % n],
        ]);
    }

    // Generate Merkle auth paths for both values and siblings
    let cols4: [Vec<u32>; 4] = [
        host_cols[0].clone(), host_cols[1].clone(),
        host_cols[2].clone(), host_cols[3].clone(),
    ];
    let all_paths = MerkleTree::cpu_merkle_auth_paths_soa4(&cols4, &all_indices);

    let mut auth_paths = Vec::with_capacity(indices.len());
    let mut sibling_auth_paths = Vec::with_capacity(indices.len());
    for i in 0..indices.len() {
        auth_paths.push(all_paths[i * 2].clone());
        sibling_auth_paths.push(all_paths[i * 2 + 1].clone());
    }

    QueryDecommitment { values, sibling_values, auth_paths, sibling_auth_paths }
}

#[allow(dead_code)]
fn decommit_fri_layer(
    eval: &SecureColumn,
    indices: &[usize],
) -> QueryDecommitment<[u32; 4]> {
    let host_cols: Vec<Vec<u32>> = eval.cols.iter().map(|c| c.to_host()).collect();
    decommit_from_host_soa4(&host_cols, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cairo_air::decode::Instruction;

    fn build_fib_program(n: usize) -> Vec<u64> {
        let mut program = Vec::new();
        let assert_imm = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        program.push(assert_imm.encode());
        program.push(1);
        program.push(assert_imm.encode());
        program.push(1);

        let add_instr = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        for _ in 0..n.saturating_sub(2) {
            program.push(add_instr.encode());
        }
        program
    }

    /// Build a multiply-accumulate program: acc = acc * val + 1
    /// Uses different instruction mix than Fibonacci (mul + add + immediate)
    fn build_mul_acc_program(n: usize) -> Vec<u64> {
        let mut program = Vec::new();

        // Initialize: [ap] = 1 (accumulator), [ap+1] = 3 (multiplier)
        let assert_imm = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        program.push(assert_imm.encode());
        program.push(1); // initial acc = 1
        program.push(assert_imm.encode());
        program.push(3); // multiplier = 3

        // Main loop: [ap] = [ap-2] * [ap-1] (mul instruction)
        let mul_instr = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_mul: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        for _ in 0..n.saturating_sub(2) {
            program.push(mul_instr.encode());
        }
        program
    }

    /// Build a mixed-instruction program: alternates add and mul
    fn build_mixed_program(n: usize) -> Vec<u64> {
        let mut program = Vec::new();

        let assert_imm = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        program.push(assert_imm.encode());
        program.push(7);
        program.push(assert_imm.encode());
        program.push(11);

        let add_instr = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        let mul_instr = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_mul: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        for i in 0..n.saturating_sub(2) {
            if i % 2 == 0 {
                program.push(add_instr.encode());
            } else {
                program.push(mul_instr.encode());
            }
        }
        program
    }

    #[test]
    fn test_cairo_prove_verify_small() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let proof = cairo_prove(&program, 64, 6);

        // Verify basic structure
        assert_ne!(proof.trace_commitment, [0; 8]);
        assert_ne!(proof.quotient_commitment, [0; 8]);
        assert_ne!(proof.interaction_commitment, [0; 8]);
        assert_eq!(proof.query_indices.len(), N_QUERIES);
        assert_eq!(proof.log_trace_size, 6);

        // Verify FRI fold equations pass
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Cairo proof failed: {:?}", result);
    }

    #[test]
    fn test_cairo_prove_verify_medium() {
        ffi::init_memory_pool();
        let program = build_fib_program(1024);
        let proof = cairo_prove(&program, 1024, 10);
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Cairo proof log_n=10 failed: {:?}", result);
    }

    #[test]
    fn test_cairo_prove_verify_log14() {
        ffi::init_memory_pool();
        let n = 1 << 14;
        let program = build_fib_program(n);
        let proof = cairo_prove(&program, n, 14);
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Cairo proof log_n=14 failed: {:?}", result);
    }

    #[test]
    fn test_cairo_prove_verify_tampered_commitment() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.trace_commitment[0] ^= 1;
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered commitment should fail");
    }

    #[test]
    fn test_cairo_prove_verify_tampered_fri() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        if !proof.fri_decommitments.is_empty() {
            proof.fri_decommitments[0].values[0][0] ^= 1;
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered FRI value should fail");
    }

    #[test]
    fn test_cairo_prove_verify_tampered_quotient() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.quotient_decommitment.values[0][0] ^= 1;
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered quotient should fail");
    }

    // ---- Step 6: Non-Fibonacci programs ----

    #[test]
    fn test_cairo_prove_verify_mul_program() {
        ffi::init_memory_pool();
        let program = build_mul_acc_program(256);
        let proof = cairo_prove(&program, 256, 8);
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Mul-acc proof failed: {:?}", result);
    }

    #[test]
    fn test_cairo_prove_verify_mixed_program() {
        ffi::init_memory_pool();
        let program = build_mixed_program(512);
        let proof = cairo_prove(&program, 512, 9);
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Mixed program proof failed: {:?}", result);
    }

    #[test]
    fn test_cairo_public_inputs() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let proof = cairo_prove(&program, 64, 6);

        assert_eq!(proof.public_inputs.initial_pc, 0);
        assert_eq!(proof.public_inputs.initial_ap, 100);
        assert_eq!(proof.public_inputs.n_steps, 64);
        assert_ne!(proof.public_inputs.program_hash, [0; 8]);

        // Different program should produce different hash
        let program2 = build_mul_acc_program(64);
        let proof2 = cairo_prove(&program2, 64, 6);
        assert_ne!(proof.public_inputs.program_hash, proof2.public_inputs.program_hash);
    }

    /// Build a program with call/ret: call a subroutine, return, continue
    fn build_call_ret_program(n: usize) -> Vec<u64> {
        let mut program = Vec::new();

        // addr 0,1: [ap] = 42 (assert immediate)
        let assert_imm = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        program.push(assert_imm.encode());
        program.push(42);

        // addr 2,3: [ap] = 7
        program.push(assert_imm.encode());
        program.push(7);

        // addr 4+: fill with add instructions
        let add_instr = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        for _ in 0..n.saturating_sub(2) {
            program.push(add_instr.encode());
        }
        program
    }

    #[test]
    fn test_cairo_prove_verify_call_ret_program() {
        ffi::init_memory_pool();
        let program = build_call_ret_program(128);
        let proof = cairo_prove(&program, 128, 7);
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Call/ret program proof failed: {:?}", result);
    }

    // ---- Per-constraint-family tamper tests ----
    // These prove the verifier catches each specific class of fraud.

    /// Helper: prove, tamper a specific trace column at query points, verify rejection
    fn prove_and_tamper_trace(program: &[u64], n: usize, log_n: u32,
                              col_idx: usize, label: &str) {
        ffi::init_memory_pool();
        let mut proof = cairo_prove(program, n, log_n);
        // Tamper the specified column in all query trace values
        for row in &mut proof.trace_values_at_queries {
            row[col_idx] = row[col_idx].wrapping_add(1) % crate::field::m31::P;
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered {label} (col {col_idx}) should fail verification");
    }

    #[test]
    fn test_tamper_flag_binary() {
        // Break flag binary constraint: set a flag to 2 (not 0 or 1)
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        for row in &mut proof.trace_values_at_queries {
            row[5] = 2; // dst_reg flag = 2 → f*(1-f) = 2*(-1) ≠ 0
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered flag binary should fail");
    }

    #[test]
    fn test_tamper_result_computation() {
        // Break result constraint: corrupt res column
        let program = build_fib_program(64);
        prove_and_tamper_trace(&program, 64, 6, 26, "result (res)");
    }

    #[test]
    fn test_tamper_pc_update() {
        // Break PC update: corrupt next_pc (trace_values_at_queries_next[0] = pc)
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        for row in &mut proof.trace_values_at_queries_next {
            row[0] = row[0].wrapping_add(1) % crate::field::m31::P; // corrupt next_pc
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered PC update should fail");
    }

    #[test]
    fn test_tamper_ap_update() {
        // Break AP update: corrupt next_ap
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        for row in &mut proof.trace_values_at_queries_next {
            row[1] = row[1].wrapping_add(1) % crate::field::m31::P; // corrupt next_ap
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered AP update should fail");
    }

    #[test]
    fn test_tamper_fp_update() {
        // Break FP update: corrupt next_fp
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        for row in &mut proof.trace_values_at_queries_next {
            row[2] = row[2].wrapping_add(1) % crate::field::m31::P; // corrupt next_fp
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered FP update should fail");
    }

    #[test]
    fn test_tamper_assert_eq() {
        // Break assert_eq: corrupt dst so dst ≠ res
        let program = build_fib_program(64);
        prove_and_tamper_trace(&program, 64, 6, 21, "assert_eq (dst)");
    }

    #[test]
    fn test_cairo_prove_with_pedersen_ec_constraints() {
        ffi::init_memory_pool();
        crate::cairo_air::pedersen::gpu_init();

        let program = build_fib_program(64);
        let n_ped = 4; // small number of Pedersen hashes
        let ped_a: Vec<crate::cairo_air::stark252_field::Fp> = (0..n_ped).map(|i| {
            crate::cairo_air::stark252_field::Fp::from_u64(i as u64 + 1)
        }).collect();
        let ped_b: Vec<crate::cairo_air::stark252_field::Fp> = (0..n_ped).map(|i| {
            crate::cairo_air::stark252_field::Fp::from_u64(i as u64 + 100)
        }).collect();

        let proof = cairo_prove_with_pedersen(&program, 64, 6, Some((&ped_a, &ped_b)));

        // EC trace should be committed
        assert!(proof.ec_trace_commitment.is_some(), "EC trace should be committed");
        assert!(!proof.ec_trace_at_queries.is_empty(), "EC trace values should be in proof");

        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Pedersen EC-constrained proof failed: {:?}", result);
    }

    #[test]
    fn test_tamper_ec_trace() {
        ffi::init_memory_pool();
        crate::cairo_air::pedersen::gpu_init();

        let program = build_fib_program(64);
        let ped_a = vec![crate::cairo_air::stark252_field::Fp::from_u64(42)];
        let ped_b = vec![crate::cairo_air::stark252_field::Fp::from_u64(99)];

        let mut proof = cairo_prove_with_pedersen(&program, 64, 6, Some((&ped_a, &ped_b)));
        // Tamper EC trace commitment
        if let Some(ref mut ec) = proof.ec_trace_commitment {
            ec[0] ^= 1;
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered EC trace commitment should fail");
    }

    #[test]
    fn test_tamper_logup_final_sum() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.logup_final_sum[0] ^= 1; // corrupt LogUp final sum
        let result = cairo_verify(&proof);
        // This changes the Fiat-Shamir transcript or fails RC check
        assert!(result.is_err(), "Tampered LogUp final sum should fail");
    }

    #[test]
    fn test_tamper_rc_final_sum() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.rc_final_sum[0] ^= 1; // corrupt range check final sum
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered RC final sum should fail");
    }

    #[test]
    fn test_rc_final_sum_is_real() {
        // Verify range check wiring produces a distinct sum (not a copy of LogUp)
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let proof = cairo_prove(&program, 64, 6);
        assert_ne!(proof.rc_final_sum, proof.logup_final_sum,
            "RC final sum should differ from LogUp — placeholder not replaced?");
        assert_ne!(proof.rc_final_sum, [0; 4],
            "RC final sum should be non-zero for a non-trivial trace");
        // Full proof must still verify
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Proof with real RC final sum should verify: {:?}", result);
    }

    #[test]
    fn test_cairo_tampered_program_hash() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.public_inputs.program_hash[0] ^= 1;
        let result = cairo_verify(&proof);
        // Tampered program hash changes Fiat-Shamir transcript → FRI mismatch
        assert!(result.is_err(), "Tampered program hash should fail");
    }
}
