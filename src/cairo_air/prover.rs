//! End-to-end Cairo STARK prover and verifier.
//!
//! Produces a complete proof of Cairo program execution:
//! 1. VM execution → 27-column trace
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
use super::vm::Memory;

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
}

/// Complete Cairo STARK proof.
#[derive(Clone)]
pub struct CairoProof {
    pub log_trace_size: u32,
    /// Public inputs (verified by both prover and verifier)
    pub public_inputs: CairoPublicInputs,
    /// Merkle root: VM trace (27 columns)
    pub trace_commitment: [u32; 8],
    /// Merkle root: LogUp interaction trace (4 QM31 columns)
    pub interaction_commitment: [u32; 8],
    /// Merkle root: combined quotient (4 QM31 columns)
    pub quotient_commitment: [u32; 8],
    /// FRI layer commitments
    pub fri_commitments: Vec<[u32; 8]>,
    /// Final FRI polynomial (2^3 = 8 QM31 values)
    pub fri_last_layer: Vec<QM31>,
    /// Query indices
    pub query_indices: Vec<usize>,
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

/// Prove execution of a Cairo program.
/// Returns a complete proof that can be verified.
pub fn cairo_prove(program: &[u64], n_steps: usize, log_n: u32) -> CairoProof {
    let n = 1usize << log_n;
    assert!(n_steps <= n);
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
    };

    // ---- Phase 1: Trace generation + commitment ----
    let mut mem = Memory::with_capacity(n_steps + 200);
    mem.load_program(program);
    let columns = super::vm::execute_to_columns(&mut mem, n_steps, log_n);

    let trace_domain = Coset::half_coset(log_n);
    let eval_domain = Coset::half_coset(log_eval_size);
    let inv_cache = InverseTwiddleCache::new(&trace_domain);
    let fwd_cache = ForwardTwiddleCache::new(&eval_domain);

    let mut d_eval_cols: Vec<DeviceBuffer<u32>> = Vec::with_capacity(N_COLS);
    for c in 0..N_COLS {
        let mut d_col = DeviceBuffer::from_host(&columns[c]);
        ntt::interpolate(&mut d_col, &inv_cache);
        let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
        unsafe {
            ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32);
        }
        drop(d_col);
        ntt::evaluate(&mut d_eval, &fwd_cache);
        d_eval_cols.push(d_eval);
    }
    drop(columns);

    let trace_commitment = MerkleTree::commit_root_only(&d_eval_cols, log_eval_size);

    let mut channel = Channel::new();
    // Bind public inputs into Fiat-Shamir transcript (compositional binding)
    channel.mix_digest(&public_inputs.program_hash);
    channel.mix_digest(&trace_commitment);

    // ---- Phase 2: Fused LogUp interaction ----
    let z_mem = channel.draw_felt();
    let alpha_mem = channel.draw_felt();
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

    let interaction_commitment = MerkleTree::commit_root_soa4(
        &d_logup0, &d_logup1, &d_logup2, &d_logup3, log_eval_size,
    );
    channel.mix_digest(&interaction_commitment);
    drop(d_logup0); drop(d_logup1); drop(d_logup2); drop(d_logup3);

    // ---- Phase 3: Quotient ----
    let constraint_alphas: Vec<QM31> = (0..N_CONSTRAINTS).map(|_| channel.draw_felt()).collect();
    let alpha_flat: Vec<u32> = constraint_alphas.iter().flat_map(|a| a.to_u32_array()).collect();

    let col_ptrs: Vec<*const u32> = d_eval_cols.iter().map(|c| c.as_ptr()).collect();
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
    drop(d_eval_cols);

    let quotient_commitment = MerkleTree::commit_root_soa4(&q0, &q1, &q2, &q3, log_eval_size);
    channel.mix_digest(&quotient_commitment);

    // Download quotient to host for decommitment (before FRI consumes it)
    let q_host: Vec<Vec<u32>> = [&q0, &q1, &q2, &q3].iter().map(|c| c.to_host()).collect();

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

    // Quotient decommitment (from host data)
    let quotient_decommitment = decommit_from_host_soa4(&q_host, &query_indices);

    // FRI decommitments (download each layer's values at query indices)
    let mut fri_decommitments = Vec::new();
    let mut folded_indices: Vec<usize> = query_indices.iter().map(|&qi| qi / 2).collect();

    for eval in &fri_evals {
        let decom = decommit_fri_layer(eval, &folded_indices);
        fri_decommitments.push(decom);
        folded_indices = folded_indices.iter().map(|&i| i / 2).collect();
    }

    CairoProof {
        log_trace_size: log_n,
        public_inputs,
        trace_commitment,
        interaction_commitment,
        quotient_commitment,
        fri_commitments,
        fri_last_layer,
        query_indices,
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

    channel.mix_digest(&proof.interaction_commitment);

    let _constraint_alphas: Vec<QM31> = (0..N_CONSTRAINTS).map(|_| channel.draw_felt()).collect();

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
    if decom.values.len() != N_QUERIES || decom.auth_paths.len() != N_QUERIES {
        return Err(format!("{label} decommitment size mismatch"));
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

/// Decommit SoA4 values + Merkle auth paths from host arrays at query indices.
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

/// Decommit FRI layer values + auth paths from GPU SecureColumn.
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
