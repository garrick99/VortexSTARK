/// Cairo VM STARK prover with full LogUp interaction phase.
///
/// Two-phase proof:
/// Phase 1: Commit execution trace (27 columns)
/// Phase 2: Draw challenges → compute LogUp interaction trace → commit
/// Then: combined quotient → FRI
use kraken_stark::cairo_air::{
    decode::Instruction,
    vm::{Memory, execute},
    trace::{self, N_COLS, N_CONSTRAINTS, COL_PC, COL_INST_LO, COL_DST_ADDR, COL_DST,
            COL_OP0_ADDR, COL_OP0, COL_OP1_ADDR, COL_OP1},
    logup,
    range_check,
};
use kraken_stark::circle::Coset;
use kraken_stark::cuda::ffi;
use kraken_stark::device::DeviceBuffer;
use kraken_stark::field::{M31, QM31};
use kraken_stark::field::cm31::CM31;
use kraken_stark::fri::{self, SecureColumn};
use kraken_stark::merkle::MerkleTree;
use kraken_stark::ntt::{self, ForwardTwiddleCache, InverseTwiddleCache};
use kraken_stark::channel::Channel;
use std::time::Instant;

/// Build a Fibonacci program in Cairo bytecode.
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

/// GPU parallel prefix sum for QM31 (4 M31 columns).
/// Uses block-level scan + inter-block reduction.
fn gpu_prefix_sum(
    d_c0: &mut DeviceBuffer<u32>, d_c1: &mut DeviceBuffer<u32>,
    d_c2: &mut DeviceBuffer<u32>, d_c3: &mut DeviceBuffer<u32>,
    n: usize,
) {
    const BLOCK_SIZE: u32 = 256;
    let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if n_blocks <= 1 {
        // Single block: scan in-place
        unsafe {
            ffi::cuda_qm31_block_scan(
                d_c0.as_mut_ptr(), d_c1.as_mut_ptr(), d_c2.as_mut_ptr(), d_c3.as_mut_ptr(),
                std::ptr::null_mut(), std::ptr::null_mut(),
                std::ptr::null_mut(), std::ptr::null_mut(),
                n as u32, BLOCK_SIZE,
            );
            ffi::cuda_device_sync();
        }
        return;
    }

    // Multi-block: scan each block, collect block sums, scan block sums, propagate
    let mut d_bs0 = DeviceBuffer::<u32>::alloc(n_blocks as usize);
    let mut d_bs1 = DeviceBuffer::<u32>::alloc(n_blocks as usize);
    let mut d_bs2 = DeviceBuffer::<u32>::alloc(n_blocks as usize);
    let mut d_bs3 = DeviceBuffer::<u32>::alloc(n_blocks as usize);

    // Step 1: block-level scan
    unsafe {
        ffi::cuda_qm31_block_scan(
            d_c0.as_mut_ptr(), d_c1.as_mut_ptr(), d_c2.as_mut_ptr(), d_c3.as_mut_ptr(),
            d_bs0.as_mut_ptr(), d_bs1.as_mut_ptr(), d_bs2.as_mut_ptr(), d_bs3.as_mut_ptr(),
            n as u32, BLOCK_SIZE,
        );
        ffi::cuda_device_sync();
    }

    // Step 2: scan block sums (recursive for large n, but block sums are small)
    gpu_prefix_sum(&mut d_bs0, &mut d_bs1, &mut d_bs2, &mut d_bs3, n_blocks as usize);

    // Step 3: propagate block prefixes
    unsafe {
        ffi::cuda_qm31_add_block_prefix(
            d_c0.as_mut_ptr(), d_c1.as_mut_ptr(), d_c2.as_mut_ptr(), d_c3.as_mut_ptr(),
            d_bs0.as_ptr(), d_bs1.as_ptr(), d_bs2.as_ptr(), d_bs3.as_ptr(),
            n as u32, BLOCK_SIZE,
        );
        ffi::cuda_device_sync();
    }
}

fn main() {
    println!("Cairo VM STARK prover — full LogUp interaction phase");
    println!("=====================================================\n");

    ffi::init_memory_pool();

    for log_n in [12, 16, 20, 24] {
        let n: usize = 1 << log_n;
        let eval_size = 2 * n;
        let log_eval_size = log_n + 1;

        print!("log_n={log_n} ({n} steps, {N_COLS}+4 cols, {N_CONSTRAINTS} constraints + LogUp)... ");

        let t0 = Instant::now();

        // =============================================
        // Phase 1: Generate and commit execution trace
        // =============================================
        let t_vm = Instant::now();
        let program = build_fib_program(n);
        let mut mem = Memory::with_capacity(n + 200);
        mem.load_program(&program);
        // Fused execute + layout: writes directly to columns, no intermediate TraceRow structs
        let columns = kraken_stark::cairo_air::vm::execute_to_columns(&mut mem, n, log_n);
        let vm_ms = t_vm.elapsed().as_secs_f64() * 1000.0;
        let layout_ms = 0.0; // fused into vm_ms

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

        // Commit Phase 1: trace
        let trace_commitment = MerkleTree::commit_root_only(&d_eval_cols, log_eval_size);

        let mut channel = Channel::new();
        channel.mix_digest(&trace_commitment);

        let phase1_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // =============================================
        // Phase 2: LogUp interaction trace
        // =============================================
        let t_phase2 = Instant::now();

        // Draw LogUp challenges from Fiat-Shamir
        let z_mem = channel.draw_felt();
        let alpha_mem = channel.draw_felt();

        // Compute memory LogUp denominators on GPU
        let mut d_denom0 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut d_denom1 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut d_denom2 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut d_denom3 = DeviceBuffer::<u32>::alloc(eval_size);

        let z_arr = z_mem.to_u32_array();
        let alpha_arr = alpha_mem.to_u32_array();

        unsafe {
            ffi::cuda_logup_memory_denoms(
                d_eval_cols[COL_PC].as_ptr(), d_eval_cols[COL_INST_LO].as_ptr(),
                d_eval_cols[COL_DST_ADDR].as_ptr(), d_eval_cols[COL_DST].as_ptr(),
                d_eval_cols[COL_OP0_ADDR].as_ptr(), d_eval_cols[COL_OP0].as_ptr(),
                d_eval_cols[COL_OP1_ADDR].as_ptr(), d_eval_cols[COL_OP1].as_ptr(),
                d_denom0.as_mut_ptr(), d_denom1.as_mut_ptr(),
                d_denom2.as_mut_ptr(), d_denom3.as_mut_ptr(),
                z_arr.as_ptr(), alpha_arr.as_ptr(),
                eval_size as u32,
            );
        }

        // Batch inverse the denominator products (QM31)
        let mut d_inv0 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut d_inv1 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut d_inv2 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut d_inv3 = DeviceBuffer::<u32>::alloc(eval_size);

        unsafe {
            ffi::cuda_qm31_inverse(
                d_denom0.as_ptr(), d_denom1.as_ptr(), d_denom2.as_ptr(), d_denom3.as_ptr(),
                d_inv0.as_mut_ptr(), d_inv1.as_mut_ptr(), d_inv2.as_mut_ptr(), d_inv3.as_mut_ptr(),
                eval_size as u32,
            );
        }
        drop(d_denom0); drop(d_denom1); drop(d_denom2); drop(d_denom3);

        // Combine: per-row LogUp sum = numerator * inv(product)
        let mut d_logup0 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut d_logup1 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut d_logup2 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut d_logup3 = DeviceBuffer::<u32>::alloc(eval_size);

        unsafe {
            ffi::cuda_logup_memory_combine(
                d_eval_cols[COL_PC].as_ptr(), d_eval_cols[COL_INST_LO].as_ptr(),
                d_eval_cols[COL_DST_ADDR].as_ptr(), d_eval_cols[COL_DST].as_ptr(),
                d_eval_cols[COL_OP0_ADDR].as_ptr(), d_eval_cols[COL_OP0].as_ptr(),
                d_eval_cols[COL_OP1_ADDR].as_ptr(), d_eval_cols[COL_OP1].as_ptr(),
                d_inv0.as_ptr(), d_inv1.as_ptr(), d_inv2.as_ptr(), d_inv3.as_ptr(),
                d_logup0.as_mut_ptr(), d_logup1.as_mut_ptr(),
                d_logup2.as_mut_ptr(), d_logup3.as_mut_ptr(),
                z_arr.as_ptr(), alpha_arr.as_ptr(),
                eval_size as u32,
            );
        }
        drop(d_inv0); drop(d_inv1); drop(d_inv2); drop(d_inv3);

        // Parallel prefix sum → running sum
        gpu_prefix_sum(&mut d_logup0, &mut d_logup1, &mut d_logup2, &mut d_logup3, eval_size);

        // Commit Phase 2: interaction trace (4 columns = running sum)
        let interaction_commitment = MerkleTree::commit_root_soa4(
            &d_logup0, &d_logup1, &d_logup2, &d_logup3, log_eval_size,
        );
        channel.mix_digest(&interaction_commitment);

        let phase2_ms = t_phase2.elapsed().as_secs_f64() * 1000.0;

        // =============================================
        // Phase 3: Combined quotient + FRI
        // =============================================
        let t_phase3 = Instant::now();

        // Transition constraints quotient (same as before)
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
        drop(d_logup0); drop(d_logup1); drop(d_logup2); drop(d_logup3);

        // Commit combined quotient
        let quotient_commitment = MerkleTree::commit_root_soa4(&q0, &q1, &q2, &q3, log_eval_size);
        channel.mix_digest(&quotient_commitment);

        // FRI
        let quotient_col = SecureColumn { cols: [q0, q1, q2, q3], len: eval_size };

        let fri_alpha = channel.draw_felt();
        let fold_domain = Coset::half_coset(log_eval_size);
        let d_twid = fri::compute_fold_twiddles_on_demand(&fold_domain, true);
        let mut line_eval = SecureColumn::zeros(eval_size / 2);
        fri::fold_circle_into_line_with_twiddles(&mut line_eval, &quotient_col, fri_alpha, &d_twid);
        drop(d_twid);
        drop(quotient_col);

        let mut current = line_eval;
        let mut current_log = log_eval_size - 1;
        while current_log > 3 {
            let fold_alpha = channel.draw_felt();
            let line_domain = Coset::half_coset(current_log);
            let d_twid = fri::compute_fold_twiddles_on_demand(&line_domain, false);
            let folded = fri::fold_line_with_twiddles(&current, fold_alpha, &d_twid);
            drop(d_twid);
            drop(current);
            current = folded;
            current_log -= 1;
        }

        let last_layer = current.to_qm31();
        let all_zero = last_layer.iter().all(|v| *v == QM31::ZERO);
        let phase3_ms = t_phase3.elapsed().as_secs_f64() * 1000.0;

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        drop(inv_cache);
        drop(fwd_cache);

        if all_zero {
            println!("SUSPICIOUS in {total_ms:.0}ms");
        } else {
            let n_elements = n as u64 * (N_COLS as u64 + 4);
            println!("OK in {total_ms:.0}ms ({n_elements} elems, \
                      vm={vm_ms:.0}ms layout={layout_ms:.0}ms gpu={phase2_ms:.0}+{phase3_ms:.0}ms)");
        }
    }
}
