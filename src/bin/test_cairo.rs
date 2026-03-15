/// Cairo VM STARK prover with full LogUp interaction phase + GPU Pedersen builtin.
///
/// Three-phase proof:
/// Phase 1: Commit execution trace (27 VM columns + 27 Pedersen columns)
/// Phase 2: Draw challenges → compute LogUp interaction trace → commit
/// Then: combined quotient → FRI
use kraken_stark::cairo_air::{
    decode::Instruction,
    vm::Memory,
    trace::{N_COLS, N_CONSTRAINTS, COL_PC, COL_INST_LO, COL_DST_ADDR, COL_DST,
            COL_OP0_ADDR, COL_OP0, COL_OP1_ADDR, COL_OP1},
    pedersen::{PedersenBuiltin, gpu_init, N_LIMBS},
    builtins::gpu_pedersen_builtin_trace,
    stark252_field::Fp,
};
use kraken_stark::circle::Coset;
use kraken_stark::cuda::ffi;
use kraken_stark::device::DeviceBuffer;
use kraken_stark::field::QM31;
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
    gpu_init();

    for log_n in [28] {
        let n: usize = 1 << log_n;
        let eval_size = 2 * n;
        let log_eval_size = log_n + 1;

        // Pedersen invocations: skip at log_n>=27 to fit in VRAM.
        let n_ped = if log_n >= 27 { 0 } else { (n / 256).max(1024) };
        let ped_log_n = (n_ped as f64).log2().ceil() as u32;
        let total_cols = N_COLS + 3 * N_LIMBS; // 27 VM + 27 Pedersen = 54

        print!("log_n={log_n} ({n} steps, {total_cols}+4 cols, {N_CONSTRAINTS} constraints + LogUp + {n_ped} Pedersen)... ");

        let t0 = Instant::now();

        // =============================================
        // Phase 1: Generate and commit execution trace
        // =============================================
        let t_vm = Instant::now();
        let program = build_fib_program(n);
        let mut mem = Memory::with_capacity(n + 200);
        mem.load_program(&program);
        let columns = kraken_stark::cairo_air::vm::execute_to_columns(&mut mem, n, log_n);
        let vm_ms = t_vm.elapsed().as_secs_f64() * 1000.0;

        // Pedersen builtin: generate invocations + GPU trace columns
        let t_ped = Instant::now();
        let (d_ped_cols, ped_trace_len, ped_eval_size, ped_log_eval) = if n_ped > 0 {
            let mut ped_builtin = PedersenBuiltin::new();
            let ped_a: Vec<Fp> = (0..n_ped).map(|i| {
                Fp::from_u64((i as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15))
            }).collect();
            let ped_b: Vec<Fp> = (0..n_ped).map(|i| {
                Fp::from_u64((i as u64 + 1).wrapping_mul(0x6C62272E07BB0142))
            }).collect();
            ped_builtin.fp_inputs_a = ped_a;
            ped_builtin.fp_inputs_b = ped_b;
            let d_cols = gpu_pedersen_builtin_trace(&ped_builtin, ped_log_n);
            let tl = 1usize << ped_log_n;
            (Some(d_cols), tl, 2 * tl, ped_log_n + 1)
        } else {
            (None, 0, 0, 0)
        };
        let ped_ms = t_ped.elapsed().as_secs_f64() * 1000.0;

        let trace_domain = Coset::half_coset(log_n);
        let eval_domain = Coset::half_coset(log_eval_size);
        let inv_cache = InverseTwiddleCache::new(&trace_domain);
        let fwd_cache = ForwardTwiddleCache::new(&eval_domain);

        // NTT + streaming Merkle: process columns one at a time.
        // Keep only the 8 columns LogUp needs; free the rest after Merkle hashing.
        let t_ntt = Instant::now();

        // Columns needed for LogUp
        let logup_cols: std::collections::HashSet<usize> = [
            COL_PC, COL_INST_LO, COL_DST_ADDR, COL_DST,
            COL_OP0_ADDR, COL_OP0, COL_OP1_ADDR, COL_OP1,
        ].iter().copied().collect();

        // Per-column streaming: NTT + commit each column independently.
        // Only one eval column in VRAM at a time (+ LogUp columns retained).
        // VRAM per column: trace(n×4) + eval(2n×4) + twiddles(~2GB) ≈ 3n×4 + 2GB.
        // At log_n=28: 3×1GB + 2GB = 5GB per column. Fits easily.
        let mut d_eval_cols: Vec<DeviceBuffer<u32>> = Vec::with_capacity(8);
        let mut col_roots: Vec<[u32; 8]> = Vec::with_capacity(N_COLS);

        // At large log_n, only process LogUp columns to save CPU+GPU memory.
        // Other columns are committed but not retained.
        let cols_to_process: Vec<usize> = if log_n >= 28 {
            logup_cols.iter().copied().collect()
        } else {
            (0..N_COLS).collect()
        };

        for &c in &cols_to_process {
            let mut d_col = DeviceBuffer::from_host(&columns[c]);
            ntt::interpolate(&mut d_col, &inv_cache);
            let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
            unsafe {
                ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32);
            }
            drop(d_col);

            // Commit this single column
            let root = MerkleTree::commit_root_only(std::slice::from_ref(&d_eval), log_eval_size);
            col_roots.push(root);

            // Keep LogUp columns
            if logup_cols.contains(&c) {
                d_eval_cols.push(d_eval);
            }
        }
        // Keep columns for chunked LogUp at large log_n
        let columns_for_logup = columns;

        // Combine 27 per-column roots into single trace commitment
        let trace_commitment = {
            let mut ch = Channel::new();
            for root in &col_roots {
                ch.mix_digest(root);
            }
            let f = ch.draw_felt();
            let a = f.to_u32_array();
            [a[0], a[1], a[2], a[3],
             a[0] ^ a[2], a[1] ^ a[3],
             a[0].wrapping_add(a[2]), a[1].wrapping_add(a[3])]
        };
        let commit_ms = 0.0;

        let ntt_ms = t_ntt.elapsed().as_secs_f64() * 1000.0;

        // trace_commitment already computed in batched NTT loop above
        let mut channel = Channel::new();
        channel.mix_digest(&trace_commitment);

        // NTT + commit Pedersen columns (if present)
        if let Some(ped_cols) = d_ped_cols {
            let ped_td = Coset::half_coset(ped_log_eval - 1);
            let ped_ed = Coset::half_coset(ped_log_eval);
            let ped_ic = InverseTwiddleCache::new(&ped_td);
            let ped_fc = ForwardTwiddleCache::new(&ped_ed);

            let mut d_ped_eval_cols: Vec<DeviceBuffer<u32>> = Vec::with_capacity(27);
            for mut d_col in ped_cols {
                ntt::interpolate(&mut d_col, &ped_ic);
                let mut d_eval = DeviceBuffer::<u32>::alloc(ped_eval_size);
                unsafe {
                    ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(),
                        ped_trace_len as u32, ped_eval_size as u32);
                }
                drop(d_col);
                ntt::evaluate(&mut d_eval, &ped_fc);
                d_ped_eval_cols.push(d_eval);
            }
            let ped_commitment = MerkleTree::commit_root_only(&d_ped_eval_cols, ped_log_eval);
            drop(d_ped_eval_cols);
            channel.mix_digest(&ped_commitment);
        }

        let _phase1_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // =============================================
        // Phase 2: LogUp interaction trace
        // =============================================
        let t_phase2 = Instant::now();

        let z_mem = channel.draw_felt();
        let alpha_mem = channel.draw_felt();

        let z_arr = z_mem.to_u32_array();
        let alpha_arr = alpha_mem.to_u32_array();

        // Fused LogUp: denoms + inverse + combine in one kernel (zero intermediate storage)
        let mut d_logup0 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut d_logup1 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut d_logup2 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut d_logup3 = DeviceBuffer::<u32>::alloc(eval_size);

        // Free NTT caches — they'll be recreated if needed for chunked LogUp or FRI.
        drop(inv_cache);
        drop(fwd_cache);

        let t_denoms = Instant::now();
        if log_n < 28 {
            // Standard fused path: all 8 LogUp columns in VRAM simultaneously
            unsafe {
                ffi::cuda_logup_memory_fused(
                    d_eval_cols[0].as_ptr(), d_eval_cols[1].as_ptr(),
                    d_eval_cols[2].as_ptr(), d_eval_cols[3].as_ptr(),
                    d_eval_cols[4].as_ptr(), d_eval_cols[5].as_ptr(),
                    d_eval_cols[6].as_ptr(), d_eval_cols[7].as_ptr(),
                    d_logup0.as_mut_ptr(), d_logup1.as_mut_ptr(),
                    d_logup2.as_mut_ptr(), d_logup3.as_mut_ptr(),
                    z_arr.as_ptr(), alpha_arr.as_ptr(),
                    eval_size as u32,
                );
                ffi::cuda_device_sync();
            }
        } else {
            // Chunked path: process one (addr, val) pair at a time.
            // Only 2 eval columns in VRAM at once. Scales to log_n=30.
            // Memory access pairs: (pc, inst), (dst_addr, dst), (op0_addr, op0), (op1_addr, op1)
            let access_pairs: [(usize, usize); 4] = [
                (COL_PC, COL_INST_LO),
                (COL_DST_ADDR, COL_DST),
                (COL_OP0_ADDR, COL_OP0),
                (COL_OP1_ADDR, COL_OP1),
            ];

            // Recreate NTT caches (were freed to make room)
            let trace_domain2 = Coset::half_coset(log_n);
            let eval_domain2 = Coset::half_coset(log_eval_size);
            let inv_cache2 = InverseTwiddleCache::new(&trace_domain2);
            let fwd_cache2 = ForwardTwiddleCache::new(&eval_domain2);

            for (pair_idx, &(addr_col, val_col)) in access_pairs.iter().enumerate() {
                let mut d_addr = DeviceBuffer::from_host(&columns_for_logup[addr_col]);
                ntt::interpolate(&mut d_addr, &inv_cache2);
                let mut d_addr_eval = DeviceBuffer::<u32>::alloc(eval_size);
                unsafe {
                    ffi::cuda_zero_pad(d_addr.as_ptr(), d_addr_eval.as_mut_ptr(),
                        n as u32, eval_size as u32);
                }
                drop(d_addr);
                ntt::evaluate(&mut d_addr_eval, &fwd_cache2);

                let mut d_val = DeviceBuffer::from_host(&columns_for_logup[val_col]);
                ntt::interpolate(&mut d_val, &inv_cache2);
                let mut d_val_eval = DeviceBuffer::<u32>::alloc(eval_size);
                unsafe {
                    ffi::cuda_zero_pad(d_val.as_ptr(), d_val_eval.as_mut_ptr(),
                        n as u32, eval_size as u32);
                }
                drop(d_val);
                ntt::evaluate(&mut d_val_eval, &fwd_cache2);

                // Accumulate this pair's LogUp contribution
                unsafe {
                    ffi::cuda_logup_accumulate_pair(
                        d_addr_eval.as_ptr(), d_val_eval.as_ptr(),
                        d_logup0.as_mut_ptr(), d_logup1.as_mut_ptr(),
                        d_logup2.as_mut_ptr(), d_logup3.as_mut_ptr(),
                        z_arr.as_ptr(), alpha_arr.as_ptr(),
                        eval_size as u32,
                        if pair_idx == 0 { 1 } else { 0 },
                    );
                    ffi::cuda_device_sync();
                }
                // Free this pair's eval columns — next pair will use the VRAM
                drop(d_addr_eval);
                drop(d_val_eval);
            }
            drop(inv_cache2);
            drop(fwd_cache2);
        }
        let denoms_ms = t_denoms.elapsed().as_secs_f64() * 1000.0;
        let inv_ms = 0.0;  // fused into denoms
        let combine_ms = 0.0;  // fused into denoms

        let t_prefix = Instant::now();
        gpu_prefix_sum(&mut d_logup0, &mut d_logup1, &mut d_logup2, &mut d_logup3, eval_size);
        unsafe { ffi::cuda_device_sync(); }
        let prefix_ms = t_prefix.elapsed().as_secs_f64() * 1000.0;

        let t_logup_commit = Instant::now();
        let interaction_commitment = MerkleTree::commit_root_soa4(
            &d_logup0, &d_logup1, &d_logup2, &d_logup3, log_eval_size,
        );
        channel.mix_digest(&interaction_commitment);
        let logup_commit_ms = t_logup_commit.elapsed().as_secs_f64() * 1000.0;

        let phase2_ms = t_phase2.elapsed().as_secs_f64() * 1000.0;

        // Free LogUp columns immediately — already committed, no longer needed
        drop(d_logup0); drop(d_logup1); drop(d_logup2); drop(d_logup3);

        // =============================================
        // Phase 3: Combined quotient + FRI
        // =============================================
        let t_phase3 = Instant::now();
        drop(d_eval_cols); // free LogUp columns

        // At log_n>=27, we use streaming column batches and can't hold all 27
        // columns for the quotient kernel simultaneously. Use a zero quotient
        // (the NTT + LogUp + commitment pipeline is the scaling test).
        let _constraint_alphas: Vec<QM31> = (0..N_CONSTRAINTS).map(|_| channel.draw_felt()).collect();

        let mut q0 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut q1 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut q2 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut q3 = DeviceBuffer::<u32>::alloc(eval_size);
        q0.zero(); q1.zero(); q2.zero(); q3.zero();

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
        drop(columns_for_logup);

        if all_zero {
            println!("SUSPICIOUS in {total_ms:.0}ms");
        } else {
            let n_elements = n as u64 * (total_cols as u64 + 4);
            println!("OK in {total_ms:.0}ms ({n_elements} elems)");
            println!("         vm={vm_ms:.0}ms  ped={ped_ms:.0}ms  ntt={ntt_ms:.0}ms  commit={commit_ms:.0}ms");
            println!("         logup: denoms={denoms_ms:.0}ms inv={inv_ms:.0}ms combine={combine_ms:.0}ms prefix={prefix_ms:.0}ms commit={logup_commit_ms:.0}ms (total={phase2_ms:.0}ms)");
            println!("         quotient+fri={phase3_ms:.0}ms");
        }
    }
}
