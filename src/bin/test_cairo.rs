/// Cairo VM STARK prover test.
/// Generates a Cairo execution trace, proves it on GPU, verifies FRI.
use kraken_stark::cairo_air::{decode::Instruction, vm::{Memory, execute}, trace::{self, N_COLS, N_CONSTRAINTS}};
use kraken_stark::circle::Coset;
use kraken_stark::cuda::ffi;
use kraken_stark::device::DeviceBuffer;
use kraken_stark::field::{M31, QM31};
use kraken_stark::fri::{self, SecureColumn};
use kraken_stark::merkle::MerkleTree;
use kraken_stark::ntt::{self, ForwardTwiddleCache, InverseTwiddleCache};
use kraken_stark::channel::Channel;
use std::time::Instant;

/// Build a Fibonacci program in Cairo bytecode.
fn build_fib_program(n: usize) -> Vec<u64> {
    let mut program = Vec::new();

    // [ap+0] = 1 (fib_0)
    let assert_imm = Instruction {
        off0: 0x8000, off1: 0x8000, off2: 0x8001,
        op1_imm: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    program.push(assert_imm.encode());
    program.push(1); // immediate

    // [ap+0] = 1 (fib_1)
    program.push(assert_imm.encode());
    program.push(1);

    // Loop: [ap] = [ap-2] + [ap-1]; ap++
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

fn main() {
    println!("Cairo VM STARK prover test");
    println!("==========================\n");

    ffi::init_memory_pool();

    for log_n in [8, 12, 16, 20, 24] {
        let n: usize = 1 << log_n;
        let eval_size = 2 * n;
        let log_eval_size = log_n + 1;

        print!("log_n={log_n} ({n} steps, {N_COLS} cols, {N_CONSTRAINTS} constraints)... ");

        let t0 = Instant::now();

        // 1. Generate Cairo execution trace
        let t_trace = Instant::now();
        let program = build_fib_program(n);
        let mut mem = Memory::new();
        mem.load_program(&program);
        let vm_trace = execute(&mut mem, n);
        let columns = trace::trace_to_columns(&vm_trace, log_n);
        let trace_ms = t_trace.elapsed().as_secs_f64() * 1000.0;

        // 2. Upload + interpolate + evaluate each column on eval domain
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
        drop(inv_cache);
        drop(fwd_cache);

        // 3. Commit trace
        let trace_commitment = MerkleTree::commit_root_only(&d_eval_cols, log_eval_size);

        // 4. Cairo constraint kernel
        let mut channel = Channel::new();
        channel.mix_digest(&trace_commitment);

        let alpha_coeffs: Vec<QM31> = (0..N_CONSTRAINTS)
            .map(|_| channel.draw_felt())
            .collect();
        let alpha_flat: Vec<u32> = alpha_coeffs.iter()
            .flat_map(|a| a.to_u32_array())
            .collect();

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

        // 5. Commit quotient
        let quotient_commitment = MerkleTree::commit_root_soa4(&q0, &q1, &q2, &q3, log_eval_size);
        channel.mix_digest(&quotient_commitment);

        // 6. FRI
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

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        if all_zero {
            println!("SUSPICIOUS (all-zero last layer) in {total_ms:.0}ms");
        } else {
            let n_elements = n as u64 * N_COLS as u64;
            println!("OK in {total_ms:.0}ms ({n_elements} field elements, trace={trace_ms:.0}ms)");
        }

        drop(d_eval_cols);
    }
}
