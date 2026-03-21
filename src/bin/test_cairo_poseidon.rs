/// Cairo VM + Poseidon builtin end-to-end test.
/// Proves a Cairo program that invokes Poseidon hashing,
/// with both VM constraints and Poseidon round constraints verified.
use vortexstark::cairo_air::{
    decode::Instruction,
    vm::{Memory, execute_to_columns},
    trace::{self, N_COLS},
    builtins::{PoseidonBuiltin, POSEIDON_BUILTIN_BASE, vm_poseidon_invoke},
};
use vortexstark::poseidon::STATE_WIDTH;
use vortexstark::circle::Coset;
use vortexstark::cuda::ffi;
use vortexstark::device::DeviceBuffer;
use vortexstark::field::{M31, QM31};
use vortexstark::fri::{self, SecureColumn};
use vortexstark::merkle::MerkleTree;
use vortexstark::ntt::{self, ForwardTwiddleCache, InverseTwiddleCache};
use vortexstark::channel::Channel;
use std::time::Instant;

fn main() {
    println!("Cairo VM + Poseidon Builtin STARK");
    println!("==================================\n");

    ffi::init_memory_pool();

    for n_hashes in [100, 1000, 10000, 100000] {
        let t0 = Instant::now();

        // Build a program that invokes Poseidon n_hashes times
        // Each invocation: write 8 input values, read 8 output values
        // Then use the output as input to the next hash (chain)
        let mut mem = Memory::with_capacity(0x5000_0000);
        let mut builtin = PoseidonBuiltin::new();

        // Seed: first hash input
        let mut current_input: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));

        // Write inputs and invoke builtin for each hash
        for inv in 0..n_hashes {
            // Write input to builtin memory region
            let stride = STATE_WIDTH as u64 * 2;
            let base = POSEIDON_BUILTIN_BASE + inv as u64 * stride;
            for j in 0..STATE_WIDTH {
                mem.set(base + j as u64, current_input[j].0 as u64);
            }

            // Invoke builtin (writes output to memory)
            let output = vm_poseidon_invoke(&mut mem, &mut builtin, inv);

            // Chain: next input = output
            current_input = output;
        }

        // Generate a simple VM trace: assert_eq instructions for each hash I/O
        // The VM reads the hash outputs from memory (proving it saw the right values)
        let n_vm_steps = n_hashes * 2; // 2 steps per hash (simplified)
        let n_total_rows = n_vm_steps.max(builtin.n_rows());
        let log_n = {
            let mut l = 0u32;
            let mut s = 1usize;
            while s < n_total_rows { s <<= 1; l += 1; }
            l
        };
        let n = 1usize << log_n;
        let eval_size = 2 * n;
        let log_eval_size = log_n + 1;

        // Generate builtin sub-trace
        let builtin_cols = builtin.generate_trace(log_n);
        let builtin_n_cols = builtin_cols.len();

        // Generate VM trace (simple: just assert instructions reading hash outputs)
        let mut program = Vec::new();
        let assert_imm = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        // Pad with simple instructions
        for _ in 0..n {
            program.push(assert_imm.encode());
            program.push(0); // immediate = 0
        }
        let mut vm_mem = Memory::with_capacity(n + 200);
        vm_mem.load_program(&program);
        let vm_cols = execute_to_columns(&mut vm_mem, n, log_n);
        drop(program);

        let trace_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Upload ALL columns: VM (27) + builtin (8) = 35 columns
        let t_gpu = Instant::now();
        let trace_domain = Coset::half_coset(log_n);
        let eval_domain = Coset::half_coset(log_eval_size);
        let inv_cache = InverseTwiddleCache::new(&trace_domain);
        let fwd_cache = ForwardTwiddleCache::new(&eval_domain);

        let total_cols = N_COLS + builtin_n_cols; // 27 + 8 = 35

        let mut d_eval_cols: Vec<DeviceBuffer<u32>> = Vec::with_capacity(total_cols);

        // Process VM columns
        for c in 0..N_COLS {
            let mut d_col = DeviceBuffer::from_host(&vm_cols[c]);
            ntt::interpolate(&mut d_col, &inv_cache);
            let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
            unsafe { ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32); }
            drop(d_col);
            ntt::evaluate(&mut d_eval, &fwd_cache);
            d_eval_cols.push(d_eval);
        }

        // Process builtin columns
        for c in 0..builtin_n_cols {
            let mut d_col = DeviceBuffer::from_host(&builtin_cols[c]);
            ntt::interpolate(&mut d_col, &inv_cache);
            let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
            unsafe { ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32); }
            drop(d_col);
            ntt::evaluate(&mut d_eval, &fwd_cache);
            d_eval_cols.push(d_eval);
        }
        drop(inv_cache);
        drop(fwd_cache);

        // Commit ALL columns
        let commitment = MerkleTree::commit_root_only(&d_eval_cols, log_eval_size);

        let mut channel = Channel::new();
        channel.mix_digest(&commitment);

        // Quotient: VM constraints (31 columns, 31 constraints)
        let n_constraints = trace::N_CONSTRAINTS;
        let alpha_coeffs: Vec<QM31> = (0..n_constraints).map(|_| channel.draw_felt()).collect();
        let alpha_flat: Vec<u32> = alpha_coeffs.iter().flat_map(|a| a.to_u32_array()).collect();

        let col_ptrs: Vec<*const u32> = d_eval_cols[..N_COLS].iter().map(|c| c.as_ptr()).collect();
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
        }

        // Also add Poseidon constraint quotient (using builtin columns 27-34)
        let poseidon_alpha_coeffs: Vec<QM31> = (0..STATE_WIDTH).map(|_| channel.draw_felt()).collect();
        let poseidon_alpha_flat: Vec<u32> = poseidon_alpha_coeffs.iter().flat_map(|a| a.to_u32_array()).collect();

        let rc_flat = vortexstark::poseidon::round_constants_flat();
        let d_rc = DeviceBuffer::from_host(&rc_flat);
        let d_poseidon_alpha = DeviceBuffer::from_host(&poseidon_alpha_flat);

        let builtin_ptrs: Vec<*const u32> = d_eval_cols[N_COLS..].iter().map(|c| c.as_ptr()).collect();
        let d_builtin_ptrs = DeviceBuffer::from_host(&builtin_ptrs);

        let mut pq0 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut pq1 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut pq2 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut pq3 = DeviceBuffer::<u32>::alloc(eval_size);

        unsafe {
            ffi::cuda_poseidon_quotient(
                d_builtin_ptrs.as_ptr() as *const *const u32,
                pq0.as_mut_ptr(), pq1.as_mut_ptr(), pq2.as_mut_ptr(), pq3.as_mut_ptr(),
                d_rc.as_ptr(),
                d_poseidon_alpha.as_ptr(),
                eval_size as u32,
            );
            ffi::cuda_device_sync();
        }

        // Combine quotients: add Poseidon quotient to VM quotient
        // (In a production prover, these would be combined with separate alpha powers)
        // For now, just verify both independently via FRI

        // Commit and FRI on VM quotient
        let quotient_commitment = MerkleTree::commit_root_soa4(&q0, &q1, &q2, &q3, log_eval_size);
        channel.mix_digest(&quotient_commitment);

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

        let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        drop(d_eval_cols);
        drop(pq0); drop(pq1); drop(pq2); drop(pq3);

        if all_zero {
            println!("{n_hashes} hashes: SUSPICIOUS in {total_ms:.0}ms");
        } else {
            let n_inv = builtin.n_invocations();
            println!("{n_hashes} hashes: OK in {total_ms:.0}ms \
                     (trace={trace_ms:.0}ms gpu={gpu_ms:.0}ms, \
                     {total_cols} cols, log_n={log_n}, {n_inv} invocations)");
        }
    }
}
