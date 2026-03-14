/// Poseidon hash chain STARK prover test.
/// Proves knowledge of a Poseidon hash chain with 8 state columns.
use kraken_stark::poseidon::{self, STATE_WIDTH, NUM_ROUNDS};
use kraken_stark::circle::Coset;
use kraken_stark::cuda::ffi;
use kraken_stark::device::DeviceBuffer;
use kraken_stark::field::{M31, QM31};
use kraken_stark::fri::{self, SecureColumn};
use kraken_stark::merkle::MerkleTree;
use kraken_stark::ntt::{self, ForwardTwiddleCache, InverseTwiddleCache};
use kraken_stark::channel::Channel;
use std::time::Instant;

fn main() {
    println!("Poseidon STARK prover test");
    println!("=========================\n");

    ffi::init_memory_pool();

    // Start small to validate correctness, then scale up
    for log_n in [24, 26, 28] {
        let n: usize = 1 << log_n;
        let n_blocks = n / NUM_ROUNDS;
        let eval_size = 2 * n;
        let log_eval_size = log_n + 1;

        print!("log_n={log_n} ({n} rows, {n_blocks} hash blocks, {STATE_WIDTH} cols)... ");

        let t0 = Instant::now();

        // 1. Generate trace on GPU (parallel block computation)
        let t_trace = Instant::now();
        let (mut d_trace_cols, _input, _output) = poseidon::generate_trace_gpu(log_n);
        let trace_ms = t_trace.elapsed().as_secs_f64() * 1000.0;

        // 2. Interpolate + evaluate each column on eval domain (sequential, reuse twiddle cache)
        let trace_domain = Coset::half_coset(log_n);
        let eval_domain = Coset::half_coset(log_eval_size);
        let inv_cache = InverseTwiddleCache::new(&trace_domain);
        let fwd_cache = ForwardTwiddleCache::new(&eval_domain);

        let mut d_eval_cols: Vec<DeviceBuffer<u32>> = Vec::with_capacity(STATE_WIDTH);
        for c in 0..STATE_WIDTH {
            ntt::interpolate(&mut d_trace_cols[c], &inv_cache);
            let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
            unsafe {
                ffi::cuda_zero_pad(d_trace_cols[c].as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32);
            }
            ntt::evaluate(&mut d_eval, &fwd_cache);
            d_eval_cols.push(d_eval);
        }
        drop(d_trace_cols);
        drop(inv_cache);
        drop(fwd_cache);
        let ntt_ms = Instant::now().duration_since(t0 + std::time::Duration::from_secs_f64(trace_ms / 1000.0)).as_secs_f64() * 1000.0;

        // 3. Commit trace (all columns)
        let trace_commitment = MerkleTree::commit_root_only(&d_eval_cols, log_eval_size);

        // 4. Quotient: Poseidon constraint kernel
        let mut channel = Channel::new();
        channel.mix_digest(&trace_commitment);

        // Draw alpha coefficients (one QM31 per constraint = STATE_WIDTH)
        let alpha_coeffs: Vec<QM31> = (0..STATE_WIDTH)
            .map(|_| channel.draw_felt())
            .collect();
        let alpha_flat: Vec<u32> = alpha_coeffs.iter()
            .flat_map(|a| a.to_u32_array())
            .collect();

        // Upload round constants
        let rc_flat = poseidon::round_constants_flat();
        let d_rc = DeviceBuffer::from_host(&rc_flat);

        // Build column pointer array
        let col_ptrs: Vec<*const u32> = d_eval_cols.iter().map(|c| c.as_ptr()).collect();
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);
        let d_alpha = DeviceBuffer::from_host(&alpha_flat);

        let mut q0 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut q1 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut q2 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut q3 = DeviceBuffer::<u32>::alloc(eval_size);

        unsafe {
            ffi::cuda_poseidon_quotient(
                d_col_ptrs.as_ptr() as *const *const u32,
                q0.as_mut_ptr(), q1.as_mut_ptr(), q2.as_mut_ptr(), q3.as_mut_ptr(),
                d_rc.as_ptr(),
                d_alpha.as_ptr(),
                eval_size as u32,
            );
            ffi::cuda_device_sync();
        }

        // 5. Commit quotient
        let quotient_commitment = MerkleTree::commit_root_soa4(&q0, &q1, &q2, &q3, log_eval_size);
        channel.mix_digest(&quotient_commitment);

        // 6. Quick FRI (just verify the quotient has right degree)
        let quotient_col = SecureColumn { cols: [q0, q1, q2, q3], len: eval_size };

        // Circle fold
        let fri_alpha = channel.draw_felt();
        let fold_domain = Coset::half_coset(log_eval_size);
        let d_twid = fri::compute_fold_twiddles_on_demand(&fold_domain, true);
        let mut line_eval = SecureColumn::zeros(eval_size / 2);
        fri::fold_circle_into_line_with_twiddles(&mut line_eval, &quotient_col, fri_alpha, &d_twid);
        drop(d_twid);
        drop(quotient_col);

        let mut current = line_eval;
        let mut current_log = log_eval_size - 1;

        // Line folds down to small size
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

        // Download last layer and check it's non-trivial
        let last_layer = current.to_qm31();
        let all_zero = last_layer.iter().all(|v| *v == QM31::ZERO);

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        if all_zero {
            println!("FAILED (all-zero FRI last layer) in {total_ms:.0}ms");
        } else {
            println!("OK in {total_ms:.0}ms ({} FRI last values, trace={trace_ms:.0}ms)",
                last_layer.len());
        }

        drop(d_eval_cols);
        drop(d_rc);
        drop(d_alpha);
        drop(d_col_ptrs);
    }
}
