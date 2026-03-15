/// Cairo VM + GPU Pedersen builtin — prover pipeline comparison.
///
/// Compares two prover paths for Pedersen trace commitment:
///   OLD: CPU generate_trace() → 27 × H2D upload → NTT → Merkle
///   NEW: GPU fused hash + trace (zero host transfer) → NTT → Merkle
///
/// VM execution with CPU Pedersen is a one-time cost (same in both paths).
/// The GPU path eliminates CPU trace generation + 27 H2D transfers.

use vortex_stark::cairo_air::{
    builtins::gpu_pedersen_builtin_trace,
    pedersen::{self, PedersenBuiltin, gpu_init, fp_to_stark252},
    stark252_field::Fp,
};
use vortex_stark::circle::Coset;
use vortex_stark::cuda::ffi;
use vortex_stark::device::DeviceBuffer;
use vortex_stark::merkle::MerkleTree;
use vortex_stark::ntt::{self, ForwardTwiddleCache, InverseTwiddleCache};
use std::time::Instant;

fn main() {
    println!("Cairo Pedersen Prover Pipeline — CPU vs GPU Path");
    println!("=================================================\n");

    ffi::init_memory_pool();
    gpu_init();

    // Pre-compute Pedersen hashes on GPU (batch), then use results to populate builtin.
    // This simulates the VM having already executed with Pedersen calls.
    for (n_invocations, log_n) in [(1_000, 10u32), (10_000, 14), (100_000, 17), (1_000_000, 20)] {
        let trace_len = 1usize << log_n;
        let eval_size = 2 * trace_len;
        let log_eval_size = log_n + 1;

        println!("--- {n_invocations} invocations, log_n={log_n} ({trace_len} trace rows, 27 cols) ---");

        // Generate deterministic inputs
        let inputs_a: Vec<Fp> = (0..n_invocations).map(|i| {
            Fp::from_u64((i as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15))
        }).collect();
        let inputs_b: Vec<Fp> = (0..n_invocations).map(|i| {
            Fp::from_u64((i as u64 + 1).wrapping_mul(0x6C62272E07BB0142))
        }).collect();

        // Pre-compute outputs on GPU (simulating VM execution with GPU batch hash)
        let outputs = pedersen::gpu_hash_batch(&inputs_a, &inputs_b);

        // Populate PedersenBuiltin with pre-computed entries
        // Use invoke() so fp_inputs_a/b are populated for GPU path
        let mut builtin = PedersenBuiltin::new();
        for i in 0..n_invocations {
            let a_s = fp_to_stark252(&inputs_a[i]);
            let b_s = fp_to_stark252(&inputs_b[i]);
            let o_s = fp_to_stark252(&outputs[i]);
            builtin.entries.push((a_s, b_s, o_s));
            builtin.fp_inputs_a.push(inputs_a[i]);
            builtin.fp_inputs_b.push(inputs_b[i]);
        }

        // Setup NTT caches
        let trace_domain = Coset::half_coset(log_n);
        let eval_domain = Coset::half_coset(log_eval_size);
        let inv_cache = InverseTwiddleCache::new(&trace_domain);
        let fwd_cache = ForwardTwiddleCache::new(&eval_domain);

        // =============================================
        // PATH A: Old CPU path
        //   CPU generate_trace() → 27 × H2D → NTT → Merkle
        // =============================================
        let t0 = Instant::now();
        let cpu_cols = builtin.generate_trace(log_n);
        let cpu_trace_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let mut d_cpu_eval_cols: Vec<DeviceBuffer<u32>> = Vec::with_capacity(27);
        for c in 0..27 {
            let mut d_col = DeviceBuffer::from_host(&cpu_cols[c]);
            ntt::interpolate(&mut d_col, &inv_cache);
            let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
            unsafe {
                ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(),
                    trace_len as u32, eval_size as u32);
            }
            drop(d_col);
            ntt::evaluate(&mut d_eval, &fwd_cache);
            d_cpu_eval_cols.push(d_eval);
        }
        let cpu_ntt_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let cpu_root = MerkleTree::commit_root_only(&d_cpu_eval_cols, log_eval_size);
        let cpu_merkle_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let cpu_total = cpu_trace_ms + cpu_ntt_ms + cpu_merkle_ms;
        drop(d_cpu_eval_cols);

        // =============================================
        // PATH B: New GPU path
        //   GPU fused hash+trace (inputs → columns, never leaves GPU) → NTT → Merkle
        // =============================================
        let t0 = Instant::now();
        // gpu_pedersen_builtin_trace uses stored Fp inputs — fused hash + decompose on GPU.
        let d_gpu_cols = gpu_pedersen_builtin_trace(&builtin, log_n);
        let gpu_trace_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let mut d_gpu_eval_cols: Vec<DeviceBuffer<u32>> = Vec::with_capacity(27);
        for mut d_col in d_gpu_cols {
            ntt::interpolate(&mut d_col, &inv_cache);
            let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
            unsafe {
                ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(),
                    trace_len as u32, eval_size as u32);
            }
            drop(d_col);
            ntt::evaluate(&mut d_eval, &fwd_cache);
            d_gpu_eval_cols.push(d_eval);
        }
        let gpu_ntt_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let gpu_root = MerkleTree::commit_root_only(&d_gpu_eval_cols, log_eval_size);
        let gpu_merkle_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let gpu_total = gpu_trace_ms + gpu_ntt_ms + gpu_merkle_ms;

        let roots_match = cpu_root == gpu_root;

        println!("  CPU path: trace={cpu_trace_ms:.1}ms  NTT+upload={cpu_ntt_ms:.1}ms  merkle={cpu_merkle_ms:.1}ms  TOTAL={cpu_total:.1}ms");
        println!("  GPU path: trace={gpu_trace_ms:.1}ms  NTT={gpu_ntt_ms:.1}ms  merkle={gpu_merkle_ms:.1}ms  TOTAL={gpu_total:.1}ms");
        println!("  Merkle roots match: {roots_match}");
        if cpu_total > 0.0 {
            println!("  Prover pipeline speedup: {:.1}x", cpu_total / gpu_total);
        }
        println!();
    }
}
