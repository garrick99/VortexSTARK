//! Standalone RPO-M31 benchmark.
use vortexstark::{cuda::ffi, rpo_m31, poseidon};
use vortexstark::ntt::{InverseTwiddleCache, ForwardTwiddleCache, interpolate, evaluate};
use vortexstark::circle::Coset;
use vortexstark::device::DeviceBuffer;
use std::time::Instant;

fn main() {
    let gpu_info = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.free,memory.total,compute_cap", "--format=csv,noheader"])
        .output().ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "Unknown GPU".to_string());
    println!("GPU: {gpu_info}\n");

    ffi::init_memory_pool_greedy();

    // ── RPO-M31 ────────────────────────────────────────────────────────────
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  RPO-M31 (24 cols, 14 rows/perm)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for log_n in [20u32, 24, 28] {
        let n: usize = 1 << log_n;
        let n_hashes = n / rpo_m31::ROWS_PER_PERM;

        let t = Instant::now();
        let d_cols = rpo_m31::generate_trace_gpu(log_n);
        let trace_ms = t.elapsed().as_secs_f64() * 1000.0;

        let log_eval = log_n + 1;
        let eval_size = 2 * n;
        let trace_domain = Coset::half_coset(log_n);
        let eval_domain  = Coset::half_coset(log_eval);
        let inv_cache = InverseTwiddleCache::new(&trace_domain);
        let fwd_cache = ForwardTwiddleCache::new(&eval_domain);

        // NTT: process one column at a time, dropping each eval col immediately.
        // Do NOT accumulate all eval cols — for RPO at log_n=28 that would be
        // 24 × 2 GB = 48 GB simultaneously, which exceeds 32 GB VRAM.
        // Peak with this pattern: trace_all (24 GB) + one_eval (2 GB) = 26 GB.
        let t2 = Instant::now();
        for mut col in d_cols {
            interpolate(&mut col, &inv_cache);
            let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
            unsafe { ffi::cuda_zero_pad(col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32); }
            drop(col);
            evaluate(&mut d_eval, &fwd_cache);
            // d_eval drops here — only one eval col live at a time
        }
        let ntt_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;
        let mhps = (n_hashes as f64) / (total_ms / 1000.0) / 1e6;

        println!("  log_n={log_n:>2} | {n_hashes:>12} hashes | trace: {trace_ms:>6.0}ms | NTT: {ntt_ms:>6.0}ms | total: {total_ms:>8.1}ms | {mhps:.2}M hash/s");
    }

    // ── Poseidon2 comparison ────────────────────────────────────────────────
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Poseidon2 (8 cols, 30 rows/perm) — comparison");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for log_n in [20u32, 24, 28] {
        let n: usize = 1 << log_n;
        let n_hashes = n / poseidon::NUM_ROUNDS;

        let t = Instant::now();
        let (d_cols, _, _) = poseidon::generate_trace_gpu(log_n);
        let trace_ms = t.elapsed().as_secs_f64() * 1000.0;

        let log_eval = log_n + 1;
        let eval_size = 2 * n;
        let trace_domain = Coset::half_coset(log_n);
        let eval_domain  = Coset::half_coset(log_eval);
        let inv_cache = InverseTwiddleCache::new(&trace_domain);
        let fwd_cache = ForwardTwiddleCache::new(&eval_domain);

        let t2 = Instant::now();
        for mut col in d_cols {
            interpolate(&mut col, &inv_cache);
            let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
            unsafe { ffi::cuda_zero_pad(col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32); }
            drop(col);
            evaluate(&mut d_eval, &fwd_cache);
        }
        let ntt_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;
        let mhps = (n_hashes as f64) / (total_ms / 1000.0) / 1e6;

        println!("  log_n={log_n:>2} | {n_hashes:>12} hashes | trace: {trace_ms:>6.0}ms | NTT: {ntt_ms:>6.0}ms | total: {total_ms:>8.1}ms | {mhps:.2}M hash/s");
    }
}
