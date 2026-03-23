//! Three-way hash benchmark: Poseidon2 (30 rows) vs RPO-M31 (14 rows) vs Poseidon2-Full (8 rows).
use vortexstark::{cuda::ffi, poseidon, rpo_m31, poseidon2f};
use vortexstark::ntt::{InverseTwiddleCache, ForwardTwiddleCache, interpolate, evaluate};
use vortexstark::circle::Coset;
use vortexstark::device::DeviceBuffer;
use std::time::Instant;

fn bench_hash(
    name: &str, cols: usize, rows_per_perm: usize,
    run: impl Fn(u32) -> Vec<DeviceBuffer<u32>>,
) {
    println!("\n  {name} ({cols} cols, {rows_per_perm} rows/perm — product {})", cols * rows_per_perm);
    println!("  {}", "─".repeat(70));
    for log_n in [20u32, 24, 28] {
        let n = 1usize << log_n;
        let n_hashes = n / rows_per_perm;

        let t0 = Instant::now();
        let d_cols = run(log_n);
        let trace_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let log_eval = log_n + 1;
        let eval_n   = 2 * n;
        let td = Coset::half_coset(log_n);
        let ed = Coset::half_coset(log_eval);
        let inv = InverseTwiddleCache::new(&td);
        let fwd = ForwardTwiddleCache::new(&ed);

        // Drop each eval col immediately — accumulating all cols would OOM for
        // wide traces (RPO: 24 cols × 2 GB at log_n=28 = 48 GB).
        let t1 = Instant::now();
        for mut col in d_cols {
            interpolate(&mut col, &inv);
            let mut ev = DeviceBuffer::<u32>::alloc(eval_n);
            unsafe { ffi::cuda_zero_pad(col.as_ptr(), ev.as_mut_ptr(), n as u32, eval_n as u32); }
            drop(col);
            evaluate(&mut ev, &fwd);
            // ev drops here
        }
        let ntt_ms  = t1.elapsed().as_secs_f64() * 1000.0;
        let total   = t0.elapsed().as_secs_f64() * 1000.0;
        let mhps    = n_hashes as f64 / (total / 1000.0) / 1e6;

        println!("  log_n={log_n:>2} | {n_hashes:>12} hashes | \
            trace {trace_ms:>6.0}ms | NTT {ntt_ms:>6.0}ms | \
            total {total:>8.1}ms | {mhps:.2}M hash/s");
    }
}

fn main() {
    let gpu = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.free,memory.total", "--format=csv,noheader"])
        .output().ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "Unknown GPU".to_string());
    println!("GPU: {gpu}");

    ffi::init_memory_pool_greedy();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  HASH THROUGHPUT BENCHMARK  (trace + NTT, RTX 5090)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // ── Poseidon2 (baseline) ───────────────────────────────────────────────
    bench_hash("Poseidon2 [baseline]", poseidon::STATE_WIDTH, poseidon::NUM_ROUNDS, |log_n| {
        let (d, _, _) = poseidon::generate_trace_gpu(log_n);
        d
    });

    // ── RPO-M31 ───────────────────────────────────────────────────────────
    bench_hash("RPO-M31", rpo_m31::STATE_WIDTH, rpo_m31::ROWS_PER_PERM, |log_n| {
        rpo_m31::generate_trace_gpu(log_n)
    });

    // ── Poseidon2-Full (experimental) ─────────────────────────────────────
    bench_hash("Poseidon2-Full [EXPERIMENTAL — RF=8, RP=0]",
        poseidon2f::ROWS_PER_PERM, poseidon2f::ROWS_PER_PERM,
        |log_n| poseidon2f::generate_trace_gpu(log_n));

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  cols×rows product: P2={}, RPO={}, P2F={}",
        poseidon::STATE_WIDTH * poseidon::NUM_ROUNDS,
        rpo_m31::STATE_WIDTH * rpo_m31::ROWS_PER_PERM,
        poseidon2f::ROWS_PER_PERM * poseidon2f::ROWS_PER_PERM);
    println!("  P2F vs P2:   {:.2}× better VRAM efficiency",
        (poseidon::STATE_WIDTH * poseidon::NUM_ROUNDS) as f64
        / (poseidon2f::ROWS_PER_PERM * poseidon2f::ROWS_PER_PERM) as f64);
}
