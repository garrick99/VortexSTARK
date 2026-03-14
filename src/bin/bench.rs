use kraken_stark::circle::Coset;
use kraken_stark::cuda::ffi;
use kraken_stark::device::DeviceBuffer;
use kraken_stark::field::M31;
use kraken_stark::ntt::{self, TwiddleCache};
use std::time::Instant;

/// Benchmark statistics for a set of timings (in milliseconds).
struct Stats {
    median: f64,
    min: f64,
    max: f64,
    mean: f64,
    stddev: f64,
    samples: usize,
}

fn compute_stats(times_ms: &mut Vec<f64>) -> Stats {
    let n = times_ms.len();
    assert!(n > 0);
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times_ms[0];
    let max = times_ms[n - 1];
    let median = if n % 2 == 1 {
        times_ms[n / 2]
    } else {
        (times_ms[n / 2 - 1] + times_ms[n / 2]) / 2.0
    };
    let mean = times_ms.iter().sum::<f64>() / n as f64;
    let variance = times_ms.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n as f64;
    let stddev = variance.sqrt();
    Stats { median, min, max, mean, stddev, samples: n }
}

fn print_stats(label: &str, stats: &Stats) {
    println!(
        "  {label:<24} median {:.3}ms  min {:.3}ms  max {:.3}ms  mean {:.3}ms  stddev {:.3}ms  (n={})",
        stats.median, stats.min, stats.max, stats.mean, stats.stddev, stats.samples,
    );
}

fn main() {
    println!("kraken-stark robust benchmark");
    println!("==============================");
    println!("GPU: RTX 5090 (SM 12.0), CUDA 13.0\n");

    // --- Init CUDA memory pool (async alloc/free, zero overhead reuse) ---
    ffi::init_memory_pool();

    // --- GPU warmup ---
    let _ = DeviceBuffer::<u32>::alloc(1024);
    unsafe { ffi::cuda_device_sync() };
    // Run a throwaway prove to fully warm the GPU (context, caches, JIT)
    let _ = kraken_stark::prover::prove(M31(1), M31(1), 8);

    // =============================================
    // Field operations
    // =============================================
    println!("=== Field Operations ===");
    {
        let n: u32 = 1 << 20;
        let a: Vec<u32> = (0..n).map(|i| i % 0x7FFF_FFFF).collect();
        let b: Vec<u32> = (0..n).map(|i| (i * 3) % 0x7FFF_FFFF).collect();
        let d_a = DeviceBuffer::from_host(&a);
        let d_b = DeviceBuffer::from_host(&b);
        let mut d_out = DeviceBuffer::<u32>::alloc(n as usize);

        // warmup
        unsafe { ffi::cuda_m31_mul(d_a.as_ptr(), d_b.as_ptr(), d_out.as_mut_ptr(), n); ffi::cuda_device_sync(); }

        let iters = 200;
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            unsafe {
                ffi::cuda_m31_mul(d_a.as_ptr(), d_b.as_ptr(), d_out.as_mut_ptr(), n);
                ffi::cuda_device_sync();
            }
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let stats = compute_stats(&mut times);
        let throughput = (n as f64 / 1e6) / (stats.median / 1000.0);
        print_stats(&format!("M31 mul ({n} elems)"), &stats);
        println!("  throughput: {throughput:.0} M elem/s\n");
    }

    // =============================================
    // NTT benchmarks
    // =============================================
    println!("=== NTT (Circle NTT) ===");
    for log_n in [12u32, 16, 20] {
        let size = 1usize << log_n;
        let coset = Coset::half_coset(log_n);
        let cache = TwiddleCache::new(&coset);

        let coeffs: Vec<u32> = (0..size).map(|i| (i as u32 * 7 + 13) % 0x7FFF_FFFF).collect();
        let mut d_data = DeviceBuffer::from_host(&coeffs);

        // warmup
        ntt::evaluate(&mut d_data, &cache);
        ntt::interpolate(&mut d_data, &cache);

        let iters = if log_n <= 16 { 200 } else { 50 };

        let mut fwd_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            ntt::evaluate(&mut d_data, &cache);
            unsafe { ffi::cuda_device_sync(); }
            fwd_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        let mut inv_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            ntt::interpolate(&mut d_data, &cache);
            unsafe { ffi::cuda_device_sync(); }
            inv_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        let fwd = compute_stats(&mut fwd_times);
        let inv = compute_stats(&mut inv_times);
        println!("  log_n={log_n} (n={size}):");
        print_stats("  forward", &fwd);
        print_stats("  inverse", &inv);
        println!();
    }

    // =============================================
    // Batch NTT
    // =============================================
    println!("=== Batch NTT (8 columns) ===");
    {
        let log_n = 16u32;
        let size = 1usize << log_n;
        let coset = Coset::half_coset(log_n);
        let cache = TwiddleCache::new(&coset);
        let n_cols = 8;

        let originals: Vec<Vec<u32>> = (0..n_cols)
            .map(|c| (0..size).map(|i| ((i * (c + 3) + 17) % 0x7FFF_FFFF as usize) as u32).collect())
            .collect();
        let mut columns: Vec<DeviceBuffer<u32>> = originals
            .iter()
            .map(|v| DeviceBuffer::from_host(v))
            .collect();

        // warmup
        ntt::evaluate_batch(&mut columns, &cache);
        ntt::interpolate_batch(&mut columns, &cache);

        let iters = 100;
        let mut fwd_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            ntt::evaluate_batch(&mut columns, &cache);
            unsafe { ffi::cuda_device_sync(); }
            fwd_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut inv_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            ntt::interpolate_batch(&mut columns, &cache);
            unsafe { ffi::cuda_device_sync(); }
            inv_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let fwd = compute_stats(&mut fwd_times);
        let inv = compute_stats(&mut inv_times);
        println!("  log_n={log_n} (n={size}), {n_cols} columns:");
        print_stats("  forward", &fwd);
        print_stats("  inverse", &inv);
        println!();
    }

    // =============================================
    // STARK prover — sustained throughput
    // =============================================
    println!("=== STARK Prover (sustained) ===");
    for log_n in [8u32, 12, 16, 20] {
        let a = M31(1);
        let b = M31(1);
        let n_elem = 1u32 << log_n;

        // warmup (2 runs)
        let _ = kraken_stark::prover::prove(a, b, log_n);
        let _ = kraken_stark::prover::prove(a, b, log_n);

        let iters = match log_n {
            20 => 20,
            16 => 50,
            _ => 100,
        };

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _proof = kraken_stark::prover::prove(a, b, log_n);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let stats = compute_stats(&mut times);
        let proofs_per_sec = 1000.0 / stats.median;
        print_stats(&format!("log_n={log_n} (n={n_elem})"), &stats);
        println!("  → {proofs_per_sec:.1} proofs/sec\n");
    }

    // =============================================
    // STARK prover — cached (amortized twiddle setup)
    // =============================================
    println!("=== STARK Prover (cached, amortized setup) ===");
    for log_n in [8u32, 12, 16, 20] {
        let a = M31(1);
        let b = M31(1);
        let n_elem = 1u32 << log_n;
        let cache = kraken_stark::prover::ProverCache::new(log_n);

        // warmup
        let _ = kraken_stark::prover::prove_cached(a, b, &cache);
        let _ = kraken_stark::prover::prove_cached(a, b, &cache);

        let iters = match log_n {
            20 => 30,
            16 => 100,
            _ => 200,
        };

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _proof = kraken_stark::prover::prove_cached(a, b, &cache);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let stats = compute_stats(&mut times);
        let proofs_per_sec = 1000.0 / stats.median;
        print_stats(&format!("log_n={log_n} (n={n_elem})"), &stats);
        println!("  → {proofs_per_sec:.1} proofs/sec\n");
    }

    // =============================================
    // STARK prover — pipelined (overlapped trace gen)
    // =============================================
    println!("=== STARK Prover (pipelined, overlapped trace gen) ===");
    for log_n in [8u32, 12, 16, 20] {
        let n_elem = 1u32 << log_n;
        let pipeline = kraken_stark::prover::ProverPipeline::new(log_n);

        // warmup
        let _ = pipeline.prove_batch(&[(M31(1), M31(1)); 3]);

        let batch_size = match log_n {
            20 => 30,
            16 => 100,
            _ => 200,
        };

        let inputs: Vec<(M31, M31)> = (0..batch_size)
            .map(|i| (M31((i + 1) as u32), M31((i + 2) as u32)))
            .collect();

        let t0 = Instant::now();
        let _proofs = pipeline.prove_batch(&inputs);
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let per_proof = total_ms / batch_size as f64;
        let proofs_per_sec = 1000.0 / per_proof;

        println!(
            "  log_n={log_n} (n={n_elem}){:>8}  {per_proof:.3}ms/proof  {proofs_per_sec:.1} proofs/sec  (batch={batch_size})",
            ""
        );
    }
    println!();

    // =============================================
    // Detailed profile (single-run, timed stages)
    // =============================================
    for profile_log_n in [16u32, 20] {
        println!("=== Stage Profile uncached (log_n={profile_log_n}) ===");
        let _ = kraken_stark::prover::prove_timed(M31(1), M31(1), profile_log_n);
        println!();
    }
    for profile_log_n in [16u32, 20] {
        println!("=== Stage Profile cached (log_n={profile_log_n}) ===");
        let cache = kraken_stark::prover::ProverCache::new(profile_log_n);
        let _ = kraken_stark::prover::prove_cached(M31(1), M31(1), &cache); // warmup
        let _ = kraken_stark::prover::prove_cached_timed(M31(1), M31(1), &cache);
        println!();
    }

    // =============================================
    // Summary table
    // =============================================
    println!("=== Summary (uncached / cached / pipelined) ===");
    println!("  {:>8}  {:>14}  {:>14}  {:>14}  {:>14}  {:>14}  {:>14}", "log_n", "uncached (ms)", "cached (ms)", "pipeline (ms)", "uncached p/s", "cached p/s", "pipeline p/s");
    println!("  {:>8}  {:>14}  {:>14}  {:>14}  {:>14}  {:>14}  {:>14}", "-----", "-------------", "-----------", "-------------", "------------", "----------", "------------");
    for log_n in [8u32, 12, 16, 20] {
        let a = M31(1);
        let b = M31(1);
        let cache = kraken_stark::prover::ProverCache::new(log_n);
        let pipeline = kraken_stark::prover::ProverPipeline::new(log_n);

        // warmup
        let _ = kraken_stark::prover::prove(a, b, log_n);
        let _ = kraken_stark::prover::prove_cached(a, b, &cache);
        let _ = pipeline.prove_batch(&[(a, b); 3]);

        let iters = match log_n {
            20 => 30,
            16 => 100,
            _ => 200,
        };

        let mut uncached_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _proof = kraken_stark::prover::prove(a, b, log_n);
            uncached_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut cached_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _proof = kraken_stark::prover::prove_cached(a, b, &cache);
            cached_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // Pipeline batch
        let inputs: Vec<(M31, M31)> = (0..iters)
            .map(|i| (M31((i + 1) as u32), M31((i + 2) as u32)))
            .collect();
        let t0 = Instant::now();
        let _proofs = pipeline.prove_batch(&inputs);
        let total_pipeline_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let pipeline_per_proof = total_pipeline_ms / iters as f64;

        let u = compute_stats(&mut uncached_times);
        let c = compute_stats(&mut cached_times);
        println!(
            "  {:>8}  {:>14.3}  {:>14.3}  {:>14.3}  {:>14.1}  {:>14.1}  {:>14.1}",
            log_n, u.median, c.median, pipeline_per_proof,
            1000.0 / u.median, 1000.0 / c.median, 1000.0 / pipeline_per_proof,
        );
    }

    // =============================================
    // Scaling curve: push log_n until we hit limits
    // =============================================
    println!("\n=== Scaling Curve (cached) ===");
    println!("  {:>8}  {:>10}  {:>14}  {:>14}  {:>14}  {:>14}",
        "log_n", "n", "median (ms)", "proofs/sec", "M elem/s", "notes");
    println!("  {:>8}  {:>10}  {:>14}  {:>14}  {:>14}  {:>14}",
        "-----", "---", "-----------", "----------", "---------", "-----");

    for log_n in 8..=28u32 {
        let n: u64 = 1u64 << log_n;

        // Estimate memory: ~60 bytes per element (trace + blowup + FRI SoA + merkle + twiddles)
        let est_bytes = n * 60;
        let est_gb = est_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        if est_gb > 28.0 {
            println!("  {:>8}  {:>10}  {:>14}  {:>14}  {:>14}  skipped (~{:.1}GB > 28GB VRAM)",
                log_n, format_n(n), "-", "-", "-", est_gb);
            continue;
        }

        let a = M31(1);
        let b = M31(1);

        // Try to create cache; if it fails (OOM), stop gracefully
        let cache = match std::panic::catch_unwind(|| {
            kraken_stark::prover::ProverCache::new(log_n)
        }) {
            Ok(c) => c,
            Err(_) => {
                println!("  {:>8}  {:>10}  {:>14}  {:>14}  {:>14}  OOM at cache alloc",
                    log_n, format_n(n), "-", "-", "-");
                break;
            }
        };

        // warmup — also catch OOM during prove
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = kraken_stark::prover::prove_cached(a, b, &cache);
            let _ = kraken_stark::prover::prove_cached(a, b, &cache);
        })).is_err() {
            println!("  {:>8}  {:>10}  {:>14}  {:>14}  {:>14}  OOM during prove",
                log_n, format_n(n), "-", "-", "-");
            break;
        }

        let iters = if log_n >= 26 {
            3
        } else if log_n >= 24 {
            5
        } else if log_n >= 22 {
            10
        } else if log_n >= 20 {
            20
        } else if log_n >= 16 {
            50
        } else {
            100
        };

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _proof = kraken_stark::prover::prove_cached(a, b, &cache);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let stats = compute_stats(&mut times);
        let proofs_per_sec = 1000.0 / stats.median;
        let m_elem_per_sec = (n as f64 / 1e6) / (stats.median / 1000.0);
        let note = if est_gb > 1.0 {
            format!("~{:.1}GB", est_gb)
        } else {
            String::new()
        };
        println!("  {:>8}  {:>10}  {:>14.3}  {:>14.1}  {:>14.1}  {}",
            log_n, format_n(n), stats.median, proofs_per_sec, m_elem_per_sec, note);
    }

    println!("\nDone.");
}

fn format_n(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}
