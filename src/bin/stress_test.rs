//! 15-minute GPU stress test with continuous throughput logging.
//! Runs Pedersen hash batches in a tight loop, reporting per-interval stats.
//! Designed to detect thermal throttling, power spikes, and sustained throughput degradation.

use std::time::{Duration, Instant};
use kraken_stark::cairo_air::pedersen::gpu_init;
use kraken_stark::cairo_air::pedersen::gpu_hash_batch;
use kraken_stark::cairo_air::stark252_field::Fp;
use kraken_stark::cuda::ffi;

fn main() {
    let test_duration = Duration::from_secs(15 * 60); // 15 minutes
    let report_interval = Duration::from_secs(10);    // report every 10 seconds
    let batch_size: usize = 100_000;                   // 100K per batch (sweet spot)
    let warmup_batches = 3;

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║          KRAKEN-STARK GPU STRESS TEST — RTX 5090 @ 450W CAP            ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Duration:       15 minutes                                            ║");
    println!("║  Batch size:     100,000 Pedersen hashes                               ║");
    println!("║  Report every:   10 seconds                                            ║");
    println!("║  Power cap:      450W (max 600W)                                       ║");
    println!("║  Kernel:         Windowed 4-bit scalar mul + mixed affine-Jacobian      ║");
    println!("║  Field:          Stark252 (p = 2^251 + 17*2^192 + 1)                   ║");
    println!("║  EC ops:         Montgomery mul, Jacobian projective coords             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Init
    println!("[INIT] Initializing CUDA memory pool...");
    ffi::init_memory_pool();
    println!("[INIT] Uploading Pedersen tables (windowed + P0 Montgomery)...");
    gpu_init();

    // Generate deterministic inputs
    println!("[INIT] Generating {batch_size} deterministic input pairs...");
    let inputs_a: Vec<Fp> = (0..batch_size).map(|i| {
        let seed = (i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x517CC1B727220A95);
        Fp::from_u64(seed)
    }).collect();
    let inputs_b: Vec<Fp> = (0..batch_size).map(|i| {
        let seed = (i as u64).wrapping_mul(0x6C62272E07BB0142).wrapping_add(0x62B821756295C58D);
        Fp::from_u64(seed)
    }).collect();

    // Correctness check — hash first pair on CPU and GPU
    println!("[INIT] Correctness check...");
    let gpu_first = gpu_hash_batch(&inputs_a[..1], &inputs_b[..1]);
    let cpu_first = kraken_stark::cairo_air::stark252_field::pedersen_hash(inputs_a[0], inputs_b[0]);
    if gpu_first[0] == cpu_first {
        println!("[INIT] Correctness: PASS (GPU == CPU for test vector)");
    } else {
        println!("[INIT] Correctness: FAIL — aborting!");
        println!("  CPU: [{:016x}, {:016x}, {:016x}, {:016x}]",
            cpu_first.v[0], cpu_first.v[1], cpu_first.v[2], cpu_first.v[3]);
        println!("  GPU: [{:016x}, {:016x}, {:016x}, {:016x}]",
            gpu_first[0].v[0], gpu_first[0].v[1], gpu_first[0].v[2], gpu_first[0].v[3]);
        return;
    }

    // Warmup
    println!("[WARMUP] Running {warmup_batches} warmup batches...");
    for _ in 0..warmup_batches {
        let _ = gpu_hash_batch(&inputs_a, &inputs_b);
    }
    println!("[WARMUP] Done. Starting sustained test.");
    println!();

    // Header
    println!("{:<8} {:>10} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "Time", "Batches", "Hashes", "Interval", "Sustained", "Peak", "Batch ms");
    println!("{:<8} {:>10} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "", "", "", "hash/sec", "hash/sec", "hash/sec", "");
    println!("{}", "─".repeat(80));

    let start = Instant::now();
    let mut interval_start = start;
    let mut total_hashes: u64 = 0;
    let mut interval_hashes: u64 = 0;
    let mut total_batches: u64 = 0;
    let mut interval_batches: u64 = 0;
    let mut peak_throughput: f64 = 0.0;
    let mut min_throughput: f64 = f64::MAX;
    let mut max_batch_ms: f64 = 0.0;
    let mut min_batch_ms: f64 = f64::MAX;
    let mut interval_count: u32 = 0;

    // Per-interval tracking for variance
    let mut interval_throughputs: Vec<f64> = Vec::new();

    loop {
        let elapsed = start.elapsed();
        if elapsed >= test_duration {
            break;
        }

        // Run one batch
        let batch_start = Instant::now();
        let results = gpu_hash_batch(&inputs_a, &inputs_b);
        let batch_elapsed = batch_start.elapsed();
        let batch_ms = batch_elapsed.as_secs_f64() * 1000.0;

        // Quick sanity: check last result isn't zero
        let last = &results[results.len() - 1];
        if last.v[0] == 0 && last.v[1] == 0 && last.v[2] == 0 && last.v[3] == 0 {
            println!("[ERROR] GPU returned zero result at batch {total_batches} — possible GPU error!");
        }

        total_hashes += batch_size as u64;
        interval_hashes += batch_size as u64;
        total_batches += 1;
        interval_batches += 1;

        if batch_ms > max_batch_ms { max_batch_ms = batch_ms; }
        if batch_ms < min_batch_ms { min_batch_ms = batch_ms; }

        // Report every interval
        let interval_elapsed = interval_start.elapsed();
        if interval_elapsed >= report_interval {
            interval_count += 1;
            let interval_secs = interval_elapsed.as_secs_f64();
            let interval_tp = interval_hashes as f64 / interval_secs;
            let sustained_tp = total_hashes as f64 / elapsed.as_secs_f64();

            if interval_tp > peak_throughput { peak_throughput = interval_tp; }
            if interval_tp < min_throughput { min_throughput = interval_tp; }
            interval_throughputs.push(interval_tp);

            let time_str = format!("{}:{:02}",
                elapsed.as_secs() / 60,
                elapsed.as_secs() % 60);

            println!("{:<8} {:>10} {:>12} {:>12.0} {:>12.0} {:>12.0} {:>10.1}",
                time_str,
                total_batches,
                total_hashes,
                interval_tp,
                sustained_tp,
                peak_throughput,
                batch_ms);

            interval_start = Instant::now();
            interval_hashes = 0;
            interval_batches = 0;
        }
    }

    let total_secs = start.elapsed().as_secs_f64();
    let sustained_tp = total_hashes as f64 / total_secs;

    println!("{}", "─".repeat(80));
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                         STRESS TEST RESULTS                             ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Total duration:      {:>8.1}s ({:.1} min){:<27}║",
        total_secs, total_secs / 60.0, "");
    println!("║  Total hashes:        {:>12}{:<30}║", total_hashes, "");
    println!("║  Total batches:       {:>12}{:<30}║", total_batches, "");
    println!("║  Batch size:          {:>12}{:<30}║", batch_size, "");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  THROUGHPUT{:<61}║", "");
    println!("║  Sustained avg:       {:>12.0} hashes/sec{:<20}║", sustained_tp, "");
    println!("║  Peak interval:       {:>12.0} hashes/sec{:<20}║", peak_throughput, "");
    println!("║  Min interval:        {:>12.0} hashes/sec{:<20}║", min_throughput, "");
    println!("║  Variance:            {:>11.2}%{:<30}║",
        if interval_throughputs.len() > 1 {
            let mean = interval_throughputs.iter().sum::<f64>() / interval_throughputs.len() as f64;
            let variance = interval_throughputs.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / interval_throughputs.len() as f64;
            let stddev = variance.sqrt();
            (stddev / mean) * 100.0
        } else { 0.0 }, "");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  BATCH LATENCY{:<58}║", "");
    println!("║  Min batch:           {:>10.1}ms{:<30}║", min_batch_ms, "");
    println!("║  Max batch:           {:>10.1}ms{:<30}║", max_batch_ms, "");
    println!("║  Avg batch:           {:>10.1}ms{:<30}║",
        (total_secs * 1000.0) / total_batches as f64, "");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  GPU SPEEDUP{:<60}║", "");
    println!("║  vs CPU (61 hash/s):  {:>10.0}x{:<31}║", sustained_tp / 61.0, "");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    // Throttle detection
    println!();
    if interval_throughputs.len() >= 6 {
        let first_3: f64 = interval_throughputs[..3].iter().sum::<f64>() / 3.0;
        let last_3: f64 = interval_throughputs[interval_throughputs.len()-3..].iter().sum::<f64>() / 3.0;
        let drift = ((last_3 - first_3) / first_3) * 100.0;
        println!("[ANALYSIS] First 30s avg: {:.0} hash/sec", first_3);
        println!("[ANALYSIS] Last  30s avg: {:.0} hash/sec", last_3);
        println!("[ANALYSIS] Throughput drift: {:+.2}%", drift);
        if drift.abs() < 2.0 {
            println!("[ANALYSIS] Verdict: STABLE — no thermal throttling detected");
        } else if drift < -5.0 {
            println!("[ANALYSIS] Verdict: THROTTLING — sustained throughput dropped {:.1}%", drift.abs());
        } else if drift > 2.0 {
            println!("[ANALYSIS] Verdict: WARMING UP — throughput increased {:.1}% (GPU boosted)", drift);
        } else {
            println!("[ANALYSIS] Verdict: MINOR FLUCTUATION — within normal range");
        }
    }

    // Spike detection
    if interval_throughputs.len() >= 3 {
        let mean = interval_throughputs.iter().sum::<f64>() / interval_throughputs.len() as f64;
        let mut spikes = 0;
        for (i, &tp) in interval_throughputs.iter().enumerate() {
            let deviation = ((tp - mean) / mean * 100.0).abs();
            if deviation > 5.0 {
                let time = (i + 1) * 10;
                println!("[SPIKE] Interval {} ({}:{:02}): {:.0} hash/sec ({:+.1}% from mean)",
                    i + 1, time / 60, time % 60, tp, (tp - mean) / mean * 100.0);
                spikes += 1;
            }
        }
        if spikes == 0 {
            println!("[ANALYSIS] No throughput spikes detected (all intervals within 5% of mean)");
        }
    }
}
