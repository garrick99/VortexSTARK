use kraken_stark::circle::Coset;
use kraken_stark::cuda::ffi;
use kraken_stark::device::DeviceBuffer;
use kraken_stark::ntt::{self, TwiddleCache};
use std::time::Instant;

fn main() {
    println!("kraken-stark GPU benchmark");
    println!("==========================\n");

    // Warmup GPU
    let _ = DeviceBuffer::<u32>::alloc(1);
    unsafe { ffi::cuda_device_sync() };

    // --- Field ops ---
    let n: u32 = 1 << 20;
    let a: Vec<u32> = (0..n).map(|i| i % 0x7FFF_FFFF).collect();
    let b: Vec<u32> = (0..n).map(|i| (i * 3) % 0x7FFF_FFFF).collect();
    let d_a = DeviceBuffer::from_host(&a);
    let d_b = DeviceBuffer::from_host(&b);
    let mut d_out = DeviceBuffer::<u32>::alloc(n as usize);

    let t0 = Instant::now();
    unsafe {
        ffi::cuda_m31_mul(d_a.as_ptr(), d_b.as_ptr(), d_out.as_mut_ptr(), n);
        ffi::cuda_device_sync();
    }
    let mul_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("[OK] M31 mul: {n} elements in {mul_ms:.3}ms");

    // --- NTT benchmarks ---
    for log_n in [12u32, 16, 20] {
        let size = 1usize << log_n;
        let coset = Coset::half_coset(log_n);
        let cache = TwiddleCache::new(&coset);

        let coeffs: Vec<u32> = (0..size).map(|i| (i as u32 * 7 + 13) % 0x7FFF_FFFF).collect();
        let mut d_data = DeviceBuffer::from_host(&coeffs);

        // Warmup
        ntt::evaluate(&mut d_data, &cache);
        ntt::interpolate(&mut d_data, &cache);

        // Benchmark forward NTT
        let iters = if log_n <= 16 { 100 } else { 10 };
        let t0 = Instant::now();
        for _ in 0..iters {
            ntt::evaluate(&mut d_data, &cache);
        }
        let fwd_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

        // Benchmark inverse NTT
        let t0 = Instant::now();
        for _ in 0..iters {
            ntt::interpolate(&mut d_data, &cache);
        }
        let inv_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

        // Verify roundtrip
        let mut d_check = DeviceBuffer::from_host(&coeffs);
        ntt::evaluate(&mut d_check, &cache);
        ntt::interpolate(&mut d_check, &cache);
        let result = d_check.to_host();
        let ok = result == coeffs;

        println!(
            "[{}] NTT log_n={log_n} (n={}): fwd {fwd_us:.1}us, inv {inv_us:.1}us",
            if ok { "OK" } else { "FAIL" },
            size,
        );
    }

    // --- Batch NTT ---
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

    // Warmup
    ntt::evaluate_batch(&mut columns, &cache);
    ntt::interpolate_batch(&mut columns, &cache);

    let iters = 50;
    let t0 = Instant::now();
    for _ in 0..iters {
        ntt::evaluate_batch(&mut columns, &cache);
    }
    let batch_fwd_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

    let t0 = Instant::now();
    for _ in 0..iters {
        ntt::interpolate_batch(&mut columns, &cache);
    }
    let batch_inv_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

    println!(
        "[OK] Batch NTT {n_cols}x log_n={log_n}: fwd {batch_fwd_us:.1}us, inv {batch_inv_us:.1}us"
    );

    // --- STARK prover benchmark ---
    println!("\n--- STARK Prover ---");
    for log_n in [8u32, 12, 16, 20] {
        let a = kraken_stark::field::M31(1);
        let b = kraken_stark::field::M31(1);

        // Warmup
        let _ = kraken_stark::prover::prove(a, b, log_n);

        let t0 = Instant::now();
        let proof = kraken_stark::prover::prove(a, b, log_n);
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!(
            "[OK] prove log_n={log_n} (n={}): {prove_ms:.1}ms, {} FRI layers",
            1u32 << log_n,
            proof.fri_commitments.len(),
        );
    }

    // Detailed timing for log_n=16
    println!("\n--- Detailed Profile (log_n=16) ---");
    let _ = kraken_stark::prover::prove_timed(
        kraken_stark::field::M31(1),
        kraken_stark::field::M31(1),
        16,
    );

    println!("\nDone.");
}
