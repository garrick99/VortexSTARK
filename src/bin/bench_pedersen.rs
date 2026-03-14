/// Pedersen hash benchmark: CPU vs GPU throughput.
use kraken_stark::cairo_air::pedersen;
use kraken_stark::cairo_air::stark252_field::{Fp, pedersen_hash};
use kraken_stark::cuda::ffi;
use std::time::Instant;

fn main() {
    println!("Pedersen Hash Benchmark: CPU vs GPU");
    println!("====================================\n");

    ffi::init_memory_pool();

    // === CPU benchmark ===
    println!("--- CPU (projective coordinates) ---");
    let _ = pedersen_hash(Fp::from_u64(1), Fp::from_u64(2)); // warmup

    let n_cpu = 100;
    let t0 = Instant::now();
    for i in 0..n_cpu {
        let _ = pedersen_hash(Fp::from_u64(i as u64 + 1), Fp::from_u64(i as u64 + 100));
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let cpu_per = cpu_ms / n_cpu as f64;
    println!("{n_cpu} hashes: {cpu_ms:.1}ms ({cpu_per:.2}ms/hash, {:.0} hashes/sec)\n",
        1000.0 / cpu_per);

    // === GPU benchmark ===
    println!("--- GPU (parallel, projective, CUDA) ---");
    pedersen::gpu_init();

    // Warmup
    let warmup_a = vec![Fp::from_u64(1)];
    let warmup_b = vec![Fp::from_u64(2)];
    let _ = pedersen::gpu_hash_batch(&warmup_a, &warmup_b);

    for n in [10, 100, 1000] {
        let inputs_a: Vec<Fp> = (0..n).map(|i| Fp::from_u64(i as u64 + 1)).collect();
        let inputs_b: Vec<Fp> = (0..n).map(|i| Fp::from_u64(i as u64 + 1000)).collect();

        let t0 = Instant::now();
        let results = pedersen::gpu_hash_batch(&inputs_a, &inputs_b);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let rate = n as f64 / (ms / 1000.0);

        // Verify first result matches CPU
        let cpu_check = pedersen_hash(inputs_a[0], inputs_b[0]);
        let ok = results[0] == cpu_check;

        println!("{n:>6} hashes: {ms:>8.1}ms ({:.0} hashes/sec) {}",
            rate, if ok { "✓" } else { "MISMATCH!" });
    }

    // Debug: show first hash values
    let a0 = Fp::from_u64(1);
    let b0 = Fp::from_u64(1000);
    let cpu_hash = pedersen_hash(a0, b0);
    let gpu_results = pedersen::gpu_hash_batch(&[a0], &[b0]);
    println!("\nDebug - input a=1, b=1000:");
    println!("  CPU: [{:016x}, {:016x}, {:016x}, {:016x}]",
        cpu_hash.v[0], cpu_hash.v[1], cpu_hash.v[2], cpu_hash.v[3]);
    println!("  GPU: [{:016x}, {:016x}, {:016x}, {:016x}]",
        gpu_results[0].v[0], gpu_results[0].v[1], gpu_results[0].v[2], gpu_results[0].v[3]);
    println!("  Match: {}", cpu_hash == gpu_results[0]);

    println!("\nNote: GPU includes upload + kernel + download time.");
}
