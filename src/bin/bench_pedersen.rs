/// Pedersen hash benchmark: measures throughput of real EC-based Pedersen.
use kraken_stark::cairo_air::pedersen::{PedersenBuiltin, Stark252};
use kraken_stark::cairo_air::stark252_field::{Fp, pedersen_hash};
use std::time::Instant;

fn main() {
    println!("Pedersen Hash Benchmark");
    println!("========================\n");

    // Warm up
    let _ = pedersen_hash(Fp::from_u64(1), Fp::from_u64(2));

    // Single hash timing
    let t0 = Instant::now();
    let n_single = 100;
    for i in 0..n_single {
        let _ = pedersen_hash(Fp::from_u64(i as u64 + 1), Fp::from_u64(i as u64 + 100));
    }
    let single_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_hash_ms = single_ms / n_single as f64;
    println!("Single hash: {per_hash_ms:.2}ms ({:.0} hashes/sec)", 1000.0 / per_hash_ms);

    // Builtin invocation benchmark
    for n in [10, 100, 1000] {
        let mut builtin = PedersenBuiltin::new();
        let t0 = Instant::now();
        for i in 0..n {
            let a = Stark252::from_hex(&format!("{:064x}", i + 1));
            let b = Stark252::from_hex(&format!("{:064x}", i + 1000));
            builtin.invoke(a, b);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let rate = n as f64 / (ms / 1000.0);
        println!("{n:>5} hashes: {ms:>8.1}ms ({rate:.0} hashes/sec)");
    }

    println!("\nNote: this is CPU-only EC arithmetic (4 scalar muls per hash).");
    println!("GPU acceleration would use precomputed point tables + parallel scalar mul.");
}
