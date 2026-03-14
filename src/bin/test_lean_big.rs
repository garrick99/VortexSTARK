/// Quick test: prove_lean at large log_n values to verify VRAM fits.
use kraken_stark::field::M31;
use kraken_stark::prover;
use std::time::Instant;

fn main() {
    println!("prove_lean VRAM scaling test");
    println!("============================\n");

    for log_n in [20, 24, 26, 28, 29] {
        let n: u64 = 1u64 << log_n;
        let est_gb = (n * 60) as f64 / (1024.0 * 1024.0 * 1024.0);

        print!("log_n={log_n} (n={n}, ~{est_gb:.1}GB est)... ");

        let t0 = Instant::now();
        match std::panic::catch_unwind(|| {
            prover::prove_lean_timed(M31(1), M31(1), log_n)
        }) {
            Ok(proof) => {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                println!("OK in {ms:.0}ms, {} FRI layers, {} queries",
                    proof.fri_commitments.len(), proof.query_indices.len());

                // Quick verify
                match kraken_stark::verifier::verify(&proof) {
                    Ok(()) => println!("  -> verified OK"),
                    Err(e) => println!("  -> VERIFY FAILED: {e}"),
                }
            }
            Err(_) => {
                println!("FAILED (OOM or panic)");
                break;
            }
        }
        println!();
    }
}
