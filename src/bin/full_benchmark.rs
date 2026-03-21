/// Full system benchmark: all verified VortexSTARK components.
use vortexstark::cuda::ffi;
use vortexstark::field::M31;
use vortexstark::poseidon;
use vortexstark::cairo_air::{decode::Instruction, prover::cairo_prove, prover::cairo_verify};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           VORTEXSTARK FULL SYSTEM BENCHMARK                ║");
    println!("║              Rust + CUDA · Circle STARK                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Auto-detect GPU
    let gpu_info = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"])
        .output().ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "Unknown GPU".to_string());
    println!("  GPU: {gpu_info}\n");

    ffi::init_memory_pool_greedy();

    // Warmup
    let _ = vortexstark::prover::prove(M31(1), M31(1), 8);

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  FIBONACCI STARK (1 column, degree-1 constraint)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for log_n in [20, 24, 28, 29, 30] {
        let n: u64 = 1 << log_n;
        let t = Instant::now();
        let proof = vortexstark::prover::prove_lean(M31(1), M31(1), log_n);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let t2 = Instant::now();
        let ok = vortexstark::verifier::verify(&proof).is_ok();
        let verify_ms = t2.elapsed().as_secs_f64() * 1000.0;
        println!("  log_n={log_n:>2} | {n:>12} elements | prove: {ms:>8.1}ms | verify: {verify_ms:.1}ms | {}", if ok {"✓"} else {"✗"});
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  POSEIDON (8 cols, degree-5 S-box — trace+NTT throughput)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for log_n in [20, 24, 28] {
        let n: usize = 1 << log_n;
        let n_hashes = n / poseidon::NUM_ROUNDS;

        let t = Instant::now();
        let (d_cols, _, _) = poseidon::generate_trace_gpu(log_n);
        let trace_ms = t.elapsed().as_secs_f64() * 1000.0;

        let eval_size = 2 * n;
        let log_eval = log_n + 1;
        let trace_domain = vortexstark::circle::Coset::half_coset(log_n);
        let eval_domain = vortexstark::circle::Coset::half_coset(log_eval);
        let inv = vortexstark::ntt::InverseTwiddleCache::new(&trace_domain);
        let fwd = vortexstark::ntt::ForwardTwiddleCache::new(&eval_domain);

        let t2 = Instant::now();
        let mut d_eval_cols = Vec::new();
        for mut col in d_cols {
            vortexstark::ntt::interpolate(&mut col, &inv);
            let mut d_eval = vortexstark::device::DeviceBuffer::<u32>::alloc(eval_size);
            unsafe { ffi::cuda_zero_pad(col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32); }
            drop(col);
            vortexstark::ntt::evaluate(&mut d_eval, &fwd);
            d_eval_cols.push(d_eval);
        }
        let ntt_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;
        let hashes_per_sec = n_hashes as f64 / (total_ms / 1000.0);

        println!("  log_n={log_n:>2} | {n_hashes:>12} hashes | trace: {trace_ms:>6.0}ms | NTT: {ntt_ms:>6.0}ms | total: {total_ms:>8.1}ms | {:.1}M hash/s", hashes_per_sec / 1e6);
        drop(d_eval_cols);
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  CAIRO VM STARK (31 columns, 31 constraints, LogUp + range checks)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for log_n in [20, 24, 26] {
        let n: usize = 1 << log_n;

        // Build fib program
        let assert_imm = Instruction { off0: 0x8000, off1: 0x8000, off2: 0x8001, op1_imm: 1, opcode_assert: 1, ap_add1: 1, ..Default::default() };
        let add_instr = Instruction { off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1, op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1, ..Default::default() };
        let mut program = Vec::new();
        program.push(assert_imm.encode()); program.push(1);
        program.push(assert_imm.encode()); program.push(1);
        for _ in 0..n-2 { program.push(add_instr.encode()); }

        let t = Instant::now();
        let proof = cairo_prove(&program, n, log_n);
        let prove_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let ok = cairo_verify(&proof).is_ok();
        let verify_ms = t2.elapsed().as_secs_f64() * 1000.0;

        println!("  log_n={log_n:>2} | {n:>12} steps | prove: {prove_ms:>8.1}ms | verify: {verify_ms:>6.1}ms | {}", if ok { "✓" } else { "✗" });
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PEDERSEN HASH (CPU, STARK curve EC, projective coordinates)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let t = Instant::now();
    let n_ped = 100;
    for i in 0..n_ped {
        let _ = vortexstark::cairo_air::stark252_field::pedersen_hash(
            vortexstark::cairo_air::stark252_field::Fp::from_u64(i + 1),
            vortexstark::cairo_air::stark252_field::Fp::from_u64(i + 100),
        );
    }
    let ped_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("  {n_ped} hashes: {ped_ms:.1}ms ({:.1}ms/hash, {:.0} hashes/sec)",
        ped_ms / n_ped as f64, n_ped as f64 / (ped_ms / 1000.0));

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  SYSTEM SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Engine:     GPU-native Circle STARK (M31, 100-bit security)");
    println!("  GPU:        {gpu_info}");
  
    println!("  Builtins:   Poseidon (GPU), Pedersen (GPU), Bitwise");
    println!("  Features:   LogUp memory consistency, range checks, 2-phase commitment");
    println!("  CLI:        stark_cli prove/verify (binary proof serialization)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
