/// Full system benchmark: all verified VortexSTARK components.
use vortexstark::cuda::ffi;
use vortexstark::field::M31;
use vortexstark::poseidon;
use vortexstark::cairo_air::{decode::Instruction, prover::cairo_prove, prover::cairo_verify};
use vortexstark::rpo_m31;
use vortexstark::poseidon2f;
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

    // Wake the GPU: nvidia-smi query forces the driver out of idle/P8 state
    // before any CUDA work begins, eliminating first-run latency spikes.
    let _ = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=power.draw", "--format=csv,noheader"])
        .output();

    // Warmup
    let _ = vortexstark::prover::prove(M31(1), M31(1), 8);

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  FIBONACCI STARK (1 column, degree-1 constraint)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let max_fib_log_n: u32 = std::env::var("VORTEX_MAX_LOG_N").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(30);
    for log_n in [20u32, 24, 28, 29, 30].iter().copied().filter(|&n| n <= max_fib_log_n) {
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

        // Drop each eval col immediately — do NOT accumulate. Wide traces (RPO: 24 cols,
        // Cairo: 31 cols) would OOM if all eval cols were kept alive simultaneously.
        let t2 = Instant::now();
        for mut col in d_cols {
            vortexstark::ntt::interpolate(&mut col, &inv);
            let mut d_eval = vortexstark::device::DeviceBuffer::<u32>::alloc(eval_size);
            unsafe { ffi::cuda_zero_pad(col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32); }
            drop(col);
            vortexstark::ntt::evaluate(&mut d_eval, &fwd);
            // d_eval drops here
        }
        let ntt_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;
        let hashes_per_sec = n_hashes as f64 / (total_ms / 1000.0);

        println!("  log_n={log_n:>2} | {n_hashes:>12} hashes | trace: {trace_ms:>6.0}ms | NTT: {ntt_ms:>6.0}ms | total: {total_ms:>8.1}ms | {:.1}M hash/s", hashes_per_sec / 1e6);
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
    println!("  RPO-M31 (24 cols, 14 rows/perm — Circle STARK optimised)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for log_n in [20, 24, 28] {
        let n: usize = 1 << log_n;
        let n_hashes = n / rpo_m31::ROWS_PER_PERM;

        let t = Instant::now();
        let d_cols = rpo_m31::generate_trace_gpu(log_n);
        let trace_ms = t.elapsed().as_secs_f64() * 1000.0;

        let eval_size = 2 * n;
        let log_eval = log_n + 1;
        let trace_domain = vortexstark::circle::Coset::half_coset(log_n);
        let eval_domain = vortexstark::circle::Coset::half_coset(log_eval);
        let inv = vortexstark::ntt::InverseTwiddleCache::new(&trace_domain);
        let fwd = vortexstark::ntt::ForwardTwiddleCache::new(&eval_domain);

        // Drop each eval col immediately — do NOT accumulate. Wide traces (RPO: 24 cols,
        // Cairo: 31 cols) would OOM if all eval cols were kept alive simultaneously.
        let t2 = Instant::now();
        for mut col in d_cols {
            vortexstark::ntt::interpolate(&mut col, &inv);
            let mut d_eval = vortexstark::device::DeviceBuffer::<u32>::alloc(eval_size);
            unsafe { ffi::cuda_zero_pad(col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32); }
            drop(col);
            vortexstark::ntt::evaluate(&mut d_eval, &fwd);
            // d_eval drops here
        }
        let ntt_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;
        let hashes_per_sec = n_hashes as f64 / (total_ms / 1000.0);

        println!("  log_n={log_n:>2} | {n_hashes:>12} hashes | trace: {trace_ms:>6.0}ms | NTT: {ntt_ms:>6.0}ms | total: {total_ms:>8.1}ms | {:.1}M hash/s", hashes_per_sec / 1e6);
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  POSEIDON2-FULL [EXPERIMENTAL] (8 cols, 8 rows/perm — RF=8, RP=0)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for log_n in [20, 24, 28] {
        let n: usize = 1 << log_n;
        let n_hashes = n / poseidon2f::ROWS_PER_PERM;

        let t = Instant::now();
        let d_cols = poseidon2f::generate_trace_gpu(log_n);
        let trace_ms = t.elapsed().as_secs_f64() * 1000.0;

        let eval_size = 2 * n;
        let log_eval = log_n + 1;
        let trace_domain = vortexstark::circle::Coset::half_coset(log_n);
        let eval_domain = vortexstark::circle::Coset::half_coset(log_eval);
        let inv = vortexstark::ntt::InverseTwiddleCache::new(&trace_domain);
        let fwd = vortexstark::ntt::ForwardTwiddleCache::new(&eval_domain);

        // Drop each eval col immediately — do NOT accumulate. Wide traces (RPO: 24 cols,
        // Cairo: 31 cols) would OOM if all eval cols were kept alive simultaneously.
        let t2 = Instant::now();
        for mut col in d_cols {
            vortexstark::ntt::interpolate(&mut col, &inv);
            let mut d_eval = vortexstark::device::DeviceBuffer::<u32>::alloc(eval_size);
            unsafe { ffi::cuda_zero_pad(col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32); }
            drop(col);
            vortexstark::ntt::evaluate(&mut d_eval, &fwd);
            // d_eval drops here
        }
        let ntt_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;
        let hashes_per_sec = n_hashes as f64 / (total_ms / 1000.0);

        println!("  log_n={log_n:>2} | {n_hashes:>12} hashes | trace: {trace_ms:>6.0}ms | NTT: {ntt_ms:>6.0}ms | total: {total_ms:>8.1}ms | {:.1}M hash/s", hashes_per_sec / 1e6);
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PEDERSEN HASH (GPU, STARK curve EC, windowed scalar mul)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    use vortexstark::cairo_air::{pedersen, stark252_field::Fp};
    pedersen::gpu_init();
    for &n_ped in &[1_000u64, 10_000, 100_000, 1_000_000] {
        let inputs_a: Vec<Fp> = (0..n_ped).map(|i| Fp::from_u64(i + 1)).collect();
        let inputs_b: Vec<Fp> = (0..n_ped).map(|i| Fp::from_u64(i + 1_000_000)).collect();
        // Warmup on first small batch
        let _ = pedersen::gpu_hash_batch(&inputs_a[..1], &inputs_b[..1]);
        let t = Instant::now();
        let results = pedersen::gpu_hash_batch(&inputs_a, &inputs_b);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let hps = n_ped as f64 / (ms / 1000.0);
        println!("  {n_ped:>9} hashes: {ms:>7.1}ms  ({:.0} hashes/sec)", hps);
        let _ = results;
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  SYSTEM SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Engine:     GPU-native Circle STARK (M31, 100-bit security)");
    println!("  GPU:        {gpu_info}");
  
    println!("  Builtins:   Poseidon2 (GPU, 30 rows/perm), RPO-M31 (GPU, 14 rows/perm), Poseidon2-Full (GPU, 8 rows/perm, experimental), Pedersen (GPU)");
    println!("  Features:   LogUp memory consistency, range checks, 2-phase commitment");
    println!("  CLI:        stark_cli prove/verify (binary proof serialization)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
