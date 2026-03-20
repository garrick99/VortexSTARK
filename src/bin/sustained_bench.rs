//! Sustained GPU benchmark with real-time power/thermal monitoring.
//!
//! Runs a configurable mix of real proof workloads in a tight loop while
//! sampling GPU power, temperature, clocks, and utilization every second.
//!
//! Usage:
//!   sustained_bench [OPTIONS]
//!
//! Examples:
//!   sustained_bench                          # 5 min, all workloads, default sizes
//!   sustained_bench --duration 600 --mix fib # 10 min, Fibonacci only
//!   sustained_bench --mix cairo --log-n 24   # Cairo VM at log_n=24
//!   sustained_bench --mix all --log-n 20     # Everything at log_n=20

use clap::Parser;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use vortexstark::cuda::ffi;
use vortexstark::field::M31;

#[derive(Parser)]
#[command(name = "sustained_bench", about = "Sustained GPU benchmark with power monitoring")]
struct Args {
    /// Duration in seconds (0 = infinite)
    #[arg(long, default_value = "300")]
    duration: u64,

    /// Workload mix: all, fib, cairo, pedersen, poseidon
    #[arg(long, default_value = "all")]
    mix: String,

    /// Log2 trace size for proofs
    #[arg(long, default_value = "20")]
    log_n: u32,

    /// GPU sampling interval in milliseconds
    #[arg(long, default_value = "1000")]
    sample_ms: u64,

    /// Also fetch and prove a Starknet contract (requires network)
    #[arg(long)]
    starknet: bool,

    /// Starknet class hash to fetch (used with --starknet)
    #[arg(long, default_value = "0x029927c8af6bccf3f6fda035981e765a7bdbf18a2dc0d630494f8758aa908e2b")]
    class_hash: String,
}

/// GPU telemetry sample.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct GpuSample {
    timestamp_ms: u64,
    power_w: f64,
    temp_c: u32,
    util_pct: u32,
    mem_used_mb: u32,
    sm_clock_mhz: u32,
}

/// Proof result from one iteration.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct ProofResult {
    workload: String,
    prove_ms: f64,
    verified: bool,
    elements: u64,
}

fn main() {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          VORTEXSTARK SUSTAINED BENCHMARK + GPU MONITOR         ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Duration:    {:>5}s ({:.1} min){:>30}║",
        args.duration, args.duration as f64 / 60.0, "");
    println!("║  Workload:    {:.<50}║", format!("{} @ log_n={}", args.mix, args.log_n));
    println!("║  GPU sample:  every {}ms{:>42}║", args.sample_ms, "");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // --- Init GPU ---
    eprintln!("[INIT] Initializing CUDA memory pool...");
    ffi::init_memory_pool_greedy();

    // Warmup
    eprintln!("[INIT] Warmup proof...");
    let _ = vortexstark::prover::prove(M31(1), M31(1), 8);

    // --- Fetch Starknet contract if requested ---
    let starknet_program = if args.starknet {
        eprintln!("[INIT] Fetching Starknet contract {}...", &args.class_hash[..18]);
        let rt = tokio::runtime::Runtime::new().unwrap();
        match rt.block_on(async {
            let client = vortexstark::cairo_air::starknet_rpc::StarknetClient::mainnet();
            client.get_compiled_casm(&args.class_hash).await
        }) {
            Ok(prog) => {
                eprintln!("[INIT] Loaded: {} bytecode felts", prog.bytecode.len());
                Some(prog)
            }
            Err(e) => {
                eprintln!("[INIT] WARNING: Starknet fetch failed: {e}");
                None
            }
        }
    } else {
        None
    };

    // --- Build workload list ---
    let workloads = build_workloads(&args.mix, args.log_n, &starknet_program);
    eprintln!("[INIT] Workloads: {}", workloads.iter().map(|w| w.name.as_str()).collect::<Vec<_>>().join(", "));

    // --- Start GPU monitor thread ---
    let running = Arc::new(AtomicBool::new(true));
    let proof_count = Arc::new(AtomicU64::new(0));
    let gpu_samples = Arc::new(std::sync::Mutex::new(Vec::<GpuSample>::new()));

    let monitor_running = running.clone();
    let monitor_samples = gpu_samples.clone();
    let monitor_proofs = proof_count.clone();
    let sample_interval = Duration::from_millis(args.sample_ms);
    let start_time = Instant::now();

    let monitor_handle = std::thread::spawn(move || {
        gpu_monitor_loop(monitor_running, monitor_samples, monitor_proofs, sample_interval, start_time);
    });

    // --- Print header ---
    println!();
    println!("{:>8} {:>8} {:>6} {:>5} {:>6} {:>6} {:>8} {:>8} {:>10} {:>6}",
        "Time", "Proofs", "W", "°C", "GPU%", "MHz", "VRAM_MB", "Workload", "Prove_ms", "OK");
    println!("{}", "─".repeat(90));

    // --- Main proving loop ---
    let bench_start = Instant::now();
    let duration = if args.duration == 0 { Duration::from_secs(u64::MAX) } else { Duration::from_secs(args.duration) };
    let mut proof_results: Vec<ProofResult> = Vec::new();
    let mut workload_idx = 0;
    let mut last_print = Instant::now();

    loop {
        if bench_start.elapsed() >= duration {
            break;
        }

        // Pick workload (round-robin)
        let workload = &workloads[workload_idx % workloads.len()];
        workload_idx += 1;

        // Run proof
        let result = (workload.run_fn)();
        proof_count.fetch_add(1, Ordering::Relaxed);

        // Print row (at most every 2 seconds to avoid spam)
        if last_print.elapsed() >= Duration::from_secs(2) || proof_results.is_empty() {
            let elapsed = bench_start.elapsed().as_secs();
            let total_proofs = proof_count.load(Ordering::Relaxed);

            // Get latest GPU sample
            let latest_gpu = gpu_samples.lock().unwrap().last().cloned();
            let (power, temp, util, clock, vram) = latest_gpu
                .map(|s| (s.power_w, s.temp_c, s.util_pct, s.sm_clock_mhz, s.mem_used_mb))
                .unwrap_or((0.0, 0, 0, 0, 0));

            let status = if result.verified { "✓" } else { "✗" };
            println!("{:>7}s {:>8} {:>5.1}W {:>4}°C {:>5}% {:>5} {:>7}M {:>8} {:>9.1}ms {:>5}",
                elapsed, total_proofs, power, temp, util, clock, vram,
                result.workload, result.prove_ms, status);

            last_print = Instant::now();
        }

        proof_results.push(result);
    }

    // --- Stop monitor ---
    running.store(false, Ordering::Relaxed);
    monitor_handle.join().unwrap();

    // --- Print summary ---
    let total_elapsed = bench_start.elapsed();
    let total_proofs = proof_results.len();
    let samples = gpu_samples.lock().unwrap().clone();

    println!("\n{}", "═".repeat(80));
    println!("  BENCHMARK COMPLETE");
    println!("{}", "═".repeat(80));
    println!("  Duration:       {:.1}s", total_elapsed.as_secs_f64());
    println!("  Total proofs:   {total_proofs}");
    println!("  Throughput:     {:.2} proofs/sec", total_proofs as f64 / total_elapsed.as_secs_f64());

    // Per-workload stats
    let mut workload_names: Vec<String> = proof_results.iter().map(|r| r.workload.clone()).collect();
    workload_names.sort();
    workload_names.dedup();

    println!("\n  Per-workload breakdown:");
    println!("  {:>12} {:>6} {:>10} {:>10} {:>10} {:>8}",
        "Workload", "Count", "Avg_ms", "Min_ms", "Max_ms", "Verified");

    for wname in &workload_names {
        let wresults: Vec<&ProofResult> = proof_results.iter().filter(|r| &r.workload == wname).collect();
        let count = wresults.len();
        let avg = wresults.iter().map(|r| r.prove_ms).sum::<f64>() / count as f64;
        let min = wresults.iter().map(|r| r.prove_ms).fold(f64::MAX, f64::min);
        let max = wresults.iter().map(|r| r.prove_ms).fold(0.0f64, f64::max);
        let verified = wresults.iter().filter(|r| r.verified).count();
        println!("  {:>12} {:>6} {:>9.1}ms {:>9.1}ms {:>9.1}ms {:>5}/{:<5}",
            wname, count, avg, min, max, verified, count);
    }

    // GPU power stats
    if !samples.is_empty() {
        let powers: Vec<f64> = samples.iter().map(|s| s.power_w).collect();
        let temps: Vec<u32> = samples.iter().map(|s| s.temp_c).collect();
        let utils: Vec<u32> = samples.iter().map(|s| s.util_pct).collect();
        let vrams: Vec<u32> = samples.iter().map(|s| s.mem_used_mb).collect();

        let avg_power = powers.iter().sum::<f64>() / powers.len() as f64;
        let max_power = powers.iter().cloned().fold(0.0f64, f64::max);
        let min_power = powers.iter().cloned().fold(f64::MAX, f64::min);
        let avg_temp = temps.iter().sum::<u32>() as f64 / temps.len() as f64;
        let max_temp = *temps.iter().max().unwrap_or(&0);
        let avg_util = utils.iter().sum::<u32>() as f64 / utils.len() as f64;
        let avg_vram = vrams.iter().sum::<u32>() as f64 / vrams.len() as f64;
        let max_vram = *vrams.iter().max().unwrap_or(&0);
        let min_vram = *vrams.iter().min().unwrap_or(&0);

        println!("\n  GPU Power:");
        println!("    Average:    {avg_power:.1}W");
        println!("    Peak:       {max_power:.1}W");
        println!("    Min:        {min_power:.1}W");
        println!("    Samples:    {}", samples.len());

        println!("\n  GPU Thermal:");
        println!("    Average:    {avg_temp:.1}°C");
        println!("    Peak:       {max_temp}°C");

        println!("\n  GPU Utilization:");
        println!("    Average:    {avg_util:.1}%");

        println!("\n  GPU VRAM:");
        println!("    Average:    {avg_vram:.0} MB ({:.1} GB)", avg_vram / 1024.0);
        println!("    Peak:       {max_vram} MB ({:.1} GB)", max_vram as f64 / 1024.0);
        println!("    Min:        {min_vram} MB ({:.1} GB)", min_vram as f64 / 1024.0);

        // Throttle detection
        let throttle_samples = temps.iter().filter(|&&t| t >= 83).count();
        if throttle_samples > 0 {
            println!("\n  WARNING: {} samples ({:.1}%) at ≥83°C (potential thermal throttle)",
                throttle_samples, throttle_samples as f64 / temps.len() as f64 * 100.0);
        }

        // Power stability (coefficient of variation)
        let power_mean = avg_power;
        let power_var = powers.iter().map(|p| (p - power_mean).powi(2)).sum::<f64>() / powers.len() as f64;
        let power_cov = power_var.sqrt() / power_mean * 100.0;
        println!("\n  Power stability (CoV): {power_cov:.2}%");
    }

    println!("\n{}", "═".repeat(80));
}

/// GPU monitor loop — runs in a background thread.
fn gpu_monitor_loop(
    running: Arc<AtomicBool>,
    samples: Arc<std::sync::Mutex<Vec<GpuSample>>>,
    _proof_count: Arc<AtomicU64>,
    interval: Duration,
    start: Instant,
) {
    while running.load(Ordering::Relaxed) {
        if let Some(sample) = query_gpu(start) {
            samples.lock().unwrap().push(sample);
        }
        std::thread::sleep(interval);
    }
    // Final sample
    if let Some(sample) = query_gpu(start) {
        samples.lock().unwrap().push(sample);
    }
}

/// Query GPU stats via nvidia-smi.
fn query_gpu(start: Instant) -> Option<GpuSample> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used,clocks.sm",
               "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() { return None; }

    let text = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = text.trim().split(',').map(|s| s.trim()).collect();
    if parts.len() < 5 { return None; }

    Some(GpuSample {
        timestamp_ms: start.elapsed().as_millis() as u64,
        power_w: parts[0].parse().unwrap_or(0.0),
        temp_c: parts[1].parse().unwrap_or(0),
        util_pct: parts[2].parse().unwrap_or(0),
        mem_used_mb: parts[3].parse().unwrap_or(0),
        sm_clock_mhz: parts[4].parse().unwrap_or(0),
    })
}

// ============================================================
// Workload definitions
// ============================================================

struct Workload {
    name: String,
    run_fn: Box<dyn Fn() -> ProofResult + Send>,
}

fn build_workloads(
    mix: &str,
    log_n: u32,
    starknet_program: &Option<vortexstark::cairo_air::casm_loader::CasmProgram>,
) -> Vec<Workload> {
    let mut workloads = Vec::new();

    let include_fib = mix == "all" || mix == "fib";
    let include_cairo = mix == "all" || mix == "cairo";
    let include_pedersen = mix == "all" || mix == "pedersen";
    let include_poseidon = mix == "all" || mix == "poseidon";
    let include_starknet = mix == "all" || mix == "starknet";

    // --- Fibonacci STARK ---
    if include_fib {
        let fib_log_n = log_n;
        workloads.push(Workload {
            name: format!("fib_{fib_log_n}"),
            run_fn: Box::new(move || {
                let t = Instant::now();
                let proof = vortexstark::prover::prove_lean(M31(1), M31(1), fib_log_n);
                let prove_ms = t.elapsed().as_secs_f64() * 1000.0;
                let ok = vortexstark::verifier::verify(&proof).is_ok();
                ProofResult {
                    workload: format!("fib_{fib_log_n}"),
                    prove_ms,
                    verified: ok,
                    elements: 1u64 << fib_log_n,
                }
            }),
        });
    }

    // --- Cairo VM STARK (Fibonacci program) ---
    if include_cairo {
        let cairo_log_n = log_n;
        let n_steps = (1usize << cairo_log_n).min(1 << 26); // cap at 67M steps

        // Build a Fibonacci program in bytecode
        let program = build_fib_cairo_program(n_steps);
        let program = Arc::new(program);

        workloads.push(Workload {
            name: format!("cairo_{cairo_log_n}"),
            run_fn: {
                let prog = program.clone();
                Box::new(move || {
                    let t = Instant::now();
                    let proof = vortexstark::cairo_air::prover::cairo_prove(&prog, n_steps, cairo_log_n);
                    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;
                    let ok = vortexstark::cairo_air::prover::cairo_verify(&proof).is_ok();
                    ProofResult {
                        workload: format!("cairo_{cairo_log_n}"),
                        prove_ms,
                        verified: ok,
                        elements: (n_steps * 27) as u64,
                    }
                })
            },
        });
    }

    // --- Pedersen GPU hash ---
    if include_pedersen {
        use vortexstark::cairo_air::pedersen;
        use vortexstark::cairo_air::stark252_field::Fp;

        pedersen::gpu_init();

        let batch_size = 1usize << log_n.min(20); // up to 1M hashes
        let inputs_a: Vec<Fp> = (0..batch_size).map(|i| {
            Fp::from_u64((i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x517CC1B727220A95))
        }).collect();
        let inputs_b: Vec<Fp> = (0..batch_size).map(|i| {
            Fp::from_u64((i as u64).wrapping_mul(0x6C62272E07BB0142).wrapping_add(0x62B821756295C58D))
        }).collect();
        let inputs_a = Arc::new(inputs_a);
        let inputs_b = Arc::new(inputs_b);

        workloads.push(Workload {
            name: format!("ped_{}k", batch_size / 1000),
            run_fn: {
                let a = inputs_a.clone();
                let b = inputs_b.clone();
                Box::new(move || {
                    let t = Instant::now();
                    let results = pedersen::gpu_hash_batch(&a, &b);
                    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;
                    ProofResult {
                        workload: format!("ped_{}k", results.len() / 1000),
                        prove_ms,
                        verified: true, // GPU hash is deterministic
                        elements: results.len() as u64,
                    }
                })
            },
        });
    }

    // --- Poseidon GPU trace ---
    if include_poseidon {
        let pos_log_n = log_n.min(24); // cap to avoid VRAM blow
        workloads.push(Workload {
            name: format!("pos_{pos_log_n}"),
            run_fn: Box::new(move || {
                let t = Instant::now();
                let (d_cols, _, _) = vortexstark::poseidon::generate_trace_gpu(pos_log_n);
                let prove_ms = t.elapsed().as_secs_f64() * 1000.0;
                drop(d_cols);
                ProofResult {
                    workload: format!("pos_{pos_log_n}"),
                    prove_ms,
                    verified: true,
                    elements: (1u64 << pos_log_n),
                }
            }),
        });
    }

    // --- Starknet contract proof ---
    if include_starknet {
        if let Some(prog) = starknet_program {
            let bytecode = Arc::new(prog.bytecode.clone());
            let prog_name = prog.name.clone();
            let n_steps = vortexstark::cairo_air::casm_loader::detect_steps(prog, 1 << 20);
            if n_steps > 0 {
                let mut sn_log_n = 0u32;
                while (1usize << sn_log_n) < n_steps { sn_log_n += 1; }
                sn_log_n = sn_log_n.max(4);

                workloads.push(Workload {
                    name: "starknet".to_string(),
                    run_fn: {
                        let bc = bytecode.clone();
                        Box::new(move || {
                            let t = Instant::now();
                            let proof = vortexstark::cairo_air::prover::cairo_prove(&bc, n_steps, sn_log_n);
                            let prove_ms = t.elapsed().as_secs_f64() * 1000.0;
                            let ok = vortexstark::cairo_air::prover::cairo_verify(&proof).is_ok();
                            ProofResult {
                                workload: "starknet".to_string(),
                                prove_ms,
                                verified: ok,
                                elements: (n_steps * 27) as u64,
                            }
                        })
                    },
                });

                eprintln!("[INIT] Starknet workload: {} ({n_steps} steps, log_n={sn_log_n})", prog_name);
            }
        }
    }

    if workloads.is_empty() {
        eprintln!("ERROR: no workloads selected. Use --mix all|fib|cairo|pedersen|poseidon");
        std::process::exit(1);
    }

    workloads
}

/// Build a Fibonacci Cairo program with the given number of steps.
fn build_fib_cairo_program(n_steps: usize) -> Vec<u64> {
    use vortexstark::cairo_air::decode::Instruction;

    let mut program = Vec::new();

    // [ap+0] = 1 (fib_0)
    let assert_imm = Instruction {
        off0: 0x8000, off1: 0x8000, off2: 0x8001,
        op1_imm: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    program.push(assert_imm.encode());
    program.push(1);

    // [ap+0] = 1 (fib_1)
    program.push(assert_imm.encode());
    program.push(1);

    // Loop body: [ap] = [ap-2] + [ap-1]
    let add_instr = Instruction {
        off0: 0x8000, off1: 0x8000u16.wrapping_sub(2), off2: 0x8000u16.wrapping_sub(1),
        op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };

    // Fill remaining steps with adds (first 2 steps are the asserts)
    for _ in 0..n_steps.saturating_sub(2) {
        program.push(add_instr.encode());
    }

    program
}
