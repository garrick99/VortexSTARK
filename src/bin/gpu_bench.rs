//! Comprehensive GPU benchmark with strict pre-flight condition checking.
//!
//! Before any workload begins this tool:
//!   1. Detects competing GPU compute processes (aborts unless --force)
//!   2. Estimates VRAM required and verifies availability
//!   3. Rejects a hot GPU (≥83°C abort, ≥70°C warn)
//!   4. Reports power draw, clock state, driver version
//!
//! During each section a background sampler records temp/power/VRAM/clock
//! every 500 ms. Peak and average values are printed per section and
//! summarised at the end.
//!
//! Usage:
//!   gpu_bench                        # default: all sections, auto log_n
//!   gpu_bench --max-log-n 26         # cap trace size
//!   gpu_bench --force                # run even if competing processes found
//!   gpu_bench --skip-preflight       # bypass all pre-flight checks
//!   gpu_bench --sections fib,cairo   # comma-separated subset

use clap::Parser;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "gpu_bench",
    about = "VortexSTARK comprehensive GPU benchmark with pre-flight condition checks"
)]
struct Args {
    /// Cap trace size (log2). Default 28; use 30 only on a clean 32 GB GPU.
    #[arg(long, default_value = "28")]
    max_log_n: u32,

    /// Run even if competing GPU processes are detected.
    #[arg(long)]
    force: bool,

    /// Skip all pre-flight checks and just benchmark.
    #[arg(long)]
    skip_preflight: bool,

    /// Comma-separated sections to run: fib, poseidon, rpo, poseidon2f, cairo, pedersen
    #[arg(long, default_value = "fib,poseidon,rpo,poseidon2f,cairo,pedersen")]
    sections: String,

    /// GPU telemetry sample interval in milliseconds.
    #[arg(long, default_value = "500")]
    sample_ms: u64,
}

// ── GPU telemetry ─────────────────────────────────────────────────────────────

/// Lightweight snapshot used by the background sampler (4 fields only).
/// Power limit is queried once at preflight and does not change during a run.
#[derive(Clone, Debug)]
struct GpuSnapshot {
    #[allow(dead_code)]
    ts_ms: u64,
    power_w: f64,
    temp_c: u32,
    mem_used_mb: u64,
    sm_clock_mhz: u32,
}

/// Fast background-sampler query: only the four metrics that change during a run.
///
///   nvidia-smi --query-gpu=power.draw,temperature.gpu,memory.used,clocks.sm
///              --format=csv,noheader,nounits
fn query_gpu_snapshot(start: Instant) -> Option<GpuSnapshot> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=power.draw,temperature.gpu,memory.used,clocks.sm",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let p: Vec<&str> = text.trim().split(',').map(|s| s.trim()).collect();
    if p.len() < 4 {
        return None;
    }
    Some(GpuSnapshot {
        ts_ms: start.elapsed().as_millis() as u64,
        power_w:     p[0].parse().unwrap_or(0.0),
        temp_c:      p[1].parse().unwrap_or(0),
        mem_used_mb: p[2].parse().unwrap_or(0),
        sm_clock_mhz:p[3].parse().unwrap_or(0),
    })
}

/// Detailed power state from `nvidia-smi -q -d POWER`.
///
/// Parses the "Power Readings" / "Power Draw" block, giving us draw, the
/// current enforced limit, the user-settable limit, and the hardware min/max.
/// This is much richer than the single `power.limit` CSV field.
#[derive(Debug, Default)]
struct PowerDetail {
    draw_w:           f64,
    limit_w:          f64,   // user-set limit (nvidia-smi -pl)
    enforced_limit_w: f64,   // actually enforced (may differ when TDP-capped)
    default_limit_w:  f64,
    min_limit_w:      f64,
    max_limit_w:      f64,
}

/// Query full power information using `nvidia-smi -q -d POWER`.
fn query_power_detail() -> Option<PowerDetail> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["-q", "-d", "POWER"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);

    // Helper: extract the watt value from a line like
    //   "    Power Draw                  : 45.23 W"
    let parse_watts = |needle: &str| -> f64 {
        text.lines()
            .find(|l| l.contains(needle))
            .and_then(|l| l.split(':').nth(1))
            .and_then(|v| v.trim().trim_end_matches(" W").trim().parse::<f64>().ok())
            .unwrap_or(0.0)
    };

    // Actual nvidia-smi -q -d POWER labels (verified against RTX 5090 driver 595.79):
    //   "Instantaneous Power Draw"  — current draw (not the rolling average)
    //   "Current Power Limit"       — the active limit (set via nvidia-smi -pl)
    //   "Requested Power Limit"     — what was requested (equivalent to "enforced" on this driver)
    //   "Default Power Limit"       — factory default
    //   "Min Power Limit"           — hardware floor
    //   "Max Power Limit"           — hardware ceiling
    Some(PowerDetail {
        draw_w:           parse_watts("Instantaneous Power Draw"),
        limit_w:          parse_watts("Current Power Limit"),
        enforced_limit_w: parse_watts("Requested Power Limit"),
        default_limit_w:  parse_watts("Default Power Limit"),
        min_limit_w:      parse_watts("Min Power Limit"),
        max_limit_w:      parse_watts("Max Power Limit"),
    })
}

/// VRAM breakdown from `nvidia-smi -q -d MEMORY`.
///
/// Parses the "FB Memory Usage" block (framebuffer — the VRAM we care about).
/// Reports Total / Used / Free in MB, converting from MiB as reported.
#[derive(Debug, Default)]
struct VramDetail {
    total_mb: u64,
    used_mb:  u64,
    free_mb:  u64,
}

/// Query VRAM state using `nvidia-smi -q -d MEMORY`.
fn query_vram_detail() -> Option<VramDetail> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["-q", "-d", "MEMORY"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);

    // The FB Memory Usage block comes first in the output; BAR1 follows.
    // We grab lines between "FB Memory Usage" and the next blank/section line.
    let mut in_fb = false;
    let mut detail = VramDetail::default();

    let parse_mib = |line: &str| -> u64 {
        line.split(':')
            .nth(1)
            .and_then(|v| v.trim().trim_end_matches(" MiB").trim().parse::<u64>().ok())
            .unwrap_or(0)
    };

    for line in text.lines() {
        if line.contains("FB Memory Usage") {
            in_fb = true;
            continue;
        }
        // Next section header ends the FB block
        if in_fb && !line.starts_with(' ') && !line.is_empty() {
            break;
        }
        if in_fb {
            if line.contains("Total") {
                detail.total_mb = parse_mib(line);
            } else if line.contains("Used") {
                detail.used_mb = parse_mib(line);
            } else if line.contains("Free") {
                detail.free_mb = parse_mib(line);
            }
        }
    }

    if detail.total_mb == 0 {
        None
    } else {
        Some(detail)
    }
}

/// Static GPU info: name + driver version in one CSV call.
fn query_gpu_identity() -> (String, String) {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
        .output();
    let Ok(out) = out else {
        return ("Unknown GPU".to_string(), "Unknown".to_string());
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut parts = text.trim().splitn(2, ',');
    let name    = parts.next().unwrap_or("Unknown GPU").trim().to_string();
    let driver  = parts.next().unwrap_or("Unknown").trim().to_string();
    (name, driver)
}

/// Temperature + SM clock: used in preflight where we don't need VRAM detail.
///
///   nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader,nounits
fn query_temp_and_clock() -> (u32, u32) {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=temperature.gpu,clocks.sm",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(out) = out else { return (0, 0) };
    let text = String::from_utf8_lossy(&out.stdout);
    let p: Vec<&str> = text.trim().split(',').map(|s| s.trim()).collect();
    if p.len() < 2 {
        return (0, 0);
    }
    (p[0].parse().unwrap_or(0), p[1].parse().unwrap_or(0))
}

// ── Competing-process detection ───────────────────────────────────────────────

#[derive(Debug)]
struct ComputeProcess {
    pid: u32,
    name: String,
    mem_mb: u64,
}

fn query_compute_processes() -> Vec<ComputeProcess> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let Ok(out) = out else { return vec![] };
    if !out.status.success() {
        return vec![];
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let mut procs = Vec::new();
    for line in text.lines() {
        let p: Vec<&str> = line.trim().split(',').map(|s| s.trim()).collect();
        if p.len() < 3 || p[0].is_empty() {
            continue;
        }
        // Skip ourselves (we haven't initialised CUDA yet, but be safe)
        let pid: u32 = p[0].parse().unwrap_or(0);
        let mem_mb: u64 = p[2].parse().unwrap_or(0);
        // Ignore tiny reservations (display driver, etc.)
        if mem_mb < 64 {
            continue;
        }
        procs.push(ComputeProcess {
            pid,
            name: p[1].to_string(),
            mem_mb,
        });
    }
    procs
}

// ── VRAM requirement estimates ────────────────────────────────────────────────

/// Conservative peak VRAM estimate in MB for a given workload at log_n.
///
/// Formula: (trace_cols + 1_eval_col_blowup2 + prover_scratch) × n × 4 bytes
///
/// The NTT benchmark loops drop each eval column immediately, so only one is
/// live at a time. Peak = all trace cols + one eval col (2× size) + overhead.
/// Prover scratch (Merkle, FRI, twiddle caches) adds ~4 extra "col-equivalents".
fn vram_estimate_mb(section: &str, log_n: u32) -> u64 {
    let n: u64 = 1 << log_n;
    let bytes: u64 = 4; // M31 is 4 bytes

    // Column count for each hash/proof section
    let trace_cols: u64 = match section {
        "fib"        => 1,
        "poseidon"   => 8,
        "rpo"        => 24,
        "poseidon2f" => 8,
        "cairo"      => 31,
        "pedersen"   => return 2048, // constant-ish regardless of log_n
        _            => 8,
    };

    // Peak = all trace cols + one eval col (blowup 2×) + 4 cols of scratch
    let scratch_cols: u64 = 4;
    let total_cols = trace_cols + 2 + scratch_cols; // +2 for one 2× eval col
    (total_cols * n * bytes) / (1024 * 1024)
}

// ── Pre-flight ────────────────────────────────────────────────────────────────

#[derive(Debug)]
enum CheckLevel {
    Ok,
    Warn(String),
    Fail(String),
}

struct PreflightResult {
    gpu_name: String,
    driver_version: String,
    power: PowerDetail,
    vram: VramDetail,
    temp_c: u32,
    sm_clock_mhz: u32,
    competing_procs: Vec<ComputeProcess>,
    checks: Vec<(&'static str, CheckLevel)>,
}

fn run_preflight(sections: &[&str], max_log_n: u32) -> PreflightResult {
    // Each query uses the narrowest nvidia-smi switch that gives us what we need.
    // No single omnibus call — each subsystem is queried independently.
    let (gpu_name, driver_version) = query_gpu_identity();
    let power  = query_power_detail().unwrap_or_default();
    let vram   = query_vram_detail().unwrap_or_default();
    let (temp_c, sm_clock_mhz) = query_temp_and_clock();
    let competing_procs = query_compute_processes();

    let mut checks: Vec<(&'static str, CheckLevel)> = Vec::new();

    // Check 1: Competing processes
    if competing_procs.is_empty() {
        checks.push(("Competing GPU processes", CheckLevel::Ok));
    } else {
        let total_mb: u64 = competing_procs.iter().map(|p| p.mem_mb).sum();
        checks.push((
            "Competing GPU processes",
            CheckLevel::Fail(format!(
                "{} process(es) using {} MB VRAM — timings will be skewed",
                competing_procs.len(),
                total_mb
            )),
        ));
    }

    // Check 2: Temperature
    checks.push((
        "GPU temperature",
        if temp_c >= 83 {
            CheckLevel::Fail(format!("{temp_c}°C — GPU is throttling (≥83°C). Cool down first."))
        } else if temp_c >= 70 {
            CheckLevel::Warn(format!("{temp_c}°C — warm, results may be thermally limited"))
        } else {
            CheckLevel::Ok
        },
    ));

    // Check 3: VRAM sufficiency — uses the detailed FB memory block
    let max_required: u64 = sections
        .iter()
        .map(|s| vram_estimate_mb(s, max_log_n))
        .max()
        .unwrap_or(0);

    checks.push((
        "VRAM availability",
        if vram.total_mb == 0 {
            CheckLevel::Warn("Could not query VRAM (nvidia-smi -q -d MEMORY failed)".to_string())
        } else if vram.free_mb < max_required {
            CheckLevel::Fail(format!(
                "{} MB free, need ~{} MB for log_n={max_log_n}. Lower --max-log-n or free VRAM.",
                vram.free_mb, max_required
            ))
        } else if vram.free_mb < max_required * 2 {
            CheckLevel::Warn(format!(
                "{} MB free (tight — need ~{} MB). OOM possible at max size.",
                vram.free_mb, max_required
            ))
        } else {
            CheckLevel::Ok
        },
    ));

    // Check 4: Power — compared against the enforced limit (the one the GPU
    // actually uses), not just the nominal limit.
    let effective_limit = if power.enforced_limit_w > 0.0 {
        power.enforced_limit_w
    } else {
        power.limit_w
    };
    checks.push((
        "Power draw at idle",
        if effective_limit > 0.0 && power.draw_w > effective_limit * 0.5 {
            CheckLevel::Warn(format!(
                "{:.0}W draw / {:.0}W enforced limit ({:.0}W nominal) — GPU already under load",
                power.draw_w, effective_limit, power.limit_w
            ))
        } else {
            CheckLevel::Ok
        },
    ));

    // Check 5: Power headroom — warn if user has lowered the limit far below HW max
    if power.max_limit_w > 0.0 && power.limit_w < power.max_limit_w * 0.7 {
        checks.push((
            "Power limit headroom",
            CheckLevel::Warn(format!(
                "Limit set to {:.0}W but HW max is {:.0}W — GPU may be power-capped. \
                 Run: nvidia-smi -pl {:.0}",
                power.limit_w, power.max_limit_w, power.max_limit_w
            )),
        ));
    }

    // Check 6: Clock state (elevated clock at idle suggests background load)
    checks.push((
        "SM clock at idle",
        if sm_clock_mhz > 1000 {
            CheckLevel::Warn(format!(
                "{sm_clock_mhz} MHz — clock is boosted at idle (background GPU workload?)"
            ))
        } else {
            CheckLevel::Ok
        },
    ));

    PreflightResult {
        gpu_name,
        driver_version,
        power,
        vram,
        temp_c,
        sm_clock_mhz,
        competing_procs,
        checks,
    }
}

// ── Background GPU monitor ─────────────────────────────────────────────────────

struct GpuMonitor {
    running: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<GpuSnapshot>>>,
    handle: Option<std::thread::JoinHandle<()>>,
    start: Instant,
}

impl GpuMonitor {
    fn start_new(interval_ms: u64) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let samples: Arc<Mutex<Vec<GpuSnapshot>>> = Arc::new(Mutex::new(Vec::new()));
        let r = running.clone();
        let s = samples.clone();
        let t0 = Instant::now();
        let handle = std::thread::spawn(move || {
            while r.load(Ordering::Relaxed) {
                if let Some(snap) = query_gpu_snapshot(t0) {
                    s.lock().unwrap().push(snap);
                }
                std::thread::sleep(Duration::from_millis(interval_ms));
            }
            // Final sample after stop
            if let Some(snap) = query_gpu_snapshot(t0) {
                s.lock().unwrap().push(snap);
            }
        });
        Self { running, samples, handle: Some(handle), start: t0 }
    }

    fn stop(mut self) -> Vec<GpuSnapshot> {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.samples.lock().unwrap().clone()
    }

    /// Take a snapshot right now (while still running).
    fn snapshot_now(&self) -> Option<GpuSnapshot> {
        query_gpu_snapshot(self.start)
    }
}

// ── Section stats ─────────────────────────────────────────────────────────────

#[derive(Debug)]
struct SectionStats {
    name: String,
    wall_ms: f64,
    peak_temp_c: u32,
    #[allow(dead_code)]
    avg_temp_c: f32,
    peak_power_w: f64,
    avg_power_w: f64,
    peak_vram_mb: u64,
    peak_sm_mhz: u32,
    #[allow(dead_code)]
    samples: usize,
}

fn compute_section_stats(name: &str, wall_ms: f64, snaps: &[GpuSnapshot]) -> SectionStats {
    if snaps.is_empty() {
        return SectionStats {
            name: name.to_string(),
            wall_ms,
            peak_temp_c: 0,
            avg_temp_c: 0.0,
            peak_power_w: 0.0,
            avg_power_w: 0.0,
            peak_vram_mb: 0,
            peak_sm_mhz: 0,
            samples: 0,
        };
    }
    let peak_temp_c  = snaps.iter().map(|s| s.temp_c).max().unwrap_or(0);
    let avg_temp_c   = snaps.iter().map(|s| s.temp_c as f32).sum::<f32>() / snaps.len() as f32;
    let peak_power_w = snaps.iter().map(|s| s.power_w).fold(0.0f64, f64::max);
    let avg_power_w  = snaps.iter().map(|s| s.power_w).sum::<f64>() / snaps.len() as f64;
    let peak_vram_mb = snaps.iter().map(|s| s.mem_used_mb).max().unwrap_or(0);
    let peak_sm_mhz  = snaps.iter().map(|s| s.sm_clock_mhz).max().unwrap_or(0);
    SectionStats {
        name: name.to_string(),
        wall_ms,
        peak_temp_c,
        avg_temp_c,
        peak_power_w,
        avg_power_w,
        peak_vram_mb,
        peak_sm_mhz,
        samples: snaps.len(),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn print_rule(c: char, n: usize) {
    println!("{}", c.to_string().repeat(n));
}

fn check_icon(c: &CheckLevel) -> &'static str {
    match c {
        CheckLevel::Ok        => "  ✓",
        CheckLevel::Warn(_)   => "  ⚠",
        CheckLevel::Fail(_)   => "  ✗",
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    let sections_list: Vec<&str> = args.sections
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║          VORTEXSTARK — COMPREHENSIVE GPU BENCHMARK                 ║");
    println!("║          Rust + CUDA · Circle STARK · RTX 5090 / 4090             ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Pre-flight ────────────────────────────────────────────────────────────
    if !args.skip_preflight {
        println!("┌─ PRE-FLIGHT GPU CONDITIONS ───────────────────────────────────────────");
        let pf = run_preflight(&sections_list, args.max_log_n);

        println!("│  GPU:             {}", pf.gpu_name);
        println!("│  Driver:          {}", pf.driver_version);
        println!("│  VRAM:            {} MB total / {} MB free ({} MB used)",
            pf.vram.total_mb, pf.vram.free_mb, pf.vram.used_mb);
        println!("│  Temperature:     {}°C", pf.temp_c);
        println!("│  Power draw:      {:.0}W", pf.power.draw_w);
        println!("│  Power limit:     {:.0}W (enforced {:.0}W, default {:.0}W)",
            pf.power.limit_w, pf.power.enforced_limit_w, pf.power.default_limit_w);
        println!("│  Power range:     {:.0}W – {:.0}W (HW min/max)",
            pf.power.min_limit_w, pf.power.max_limit_w);
        println!("│  SM clock:        {} MHz", pf.sm_clock_mhz);
        println!("│");

        if !pf.competing_procs.is_empty() {
            println!("│  Competing processes detected:");
            for p in &pf.competing_procs {
                println!("│    PID {:>6}  {:40}  {} MB", p.pid, p.name, p.mem_mb);
            }
            println!("│");
        }

        let mut has_fail = false;
        let mut has_warn = false;
        for (label, level) in &pf.checks {
            let icon = check_icon(level);
            match level {
                CheckLevel::Ok => {
                    println!("│ {icon}  {label}");
                }
                CheckLevel::Warn(msg) => {
                    has_warn = true;
                    println!("│ {icon}  {label}: {msg}");
                }
                CheckLevel::Fail(msg) => {
                    has_fail = true;
                    println!("│ {icon}  {label}: {msg}");
                }
            }
        }
        println!("└──────────────────────────────────────────────────────────────────────");
        println!();

        if has_fail && !args.force {
            eprintln!("ABORT: Pre-flight check(s) failed. Fix the issues above or re-run with --force.");
            std::process::exit(1);
        }
        if has_warn {
            println!("WARNING: Pre-flight warnings present. Results may be affected.");
            println!("         Re-run with a cool, idle GPU for clean numbers.");
            println!();
        }
    } else {
        println!("(--skip-preflight: GPU condition checks bypassed)");
        println!();
    }

    // ── CUDA init ─────────────────────────────────────────────────────────────
    println!("Initialising CUDA memory pool...");
    vortexstark::cuda::ffi::init_memory_pool_greedy();

    // Warmup: force driver out of idle / P8 state
    println!("Warming up GPU (small proof)...");
    let _ = vortexstark::prover::prove(vortexstark::field::M31(1), vortexstark::field::M31(1), 8);
    // Brief pause to let clocks stabilise
    std::thread::sleep(Duration::from_millis(200));
    println!("GPU ready.\n");

    let mut section_results: Vec<SectionStats> = Vec::new();
    let global_start = Instant::now();
    let sample_ms = args.sample_ms;
    let max_log_n = args.max_log_n;

    // ── Section: Fibonacci STARK ──────────────────────────────────────────────
    if sections_list.contains(&"fib") {
        print_rule('━', 72);
        println!("  FIBONACCI STARK  (1 column, degree-1 constraint, blowup 2×)");
        print_rule('━', 72);

        let candidates = [20u32, 24, 28, 29, 30];
        let log_ns: Vec<u32> = candidates.iter().copied().filter(|&n| n <= max_log_n).collect();

        for log_n in log_ns {
            let n: u64 = 1 << log_n;
            let est_mb = vram_estimate_mb("fib", log_n);

            let mon = GpuMonitor::start_new(sample_ms);
            let t0 = Instant::now();
            let proof = vortexstark::prover::prove_lean(
                vortexstark::field::M31(1), vortexstark::field::M31(1), log_n);
            let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let ok = vortexstark::verifier::verify(&proof).is_ok();
            let verify_ms = t1.elapsed().as_secs_f64() * 1000.0;
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // Snap live stats for inline display before monitor stops
            let live = mon.snapshot_now();
            let snaps = mon.stop();
            let ss = compute_section_stats(&format!("fib_{log_n}"), wall_ms, &snaps);

            let (_, _, cur_vram) = live_or_peak(&live, &ss);

            println!(
                "  log_n={log_n:>2} | {n:>12} elems | prove: {prove_ms:>8.1}ms | \
                 verify: {verify_ms:>6.1}ms | est {est_mb}MB | \
                 peak {peak_t}°C {peak_p:.0}W {cur_vram}MB | {}",
                if ok { "✓" } else { "✗" },
                peak_t = ss.peak_temp_c,
                peak_p = ss.peak_power_w,
            );
            section_results.push(ss);
        }
        println!();
    }

    // ── Section: Poseidon2 (baseline hash) ───────────────────────────────────
    if sections_list.contains(&"poseidon") {
        use vortexstark::poseidon;

        print_rule('━', 72);
        println!("  POSEIDON2  (8 cols, 30 rows/perm, RF=8 RP=22)");
        print_rule('━', 72);

        for log_n in [20u32, 24, 28].iter().copied().filter(|&n| n <= max_log_n) {
            let n: usize = 1 << log_n;
            let n_hashes = n / poseidon::NUM_ROUNDS;
            let est_mb = vram_estimate_mb("poseidon", log_n);

            let mon = GpuMonitor::start_new(sample_ms);
            let t0 = Instant::now();
            let (d_cols, _, _) = poseidon::generate_trace_gpu(log_n);
            let trace_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let ntt_ms = do_ntt(d_cols, log_n);
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let live = mon.snapshot_now();
            let snaps = mon.stop();
            let ss = compute_section_stats(&format!("poseidon_{log_n}"), wall_ms, &snaps);

            let mhps = n_hashes as f64 / (wall_ms / 1000.0) / 1e6;
            let (cur_t, cur_p, cur_v) = live_or_peak(&live, &ss);

            println!(
                "  log_n={log_n:>2} | {n_hashes:>10} hashes | \
                 trace {trace_ms:>6.0}ms NTT {ntt_ms:>6.0}ms total {wall_ms:>8.1}ms | \
                 {mhps:.2}M/s | est {est_mb}MB | {cur_t}°C {cur_p:.0}W {cur_v}MB"
            );
            section_results.push(ss);
        }
        println!();
    }

    // ── Section: RPO-M31 ─────────────────────────────────────────────────────
    if sections_list.contains(&"rpo") {
        use vortexstark::rpo_m31;

        print_rule('━', 72);
        println!("  RPO-M31  (24 cols, 14 rows/perm — Circle STARK–native)");
        print_rule('━', 72);

        for log_n in [20u32, 24, 28].iter().copied().filter(|&n| n <= max_log_n) {
            let n: usize = 1 << log_n;
            let n_hashes = n / rpo_m31::ROWS_PER_PERM;
            let est_mb = vram_estimate_mb("rpo", log_n);

            let mon = GpuMonitor::start_new(sample_ms);
            let t0 = Instant::now();
            let d_cols = rpo_m31::generate_trace_gpu(log_n);
            let trace_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let ntt_ms = do_ntt(d_cols, log_n);
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let live = mon.snapshot_now();
            let snaps = mon.stop();
            let ss = compute_section_stats(&format!("rpo_{log_n}"), wall_ms, &snaps);

            let mhps = n_hashes as f64 / (wall_ms / 1000.0) / 1e6;
            let (cur_t, cur_p, cur_v) = live_or_peak(&live, &ss);

            println!(
                "  log_n={log_n:>2} | {n_hashes:>10} hashes | \
                 trace {trace_ms:>6.0}ms NTT {ntt_ms:>6.0}ms total {wall_ms:>8.1}ms | \
                 {mhps:.2}M/s | est {est_mb}MB | {cur_t}°C {cur_p:.0}W {cur_v}MB"
            );
            section_results.push(ss);
        }
        println!();
    }

    // ── Section: Poseidon2-Full [experimental] ────────────────────────────────
    if sections_list.contains(&"poseidon2f") {
        use vortexstark::poseidon2f;

        print_rule('━', 72);
        println!("  POSEIDON2-FULL [EXPERIMENTAL]  (8 rows/perm, RF=8 RP=0, no security analysis)");
        print_rule('━', 72);

        for log_n in [20u32, 24, 28].iter().copied().filter(|&n| n <= max_log_n) {
            let n: usize = 1 << log_n;
            let n_hashes = n / poseidon2f::ROWS_PER_PERM;
            let est_mb = vram_estimate_mb("poseidon2f", log_n);

            let mon = GpuMonitor::start_new(sample_ms);
            let t0 = Instant::now();
            let d_cols = poseidon2f::generate_trace_gpu(log_n);
            let trace_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let ntt_ms = do_ntt(d_cols, log_n);
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let live = mon.snapshot_now();
            let snaps = mon.stop();
            let ss = compute_section_stats(&format!("p2f_{log_n}"), wall_ms, &snaps);

            let mhps = n_hashes as f64 / (wall_ms / 1000.0) / 1e6;
            let (cur_t, cur_p, cur_v) = live_or_peak(&live, &ss);

            println!(
                "  log_n={log_n:>2} | {n_hashes:>10} hashes | \
                 trace {trace_ms:>6.0}ms NTT {ntt_ms:>6.0}ms total {wall_ms:>8.1}ms | \
                 {mhps:.2}M/s | est {est_mb}MB | {cur_t}°C {cur_p:.0}W {cur_v}MB"
            );
            section_results.push(ss);
        }
        println!();
    }

    // ── Section: Cairo VM STARK ───────────────────────────────────────────────
    if sections_list.contains(&"cairo") {
        use vortexstark::cairo_air::decode::Instruction;
        use vortexstark::cairo_air::prover::{cairo_prove, cairo_verify};

        print_rule('━', 72);
        println!("  CAIRO VM STARK  (31 cols, 31 constraints, LogUp + range checks)");
        print_rule('━', 72);

        let cairo_max = max_log_n.min(26);
        for log_n in [20u32, 24, 26].iter().copied().filter(|&n| n <= cairo_max) {
            let n: usize = 1 << log_n;
            let est_mb = vram_estimate_mb("cairo", log_n);

            // Build Fibonacci Cairo program
            let assert_imm = Instruction {
                off0: 0x8000, off1: 0x8000, off2: 0x8001,
                op1_imm: 1, opcode_assert: 1, ap_add1: 1,
                ..Default::default()
            };
            let add_instr = Instruction {
                off0: 0x8000, off1: 0x8000u16.wrapping_sub(2), off2: 0x8000u16.wrapping_sub(1),
                op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
                ..Default::default()
            };
            let mut program = Vec::new();
            program.push(assert_imm.encode()); program.push(1);
            program.push(assert_imm.encode()); program.push(1);
            for _ in 0..n.saturating_sub(2) {
                program.push(add_instr.encode());
            }

            let mon = GpuMonitor::start_new(sample_ms);
            let t0 = Instant::now();
            let proof = cairo_prove(&program, n, log_n);
            let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let ok = cairo_verify(&proof).is_ok();
            let verify_ms = t1.elapsed().as_secs_f64() * 1000.0;
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let live = mon.snapshot_now();
            let snaps = mon.stop();
            let ss = compute_section_stats(&format!("cairo_{log_n}"), wall_ms, &snaps);
            let (cur_t, cur_p, cur_v) = live_or_peak(&live, &ss);

            println!(
                "  log_n={log_n:>2} | {n:>10} steps | \
                 prove {prove_ms:>8.1}ms verify {verify_ms:>6.1}ms | \
                 est {est_mb}MB | {cur_t}°C {cur_p:.0}W {cur_v}MB | {}",
                if ok { "✓" } else { "✗" }
            );
            section_results.push(ss);
        }
        println!();
    }

    // ── Section: Pedersen hash (GPU batch) ───────────────────────────────────
    if sections_list.contains(&"pedersen") {
        use vortexstark::cairo_air::{pedersen, stark252_field::Fp};

        print_rule('━', 72);
        println!("  PEDERSEN HASH  (GPU batch, windowed 4-bit scalar mul on STARK curve)");
        print_rule('━', 72);

        pedersen::gpu_init();

        for &n_ped in &[1_000u64, 10_000, 100_000, 1_000_000] {
            let inputs_a: Vec<Fp> = (0..n_ped).map(|i| Fp::from_u64(i + 1)).collect();
            let inputs_b: Vec<Fp> = (0..n_ped).map(|i| Fp::from_u64(i + 1_000_000)).collect();

            // Per-batch warmup
            let _ = pedersen::gpu_hash_batch(&inputs_a[..1], &inputs_b[..1]);

            let mon = GpuMonitor::start_new(sample_ms);
            let t0 = Instant::now();
            let results = pedersen::gpu_hash_batch(&inputs_a, &inputs_b);
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let _ = results;

            let live = mon.snapshot_now();
            let snaps = mon.stop();
            let ss = compute_section_stats(&format!("ped_{n_ped}"), wall_ms, &snaps);
            let hps = n_ped as f64 / (wall_ms / 1000.0);
            let (cur_t, cur_p, cur_v) = live_or_peak(&live, &ss);

            println!(
                "  {n_ped:>9} hashes | {wall_ms:>8.1}ms | {:.0} hash/s | \
                 {cur_t}°C {cur_p:.0}W {cur_v}MB",
                hps
            );
            section_results.push(ss);
        }
        println!();
    }

    // ── Final report ──────────────────────────────────────────────────────────
    let total_wall_s = global_start.elapsed().as_secs_f64();

    print_rule('═', 72);
    println!("  BENCHMARK COMPLETE  ({:.1}s total)", total_wall_s);
    print_rule('═', 72);
    println!();

    // Section summary table
    println!("  {:>18}  {:>9}  {:>7}  {:>7}  {:>7}  {:>8}  {:>7}",
        "Section", "Wall_ms", "PkTemp", "AvgPow", "PkPow", "PkVRAM", "PkClock");
    println!("  {}", "─".repeat(72));
    for ss in &section_results {
        println!(
            "  {:>18}  {:>8.1}ms  {:>5}°C  {:>5.0}W  {:>5.0}W  {:>6}MB  {:>5}MHz",
            ss.name,
            ss.wall_ms,
            ss.peak_temp_c,
            ss.avg_power_w,
            ss.peak_power_w,
            ss.peak_vram_mb,
            ss.peak_sm_mhz,
        );
    }
    println!();

    // Aggregate GPU stats across the full run
    let all_temps:  Vec<u32> = section_results.iter().map(|s| s.peak_temp_c).collect();
    let all_powers: Vec<f64> = section_results.iter().map(|s| s.peak_power_w).collect();
    let all_vrams:  Vec<u64> = section_results.iter().map(|s| s.peak_vram_mb).collect();

    let global_peak_temp  = all_temps.iter().copied().max().unwrap_or(0);
    let global_peak_power = all_powers.iter().cloned().fold(0.0f64, f64::max);
    let global_peak_vram  = all_vrams.iter().copied().max().unwrap_or(0);

    println!("  Run-wide peaks:");
    println!("    Temperature:    {}°C", global_peak_temp);
    println!("    Power draw:     {:.0}W", global_peak_power);
    println!("    VRAM used:      {} MB ({:.1} GB)", global_peak_vram, global_peak_vram as f64 / 1024.0);

    if global_peak_temp >= 83 {
        println!();
        println!("  ⚠  THERMAL THROTTLE DETECTED: peak {}°C ≥ 83°C. Review cooling.", global_peak_temp);
    }

    print_rule('═', 72);
    println!();
}

// ── Shared NTT helper ─────────────────────────────────────────────────────────

/// Run interpolate → zero-pad → evaluate on every column, timing the full pass.
/// Each eval column is dropped immediately after its NTT — NOT accumulated.
/// This keeps peak VRAM at trace_all + one_eval instead of trace_all + all_evals.
fn do_ntt(
    d_cols: Vec<vortexstark::device::DeviceBuffer<u32>>,
    log_n: u32,
) -> f64 {
    use vortexstark::{circle::Coset, ntt, device::DeviceBuffer, cuda::ffi};

    let n = 1usize << log_n;
    let log_eval = log_n + 1;
    let eval_n = 2 * n;

    let trace_domain = Coset::half_coset(log_n);
    let eval_domain  = Coset::half_coset(log_eval);
    let inv = ntt::InverseTwiddleCache::new(&trace_domain);
    let fwd = ntt::ForwardTwiddleCache::new(&eval_domain);

    let t = Instant::now();
    for mut col in d_cols {
        ntt::interpolate(&mut col, &inv);
        let mut ev = DeviceBuffer::<u32>::alloc(eval_n);
        unsafe { ffi::cuda_zero_pad(col.as_ptr(), ev.as_mut_ptr(), n as u32, eval_n as u32); }
        drop(col);
        ntt::evaluate(&mut ev, &fwd);
        // ev drops here — only one eval col live at a time
    }
    t.elapsed().as_secs_f64() * 1000.0
}

/// Extract (temp, power, vram) — prefer live snapshot, fall back to section peaks.
fn live_or_peak(live: &Option<GpuSnapshot>, ss: &SectionStats) -> (u32, f64, u64) {
    match live {
        Some(s) => (s.temp_c, s.power_w, s.mem_used_mb),
        None    => (ss.peak_temp_c, ss.peak_power_w, ss.peak_vram_mb),
    }
}
