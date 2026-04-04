/// VortexSTARK CLI: prove, verify, inspect Cairo programs and Fibonacci STARKs.
///
/// Usage:
///   stark_cli prove <log_n> [a] [b] [-o proof.bin]              — Fibonacci STARK
///   stark_cli prove-file <program.casm> [-o proof.bin]           — prove a CASM/Cairo0 file (M31 AIR)
///   stark_cli prove-starknet --class-hash <0x...>                — prove from Starknet RPC (M31 AIR)
///   stark_cli prove-cairo <program.json> [-o proof.json]         — stwo-cairo GPU prover (Starknet-ready)
///   stark_cli verify-stwo <proof.json>                           — verify stwo-compatible proof
///   stark_cli inspect <program.casm>                             — disassemble a CASM file
///   stark_cli fetch-block [--block <id>]                         — fetch Starknet block info
///   stark_cli verify <proof.bin>
///   stark_cli bench <log_n>

use clap::{Parser, Subcommand};
use vortexstark::cuda::ffi;
use vortexstark::field::M31;
use vortexstark::prover::{self, StarkProof, QueryDecommitment};
use vortexstark::verifier;
use vortexstark::field::QM31;
use vortexstark::cairo_air::casm_loader;
use vortexstark::cairo_air::starknet_rpc;
use vortexstark::stwo_export;
use std::io::{Read, Write, BufWriter, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "stark_cli", about = "VortexSTARK: GPU-native Circle STARK prover")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Prove a Fibonacci STARK
    Prove {
        /// Log2 of trace size
        log_n: u32,
        /// First Fibonacci input (default: 1)
        #[arg(default_value = "1")]
        a: u32,
        /// Second Fibonacci input (default: 1)
        #[arg(default_value = "1")]
        b: u32,
        /// Output proof file
        #[arg(short, long, default_value = "proof.bin")]
        output: String,
    },

    /// Prove a CASM or Cairo 0 program from file
    ProveFile {
        /// Path to .casm or compiled Cairo 0 JSON file
        program: PathBuf,
        /// Output proof file (native binary format)
        #[arg(short, long, default_value = "proof.bin")]
        output: String,
        /// Also export stwo-compatible proof as JSON (for stwo verifier ecosystem)
        #[arg(long)]
        stwo_output: Option<String>,
        /// Maximum execution steps (auto-detect if omitted)
        #[arg(long)]
        steps: Option<usize>,
        /// Log2 of trace size (auto from steps if omitted)
        #[arg(long)]
        log_n: Option<u32>,
        /// Entry point offset (uses first External entry point if omitted)
        #[arg(long)]
        entry_point: Option<u64>,
    },

    /// Prove a contract from Starknet RPC
    ProveStarknet {
        /// Contract class hash (0x...)
        #[arg(long)]
        class_hash: String,
        /// Starknet RPC endpoint URL
        #[arg(long, default_value = starknet_rpc::MAINNET_RPC)]
        rpc: String,
        /// Output proof file (native binary format, or cairo-serde JSON with --stwo-cairo)
        #[arg(short, long, default_value = "proof.bin")]
        output: String,
        /// Also export stwo-compatible proof as JSON (for stwo verifier ecosystem)
        #[arg(long)]
        stwo_output: Option<String>,
        /// Maximum execution steps
        #[arg(long, default_value = "1000000")]
        steps: usize,
        /// Initial contract storage as JSON: '{"0xkey": "0xvalue", ...}'
        #[arg(long)]
        storage: Option<String>,
        /// Caller address (felt252 hex, e.g. 0x1234)
        #[arg(long, default_value = "0x0")]
        caller: String,
        /// Contract address (felt252 hex)
        #[arg(long, default_value = "0x1")]
        contract_address: String,
        /// Entry point selector (felt252 hex)
        #[arg(long, default_value = "0x0")]
        entry_point_selector: String,
        /// Block number for get_block_hash / get_execution_info
        #[arg(long, default_value = "1000")]
        block_number: u64,
        /// Use the stwo-cairo Starknet-compatible pipeline (outputs cairo-serde proof).
        /// Fetches CASM via RPC and passes to the stwo-cairo `run_and_prove` prover.
        /// Requires stwo-cairo binaries to be built in WSL.
        #[arg(long)]
        stwo_cairo: bool,
        /// Hash function for stwo-cairo path: blake2s (default) or poseidon252 (mainnet)
        #[arg(long, default_value = "blake2s")]
        hash: String,
    },

    /// Prove a Cairo program with the Starknet-compatible stwo-cairo GPU pipeline.
    ///
    /// Uses the stwo-cairo prover (CudaBackend) via WSL. Outputs a proof in
    /// `cairo_serde` format (array of field elements) that can be submitted
    /// directly to the stwo_cairo_verifier contract on Starknet.
    ///
    /// Requires the stwo-cairo `run_and_prove` binary to be built first:
    ///   cd /path/to/stwo-cairo/stwo_cairo_prover
    ///   wsl -- cargo build --release --bin run_and_prove
    ProveCairo {
        /// Path to compiled Cairo JSON program (output of `cairo-compile` or `starknet-compile`)
        program: PathBuf,
        /// Output proof file (cairo-serde JSON by default — submittable to Starknet)
        #[arg(short, long, default_value = "proof.cairo-serde.json")]
        output: PathBuf,
        /// Proof format: cairo-serde (for Starknet), json (debug), binary (compact)
        #[arg(long, default_value = "cairo-serde")]
        format: String,
        /// Hash function: blake2s (fastest), blake2s_m31, poseidon252 (Starknet mainnet)
        #[arg(long, default_value = "blake2s")]
        hash: String,
        /// Proof-of-work bits (security: pow_bits + log_blowup*n_queries ≥ 96)
        #[arg(long, default_value = "26")]
        pow_bits: u32,
        /// Number of FRI queries (with log_blowup=1: 70 queries → 96-bit security)
        #[arg(long, default_value = "70")]
        n_queries: u32,
        /// Verify the proof after generation (runs Rust verifier)
        #[arg(long)]
        verify: bool,
        /// Path to prover params JSON file (overrides --hash/--pow-bits/--n-queries)
        #[arg(long)]
        params: Option<PathBuf>,
        /// Path to stwo-cairo `run_and_prove` binary (WSL Linux path)
        /// Default: /mnt/c/<username>/stwo-cairo/stwo_cairo_prover/target/release/run_and_prove
        #[arg(long)]
        prover_bin: Option<String>,
    },

    /// Verify a stwo-compatible proof JSON (offline verification without GPU)
    VerifyStwo {
        /// Path to stwo-compatible proof JSON file
        proof: PathBuf,
    },

    /// Inspect/disassemble a CASM file
    Inspect {
        /// Path to .casm or compiled Cairo 0 JSON file
        program: PathBuf,
        /// Max instructions to disassemble
        #[arg(long, default_value = "50")]
        max: usize,
    },

    /// Fetch and display a Starknet block
    FetchBlock {
        /// Block number (uses "latest" if omitted)
        #[arg(long)]
        block: Option<u64>,
        /// Starknet RPC endpoint URL
        #[arg(long, default_value = starknet_rpc::MAINNET_RPC)]
        rpc: String,
        /// Also fetch CASM for deploy transactions
        #[arg(long)]
        fetch_classes: bool,
    },

    /// Verify a proof file
    Verify {
        /// Path to proof file
        proof: String,
    },

    /// Benchmark Fibonacci STARK proving
    Bench {
        /// Log2 of trace size
        log_n: u32,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Prove { log_n, a, b, output } => cmd_prove(log_n, a, b, &output),
        Commands::ProveFile { program, output, stwo_output, steps, log_n, entry_point } => {
            cmd_prove_file(&program, &output, stwo_output.as_deref(), steps, log_n, entry_point);
        }
        Commands::ProveStarknet {
            class_hash, rpc, output, stwo_output, steps,
            storage, caller, contract_address, entry_point_selector, block_number,
            stwo_cairo, hash,
        } => {
            if stwo_cairo {
                cmd_prove_starknet_stwo_cairo(
                    &class_hash, &rpc, &output, &hash, steps, storage.as_deref(),
                    &caller, &contract_address, &entry_point_selector, block_number,
                );
            } else {
                cmd_prove_starknet(
                    &class_hash, &rpc, &output, stwo_output.as_deref(), steps,
                    storage.as_deref(), &caller, &contract_address,
                    &entry_point_selector, block_number,
                );
            }
        }
        Commands::ProveCairo { program, output, format, hash, pow_bits, n_queries, verify, params, prover_bin } => {
            cmd_prove_cairo(&program, &output, &format, &hash, pow_bits, n_queries, verify, params.as_deref(), prover_bin.as_deref());
        }
        Commands::VerifyStwo { proof } => cmd_verify_stwo(&proof),
        Commands::Inspect { program, max } => cmd_inspect(&program, max),
        Commands::FetchBlock { block, rpc, fetch_classes } => {
            cmd_fetch_block(block, &rpc, fetch_classes);
        }
        Commands::Verify { proof } => cmd_verify(&proof),
        Commands::Bench { log_n } => cmd_bench(log_n),
    }
}

fn cmd_prove(log_n: u32, a: u32, b: u32, output: &str) {
    ffi::init_memory_pool();

    let n: u64 = 1u64 << log_n;
    eprintln!("Proving Fibonacci STARK: log_n={log_n}, n={n}, a={a}, b={b}");

    let t0 = Instant::now();
    let proof = prover::prove_lean_timed(M31(a), M31(b), log_n);
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    match verifier::verify(&proof) {
        Ok(()) => {
            let verify_ms = t1.elapsed().as_secs_f64() * 1000.0;
            eprintln!("Proof verified OK");
            eprintln!("  prove:  {prove_ms:.1}ms");
            eprintln!("  verify: {verify_ms:.1}ms");
        }
        Err(e) => {
            eprintln!("VERIFICATION FAILED: {e}");
            std::process::exit(1);
        }
    }

    let t2 = Instant::now();
    let bytes = serialize_proof(&proof);
    let proof_size = bytes.len();
    let file = std::fs::File::create(output).expect("cannot create output file");
    let mut w = BufWriter::new(file);
    w.write_all(&bytes).expect("write failed");
    w.flush().expect("flush failed");
    let write_ms = t2.elapsed().as_secs_f64() * 1000.0;

    eprintln!("  proof size: {} bytes ({:.1} KB)", proof_size, proof_size as f64 / 1024.0);
    eprintln!("  written to: {output} ({write_ms:.1}ms)");
}

fn cmd_prove_file(path: &PathBuf, output: &str, stwo_output: Option<&str>, max_steps: Option<usize>, log_n_override: Option<u32>, entry_point_override: Option<u64>) {
    eprintln!("Loading program: {}", path.display());

    let mut program = casm_loader::load_program(path)
        .unwrap_or_else(|e| { eprintln!("ERROR: {e}"); std::process::exit(1); });

    if let Some(ep) = entry_point_override {
        program.entry_point = ep;
    }

    casm_loader::print_summary(&program);

    // Detect execution steps
    let default_max = 1 << 20; // 1M steps max for auto-detection
    let n_steps = max_steps.unwrap_or_else(|| {
        eprintln!("Auto-detecting execution steps (max {default_max})...");
        let steps = casm_loader::detect_steps(&program, default_max);
        eprintln!("  Detected: {steps} steps");
        steps
    });

    if n_steps == 0 {
        eprintln!("ERROR: program executed 0 steps");
        std::process::exit(1);
    }

    // Compute log_n: smallest l such that 2^l >= n_steps.
    let log_n = log_n_override.unwrap_or_else(|| {
        let mut l = 0u32;
        while (1usize << l) < n_steps { l += 1; }
        l.max(4) // minimum log_n=4 for FRI
    });

    let n = 1usize << log_n;
    eprintln!("Proving Cairo STARK: {n_steps} steps → padded to {n} (log_n={log_n})");
    ffi::init_memory_pool();

    let t0 = Instant::now();
    let proof = vortexstark::cairo_air::prover::cairo_prove_program(&program, n_steps, log_n)
        .unwrap_or_else(|e| { eprintln!("ERROR: {e}"); std::process::exit(1); });
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

    eprintln!("  prove: {prove_ms:.1}ms");
    if !program.hints.is_empty() {
        eprintln!("  hints executed: {} hint sites", program.hints.len());
    }
    eprintln!("  program hash: {:08x}{:08x}...",
        proof.public_inputs.program_hash[0],
        proof.public_inputs.program_hash[1]);

    // Serialize Cairo proof
    let bytes = serialize_cairo_proof(&proof);
    let proof_size = bytes.len();
    let file = std::fs::File::create(output).expect("cannot create output file");
    let mut w = BufWriter::new(file);
    w.write_all(&bytes).expect("write failed");
    w.flush().expect("flush failed");

    eprintln!("  proof size: {} bytes ({:.1} KB)", proof_size, proof_size as f64 / 1024.0);
    eprintln!("  written to: {output}");

    if let Some(stwo_path) = stwo_output {
        export_stwo_proof(&proof, stwo_path);
    }
}

fn parse_hex_u64(s: &str) -> u64 {
    let s = s.trim().trim_start_matches("0x").trim_start_matches("0X");
    u64::from_str_radix(s, 16).unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
fn cmd_prove_starknet(
    class_hash: &str,
    rpc_url: &str,
    output: &str,
    stwo_output: Option<&str>,
    max_steps: usize,
    storage_json: Option<&str>,
    caller: &str,
    contract_address: &str,
    entry_point_selector: &str,
    block_number: u64,
) {
    use vortexstark::cairo_air::hints::SyscallState;

    eprintln!("Fetching CASM from Starknet RPC...");
    eprintln!("  RPC:        {rpc_url}");
    eprintln!("  Class hash: {class_hash}");

    let rt = tokio::runtime::Runtime::new().expect("cannot create tokio runtime");
    let program = rt.block_on(async {
        let client = starknet_rpc::StarknetClient::new(rpc_url);
        client.get_compiled_casm(class_hash).await
    }).unwrap_or_else(|e| { eprintln!("ERROR: {e}"); std::process::exit(1); });

    casm_loader::print_summary(&program);

    // Build SyscallState from CLI args
    let mut sc = SyscallState::default();
    sc.caller_address       = parse_hex_u64(caller);
    sc.contract_address     = parse_hex_u64(contract_address);
    sc.entry_point_selector = parse_hex_u64(entry_point_selector);
    sc.block_number         = block_number;

    if let Some(json) = storage_json {
        match serde_json::from_str::<serde_json::Value>(json) {
            Ok(serde_json::Value::Object(map)) => {
                for (k, v) in &map {
                    let key = parse_hex_u64(k.as_str());
                    let val = v.as_str().map(parse_hex_u64)
                        .or_else(|| v.as_u64())
                        .unwrap_or(0);
                    sc.storage.insert(key, val);
                }
                eprintln!("  storage:    {} entries", sc.storage.len());
            }
            _ => {
                eprintln!("WARN: --storage JSON parse failed, using empty storage");
            }
        }
    }

    // Detect steps
    eprintln!("Auto-detecting execution steps (max {max_steps})...");
    let n_steps = casm_loader::detect_steps(&program, max_steps);
    eprintln!("  Detected: {n_steps} steps");

    if n_steps == 0 {
        eprintln!("ERROR: program executed 0 steps (may require hints or calldata)");
        std::process::exit(1);
    }

    let mut log_n = 0u32;
    while (1usize << log_n) < n_steps { log_n += 1; }
    log_n = log_n.max(4);

    let n = 1usize << log_n;
    eprintln!("Proving Cairo STARK: {n_steps} steps → padded to {n} (log_n={log_n})");
    ffi::init_memory_pool();

    let t0 = Instant::now();
    let (proof, sc_out) =
        vortexstark::cairo_air::prover::cairo_prove_program_with_syscalls(
            &program, n_steps, log_n, sc,
        ).unwrap_or_else(|e| { eprintln!("ERROR: {e}"); std::process::exit(1); });
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

    eprintln!("  prove: {prove_ms:.1}ms");
    if !program.hints.is_empty() {
        eprintln!("  hints executed: {} hint sites", program.hints.len());
    }
    if !sc_out.events.is_empty() {
        eprintln!("  events emitted: {}", sc_out.events.len());
    }
    if sc_out.storage.values().any(|&v| v != 0) {
        let writes: Vec<_> = sc_out.storage.iter().filter(|(_, v)| **v != 0).collect();
        eprintln!("  storage writes: {}", writes.len());
    }

    let bytes = serialize_cairo_proof(&proof);
    let proof_size = bytes.len();
    let file = std::fs::File::create(output).expect("cannot create output file");
    let mut w = BufWriter::new(file);
    w.write_all(&bytes).expect("write failed");
    w.flush().expect("flush failed");

    eprintln!("  proof size: {} bytes ({:.1} KB)", proof_size, proof_size as f64 / 1024.0);
    eprintln!("  written to: {output}");

    if let Some(stwo_path) = stwo_output {
        export_stwo_proof(&proof, stwo_path);
    }
}

/// Export a CairoProof as stwo-compatible JSON to a file.
fn export_stwo_proof(proof: &vortexstark::cairo_air::prover::CairoProof, path: &str) {
    let t0 = Instant::now();
    match stwo_export::cairo_proof_to_json(proof) {
        Ok(json) => {
            match std::fs::write(path, &json) {
                Ok(()) => {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("  stwo proof: {} bytes ({:.1} KB) → {path} ({ms:.1}ms)",
                        json.len(), json.len() as f64 / 1024.0);
                }
                Err(e) => eprintln!("ERROR writing stwo proof to {path}: {e}"),
            }
        }
        Err(e) => eprintln!("ERROR serializing stwo proof: {e}"),
    }
}

/// Verify a stwo-compatible proof JSON file (offline, no GPU required).
///
/// Checks structural validity: JSON parse, commitment count, FRI last layer,
/// and PoW nonce. Full Merkle + FRI integrity requires the native CairoProof;
/// use `verify <proof.bin>` for end-to-end verification.
fn cmd_verify_stwo(proof_path: &PathBuf) {
    let json = std::fs::read_to_string(proof_path)
        .unwrap_or_else(|e| { eprintln!("ERROR reading {}: {e}", proof_path.display()); std::process::exit(1); });

    let two_proof: stwo_export::TwoStarkProof = serde_json::from_str(&json)
        .unwrap_or_else(|e| { eprintln!("ERROR parsing proof JSON: {e}"); std::process::exit(1); });

    eprintln!("stwo proof: {}", proof_path.display());
    eprintln!("  commitments:   {}", two_proof.commitments.len());
    eprintln!("  query count:   {}", two_proof.query_indices.len());
    eprintln!("  trees:         {}", two_proof.tree_meta.len());
    eprintln!("  FRI layers:    {} inner + last", two_proof.fri_proof.inner_layers.len());
    eprintln!("  last layer sz: {} coeffs", two_proof.fri_proof.last_layer_poly.len());
    eprintln!("  pow nonce:     {}", two_proof.proof_of_work);
    eprintln!("  log_blowup:    {}", two_proof.config.log_blowup_factor);
    eprintln!("  n_queries:     {}", two_proof.config.n_queries);

    // Structural checks
    let mut ok = true;

    if two_proof.commitments.len() != two_proof.tree_meta.len() {
        eprintln!("ERROR: commitments.len() ({}) != tree_meta.len() ({})",
            two_proof.commitments.len(), two_proof.tree_meta.len());
        ok = false;
    }
    if two_proof.decommitments.len() != two_proof.tree_meta.len() {
        eprintln!("ERROR: decommitments.len() ({}) != tree_meta.len() ({})",
            two_proof.decommitments.len(), two_proof.tree_meta.len());
        ok = false;
    }
    if two_proof.queried_values.len() != two_proof.tree_meta.len() {
        eprintln!("ERROR: queried_values.len() ({}) != tree_meta.len() ({})",
            two_proof.queried_values.len(), two_proof.tree_meta.len());
        ok = false;
    }
    if two_proof.query_indices.len() != two_proof.config.n_queries as usize {
        eprintln!("ERROR: query_indices.len() ({}) != config.n_queries ({})",
            two_proof.query_indices.len(), two_proof.config.n_queries);
        ok = false;
    }
    let last_sz = 1usize << two_proof.config.log_last_layer_degree_bound;
    if two_proof.fri_proof.last_layer_poly.len() != last_sz {
        eprintln!("ERROR: last_layer_poly.len() ({}) != 2^log_last ({})",
            two_proof.fri_proof.last_layer_poly.len(), last_sz);
        ok = false;
    }

    if ok {
        eprintln!("Structure: OK");
        eprintln!("Proof verified OK (structural check)");
    } else {
        eprintln!("Proof INVALID");
        std::process::exit(1);
    }
}

/// Prove a Starknet contract using the stwo-cairo Starknet-compatible pipeline.
///
/// Fetches CASM via RPC, writes to a temp file, then delegates to `cmd_prove_cairo`
/// which calls `run_and_prove` via WSL. The output is a cairo-serde proof ready for
/// submission to the `stwo_cairo_verifier` contract on Starknet.
#[allow(clippy::too_many_arguments)]
fn cmd_prove_starknet_stwo_cairo(
    class_hash: &str,
    rpc_url: &str,
    output: &str,
    hash: &str,
    _max_steps: usize,
    _storage_json: Option<&str>,
    _caller: &str,
    _contract_address: &str,
    _entry_point_selector: &str,
    _block_number: u64,
) {
    eprintln!("Fetching CASM from Starknet RPC (stwo-cairo path)...");
    eprintln!("  RPC:        {rpc_url}");
    eprintln!("  Class hash: {class_hash}");

    let rt = tokio::runtime::Runtime::new().expect("cannot create tokio runtime");
    // Fetch raw compiled CASM JSON — pass directly to run_and_prove without parsing.
    let raw_json: serde_json::Value = rt.block_on(async {
        let client = vortexstark::cairo_air::starknet_rpc::StarknetClient::new(rpc_url);
        client.get_compiled_casm_raw(class_hash).await
    }).unwrap_or_else(|e| { eprintln!("ERROR fetching CASM: {e}"); std::process::exit(1); });

    // Write to temp file accessible from WSL
    let tmp_casm = std::env::temp_dir().join("vortex_starknet_casm.json");
    std::fs::write(&tmp_casm, serde_json::to_string_pretty(&raw_json).unwrap())
        .expect("failed to write CASM temp file");
    eprintln!("  CASM size:  {} bytes", serde_json::to_string(&raw_json).unwrap().len());

    let output_path = PathBuf::from(output);
    cmd_prove_cairo(
        &tmp_casm,
        &output_path,
        "cairo-serde",
        hash,
        26,
        70,
        true,   // always verify
        None,
        None,
    );
}

/// Convert a Windows path to a WSL /mnt/… path.
///
/// Handles Windows extended-length paths (\\?\C:\…) and regular paths (C:\…).
fn win_to_wsl(p: &std::path::Path) -> String {
    // Prefer canonicalized absolute path; fall back to the raw path.
    let raw = p.canonicalize()
        .unwrap_or_else(|_| p.to_path_buf())
        .to_string_lossy()
        .replace('\\', "/");

    // Strip Windows extended-length prefix: \\?\ or //./
    let s = if raw.starts_with("//") {
        // e.g. "//\?/C:/Users/…" or "//?/C:/Users/…"
        // Find the drive letter after the prefix
        if let Some(pos) = raw[2..].find('/') {
            let after_prefix = &raw[2 + pos + 1..]; // skip "//xxxx/"
            after_prefix
        } else {
            raw.as_str()
        }
    } else {
        raw.as_str()
    };

    // Now s should look like "C:/Users/…"
    if s.len() >= 2 && s.as_bytes()[1] == b':' {
        let drive = s[..1].to_lowercase();
        format!("/mnt/{}{}", drive, &s[2..])
    } else if s.starts_with('/') {
        // Already a Unix path (e.g. relative paths that canonicalize weirdly)
        s.to_owned()
    } else {
        // Fallback: just forward-slash the path
        format!("/{s}")
    }
}

/// Invoke the stwo-cairo `run_and_prove` binary through WSL to produce a
/// Starknet-compatible Cairo STARK proof.
///
/// Security: all arguments are passed as individual WSL args (no shell interpolation).
#[allow(clippy::too_many_arguments)]
fn cmd_prove_cairo(
    program: &PathBuf,
    output: &PathBuf,
    format: &str,
    hash: &str,
    pow_bits: u32,
    n_queries: u32,
    verify: bool,
    params_file: Option<&std::path::Path>,
    prover_bin: Option<&str>,
) {
    let default_bin =
        "/mnt/c/Users/user/stwo-cairo/stwo_cairo_prover/target/release/run_and_prove";
    let bin = prover_bin.unwrap_or(default_bin);

    // Build params JSON into a temp file (accessible from both Windows and WSL)
    let params_wsl_path: String;
    let _tmp_holder;   // keep NamedTempFile alive until after the subprocess exits
    let params_arg: String;

    if let Some(pf) = params_file {
        params_arg = win_to_wsl(pf);
    } else {
        let params_json = serde_json::json!({
            "channel_hash": hash,
            "channel_salt": 0,
            "pcs_config": {
                "pow_bits": pow_bits,
                "fri_config": {
                    "log_last_layer_degree_bound": 0,
                    "log_blowup_factor": 1,
                    "n_queries": n_queries,
                    "line_fold_step": 1
                },
                "lifting_log_size": null
            },
            "preprocessed_trace": "canonical_without_pedersen",
            "store_polynomials_coefficients": true,
            "include_all_preprocessed_columns": false
        });
        // Write to %TEMP% — accessible as /mnt/c/Users/…/AppData/Local/Temp/… in WSL
        let tmp_path = std::env::temp_dir().join("vortex_prove_cairo_params.json");
        std::fs::write(&tmp_path, serde_json::to_string_pretty(&params_json).unwrap())
            .expect("failed to write params JSON");
        params_wsl_path = win_to_wsl(&tmp_path);
        params_arg = params_wsl_path.clone();
        _tmp_holder = tmp_path;
    }

    let program_wsl = win_to_wsl(program);
    // For output, use absolute path; create parent dirs if needed
    let output_abs = if output.is_absolute() {
        output.clone()
    } else {
        std::env::current_dir().unwrap().join(output)
    };
    if let Some(parent) = output_abs.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let output_wsl = win_to_wsl(&output_abs);

    eprintln!("stwo-cairo GPU prover (Starknet-compatible)");
    eprintln!("  program:    {}", program.display());
    eprintln!("  output:     {}", output_abs.display());
    eprintln!("  format:     {format}");
    eprintln!("  hash:       {hash}");
    eprintln!("  security:   {}+{}×{}={} bits", pow_bits, 1u32, n_queries, pow_bits + n_queries);
    eprintln!("  binary:     {bin}");

    let mut args: Vec<&str> = vec![
        "--program",      &program_wsl,
        "--proof_path",   &output_wsl,
        "--proof-format", format,
        "--params_json",  &params_arg,
    ];
    let verify_str;
    if verify {
        verify_str = "--verify";
        args.push(verify_str);
    }

    // Build a shell command that:
    //  1. Sources the login profile (CUDA paths, cargo paths)
    //  2. Invokes run_and_prove with properly single-quoted arguments
    // Single-quoting all user-supplied paths prevents shell injection.
    fn shell_quote(s: &str) -> String {
        // Wrap in single quotes, escaping embedded single quotes as '\''
        format!("'{}'", s.replace('\'', r"'\''"))
    }

    let mut shell_cmd = format!(
        "export PATH=\"$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH\" && \
         export LD_LIBRARY_PATH=\"/usr/local/cuda-13.2/lib64:${{LD_LIBRARY_PATH:-}}\" && \
         {} --program {} --proof_path {} --proof-format {}",
        shell_quote(bin),
        shell_quote(&program_wsl),
        shell_quote(&output_wsl),
        shell_quote(format),
    );
    shell_cmd.push_str(&format!(" --params_json {}", shell_quote(&params_arg)));
    if verify {
        shell_cmd.push_str(" --verify");
    }

    let t0 = Instant::now();
    let status = std::process::Command::new("wsl")
        .args(["-d", "Ubuntu", "--", "bash", "-lc", &shell_cmd])
        .status();

    match status {
        Ok(s) if s.success() => {
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            // Report proof file size
            let proof_size = std::fs::metadata(&output_abs).map(|m| m.len()).unwrap_or(0);
            eprintln!("\n  total time: {elapsed_ms:.0}ms");
            eprintln!("  proof size: {} bytes ({:.1} KB)", proof_size, proof_size as f64 / 1024.0);
            eprintln!("  written to: {}", output_abs.display());
            if format == "cairo-serde" {
                let verifier_feature = if hash.starts_with("poseidon") {
                    "poseidon252_verifier"
                } else {
                    "blake2s_verifier"
                };
                eprintln!("\n  Proof is in cairo-serde format (Starknet-submittable).");
                eprintln!("  Verify locally with scarb (requires scarb + stwo_cairo_verifier):");
                eprintln!("    cd /path/to/stwo-cairo/stwo_cairo_verifier");
                eprintln!("    scarb execute --package stwo_cairo_verifier \\");
                eprintln!("      --arguments-file {} \\", output_abs.display());
                eprintln!("      --output standard --target standalone \\");
                eprintln!("      --features {verifier_feature}");
            }
        }
        Ok(s) => {
            eprintln!("ERROR: run_and_prove exited with status {s}");
            eprintln!("Ensure the binary is built:");
            eprintln!("  wsl -- bash -lc 'cd /path/to/stwo-cairo/stwo_cairo_prover && cargo build --release --bin run_and_prove'");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("ERROR launching WSL: {e}");
            eprintln!("Ensure WSL2 with Ubuntu is installed and the binary exists at:");
            eprintln!("  {bin}");
            std::process::exit(1);
        }
    }
}

fn cmd_inspect(path: &PathBuf, max_instructions: usize) {
    let program = casm_loader::load_program(path)
        .unwrap_or_else(|e| { eprintln!("ERROR: {e}"); std::process::exit(1); });

    casm_loader::print_summary(&program);
    eprintln!("\nDisassembly:");
    casm_loader::disassemble(&program, max_instructions);
}

fn cmd_fetch_block(block_num: Option<u64>, rpc_url: &str, fetch_classes: bool) {
    let rt = tokio::runtime::Runtime::new().expect("cannot create tokio runtime");
    rt.block_on(async {
        let client = starknet_rpc::StarknetClient::new(rpc_url);

        let block_id = match block_num {
            Some(n) => starknet_rpc::BlockId::number(n),
            None => starknet_rpc::BlockId::latest(),
        };

        let block = client.get_block_with_txs(&block_id).await
            .unwrap_or_else(|e| { eprintln!("ERROR: {e}"); std::process::exit(1); });

        starknet_rpc::print_block_summary(&block);

        if fetch_classes {
            // Collect unique class hashes from deploy transactions
            let class_hashes: Vec<String> = block.transactions.iter()
                .filter_map(|tx| tx.class_hash.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            if class_hashes.is_empty() {
                eprintln!("\nNo deploy transactions with class hashes in this block.");
                return;
            }

            eprintln!("\nFetching CASM for {} unique classes...", class_hashes.len());
            for ch in &class_hashes {
                match client.get_compiled_casm(ch).await {
                    Ok(program) => {
                        casm_loader::print_summary(&program);
                    }
                    Err(e) => {
                        eprintln!("  {}: ERROR: {e}", &ch[..18.min(ch.len())]);
                    }
                }
            }
        }
    });
}

fn cmd_verify(path: &str) {
    let mut file = BufReader::new(std::fs::File::open(path).expect("cannot open proof file"));
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read failed");

    eprintln!("Loaded proof: {} bytes ({:.1} KB)", bytes.len(), bytes.len() as f64 / 1024.0);

    assert!(bytes.len() >= 4, "proof file too small");
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());

    ffi::init_memory_pool();

    if magic == MAGIC {
        // Fibonacci STARK proof
        let proof = deserialize_proof(&bytes);
        let n: u64 = 1u64 << proof.log_trace_size;
        eprintln!("  Fibonacci STARK: log_n={}, n={n}, public_inputs=({}, {})",
            proof.log_trace_size, proof.public_inputs.0.0, proof.public_inputs.1.0);
        eprintln!("  {} FRI layers, {} queries", proof.fri_commitments.len(), proof.query_indices.len());

        let t0 = Instant::now();
        match verifier::verify(&proof) {
            Ok(()) => {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("VERIFIED OK ({ms:.1}ms)");
            }
            Err(e) => {
                eprintln!("VERIFICATION FAILED: {e}");
                std::process::exit(1);
            }
        }
    } else if magic == CAIRO_MAGIC {
        // Cairo VM STARK proof
        let proof = deserialize_cairo_proof(&bytes);
        let n: u64 = 1u64 << proof.log_trace_size;
        eprintln!("  Cairo STARK: log_n={}, n={n}, n_steps={}",
            proof.log_trace_size, proof.public_inputs.n_steps);
        eprintln!("  program_hash: {:08x}{:08x}...",
            proof.public_inputs.program_hash[0], proof.public_inputs.program_hash[1]);
        eprintln!("  {} FRI layers, {} queries",
            proof.fri_commitments.len(), proof.query_indices.len());

        let t0 = Instant::now();
        match vortexstark::cairo_air::prover::cairo_verify(&proof) {
            Ok(()) => {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("VERIFIED OK ({ms:.1}ms)");
            }
            Err(e) => {
                eprintln!("VERIFICATION FAILED: {e}");
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("ERROR: unknown proof format (magic={magic:#010x})");
        eprintln!("  Expected Fibonacci (0x{MAGIC:08x}) or Cairo (0x{CAIRO_MAGIC:08x})");
        std::process::exit(1);
    }
}

fn cmd_bench(log_n: u32) {
    ffi::init_memory_pool();

    let n: u64 = 1u64 << log_n;
    eprintln!("Benchmarking: log_n={log_n}, n={n}");

    // Warmup
    if log_n <= 20 {
        let _ = prover::prove(M31(1), M31(1), log_n.min(8));
    }

    let t0 = Instant::now();
    let proof = prover::prove_lean_timed(M31(1), M31(1), log_n);
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let result = verifier::verify(&proof);
    let verify_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let bytes = serialize_proof(&proof);

    eprintln!("\nResults:");
    eprintln!("  prove:      {prove_ms:.1}ms");
    eprintln!("  verify:     {verify_ms:.1}ms");
    eprintln!("  proof size: {} bytes ({:.1} KB)", bytes.len(), bytes.len() as f64 / 1024.0);
    eprintln!("  verified:   {}", if result.is_ok() { "YES" } else { "FAILED" });
    if let Err(e) = result {
        eprintln!("  error: {e}");
    }
}

// --- Fibonacci proof serialization ---

const MAGIC: u32 = 0x4B52_414B; // "KRAK"
const VERSION: u32 = 1;

fn write_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}

fn write_u32_slice(w: &mut Vec<u8>, s: &[u32]) {
    for &v in s { write_u32(w, v); }
}

fn write_decommitment_u32(w: &mut Vec<u8>, d: &QueryDecommitment<u32>) {
    let n = d.values.len() as u32;
    write_u32(w, n);
    write_u32_slice(w, &d.values);
    write_u32_slice(w, &d.sibling_values);
    for path in &d.auth_paths {
        write_u32(w, path.len() as u32);
        for hash in path { write_u32_slice(w, hash); }
    }
    for path in &d.sibling_auth_paths {
        write_u32(w, path.len() as u32);
        for hash in path { write_u32_slice(w, hash); }
    }
}

fn write_decommitment_4(w: &mut Vec<u8>, d: &QueryDecommitment<[u32; 4]>) {
    let n = d.values.len() as u32;
    write_u32(w, n);
    for v in &d.values { write_u32_slice(w, v); }
    for v in &d.sibling_values { write_u32_slice(w, v); }
    for path in &d.auth_paths {
        write_u32(w, path.len() as u32);
        for hash in path { write_u32_slice(w, hash); }
    }
    for path in &d.sibling_auth_paths {
        write_u32(w, path.len() as u32);
        for hash in path { write_u32_slice(w, hash); }
    }
}

fn serialize_proof(proof: &StarkProof) -> Vec<u8> {
    let mut w = Vec::new();
    write_u32(&mut w, MAGIC);
    write_u32(&mut w, VERSION);
    write_u32(&mut w, proof.log_trace_size);
    write_u32(&mut w, proof.public_inputs.0.0);
    write_u32(&mut w, proof.public_inputs.1.0);
    write_u32_slice(&mut w, &proof.trace_commitment);
    write_u32_slice(&mut w, &proof.quotient_commitment);

    write_u32(&mut w, proof.fri_commitments.len() as u32);
    for c in &proof.fri_commitments { write_u32_slice(&mut w, c); }

    write_u32(&mut w, proof.fri_last_layer.len() as u32);
    for v in &proof.fri_last_layer {
        write_u32_slice(&mut w, &v.to_u32_array());
    }

    write_u32(&mut w, proof.query_indices.len() as u32);
    for &qi in &proof.query_indices { write_u32(&mut w, qi as u32); }

    write_decommitment_u32(&mut w, &proof.trace_decommitment);
    write_decommitment_4(&mut w, &proof.quotient_decommitment);

    write_u32(&mut w, proof.fri_decommitments.len() as u32);
    for d in &proof.fri_decommitments { write_decommitment_4(&mut w, d); }

    w
}

// --- Cairo proof serialization (complete, serde_json) ---

const CAIRO_MAGIC: u32 = 0x4341_4952; // "CAIR"
const CAIRO_VERSION: u32 = 1;

fn serialize_cairo_proof(proof: &vortexstark::cairo_air::prover::CairoProof) -> Vec<u8> {
    let mut w = Vec::new();
    write_u32(&mut w, CAIRO_MAGIC);
    write_u32(&mut w, CAIRO_VERSION);
    let json = serde_json::to_vec(proof).expect("Cairo proof serialization failed");
    w.extend_from_slice(&json);
    w
}

fn deserialize_cairo_proof(data: &[u8]) -> vortexstark::cairo_air::prover::CairoProof {
    assert!(data.len() >= 8, "proof file too small");
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    assert_eq!(magic, CAIRO_MAGIC, "not a Cairo proof file (magic={magic:#010x})");
    assert_eq!(version, CAIRO_VERSION, "unsupported Cairo proof version {version}");
    serde_json::from_slice(&data[8..]).expect("Cairo proof deserialization failed")
}

// --- Fibonacci proof deserialization ---

fn read_u32(r: &mut &[u8]) -> u32 {
    let (bytes, rest) = r.split_at(4);
    *r = rest;
    u32::from_le_bytes(bytes.try_into().unwrap())
}

fn read_u32_array<const N: usize>(r: &mut &[u8]) -> [u32; N] {
    let mut arr = [0u32; N];
    for v in &mut arr { *v = read_u32(r); }
    arr
}

fn read_decommitment_u32(r: &mut &[u8]) -> QueryDecommitment<u32> {
    let n = read_u32(r) as usize;
    let values: Vec<u32> = (0..n).map(|_| read_u32(r)).collect();
    let sibling_values: Vec<u32> = (0..n).map(|_| read_u32(r)).collect();
    let auth_paths: Vec<Vec<[u32; 8]>> = (0..n).map(|_| {
        let depth = read_u32(r) as usize;
        (0..depth).map(|_| read_u32_array::<8>(r)).collect()
    }).collect();
    let sibling_auth_paths: Vec<Vec<[u32; 8]>> = (0..n).map(|_| {
        let depth = read_u32(r) as usize;
        (0..depth).map(|_| read_u32_array::<8>(r)).collect()
    }).collect();
    QueryDecommitment { values, sibling_values, auth_paths, sibling_auth_paths }
}

fn read_decommitment_4(r: &mut &[u8]) -> QueryDecommitment<[u32; 4]> {
    let n = read_u32(r) as usize;
    let values: Vec<[u32; 4]> = (0..n).map(|_| read_u32_array::<4>(r)).collect();
    let sibling_values: Vec<[u32; 4]> = (0..n).map(|_| read_u32_array::<4>(r)).collect();
    let auth_paths: Vec<Vec<[u32; 8]>> = (0..n).map(|_| {
        let depth = read_u32(r) as usize;
        (0..depth).map(|_| read_u32_array::<8>(r)).collect()
    }).collect();
    let sibling_auth_paths: Vec<Vec<[u32; 8]>> = (0..n).map(|_| {
        let depth = read_u32(r) as usize;
        (0..depth).map(|_| read_u32_array::<8>(r)).collect()
    }).collect();
    QueryDecommitment { values, sibling_values, auth_paths, sibling_auth_paths }
}

fn deserialize_proof(data: &[u8]) -> StarkProof {
    let mut r = data;
    let magic = read_u32(&mut r);
    assert_eq!(magic, MAGIC, "invalid proof file (bad magic)");
    let version = read_u32(&mut r);
    assert_eq!(version, VERSION, "unsupported proof version {version}");

    let log_n = read_u32(&mut r);
    let a = read_u32(&mut r);
    let b = read_u32(&mut r);
    let trace_commitment = read_u32_array::<8>(&mut r);
    let quotient_commitment = read_u32_array::<8>(&mut r);

    let n_fri = read_u32(&mut r) as usize;
    let fri_commitments: Vec<[u32; 8]> = (0..n_fri).map(|_| read_u32_array::<8>(&mut r)).collect();

    let n_last = read_u32(&mut r) as usize;
    let fri_last_layer: Vec<QM31> = (0..n_last)
        .map(|_| QM31::from_u32_array(read_u32_array::<4>(&mut r)))
        .collect();

    let n_queries = read_u32(&mut r) as usize;
    let query_indices: Vec<usize> = (0..n_queries).map(|_| read_u32(&mut r) as usize).collect();

    let trace_decommitment = read_decommitment_u32(&mut r);
    let quotient_decommitment = read_decommitment_4(&mut r);

    let n_fri_decom = read_u32(&mut r) as usize;
    let fri_decommitments: Vec<QueryDecommitment<[u32; 4]>> = (0..n_fri_decom)
        .map(|_| read_decommitment_4(&mut r))
        .collect();

    StarkProof {
        trace_commitment,
        quotient_commitment,
        fri_commitments,
        fri_last_layer,
        log_trace_size: log_n,
        public_inputs: (M31(a), M31(b)),
        query_indices,
        trace_decommitment,
        quotient_decommitment,
        fri_decommitments,
    }
}
