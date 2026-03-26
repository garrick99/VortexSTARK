/// VortexSTARK CLI: prove, verify, inspect Cairo programs and Fibonacci STARKs.
///
/// Usage:
///   stark_cli prove <log_n> [a] [b] [-o proof.bin]        — Fibonacci STARK
///   stark_cli prove-file <program.casm> [-o proof.bin]     — prove a CASM/Cairo0 file
///   stark_cli prove-starknet --class-hash <0x...>          — prove from Starknet RPC
///   stark_cli inspect <program.casm>                       — disassemble a CASM file
///   stark_cli fetch-block [--block <id>]                   — fetch Starknet block info
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
        /// Output proof file
        #[arg(short, long, default_value = "proof.bin")]
        output: String,
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
        /// Output proof file
        #[arg(short, long, default_value = "proof.bin")]
        output: String,
        /// Maximum execution steps
        #[arg(long, default_value = "1000000")]
        steps: usize,
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
        Commands::ProveFile { program, output, steps, log_n, entry_point } => {
            cmd_prove_file(&program, &output, steps, log_n, entry_point);
        }
        Commands::ProveStarknet { class_hash, rpc, output, steps } => {
            cmd_prove_starknet(&class_hash, &rpc, &output, steps);
        }
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

fn cmd_prove_file(path: &PathBuf, output: &str, max_steps: Option<usize>, log_n_override: Option<u32>, entry_point_override: Option<u64>) {
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

    // Compute log_n (next power of 2)
    let log_n = log_n_override.unwrap_or_else(|| {
        let mut l = 0u32;
        while (1usize << l) < n_steps { l += 1; }
        l.max(4) // minimum log_n=4 for FRI
    });

    eprintln!("Proving Cairo STARK: {n_steps} steps, log_n={log_n}");
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
}

fn cmd_prove_starknet(class_hash: &str, rpc_url: &str, output: &str, max_steps: usize) {
    eprintln!("Fetching CASM from Starknet RPC...");
    eprintln!("  RPC:        {rpc_url}");
    eprintln!("  Class hash: {class_hash}");

    let rt = tokio::runtime::Runtime::new().expect("cannot create tokio runtime");
    let program = rt.block_on(async {
        let client = starknet_rpc::StarknetClient::new(rpc_url);
        client.get_compiled_casm(class_hash).await
    }).unwrap_or_else(|e| { eprintln!("ERROR: {e}"); std::process::exit(1); });

    casm_loader::print_summary(&program);

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

    eprintln!("Proving Cairo STARK: {n_steps} steps, log_n={log_n}");
    ffi::init_memory_pool();

    let t0 = Instant::now();
    let proof = vortexstark::cairo_air::prover::cairo_prove_program(&program, n_steps, log_n)
        .unwrap_or_else(|e| { eprintln!("ERROR: {e}"); std::process::exit(1); });
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

    eprintln!("  prove: {prove_ms:.1}ms");
    if !program.hints.is_empty() {
        eprintln!("  hints executed: {} hint sites", program.hints.len());
    }

    let bytes = serialize_cairo_proof(&proof);
    let proof_size = bytes.len();
    let file = std::fs::File::create(output).expect("cannot create output file");
    let mut w = BufWriter::new(file);
    w.write_all(&bytes).expect("write failed");
    w.flush().expect("flush failed");

    eprintln!("  proof size: {} bytes ({:.1} KB)", proof_size, proof_size as f64 / 1024.0);
    eprintln!("  written to: {output}");
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

    ffi::init_memory_pool();

    let proof = deserialize_proof(&bytes);
    let n: u64 = 1u64 << proof.log_trace_size;
    eprintln!("  log_n={}, n={n}, public_inputs=({}, {})",
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

// --- Cairo proof serialization (minimal) ---

const CAIRO_MAGIC: u32 = 0x4341_4952; // "CAIR"
const CAIRO_VERSION: u32 = 1;

fn serialize_cairo_proof(proof: &vortexstark::cairo_air::prover::CairoProof) -> Vec<u8> {
    let mut w = Vec::new();
    write_u32(&mut w, CAIRO_MAGIC);
    write_u32(&mut w, CAIRO_VERSION);
    write_u32(&mut w, proof.log_trace_size);
    write_u32(&mut w, proof.public_inputs.initial_pc);
    write_u32(&mut w, proof.public_inputs.initial_ap);
    write_u32(&mut w, proof.public_inputs.n_steps as u32);
    write_u32_slice(&mut w, &proof.public_inputs.program_hash);
    write_u32_slice(&mut w, &proof.trace_commitment);
    write_u32_slice(&mut w, &proof.interaction_commitment);
    write_u32_slice(&mut w, &proof.quotient_commitment);

    write_u32(&mut w, proof.fri_commitments.len() as u32);
    for c in &proof.fri_commitments { write_u32_slice(&mut w, c); }

    write_u32(&mut w, proof.fri_last_layer.len() as u32);
    for v in &proof.fri_last_layer {
        write_u32_slice(&mut w, &v.to_u32_array());
    }

    write_u32(&mut w, proof.query_indices.len() as u32);
    for &qi in &proof.query_indices { write_u32(&mut w, qi as u32); }

    write_decommitment_4(&mut w, &proof.quotient_decommitment);

    write_u32(&mut w, proof.fri_decommitments.len() as u32);
    for d in &proof.fri_decommitments { write_decommitment_4(&mut w, d); }

    w
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
