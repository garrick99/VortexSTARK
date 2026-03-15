/// VortexSTARK CLI: prove and verify Fibonacci STARK proofs.
///
/// Usage:
///   stark_cli prove <log_n> [a] [b] [-o proof.bin]
///   stark_cli verify <proof.bin>
///   stark_cli bench <log_n>

use vortex_stark::cuda::ffi;
use vortex_stark::field::M31;
use vortex_stark::prover::{self, StarkProof, QueryDecommitment};
use vortex_stark::verifier;
use vortex_stark::field::QM31;
use std::io::{Read, Write, BufWriter, BufReader};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  stark_cli prove <log_n> [a] [b] [-o proof.bin]");
        eprintln!("  stark_cli verify <proof.bin>");
        eprintln!("  stark_cli bench <log_n>");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "prove" => cmd_prove(&args[2..]),
        "verify" => cmd_verify(&args[2..]),
        "bench" => cmd_bench(&args[2..]),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
}

fn cmd_prove(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: stark_cli prove <log_n> [a] [b] [-o proof.bin]");
        std::process::exit(1);
    }

    let log_n: u32 = args[0].parse().expect("invalid log_n");
    let a = args.get(1).map(|s| s.parse::<u32>().expect("invalid a")).unwrap_or(1);
    let b = args.get(2).map(|s| s.parse::<u32>().expect("invalid b")).unwrap_or(1);

    let mut output_path = "proof.bin".to_string();
    for i in 0..args.len() {
        if args[i] == "-o" && i + 1 < args.len() {
            output_path = args[i + 1].clone();
        }
    }

    ffi::init_memory_pool();

    let n: u64 = 1u64 << log_n;
    eprintln!("Proving Fibonacci STARK: log_n={log_n}, n={n}, a={a}, b={b}");

    let t0 = Instant::now();
    let proof = if log_n <= 28 {
        prover::prove_lean(M31(a), M31(b), log_n)
    } else {
        prover::prove_lean_timed(M31(a), M31(b), log_n)
    };
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Verify before saving
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

    // Serialize and save
    let t2 = Instant::now();
    let bytes = serialize_proof(&proof);
    let proof_size = bytes.len();
    let file = std::fs::File::create(&output_path).expect("cannot create output file");
    let mut w = BufWriter::new(file);
    w.write_all(&bytes).expect("write failed");
    w.flush().expect("flush failed");
    let write_ms = t2.elapsed().as_secs_f64() * 1000.0;

    eprintln!("  proof size: {} bytes ({:.1} KB)", proof_size, proof_size as f64 / 1024.0);
    eprintln!("  written to: {output_path} ({write_ms:.1}ms)");
}

fn cmd_verify(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: stark_cli verify <proof.bin>");
        std::process::exit(1);
    }

    let path = &args[0];
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

fn cmd_bench(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: stark_cli bench <log_n>");
        std::process::exit(1);
    }

    let log_n: u32 = args[0].parse().expect("invalid log_n");
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

// --- Proof serialization (simple binary format) ---
// All values are little-endian u32. Format:
// [magic: u32] [version: u32] [log_n: u32] [a: u32] [b: u32]
// [trace_commitment: 8×u32] [quotient_commitment: 8×u32]
// [n_fri_layers: u32] [fri_commitments: n_fri_layers × 8×u32]
// [n_last_layer: u32] [fri_last_layer: n_last_layer × 4×u32]
// [n_queries: u32] [query_indices: n_queries × u32]
// [trace_decommitment] [quotient_decommitment] [fri_decommitments × n_fri_layers]

const MAGIC: u32 = 0x4B52_414B; // "KRAK"
const VERSION: u32 = 1;

fn write_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}

fn write_u32_slice(w: &mut Vec<u8>, s: &[u32]) {
    for &v in s {
        write_u32(w, v);
    }
}

fn write_decommitment_u32(w: &mut Vec<u8>, d: &QueryDecommitment<u32>) {
    let n = d.values.len() as u32;
    write_u32(w, n);
    write_u32_slice(w, &d.values);
    write_u32_slice(w, &d.sibling_values);
    for path in &d.auth_paths {
        write_u32(w, path.len() as u32);
        for hash in path {
            write_u32_slice(w, hash);
        }
    }
    for path in &d.sibling_auth_paths {
        write_u32(w, path.len() as u32);
        for hash in path {
            write_u32_slice(w, hash);
        }
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
    for d in &proof.fri_decommitments {
        write_decommitment_4(&mut w, d);
    }

    w
}

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
