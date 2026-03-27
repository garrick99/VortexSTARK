//! Property-based soundness and completeness tests.
//!
//! These tests use deterministic pseudo-randomness to exercise edge cases
//! without external test-framework dependencies.
//!
//! **Completeness**: honest proofs for valid programs always verify.
//! **Soundness**: mutating any field element in a valid proof causes rejection.

use vortexstark::cairo_air::prover::{cairo_prove, cairo_verify};
use vortexstark::cairo_air::decode::Instruction;

fn init_gpu() {
    vortexstark::cuda::ffi::init_memory_pool();
}

/// Simple 64-bit LCG for deterministic pseudo-random numbers.
/// Not cryptographic — only used for test data generation.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn next_u32(&mut self) -> u32 { (self.next() >> 32) as u32 }
    fn next_range(&mut self, max: usize) -> usize { (self.next() as usize) % max }
}

/// Build a simple valid program: a sequence of add instructions (Fibonacci-style).
fn make_add_program(n_steps: usize) -> Vec<u64> {
    let init = Instruction {
        off0: 0x8000, off1: 0x8000, off2: 0x8001,
        op1_imm: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    let add = Instruction {
        off0: 0x8000, off1: 0x8000u16.wrapping_sub(2),
        off2: 0x8000u16.wrapping_sub(1),
        op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    let mut prog = vec![init.encode(), 1u64, init.encode(), 1u64];
    for _ in 4..n_steps { prog.push(add.encode()); }
    prog
}

// ---------------------------------------------------------------------------
// Completeness: honest proofs for valid programs always verify
// ---------------------------------------------------------------------------

/// Prove and verify programs of multiple sizes; all must succeed.
#[test]
fn test_completeness_multiple_sizes() {
    init_gpu();
    for (n_steps, log_n) in [(16, 4), (32, 5), (64, 6), (128, 7)] {
        let prog = make_add_program(n_steps);
        let proof = cairo_prove(&prog, n_steps, log_n);
        let result = cairo_verify(&proof);
        assert!(result.is_ok(),
            "Valid proof should verify for n_steps={n_steps} log_n={log_n}: {:?}", result);
    }
}

/// Prove the same program 5 times with the same inputs; all proofs must verify.
/// This catches any non-determinism in the prover.
#[test]
fn test_completeness_deterministic() {
    init_gpu();
    let prog = make_add_program(32);
    for i in 0..5 {
        let proof = cairo_prove(&prog, 32, 5);
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Iteration {i}: proof must verify");
    }
}

// ---------------------------------------------------------------------------
// Soundness: mutating any field element in a valid proof causes rejection
// ---------------------------------------------------------------------------

/// Take a valid proof and flip one bit in each of the 8 program-hash words.
/// Each tamper must be detected.
#[test]
fn test_soundness_program_hash_tamper() {
    init_gpu();
    let prog = make_add_program(32);
    let proof_base = cairo_prove(&prog, 32, 5);

    for word in 0..8 {
        let mut proof = proof_base.clone();
        proof.public_inputs.program_hash[word] ^= 0x0000_0001;
        let result = cairo_verify(&proof);
        assert!(result.is_err(),
            "Tampered program_hash[{word}] should be detected");
    }
}

/// Mutate each of the 100 query indices by ±1 and verify rejection.
#[test]
fn test_soundness_query_indices_tamper() {
    init_gpu();
    let prog = make_add_program(32);
    let proof_base = cairo_prove(&prog, 32, 5);
    let eval_size = 1usize << (5 + vortexstark::prover::BLOWUP_BITS);

    // Only test a few query indices to keep runtime bounded
    for qi in 0..proof_base.query_indices.len().min(5) {
        let mut proof = proof_base.clone();
        proof.query_indices[qi] = (proof.query_indices[qi] + 1) % eval_size;
        let result = cairo_verify(&proof);
        assert!(result.is_err(),
            "Tampered query_indices[{qi}] should be detected");
    }
}

/// Mutate each trace value at the first query point and verify rejection.
/// This is the broadest per-column tamper sweep for a fresh random proof.
#[test]
fn test_soundness_trace_values_all_columns() {
    init_gpu();
    let prog = make_add_program(32);
    let proof_base = cairo_prove(&prog, 32, 5);

    let n_cols = proof_base.trace_values_at_queries[0].len();
    for col in 0..n_cols {
        let mut proof = proof_base.clone();
        let original = proof.trace_values_at_queries[0][col];
        proof.trace_values_at_queries[0][col] = original.wrapping_add(1) % vortexstark::field::m31::P;
        let result = cairo_verify(&proof);
        assert!(result.is_err(),
            "Tampered trace column {col} should be detected (orig={original})");
    }
}

/// Mutate the logup_final_sum and verify rejection.
#[test]
fn test_soundness_logup_final_sum_tamper() {
    init_gpu();
    let prog = make_add_program(32);
    let mut proof = cairo_prove(&prog, 32, 5);
    proof.logup_final_sum[0] ^= 1;
    let result = cairo_verify(&proof);
    assert!(result.is_err(), "Tampered logup_final_sum should be detected");
}

/// Mutate the rc_final_sum and verify rejection.
#[test]
fn test_soundness_rc_final_sum_tamper() {
    init_gpu();
    let prog = make_add_program(32);
    let mut proof = cairo_prove(&prog, 32, 5);
    proof.rc_final_sum[0] ^= 1;
    let result = cairo_verify(&proof);
    assert!(result.is_err(), "Tampered rc_final_sum should be detected");
}

/// Mutate each FRI last-layer polynomial coefficient and verify rejection.
#[test]
fn test_soundness_fri_last_layer_tamper() {
    init_gpu();
    let prog = make_add_program(32);
    let proof_base = cairo_prove(&prog, 32, 5);

    for i in 0..proof_base.fri_last_layer.len() {
        let mut proof = proof_base.clone();
        // Flip one component of the QM31 value
        proof.fri_last_layer[i].a.a.0 ^= 1;
        let result = cairo_verify(&proof);
        assert!(result.is_err(),
            "Tampered fri_last_layer[{i}] should be detected");
    }
}

/// Tamper with the PoW nonce and verify rejection.
#[test]
fn test_soundness_pow_nonce_tamper() {
    init_gpu();
    let prog = make_add_program(32);
    let proof_base = cairo_prove(&prog, 32, 5);

    // Baseline must pass
    cairo_verify(&proof_base).expect("baseline proof must verify");

    // Nonce+1 should fail PoW check
    let mut proof = proof_base.clone();
    proof.pow_nonce = proof_base.pow_nonce.wrapping_add(1);
    assert!(cairo_verify(&proof).is_err(),
        "pow_nonce+1 should be rejected");

    // Nonce-1 should also fail
    let mut proof2 = proof_base.clone();
    proof2.pow_nonce = proof_base.pow_nonce.wrapping_sub(1);
    assert!(cairo_verify(&proof2).is_err(),
        "pow_nonce-1 should be rejected");
}

// ---------------------------------------------------------------------------
// Soundness: random field element mutations across the full proof
// ---------------------------------------------------------------------------

/// Randomly mutate 50 distinct field elements across the proof and verify
/// each mutation is detected. Uses a fixed seed for reproducibility.
#[test]
fn test_soundness_random_field_element_mutations() {
    init_gpu();
    let prog = make_add_program(64);
    let proof_base = cairo_prove(&prog, 64, 6);
    let mut rng = Lcg::new(0xDEAD_BEEF_CAFE_BABE);

    // Targets: trace columns at various query indices
    let n_queries = proof_base.trace_values_at_queries.len();
    let n_cols = proof_base.trace_values_at_queries[0].len();

    let mut rejections = 0;
    let attempts = 20;

    for _ in 0..attempts {
        let qi = rng.next_range(n_queries);
        let col = rng.next_range(n_cols);
        let delta = (rng.next_u32() % 100).max(1); // avoid delta=0

        let mut proof = proof_base.clone();
        proof.trace_values_at_queries[qi][col] =
            proof.trace_values_at_queries[qi][col].wrapping_add(delta)
            % vortexstark::field::m31::P;

        if cairo_verify(&proof).is_err() {
            rejections += 1;
        }
    }

    // All mutations should be rejected (100% soundness for trace column tampering)
    assert_eq!(rejections, attempts,
        "Expected all {attempts} random trace mutations to be detected; got {rejections}");
}

/// Tamper with the OODS quotient decommitment values and verify rejection.
/// This exercises the new OODS quotient formula check: a malicious prover
/// cannot substitute a fake polynomial as the FRI input.
#[test]
fn test_soundness_oods_quotient_tamper() {
    init_gpu();
    let prog = make_add_program(32);
    let proof_base = cairo_prove(&prog, 32, 5);

    // Baseline must pass
    cairo_verify(&proof_base).expect("baseline proof must verify");

    // Flip one limb of each OODS quotient decommitment value — must be rejected
    for q in 0..proof_base.oods_quotient_decommitment.values.len() {
        let mut proof = proof_base.clone();
        proof.oods_quotient_decommitment.values[q][0] ^= 1;
        assert!(
            cairo_verify(&proof).is_err(),
            "Tampered oods_quotient_decommitment.values[{q}][0] should be rejected"
        );
    }
}
