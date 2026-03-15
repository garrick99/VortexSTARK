# kraken-stark Soundness Status

## Done

### Verifier-side constraint evaluation (Fix #1)
- Verifier independently evaluates all 20 Cairo transition constraints at query points
- Checks constraint_sum == quotient_decommitment_value (exact equality)
- Per-constraint-family tamper tests:
  - test_tamper_flag_binary: f*(1-f)=0 violation → DETECTED
  - test_tamper_result_computation: corrupt res → DETECTED
  - test_tamper_pc_update: corrupt next_pc → DETECTED
  - test_tamper_ap_update: corrupt next_ap → DETECTED
  - test_tamper_fp_update: corrupt next_fp → DETECTED
  - test_tamper_assert_eq: corrupt dst≠res → DETECTED

### LogUp final-value enforcement (Fix #2)
- Prover downloads final LogUp running sum from GPU prefix scan
- Final sum bound into Fiat-Shamir transcript (tampering breaks FRI verification)
- Verifier recomputes expected memory table sum from program (public input)
- Verifier re-executes program to extract memory table for validation
- test_tamper_logup_final_sum: corrupt final sum → DETECTED

### Range check wiring (Fix #3)
- extract_offsets() extracts 3×16-bit offsets per instruction from trace
- Verifier checks all offsets are in [0, 2^16) range
- compute_rc_interaction_trace() computes range check LogUp running sum
- compute_rc_table_sum() computes expected table contribution
- Verifier checks execution sum + table sum == 0 (LogUp cancellation)
- Verifier checks claimed RC final sum matches recomputed value
- test_tamper_rc_final_sum: corrupt RC sum → DETECTED

### Public inputs
- initial_pc, initial_ap, n_steps, program_hash, program bytecode
- All bound into Fiat-Shamir transcript
- test_cairo_tampered_program_hash → DETECTED

### Merkle auth paths
- Generated via cpu_merkle_auth_paths_soa4 (targeted, efficient)
- Verified for quotient commitment + all FRI layers
- test_cairo_prove_verify_tampered_quotient → DETECTED
- test_cairo_prove_verify_tampered_fri → DETECTED

### Multi-program testing
- Fibonacci (add-only): proven + verified
- Multiply-accumulate (mul-only): proven + verified
- Mixed add/mul alternating: proven + verified
- Call/ret initialization pattern: proven + verified

## Documented Assumptions

### Pedersen honest-prover path
- GPU computes correct Pedersen hashes (10K regression test, byte-for-byte)
- Trace columns committed via Merkle tree
- Full EC constraint kernel (stwo-style partial_ec_mul) would require ~500 columns
- Current approach: honest prover computes correctly, commitment binds values

## Confidence Summary

| Component | Confidence |
|-----------|-----------|
| GPU kernels / benchmarks | 95% |
| Fibonacci STARK (prove+verify) | 90% |
| Cairo verifier soundness | 85% |
| Cairo system completeness | 80% |
| Production readiness | 40% |

### What would move production readiness higher
- Real Cairo compiler output (not hand-crafted bytecode)
- Security audit
- Pedersen EC constraint kernel
- Formal verification of constraint polynomials
- Adversarial testing
