# kraken-stark Soundness Status

## Done

### Verifier-side constraint evaluation (Fix #1)
- Verifier independently evaluates all 20 Cairo transition constraints at query points
- Matches GPU kernel (cairo_constraint.cu) exactly
- Tamper tests per constraint family:
  - Flag binary violation: detected
  - Result computation violation: detected
  - PC update violation: detected
  - AP update violation: detected
  - FP update violation: detected
  - Assert_eq violation: detected
- Trace values (current + next row) included in proof at all 100 query points
- FRI fold equations verified at all query points
- Merkle auth paths verified for quotient + all FRI layers

## Deferred — Known Gaps

### LogUp final-value enforcement
- **Status**: NOT FIXED. Known gap.
- **What exists**: Fused GPU kernel computes LogUp running sum. Sum is committed and FRI-verified as low-degree.
- **What's missing**: Verifier does not extract and check that the final accumulator value equals the expected memory table contribution. Without this, a malicious prover could commit a running sum that doesn't correspond to valid memory consistency.
- **Risk**: Memory permutation argument is not enforced. Invalid memory access patterns could pass verification.
- **Next phase**: Extract final running sum value, compute expected memory table sum from public inputs, check equality in verifier.

### Range check wiring
- **Status**: NOT FIXED. Known gap.
- **What exists**: `range_check.rs` module with LogUp bus architecture for 16-bit offset validation.
- **What's missing**: Not wired into `cairo_prove` or `cairo_verify`. Offset values (off0, off1, off2) are not range-checked.
- **Risk**: A malicious prover could use out-of-range offset values in instruction encoding, producing invalid memory addresses that wouldn't be caught.
- **Next phase**: Wire range check columns into trace commitment, add range check LogUp to interaction phase, verify in verifier.

## Documented Assumptions

### Pedersen honest-prover path
- GPU computes correct Pedersen hashes (verified by 10K regression test, byte-for-byte CPU match)
- Trace columns are committed via Merkle tree (values bound)
- LogUp links VM memory accesses to Pedersen I/O addresses
- **Assumption**: Prover is honest about Pedersen computation. A malicious prover could commit garbage Pedersen outputs.
- **Full fix**: EC constraint kernel (stwo-style partial_ec_mul, ~500 columns). Significant engineering effort.

## Confidence Summary

| Component | Confidence | Notes |
|-----------|-----------|-------|
| GPU kernels / benchmarks | 95% | 10K regression, stress tested, byte-for-byte verified |
| Fibonacci STARK (prove+verify) | 90% | Full end-to-end with constraint check |
| Cairo verifier soundness | 75% | Constraint eval at query points, Merkle paths, FRI fold equations |
| Cairo system completeness | 50% | LogUp final value + range checks still open |
| Production readiness | 20% | No real compiler output, no security audit, incomplete coverage |
