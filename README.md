# VortexSTARK

GPU-native Circle STARK prover with end-to-end proof generation and verification on NVIDIA Blackwell (RTX 5090) and Ada Lovelace (RTX 4090). Rust + CUDA.

## Status

### End-to-end proven and verified on hardware

- **Fibonacci STARK**: Full prove → verify pipeline, 100-bit conjectured security, 12 tamper-detection tests
- **Cairo VM STARK**: 27-column trace, 20 transition constraints, verifier independently evaluates all constraints at query points
- **LogUp memory consistency**: Execution sum + memory table sum cancellation, final value bound into Fiat-Shamir
- **Pedersen hash**: GPU windowed EC scalar multiplication, 37.7M hashes/sec, verified against CPU reference
- **Poseidon hash**: GPU trace generation, 34.7M hashes/sec
- **FRI**: Circle fold + line folds, GPU-resident decommitment, all fold equations verified

### Benchmarked (measured on RTX 5090, CUDA 13.0, March 2026)

| Workload | Scale | Prove | Verify |
|----------|-------|-------|--------|
| Fibonacci log_n=24 | 16.8M elements | 390ms | 6.4ms |
| Fibonacci log_n=28 | 268M elements | 2.68s | 11.9ms |
| Fibonacci log_n=30 | 1.07B elements | 7.0s | 14.2ms |
| Cairo VM log_n=24 | 16.8M steps, 453M elements | 818ms | — |
| Cairo VM log_n=27 | 134M steps, 3.6B elements | 7.3s | — |
| Poseidon trace gen | 12.2M hashes | 1.4s | — |
| Pedersen GPU batch | 1M hashes | 26.5ms | — |

### Not yet adversarially complete

These are documented gaps in the constraint system. The prover produces correct proofs for honest executions, but a malicious prover could exploit these to forge proofs:

- **JNZ fall-through**: When `dst=0` (conditional not taken), `next_pc` is unconstrained
- **Instruction decomposition**: No constraint linking flag columns to the actual instruction word
- **Operand address constraints**: No constraint verifying dst/op0/op1 addresses match register + offset
- **Flag mutual exclusivity**: No constraint enforcing at most one op1 source, one PC update mode, etc.
- **Range checks**: Implemented and tested, but not yet wired into the prover pipeline
- **Hint execution**: Not implemented — limits provable programs to straight-line arithmetic and simple loops
- **Full Starknet programs**: Requires hints, high-address builtin memory segments, and full felt252 support

Production readiness is self-rated at **50%**. See [SOUNDNESS.md](SOUNDNESS.md) for the full analysis.

## Architecture

- **Full-group NTT** on subgroup(2^31) via ForwardTwiddleCache
- **Fused quotient + circle fold**: zero host transfer for quotient data
- **FRI arena**: MOVE layers into storage, zero clones, GPU-resident decommitment
- **Two-phase commitment**: trace → Fiat-Shamir → LogUp interaction → quotient → FRI
- **GPU LogUp interaction**: batch QM31 inverse + parallel prefix sum (16ms at 16.7M steps)
- **Pinned DMA**: async trace download overlapped with FRI folding at PCIe 5.0 bandwidth

## Cairo VM AIR

- **Instruction decoder**: 15 flags, 3 biased offsets, full Cairo encoding
- **VM executor**: add, mul, jump, jnz, call, ret, assert_eq (26ns/step fused)
- **27-column trace**: registers(3), instruction(2), flags(15), operands(7)
- **20 transition constraints** evaluated on GPU (single CUDA kernel) and independently by verifier
- **LogUp memory consistency**: permutation argument with execution + table sum cancellation
- **Range checks**: 16-bit offset validation via LogUp bus (implemented, wiring in progress)

## Builtins

| Builtin | Status | Throughput |
|---------|--------|-----------|
| Poseidon | GPU kernel, proven | 34.7M hashes/sec |
| Pedersen | GPU kernel, proven (windowed 4-bit EC, Montgomery Jacobian) | 37.7M hashes/sec |
| Bitwise | Trace generation | AND/XOR/OR on 252-bit |

## CLI

```bash
stark_cli prove 24 1 1 -o proof.bin          # Fibonacci STARK
stark_cli prove-file program.casm -o proof.bin # Cairo program
stark_cli prove-starknet --class-hash 0x...    # From Starknet mainnet
stark_cli inspect program.casm                 # Disassemble CASM
stark_cli fetch-block --block 100000           # Starknet block info
stark_cli verify proof.bin                     # Verify a proof
stark_cli bench 28                             # Benchmark
```

## Building

Requires: Rust nightly (1.89+), CUDA 13.0+, RTX 5090 (SM 12.0) or RTX 4090 (SM 8.9).

```bash
cargo build --release
cargo test -- --test-threads=1    # 148 tests
cargo run --release --bin full_benchmark
```

## Tests

148 tests covering: M31/CM31/QM31 field arithmetic, Circle NTT, Merkle tree (commit, auth paths, tiled, SoA4), FRI (fold, circle fold, deterministic), STARK prover + verifier (multiple sizes, tamper detection), Cairo VM (decoder, executor, Fibonacci, constraints, LogUp, range checks), Poseidon, Pedersen (Stark252 field, EC ops, GPU vs CPU), Bitwise, GPU constraint eval (bytecode VM, warp-cooperative), GPU leaf hashing (Blake2s), CASM loader.

## Break This System

If you can craft a malformed trace that the verifier accepts, that is a real bug. Open an issue.

### Known weak points (expected to fail under adversarial input)

These are documented constraint gaps. A sufficiently motivated adversary can likely exploit them to forge proofs for specific program patterns:

- **JNZ fall-through**: Set `pc_jnz=1, dst=0` and `next_pc` to any value — constraint contributes zero
- **Instruction decomposition**: Commit flag columns that don't match the instruction word — no constraint links them
- **Operand addresses**: Commit arbitrary `dst_addr/op0_addr/op1_addr` unrelated to registers + offsets
- **Flag exclusivity**: Set `op1_imm=1, op1_fp=1, op1_ap=1` simultaneously — no constraint prevents it
- **Range checks**: Offsets outside [0, 2^16) are not rejected by the prover (code exists, not wired)

### Expected to hold (guarantees today)

These should **not** break. If they do, that's a real soundness bug:

- Honest prover produces proofs that verify for Fibonacci, add, mul, call/ret, and mixed programs
- Verifier independently evaluates all 20 constraints at query points and rejects any mismatch
- Tampering any committed value (trace, quotient, FRI, commitment) is detected
- LogUp final-value enforcement: corrupting the memory consistency sum breaks FRI verification
- FRI fold equations: algebraic consistency checked at every query across all layers
- Merkle auth paths: data integrity verified for trace, quotient, and all FRI layers
- Fiat-Shamir transcript: any mutation to public inputs, commitments, or challenges cascades into rejection

### Tamper tests (all passing)

| Test | What it tampers | Result |
|------|----------------|--------|
| `test_tamper_flag_binary` | Set flag to non-binary value | REJECTED |
| `test_tamper_result_computation` | Corrupt `res` column | REJECTED |
| `test_tamper_pc_update` | Corrupt `next_pc` | REJECTED |
| `test_tamper_ap_update` | Corrupt `next_ap` | REJECTED |
| `test_tamper_fp_update` | Corrupt `next_fp` | REJECTED |
| `test_tamper_assert_eq` | Corrupt `dst != res` | REJECTED |
| `test_tamper_logup_final_sum` | Corrupt LogUp sum | REJECTED |
| `test_tamper_rc_final_sum` | Corrupt range check sum | REJECTED |
| `test_cairo_tampered_program_hash` | Corrupt program hash | REJECTED |
| `test_cairo_prove_verify_tampered_quotient` | Corrupt quotient value | REJECTED |
| `test_cairo_prove_verify_tampered_fri` | Corrupt FRI value | REJECTED |
| `test_tamper_ec_trace` | Corrupt EC trace commitment | REJECTED |

## License

Business Source License 1.1 ([LICENSE](LICENSE)). Non-production use permitted. Converts to Apache License 2.0 on 2029-03-20. Contact for commercial licensing.
