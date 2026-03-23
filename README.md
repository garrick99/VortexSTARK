# VortexSTARK

GPU-native Circle STARK prover with end-to-end proof generation and verification on NVIDIA Blackwell (RTX 5090) and Ada Lovelace (RTX 4090). Rust + CUDA.

## Status

### End-to-end proven and verified on hardware

- **Fibonacci STARK**: Full prove → verify pipeline, 100-bit conjectured security, 12 tamper-detection tests
- **Cairo VM STARK**: 31-column trace, 31 transition constraints, verifier independently evaluates all constraints at query points
- **LogUp memory consistency**: Execution sum + memory table sum cancellation, final value bound into Fiat-Shamir
- **Pedersen hash**: GPU windowed EC scalar multiplication, 37.7M hashes/sec, verified against CPU reference
- **Poseidon2 hash**: GPU trace generation + NTT, 4.7M hashes/sec at log_n=28 (30 rows/perm, RF=8 RP=22)
- **RPO-M31 hash**: Circle STARK–native hash (eprint 2024/1635), 3.5M hashes/sec at log_n=28 (14 rows/perm, 24 cols)
- **FRI**: Circle fold + line folds, GPU-resident decommitment, all fold equations verified

### Benchmarked (RTX 5090, CUDA 13.2, driver 595.79, 2026-03-22)

| Workload | Scale | Prove | Verify |
|----------|-------|-------|--------|
| Fibonacci log_n=24 | 16.8M elements | 214ms | 6.2ms |
| Fibonacci log_n=28 | 268M elements | 1.56s | 8.2ms |
| Fibonacci log_n=30 | 1.07B elements | 9.79s | 10.7ms |
| Cairo VM log_n=24 | 16.8M steps | 7.24s | 0.4ms |
| Cairo VM log_n=26 | 67.1M steps | 29.2s | 0.5ms |
| Poseidon2 trace+NTT log_n=28 | 8.9M hashes | 1.92s | — |
| RPO-M31 trace+NTT log_n=28 | 19.2M hashes | 5.51s | — |
| Poseidon2-Full trace+NTT log_n=28 | 33.6M hashes [experimental] | 1.92s | — |
| Pedersen GPU batch | 1M hashes | 26.6ms | — |

### Adversarial soundness (constraint coverage)

31-column trace, 31 transition constraints. The following are now enforced:

- **Operand address verification**: dst_addr, op0_addr, op1_addr constrained against register + offset - bias
- **JNZ fall-through**: dst_inv auxiliary column, fall-through constrained to pc + inst_size when dst=0
- **JNZ inverse consistency**: dst * dst_inv = 1 enforced when jnz and dst != 0
- **Op1 source exclusivity**: pairwise products of op1_imm, op1_fp, op1_ap all constrained to zero
- **PC update exclusivity**: pairwise products of jump_abs, jump_rel, jnz constrained to zero
- **Opcode exclusivity**: pairwise products of call, ret, assert constrained to zero
- **Flag binary**: all 15 flags constrained to {0, 1}
- **Instruction decomposition**: all 63 bits verified — inst_lo + inst_hi ≡ off0 + off1·2^16 + off2·2^32 + flags·2^48 (mod P)
- **LogUp memory consistency**: execution sum + table sum cancellation with final value bound into Fiat-Shamir
- **Range check argument**: all 16-bit offsets verified via LogUp against precomputed table, wired into prover with z_rc challenge
- **Merkle domain separation**: internal nodes use Blake2s personalization (h[6] ^= 0x01), preventing leaf/node confusion

### Remaining limitations

- **Hint execution**: Not implemented — limits provable programs to straight-line arithmetic and simple loops
- **Full Starknet programs**: Requires hints, high-address builtin memory segments, and full felt252 support

See [SOUNDNESS.md](SOUNDNESS.md) for the full constraint-by-constraint analysis.

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
- **31-column trace**: registers(3), instruction(2), flags(15), operands(7), offsets(3), dst_inv(1)
- **31 transition constraints** evaluated on GPU (single CUDA kernel) and independently by verifier
- **LogUp memory consistency**: permutation argument with execution + table sum cancellation
- **Range check argument**: 16-bit offset validation via LogUp bus, wired into prover pipeline
- **Instruction decomposition**: algebraic constraint tying inst_lo/inst_hi to offsets and flags

## Builtins

| Builtin | Status | Throughput (log_n=28) |
|---------|--------|----------------------|
| Poseidon2 | GPU kernel, proven (RF=8 RP=22, 30 rows/perm) | 4.7M hashes/sec |
| RPO-M31 | GPU kernel, proven (14 rows/perm, 24 cols) | 3.5M hashes/sec |
| Poseidon2-Full | GPU kernel, **experimental — no security analysis** (8 rows/perm) | 17.5M hashes/sec |
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

Requires: Rust nightly (1.89+), CUDA 13.2+, RTX 5090 (SM 12.0) or RTX 4090 (SM 8.9).

```bash
cargo build --release
cargo test -- --test-threads=1    # 149 tests
cargo run --release --bin full_benchmark
cargo run --release --bin gpu_bench     # pre-flight checks + per-section GPU telemetry
```

## Tests

149 tests covering: M31/CM31/QM31 field arithmetic, Circle NTT, Merkle tree (commit, auth paths, tiled, SoA4), FRI (fold, circle fold, deterministic), STARK prover + verifier (multiple sizes, tamper detection), Cairo VM (decoder, executor, Fibonacci, constraints, LogUp, range checks, instruction decomposition), Poseidon, Pedersen (Stark252 field, EC ops, GPU vs CPU), Bitwise, GPU constraint eval (bytecode VM, warp-cooperative), GPU leaf hashing (Blake2s, domain separation), CASM loader.

## Break This System

If you can craft a malformed trace that the verifier accepts, that is a real bug. Open an issue.

### Known remaining weak points

- **Hint execution**: Not implemented — limits provable programs to straight-line arithmetic and simple loops
- **Full Starknet programs**: Requires hints, high-address builtin memory segments, and full felt252 support

### Expected to hold (guarantees today)

These should **not** break. If they do, that's a real soundness bug:

- Honest prover produces proofs that verify for Fibonacci, add, mul, call/ret, and mixed programs
- Verifier independently evaluates all 31 constraints at query points and rejects any mismatch
- Operand addresses verified against register + offset for all three operands (dst, op0, op1)
- JNZ fall-through constrained: dst_inv auxiliary column forces next_pc = pc + inst_size when dst = 0
- Flag exclusivity enforced: op1 source, PC update mode, and opcode are pairwise mutually exclusive
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
