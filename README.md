# VortexSTARK

GPU-native Circle STARK prover. Rust + CUDA. RTX 5090.

## Performance

```
FIBONACCI STARK (proven + verified, 100-bit security)
  log_n=24 │   16.8M elements │    390ms prove │  6.4ms verify ✓
  log_n=28 │    268M elements │   2.68s prove  │ 11.9ms verify ✓
  log_n=29 │    537M elements │   4.39s prove  │  9.5ms verify ✓
  log_n=30 │  1.07B elements  │   7.0s  prove  │ 14.2ms verify ✓ (amortized)

POSEIDON (8 columns, x^5 S-box, 22 full rounds, MDS)
  log_n=28 │  12.2M hashes │  1.4s  (34.7M hashes/sec GPU trace gen)

CAIRO VM (27 columns, 20 constraints, LogUp memory consistency)
  log_n=24 │  16.8M steps │  453M elements │   818ms (sub-second!)
  log_n=26 │    67M steps │  1.8B elements │   3.5s
  log_n=27 │   134M steps │  3.6B elements │   7.3s (per-column VRAM streaming)
```

## Architecture

- **Full-group NTT** on subgroup(2^31) via ForwardTwiddleCache
- **Fused quotient + circle fold**: zero host transfer for quotient data
- **FRI arena**: MOVE layers into storage, zero clones, GPU-resident decommitment
- **Two-phase commitment**: trace → Fiat-Shamir → LogUp interaction → quotient → FRI
- **GPU LogUp interaction**: batch QM31 inverse + parallel prefix sum (16ms at 16.7M steps)

## Cairo VM AIR

Complete core AIR for the Cairo instruction set:

- **Instruction decoder**: 15 flags, 3 biased offsets, full Cairo encoding
- **VM executor**: add, mul, jump, jnz, call, ret, assert_eq (26ns/step fused)
- **27-column trace**: registers(3), instruction(2), flags(15), operands(7)
- **20 transition constraints** in a single CUDA kernel
- **LogUp memory consistency**: permutation argument proving all reads are valid
- **Range checks**: 16-bit offset validation via LogUp bus

## Builtins

| Builtin | Status | Throughput |
|---------|--------|-----------|
| Poseidon | GPU kernel, proven | 34.7M hashes/sec |
| Pedersen | GPU kernel, proven (windowed 4-bit EC, Montgomery Jacobian) | 37.7M hashes/sec |
| Bitwise | Trace + constraints | AND/XOR/OR on 252-bit |

## Stark252 Field

Full modular arithmetic over the STARK prime (p = 2^251 + 17·2^192 + 1):

- Add, sub, mul, neg, inverse (Fermat)
- Elliptic curve: point add, double, scalar mul (projective Jacobian)
- Pedersen hash: H(a,b) = [P₀ + a_low·P₁ + a_high·P₂ + b_low·P₃ + b_high·P₄]_x
- 30 field/EC tests passing

## CLI

```bash
# Prove Fibonacci STARK
stark_cli prove 24 1 1 -o proof.bin

# Prove a CASM file (auto-detects steps)
stark_cli prove-file program.casm -o proof.bin

# Prove a contract from Starknet mainnet
stark_cli prove-starknet --class-hash 0x029927c8af6b...

# Inspect/disassemble a CASM file
stark_cli inspect program.casm

# Fetch Starknet block info
stark_cli fetch-block --block 100000

# Verify
stark_cli verify proof.bin

# Benchmark
stark_cli bench 28
```

Accepts CASM JSON (Cairo 1 compiled) and Cairo 0 compiled JSON formats.
Starknet RPC integration fetches contract CASM directly from mainnet/testnet.

## Building

Requires: Rust 1.94+, CUDA 13.0, RTX 5090 (SM 12.0) or RTX 4090 (SM 8.9).

```bash
cargo build --release
cargo test -- --test-threads=1    # 148 tests
cargo run --release --bin full_benchmark
```

## Tests

148 tests covering:
- M31/CM31/QM31 field arithmetic
- Circle NTT (forward, inverse, roundtrip, batched)
- Merkle tree (commit, auth paths, tiled, SoA4)
- FRI (fold, circle fold, deterministic)
- STARK prover + verifier (multiple sizes, tamper detection)
- Cairo VM (decoder, executor, Fibonacci, constraints, LogUp, range checks)
- Poseidon (permutation, trace, GPU trace match)
- Pedersen (Stark252 field, EC ops, hash, GPU vs CPU correctness)
- Bitwise (AND/XOR/OR, trace)
- GPU constraint eval (bytecode VM, warp-cooperative kernels)
- GPU leaf hashing (Blake2s, buffered chunks)
- CASM loader (hex parsing, step detection, disassembly)

## License

Private. Contact for licensing inquiries.
