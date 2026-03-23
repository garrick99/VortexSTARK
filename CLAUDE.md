# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

VortexSTARK is a GPU-native Circle STARK prover in Rust + CUDA, targeting RTX 5090 (Blackwell SM 12.0) and RTX 4090 (Ada SM 8.9). It implements a full end-to-end STARK proving and verification system with ~100-bit conjectured security. All core proving phases are GPU-resident — no host transfers mid-pipeline.

## Build & Test

```bash
cargo build --release                          # Requires CUDA 13+ on Windows (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0)
cargo test -- --test-threads=1                 # 149 tests; must run single-threaded (GPU contention)
cargo clippy --all-targets --release
```

### Binaries

```bash
# CLI
cargo run --release --bin stark_cli prove 24              # Fibonacci STARK (log_n=24)
cargo run --release --bin stark_cli prove-file prog.casm  # Cairo program
cargo run --release --bin stark_cli verify proof.bin

# Benchmarks
cargo run --release --bin full_benchmark   # Complete system
cargo run --release --bin hash_bench       # Poseidon2 vs RPO-M31 vs Poseidon2-Full
cargo run --release --bin bench            # NTT + constraint benchmarks
cargo run --release --bin rpo_bench        # RPO-M31 vs Poseidon2 comparison
```

## Architecture

### Proving Pipeline

```
Input (Fibonacci or Cairo CASM)
  → [GPU] Trace generation & upload
  → [GPU] Circle NTT interpolation
  → [GPU] Blowup domain evaluation
  → [GPU] Merkle commit (Blake2s, SoA4 layout)
  → [GPU] Constraint evaluation + quotient (stays GPU-resident)
  → [GPU] FRI: circle fold → line folds
  → [CPU] Package proof → StarkProof (serializable)
  → [CPU] Verify: Fiat-Shamir replay, Merkle checks, fold equations
```

### Field Tower

- **M31** — 2³¹−1 prime field, base layer
- **CM31** — Complex extension a + b·√−1, used for Circle group arithmetic
- **QM31** — Quadratic extension (a + b·i + c·α + d·β), used for FRI security

### STARK Flavors

**Fibonacci STARK** (`src/prover.rs`, `src/verifier.rs`): 1-column trace, single transition constraint. Blowup factor 2×, 100 FRI queries.

**Cairo VM STARK** (`src/cairo_air/`): 31-column trace with full VM execution semantics — decode, register updates, memory consistency (LogUp), range checks. Supports: add, mul, jump, jnz, call, ret, assert_eq. No hint execution (limits to straight-line Cairo code).

### Hash Functions in the Trace

Three hash functions can generate the STARK trace:

| Hash | Rows/perm | Columns | Notes |
|------|-----------|---------|-------|
| Poseidon2 | 30 | 8 | RF=8, RP=22; stable |
| RPO-M31 | 14 | 24 | Circle STARK–native (eprint 2024/1635); FM/BM structure |
| Poseidon2-Full | 8 | — | RF=8, no partial rounds; **experimental, not security-analyzed** |

Pedersen EC hash (`src/cairo_air/pedersen.rs`, `cuda/pedersen_gpu.cu`) is used for Cairo builtin hashing — windowed 4-bit scalar mul, 37.7M hash/s on RTX 5090.

### CUDA Layer

- **Build**: `build.rs` detects CUDA toolkit, compiles 23 `.cu` files, links as a static lib (`vortex_cuda.lib` / `libvortex_cuda.a`). Adds SM 12.0 target if CUDA 13+ detected.
- **FFI bridge**: `src/cuda/ffi.rs` — raw bindings to CUDA runtime + all kernel entry points.
- **Key kernels**: `circle_ntt.cu`, `blake2s.cu`, `quotient.cu`, `fri.cu`, `cairo_constraint.cu`, `logup_interaction.cu`, `pedersen_gpu.cu`, `poseidon_trace.cu`, `rpo_trace.cu`.

### Merkle Tree

SoA4 layout: 4 trees striped for cache efficiency. Domain separation between leaf and internal nodes via `h[6]` personalization. Auth paths generated CPU-side (`cpu_merkle_auth_paths_soa4`).

### VRAM Management

- Async allocation pool with 2 GB release threshold (`src/cuda/ffi.rs`: `init_memory_pool`)
- Preflight check via `nvidia-smi` before CUDA init (`vram_preflight_check`)
- WSL2 detected at startup; falls back to synchronous allocation if needed

## Key Source Locations

| Path | Role |
|------|------|
| `src/prover.rs` | Fibonacci STARK prove pipeline |
| `src/verifier.rs` | Fibonacci + generic verifier |
| `src/cairo_air/` | Cairo VM: decoder, executor, AIR constraints, builtins |
| `src/cairo_air/prover.rs` | Cairo STARK prove |
| `src/field/` | M31, CM31, QM31 arithmetic |
| `src/ntt.rs` | Circle NTT with twiddle caching |
| `src/fri.rs` | FRI fold, query selection, decommit |
| `src/merkle.rs` | Merkle tree (SoA4, domain separation) |
| `src/channel.rs` | Fiat-Shamir transcript |
| `src/device/buffer.rs` | `DeviceBuffer<T>` — GPU memory RAII wrapper |
| `src/cuda/ffi.rs` | All CUDA FFI bindings and helpers |
| `src/poseidon.rs` / `src/rpo_m31.rs` | CPU-side hash logic + GPU trace dispatch |
| `src/poseidon2f.rs` | Poseidon2-Full (experimental) |
| `src/air.rs` | Abstract AIR constraint interface |
| `build.rs` | CUDA toolkit detection + kernel compilation |
| `SOUNDNESS.md` | Constraint-by-constraint security analysis + tamper test documentation |
| `BENCHMARKS.md` | Performance history and tuning notes |

## Notes

- GreenDragon (192.168.50.239, RTX 5090) runs driver 591.86 — update to 595.79 to fix OOM at log_n=30.
- Poseidon2-Full (`src/poseidon2f.rs`, `cuda/poseidon2f_trace.cu`) has no formal security analysis — research only.
- Cairo hint execution is not implemented; proofs are limited to straight-line CASM programs.
