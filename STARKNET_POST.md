# VortexSTARK: GPU-Native Circle STARK Prover — 7.0s for 1 Billion Elements on a Single RTX 5090

## TL;DR

I built a GPU-native Circle STARK prover that generates verified proofs for 1 billion trace elements in **7.0 seconds** on a single NVIDIA RTX 5090. The field stack (M31/CM31/QM31) and Circle STARK protocol match stwo exactly. Everything runs on GPU — NTT, FRI, Merkle commitments, constraint evaluation, grinding — zero CPU fallbacks in the proving pipeline.

The stwo `CudaBackend` integration is **complete** — the full stwo-cairo AIR (all 67 Cairo components) runs through GPU-accelerated constraint evaluation. Every test passes, every proof verifies.

## Benchmarks

### Fibonacci STARK (standalone prover)

Single RTX 5090 (Blackwell, SM_120), Fibonacci AIR, 100-bit security:

| Trace Size | Elements | Proving Time | Proof Size |
|---|---|---|---|
| 2^20 | 1M | 116ms | 1.9 KB |
| 2^22 | 4M | 143ms | 1.9 KB |
| 2^24 | 16M | 217ms | 2.3 KB |
| 2^26 | 67M | 471ms | 2.7 KB |
| 2^28 | 268M | 1.5s | 3.1 KB |
| 2^30 | 1.07B | **7.0s** | 3.4 KB |

Steady-state throughput: **~8.6 proofs/minute** at 1B elements (amortized). All proofs verified.

### stwo-cairo Full Proving Pipeline (GPU CudaBackend)

Full end-to-end proving with `prove_cairo()` — VM execution, trace generation, GPU STARK proving, verification, and serialization. Single RTX 5090:

| Cairo Program | Total Time | STARK Proving | Verified |
|---|---|---|---|
| All opcodes (67 components) | 42s | **9.4s** | yes |
| All builtins (Pedersen, Poseidon, Bitwise, Range, Blake, EC) | 57s | **21.1s** | yes |
| Pedersen aggregator | 48s | **14.3s** | yes |
| Poseidon aggregator | 40s | **7.8s** | yes |

**What the columns mean:**
- **Total Time** — wall clock from program start to proof serialized, including Cairo VM execution (CPU), trace generation (CPU), STARK proving (GPU), verification, and proof output
- **STARK Proving** — time spent in the GPU proving pipeline only (`prove_ex`): NTT, constraint evaluation, FRI protocol, Merkle commitments, grinding

The GPU STARK proving is **2-5x faster** than the trace generation phase. As trace sizes scale up (millions of Cairo steps for real Starknet blocks), the GPU proving advantage widens because GPU kernels scale with trace size while per-component overhead stays constant.

### Phase Breakdown (all-builtins proof, 21.1s GPU proving)

| Phase | Time | Where |
|---|---|---|
| Constraint evaluation | ~8s | GPU bytecode interpreter (67 components) |
| FRI protocol | ~5s | GPU fold + commit |
| FRI quotients | ~2.4s | GPU accumulate + combine |
| Merkle commitments | ~3s | GPU Blake2s |
| Decommitment | ~2s | GPU tile trees + auth paths |
| Grinding | 1.3ms | GPU Blake2s PoW (26-bit) |

## What's Implemented

**stwo CudaBackend (4,200+ lines, 50+ tests):**
- `PolyOps` — GPU NTT evaluate/interpolate with cached twiddles
- `FriOps` — GPU circle-to-line and line folding
- `MerkleOpsLifted` — GPU Blake2s leaf hashing and tree construction
- `GrindOps` — GPU Blake2s proof-of-work (1ms for 26-bit)
- `QuotientOps` — GPU DEEP quotient accumulation
- `ComponentProver` — GPU bytecode constraint evaluator (per-thread + warp-cooperative kernels)
- `ColumnOps` — GPU column types with bit-reverse, backed by DeviceBuffer

**GPU Kernels (all CUDA, SM_120 optimized):**
- Circle NTT — stwo-compatible twiddle format, fused shared-memory tiling, radix-4
- FRI — fold_line, fold_circle_into_line, on-demand twiddle computation
- Merkle trees — GPU Blake2s (leaves, nodes, tiled subtree roots)
- Constraint evaluation — two kernels: per-thread (≤1024 registers) and warp-cooperative via `__shfl_sync` (>1024 registers, distributes register file across 32 lanes)
- FRI quotient accumulation with batch column evaluation
- Barycentric polynomial evaluation
- Batch modular inverse, bit-reverse permutation

**Architecture:**
- Field stack: M31, CM31, QM31 — identical to stwo
- Pinned DMA transfers at full PCIe 5.0 bandwidth (37 GB/s measured)
- GPU-resident FRI layer decommitment (tile-based, ~3MB vs 16GB)
- Lazy pinned buffer pool for amortized allocation across proofs

**Also built during this project:**
- **OpenPTXas** — open-source PTX-to-CUBIN assembler for SM_120, with automated scoreboard emulation and a fix for a ptxas miscompilation bug affecting every NVIDIA GPU since 2014 (SM_50 through SM_120)
- **OpenCUDA** — CUDA-subset C compiler targeting PTX/SM_120, 54 tests passing, GPU-verified on RTX 5090

## Why This Matters for Starknet

Starknet's production prover (stwo) runs on CPU. When proving decentralizes, GPU provers will be critical for competitive proving economics. VortexSTARK demonstrates that the stwo Circle STARK protocol runs entirely on GPU with significant speedups — and the CudaBackend integration means it plugs directly into the existing stwo-cairo proving pipeline.

Nethermind's stwo-gpu effort (189 commits, no published benchmarks or releases) is the only other GPU attempt. VortexSTARK is shipping with benchmarks, a complete Backend implementation, and a passing test suite.

## About

Built over several months of GPU architecture research, kernel development, and low-level NVIDIA reverse engineering. All benchmarks run on a single RTX 5090. The GPU kernels are hand-optimized for Blackwell (SM_120) with knowledge gained from building the assembler and compiler toolchain from scratch.

I'm looking for:
- Feedback from the Starknet/stwo community on priorities
- Starknet Foundation grant support for large-scale proving infrastructure
- Partnerships with proving marketplace providers

Happy to share more technical details, answer questions, or run benchmarks on specific workloads.
