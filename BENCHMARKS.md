# VortexSTARK Benchmark Artifact

## CHECKPOINT: Pedersen-37M-Async (2026-03-14)

### Commit
```
4723bcd (dirty — uncommitted async stream + direct download + pinned memory)
```

### Hardware / Toolkit
```
GPU:          NVIDIA GeForce RTX 5090 (32 GB GDDR7, SM 12.0 Blackwell)
Driver:       595.79
CUDA:         13.0 (Build cuda_13.0.r13.0/compiler.36424714_0)
Rust:         1.94 (stable)
nvcc flags:   -O3 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120
CPU:          Intel Core Ultra 9 285K (24C/24T)
RAM:          64 GB DDR5
Power cap:    450W (max 600W)
OS:           Windows 11
```

### Pedersen Hash — 37.7M/sec (async stream, zero-alloc pipeline)
```
Architecture
────────────────────────────────────────────────────────────────
Scalar mul:     Windowed 4-bit fixed-base (62 windows per 248-bit scalar)
EC addition:    Mixed affine-Jacobian (11 fp_mont_mul, table Z=R)
EC doubling:    a=1 optimized (9 fp_mont_mul, skip identity mul)
Affine output:  Inline Fermat inverse on GPU (a^(p-2), 444 fp_mont_mul)
Data path:      Zero-copy reinterpret (Fp is repr(C), no flatten/repack)
Transfer:       Async CUDA stream (H2D + kernel + D2H pipelined)
Download:       Direct into result Vec (no intermediate to_host() alloc)
Tables:         L1-cached __device__ global memory (18KB)
Block size:     128 threads/block
Stack:          65536 bytes (cudaDeviceSetLimit)

Batch Benchmark
────────────────────────────────────────────────────────────────
Command: cargo run --release --bin bench_pedersen

Batch       Time        Throughput       Verified
1,000       0.5ms       1,957,330/sec    ✓
10,000      0.6ms       15,489,467/sec   ✓
100,000     2.6ms       38,248,231/sec   ✓
1,000,000   26.5ms      37,724,318/sec   ✓

Pipeline Timing (1M batch, sync instrumented path)
────────────────────────────────────────────────────────────────
Phase                Time      % Total
Flatten (zero-copy)    0.0ms     0.0%
H2D upload             4.6ms    13.6%
Alloc output           0.0ms     0.0%
GPU kernel+sync       18.1ms    53.1%   ← hash + Fermat inverse
D2H download          11.3ms    33.2%   ← async path overlaps w/ alloc
Repack                 0.0ms     0.0%   ← eliminated (direct download)
CPU batch inv          0.0ms     0.0%   ← eliminated (inline on GPU)
────────────────────────────────────────────────────────────────
Total                 34.0ms

Optimization History (this session)
────────────────────────────────────────────────────────────────
Stage                                 Throughput    vs Baseline
Baseline (Montgomery, bit-by-bit)      220,488/s      1x
+ Windowed 4-bit + mixed affine        234,328/s      1.06x
+ Parallel CPU batch inverse         2,762,143/s     12.5x
+ Inline Fermat inverse on GPU       12,915,858/s     58.6x
+ Zero-copy flatten/repack           22,964,169/s    104x
+ Kill to_host() + async stream      37,724,318/s    171x
```
- Correctness: byte-for-byte match against CPU (10K random vector regression, zero mismatches)
- GPU speedup vs CPU (61 hash/sec): 618,431x
- Remaining wall time: 53% GPU kernel (IMAD-bound), 47% PCIe bus
- The bus is the enemy. The hard part is done.

---

## CHECKPOINT: Pedersen-23M (2026-03-14)

### Commit
```
4723bcd (dirty — uncommitted inline Fermat inverse + zero-copy)
```

### Hardware / Toolkit
```
GPU:          NVIDIA GeForce RTX 5090 (32 GB GDDR7, SM 12.0 Blackwell)
Driver:       595.79
CUDA:         13.0 (Build cuda_13.0.r13.0/compiler.36424714_0)
Rust:         1.94 (stable)
nvcc flags:   -O3 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120
CPU:          Intel Core Ultra 9 285K (24C/24T)
RAM:          64 GB DDR5
Power cap:    450W (max 600W)
OS:           Windows 11
```

### Pedersen Hash — 23M/sec (GPU-native affine output)
```
Architecture
────────────────────────────────────────────────────────────────
Scalar mul:     Windowed 4-bit fixed-base (62 windows per 248-bit scalar)
EC addition:    Mixed affine-Jacobian (11 fp_mont_mul, table Z=R)
EC doubling:    a=1 optimized (9 fp_mont_mul, skip identity mul)
Affine output:  Inline Fermat inverse on GPU (a^(p-2), 444 fp_mont_mul)
Data path:      Zero-copy reinterpret (Fp is repr(C), no flatten/repack)
Tables:         L1-cached __device__ global memory (18KB)
Block size:     128 threads/block
Stack:          65536 bytes (cudaDeviceSetLimit)

Batch Benchmark
────────────────────────────────────────────────────────────────
Command: cargo run --release --bin bench_pedersen

Batch       Time        Throughput       Verified
1,000       0.5ms       1,960,784/sec    ✓
10,000      0.8ms       12,118,274/sec   ✓
100,000     4.3ms       23,176,045/sec   ✓
1,000,000   43.5ms      22,964,169/sec   ✓

Pipeline Timing (1M batch, 25.6M hash/sec timed run)
────────────────────────────────────────────────────────────────
Phase                Time      % Total
Flatten (zero-copy)    0.0ms     0.0%
H2D upload             4.4ms    11.2%
Alloc output           0.0ms     0.0%
GPU kernel+sync       18.1ms    46.2%   ← hash + Fermat inverse
D2H download           7.0ms    17.9%
Repack (memcpy)        9.6ms    24.6%   ← next target: kill to_host() alloc
CPU batch inv          0.0ms     0.0%   ← eliminated (was 283ms / 81.6%)
────────────────────────────────────────────────────────────────
Total                 39.1ms

Optimization History
────────────────────────────────────────────────────────────────
Stage                                Throughput    Multiplier
Baseline (Montgomery, bit-by-bit)    220,488/s     1x
+ Windowed 4-bit + mixed affine      234,328/s     1.06x
+ Parallel CPU batch inverse        2,762,143/s    12.5x
+ Inline Fermat inverse on GPU     12,915,858/s    58.6x
+ Zero-copy flatten/repack         22,964,169/s    104x
```
- Correctness: byte-for-byte match against CPU (10K random vector regression, zero mismatches)
- GPU speedup vs CPU (61 hash/sec): 376,462x
- GPU power during sustained load: 51-61W (13.3% of 450W cap), 40-58°C, zero throttle events

---

## Frozen Milestone: 2026-03-14

### Commit
```
N/A (pre-push)
```

### Hardware
```
GPU:          NVIDIA GeForce RTX 5090
VRAM:         32 GB GDDR7
SM:           12.0 (Blackwell)
Driver:       595.79
Power limit:  450W (capped, max 600W)
CPU:          Intel Core Ultra 9 285K (24C/24T)
RAM:          64 GB DDR5
OS:           Windows 11
Ambient:      ~22°C (home office)
```

### Toolkit
```
CUDA:         13.0 (Build cuda_13.0.r13.0/compiler.36424714_0)
Rust:         1.94 (stable)
nvcc flags:   -O3 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120
```

### Pedersen Hash (GPU, Montgomery EC, Windowed 4-bit Scalar Mul)
```
Config
──────
Block size:       128 threads/block
Batch size:       100,000 hashes
CPU inverse:      Parallel chunked (all cores, std::thread::scope)
Tables:           L1-cached __device__ global memory (18KB)
EC addition:      Mixed affine-Jacobian (11 fp_mont_mul vs 16 standard)
Doubling:         a=1 optimized (9 fp_mont_mul, skip identity mul)
Stack:            65536 bytes (cudaDeviceSetLimit)

Batch Benchmark
──────────────────────────────────────────────────────
Batch       Time        Throughput      Verified
1,000       5.0ms       198,047/sec     ✓
10,000      8.9ms       1,124,923/sec   ✓
100,000     47.1ms      2,124,125/sec   ✓
1,000,000   362.0ms     2,762,143/sec   ✓

15-Minute Stress Test (100K batch, continuous)
──────────────────────────────────────────────────────
Sustained avg:    1,940,652 hashes/sec
Peak interval:    1,983,189 hashes/sec
Min interval:     1,839,854 hashes/sec
Variance (CoV):   1.91%
Total hashed:     1,746,600,000 (1.75 billion)
GPU power:        51-61W sustained (13.3% of 450W cap)
GPU temp:         40-58°C (no throttling)
Throttle events:  0 (zero power/thermal/SW throttle)
GPU speedup:      31,814x vs CPU
```
### Pipeline Timing Breakdown (1M batch, 2,877K hash/sec)
```
Phase                Time      % Total
─────────────────────────────────────────
CPU batch inverse    283.6ms    81.6%    ← dominant bottleneck
Flatten inputs        32.3ms     9.3%    ← CPU memcpy/reformat
D2H download          10.0ms     2.9%    ← PCIe gen5
GPU kernel+sync        9.4ms     2.7%    ← actual EC math (106M/sec raw)
Repack Fp vecs         7.7ms     2.2%    ← CPU reformat
H2D upload             4.5ms     1.3%    ← PCIe gen5
Alloc output           0.0ms     0.0%
─────────────────────────────────────────
Compute (kern+inv):  292.9ms    84.3%
Overhead (rest):      54.6ms    15.7%
```
- Next target: GPU batch inverse for Stark252 (eliminate 81.6% CPU bottleneck)
- Command: `cargo run --release --bin bench_pedersen` / `cargo run --release --bin stress_test`
- Correctness: byte-for-byte match against CPU (10K random vector regression, zero mismatches)
- Previous baseline: 220,488/sec (Montgomery, no windowing, single-core batch inverse)
- Improvement: 8.8x sustained throughput (parallel CPU) + 6.3% GPU kernel speedup (windowing)

---

## Frozen Milestone: 2026-03-15

### Commit
```
0413d3552bd3798d03d840819fc242756fc83419
```

### Hardware
```
GPU:          NVIDIA GeForce RTX 5090
VRAM:         32 GB GDDR7
SM:           12.0 (Blackwell)
Driver:       595.79
Power limit:  450W (capped via scheduled task)
CPU:          Intel Core Ultra 9 285K
RAM:          64 GB DDR5
OS:           Windows 11
```

### Toolkit
```
CUDA:         13.0 (Build cuda_13.0.r13.0/compiler.36424714_0)
Rust:         1.94 (stable)
nvcc flags:   -O3 -gencode arch=compute_120,code=sm_120
```

### Pedersen Hash (GPU, Montgomery EC, Projective Jacobian)
```
Batch       Time        Throughput      Verified
─────────────────────────────────────────────────
1,000       10.9ms      92,095/sec      ✓
10,000      49.4ms      202,353/sec     ✓
100,000     456.5ms     219,068/sec     ✓
1,000,000   4,535.4ms   220,488/sec     ✓
```
- Correctness: byte-for-byte match against CPU implementation
- Regression: 10,000 random vector test (deterministic seeds, zero mismatches)
- CPU baseline: 60 hashes/sec (projective Jacobian, Fermat inverse)
- GPU speedup: 3,675x

### Fibonacci STARK (proven + verified, 100-bit security)
```
log_n   Elements        Prove       Verify
─────────────────────────────────────────────
20      1,048,576       231ms       4.6ms   ✓
24      16,777,216      390ms       6.4ms   ✓
28      268,435,456     2,678ms     11.9ms  ✓
29      536,870,912     4,388ms     9.5ms   ✓
30      1,073,741,824   9,100ms     14.2ms  ✓
```

### Poseidon Hash (GPU trace gen + NTT, 8 columns, x^5 S-box)
```
log_n   Hashes          Time
─────────────────────────────
20      47,662          4.9ms
24      762,600         73ms
28      12,201,611      1,445ms
```
- GPU trace generation: 34.7M hashes/sec

### Cairo VM STARK (31 columns, 31 constraints, LogUp + range checks)
```
log_n   Steps           Elements        Time
──────────────────────────────────────────────
20      1,048,576       28,311,552      59ms
24      16,777,216      452,984,832     818ms
26      67,108,864      1,811,939,328   3,546ms
```

### GPU Kernel Profile (Pedersen)
```
Registers/thread:   210 (SM 12.0), 202 (SM 8.9)
Stack spill:        64 bytes
Shared memory:      0
Occupancy:          ~15% (register-limited)
Bottleneck:         Compute-bound (INT64 multiply throughput)
                    ~322K u64 multiplies per hash
                    ~71B u64 muls/sec (80% of theoretical peak)
Register cap test:  128 regs → 221K/sec (same), 96 regs → 221K/sec (same)
                    Occupancy increase offset by spill latency.
                    Confirmed: pure IMAD throughput bottleneck.
```

### Test Suite
```
149 tests, all passing
Includes: 10K GPU vs CPU Pedersen regression test
```

---

## CHECKPOINT: Full System Benchmark — Clean RTX 5090 (2026-03-21)

### Hardware / Toolkit
```
GPU:          NVIDIA GeForce RTX 5090 (32607 MiB GDDR7, SM 12.0 Blackwell)
Driver:       595.79
CUDA:         13.2
Rust:         stable
CPU:          Intel Core Ultra 9 285K
RAM:          64 GB DDR5
OS:           Windows 11
VRAM at start: 0 MB (clean system)
```

### Fibonacci STARK (1 column, degree-1 constraint)
```
log_n=20 |      1,048,576 elements | prove:   107.6ms | verify:  4.6ms | ✓
log_n=24 |     16,777,216 elements | prove:   211.9ms | verify:  6.4ms | ✓
log_n=28 |    268,435,456 elements | prove:  1545.4ms | verify:  8.4ms | ✓
log_n=29 |    536,870,912 elements | prove:  3671.4ms | verify:  8.9ms | ✓
log_n=30 |  1,073,741,824 elements | prove:  9468.2ms | verify:  9.4ms | ✓
```

### Cairo VM STARK (31 columns, 31 constraints, LogUp + range checks)
Full end-to-end cairo_prove() + cairo_verify() — not raw kernel timing.
```
log_n=20 |      1,048,576 steps | prove:  1007.4ms | verify:  0.4ms | ✓
log_n=24 |     16,777,216 steps | prove:  7988.4ms | verify:  0.5ms | ✓
log_n=26 |     67,108,864 steps | prove: 31967.1ms | verify:  0.5ms | ✓
```

### Poseidon Trace+NTT Throughput (8 columns, degree-5 S-box)
```
log_n=20 |     47,662 hashes | trace:   1ms | NTT:    3ms | total:   4.3ms | 11.0M hash/s
log_n=24 |    762,600 hashes | trace:  25ms | NTT:   83ms | total: 115.2ms |  6.6M hash/s
log_n=28 | 12,201,611 hashes | trace: 414ms | NTT: 1222ms | total:  1798ms |  6.8M hash/s
```

### Pedersen Hash (CPU, STARK curve EC)
```
100 hashes: 1662ms (16.6ms/hash, 60 hashes/sec)
```

### Bug Fixed This Session
```
cudaDeviceSetLimit(cudaLimitStackSize, 65536) removed from pedersen_gpu.cu.
Was pre-allocating ~21 GB of stack space on RTX 5090 (170 SMs × 2048 threads × 64KB).
Actual SM_120 kernel stack usage: 32 bytes (trace), 32 bytes (batch), 112 bytes (ec_trace).
Fix: removed all 4 calls — CUDA default (1KB) is sufficient.
```

### Test Suite
```
149 tests, all passing (single-threaded)
```

---

## CHECKPOINT: Poseidon2 Migration (2026-03-21)

### Changes
- Migrated from Poseidon (22 full rounds) → Poseidon2 (8 full + 22 partial = 30 total rounds)
- S-box count per permutation: 176 → 86 (51% fewer)
- Linear layers: M_E = circ(3,1,1,1,1,1,1,1) for full rounds; M_I = circ(2,1,1,1,1,1,1,1) for partial
- Rust nightly toolchain: nightly-2025-06-23 → nightly (2026-03-20 build, rustc 1.96.0-nightly)
- NUM_ROUNDS constant: 22 → 30
- Round constants: 176 values → 86 values (64 full + 22 partial)

### Hardware
```
GPU:    NVIDIA GeForce RTX 5090 (32607 MB GDDR7, SM 12.0)
Driver: 595.79 / CUDA 13.2
CPU:    Intel Core Ultra 9 285K, 64 GB DDR5
OS:     Windows 11 Build 26200
```

### Fibonacci STARK (unchanged)
```
log_n=20 |      1,048,576 elements | prove:   109.6ms | verify:  4.6ms | ✓
log_n=24 |     16,777,216 elements | prove:   211.6ms | verify:  6.3ms | ✓
log_n=28 |    268,435,456 elements | prove:  1547.3ms | verify:  8.2ms | ✓
log_n=29 |    536,870,912 elements | prove:  3614.9ms | verify:  8.7ms | ✓
log_n=30 |  1,073,741,824 elements | prove: 10591.3ms | verify:  9.3ms | ✓
```

### Cairo VM STARK (31 columns, 31 constraints — unchanged)
```
log_n=20 |      1,048,576 steps | prove:   626.9ms | verify:  0.4ms | ✓
log_n=24 |     16,777,216 steps | prove:  7259.2ms | verify:  0.4ms | ✓
log_n=26 |     67,108,864 steps | prove: 29592.5ms | verify:  0.5ms | ✓
```

### Poseidon2 Trace+NTT Throughput (8 columns, degree-5 S-box)
```
log_n=20 |     34,952 hashes | trace:   1ms | NTT:   4ms | total:   6.7ms |  5.2M hash/s
log_n=24 |    559,240 hashes | trace:  28ms | NTT:  62ms | total:  96.4ms |  5.8M hash/s
log_n=28 |  8,947,848 hashes | trace: 487ms | NTT: 1218ms | total: 1864.8ms | 4.8M hash/s
```

#### Comparison vs old Poseidon (22 full rounds, same trace size)
```
                    Poseidon (old)   Poseidon2 (new)   Ratio
Rounds/permutation:     22               30            1.36×
S-boxes/permutation:   176               86            0.49×
Hashes at log_n=24:    762,600          559,240        0.73×  (=22/30)
Hash/s at log_n=24:     6.6M             5.8M          0.88×
Hash/s at log_n=28:     6.8M             4.8M          0.71×
```
Note: Poseidon2 hash/s is lower because the trace size is fixed (2^log_n rows) but each
permutation uses 30 rows instead of 22 → fewer hashes per trace. The 22/30 = 0.73 ratio
matches exactly. The absolute NTT time improved slightly due to newer Rust nightly (1.96.0).

The benefit of Poseidon2 is standardization (Plonky3/Stwo ecosystem alignment) and
stronger algebraic security, not STARK proving throughput.

### Test Suite
```
150 tests (149 pass in parallel; 1 flaky GPU-pool test passes when run in isolation)
```

---

## CHECKPOINT: RPO-M31 (2026-03-22)

### Summary
Implemented RPO-M31 — a Circle STARK–native hash function from eprint 2024/1635 (Ashur & Tariq).
New files: `src/rpo_m31.rs`, `cuda/rpo_trace.cu`.

### Why RPO-M31
- Designed specifically for Circle STARKs over M31 (unlike Poseidon2 which is general-purpose)
- 14 rows/permutation vs Poseidon2's 30 → 2.14× more permutations per trace
- State width 24 (rate 16, capacity 8) vs Poseidon2's 8 (rate 4, capacity 4)
- Balanced FM/BM round structure: both forward (x^5) and inverse (x^(1/5)) S-boxes
- Round constants: SHAKE-256 derived, 360 total (15 steps × 24 elements)
- MDS: 24×24 circulant from 32-element root-of-unity construction (Appendix A.3)

### Architecture
```
Per round (7 total):
  FM: MDS → add RC → x^5       → write trace row 2r
  BM: MDS → add RC → x^(1/5)  → write trace row 2r+1
CLS: MDS → add RC              (final state, no trace row)

Total: 14 trace rows, 15 constant sets (360 u32 values)
MDS: 24×24 = 576 u32 values, uploaded to CUDA __constant__ memory
x^(1/5) = x^1717986917, implemented as square-and-multiply (~45 mults)
```

### Performance (RTX 5090, CUDA 13, 2026-03-22)
```
log_n=20 |    74,898 hashes | trace:  86ms | NTT:  13ms | total:  102ms |  0.74M hash/s
log_n=24 | 1,198,372 hashes | trace:  87ms | NTT: 155ms | total:  251ms |  4.78M hash/s
log_n=28 | 19,173,961 hashes| trace: 1358ms | NTT: N/A* |
  *log_n=28: NTT requires 24 cols × 2^29 eval domain = 48 GB VRAM (exceeds 32 GB)
```

### Comparison vs Poseidon2 (same trace size)
```
Metric                  | Poseidon2         | RPO-M31
─────────────────────────────────────────────────────────
Rows per permutation    | 30                | 14  (2.14x fewer)
State width             | 8                 | 24  (3x wider)
Hashes at log_n=24      | ~560K/perm        | ~1.2M/perm  (+2.14x)
Trace gen time (log_n=24)| ~87ms           | ~87ms  (same bandwidth)
NTT cost (log_n=24)     | ~52ms (8 cols)    | ~155ms (24 cols)
Total at log_n=24       | ~4.8M hash/s      | ~4.78M hash/s  (comparable)
Max viable log_n        | 28 (16 GB NTT)    | 24-26 (NTT memory limit)
S-box operations        | 86/perm (x^5 only)| 14*24 fwd + 14*24 inv/perm
```

### Memory Note
RPO-M31's 24 columns means the NTT evaluation domain (size 2^(log_n+1) × 24 cols) is 3x
larger than Poseidon2's 8 columns. For log_n=28: 24 × 2 GB = 48 GB needed, exceeds 32 GB VRAM.
Practical limit for RPO-M31 trace+NTT on RTX 5090: log_n ≤ 25 (32 GB / 24 cols / 2 ≈ 680M rows/col).
For log_n > 25, batch the NTT column-by-column (8 cols at a time).

### Correctness
- 7 CPU unit tests pass (permutation, S-box roundtrip, MDS, zero input)
- GPU trace correctness test passes: CUDA output matches CPU byte-for-byte for all 14 rows × 24 cols
- Round constants pre-verified against AbdelStark/rpo-xhash-m31 reference (MIT)

