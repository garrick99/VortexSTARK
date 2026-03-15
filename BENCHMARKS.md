# kraken-stark Benchmark Artifact

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

### Cairo VM STARK (27 columns, 20 constraints, LogUp + range checks)
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
114 tests, all passing
Includes: 10K GPU vs CPU Pedersen regression test
```
