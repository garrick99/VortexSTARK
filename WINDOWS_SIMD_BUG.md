# Bug Report: STATUS_HEAP_CORRUPTION in stwo-cairo on Windows MSVC

## Summary

stwo-cairo's interaction trace generation crashes with `STATUS_HEAP_CORRUPTION` (0xC0000374) on Windows MSVC when using `portable_simd` with rayon thread pools. The same code, same compiler version, and same binary crate runs correctly on Linux (tested via WSL2 on the same machine).

## Environment

**Crashes (Windows MSVC):**
- OS: Windows 11 (10.0.26200)
- Toolchain: `nightly-2025-06-23-x86_64-pc-windows-msvc`
- Compiler: `rustc 1.89.0-nightly (be19eda0d 2025-06-22)`
- RUSTFLAGS: `-C target-cpu=native` (also crashes without it)
- GPU: NVIDIA RTX 5090, CUDA 13.0 (not relevant — crash is in CPU SIMD code)

**Works (Linux via WSL2, same machine):**
- OS: Linux 6.6.87.2-microsoft-standard-WSL2 x86_64
- Toolchain: `nightly-2025-06-23-x86_64-unknown-linux-gnu`
- Compiler: `rustc 1.89.0-nightly (be19eda0d 2025-06-22)` (identical)
- Same source code, same Cargo.lock

## Reproduction

```bash
# Clone stwo-cairo (or use existing checkout)
# Ensure workspace Cargo.toml points to stwo fork with cuda-backend branch

# Build
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin run_and_prove

# Run (crashes on Windows, works on Linux)
./target/release/run_and_prove \
  --program test_data/test_prove_verify_ret_opcode/compiled.json \
  --proof_path /tmp/proof.json
```

**Any** test program crashes — tested with `test_prove_verify_ret_opcode` (simplest, 15 Cairo steps) and `test_prove_verify_all_opcode_components` (1495 steps, 67 components). Both crash identically on Windows.

## Crash Details

**Windows exit code:** `0xC0000374` = `STATUS_HEAP_CORRUPTION`

**No Rust panic.** No backtrace. No error message. The process is terminated by the Windows heap manager after detecting corruption. `std::panic::set_hook` does not fire. `RUST_BACKTRACE=full` produces no output.

**Crash location:** Inside `rayon::scope()` in `CairoInteractionClaimGenerator::write_interaction_trace()`. The scope spawns ~150 parallel tasks that each call component-specific `write_interaction_trace()` methods using `PackedM31`/`PackedQM31` SIMD operations.

**Narrowing:**
- Individual spawned tasks can complete (e.g., `add_opcode` prints "DONE")
- The `rayon::scope()` itself never returns — at least one spawned task corrupts the heap
- `RAYON_NUM_THREADS=1` does not fix it (still crashes)
- `RUST_MIN_STACK=67108864` does not fix it (not a stack overflow)
- `-C target-cpu=native` does not fix it (changes which task crashes first, but crash persists)

## What the Crashing Code Does

The interaction trace generator (`write_interaction_trace`) performs LogUp argument computation using SIMD-packed field arithmetic:

```rust
// From add_opcode.rs (representative of all ~60 component files)
(col_gen.par_iter_mut(), &self.lookup_data.verify_instruction_0, &self.lookup_data.memory_address_to_id_0)
    .into_par_iter()
    .for_each(|(writer, values0, values1)| {
        let denom0: PackedQM31 = common_lookup_elements.combine(values0);
        let denom1: PackedQM31 = common_lookup_elements.combine(values1);
        writer.write_frac(denom0 + denom1, denom0 * denom1);
    });
```

This uses `std::simd` types (`PackedM31` = `Simd<u32, 16>`, `PackedQM31` = 4× `PackedM31`) from stwo's SIMD backend.

## Pipeline Progress Before Crash

The following phases complete successfully on Windows before the crash:

| Phase | Status | Notes |
|-------|--------|-------|
| Cairo VM execution | ✅ | 15-1495 steps depending on program |
| Trace adaptation | ✅ | Relocator, memory builder, state transitions |
| Preprocessed trace conversion (SimdBackend → CudaBackend) | ✅ | 161 columns converted |
| GPU NTT interpolation | ✅ | All columns interpolated on GPU |
| GPU Merkle leaf hashing | ✅ | 67M leaves hashed in 14ms |
| GPU Merkle tree building | ✅ | All layers built |
| Base trace commit | ✅ | 87ms GPU |
| GPU proof-of-work grinding | ✅ | <1ms |
| **Interaction trace generation** | **❌ CRASH** | `STATUS_HEAP_CORRUPTION` |

## Linux (WSL2) Results

The identical code produces correct, verified proofs on Linux:

```
[TIMING] Base trace generation: 0.169s
[TIMING] Base trace commit: 0.127s
[TIMING] Grind (pow=176781): 0.000s
[TIMING] Interaction trace gen: 0.147s
[TIMING] Interaction trace commit: 0.123s
[TIMING] prove_ex (constraint+OODS+FRI+Merkle): 12.661s
[TIMING] Total prove_cairo: 13.229s
✅ Proved successfully!
```

Full 67-component Cairo AIR proof generated and verified in 13.2s.

## Analysis

This appears to be a memory safety issue in the interaction between:
1. `std::simd` / `portable_simd` packed field operations
2. `rayon` thread pool work-stealing
3. Windows MSVC heap allocator

The heap corruption manifests specifically during parallel SIMD-packed field arithmetic across rayon worker threads. Since the same binary logic works on Linux with the same compiler, the bug is likely in:
- MSVC codegen for `portable_simd` intrinsics (alignment, bounds)
- Windows heap allocator sensitivity to SIMD-related alignment violations
- A subtle UB in packed field operations that Linux's allocator tolerates but Windows catches

## Workaround

Use Linux (native or WSL2). The full proving pipeline works correctly on Linux with the RTX 5090 via WSL2 GPU passthrough.

## Versions

- stwo: v2.1.0 (from fork, branch `cuda-backend`, commit `dc081359`)
- stwo-constraint-framework: v2.1.0 (same fork)
- stwo-cairo: v1.1.0
- cairo-vm: v3.2.0
- rustc: 1.89.0-nightly (be19eda0d 2025-06-22)
- rayon: (workspace version from stwo-cairo)

## Related

- Rust `portable_simd` tracking issue: https://github.com/rust-lang/rust/issues/86656
- stwo uses `#![feature(portable_simd)]` extensively for `PackedM31` / `PackedQM31`
- This may be related to known MSVC codegen issues with nightly SIMD features
