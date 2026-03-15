//! Pedersen hash builtin for Cairo VM.
//!
//! The Pedersen hash operates on the STARK curve (NOT M31):
//!   y² = x³ + x + β  over  F_p  where  p = 2^251 + 17·2^192 + 1
//!
//! H(a, b) = [P₀ + a_low·P₁ + a_high·P₂ + b_low·P₃ + b_high·P₄]_x
//!
//! where a_low = lowest 248 bits, a_high = highest 4 bits,
//! and P₀..P₄ are fixed curve points derived from digits of π.
//!
//! STARK AIR representation:
//! The 252-bit field elements are decomposed into limbs of M31-sized values
//! for the constraint system. Each Stark252 element requires ~9 M31 limbs
//! (252 bits / 31 bits per limb ≈ 9 limbs).
//!
//! The EC point addition and scalar multiplication are expressed as
//! polynomial constraints over these M31 limbs.
//!
//! Architecture (matching stwo-cairo):
//! - pedersen_builtin: 3 trace columns (input_a_id, input_b_id, output_id)
//!   Links to memory via LogUp
//! - pedersen_aggregator: 206 columns
//!   Aggregates the windowed scalar multiplication results
//! - partial_ec_mul: 297 columns (windowed, 18-bit windows)
//!   Performs the actual elliptic curve scalar multiplication
//!
//! Total: ~500+ columns for the full Pedersen AIR
//!
//! For our implementation, we take a pragmatic approach:
//! 1. The COMPUTATION runs natively (Stark252 arithmetic on CPU/GPU)
//! 2. The PROOF uses a lookup-based approach: the Pedersen builtin
//!    produces (input_a, input_b, output) tuples, and LogUp proves
//!    the VM's memory accesses match these tuples.
//! 3. A separate "Pedersen table" component proves the tuples are
//!    valid Pedersen hashes (via the EC arithmetic constraints).

use crate::field::M31;

/// STARK curve prime: p = 2^251 + 17·2^192 + 1
pub const STARK_PRIME_HEX: &str =
    "0800000000000011000000000000000000000000000000000000000000000001";

/// STARK curve parameters: y² = x³ + αx + β where α = 1
pub const CURVE_ALPHA: u64 = 1;

/// β = 3141592653589793238462643383279502884197169399375105820974944592307816406665
pub const CURVE_BETA_HEX: &str =
    "06f21413efbe40de150e596d72f7a8c5609ad26c15c915c1f4cdfcb99cee9e89";

/// Number of M31 limbs to represent a Stark252 field element.
/// 252 bits / 31 bits = 9 limbs (with the top limb < 2^(252-8*31) = 2^4 = 16)
pub const N_LIMBS: usize = 9;

/// A 252-bit field element represented as 9 M31 limbs (little-endian).
/// limb[0] is the least significant 31 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stark252 {
    pub limbs: [u32; N_LIMBS],
}

impl Stark252 {
    pub const ZERO: Self = Self { limbs: [0; N_LIMBS] };

    /// Create from a u64 (for small values).
    pub fn from_u64(v: u64) -> Self {
        let mut limbs = [0u32; N_LIMBS];
        limbs[0] = (v & 0x7FFF_FFFF) as u32;
        limbs[1] = ((v >> 31) & 0x7FFF_FFFF) as u32;
        limbs[2] = ((v >> 62) & 0x3) as u32; // top 2 bits
        Self { limbs }
    }

    /// Create from a hex string (big-endian).
    pub fn from_hex(s: &str) -> Self {
        let s = s.trim_start_matches("0x").trim_start_matches("0X");
        let bytes = hex_to_bytes(s);
        let mut limbs = [0u32; N_LIMBS];
        // Convert big-endian bytes to little-endian 31-bit limbs
        let mut bit_pos = 0usize;
        for &byte in bytes.iter().rev() {
            for bit in 0..8 {
                if byte & (1 << bit) != 0 {
                    let limb_idx = bit_pos / 31;
                    let limb_bit = bit_pos % 31;
                    if limb_idx < N_LIMBS {
                        limbs[limb_idx] |= 1 << limb_bit;
                    }
                }
                bit_pos += 1;
            }
        }
        Self { limbs }
    }

    /// Convert to M31 limbs (for trace columns).
    pub fn to_m31_limbs(&self) -> [M31; N_LIMBS] {
        std::array::from_fn(|i| M31(self.limbs[i]))
    }
}

/// A point on the STARK curve.
#[derive(Clone, Copy, Debug, Default)]
pub struct StarkPoint {
    pub x: Stark252,
    pub y: Stark252,
}

/// The 5 Pedersen hash constant points (from digits of π).
pub fn pedersen_points() -> [StarkPoint; 5] {
    [
        StarkPoint {
            x: Stark252::from_hex("49ee3eba8c1600700ee1b87eb599f16716b0b1022947733551fde4050ca6804"),
            y: Stark252::from_hex("3ca0cfe4b3bc6ddf346d49d06ea0ed34e621062c0e056c1d0405d266e10268a"),
        },
        StarkPoint {
            x: Stark252::from_hex("234287dcbaffe7f969c748655fca9e58fa8120b6d56eb0c1080d17957ebe47b"),
            y: Stark252::from_hex("3b056f100f96fb21e889527d41f4e39940135dd7a6c7e6c2f8116572f578e85"),
        },
        StarkPoint {
            x: Stark252::from_hex("4fa56f376c83db33f9dab2656558f3399099ec1de5e3018b7571f510a2c7768"),
            y: Stark252::from_hex("3f42a042e45b8a3e3821a7133325bfa989e2bc26485dbe63ac6eadc28fc2fad"),
        },
        StarkPoint {
            x: Stark252::from_hex("4ba4cc166be8dec764910f75b45f74b40642ad9b32d50d8865e3e7caa740577"),
            y: Stark252::from_hex("00416a975392d0e71777ab65e5e7e4c54daee0efbb7d00b8d2ccacfefa2d8e1c"),
        },
        StarkPoint {
            x: Stark252::from_hex("54302dcb0e6cc1c6e44cca8f61a63bb2ca65048d53fb325d36ff12c49a58202"),
            y: Stark252::from_hex("01b77b3e37d13504b348046268d8ae25ce98ad783c25561a879dcc77e99c2426"),
        },
    ]
}

/// Pedersen builtin for Cairo VM.
/// Manages invocations and generates trace data.
/// Stores both Stark252 (for CPU trace) and Fp (for GPU trace) representations.
pub struct PedersenBuiltin {
    /// (input_a, input_b, output) tuples in Stark252 format
    pub entries: Vec<(Stark252, Stark252, Stark252)>,
    /// Fp inputs for GPU trace generation (avoids Stark252→Fp conversion)
    pub fp_inputs_a: Vec<super::stark252_field::Fp>,
    pub fp_inputs_b: Vec<super::stark252_field::Fp>,
}

impl PedersenBuiltin {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            fp_inputs_a: Vec::new(),
            fp_inputs_b: Vec::new(),
        }
    }

    /// Invoke Pedersen hash using real EC arithmetic on the STARK curve.
    pub fn invoke(&mut self, a: Stark252, b: Stark252) -> Stark252 {
        use super::stark252_field::pedersen_hash;
        let fp_a = stark252_to_fp(&a);
        let fp_b = stark252_to_fp(&b);
        let fp_out = pedersen_hash(fp_a, fp_b);
        let out = fp_to_stark252(&fp_out);
        self.entries.push((a, b, out));
        self.fp_inputs_a.push(fp_a);
        self.fp_inputs_b.push(fp_b);
        out
    }

    pub fn n_invocations(&self) -> usize {
        self.entries.len()
    }

    /// Generate trace columns for the Pedersen builtin.
    /// Each invocation produces one row with 3 * N_LIMBS = 27 columns:
    /// [a_limbs(9), b_limbs(9), output_limbs(9)]
    pub fn generate_trace(&self, log_n: u32) -> Vec<Vec<u32>> {
        let n = 1usize << log_n;
        let n_cols = 3 * N_LIMBS; // 27 columns
        let mut cols: Vec<Vec<u32>> = (0..n_cols).map(|_| vec![0u32; n]).collect();

        for (i, (a, b, out)) in self.entries.iter().enumerate() {
            if i >= n { break; }
            for j in 0..N_LIMBS {
                cols[j][i] = a.limbs[j];
                cols[N_LIMBS + j][i] = b.limbs[j];
                cols[2 * N_LIMBS + j][i] = out.limbs[j];
            }
        }

        cols
    }

    /// Generate LogUp entries for memory consistency.
    pub fn logup_entries(&self, base_addr: u64) -> Vec<(M31, M31)> {
        let mut entries = Vec::new();
        let stride = 3u64; // 3 memory cells per invocation (a_id, b_id, out_id)

        for (inv_idx, (a, b, out)) in self.entries.iter().enumerate() {
            let base = base_addr + inv_idx as u64 * stride;
            // For simplicity, use first limb as the memory value
            entries.push((M31((base) as u32), M31(a.limbs[0])));
            entries.push((M31((base + 1) as u32), M31(b.limbs[0])));
            entries.push((M31((base + 2) as u32), M31(out.limbs[0])));
        }
        entries
    }
}

/// Convert Stark252 (9×31-bit limbs) to Fp (4×64-bit limbs).
pub fn stark252_to_fp(s: &Stark252) -> super::stark252_field::Fp {
    use super::stark252_field::Fp;
    // Reassemble 252 bits from 31-bit limbs into 64-bit limbs
    let mut bits = [0u8; 256];
    for i in 0..N_LIMBS {
        let val = s.limbs[i];
        let n_bits = if i == N_LIMBS - 1 { 252 - 31 * (N_LIMBS - 1) } else { 31 };
        for b in 0..n_bits {
            bits[i * 31 + b] = ((val >> b) & 1) as u8;
        }
    }
    let mut v = [0u64; 4];
    for i in 0..252 {
        if bits[i] == 1 {
            v[i / 64] |= 1u64 << (i % 64);
        }
    }
    Fp { v }
}

/// Convert Fp (4×64-bit limbs) to Stark252 (9×31-bit limbs).
pub fn fp_to_stark252(fp: &super::stark252_field::Fp) -> Stark252 {
    let mut bits = [0u8; 256];
    for i in 0..252 {
        if fp.v[i / 64] & (1u64 << (i % 64)) != 0 {
            bits[i] = 1;
        }
    }
    let mut limbs = [0u32; N_LIMBS];
    for i in 0..N_LIMBS {
        let n_bits = if i == N_LIMBS - 1 { 252 - 31 * (N_LIMBS - 1) } else { 31 };
        for b in 0..n_bits {
            if bits[i * 31 + b] == 1 {
                limbs[i] |= 1 << b;
            }
        }
    }
    Stark252 { limbs }
}

/// Initialize GPU Pedersen: upload constant points and precomputed windowed tables.
pub fn gpu_init() {
    use super::stark252_field::{CurvePoint, pedersen_points};
    use crate::cuda::ffi;

    let points = pedersen_points();

    // Upload raw affine points (still needed for backward compat)
    let mut px = [0u64; 20];
    let mut py = [0u64; 20];
    for (i, pt) in points.iter().enumerate() {
        if let CurvePoint::Affine(x, y) = pt {
            for j in 0..4 { px[i*4+j] = x.v[j]; py[i*4+j] = y.v[j]; }
        }
    }
    unsafe { ffi::cuda_pedersen_upload_points(px.as_ptr(), py.as_ptr()); }

    // Precompute windowed tables for P1..P4 (4 points, 16 multiples each)
    // Table[point][k] = k * P_i in Montgomery Jacobian form
    // k=0: point at infinity (0, 1, 0 in Jacobian)
    // k=1: P_i, k=2: 2*P_i, ..., k=15: 15*P_i
    let mut table_x = [0u64; 4 * 16 * 4]; // [4 points][16 multiples][4 limbs]
    let mut table_y = [0u64; 4 * 16 * 4];
    let mut table_z = [0u64; 4 * 16 * 4];

    let r_mod_p = super::stark252_field::compute_r_mod_p();
    let one_mont = r_mod_p; // 1 in Montgomery form = R mod p

    for pt_idx in 0..4 {
        let base = points[pt_idx + 1]; // P1..P4 (skip P0)
        let (bx, by) = match base {
            CurvePoint::Affine(x, y) => (x, y),
            _ => panic!("Pedersen point is infinity"),
        };

        // k=0: infinity (Montgomery Jacobian: X=0, Y=R mod p, Z=0)
        let offset = pt_idx * 16 * 4;
        // X=0, Y=R (Montgomery 1), Z=0
        for j in 0..4 { table_y[offset + j] = r_mod_p.v[j]; }
        // X and Z already 0

        // k=1: P_i in Montgomery Jacobian (X_mont, Y_mont, Z_mont=R)
        // to_mont(a) = a * R mod p. We have R mod p, so standard Fp mul gives Montgomery form.
        let bx_mont = bx * r_mod_p;
        let by_mont = by * r_mod_p;

        let off1 = offset + 1 * 4; // k=1
        for j in 0..4 {
            table_x[off1 + j] = bx_mont.v[j];
            table_y[off1 + j] = by_mont.v[j];
            table_z[off1 + j] = one_mont.v[j];
        }

        // k=2..15: compute via repeated addition on CPU (affine, then convert)
        let mut current = base;
        for k in 2..16u32 {
            current = current.add(base);
            let (cx, cy) = match current {
                CurvePoint::Affine(x, y) => (x, y),
                _ => {
                    // Point at infinity — store as (0, R, 0) in Montgomery Jacobian
                    let off_k = offset + k as usize * 4;
                    for j in 0..4 { table_y[off_k + j] = r_mod_p.v[j]; }
                    continue;
                }
            };
            let cx_mont = cx * r_mod_p;
            let cy_mont = cy * r_mod_p;
            let off_k = offset + k as usize * 4;
            for j in 0..4 {
                table_x[off_k + j] = cx_mont.v[j];
                table_y[off_k + j] = cy_mont.v[j];
                table_z[off_k + j] = one_mont.v[j];
            }
        }
    }

    // P0 in Montgomery Jacobian
    let (p0x, p0y) = match points[0] {
        CurvePoint::Affine(x, y) => (x, y),
        _ => panic!("P0 is infinity"),
    };
    let p0x_mont = p0x * r_mod_p;
    let p0y_mont = p0y * r_mod_p;

    unsafe {
        ffi::cuda_pedersen_upload_tables(
            table_x.as_ptr(), table_y.as_ptr(), table_z.as_ptr(),
            p0x_mont.v.as_ptr(), p0y_mont.v.as_ptr(), one_mont.v.as_ptr(),
        );
    }
}

/// Batch Pedersen hash on GPU. Returns output Fp values.
/// GPU computes full affine x on-device (inline Fermat inverse).
/// Uses pinned memory + async stream for maximum transfer throughput.
pub fn gpu_hash_batch(
    inputs_a: &[super::stark252_field::Fp],
    inputs_b: &[super::stark252_field::Fp],
) -> Vec<super::stark252_field::Fp> {
    use crate::cuda::ffi;
    use crate::device::DeviceBuffer;
    use super::stark252_field::Fp;
    let n = inputs_a.len();
    assert_eq!(n, inputs_b.len());
    if n == 0 { return vec![]; }

    let n_u64 = n * 4;
    let _bytes_in = n_u64 * std::mem::size_of::<u64>();

    // Zero-copy reinterpret inputs (Fp is repr(C) with [u64; 4])
    let flat_a = unsafe { std::slice::from_raw_parts(inputs_a.as_ptr() as *const u64, n_u64) };
    let flat_b = unsafe { std::slice::from_raw_parts(inputs_b.as_ptr() as *const u64, n_u64) };

    // Use a CUDA stream for async operations
    let stream = ffi::CudaStream::new();

    // Allocate device buffers (pool-based, near-zero cost)
    let mut d_a = DeviceBuffer::<u64>::alloc(n_u64);
    let mut d_b = DeviceBuffer::<u64>::alloc(n_u64);
    let mut d_out = DeviceBuffer::<u64>::alloc(n_u64);

    // Async H2D uploads on stream
    d_a.upload_async(flat_a, &stream);
    d_b.upload_async(flat_b, &stream);

    // Launch kernel on same stream (waits for uploads to complete)
    unsafe {
        ffi::cuda_pedersen_hash_batch_stream(
            d_a.as_ptr(), d_b.as_ptr(),
            d_out.as_mut_ptr(), std::ptr::null_mut(),
            n as u32,
            stream.ptr,
        );
    }

    // Allocate result Vec while kernel is still running on GPU
    let mut results = vec![Fp::ZERO; n];

    // Async D2H download directly into result Vec (no intermediate buffer)
    d_out.download_into_async(
        unsafe { std::slice::from_raw_parts_mut(results.as_mut_ptr() as *mut u64, n_u64) },
        &stream,
    );

    // Wait for everything
    stream.sync();

    results
}

/// Fused GPU Pedersen hash + trace column generation.
/// Hashes (a, b) pairs on GPU, decomposes into 27 M31 trace columns,
/// and returns them as DeviceBuffers — results never touch the host.
///
/// Column layout: [a_limbs(9), b_limbs(9), output_limbs(9)]
/// Each column has `trace_len` rows (padded with zeros beyond `n`).
///
/// This is the "Pedersen-as-stage" API: output feeds directly into NTT → Merkle.
pub fn gpu_pedersen_trace(
    inputs_a: &[super::stark252_field::Fp],
    inputs_b: &[super::stark252_field::Fp],
    log_trace_len: u32,
) -> Vec<crate::device::DeviceBuffer<u32>> {
    use crate::cuda::ffi;
    use crate::device::DeviceBuffer;
    let n = inputs_a.len();
    assert_eq!(n, inputs_b.len());
    let trace_len = 1usize << log_trace_len;
    assert!(n <= trace_len, "more invocations ({n}) than trace rows ({trace_len})");

    let n_u64 = n * 4;

    // Zero-copy reinterpret inputs
    let flat_a = unsafe { std::slice::from_raw_parts(inputs_a.as_ptr() as *const u64, n_u64) };
    let flat_b = unsafe { std::slice::from_raw_parts(inputs_b.as_ptr() as *const u64, n_u64) };

    let stream = ffi::CudaStream::new();

    // Upload inputs async
    let mut d_a = DeviceBuffer::<u64>::alloc(n_u64);
    let mut d_b = DeviceBuffer::<u64>::alloc(n_u64);
    d_a.upload_async(flat_a, &stream);
    d_b.upload_async(flat_b, &stream);

    // Allocate 27 trace columns on GPU (zero-initialized for padding)
    let n_cols = 3 * N_LIMBS; // 27
    let mut d_cols: Vec<DeviceBuffer<u32>> = (0..n_cols)
        .map(|_| {
            let mut buf = DeviceBuffer::<u32>::alloc(trace_len);
            buf.zero();
            buf
        })
        .collect();

    // Build device pointer array: [27 device pointers]
    let col_ptrs: Vec<*mut u32> = d_cols.iter_mut().map(|c| c.as_mut_ptr()).collect();
    let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

    // Launch fused hash + trace kernel
    unsafe {
        ffi::cuda_pedersen_trace(
            d_a.as_ptr(), d_b.as_ptr(),
            d_col_ptrs.as_ptr() as *mut *mut u32,
            n as u32,
            stream.ptr,
        );
    }

    stream.sync();

    // Return 27 DeviceBuffers — ready for NTT, no host round-trip
    d_cols
}

/// Pipeline timing breakdown for a batch Pedersen hash.
#[derive(Debug, Clone)]
pub struct PipelineTiming {
    pub n: usize,
    pub flatten_us: f64,      // CPU: flatten inputs to flat arrays
    pub upload_us: f64,       // H2D: DeviceBuffer::from_host
    pub alloc_us: f64,        // GPU: output buffer allocation
    pub kernel_us: f64,       // GPU: hash kernel + inline inverse + sync
    pub download_us: f64,     // D2H: to_host for affine x
    pub repack_us: f64,       // CPU: repack flat arrays to Fp vecs
    pub inverse_us: f64,      // CPU: (eliminated — now inline on GPU)
    pub total_us: f64,        // wall clock total
}

/// Instrumented version of gpu_hash_batch that returns per-phase timing.
/// Uses sync path (not async stream) for accurate per-phase measurement.
pub fn gpu_hash_batch_timed(
    inputs_a: &[super::stark252_field::Fp],
    inputs_b: &[super::stark252_field::Fp],
) -> (Vec<super::stark252_field::Fp>, PipelineTiming) {
    use crate::cuda::ffi;
    use crate::device::DeviceBuffer;
    use super::stark252_field::Fp;
    use std::time::Instant;

    let n = inputs_a.len();
    assert_eq!(n, inputs_b.len());
    let n_u64 = n * 4;

    let wall_start = Instant::now();

    // Phase 1: Zero-copy reinterpret
    let t0 = Instant::now();
    let flat_a = unsafe { std::slice::from_raw_parts(inputs_a.as_ptr() as *const u64, n_u64) };
    let flat_b = unsafe { std::slice::from_raw_parts(inputs_b.as_ptr() as *const u64, n_u64) };
    let flatten_us = t0.elapsed().as_secs_f64() * 1e6;

    // Phase 2: H2D upload (sync for timing)
    let t0 = Instant::now();
    let d_a = DeviceBuffer::from_host(flat_a);
    let d_b = DeviceBuffer::from_host(flat_b);
    let upload_us = t0.elapsed().as_secs_f64() * 1e6;

    // Phase 3: Alloc output
    let t0 = Instant::now();
    let mut d_out = DeviceBuffer::<u64>::alloc(n_u64);
    let alloc_us = t0.elapsed().as_secs_f64() * 1e6;

    // Phase 4: Kernel + sync
    let t0 = Instant::now();
    unsafe {
        ffi::cuda_pedersen_hash_batch(
            d_a.as_ptr(), d_b.as_ptr(),
            d_out.as_mut_ptr(), std::ptr::null_mut(),
            n as u32,
        );
        ffi::cuda_device_sync();
    }
    let kernel_us = t0.elapsed().as_secs_f64() * 1e6;

    // Phase 5: D2H download directly into result Vec (no intermediate)
    let t0 = Instant::now();
    let mut results = vec![Fp::ZERO; n];
    d_out.download_into(unsafe {
        std::slice::from_raw_parts_mut(results.as_mut_ptr() as *mut u64, n_u64)
    });
    let download_us = t0.elapsed().as_secs_f64() * 1e6;

    let repack_us = 0.0; // eliminated

    let total_us = wall_start.elapsed().as_secs_f64() * 1e6;

    let timing = PipelineTiming {
        n, flatten_us, upload_us, alloc_us, kernel_us,
        download_us, repack_us, inverse_us: 0.0, total_us,
    };

    (results, timing)
}

#[allow(dead_code)]
fn batch_inverse_fp(values: &[super::stark252_field::Fp]) -> Vec<super::stark252_field::Fp> {
    use super::stark252_field::Fp;

    let n = values.len();
    if n == 0 { return vec![]; }

    // Prefix products
    let mut prefix = Vec::with_capacity(n);
    prefix.push(values[0]);
    for i in 1..n {
        prefix.push(prefix[i-1] * values[i]);
    }

    // Invert the total product
    let mut inv = prefix[n-1].inverse();

    // Unwind
    let mut result = vec![Fp::ZERO; n];
    for i in (1..n).rev() {
        result[i] = inv * prefix[i-1]; // inv(values[i]) = inv * prefix[i-1]
        inv = inv * values[i];         // update inv for next
    }
    result[0] = inv;

    result
}

pub const PEDERSEN_BUILTIN_BASE: u64 = 0x5000_0000;

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    let hex = if hex.len() % 2 == 1 {
        format!("0{hex}")
    } else {
        hex.to_string()
    };
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap_or(0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cairo_air::stark252_field::{Fp, pedersen_hash};

    #[test]
    fn test_pedersen_gpu_vs_cpu_random_vectors() {
        crate::cuda::ffi::init_memory_pool();
        gpu_init();

        let n = 10_000;

        // Deterministic "random" inputs via simple hash-like mixing
        let inputs_a: Vec<Fp> = (0..n).map(|i| {
            let seed = (i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x517CC1B727220A95);
            Fp::from_u64(seed)
        }).collect();

        let inputs_b: Vec<Fp> = (0..n).map(|i| {
            let seed = (i as u64).wrapping_mul(0x6C62272E07BB0142).wrapping_add(0x62B821756295C58D);
            Fp::from_u64(seed)
        }).collect();

        // GPU batch
        let gpu_results = gpu_hash_batch(&inputs_a, &inputs_b);

        // CPU one-by-one
        let cpu_results: Vec<Fp> = inputs_a.iter().zip(&inputs_b)
            .map(|(&a, &b)| pedersen_hash(a, b))
            .collect();

        // Byte-for-byte equality
        let mut mismatches = 0;
        for i in 0..n {
            if gpu_results[i] != cpu_results[i] {
                if mismatches < 3 {
                    eprintln!("MISMATCH at index {i}:");
                    eprintln!("  CPU: [{:016x}, {:016x}, {:016x}, {:016x}]",
                        cpu_results[i].v[0], cpu_results[i].v[1], cpu_results[i].v[2], cpu_results[i].v[3]);
                    eprintln!("  GPU: [{:016x}, {:016x}, {:016x}, {:016x}]",
                        gpu_results[i].v[0], gpu_results[i].v[1], gpu_results[i].v[2], gpu_results[i].v[3]);
                }
                mismatches += 1;
            }
        }

        assert_eq!(mismatches, 0,
            "{mismatches}/{n} Pedersen hashes differ between GPU and CPU");
    }

    #[test]
    fn test_pedersen_gpu_trace_vs_cpu() {
        crate::cuda::ffi::init_memory_pool();
        gpu_init();

        let n = 1000;
        let log_n = 10; // trace_len = 1024

        // Deterministic inputs
        let inputs_a: Vec<Fp> = (0..n).map(|i| {
            Fp::from_u64((i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x517CC1B727220A95))
        }).collect();
        let inputs_b: Vec<Fp> = (0..n).map(|i| {
            Fp::from_u64((i as u64).wrapping_mul(0x6C62272E07BB0142).wrapping_add(0x62B821756295C58D))
        }).collect();

        // GPU: fused hash + trace columns (never leaves GPU)
        let gpu_cols = gpu_pedersen_trace(&inputs_a, &inputs_b, log_n);
        assert_eq!(gpu_cols.len(), 27);

        // Download GPU columns to host for comparison
        let gpu_host: Vec<Vec<u32>> = gpu_cols.iter().map(|c| c.to_host()).collect();

        // CPU: hash via PedersenBuiltin, then generate_trace
        let mut builtin = PedersenBuiltin::new();
        for i in 0..n {
            let a = fp_to_stark252(&inputs_a[i]);
            let b = fp_to_stark252(&inputs_b[i]);
            builtin.invoke(a, b);
        }
        let cpu_cols = builtin.generate_trace(log_n);

        // Compare all 27 columns, all rows
        let mut mismatches = 0;
        for col in 0..27 {
            for row in 0..n {
                if gpu_host[col][row] != cpu_cols[col][row] {
                    if mismatches < 5 {
                        eprintln!("MISMATCH col={col} row={row}: GPU={} CPU={}",
                            gpu_host[col][row], cpu_cols[col][row]);
                    }
                    mismatches += 1;
                }
            }
            // Verify padding is zero
            for row in n..(1 << log_n) {
                if gpu_host[col][row] != 0 {
                    if mismatches < 5 {
                        eprintln!("NON-ZERO PADDING col={col} row={row}: {}", gpu_host[col][row]);
                    }
                    mismatches += 1;
                }
            }
        }
        assert_eq!(mismatches, 0,
            "{mismatches} trace column mismatches between GPU and CPU");
    }

    #[test]
    fn test_stark252_from_u64() {
        let val = Stark252::from_u64(42);
        assert_eq!(val.limbs[0], 42);
        assert_eq!(val.limbs[1], 0);
    }

    #[test]
    fn test_stark252_from_hex() {
        let val = Stark252::from_hex("ff");
        assert_eq!(val.limbs[0], 0xFF);
        assert_eq!(val.limbs[1], 0);
    }

    #[test]
    fn test_stark252_large() {
        // 2^31 should be in limb[1]
        let val = Stark252::from_hex("80000000"); // 2^31
        assert_eq!(val.limbs[0], 0);
        assert_eq!(val.limbs[1], 1);
    }

    #[test]
    fn test_pedersen_points_loaded() {
        let points = pedersen_points();
        // P0.x should be non-zero
        assert_ne!(points[0].x, Stark252::ZERO);
        assert_ne!(points[0].y, Stark252::ZERO);
    }

    #[test]
    fn test_pedersen_builtin_invoke() {
        let mut builtin = PedersenBuiltin::new();
        let a = Stark252::from_u64(42);
        let b = Stark252::from_u64(99);
        let out = builtin.invoke(a, b);

        assert_eq!(builtin.n_invocations(), 1);
        assert_ne!(out, Stark252::ZERO);

        // Deterministic
        let mut builtin2 = PedersenBuiltin::new();
        let out2 = builtin2.invoke(a, b);
        assert_eq!(out, out2);
    }

    #[test]
    fn test_pedersen_trace() {
        let mut builtin = PedersenBuiltin::new();
        builtin.invoke(Stark252::from_u64(1), Stark252::from_u64(2));
        builtin.invoke(Stark252::from_u64(3), Stark252::from_u64(4));

        let cols = builtin.generate_trace(2); // 4 rows
        assert_eq!(cols.len(), 3 * N_LIMBS); // 27 columns
        assert_eq!(cols[0].len(), 4);

        // First invocation: a.limbs[0] = 1
        assert_eq!(cols[0][0], 1);
        // Second invocation: a.limbs[0] = 3
        assert_eq!(cols[0][1], 3);
    }
}
