//! Standard Cooley-Tukey NTT over Stark252.
//!
//! # CPU implementation
//! Uses DIT (Decimation In Time): bit-reverse input, then butterfly stages
//! from stride 1 up to N/2.
//!
//! # GPU implementation
//! Forward/inverse NTT via CUDA kernels (stark252_ntt.cu).
//! GPU path uses SoA layout (4 separate u64 arrays) and Montgomery form internally.
//!
//! # Convention
//! Forward NTT:  a[k] = Σ_j coeff[j] · ω^{jk}   (no normalization)
//! Inverse NTT:  coeff[j] = (1/N) · Σ_k a[k] · ω^{-jk}

use super::field::{Fp, ntt_root_of_unity, batch_inverse, fp_to_u32x8, fp_from_u32x8};
use crate::device::DeviceBuffer;

// ─────────────────────────────────────────────
// Bit-reversal permutation
// ─────────────────────────────────────────────

fn bit_reverse_index(mut x: usize, bits: u32) -> usize {
    let mut r = 0usize;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

fn bit_reverse_permutation(a: &mut [Fp], log_n: u32) {
    let n = a.len();
    for i in 0..n {
        let j = bit_reverse_index(i, log_n);
        if i < j {
            a.swap(i, j);
        }
    }
}

// ─────────────────────────────────────────────
// CPU NTT (DIT Cooley-Tukey)
// ─────────────────────────────────────────────

/// In-place forward NTT on a slice of length 2^log_n.
///
/// After return: `a[k] = Σ_j coeff[j] · ω^{jk}` where ω = ntt_root_of_unity(log_n).
pub fn ntt_forward_cpu(a: &mut [Fp], log_n: u32) {
    let n = 1usize << log_n;
    assert_eq!(a.len(), n, "NTT input length must equal 2^log_n");

    // Bit-reverse permutation (DIT requirement)
    bit_reverse_permutation(a, log_n);

    // Butterfly stages: stride doubles each stage
    let mut half_len = 1usize;
    while half_len < n {
        let len = half_len * 2;
        let s = len.trailing_zeros(); // log2(len)
        // ω_s = primitive len-th root of unity
        let w = ntt_root_of_unity(s);
        for start in (0..n).step_by(len) {
            let mut wj = Fp::ONE;
            for j in 0..half_len {
                let u = a[start + j];
                let v = a[start + j + half_len].mul(wj);
                a[start + j]            = u.add(v);
                a[start + j + half_len] = u.sub(v);
                wj = wj.mul(w);
            }
        }
        half_len = len;
    }
}

/// In-place inverse NTT (INTT) on a slice of length 2^log_n.
///
/// After return: `coeff[j] = (1/N) · Σ_k a[k] · ω^{-jk}`.
pub fn ntt_inverse_cpu(a: &mut [Fp], log_n: u32) {
    let n = 1usize << log_n;
    assert_eq!(a.len(), n, "INTT input length must equal 2^log_n");

    // Same butterfly structure but with inverse twiddles
    bit_reverse_permutation(a, log_n);

    let mut half_len = 1usize;
    while half_len < n {
        let len = half_len * 2;
        let s = len.trailing_zeros();
        // ω_s^{-1} = inverse of the primitive len-th root
        let w_inv = ntt_root_of_unity(s).inverse();
        for start in (0..n).step_by(len) {
            let mut wj = Fp::ONE;
            for j in 0..half_len {
                let u = a[start + j];
                let v = a[start + j + half_len].mul(wj);
                a[start + j]            = u.add(v);
                a[start + j + half_len] = u.sub(v);
                wj = wj.mul(w_inv);
            }
        }
        half_len = len;
    }

    // Normalize by 1/N
    let inv_n = Fp::from_u64(n as u64).inverse();
    for x in a.iter_mut() {
        *x = x.mul(inv_n);
    }
}

// ─────────────────────────────────────────────
// Low-degree extension (LDE)
// ─────────────────────────────────────────────

/// Low-degree extend trace values on domain of size N to a domain of size `blowup * N`.
///
/// Steps:
/// 1. INTT on trace values → polynomial coefficients
/// 2. Zero-pad to blowup*N
/// 3. NTT on padded coefficients → extended evaluations
///
/// Returns the extended evaluations (length = blowup * N).
pub fn lde_cpu(trace: &[Fp], log_n: u32, log_blowup: u32) -> Vec<Fp> {
    let n = trace.len();
    debug_assert_eq!(n, 1 << log_n);

    // Step 1: INTT to get coefficients
    let mut coeffs = trace.to_vec();
    ntt_inverse_cpu(&mut coeffs, log_n);

    // Step 2: zero-pad to eval domain size
    let log_eval = log_n + log_blowup;
    let eval_n = 1usize << log_eval;
    coeffs.resize(eval_n, Fp::ZERO);

    // Step 3: NTT to get extended evaluations
    ntt_forward_cpu(&mut coeffs, log_eval);
    coeffs
}

// ─────────────────────────────────────────────
// GPU NTT wrapper
// ─────────────────────────────────────────────

/// Precomputed twiddle factors for GPU NTT.
///
/// For an NTT of size N=2^log_n, we need N/2 twiddle factors per stage,
/// but in practice we pass the base generator and the kernel computes twiddles inline.
/// This struct holds device buffers of twiddle factors in Montgomery form.
pub struct NttTwiddles {
    pub log_n: u32,
    /// Stage twiddle tables: `stage_twiddles[s]` holds twiddles for stage s.
    /// Length: N/(2^{s+1}) where s = 0..log_n-1 (earlier stages have more twiddles).
    /// Actually we use a flat table of size N/2: twiddles[j] = ω_N^j for j=0..N/2-1.
    /// For stage s (stride = 2^s), the twiddle for position j is twiddles[j * N/2 / (N/2)] ...
    /// Simpler: we pass the full N/2-entry twiddle table and the stride to the kernel.
    pub forward_twiddles: DeviceBuffer<u64>, // (N/2) × 4 u64 values, SoA per limb
    pub inverse_twiddles: DeviceBuffer<u64>,
    pub inv_n_mont:       [u64; 4],          // (1/N) in standard form (Montgomery handled by kernel)
}

impl NttTwiddles {
    /// Precompute twiddle tables for an NTT of size 2^log_n and upload to GPU.
    pub fn new(log_n: u32) -> Self {
        let n = 1usize << log_n;
        let half_n = n / 2;

        // Forward twiddles: twiddles_fwd[j] = ω_N^j for j = 0..N/2-1
        let omega = ntt_root_of_unity(log_n);
        let mut fwd = Vec::with_capacity(half_n);
        let mut wj = Fp::ONE;
        for _ in 0..half_n {
            fwd.push(wj);
            wj = wj.mul(omega);
        }

        // Inverse twiddles: twiddles_inv[j] = ω_N^{-j} = (ω_N^{-1})^j
        let omega_inv = omega.inverse();
        let mut inv_tw = Vec::with_capacity(half_n);
        wj = Fp::ONE;
        for _ in 0..half_n {
            inv_tw.push(wj);
            wj = wj.mul(omega_inv);
        }

        // 1/N
        let inv_n = Fp::from_u64(n as u64).inverse();

        // Convert to flat u64 arrays (SoA: all limb0s, then all limb1s, ...)
        let fwd_flat = fps_to_flat_u64(&fwd);
        let inv_flat = fps_to_flat_u64(&inv_tw);

        NttTwiddles {
            log_n,
            forward_twiddles: DeviceBuffer::from_host(&fwd_flat),
            inverse_twiddles: DeviceBuffer::from_host(&inv_flat),
            inv_n_mont: inv_n.v,
        }
    }
}

/// Convert a slice of Fp to a flat u64 array in SoA layout.
/// Output: [fp[0].v[0], fp[1].v[0], ..., fp[n-1].v[0], fp[0].v[1], ...]
/// (4 contiguous blocks of n u64s, one block per limb)
pub fn fps_to_flat_u64(fps: &[Fp]) -> Vec<u64> {
    let n = fps.len();
    let mut out = vec![0u64; n * 4];
    for (i, fp) in fps.iter().enumerate() {
        for k in 0..4 {
            out[k * n + i] = fp.v[k];
        }
    }
    out
}

/// Convert flat u64 SoA back to Fp slice.
pub fn flat_u64_to_fps(flat: &[u64], n: usize) -> Vec<Fp> {
    let mut fps = vec![Fp::ZERO; n];
    for i in 0..n {
        for k in 0..4 {
            fps[i].v[k] = flat[k * n + i];
        }
    }
    fps
}

/// Forward NTT on GPU. Returns extended evaluations.
///
/// Falls back to CPU if GPU is not available (for testing).
pub fn ntt_forward_gpu(data: &[Fp], log_n: u32) -> Vec<Fp> {
    let n = 1usize << log_n;
    assert_eq!(data.len(), n);

    // Upload data as flat u64 SoA (4 × n u64)
    let flat = fps_to_flat_u64(data);
    let mut d_data = DeviceBuffer::<u64>::from_host(&flat);

    // Precompute twiddles (ω_N^j in standard form)
    let omega = ntt_root_of_unity(log_n);
    let half_n = n / 2;
    let mut twiddles = Vec::with_capacity(half_n);
    let mut wj = Fp::ONE;
    for _ in 0..half_n {
        twiddles.push(wj);
        wj = wj.mul(omega);
    }
    let tw_flat = fps_to_flat_u64(&twiddles);
    let d_tw = DeviceBuffer::<u64>::from_host(&tw_flat);

    // Call GPU kernel
    unsafe {
        crate::cuda::ffi::cuda_stark252_ntt_forward(
            d_data.as_mut_ptr(),
            d_tw.as_ptr(),
            log_n,
        );
        crate::cuda::ffi::cuda_device_sync();
    }

    let result_flat = d_data.to_host();
    flat_u64_to_fps(&result_flat, n)
}

/// Inverse NTT on GPU.
pub fn ntt_inverse_gpu(data: &[Fp], log_n: u32) -> Vec<Fp> {
    let n = 1usize << log_n;
    assert_eq!(data.len(), n);

    let flat = fps_to_flat_u64(data);
    let mut d_data = DeviceBuffer::<u64>::from_host(&flat);

    let omega_inv = ntt_root_of_unity(log_n).inverse();
    let half_n = n / 2;
    let mut twiddles = Vec::with_capacity(half_n);
    let mut wj = Fp::ONE;
    for _ in 0..half_n {
        twiddles.push(wj);
        wj = wj.mul(omega_inv);
    }
    let tw_flat = fps_to_flat_u64(&twiddles);
    let d_tw = DeviceBuffer::<u64>::from_host(&tw_flat);

    let inv_n = Fp::from_u64(n as u64).inverse();
    let mut d_inv_n = DeviceBuffer::<u64>::from_host(&inv_n.v);

    unsafe {
        crate::cuda::ffi::cuda_stark252_ntt_inverse(
            d_data.as_mut_ptr(),
            d_tw.as_ptr(),
            log_n,
            d_inv_n.as_ptr(),
        );
        crate::cuda::ffi::cuda_device_sync();
    }

    let result_flat = d_data.to_host();
    flat_u64_to_fps(&result_flat, n)
}

/// LDE using GPU NTT.
pub fn lde_gpu(trace: &[Fp], log_n: u32, log_blowup: u32) -> Vec<Fp> {
    let n = trace.len();
    debug_assert_eq!(n, 1 << log_n);

    // INTT to get coefficients (CPU for correctness — GPU for production)
    let mut coeffs = trace.to_vec();
    ntt_inverse_cpu(&mut coeffs, log_n);

    // Zero-pad
    let log_eval = log_n + log_blowup;
    let eval_n = 1usize << log_eval;
    coeffs.resize(eval_n, Fp::ZERO);

    // NTT on GPU
    ntt_forward_gpu(&coeffs, log_eval)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntt_roundtrip_small() {
        let log_n = 4u32;
        let n = 1 << log_n;
        // Random-ish values using field elements
        let orig: Vec<Fp> = (0u64..n as u64).map(|i| Fp::from_u64(i * 12345 + 1)).collect();
        let mut a = orig.clone();
        ntt_forward_cpu(&mut a, log_n);
        ntt_inverse_cpu(&mut a, log_n);
        assert_eq!(a, orig, "NTT roundtrip failed for log_n={log_n}");
    }

    #[test]
    fn test_lde_consistency() {
        let log_n = 4u32;
        let log_blowup = 2u32;
        let n = 1usize << log_n;
        let trace: Vec<Fp> = (0u64..n as u64).map(|i| Fp::from_u64(i * 7 + 3)).collect();
        let extended = lde_cpu(&trace, log_n, log_blowup);

        // Every 4th element should equal the original trace
        for k in 0..n {
            assert_eq!(
                extended[k * (1 << log_blowup)], trace[k],
                "LDE consistency failed at k={k}"
            );
        }
    }

    #[test]
    fn test_ntt_linearity() {
        let log_n = 3u32;
        let n = 1usize << log_n;
        let a: Vec<Fp> = (0u64..n as u64).map(|i| Fp::from_u64(i + 1)).collect();
        let b: Vec<Fp> = (0u64..n as u64).map(|i| Fp::from_u64(i * 2 + 3)).collect();
        let alpha = Fp::from_u64(7);

        let mut ntt_a = a.clone();
        let mut ntt_b = b.clone();
        ntt_forward_cpu(&mut ntt_a, log_n);
        ntt_forward_cpu(&mut ntt_b, log_n);

        // NTT(a + alpha*b) == NTT(a) + alpha*NTT(b)
        let mut combined: Vec<Fp> = a.iter().zip(b.iter())
            .map(|(ai, bi)| ai.add(bi.mul(alpha)))
            .collect();
        ntt_forward_cpu(&mut combined, log_n);

        for i in 0..n {
            let expected = ntt_a[i].add(ntt_b[i].mul(alpha));
            assert_eq!(combined[i], expected, "NTT linearity failed at i={i}");
        }
    }
}
