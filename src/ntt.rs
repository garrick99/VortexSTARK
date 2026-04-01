//! GPU Circle NTT: evaluate (coefficients → values) and interpolate (values → coefficients).
//!
//! Wraps the CUDA circle_ntt kernels with twiddle factor management.

use crate::circle::{compute_both_twiddles_gpu, compute_forward_twiddles_gpu, compute_inverse_twiddles_gpu, Coset};
use crate::cuda::ffi;
use crate::device::DeviceBuffer;

/// Cached twiddle factors for a given domain size (on GPU).
pub struct TwiddleCache {
    pub log_n: u32,
    // Forward
    pub d_twiddles: DeviceBuffer<u32>,
    pub d_circle_twids: DeviceBuffer<u32>,
    pub layer_offsets: Vec<u32>,
    pub layer_sizes: Vec<u32>,
    // Inverse
    pub d_itwiddles: DeviceBuffer<u32>,
    pub d_circle_itwids: DeviceBuffer<u32>,
    pub ilayer_offsets: Vec<u32>,
    pub ilayer_sizes: Vec<u32>,
}

impl TwiddleCache {
    /// Build and upload twiddle factors for a coset of given log_size.
    pub fn new(coset: &Coset) -> Self {
        let (d_twiddles, d_circle_twids, d_itwiddles, d_circle_itwids, offsets, sizes) =
            compute_both_twiddles_gpu(coset);

        Self {
            log_n: coset.log_size,
            d_twiddles,
            d_circle_twids,
            layer_offsets: offsets.clone(),
            layer_sizes: sizes.clone(),
            d_itwiddles,
            d_circle_itwids,
            ilayer_offsets: offsets,
            ilayer_sizes: sizes,
        }
    }
}

/// Forward-only twiddle cache (smaller VRAM footprint).
pub struct ForwardTwiddleCache {
    pub log_n: u32,
    pub d_twiddles: DeviceBuffer<u32>,
    pub d_circle_twids: DeviceBuffer<u32>,
    pub layer_offsets: Vec<u32>,
    pub layer_sizes: Vec<u32>,
}

impl ForwardTwiddleCache {
    /// Build forward twiddle factors only for a coset.
    pub fn new(coset: &Coset) -> Self {
        let (d_twiddles, d_circle_twids, layer_offsets, layer_sizes) =
            compute_forward_twiddles_gpu(coset);
        Self {
            log_n: coset.log_size,
            d_twiddles,
            d_circle_twids,
            layer_offsets,
            layer_sizes,
        }
    }
}

/// Inverse-only twiddle cache (smaller VRAM footprint).
pub struct InverseTwiddleCache {
    pub log_n: u32,
    pub d_itwiddles: DeviceBuffer<u32>,
    pub d_circle_itwids: DeviceBuffer<u32>,
    pub layer_offsets: Vec<u32>,
    pub layer_sizes: Vec<u32>,
}

impl InverseTwiddleCache {
    /// Build inverse twiddle factors only for a coset.
    /// Internally computes forward, inverts, and drops forward.
    pub fn new(coset: &Coset) -> Self {
        let (d_itwiddles, d_circle_itwids, layer_offsets, layer_sizes) =
            compute_inverse_twiddles_gpu(coset);
        Self {
            log_n: coset.log_size,
            d_itwiddles,
            d_circle_itwids,
            layer_offsets,
            layer_sizes,
        }
    }
}

/// Trait for types that provide forward twiddle data for NTT evaluate.
pub trait ForwardTwiddles {
    fn log_n(&self) -> u32;
    fn twiddles_ptr(&self) -> *const u32;
    fn circle_twids_ptr(&self) -> *const u32;
    fn layer_offsets(&self) -> &[u32];
    fn layer_sizes(&self) -> &[u32];
}

impl ForwardTwiddles for TwiddleCache {
    fn log_n(&self) -> u32 { self.log_n }
    fn twiddles_ptr(&self) -> *const u32 { self.d_twiddles.as_ptr() }
    fn circle_twids_ptr(&self) -> *const u32 { self.d_circle_twids.as_ptr() }
    fn layer_offsets(&self) -> &[u32] { &self.layer_offsets }
    fn layer_sizes(&self) -> &[u32] { &self.layer_sizes }
}

impl ForwardTwiddles for ForwardTwiddleCache {
    fn log_n(&self) -> u32 { self.log_n }
    fn twiddles_ptr(&self) -> *const u32 { self.d_twiddles.as_ptr() }
    fn circle_twids_ptr(&self) -> *const u32 { self.d_circle_twids.as_ptr() }
    fn layer_offsets(&self) -> &[u32] { &self.layer_offsets }
    fn layer_sizes(&self) -> &[u32] { &self.layer_sizes }
}

/// Trait for types that provide inverse twiddle data for NTT interpolate.
pub trait InverseTwiddles {
    fn log_n(&self) -> u32;
    fn itwiddles_ptr(&self) -> *const u32;
    fn circle_itwids_ptr(&self) -> *const u32;
    fn ilayer_offsets(&self) -> &[u32];
    fn ilayer_sizes(&self) -> &[u32];
}

impl InverseTwiddles for TwiddleCache {
    fn log_n(&self) -> u32 { self.log_n }
    fn itwiddles_ptr(&self) -> *const u32 { self.d_itwiddles.as_ptr() }
    fn circle_itwids_ptr(&self) -> *const u32 { self.d_circle_itwids.as_ptr() }
    fn ilayer_offsets(&self) -> &[u32] { &self.ilayer_offsets }
    fn ilayer_sizes(&self) -> &[u32] { &self.ilayer_sizes }
}

impl InverseTwiddles for InverseTwiddleCache {
    fn log_n(&self) -> u32 { self.log_n }
    fn itwiddles_ptr(&self) -> *const u32 { self.d_itwiddles.as_ptr() }
    fn circle_itwids_ptr(&self) -> *const u32 { self.d_circle_itwids.as_ptr() }
    fn ilayer_offsets(&self) -> &[u32] { &self.layer_offsets }
    fn ilayer_sizes(&self) -> &[u32] { &self.layer_sizes }
}

/// Forward NTT: coefficients → evaluation values (in-place on GPU).
/// `d_data` must contain `n = 2^log_n` M31 elements on the device.
pub fn evaluate(d_data: &mut DeviceBuffer<u32>, cache: &impl ForwardTwiddles) {
    let log_n = cache.log_n();
    let n = 1u32 << log_n;
    let n_line_layers = if log_n > 0 { log_n - 1 } else { 0 };

    unsafe {
        ffi::cuda_circle_ntt_evaluate(
            d_data.as_mut_ptr(),
            cache.twiddles_ptr(),
            cache.circle_twids_ptr(),
            cache.layer_offsets().as_ptr(),
            cache.layer_sizes().as_ptr(),
            n_line_layers,
            n,
        );
    }
}

/// Inverse NTT: evaluation values → coefficients (in-place on GPU).
pub fn interpolate(d_data: &mut DeviceBuffer<u32>, cache: &impl InverseTwiddles) {
    let log_n = cache.log_n();
    let n = 1u32 << log_n;
    let n_line_layers = if log_n > 0 { log_n - 1 } else { 0 };

    unsafe {
        ffi::cuda_circle_ntt_interpolate(
            d_data.as_mut_ptr(),
            cache.itwiddles_ptr(),
            cache.circle_itwids_ptr(),
            cache.ilayer_offsets().as_ptr(),
            cache.ilayer_sizes().as_ptr(),
            n_line_layers,
            n,
        );
    }
}

/// Bit-reverse permutation on GPU (in-place).
pub fn bit_reverse(d_data: &mut DeviceBuffer<u32>, log_n: u32) {
    unsafe {
        ffi::cuda_bit_reverse_m31(d_data.as_mut_ptr(), log_n);
        ffi::cuda_device_sync();
    }
}

/// Forward NTT on multiple columns simultaneously.
pub fn evaluate_batch(columns: &mut [DeviceBuffer<u32>], cache: &impl ForwardTwiddles) {
    if columns.is_empty() {
        return;
    }
    let log_n = cache.log_n();
    let n = 1u32 << log_n;
    let n_cols = columns.len() as u32;
    let n_line_layers = if log_n > 0 { log_n - 1 } else { 0 };

    // Build array of device pointers
    let ptrs: Vec<*mut u32> = columns.iter_mut().map(|c| c.as_mut_ptr()).collect();
    let d_ptrs = DeviceBuffer::from_host(&ptrs);

    unsafe {
        ffi::cuda_circle_ntt_evaluate_batch(
            d_ptrs.as_ptr() as *mut *mut u32,
            cache.twiddles_ptr(),
            cache.circle_twids_ptr(),
            cache.layer_offsets().as_ptr(),
            cache.layer_sizes().as_ptr(),
            n_line_layers,
            n,
            n_cols,
        );
    }
}

/// Inverse NTT on multiple columns simultaneously.
pub fn interpolate_batch(columns: &mut [DeviceBuffer<u32>], cache: &impl InverseTwiddles) {
    if columns.is_empty() {
        return;
    }
    let log_n = cache.log_n();
    let n = 1u32 << log_n;
    let n_cols = columns.len() as u32;
    let n_line_layers = if log_n > 0 { log_n - 1 } else { 0 };

    let ptrs: Vec<*mut u32> = columns.iter_mut().map(|c| c.as_mut_ptr()).collect();
    let d_ptrs = DeviceBuffer::from_host(&ptrs);

    unsafe {
        ffi::cuda_circle_ntt_interpolate_batch(
            d_ptrs.as_ptr() as *mut *mut u32,
            cache.itwiddles_ptr(),
            cache.circle_itwids_ptr(),
            cache.ilayer_offsets().as_ptr(),
            cache.ilayer_sizes().as_ptr(),
            n_line_layers,
            n,
            n_cols,
        );
    }
}

// ── Stwo-compatible NTT (different polynomial basis from old NTT) ────────────

/// Stwo NTT twiddle cache. Uses `cuda_stwo_ntt_evaluate` / `cuda_stwo_ntt_interpolate`.
/// The polynomial basis matches `eval_at_oods_from_coeffs`.
pub struct StwoNttCache {
    pub log_n: u32,
    pub d_fwd_twiddles: DeviceBuffer<u32>,
    pub d_inv_twiddles: DeviceBuffer<u32>,
}

impl StwoNttCache {
    /// Build stwo-format twiddle factors for the given domain coset.
    /// The coset should be `half_coset(log_n)`.
    /// Twiddles are computed from `half_odds(log_n - 1)` (stwo convention).
    ///
    /// Matches stwo's `slow_precompute_twiddles`: for each level, store x-coordinates
    /// of the first half of the coset in BRT order, then double the coset.
    pub fn new(coset: &Coset) -> Self {
        use crate::field::M31;
        use crate::circle::CirclePoint;

        let log_n = coset.log_size;
        let mut twiddle_coset = Coset::half_odds(log_n - 1);
        let tc_size = twiddle_coset.size(); // 2^(log_n-1)

        // Build flat twiddle buffer matching stwo's slow_precompute_twiddles:
        // For each level: store x-coordinates of first half of coset, BRT-permuted.
        // Then double the coset and repeat.
        let mut fwd = Vec::with_capacity(tc_size);
        let mut current = twiddle_coset;
        for _ in 0..current.log_size {
            let half = current.size() / 2;
            let layer_log = half.trailing_zeros();
            // Collect x-coordinates of first half of coset in natural order.
            let mut xs = vec![0u32; half];
            let mut pt = current.initial;
            for j in 0..half {
                xs[j] = pt.x.0;
                pt = pt.mul(current.step);
            }
            // BRT-permute within this layer.
            let mut brt_xs = vec![0u32; half];
            for i in 0..half {
                let brt_i = if layer_log == 0 { 0 } else { i.reverse_bits() >> (usize::BITS - layer_log) };
                brt_xs[brt_i] = xs[i];
            }
            fwd.extend_from_slice(&brt_xs);
            // Double the coset for next level.
            current = Coset {
                initial: current.initial.mul(current.initial),
                step: current.step.mul(current.step),
                log_size: current.log_size - 1,
            };
        }
        // Pad to tc_size.
        fwd.push(1);
        while fwd.len() < tc_size { fwd.push(0); }
        fwd.truncate(tc_size);

        // Inverse twiddles: batch-invert using Montgomery's trick.
        let mut inv = fwd.clone();
        let mut prefix = vec![M31::ONE; tc_size];
        prefix[0] = M31(inv[0]);
        for i in 1..tc_size {
            prefix[i] = prefix[i-1] * M31(inv[i]);
        }
        let mut inv_prod = prefix[tc_size - 1].inverse();
        for i in (1..tc_size).rev() {
            let inv_i = inv_prod * prefix[i-1];
            inv_prod = inv_prod * M31(inv[i]);
            inv[i] = inv_i.0;
        }
        inv[0] = inv_prod.0;

        Self {
            log_n,
            d_fwd_twiddles: DeviceBuffer::from_host(&fwd),
            d_inv_twiddles: DeviceBuffer::from_host(&inv),
        }
    }
}

/// Forward NTT using stwo-compatible kernel.
pub fn evaluate_stwo(d_data: &mut DeviceBuffer<u32>, cache: &StwoNttCache) {
    let n = 1u32 << cache.log_n;
    unsafe {
        ffi::cuda_stwo_ntt_evaluate(d_data.as_mut_ptr(), cache.d_fwd_twiddles.as_ptr(), n);
    }
}

/// Inverse NTT using stwo-compatible kernel.
pub fn interpolate_stwo(d_data: &mut DeviceBuffer<u32>, cache: &StwoNttCache) {
    let n = 1u32 << cache.log_n;
    unsafe {
        ffi::cuda_stwo_ntt_interpolate(d_data.as_mut_ptr(), cache.d_inv_twiddles.as_ptr(), n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::M31;

    #[test]
    fn test_ntt_roundtrip() {
        // NTT(iNTT(x)) = x
        let log_n = 10u32;
        let n = 1usize << log_n;
        let coset = Coset::half_coset(log_n);
        let cache = TwiddleCache::new(&coset);

        // Random-ish coefficients
        let coeffs: Vec<u32> = (0..n).map(|i| ((i * 7 + 13) % (M31::ONE.0 as usize)) as u32).collect();
        let mut d_data = DeviceBuffer::from_host(&coeffs);

        // Forward then inverse should be identity
        evaluate(&mut d_data, &cache);
        interpolate(&mut d_data, &cache);

        let result = d_data.to_host();
        assert_eq!(coeffs, result, "NTT roundtrip failed");
    }

    #[test]
    fn test_ntt_roundtrip_split_caches() {
        // Test with separate ForwardTwiddleCache / InverseTwiddleCache
        let log_n = 10u32;
        let n = 1usize << log_n;
        let coset = Coset::half_coset(log_n);
        let fwd = ForwardTwiddleCache::new(&coset);
        let inv = InverseTwiddleCache::new(&coset);

        let coeffs: Vec<u32> = (0..n).map(|i| ((i * 7 + 13) % (M31::ONE.0 as usize)) as u32).collect();
        let mut d_data = DeviceBuffer::from_host(&coeffs);

        evaluate(&mut d_data, &fwd);
        interpolate(&mut d_data, &inv);

        let result = d_data.to_host();
        assert_eq!(coeffs, result, "Split-cache NTT roundtrip failed");
    }

    #[test]
    fn test_ntt_roundtrip_batch() {
        let log_n = 8u32;
        let n = 1usize << log_n;
        let coset = Coset::half_coset(log_n);
        let cache = TwiddleCache::new(&coset);

        let n_cols = 4;
        let originals: Vec<Vec<u32>> = (0..n_cols)
            .map(|c| (0..n).map(|i| ((i * (c + 3) + 17) % (M31::ONE.0 as usize)) as u32).collect())
            .collect();

        let mut columns: Vec<DeviceBuffer<u32>> = originals
            .iter()
            .map(|v| DeviceBuffer::from_host(v))
            .collect();

        evaluate_batch(&mut columns, &cache);
        interpolate_batch(&mut columns, &cache);

        for (c, orig) in originals.iter().enumerate() {
            let result = columns[c].to_host();
            assert_eq!(orig, &result, "Batch NTT roundtrip failed for column {c}");
        }
    }

    /// Test: stwo forward NTT of known coefficients, then eval_at_oods at each domain point.
    #[test]
    /// Verify permute_half_coset_to_canonic + stwo INTT gives correct eval_at_oods
    /// at EVAL domain size (log_n=8, 256 points).
    #[test]
    fn test_permute_then_stwo_intt_eval_domain() {
        use crate::oods::{OodsPoint, eval_at_oods_from_coeffs, qm31_from_m31};
        crate::cuda::ffi::init_memory_pool();

        let log_n = 8u32;
        let n = 1usize << log_n;
        let coset = Coset::half_coset(log_n);
        let cache = StwoNttCache::new(&coset);

        // Create known trace data at half_coset natural positions.
        let trace: Vec<u32> = (0..n).map(|k| coset.at(k).x.0).collect();

        // Pipeline: permute → stwo INTT → eval_at_oods at first domain point.
        let canonic = Coset::permute_hc_natural_to_canonic_brt(&trace, log_n);
        let mut d = DeviceBuffer::from_host(&canonic);
        interpolate_stwo(&mut d, &cache);
        let stwo_coeffs = d.to_host();
        // First n/4 coefficients for a polynomial of degree < n (4x blowup would be n/4 for eval domain).
        // Actually for trace data with n values, ALL n coefficients matter.
        let f0 = eval_at_oods_from_coeffs(&stwo_coeffs, OodsPoint {
            x: qm31_from_m31(coset.at(0).x), y: qm31_from_m31(coset.at(0).y),
        });
        eprintln!("[perm_eval] f(hc.at(0)) = {:?}, expected = [{}, 0, 0, 0]",
            f0.to_u32_array(), trace[0]);
        let ok = f0.to_u32_array()[0] == trace[0] && f0.to_u32_array()[1] == 0;
        eprintln!("[perm_eval] match: {}", ok);

        // Check a few more.
        let mut correct = 0;
        for k in 0..n.min(16) {
            let pk = coset.at(k);
            let f = eval_at_oods_from_coeffs(&stwo_coeffs, OodsPoint {
                x: qm31_from_m31(pk.x), y: qm31_from_m31(pk.y),
            });
            if f.to_u32_array()[0] == trace[k] && f.to_u32_array()[1] == 0 { correct += 1; }
        }
        eprintln!("[perm_eval] correct at domain points: {}/{}", correct, n.min(16));

        // Check: are coefficients beyond n/4 (= 64 for blowup=4) zero?
        // For a polynomial of degree < n/4, coefficients [n/4..n] should be zero.
        let n4 = n / 4; // trace domain size
        let mut high_nonzero = 0;
        for i in n4..n {
            if stwo_coeffs[i] != 0 { high_nonzero += 1; }
        }
        eprintln!("[perm_eval] high coefficients (>{}) nonzero: {}/{}", n4, high_nonzero, n - n4);

        // Check: does truncation to n/4 still give correct eval?
        let f0_trunc = eval_at_oods_from_coeffs(&stwo_coeffs[..n4], OodsPoint {
            x: qm31_from_m31(coset.at(0).x), y: qm31_from_m31(coset.at(0).y),
        });
        let f0_full = eval_at_oods_from_coeffs(&stwo_coeffs, OodsPoint {
            x: qm31_from_m31(coset.at(0).x), y: qm31_from_m31(coset.at(0).y),
        });
        eprintln!("[perm_eval] f(p0) truncated to n/4: {:?}", f0_trunc.to_u32_array());
        eprintln!("[perm_eval] f(p0) full: {:?}", f0_full.to_u32_array());
        eprintln!("[perm_eval] truncation valid: {}", f0_trunc == f0_full);

        assert_eq!(correct, n.min(16), "permute→stwo INTT→eval_at_oods fails at eval domain size");
    }

    /// Test eval_at_oods consistency at EVAL DOMAIN size (log_n=8, 256 points).
    #[test]
    fn test_stwo_eval_at_oods_consistency_large() {
        use crate::oods::{OodsPoint, eval_at_oods_from_coeffs, qm31_from_m31};
        crate::cuda::ffi::init_memory_pool();

        let log_n = 8u32;
        let n = 1usize << log_n;
        let coset = Coset::half_coset(log_n);
        let cache = StwoNttCache::new(&coset);
        let ho = Coset::half_odds(log_n - 1);
        let half_n = n / 2;

        let mut coeffs = vec![0u32; n];
        coeffs[0] = 42; coeffs[1] = 17; coeffs[2] = 5; coeffs[3] = 99;

        let mut d_data = DeviceBuffer::from_host(&coeffs);
        evaluate_stwo(&mut d_data, &cache);
        let evals = d_data.to_host();

        // Check BRT canonic matching (should be n/n for correct twiddles).
        let mut brt_cn = 0;
        for k in 0..n {
            let brt_k = k.reverse_bits() >> (usize::BITS - log_n);
            let cn_pt = if brt_k < half_n { ho.at(brt_k) } else { ho.at(brt_k - half_n).conjugate() };
            let f_cn = eval_at_oods_from_coeffs(&coeffs, OodsPoint {
                x: qm31_from_m31(cn_pt.x), y: qm31_from_m31(cn_pt.y),
            });
            if f_cn.to_u32_array()[0] == evals[k] && f_cn.to_u32_array()[1] == 0 {
                brt_cn += 1;
            }
        }
        eprintln!("[stwo_large] BRT canonic: {}/{}", brt_cn, n);
        assert_eq!(brt_cn, n, "Stwo eval_at_oods fails at log_n=8 (eval domain size)");
    }

    #[test]
    fn test_stwo_eval_at_oods_consistency() {
        use crate::oods::{OodsPoint, eval_at_oods_from_coeffs, qm31_from_m31};
        crate::cuda::ffi::init_memory_pool();

        let log_n = 6u32;
        let n = 1usize << log_n;
        let coset = Coset::half_coset(log_n);
        let cache = StwoNttCache::new(&coset);

        // Known coefficients: first 4 nonzero, rest zero.
        let mut coeffs = vec![0u32; n];
        coeffs[0] = 42; coeffs[1] = 17; coeffs[2] = 5; coeffs[3] = 99;

        // Forward stwo NTT → evaluations.
        let mut d_data = DeviceBuffer::from_host(&coeffs);
        evaluate_stwo(&mut d_data, &cache);
        let evals = d_data.to_host();

        // eval_at_oods_from_coeffs(coeffs, point) should reproduce each evaluation.
        // But at WHICH domain point? Let's try every point of half_coset:
        let ho = Coset::half_odds(log_n - 1);
        let half_n = n / 2;
        let mut hc_match = 0;
        let mut canonic_match = 0;
        for k in 0..n {
            let f_k = eval_at_oods_from_coeffs(&coeffs, OodsPoint {
                x: qm31_from_m31(coset.at(k).x),
                y: qm31_from_m31(coset.at(k).y),
            });
            if f_k.to_u32_array()[0] == evals[k] && f_k.to_u32_array()[1] == 0 {
                hc_match += 1;
            }

            let cn_pt = if k < half_n { ho.at(k) } else { ho.at(k - half_n).conjugate() };
            let f_cn = eval_at_oods_from_coeffs(&coeffs, OodsPoint {
                x: qm31_from_m31(cn_pt.x),
                y: qm31_from_m31(cn_pt.y),
            });
            if f_cn.to_u32_array()[0] == evals[k] && f_cn.to_u32_array()[1] == 0 {
                canonic_match += 1;
            }
        }
        eprintln!("[stwo_eval] evals matched at half_coset points: {}/{}", hc_match, n);
        eprintln!("[stwo_eval] evals matched at canonic points: {}/{}", canonic_match, n);

        // If neither matches all n, the eval_at_oods basis doesn't match stwo NTT.
        // In that case, we need to understand the actual mapping.
        if hc_match < n && canonic_match < n {
            // Try BRT of each:
            let mut brt_hc = 0;
            let mut brt_cn = 0;
            for k in 0..n {
                let brt_k = k.reverse_bits() >> (usize::BITS - log_n);
                let f_hc = eval_at_oods_from_coeffs(&coeffs, OodsPoint {
                    x: qm31_from_m31(coset.at(brt_k).x),
                    y: qm31_from_m31(coset.at(brt_k).y),
                });
                if f_hc.to_u32_array()[0] == evals[k] && f_hc.to_u32_array()[1] == 0 {
                    brt_hc += 1;
                }
                let cn_pt = if brt_k < half_n { ho.at(brt_k) } else { ho.at(brt_k - half_n).conjugate() };
                let f_cn = eval_at_oods_from_coeffs(&coeffs, OodsPoint {
                    x: qm31_from_m31(cn_pt.x),
                    y: qm31_from_m31(cn_pt.y),
                });
                if f_cn.to_u32_array()[0] == evals[k] && f_cn.to_u32_array()[1] == 0 {
                    brt_cn += 1;
                }
            }
            eprintln!("[stwo_eval] BRT half_coset: {}/{}", brt_hc, n);
            eprintln!("[stwo_eval] BRT canonic: {}/{}", brt_cn, n);
        }
    }

    #[test]
    /// Determine the domain point at each old NTT output position, then build the
    /// correct permutation to stwo NTT ordering.
    #[test]
    #[ignore = "diagnostic: old NTT and stwo NTT use different polynomial bases"]
    fn test_old_to_stwo_index_mapping() {
        crate::cuda::ffi::init_memory_pool();
        let log_n = 6u32;
        let n = 1usize << log_n;
        let coset = Coset::half_coset(log_n);
        let old_fwd = ForwardTwiddleCache::new(&coset);
        let stwo_cache = StwoNttCache::new(&coset);

        let coeffs: Vec<u32> = (0..n).map(|i| ((i * 7 + 3) % (crate::field::m31::P as usize)) as u32).collect();

        let mut d_old = DeviceBuffer::from_host(&coeffs);
        evaluate(&mut d_old, &old_fwd);
        let old_evals = d_old.to_host();

        let mut d_stwo = DeviceBuffer::from_host(&coeffs);
        evaluate_stwo(&mut d_stwo, &stwo_cache);
        let stwo_evals = d_stwo.to_host();

        // Check: is old NTT output at position i the evaluation at half_coset.at(i) (natural)
        // or half_coset.at(BRT(i)) (BRT)?
        // Method: evaluate a polynomial with known values at known domain points.
        // Use f(p) = p.x (the x-coordinate projection). Then old_ntt[i] should be
        // either half_coset.at(i).x or half_coset.at(BRT(i)).x.
        {
            use crate::oods::{qm31_from_m31, eval_at_oods_from_coeffs, OodsPoint};
            // Create the polynomial f(p) = p.x by evaluating p.x at all half_coset points.
            // Represent it in old NTT coefficients.
            let x_vals: Vec<u32> = (0..n).map(|k| coset.at(k).x.0).collect();
            // Old INTT → old coefficients for "f(p) = p.x at half_coset domain"
            let inv_cache = InverseTwiddleCache::new(&coset);
            let mut d_x = DeviceBuffer::from_host(&x_vals);
            interpolate(&mut d_x, &inv_cache);
            let x_coeffs = d_x.to_host();
            // Old NTT → evaluations
            let mut d_xf = DeviceBuffer::from_host(&x_coeffs);
            evaluate(&mut d_xf, &old_fwd);
            let x_evals = d_xf.to_host();
            // Check: x_evals[k] == x_vals[k]? (Would mean natural order)
            let natural_match = (0..n).filter(|&k| x_evals[k] == x_vals[k]).count();
            let brt_match = (0..n).filter(|&k| {
                let brt_k = k.reverse_bits() >> (usize::BITS - log_n);
                x_evals[k] == x_vals[brt_k]
            }).count();
            eprintln!("[order] old NTT roundtrip natural match: {}/{}", natural_match, n);
            eprintln!("[order] old NTT roundtrip BRT match: {}/{}", brt_match, n);
            // Also check: x_evals[k] == coset.at(k).x (natural) or coset.at(BRT(k)).x (BRT)
            let nat_domain = (0..n).filter(|&k| x_evals[k] == coset.at(k).x.0).count();
            let brt_domain = (0..n).filter(|&k| {
                let brt_k = k.reverse_bits() >> (usize::BITS - log_n);
                x_evals[k] == coset.at(brt_k).x.0
            }).count();
            eprintln!("[order] old NTT position k = f(half_coset.at(k)) natural: {}/{}", nat_domain, n);
            eprintln!("[order] old NTT position k = f(half_coset.at(BRT(k))) BRT: {}/{}", brt_domain, n);

            // Now verify: permute x_vals (natural half_coset) → stwo order → stwo INTT → eval_at_oods
            let canonic_brt = Coset::permute_hc_natural_to_canonic_brt(&x_vals, log_n);
            let mut d_cn = DeviceBuffer::from_host(&canonic_brt);
            interpolate_stwo(&mut d_cn, &stwo_cache);
            let stwo_coeffs = d_cn.to_host();
            // eval_at_oods at domain point half_coset.at(0) should give x_vals[0]
            let p0 = coset.at(0);
            let f_p0 = eval_at_oods_from_coeffs(&stwo_coeffs, OodsPoint {
                x: qm31_from_m31(p0.x), y: qm31_from_m31(p0.y),
            });
            eprintln!("[order] eval_at_oods(stwo_coeffs, p0) = {:?}, expected = [{}, 0, 0, 0]",
                f_p0.to_u32_array(), x_vals[0]);
            let match_p0 = f_p0.to_u32_array()[0] == x_vals[0] && f_p0.to_u32_array()[1] == 0;
            eprintln!("[order] correct f(z) via permute→stwo INTT: {}", match_p0);

            // Check a few more
            let mut correct = 0;
            for k in 0..n.min(8) {
                let pk = coset.at(k);
                let f_pk = eval_at_oods_from_coeffs(&stwo_coeffs, OodsPoint {
                    x: qm31_from_m31(pk.x), y: qm31_from_m31(pk.y),
                });
                if f_pk.to_u32_array()[0] == x_vals[k] && f_pk.to_u32_array()[1] == 0 {
                    correct += 1;
                }
            }
            eprintln!("[order] eval_at_oods correct at domain points: {}/{}", correct, n.min(8));
        }

        // Direct approach: the old NTT roundtrip tells us that trace[k] = f(domain_point_k).
        // We know the domain points of half_coset. We need to find which half_coset
        // point maps to which stwo position.
        //
        // Method: create UNIQUE trace data, get old NTT evaluations.
        // Then reorder by checking: for each half_coset point p_k, find which stwo
        // position j has the same circle point (same x,y).
        //
        // half_coset.at(k) maps to canonic position sigma(k) where
        //   sigma(k) = k/2 (even k) or n-1-k/2 (odd k)
        //
        // But OLD NTT output position i might NOT correspond to half_coset.at(i).
        // Let me determine the mapping empirically.
        //
        // Use a constant polynomial: old_NTT([1, 0, 0, ...]) should give all 1s.
        let mut const_coeffs = vec![0u32; n];
        const_coeffs[0] = 1;
        let mut d_const = DeviceBuffer::from_host(&const_coeffs);
        evaluate(&mut d_const, &old_fwd);
        let old_const = d_const.to_host();
        let all_ones = old_const.iter().all(|&v| v == 1);
        eprintln!("[mapping] old NTT of [1,0,...]: all ones = {}", all_ones);

        // If the old NTT of a constant [1, 0, ...] gives all 1s, then the first
        // "coefficient" IS the constant term. Let me check what the old INTT
        // of all-1s trace gives:
        let trace_ones: Vec<u32> = vec![1; n];
        let mut d_ones = DeviceBuffer::from_host(&trace_ones);
        interpolate(&mut d_ones, &InverseTwiddleCache::new(&coset));
        let ones_coeffs = d_ones.to_host();
        eprintln!("[mapping] old INTT of all-1s: first 4 = {:?}", &ones_coeffs[..4]);
        // If first coeff is 1 and rest 0, the constant function is basis function 0.

        // The old NTT and stwo NTT use different polynomial bases.
        // Given the SAME coefficients, they produce different evaluations.
        // But they evaluate on the SAME set of domain points (just different ordering).
        //
        // To convert old NTT evaluations → stwo NTT evaluations:
        // 1. Determine which domain point each old NTT position represents
        // 2. Reorder to stwo domain ordering
        // 3. Stwo INTT → stwo-basis coefficients
        //
        // Method: evaluate a "probe" polynomial (the second basis function) via old NTT.
        // old_ntt[1, 0, 0, ...] gives the second basis function's values at each position.
        // These values are y-coordinates of the domain points (for circle NTT first basis = y).
        //
        // Use TWO probes (second and third basis functions = y, x after circle level)
        // to uniquely identify each domain point.

        // Probe 1: [0, 1, 0, ...]
        let mut probe_y = vec![0u32; n];
        probe_y[1] = 1;
        let mut d_py = DeviceBuffer::from_host(&probe_y);
        evaluate(&mut d_py, &old_fwd);
        let old_y = d_py.to_host();

        // Probe 2: [0, 0, 1, 0, ...]
        let mut probe_x = vec![0u32; n];
        probe_x[2] = 1;
        let mut d_px = DeviceBuffer::from_host(&probe_x);
        evaluate(&mut d_px, &old_fwd);
        let old_x = d_px.to_host();

        let ho = Coset::half_odds(log_n - 1);
        let half_n = n / 2;

        // Use ALL basis functions as probes for uniqueness.
        // Evaluate basis functions 0..5 via both NTTs to create fingerprints.
        let n_probes = 6;
        let mut old_probes: Vec<Vec<u32>> = Vec::new();
        let mut stwo_probes: Vec<Vec<u32>> = Vec::new();
        for b in 0..n_probes {
            let mut p = vec![0u32; n];
            p[b] = 1;
            let mut d1 = DeviceBuffer::from_host(&p);
            evaluate(&mut d1, &old_fwd);
            old_probes.push(d1.to_host());
            let mut d2 = DeviceBuffer::from_host(&p);
            evaluate_stwo(&mut d2, &stwo_cache);
            stwo_probes.push(d2.to_host());
        }

        // Match: position i in old → position j in stwo if ALL probes match
        let mut perm = vec![usize::MAX; n];
        let mut found = 0;
        for i in 0..n {
            for j in 0..n {
                let mut all_match = true;
                for b in 0..n_probes {
                    if old_probes[b][i] != stwo_probes[b][j] { all_match = false; break; }
                }
                if all_match {
                    perm[i] = j;
                    found += 1;
                    break;
                }
            }
        }
        eprintln!("[mapping] found {}/{}", found, n);
        eprintln!("[mapping] perm[0..16] = {:?}", &perm[..16.min(n)]);

        let mut used = vec![false; n];
        let mut valid = true;
        for &j in &perm {
            if j >= n || used[j] { valid = false; break; }
            used[j] = true;
        }
        eprintln!("[mapping] valid permutation: {}", valid);

        // Now verify: using this permutation to reorder old evaluations gives stwo evaluations
        if valid && found == n {
            let mut reordered = vec![0u32; n];
            for i in 0..n {
                reordered[perm[i]] = old_evals[i];
            }
            // Stwo INTT of reordered → stwo coefficients. Then verify with eval_at_oods.
            let mut d_reord = DeviceBuffer::from_host(&reordered);
            interpolate_stwo(&mut d_reord, &stwo_cache);
            let stwo_coeffs = d_reord.to_host();

            // Stwo forward NTT of stwo_coeffs should give stwo_evals
            let mut d_check = DeviceBuffer::from_host(&stwo_coeffs);
            evaluate_stwo(&mut d_check, &stwo_cache);
            let check = d_check.to_host();
            let check_match = check == stwo_evals;
            eprintln!("[mapping] reordered→stwo INTT→stwo NTT == stwo_evals: {}", check_match);

            // Also check eval_at_oods gives correct values at domain points
            use crate::oods::{OodsPoint, eval_at_oods_from_coeffs, qm31_from_m31};
            let mut oods_match = 0;
            for k in 0..n.min(8) {
                let cn_k = if k < half_n { ho.at(k) } else { ho.at(k - half_n).conjugate() };
                let brt_k = k.reverse_bits() >> (usize::BITS - log_n);
                let f = eval_at_oods_from_coeffs(&stwo_coeffs, OodsPoint {
                    x: qm31_from_m31(cn_k.x), y: qm31_from_m31(cn_k.y),
                });
                if f.to_u32_array()[0] == stwo_evals[brt_k] && f.to_u32_array()[1] == 0 {
                    oods_match += 1;
                }
            }
            eprintln!("[mapping] eval_at_oods at canonic BRT points: {}/{}",
                oods_match, n.min(8));
        }

        assert_eq!(found, n, "Permutation mapping failed");
    }

    /// Check: do old NTT and stwo NTT evaluate at the SAME set of circle points?
    #[test]
    fn test_old_vs_stwo_domain_points() {
        use crate::field::M31;
        use std::collections::HashSet;
        crate::cuda::ffi::init_memory_pool();

        let log_n = 6u32;
        let n = 1usize << log_n;
        let coset = Coset::half_coset(log_n);

        // Old NTT domain points: half_coset.at(j) for j=0..n-1
        let old_pts: HashSet<(u32, u32)> = (0..n).map(|j| {
            let p = coset.at(j);
            (p.x.0, p.y.0)
        }).collect();

        // Stwo NTT domain points: canonic_domain = half_odds(log_n-1) ∪ conjugate
        let ho = Coset::half_odds(log_n - 1);
        let half_n = n / 2;
        let stwo_pts: HashSet<(u32, u32)> = (0..n).map(|j| {
            let p = if j < half_n { ho.at(j) } else { ho.at(j - half_n).conjugate() };
            (p.x.0, p.y.0)
        }).collect();

        let intersection = old_pts.intersection(&stwo_pts).count();
        eprintln!("[domain] old NTT: {} unique points", old_pts.len());
        eprintln!("[domain] stwo NTT: {} unique points", stwo_pts.len());
        eprintln!("[domain] intersection: {}", intersection);
        eprintln!("[domain] same domain: {}", intersection == n);

        if intersection == n {
            // Build the permutation: old position i → stwo position j
            // where both evaluate at the same circle point
            let mut sigma = vec![usize::MAX; n];
            for i in 0..n {
                let old_p = coset.at(i);
                for j in 0..n {
                    let stwo_p = if j < half_n { ho.at(j) } else { ho.at(j - half_n).conjugate() };
                    if old_p.x.0 == stwo_p.x.0 && old_p.y.0 == stwo_p.y.0 {
                        sigma[i] = j;
                        break;
                    }
                }
            }
            let found = sigma.iter().filter(|&&v| v != usize::MAX).count();
            eprintln!("[domain] permutation found: {}/{}", found, n);
            if found == n {
                eprintln!("[domain] sigma[0..8] = {:?}", &sigma[..8]);
                // Check pattern: is sigma[k] = k/2 for even k, n-1-k/2 for odd k?
                let mut pattern_match = 0;
                for k in 0..n {
                    let expected = if k % 2 == 0 { k / 2 } else { n - 1 - k / 2 };
                    if sigma[k] == expected { pattern_match += 1; }
                }
                eprintln!("[domain] sigma matches k/2 pattern: {}/{}", pattern_match, n);
            }
        }
    }

    #[test]
    fn test_stwo_ntt_roundtrip() {
        crate::cuda::ffi::init_memory_pool();
        let log_n = 6u32;
        let n = 1usize << log_n;
        let coset = Coset::half_coset(log_n);
        let cache = StwoNttCache::new(&coset);

        let data: Vec<u32> = (0..n).map(|i| ((i * 7 + 3) % (crate::field::m31::P as usize)) as u32).collect();
        let mut d = DeviceBuffer::from_host(&data);
        evaluate_stwo(&mut d, &cache);
        interpolate_stwo(&mut d, &cache);
        let result = d.to_host();
        assert_eq!(data, result, "Stwo NTT roundtrip failed");
    }

    /// Critical test: does eval_at_oods_from_coeffs(old_intt_coeffs, z) give the correct f(z)?
    ///
    /// Computes f(z) two ways:
    /// 1. Old INTT → coeffs → eval_at_oods_from_coeffs
    /// 2. Direct sum: f(z) = Σ_{k=0}^{n-1} trace[k] * L_k(z) via barycentric interpolation
    ///    on the half_coset domain
    ///
    /// If they differ, eval_at_oods_from_coeffs is incompatible with old INTT coefficients.
    #[test]
    #[ignore = "diagnostic: old INTT coefficients are incompatible with eval_at_oods_from_coeffs"]
    fn test_old_intt_eval_at_oods_correctness() {
        use crate::oods::{OodsPoint, eval_at_oods_from_coeffs, qm31_from_m31};
        use crate::channel::Channel;
        use crate::field::QM31;
        crate::cuda::ffi::init_memory_pool();

        let log_n = 6u32;
        let n = 1usize << log_n;
        let coset = Coset::half_coset(log_n);

        // Known trace values (natural order: row k at index k).
        let trace: Vec<u32> = (0..n).map(|i| ((i * 7 + 3) % (crate::field::m31::P as usize)) as u32).collect();

        // Method 1: Old INTT → coefficients → eval_at_oods.
        let inv_cache = InverseTwiddleCache::new(&coset);
        let mut d_data = DeviceBuffer::from_host(&trace);
        interpolate(&mut d_data, &inv_cache);
        let old_coeffs = d_data.to_host();

        let mut ch = Channel::new();
        ch.mix_u64(99);
        let z = OodsPoint::from_channel(&mut ch);
        let f_z_method1 = eval_at_oods_from_coeffs(&old_coeffs, z);

        // Method 2: Direct evaluation at z via the old NTT evaluation formula.
        // The old NTT evaluates coefficients at half_coset domain points.
        // trace[k] = f(half_coset.at(k)) in the old NTT basis (natural order).
        // Wait — actually the old NTT takes coefficients in a specific order and
        // produces evaluations. Let me instead use: if old_coeffs are correct,
        // then evaluating them with the OLD NTT's forward transform at any of the
        // n domain points should give back trace[k].
        //
        // The real test: does the old INTT + forward NTT round-trip?
        let fwd_cache = ForwardTwiddleCache::new(&coset);
        let mut d_check = DeviceBuffer::from_host(&old_coeffs);
        evaluate(&mut d_check, &fwd_cache);
        let roundtrip = d_check.to_host();
        assert_eq!(roundtrip, trace, "Old NTT roundtrip broken");

        // Method 2 continued: evaluate f at z by iterating over evaluations.
        // The polynomial f has the property: f(p_k) = trace[k] where p_k = half_coset.at(k).
        // This means: Σ_{j} coeffs[j] * basis_j(p_k) = trace[k].
        // We want: Σ_{j} coeffs[j] * basis_j(z).
        // eval_at_oods_from_coeffs computes this IF the basis_j are the ones from eval_at_oods.
        //
        // Direct check: compute f(z) as a weighted sum of evaluations using the
        // half_coset vanishing polynomial (barycentric formula). For a polynomial
        // on the circle domain D of size n, evaluated at point z ∉ D:
        //   f(z) = v(z) * Σ_{k=0}^{n-1} f(p_k) * w_k / (z - p_k)
        // where v(z) = Π_{k} (z - p_k) is the vanishing polynomial of D at z,
        // and w_k = 1/Π_{j≠k} (p_k - p_j) are barycentric weights.
        //
        // On the circle group, the "subtraction" z - p_k is the pair vanishing:
        //   complex_conjugate_line(p_k, z)
        //
        // This is too complex for a quick test. Let's use a simpler approach:
        // evaluate f at each half_coset point using eval_at_oods_from_coeffs
        // and verify it matches trace.
        let mut eval_mismatch = 0;
        for k in 0..n.min(8) {
            let p_k = coset.at(k);
            let zp = OodsPoint { x: qm31_from_m31(p_k.x), y: qm31_from_m31(p_k.y) };
            let f_pk = eval_at_oods_from_coeffs(&old_coeffs, zp);
            let expected = qm31_from_m31(M31(trace[k]));
            if f_pk != expected {
                if eval_mismatch < 3 {
                    eprintln!("[basis] eval_at_oods(coeffs, p_{}) = {:?}, expected {:?}",
                        k, f_pk.to_u32_array(), expected.to_u32_array());
                }
                eval_mismatch += 1;
            }
        }
        eprintln!("[basis] eval_at_oods at half_coset domain points: {}/{} mismatches",
            eval_mismatch, n.min(8));

        if eval_mismatch > 0 {
            eprintln!("[basis] OLD INTT coefficients are NOT in the eval_at_oods basis!");
            eprintln!("[basis] f(z) via old INTT + eval_at_oods: {:?}", f_z_method1.to_u32_array());
        }
        // Now test: stwo INTT on TRACE DATA directly → stwo coefficients → eval_at_oods.
        // The stwo NTT roundtrip passes (evaluate_stwo(interpolate_stwo(x)) = x),
        // so stwo coefficients from trace data should give f(z) at stwo domain points.
        let stwo_cache = StwoNttCache::new(&coset);
        let mut d_stwo = DeviceBuffer::from_host(&trace);
        interpolate_stwo(&mut d_stwo, &stwo_cache);
        let stwo_coeffs = d_stwo.to_host();

        // But trace data is at half_coset points, not CircleDomain points!
        // The stwo NTT "thinks" the trace data is at CircleDomain points.
        // So stwo_coeffs represent a polynomial that equals trace[k] at
        // CircleDomain.at(k), NOT at half_coset.at(k).
        //
        // Verify: evaluate stwo_coeffs at CircleDomain points (canonic) should give trace.
        let ho = Coset::half_odds(log_n - 1);
        let half_n = n / 2;
        let mut stwo_mismatch = 0;
        for k in 0..n.min(8) {
            // CircleDomain.at(k) in the stwo convention:
            let cn_k = k; // natural canonic index
            let pt = if cn_k < half_n { ho.at(cn_k) } else { ho.at(cn_k - half_n).conjugate() };
            let zp = OodsPoint { x: qm31_from_m31(pt.x), y: qm31_from_m31(pt.y) };
            let f_pk = eval_at_oods_from_coeffs(&stwo_coeffs, zp);
            let expected = qm31_from_m31(M31(trace[k]));
            if f_pk != expected {
                if stwo_mismatch < 3 {
                    eprintln!("[basis] stwo eval_at_oods at canonic pt {}: {:?} vs trace {:?}",
                        k, f_pk.to_u32_array(), expected.to_u32_array());
                }
                stwo_mismatch += 1;
            }
        }
        eprintln!("[basis] stwo INTT(trace) → eval_at_oods at canonic domain: {}/{} mismatches",
            stwo_mismatch, n.min(8));

        // The right approach: stwo INTT gives coefficients that reproduce trace values
        // at the CircleDomain's NATURAL ordering. To get f(half_coset.at(k)), we'd need
        // to permute trace to canonic order first.
        // But for OODS: we just need f(z) for a random QM31 point z — which works
        // with ANY consistent polynomial representation.
        let f_z_stwo = eval_at_oods_from_coeffs(&stwo_coeffs, z);
        eprintln!("[basis] f(z) via old INTT:  {:?}", f_z_method1.to_u32_array());
        eprintln!("[basis] f(z) via stwo INTT: {:?}", f_z_stwo.to_u32_array());

        // The stwo f(z) should be the CORRECT value (polynomial passing through
        // all trace values, just at different domain assignment).
        // Both old and stwo represent the SAME polynomial, just in different bases.
        // eval_at_oods should give the same f(z) for both... unless they're DIFFERENT polynomials.
        eprintln!("[basis] old f(z) == stwo f(z): {}", f_z_method1 == f_z_stwo);

        // Both tests tell us the story: old INTT coefficients are WRONG for eval_at_oods.
        // The fix: use stwo INTT for OODS point evaluation.
        assert_eq!(eval_mismatch, 0,
            "eval_at_oods_from_coeffs(old_intt_coeffs, domain_point) != trace value \
             → old INTT basis != eval_at_oods basis. Use stwo INTT for OODS evaluation.");
    }
}
