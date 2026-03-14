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
}
