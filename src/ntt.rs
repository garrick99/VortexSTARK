//! GPU Circle NTT: evaluate (coefficients → values) and interpolate (values → coefficients).
//!
//! Wraps the CUDA circle_ntt kernels with twiddle factor management.

use crate::circle::{compute_itwiddles, compute_twiddles, Coset};
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
        let (line_twids, circle_twids, offsets, sizes) = compute_twiddles(coset);
        let (iline_twids, icircle_twids, ioffsets, isizes) = compute_itwiddles(coset);

        Self {
            log_n: coset.log_size,
            d_twiddles: DeviceBuffer::from_host(&line_twids),
            d_circle_twids: DeviceBuffer::from_host(&circle_twids),
            layer_offsets: offsets,
            layer_sizes: sizes,
            d_itwiddles: DeviceBuffer::from_host(&iline_twids),
            d_circle_itwids: DeviceBuffer::from_host(&icircle_twids),
            ilayer_offsets: ioffsets,
            ilayer_sizes: isizes,
        }
    }
}

/// Forward NTT: coefficients → evaluation values (in-place on GPU).
/// `d_data` must contain `n = 2^log_n` M31 elements on the device.
pub fn evaluate(d_data: &mut DeviceBuffer<u32>, cache: &TwiddleCache) {
    let n = 1u32 << cache.log_n;
    let n_line_layers = if cache.log_n > 0 { cache.log_n - 1 } else { 0 };

    unsafe {
        ffi::cuda_circle_ntt_evaluate(
            d_data.as_mut_ptr(),
            cache.d_twiddles.as_ptr(),
            cache.d_circle_twids.as_ptr(),
            cache.layer_offsets.as_ptr(),
            cache.layer_sizes.as_ptr(),
            n_line_layers,
            n,
        );
    }
}

/// Inverse NTT: evaluation values → coefficients (in-place on GPU).
pub fn interpolate(d_data: &mut DeviceBuffer<u32>, cache: &TwiddleCache) {
    let n = 1u32 << cache.log_n;
    let n_line_layers = if cache.log_n > 0 { cache.log_n - 1 } else { 0 };

    unsafe {
        ffi::cuda_circle_ntt_interpolate(
            d_data.as_mut_ptr(),
            cache.d_itwiddles.as_ptr(),
            cache.d_circle_itwids.as_ptr(),
            cache.ilayer_offsets.as_ptr(),
            cache.ilayer_sizes.as_ptr(),
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
pub fn evaluate_batch(columns: &mut [DeviceBuffer<u32>], cache: &TwiddleCache) {
    if columns.is_empty() {
        return;
    }
    let n = 1u32 << cache.log_n;
    let n_cols = columns.len() as u32;
    let n_line_layers = if cache.log_n > 0 { cache.log_n - 1 } else { 0 };

    // Build array of device pointers
    let ptrs: Vec<*mut u32> = columns.iter_mut().map(|c| c.as_mut_ptr()).collect();
    let d_ptrs = DeviceBuffer::from_host(&ptrs);

    unsafe {
        ffi::cuda_circle_ntt_evaluate_batch(
            d_ptrs.as_ptr() as *mut *mut u32,
            cache.d_twiddles.as_ptr(),
            cache.d_circle_twids.as_ptr(),
            cache.layer_offsets.as_ptr(),
            cache.layer_sizes.as_ptr(),
            n_line_layers,
            n,
            n_cols,
        );
    }
}

/// Inverse NTT on multiple columns simultaneously.
pub fn interpolate_batch(columns: &mut [DeviceBuffer<u32>], cache: &TwiddleCache) {
    if columns.is_empty() {
        return;
    }
    let n = 1u32 << cache.log_n;
    let n_cols = columns.len() as u32;
    let n_line_layers = if cache.log_n > 0 { cache.log_n - 1 } else { 0 };

    let ptrs: Vec<*mut u32> = columns.iter_mut().map(|c| c.as_mut_ptr()).collect();
    let d_ptrs = DeviceBuffer::from_host(&ptrs);

    unsafe {
        ffi::cuda_circle_ntt_interpolate_batch(
            d_ptrs.as_ptr() as *mut *mut u32,
            cache.d_itwiddles.as_ptr(),
            cache.d_circle_itwids.as_ptr(),
            cache.ilayer_offsets.as_ptr(),
            cache.ilayer_sizes.as_ptr(),
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
