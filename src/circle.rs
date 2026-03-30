//! Circle group and domain types over M31.
//!
//! The circle group is { (x,y) : x^2 + y^2 = 1 } over M31.
//! The generator has order 2^31, matching the M31 field structure.

use crate::cuda::ffi;
use crate::device::DeviceBuffer;
use crate::field::M31;

/// A point on the circle x^2 + y^2 = 1 over M31.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CirclePoint {
    pub x: M31,
    pub y: M31,
}

impl CirclePoint {
    /// Circle group identity: (1, 0).
    pub const IDENTITY: Self = Self {
        x: M31::ONE,
        y: M31::ZERO,
    };

    /// Generator of order 2^31 on the M31 circle group.
    /// G = (2, 1268011823) where G.x^2 + G.y^2 = 1 mod P.
    pub const GENERATOR: Self = Self {
        x: M31(2),
        y: M31(1268011823),
    };

    /// Circle group operation: (x1,y1) * (x2,y2) = (x1*x2 - y1*y2, x1*y2 + y1*x2)
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x - self.y * rhs.y,
            y: self.x * rhs.y + self.y * rhs.x,
        }
    }

    /// Inverse: conjugate (x, -y) since |point| = 1.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            x: self.x,
            y: -self.y,
        }
    }

    /// Double: p * p.
    #[inline]
    pub fn double(self) -> Self {
        self.mul(self)
    }

    /// Repeated squaring: self^(2^n).
    #[inline]
    pub fn repeated_double(self, n: u32) -> Self {
        let mut p = self;
        for _ in 0..n {
            p = p.double();
        }
        p
    }

    /// Scalar multiplication by repeated doubling.
    pub fn mul_scalar(self, mut k: u32) -> Self {
        let mut result = Self::IDENTITY;
        let mut base = self;
        while k > 0 {
            if k & 1 == 1 {
                result = result.mul(base);
            }
            base = base.double();
            k >>= 1;
        }
        result
    }

    /// Antipodal point: (-x, -y).
    #[inline]
    pub fn antipode(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

/// A coset of the circle group: { initial * G^(step*i) : i = 0..size }
/// where G is the circle group generator.
#[derive(Clone, Copy, Debug)]
pub struct Coset {
    pub initial: CirclePoint,
    pub step: CirclePoint,
    pub log_size: u32,
}

impl Coset {
    /// Create a standard power-of-2 coset.
    /// The subgroup of order 2^log_size has generator G^(2^(31-log_size)).
    pub fn subgroup(log_size: u32) -> Self {
        assert!(log_size <= 31);
        let step = CirclePoint::GENERATOR.repeated_double(31 - log_size);
        Self {
            initial: CirclePoint::IDENTITY,
            step,
            log_size,
        }
    }

    /// Half-coset: standard domain used for polynomial evaluation.
    /// Coset = G^(2^(30-log_size)) * subgroup(log_size).
    pub fn half_coset(log_size: u32) -> Self {
        assert!(log_size <= 30);
        let step = CirclePoint::GENERATOR.repeated_double(31 - log_size);
        let initial = CirclePoint::GENERATOR.repeated_double(30 - log_size);
        Self {
            initial,
            step,
            log_size,
        }
    }

    /// Half-odds coset: used by stwo for FRI line-fold domains.
    /// Coset = G^(2^(29-log_size)) * subgroup(log_size).
    /// This matches `Coset::half_odds` in the stwo library.
    pub fn half_odds(log_size: u32) -> Self {
        assert!(log_size <= 29);
        let step = CirclePoint::GENERATOR.repeated_double(31 - log_size);
        let initial = CirclePoint::GENERATOR.repeated_double(29 - log_size);
        Self {
            initial,
            step,
            log_size,
        }
    }

    /// Odds coset: matches stwo's CanonicCoset::new(log_size).coset().
    /// Coset = G^(2^(30-log_size)) * subgroup(log_size).
    /// Same initial as half_coset but covers ALL n points (not just n/2).
    /// Used as the circle fold twiddle domain for BRT-canonic FRI.
    pub fn odds(log_size: u32) -> Self {
        assert!(log_size <= 30);
        let step = CirclePoint::GENERATOR.repeated_double(31 - log_size);
        let initial = CirclePoint::GENERATOR.repeated_double(30 - log_size);
        Self { initial, step, log_size }
    }

    pub fn size(&self) -> usize {
        1 << self.log_size
    }

    /// Get the i-th point of the coset.
    pub fn at(&self, i: usize) -> CirclePoint {
        self.initial.mul(self.step.mul_scalar(i as u32))
    }

    /// Evaluate the vanishing polynomial of half_coset(log_size) at x.
    ///
    /// Z_H(x) = 0 iff x is the x-coordinate of a point in the trace domain.
    /// Formula: Z_H(x) = f_{log_size}(x) + 1,
    ///   where f_0(x) = x, f_{i+1}(x) = 2x^2 − 1  (circle group doubling).
    pub fn circle_vanishing_poly_at(x: M31, log_size: u32) -> M31 {
        let mut v = x;
        for _ in 0..log_size {
            v = M31(2) * v * v - M31::ONE;
        }
        v + M31::ONE
    }

    /// Generate all coset points at once using sequential multiplication.
    /// O(n) circle multiplications instead of O(n log n).
    pub fn all_points(&self) -> Vec<CirclePoint> {
        let n = self.size();
        let mut points = Vec::with_capacity(n);
        let mut current = self.initial;
        for _ in 0..n {
            points.push(current);
            current = current.mul(self.step);
        }
        points
    }
}

/// Compute twiddle factors for the Circle NTT on GPU.
/// Returns (line_twiddles, circle_twiddles, layer_offsets, layer_sizes).
///
/// Line twiddles: for each layer l (l = n_line_layers-1 down to 0),
///   twiddles[offset[l]..offset[l]+size[l]] = x-coordinates of subgroup points.
///
/// Circle twiddles: y-coordinates for the circle butterfly (layer 0).
pub fn compute_twiddles(
    coset: &Coset,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let log_n = coset.log_size;
    let n = coset.size();
    let n_line_layers = if log_n > 0 { log_n - 1 } else { 0 };

    // Compute all coset points on GPU in parallel
    let mut d_x = DeviceBuffer::<u32>::alloc(n);
    let mut d_y = DeviceBuffer::<u32>::alloc(n);
    unsafe {
        ffi::cuda_compute_coset_points(
            coset.initial.x.0, coset.initial.y.0,
            coset.step.x.0, coset.step.y.0,
            d_x.as_mut_ptr(), d_y.as_mut_ptr(),
            n as u32,
        );
        ffi::cuda_device_sync();
    }

    // Circle twiddles: y-coordinates of first half
    let half_n = n / 2;
    let all_y = d_y.to_host();
    let circle_twids: Vec<u32> = all_y[..half_n].to_vec();
    drop(d_y);

    // Build line twiddles entirely on GPU — no per-layer downloads
    // Total twiddle count: n/2 + n/4 + ... + 1 = n - 1
    let total_twiddles = n - 1;
    let mut d_line_twiddles = DeviceBuffer::<u32>::alloc(total_twiddles);

    let mut layer_offsets = Vec::new();
    let mut layer_sizes = Vec::new();
    let mut d_current = d_x;
    let mut current_n = n;
    let mut write_offset = 0usize;

    for _layer in 0..n_line_layers as usize {
        let layer_size = current_n / 2;
        layer_offsets.push(write_offset as u32);
        layer_sizes.push(layer_size as u32);

        // Extract even-indexed x values as twiddles + squash for next layer, all on GPU
        let mut d_squashed = DeviceBuffer::<u32>::alloc(layer_size);
        unsafe {
            ffi::cuda_extract_and_squash(
                d_current.as_ptr(),
                d_line_twiddles.as_mut_ptr().add(write_offset),
                d_squashed.as_mut_ptr(),
                layer_size as u32,
            );
        }

        write_offset += layer_size;
        d_current = d_squashed;
        current_n = layer_size;
    }
    unsafe { ffi::cuda_device_sync(); }

    // Single download of the complete twiddle array
    let line_twiddles = d_line_twiddles.to_host();

    (line_twiddles, circle_twids, layer_offsets, layer_sizes)
}

/// Compute inverse twiddle factors using GPU batch inverse.
/// For iNTT, twiddles need to be the inverses of the forward twiddles.
pub fn compute_itwiddles(
    coset: &Coset,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let (line_twids, circle_twids, offsets, sizes) = compute_twiddles(coset);

    // Batch invert on GPU (Montgomery's trick)
    let inv_line = gpu_batch_inverse(&line_twids);
    let inv_circle = gpu_batch_inverse(&circle_twids);

    (inv_line, inv_circle, offsets, sizes)
}

/// Compute both forward and inverse twiddles, returning DeviceBuffers (no host round-trip).
pub fn compute_both_twiddles_gpu(
    coset: &Coset,
) -> (DeviceBuffer<u32>, DeviceBuffer<u32>, DeviceBuffer<u32>, DeviceBuffer<u32>, Vec<u32>, Vec<u32>) {
    let log_n = coset.log_size;
    let n = coset.size();
    let n_line_layers = if log_n > 0 { log_n - 1 } else { 0 };

    // Compute all coset points on GPU in parallel
    let mut d_x = DeviceBuffer::<u32>::alloc(n);
    let mut d_y = DeviceBuffer::<u32>::alloc(n);
    unsafe {
        ffi::cuda_compute_coset_points(
            coset.initial.x.0, coset.initial.y.0,
            coset.step.x.0, coset.step.y.0,
            d_x.as_mut_ptr(), d_y.as_mut_ptr(),
            n as u32,
        );
        let sync_err = ffi::cudaDeviceSynchronize();
        let last_err = ffi::cudaGetLastError();
        eprintln!("[CUDA] compute_coset_points: n={n}, log_n={log_n}, sync={sync_err}, last={last_err}");
        if sync_err != 0 || last_err != 0 {
            panic!("[CUDA] compute_coset_points kernel failed!");
        }
    }

    // Circle twiddles: y-coordinates of first half (stay on device)
    let half_n = n / 2;
    let mut d_circle_twids = DeviceBuffer::<u32>::alloc(half_n);
    unsafe {
        ffi::cudaMemcpy(
            d_circle_twids.as_mut_ptr() as *mut std::ffi::c_void,
            d_y.as_ptr() as *const std::ffi::c_void,
            half_n * std::mem::size_of::<u32>(),
            ffi::MEMCPY_D2D,
        );
    }
    drop(d_y);

    // Build line twiddles on GPU
    let total_twiddles = n - 1;
    let mut d_line_twiddles = DeviceBuffer::<u32>::alloc(total_twiddles);

    let mut layer_offsets = Vec::new();
    let mut layer_sizes = Vec::new();
    let mut d_current = d_x;
    let mut current_n = n;
    let mut write_offset = 0usize;

    for layer in 0..n_line_layers as usize {
        let layer_size = current_n / 2;
        layer_offsets.push(write_offset as u32);
        layer_sizes.push(layer_size as u32);

        let mut d_squashed = DeviceBuffer::<u32>::alloc(layer_size);
        unsafe {
            ffi::cuda_extract_and_squash(
                d_current.as_ptr(),
                d_line_twiddles.as_mut_ptr().add(write_offset),
                d_squashed.as_mut_ptr(),
                layer_size as u32,
            );
            let err = ffi::cudaDeviceSynchronize();
            if err != 0 {
                let last = ffi::cudaGetLastError();
                panic!("[CUDA] extract_and_squash failed at layer {layer}: \
                        sync={err}, last={last}, layer_size={layer_size}, \
                        write_offset={write_offset}, current_n={current_n}");
            }
        }

        write_offset += layer_size;
        d_current = d_squashed;
        current_n = layer_size;
    }

    // Inverse twiddles via GPU batch inverse (device → device)
    eprintln!("[CUDA] batch_inverse: total_twiddles={total_twiddles}, half_n={half_n}");
    let mut d_iline_twiddles = DeviceBuffer::<u32>::alloc(total_twiddles);
    let mut d_icircle_twids = DeviceBuffer::<u32>::alloc(half_n);
    unsafe {
        ffi::cuda_batch_inverse_m31(
            d_line_twiddles.as_ptr(), d_iline_twiddles.as_mut_ptr(), total_twiddles as u32,
        );
        let err = ffi::cudaDeviceSynchronize();
        eprintln!("[CUDA] batch_inverse line: err={err}");
        ffi::cuda_batch_inverse_m31(
            d_circle_twids.as_ptr(), d_icircle_twids.as_mut_ptr(), half_n as u32,
        );
        let err = ffi::cudaDeviceSynchronize();
        eprintln!("[CUDA] batch_inverse circle: err={err}");
    }

    // Returns: (fwd_line, fwd_circle, inv_line, inv_circle, offsets, sizes)
    (d_line_twiddles, d_circle_twids, d_iline_twiddles, d_icircle_twids, layer_offsets, layer_sizes)
}

/// Compute forward twiddles only, returning DeviceBuffers.
pub fn compute_forward_twiddles_gpu(
    coset: &Coset,
) -> (DeviceBuffer<u32>, DeviceBuffer<u32>, Vec<u32>, Vec<u32>) {
    let log_n = coset.log_size;
    let n = coset.size();
    let n_line_layers = if log_n > 0 { log_n - 1 } else { 0 };

    let mut d_x = DeviceBuffer::<u32>::alloc(n);
    let mut d_y = DeviceBuffer::<u32>::alloc(n);
    unsafe {
        ffi::cuda_compute_coset_points(
            coset.initial.x.0, coset.initial.y.0,
            coset.step.x.0, coset.step.y.0,
            d_x.as_mut_ptr(), d_y.as_mut_ptr(),
            n as u32,
        );
        ffi::cuda_device_sync();
    }

    let half_n = n / 2;
    let mut d_circle_twids = DeviceBuffer::<u32>::alloc(half_n);
    unsafe {
        ffi::cudaMemcpy(
            d_circle_twids.as_mut_ptr() as *mut std::ffi::c_void,
            d_y.as_ptr() as *const std::ffi::c_void,
            half_n * std::mem::size_of::<u32>(),
            ffi::MEMCPY_D2D,
        );
    }
    drop(d_y);

    let total_twiddles = n - 1;
    let mut d_line_twiddles = DeviceBuffer::<u32>::alloc(total_twiddles);
    let mut layer_offsets = Vec::new();
    let mut layer_sizes = Vec::new();
    let mut d_current = d_x;
    let mut current_n = n;
    let mut write_offset = 0usize;

    for layer in 0..n_line_layers as usize {
        let layer_size = current_n / 2;
        layer_offsets.push(write_offset as u32);
        layer_sizes.push(layer_size as u32);

        let mut d_squashed = DeviceBuffer::<u32>::alloc(layer_size);
        unsafe {
            ffi::cuda_extract_and_squash(
                d_current.as_ptr(),
                d_line_twiddles.as_mut_ptr().add(write_offset),
                d_squashed.as_mut_ptr(),
                layer_size as u32,
            );
            let err = ffi::cudaDeviceSynchronize();
            if err != 0 {
                let last = ffi::cudaGetLastError();
                panic!("[CUDA] extract_and_squash failed at layer {layer}: \
                        sync={err}, last={last}, layer_size={layer_size}, \
                        write_offset={write_offset}, current_n={current_n}");
            }
        }

        write_offset += layer_size;
        d_current = d_squashed;
        current_n = layer_size;
    }

    (d_line_twiddles, d_circle_twids, layer_offsets, layer_sizes)
}

/// Compute inverse twiddles only, returning DeviceBuffers.
/// Internally creates forward twiddles, inverts, and drops the forward ones.
pub fn compute_inverse_twiddles_gpu(
    coset: &Coset,
) -> (DeviceBuffer<u32>, DeviceBuffer<u32>, Vec<u32>, Vec<u32>) {
    let (d_fwd_line, d_fwd_circle, offsets, sizes) = compute_forward_twiddles_gpu(coset);

    let n = coset.size();
    let half_n = n / 2;
    let total_twiddles = n - 1;

    let mut d_iline = DeviceBuffer::<u32>::alloc(total_twiddles);
    let mut d_icircle = DeviceBuffer::<u32>::alloc(half_n);
    unsafe {
        ffi::cuda_batch_inverse_m31(
            d_fwd_line.as_ptr(), d_iline.as_mut_ptr(), total_twiddles as u32,
        );
        ffi::cuda_batch_inverse_m31(
            d_fwd_circle.as_ptr(), d_icircle.as_mut_ptr(), half_n as u32,
        );
        ffi::cuda_device_sync();
    }
    // Forward twiddles dropped here
    drop(d_fwd_line);
    drop(d_fwd_circle);

    (d_iline, d_icircle, offsets, sizes)
}

/// Compute both forward and inverse twiddles in one pass (avoids double coset computation).
pub fn compute_both_twiddles(
    coset: &Coset,
) -> ((Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>), (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)) {
    let (line_twids, circle_twids, offsets, sizes) = compute_twiddles(coset);

    let inv_line = gpu_batch_inverse(&line_twids);
    let inv_circle = gpu_batch_inverse(&circle_twids);

    (
        (line_twids, circle_twids, offsets.clone(), sizes.clone()),
        (inv_line, inv_circle, offsets, sizes),
    )
}

/// Batch inverse via GPU kernel.
fn gpu_batch_inverse(values: &[u32]) -> Vec<u32> {
    if values.is_empty() {
        return Vec::new();
    }
    let d_input = DeviceBuffer::from_host(values);
    let mut d_output = DeviceBuffer::<u32>::alloc(values.len());
    unsafe {
        ffi::cuda_batch_inverse_m31(d_input.as_ptr(), d_output.as_mut_ptr(), values.len() as u32);
        ffi::cuda_device_sync();
    }
    d_output.to_host()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_on_circle() {
        let g = CirclePoint::GENERATOR;
        // x^2 + y^2 should equal 1 (mod P)
        let sum = g.x * g.x + g.y * g.y;
        assert_eq!(sum, M31::ONE, "Generator not on circle");
    }

    #[test]
    fn test_generator_order() {
        // G^(2^31) should be the identity
        let g = CirclePoint::GENERATOR;
        let result = g.repeated_double(31);
        assert_eq!(result, CirclePoint::IDENTITY, "Generator order is not 2^31");
    }

    #[test]
    fn test_subgroup() {
        // Subgroup of order 4: G^(2^29)
        let coset = Coset::subgroup(2);
        assert_eq!(coset.size(), 4);
        // The 4th power should return to identity
        let p = coset.step.mul_scalar(4);
        assert_eq!(p, CirclePoint::IDENTITY);
    }

    #[test]
    fn test_identity() {
        let g = CirclePoint::GENERATOR;
        assert_eq!(g.mul(CirclePoint::IDENTITY), g);
    }

    #[test]
    fn test_conjugate() {
        let g = CirclePoint::GENERATOR;
        let prod = g.mul(g.conjugate());
        assert_eq!(prod, CirclePoint::IDENTITY);
    }
}
