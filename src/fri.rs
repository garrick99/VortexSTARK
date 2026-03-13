//! FRI (Fast Reed-Solomon Interactive Oracle Proof of Proximity) protocol.
//!
//! Implements circle FRI folding on GPU using SoA layout.
//! - fold_circle_into_line: first fold (circle → line)
//! - fold_line: subsequent folds (line → line, halving each time)

use crate::circle::Coset;
use crate::cuda::ffi;
use crate::device::DeviceBuffer;
use crate::field::QM31;

/// A secure (QM31) column in SoA layout: 4 separate M31 columns.
pub struct SecureColumn {
    pub cols: [DeviceBuffer<u32>; 4],
    pub len: usize,
}

impl SecureColumn {
    /// Allocate an uninitialized secure column (use when all elements will be written).
    pub fn alloc(len: usize) -> Self {
        let cols = std::array::from_fn(|_| DeviceBuffer::<u32>::alloc(len));
        Self { cols, len }
    }

    /// Allocate a zero-initialized secure column of given length.
    pub fn zeros(len: usize) -> Self {
        let mut cols = std::array::from_fn(|_| DeviceBuffer::<u32>::alloc(len));
        for col in &mut cols {
            col.zero();
        }
        Self { cols, len }
    }

    /// Upload from host QM31 values.
    pub fn from_qm31(values: &[QM31]) -> Self {
        let len = values.len();
        let mut c0 = vec![0u32; len];
        let mut c1 = vec![0u32; len];
        let mut c2 = vec![0u32; len];
        let mut c3 = vec![0u32; len];
        for (i, v) in values.iter().enumerate() {
            let arr = v.to_u32_array();
            c0[i] = arr[0];
            c1[i] = arr[1];
            c2[i] = arr[2];
            c3[i] = arr[3];
        }
        Self {
            cols: [
                DeviceBuffer::from_host(&c0),
                DeviceBuffer::from_host(&c1),
                DeviceBuffer::from_host(&c2),
                DeviceBuffer::from_host(&c3),
            ],
            len,
        }
    }

    /// Download to host QM31 values.
    pub fn to_qm31(&self) -> Vec<QM31> {
        let c0 = self.cols[0].to_host();
        let c1 = self.cols[1].to_host();
        let c2 = self.cols[2].to_host();
        let c3 = self.cols[3].to_host();
        (0..self.len)
            .map(|i| QM31::from_u32_array([c0[i], c1[i], c2[i], c3[i]]))
            .collect()
    }
}

/// Compute fold twiddles on GPU: domain points → batch inverse, all on device.
/// `extract_y`: false = x-coordinates (line fold), true = y-coordinates (circle fold).
fn compute_fold_twiddles_gpu(domain: &Coset, extract_y: bool) -> DeviceBuffer<u32> {
    let half_n = domain.size() / 2;
    let n = domain.size();
    let log_n = domain.log_size;

    // GPU kernel computes bit-reversed domain points (half_n values)
    let mut d_sources = DeviceBuffer::<u32>::alloc(half_n);
    unsafe {
        ffi::cuda_compute_fold_twiddle_sources(
            domain.initial.x.0,
            domain.initial.y.0,
            domain.step.x.0,
            domain.step.y.0,
            d_sources.as_mut_ptr(),
            n as u32,
            log_n,
            if extract_y { 1 } else { 0 },
        );
    }

    // GPU batch inverse (Montgomery's trick on device)
    let mut d_result = DeviceBuffer::<u32>::alloc(half_n);
    unsafe {
        ffi::cuda_batch_inverse_m31(d_sources.as_ptr(), d_result.as_mut_ptr(), half_n as u32);
    }

    d_result
}

/// Precompute FRI fold twiddles for all layers on GPU. Returns device buffers
/// for circle fold (index 0) and line folds (indices 1..n).
pub fn precompute_fri_twiddles(start_log_size: u32, stop_log_size: u32) -> Vec<DeviceBuffer<u32>> {
    let mut result = Vec::new();

    // Circle fold twiddles (first layer) — inverse y-coordinates
    let domain = Coset::half_coset(start_log_size);
    result.push(compute_fold_twiddles_gpu(&domain, true));

    // Line fold twiddles (subsequent layers) — inverse x-coordinates
    let mut log_size = start_log_size - 1;
    while log_size > stop_log_size {
        let domain = Coset::half_coset(log_size);
        result.push(compute_fold_twiddles_gpu(&domain, false));
        log_size -= 1;
    }

    result
}

/// FRI fold_circle_into_line on GPU with pre-uploaded twiddles.
pub fn fold_circle_into_line_with_twiddles(
    dst: &mut SecureColumn,
    src: &SecureColumn,
    alpha: QM31,
    d_twiddles: &DeviceBuffer<u32>,
) {
    let half_n = src.len / 2;

    let alpha_arr = alpha.to_u32_array();
    let alpha_sq = alpha * alpha;
    let alpha_sq_arr = alpha_sq.to_u32_array();

    unsafe {
        ffi::cuda_fold_circle_into_line_soa(
            dst.cols[0].as_mut_ptr(),
            dst.cols[1].as_mut_ptr(),
            dst.cols[2].as_mut_ptr(),
            dst.cols[3].as_mut_ptr(),
            src.cols[0].as_ptr(),
            src.cols[1].as_ptr(),
            src.cols[2].as_ptr(),
            src.cols[3].as_ptr(),
            d_twiddles.as_ptr(),
            alpha_arr.as_ptr(),
            alpha_sq_arr.as_ptr(),
            half_n as u32,
        );
    }
}

/// FRI fold_line on GPU with pre-uploaded twiddles.
pub fn fold_line_with_twiddles(
    eval: &SecureColumn,
    alpha: QM31,
    d_twiddles: &DeviceBuffer<u32>,
) -> SecureColumn {
    let half_n = eval.len / 2;
    let alpha_arr = alpha.to_u32_array();

    let mut out = SecureColumn {
        cols: std::array::from_fn(|_| DeviceBuffer::<u32>::alloc(half_n)),
        len: half_n,
    };

    unsafe {
        ffi::cuda_fold_line_soa(
            eval.cols[0].as_ptr(),
            eval.cols[1].as_ptr(),
            eval.cols[2].as_ptr(),
            eval.cols[3].as_ptr(),
            d_twiddles.as_ptr(),
            out.cols[0].as_mut_ptr(),
            out.cols[1].as_mut_ptr(),
            out.cols[2].as_mut_ptr(),
            out.cols[3].as_mut_ptr(),
            alpha_arr.as_ptr(),
            half_n as u32,
        );
    }

    out
}

/// FRI fold_circle_into_line on GPU.
///
/// First FRI step: folds a circle evaluation (n QM31 elements in SoA)
/// into a line evaluation (n/2 QM31 elements in SoA).
///
/// dst[i] += (f0+f1) + alpha * inv_y[i] * (f0-f1)
/// where f0 = src[2i], f1 = src[2i+1]
pub fn fold_circle_into_line(
    dst: &mut SecureColumn,
    src: &SecureColumn,
    alpha: QM31,
    domain: &Coset,
) {
    let d_twiddles = compute_fold_twiddles_gpu(domain, true);
    fold_circle_into_line_with_twiddles(dst, src, alpha, &d_twiddles);
}

/// FRI fold_line on GPU.
///
/// Subsequent FRI steps: folds a line evaluation (n QM31 elements in SoA)
/// into a line evaluation (n/2 QM31 elements in SoA).
///
/// result[i] = (f0+f1) + alpha * inv_x[i] * (f0-f1)
pub fn fold_line(
    eval: &SecureColumn,
    alpha: QM31,
    domain: &Coset,
) -> SecureColumn {
    let d_twiddles = compute_fold_twiddles_gpu(domain, false);
    fold_line_with_twiddles(eval, alpha, &d_twiddles)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::M31;

    #[test]
    fn test_fold_line_halves_size() {
        let n = 64;
        let log_n = 6u32;
        let domain = Coset::half_coset(log_n);

        let values: Vec<QM31> = (0..n)
            .map(|i| {
                QM31::from_m31_array([
                    M31((i as u32 * 3 + 1) % 0x7FFF_FFFF),
                    M31((i as u32 * 5 + 2) % 0x7FFF_FFFF),
                    M31((i as u32 * 7 + 3) % 0x7FFF_FFFF),
                    M31((i as u32 * 11 + 4) % 0x7FFF_FFFF),
                ])
            })
            .collect();

        let eval = SecureColumn::from_qm31(&values);
        let alpha = QM31::from_m31_array([M31(42), M31(99), M31(7), M31(13)]);

        let result = fold_line(&eval, alpha, &domain);
        assert_eq!(result.len, n / 2);

        // Verify we can download the result
        let host = result.to_qm31();
        assert_eq!(host.len(), n / 2);
    }

    #[test]
    fn test_fold_circle_into_line() {
        let n = 64;
        let log_n = 6u32;
        let domain = Coset::half_coset(log_n);

        let values: Vec<QM31> = (0..n)
            .map(|i| {
                QM31::from_m31_array([
                    M31((i as u32 * 3 + 1) % 0x7FFF_FFFF),
                    M31((i as u32 * 5 + 2) % 0x7FFF_FFFF),
                    M31((i as u32 * 7 + 3) % 0x7FFF_FFFF),
                    M31((i as u32 * 11 + 4) % 0x7FFF_FFFF),
                ])
            })
            .collect();

        let src = SecureColumn::from_qm31(&values);
        let mut dst = SecureColumn::zeros(n / 2);
        let alpha = QM31::from_m31_array([M31(42), M31(99), M31(7), M31(13)]);

        fold_circle_into_line(&mut dst, &src, alpha, &domain);

        let host = dst.to_qm31();
        assert_eq!(host.len(), n / 2);
        // Result should not be all zeros (the fold produces non-trivial output)
        assert!(host.iter().any(|v| *v != QM31::ZERO));
    }

    #[test]
    fn test_fold_line_deterministic() {
        let n = 32;
        let log_n = 5u32;
        let domain = Coset::half_coset(log_n);

        let values: Vec<QM31> = (0..n)
            .map(|i| QM31::from_m31_array([M31(i as u32 + 1), M31::ZERO, M31::ZERO, M31::ZERO]))
            .collect();

        let alpha = QM31::from_m31_array([M31(100), M31(200), M31(300), M31(400)]);

        let eval1 = SecureColumn::from_qm31(&values);
        let eval2 = SecureColumn::from_qm31(&values);

        let r1 = fold_line(&eval1, alpha, &domain);
        let r2 = fold_line(&eval2, alpha, &domain);

        assert_eq!(r1.to_qm31(), r2.to_qm31());
    }
}
