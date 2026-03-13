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

/// Compute fold twiddles for fold_line: inverse x-coordinates of domain points.
/// Domain points are in bit-reversed order; we take pairs (2i, 2i+1).
fn compute_fold_line_twiddles(domain: &Coset) -> Vec<u32> {
    let n = domain.size();
    let half_n = n / 2;
    let log_size = domain.log_size;

    (0..half_n)
        .map(|i| {
            let bit_rev_idx = bit_reverse(i << 1, log_size);
            let point = domain.at(bit_rev_idx);
            point.x.inverse().0
        })
        .collect()
}

/// Compute fold twiddles for fold_circle: inverse y-coordinates.
fn compute_fold_circle_twiddles(domain: &Coset) -> Vec<u32> {
    let n = domain.size();
    let half_n = n / 2;
    let log_size = domain.log_size;

    (0..half_n)
        .map(|i| {
            let bit_rev_idx = bit_reverse(i << 1, log_size);
            let point = domain.at(bit_rev_idx);
            point.y.inverse().0
        })
        .collect()
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
    let half_n = src.len / 2;
    let twiddles = compute_fold_circle_twiddles(domain);
    let d_twiddles = DeviceBuffer::from_host(&twiddles);

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
        ffi::cuda_device_sync();
    }
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
    let half_n = eval.len / 2;
    let twiddles = compute_fold_line_twiddles(domain);
    let d_twiddles = DeviceBuffer::from_host(&twiddles);

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
        ffi::cuda_device_sync();
    }

    out
}

/// Bit-reverse an index within a given number of bits.
fn bit_reverse(mut val: usize, bits: u32) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

/// Add FRI FFI bindings
/// Update cuda/ffi.rs with fold_line_soa and fold_circle_into_line_soa

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
