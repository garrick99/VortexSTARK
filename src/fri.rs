//! FRI (Fast Reed-Solomon Interactive Oracle Proof of Proximity) protocol.
//!
//! Implements circle FRI folding on GPU using SoA layout.
//! - fold_circle_into_line: first fold (circle → line)
//! - fold_line: subsequent folds (line → line, halving each time)

use crate::circle::Coset;
use crate::cuda::ffi;
use crate::device::DeviceBuffer;
use crate::field::{M31, QM31};

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

/// Compute fold twiddles on GPU on demand (public API for lazy twiddle lifetime).
/// Returns a DeviceBuffer that the caller can drop after use.
pub fn compute_fold_twiddles_on_demand(domain: &Coset, extract_y: bool) -> DeviceBuffer<u32> {
    compute_fold_twiddles_gpu(domain, extract_y)
}

/// Start computing fold twiddles asynchronously on a CUDA stream.
/// Returns (sources_buf, result_buf) — caller must sync the stream before using result.
pub fn compute_fold_twiddles_async(
    domain: &Coset,
    extract_y: bool,
    stream: &ffi::CudaStream,
) -> (DeviceBuffer<u32>, DeviceBuffer<u32>) {
    let half_n = domain.size() / 2;
    let n = domain.size();
    let log_n = domain.log_size;

    let mut d_sources = DeviceBuffer::<u32>::alloc(half_n);
    unsafe {
        ffi::cuda_compute_fold_twiddle_sources_stream(
            domain.initial.x.0, domain.initial.y.0,
            domain.step.x.0, domain.step.y.0,
            d_sources.as_mut_ptr(),
            n as u32, log_n,
            if extract_y { 1 } else { 0 },
            stream.ptr,
        );
    }

    let mut d_result = DeviceBuffer::<u32>::alloc(half_n);
    unsafe {
        ffi::cuda_batch_inverse_m31_stream(
            d_sources.as_ptr(), d_result.as_mut_ptr(), half_n as u32, stream.ptr,
        );
    }

    (d_sources, d_result)
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
    // Use half_odds to match stwo's FriVerifier fold domain convention.
    let mut log_size = start_log_size - 1;
    while log_size > stop_log_size {
        let domain = Coset::half_odds(log_size);
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


// --- CPU-side FRI operations for small tail iterations ---
// When data is small (≤1024 elements), GPU kernel launch + D2H sync overhead
// exceeds the actual compute time. CPU avoids ~60μs/iteration overhead.

/// Log2 of the FRI last-layer size (number of raw evaluations stored in the proof).
/// After all FRI folds the last layer has 2^3 = 8 QM31 evaluations in BRT order.
pub const LOG_LAST_LAYER_DEGREE_BOUND: u32 = 3;

/// Bit-reverse a slice of QM31 values in place.
fn bit_reverse_qm31(v: &mut [QM31], log_n: u32) {
    let n = v.len();
    for i in 0..n {
        let j = {
            let mut result = 0usize;
            let mut val = i;
            for _ in 0..log_n {
                result = (result << 1) | (val & 1);
                val >>= 1;
            }
            result
        };
        if i < j {
            v.swap(i, j);
        }
    }
}

/// Convert the last-layer evaluations (bit-reversed order, from the fold chain) to
/// polynomial coefficients in the line-poly basis `{1, x, 2x²-1, x(2x²-1), ...}`.
///
/// Applies: bit-reverse → line IFFT using half_odds domain → normalize →
/// bit-reverse → truncate to `1 << LOG_LAST_LAYER_DEGREE_BOUND` coefficients.
///
/// Input evaluations must lie on a `half_odds` domain (the VortexSTARK line-fold
/// domain, matching stwo's FriVerifier convention).
pub fn last_layer_poly_coeffs(mut eval: Vec<QM31>) -> Vec<QM31> {
    let n = eval.len();
    assert!(n.is_power_of_two() && n >= 2);
    let log_n = n.ilog2();

    // Step 1: Bit-reverse to natural order.
    bit_reverse_qm31(&mut eval, log_n);

    // Step 2: Line IFFT on natural-order evaluations using half_odds domains.
    // At each level we apply ibutterfly(l[i], r[i], domain.at(i).x^{-1}).
    let mut chunk_size = n;
    let mut dom_log = log_n;
    while chunk_size > 1 {
        let domain = Coset::half_odds(dom_log);
        let half = chunk_size / 2;
        for chunk in eval.chunks_exact_mut(chunk_size) {
            let (l, r) = chunk.split_at_mut(half);
            for i in 0..half {
                let x_inv = domain.at(i).x.inverse();
                // ibutterfly: v0 = v0+v1, v1 = (v0-v1)*itwid
                let tmp = l[i];
                l[i] = tmp + r[i];
                r[i] = (tmp - r[i]) * x_inv;
            }
        }
        dom_log -= 1;
        chunk_size = half;
    }

    // Step 3: Normalize by 1/n.
    let n_inv = M31(n as u32).inverse();
    for v in eval.iter_mut() {
        *v = *v * n_inv;
    }
    // eval now contains bit-reversed polynomial coefficients.

    // Step 4: Bit-reverse to natural-order coefficients [c0, c1, c2, ..., c_{n-1}].
    bit_reverse_qm31(&mut eval, log_n);

    // Step 5: Truncate to degree bound.
    eval.truncate(1 << LOG_LAST_LAYER_DEGREE_BOUND);
    eval
}

/// Same as `last_layer_poly_coeffs` but without truncation — for degree diagnostics.
#[cfg(test)]
pub fn last_layer_poly_coeffs_full(mut eval: Vec<QM31>) -> Vec<QM31> {
    let n = eval.len();
    assert!(n.is_power_of_two() && n >= 2);
    let log_n = n.ilog2();
    bit_reverse_qm31(&mut eval, log_n);
    let mut chunk_size = n;
    let mut dom_log = log_n;
    while chunk_size > 1 {
        let domain = Coset::half_odds(dom_log);
        let half = chunk_size / 2;
        for chunk in eval.chunks_exact_mut(chunk_size) {
            let (l, r) = chunk.split_at_mut(half);
            for i in 0..half {
                let x_inv = domain.at(i).x.inverse();
                let tmp = l[i];
                l[i] = tmp + r[i];
                r[i] = (tmp - r[i]) * x_inv;
            }
        }
        dom_log -= 1;
        chunk_size = half;
    }
    let n_inv = M31(n as u32).inverse();
    for v in eval.iter_mut() { *v = *v * n_inv; }
    bit_reverse_qm31(&mut eval, log_n);
    eval // no truncation
}


/// Evaluate the last-layer line polynomial at a queried index.
///
/// `coeffs` are in NATURAL-ORDER from the line IFFT, using the LinePoly basis:
///   phi_0(x) = 1,  phi_1(x) = x,  phi_2(x) = 2x²-1,  phi_3(x) = x*(2x²-1), ...
/// Only the first `coeffs.len()` terms are evaluated; higher terms are assumed zero.
///
/// `last_idx` is the BRT (bit-reversed) index into the last-layer `half_odds` domain.
pub fn eval_last_layer_poly(coeffs: &[QM31], last_idx: usize, log_last_layer_size: u32) -> QM31 {
    let domain = Coset::half_odds(log_last_layer_size);
    let natural_idx = last_idx.reverse_bits() >> (usize::BITS - log_last_layer_size);
    let x: M31 = domain.at(natural_idx).x;

    // Evaluate using stwo's fold algorithm: recursively split coefficients and combine
    // with doublings [x, double_x(x), double_x(double_x(x)), ...].
    let n = coeffs.len();
    if n == 1 { return coeffs[0]; }

    let mut doublings = Vec::new();
    let mut xi = x;
    let log_n = n.ilog2();
    for _ in 0..log_n {
        doublings.push(xi);
        xi = M31(2) * xi * xi - M31::ONE; // double_x
    }

    fn fold_eval(brt_coeffs: &[QM31], doublings: &[M31]) -> QM31 {
        let n = brt_coeffs.len();
        if n == 1 { return brt_coeffs[0]; }
        let half = n / 2;
        let lhs = fold_eval(&brt_coeffs[..half], &doublings[1..]);
        let rhs = fold_eval(&brt_coeffs[half..], &doublings[1..]);
        lhs + rhs * doublings[0]
    }

    // BRT the natural-order coefficients for fold_eval (stwo's LinePoly convention).
    let mut brt = vec![QM31::ZERO; n];
    for i in 0..n { brt[i.reverse_bits() >> (usize::BITS - log_n)] = coeffs[i]; }
    fold_eval(&brt, &doublings)
}

/// FRI fold_line on CPU (small data path).
/// result[i] = (f0 + f1) + alpha * twiddle[i] * (f0 - f1)
pub fn fold_line_cpu(
    eval: &[QM31],
    alpha: QM31,
    twiddles: &[u32],
) -> Vec<QM31> {
    let half_n = eval.len() / 2;
    (0..half_n)
        .map(|i| {
            let f0 = eval[2 * i];
            let f1 = eval[2 * i + 1];
            let sum = f0 + f1;
            let diff = f0 - f1;
            let tw_diff = diff * M31(twiddles[i]);
            sum + alpha * tw_diff
        })
        .collect()
}

/// CPU Merkle root for QM31 SoA4 data. Returns 8-word Blake2s root.
/// Produces the same hashes as the GPU Merkle kernels.
pub fn merkle_root_cpu(values: &[QM31]) -> [u32; 8] {
    use crate::channel::{blake2s_hash, blake2s_hash_node};

    let n = values.len();
    assert!(n.is_power_of_two() && n >= 1);

    // Hash leaves: each QM31 → 4 u32 → 16-byte message → Blake2s
    let mut hashes: Vec<[u8; 32]> = values
        .iter()
        .map(|v| {
            let arr = v.to_u32_array();
            let mut input = [0u8; 64]; // padded to 64 bytes
            for (i, &w) in arr.iter().enumerate() {
                input[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            // Blake2s with t=16 (4 words × 4 bytes)
            // The blake2s_hash function uses input.len() as t, but we need t=16.
            // Pass a 16-byte slice to get the correct t value.
            blake2s_hash(&input[..16])
        })
        .collect();

    // Hash nodes bottom-up (internal node domain separation)
    while hashes.len() > 1 {
        let parent_count = hashes.len() / 2;
        hashes = (0..parent_count)
            .map(|i| {
                let mut input = [0u8; 64];
                input[..32].copy_from_slice(&hashes[2 * i]);
                input[32..64].copy_from_slice(&hashes[2 * i + 1]);
                blake2s_hash_node(&input)
            })
            .collect();
    }

    // Convert [u8; 32] → [u32; 8]
    let mut root = [0u32; 8];
    for i in 0..8 {
        root[i] = u32::from_le_bytes([
            hashes[0][i * 4],
            hashes[0][i * 4 + 1],
            hashes[0][i * 4 + 2],
            hashes[0][i * 4 + 3],
        ]);
    }
    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::M31;

    /// Helper: bit-reverse an index.
    fn br(x: usize, bits: u32) -> usize {
        let mut result = 0usize;
        let mut val = x;
        for _ in 0..bits {
            result = (result << 1) | (val & 1);
            val >>= 1;
        }
        result
    }

    /// Verify that `last_layer_poly_coeffs` correctly recovers a degree-1 polynomial
    /// from 8 BRT-ordered evaluations on `half_odds(3)`.
    #[test]
    fn test_last_layer_poly_round_trip() {
        let c0 = QM31::from_m31_array([M31(42), M31(0), M31(0), M31(0)]);
        let c1 = QM31::from_m31_array([M31(17), M31(0), M31(0), M31(0)]);
        let n = 8usize;
        let log_n = 3u32;
        let domain = Coset::half_odds(log_n);

        // Build BRT-ordered evaluations: brt_eval[j] = p(domain.at(br(j, log_n)).x)
        let brt_eval: Vec<QM31> = (0..n)
            .map(|j| {
                let x: M31 = domain.at(br(j, log_n)).x;
                c0 + c1 * x
            })
            .collect();

        let coeffs = last_layer_poly_coeffs(brt_eval.clone());
        assert_eq!(coeffs.len(), 1 << LOG_LAST_LAYER_DEGREE_BOUND, "wrong coefficient count");
        assert_eq!(coeffs[0], c0, "constant term mismatch");
        assert_eq!(coeffs[1], c1, "linear term mismatch");
        // Higher-order terms should be zero for a degree-1 polynomial.
        let zero = QM31::from_m31_array([M31(0), M31(0), M31(0), M31(0)]);
        for k in 2..coeffs.len() {
            assert_eq!(coeffs[k], zero, "coefficient c{k} should be zero");
        }

        // Also verify eval_last_layer_poly recovers the evaluations.
        for j in 0..n {
            let recovered = eval_last_layer_poly(&coeffs, j, log_n);
            assert_eq!(recovered, brt_eval[j], "eval mismatch at j={j}");
        }
    }

    /// Verify round-trip for a full degree-3 polynomial (all 4 LinePoly basis terms non-zero).
    #[test]
    fn test_last_layer_poly_round_trip_degree3() {
        let c0 = QM31::from_m31_array([M31(42), M31(0), M31(0), M31(0)]);
        let c1 = QM31::from_m31_array([M31(17), M31(0), M31(0), M31(0)]);
        let c2 = QM31::from_m31_array([M31(99), M31(0), M31(0), M31(0)]);
        let c3 = QM31::from_m31_array([M31(7), M31(0), M31(0), M31(0)]);
        let n = 8usize;
        let log_n = 3u32;
        let domain = Coset::half_odds(log_n);

        // Build BRT-ordered evaluations: p(x) = c0 + c1*x + c2*(2x²-1) + c3*x*(2x²-1)
        let brt_eval: Vec<QM31> = (0..n)
            .map(|j| {
                let x: M31 = domain.at(br(j, log_n)).x;
                let dx = M31(2) * x * x - M31::ONE;
                c0 + c1 * x + c2 * dx + c3 * x * dx
            })
            .collect();

        let coeffs = last_layer_poly_coeffs(brt_eval.clone());
        assert_eq!(coeffs.len(), 1 << LOG_LAST_LAYER_DEGREE_BOUND, "wrong coefficient count");
        assert_eq!(coeffs[0], c0, "c0 mismatch");
        assert_eq!(coeffs[1], c1, "c1 mismatch");
        if coeffs.len() > 2 { assert_eq!(coeffs[2], c2, "c2 mismatch"); }
        if coeffs.len() > 3 { assert_eq!(coeffs[3], c3, "c3 mismatch"); }
        // High coefficients should be zero for a degree-3 polynomial.
        for i in 4..coeffs.len() {
            assert_eq!(coeffs[i], QM31::ZERO, "coeff[{i}] should be zero for degree-3 poly");
        }

        for j in 0..n {
            let recovered = eval_last_layer_poly(&coeffs, j, log_n);
            assert_eq!(recovered, brt_eval[j], "eval mismatch at j={j}");
        }
    }

    #[test]
    fn test_fold_line_halves_size() {
        let n = 64;
        let log_n = 6u32;
        let domain = Coset::half_odds(log_n);

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
        let domain = Coset::half_odds(log_n);

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
