//! Out-Of-Domain Sampling (OODS) — stwo-compatible wire format extension.
//!
//! Implements the OODS protocol step for VortexSTARK:
//! after committing all trace trees, draw a random circle point z over QM31,
//! evaluate each committed polynomial at z and z_next = z·step,
//! and mix the evaluations (sampled_values) into the Fiat-Shamir channel.
//!
//! The resulting `sampled_values` field matches the structure of stwo's
//! `CommitmentSchemeProof.sampled_values`, enabling on-chain verifiability
//! by stwo-compatible verifier contracts.
//!
//! Protocol:
//!   1. All trace trees committed → mix roots into channel
//!   2. Draw OODS point z = (z_x, z_y) ∈ QM31 circle (channel.draw_oods_point())
//!   3. For each polynomial f_i committed in any tree:
//!        sampled_values[i][0] = f_i(z)
//!        sampled_values[i][1] = f_i(z_next)   (if column participates in step constraints)
//!   4. Mix all sampled_values into channel
//!   5. Verifier independently computes same evaluations at query points and checks FRI

use crate::field::{CM31, M31, QM31};
use crate::channel::Channel;
use crate::circle::CirclePoint;

// ─────────────────────────────────────────────────────────────────────────────
// OODS point: a circle point over QM31
// ─────────────────────────────────────────────────────────────────────────────

/// A circle point over the QM31 extension field.
/// Satisfies x² + y² = 1 over QM31 by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OodsPoint {
    pub x: QM31,
    pub y: QM31,
}

impl OodsPoint {
    /// Draw a random OODS point from the channel using the rational parameterization.
    ///
    /// Given t ← channel, computes:
    ///   z.x = (1 − t²) / (1 + t²)
    ///   z.y = 2t / (1 + t²)
    /// which satisfies z.x² + z.y² = 1 over QM31.
    pub fn from_channel(channel: &mut Channel) -> Self {
        let t = channel.draw_felt();
        let t2 = t * t;
        let denom_inv = (QM31::ONE + t2).inverse();
        let x = (QM31::ONE - t2) * denom_inv;
        let y = (t + t) * denom_inv;
        Self { x, y }
    }

    /// Encode as [z.x.to_u32_array(), z.y.to_u32_array()] — 8 u32 values.
    pub fn to_u32_array(&self) -> [u32; 8] {
        let xv = self.x.to_u32_array();
        let yv = self.y.to_u32_array();
        [xv[0], xv[1], xv[2], xv[3], yv[0], yv[1], yv[2], yv[3]]
    }

    /// Decode from 8 u32 values.
    pub fn from_u32_array(v: &[u32; 8]) -> Self {
        Self {
            x: QM31::from_u32_array([v[0], v[1], v[2], v[3]]),
            y: QM31::from_u32_array([v[4], v[5], v[6], v[7]]),
        }
    }

    /// Compute z_next = z * step — the OODS point for the "next row".
    ///
    /// `step` is the M31 circle group generator step for the trace domain:
    ///   step = CirclePoint::GENERATOR.repeated_double(31 - log_n)
    ///
    /// Circle mult over mixed (QM31, M31): (a.x, a.y) * (b.x, b.y) = (a.x·b.x − a.y·b.y, a.x·b.y + a.y·b.x)
    pub fn next_step(&self, step: CirclePoint) -> Self {
        let sx = qm31_from_m31(step.x);
        let sy = qm31_from_m31(step.y);
        Self {
            x: self.x * sx - self.y * sy,
            y: self.x * sy + self.y * sx,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Field helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Embed M31 into QM31 as the constant (m31 + 0i + 0u + 0iu).
#[inline]
pub fn qm31_from_m31(v: M31) -> QM31 {
    QM31::from_m31_array([v, M31::ZERO, M31::ZERO, M31::ZERO])
}

/// Circle group x-doubling over QM31: x → 2x² − 1.
#[inline]
pub fn double_x_qm31(x: QM31) -> QM31 {
    let x2 = x * x;
    x2 + x2 - QM31::ONE
}

/// Batch inverse of QM31 values using Montgomery's trick.
/// Returns vec of inverses; panics if any input is zero.
pub fn qm31_batch_inverse(values: &[QM31]) -> Vec<QM31> {
    let n = values.len();
    if n == 0 { return vec![]; }
    let mut prefix = vec![QM31::ZERO; n];
    prefix[0] = values[0];
    for i in 1..n {
        prefix[i] = prefix[i-1] * values[i];
    }
    let mut inv_acc = prefix[n-1].inverse();
    let mut result = vec![QM31::ZERO; n];
    for i in (1..n).rev() {
        result[i] = inv_acc * prefix[i-1];
        inv_acc = inv_acc * values[i];
    }
    result[0] = inv_acc;
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Polynomial evaluation at OODS point using the fold algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate a polynomial at OODS point z, given its coefficients in the circle
/// polynomial basis (output of INTT / `ntt::interpolate`).
///
/// Algorithm: stwo's `eval_at_point` / recursive `fold`.
/// O(n) time, O(log n) stack depth.
///
/// The coefficient ordering must match VortexSTARK's `ntt::interpolate` output,
/// which uses the same butterfly structure as stwo's Circle STARK NTT.
pub fn eval_at_oods_from_coeffs(coeffs: &[u32], z: OodsPoint) -> QM31 {
    let n = coeffs.len();
    assert!(n.is_power_of_two() && n >= 1, "coeffs must have power-of-two length");
    let log_n = n.trailing_zeros() as usize;
    if log_n == 0 {
        return qm31_from_m31(M31(coeffs[0]));
    }

    // Build folding factors in stwo order:
    //   mappings_raw = [z.y, z.x, double_x(z.x), double_x²(z.x), ...]  (log_n entries)
    //   mappings = mappings_raw.reversed()  =  [..., double_x(z.x), z.x, z.y]
    //
    // fold(values, [f_{k-1}, ..., f_0]):
    //   split into halves (L, R)
    //   L' = fold(L, [f_{k-2}, ..., f_0])
    //   R' = fold(R, [f_{k-2}, ..., f_0])
    //   return L' + f_{k-1} * R'
    let mut mappings_raw = Vec::with_capacity(log_n);
    mappings_raw.push(z.y);
    let mut x = z.x;
    for _ in 1..log_n {
        mappings_raw.push(x);
        x = double_x_qm31(x);
    }
    // Reverse to get outermost factor first (matches stwo's `fold` call convention)
    mappings_raw.reverse();

    fold_m31(coeffs, &mappings_raw)
}

/// Evaluate a QM31 polynomial (4 interleaved M31 limbs per row, SoA layout:
/// [limb0[0..n], limb1[0..n], limb2[0..n], limb3[0..n]]) at OODS point z.
///
/// Used for interaction traces (LogUp, RC, dict) and the quotient polynomial.
pub fn eval_qm31_col_at_oods(coeffs_soa: &[[u32; 4]], z: OodsPoint) -> QM31 {
    // coeffs_soa[i] = [limb0, limb1, limb2, limb3] for coefficient i
    let n = coeffs_soa.len();
    assert!(n.is_power_of_two() && n >= 1);
    let log_n = n.trailing_zeros() as usize;

    if log_n == 0 {
        return QM31::from_u32_array(coeffs_soa[0]);
    }

    let mut mappings_raw = Vec::with_capacity(log_n);
    mappings_raw.push(z.y);
    let mut x = z.x;
    for _ in 1..log_n {
        mappings_raw.push(x);
        x = double_x_qm31(x);
    }
    mappings_raw.reverse();

    fold_qm31(coeffs_soa, &mappings_raw)
}

/// Recursive fold on M31 coefficients with QM31 folding factors.
///
/// Matches stwo's `fold<M31, SecureField>(values, folding_factors)`:
///   fold([c0..c_{n-1}], [f_{k-1}, ..., f_0]):
///     left  = fold(c0..c_{n/2-1},  [f_{k-2},..,f_0])
///     right = fold(c_{n/2}..c_{n-1}, [f_{k-2},..,f_0])
///     return left + f_{k-1} * right
fn fold_m31(values: &[u32], factors: &[QM31]) -> QM31 {
    if values.len() == 1 {
        return qm31_from_m31(M31(values[0]));
    }
    debug_assert_eq!(values.len(), 1 << factors.len(), "fold: length/factor mismatch");
    let half = values.len() / 2;
    let (first, rest) = factors.split_first().unwrap();
    let left  = fold_m31(&values[..half], rest);
    let right = fold_m31(&values[half..], rest);
    left + *first * right
}

/// Recursive fold on QM31 coefficients (AoS layout: [u32;4] per coeff).
fn fold_qm31(values: &[[u32; 4]], factors: &[QM31]) -> QM31 {
    if values.len() == 1 {
        return QM31::from_u32_array(values[0]);
    }
    debug_assert_eq!(values.len(), 1 << factors.len());
    let half = values.len() / 2;
    let (first, rest) = factors.split_first().unwrap();
    let left  = fold_qm31(&values[..half], rest);
    let right = fold_qm31(&values[half..], rest);
    left + *first * right
}

// ─────────────────────────────────────────────────────────────────────────────
// OODS sampled values
// ─────────────────────────────────────────────────────────────────────────────

/// OODS-sampled values for one column at a single sample point.
pub type OodsSample = QM31;

/// All OODS sampled values for the VortexSTARK Cairo proof.
///
/// Structure mirrors stwo's `TreeVec<ColumnVec<Vec<SecureField>>>`:
///   - Outer: per committed tree (trace_lo, trace_hi, trace_dict, interaction, rc, dict_main, quotient)
///   - Middle: per column in the tree
///   - Inner: per sample point (curr = z, next = z_next)
///
/// For trace columns: 2 sample points (z and z_next) — both current and next row.
/// For quotient columns: 1 sample point (z) — no next-row dependency.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OodsSampledValues {
    /// OODS point z (the base sample point).
    pub z: OodsPoint,
    /// z_next = z * trace_step (one row forward in the trace domain).
    pub z_next: OodsPoint,
    /// Trace columns sampled at z.  Length = N_COLS = 34.
    pub trace_at_z:      Vec<QM31>,
    /// Trace columns sampled at z_next. Length = N_COLS = 34.
    pub trace_at_z_next: Vec<QM31>,
    /// LogUp interaction columns sampled at z. Length = 4 (one QM31 per row).
    pub interaction_at_z:      Vec<QM31>,
    /// LogUp interaction columns sampled at z_next.
    pub interaction_at_z_next: Vec<QM31>,
    /// RC interaction columns sampled at z.
    pub rc_at_z:      Vec<QM31>,
    /// RC interaction columns sampled at z_next.
    pub rc_at_z_next: Vec<QM31>,
    /// Dict main interaction columns sampled at z.
    pub dict_main_at_z:      Vec<QM31>,
    /// Dict main interaction columns sampled at z_next.
    pub dict_main_at_z_next: Vec<QM31>,
}

impl OodsSampledValues {
    /// Flatten all sampled values into a single Vec for Fiat-Shamir mixing.
    pub fn flatten_for_channel(&self) -> Vec<QM31> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.trace_at_z);
        out.extend_from_slice(&self.trace_at_z_next);
        out.extend_from_slice(&self.interaction_at_z);
        out.extend_from_slice(&self.interaction_at_z_next);
        out.extend_from_slice(&self.rc_at_z);
        out.extend_from_slice(&self.rc_at_z_next);
        out.extend_from_slice(&self.dict_main_at_z);
        out.extend_from_slice(&self.dict_main_at_z_next);
        out
    }
}

/// Evaluate the trace domain vanishing polynomial Z_H at an OODS point.
///
/// For the half_coset(log_n) trace domain:
///   Z_H(p) = coset_vanishing(half_coset(log_n), p) = T_{n/2}(p.x)
///            = double_x^{log_n - 1}(p.x)
///
/// This matches `Coset::coset_vanishing_at(&Coset::half_coset(log_n), p)` for M31 points.
/// Note: circle_vanishing_poly_at = T_n(x)+1 only vanishes at n/2 trace points — wrong V_H.
pub fn oods_vanishing(z: OodsPoint, log_n: u32) -> QM31 {
    let mut v = z.x;
    for _ in 1..log_n {
        v = double_x_qm31(v);
    }
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// Channel extension: draw OODS point
// ─────────────────────────────────────────────────────────────────────────────

/// Draw a random OODS point from the channel.
/// Convenience wrapper around `OodsPoint::from_channel`.
pub fn draw_oods_point(channel: &mut Channel) -> OodsPoint {
    OodsPoint::from_channel(channel)
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities for interaction trace (QM31 columns in SoA layout)
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a Vec<[u32;4]> (QM31 SoA, n rows × 4 limbs) into coefficient form.
///
/// The interaction traces are stored as evaluations at the trace domain.
/// INTT converts them to coefficients for OODS evaluation.
///
/// NOTE: For the current implementation, we skip INTT on interaction traces
/// and return the raw values. The verifier must use the same convention.
/// Full INTT support requires a CPU-side circle INTT implementation.
pub fn interaction_coeffs_from_evals(evals_soa: Vec<[u32; 4]>) -> Vec<[u32; 4]> {
    // TODO: apply CPU circle INTT once a CPU implementation is available.
    // For Phase 1, return the raw evaluation values (treated as "coefficients"
    // in a basis that the verifier must use consistently).
    evals_soa
}

// ─────────────────────────────────────────────────────────────────────────────
// Line coefficients for OODS quotient computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the (a, b, c) line coefficients for the OODS quotient term at a sample point.
///
/// For a real polynomial f (M31-valued) sampled at OODS point z with value v:
///   The "line" through (z.y, v) and (conj(z).y, conj(v)) is: L(t) = a + b*t
///   where:
///     b = (v − conj(v)) / (z.y − conj(z).y) = (v − conj(v)) / (2 * z.y)
///     a = v − b * z.y
///   The "c coefficient" for the rational formula is c = 1 (for M31 columns),
///   or the line's coefficient relative to the accumulation alpha.
///
/// Returns (a, b) where the numerator for column i at domain point p is:
///   c_i * f_i(p) − (a_i + b_i * p.y)
///
/// For use with `cuda_accumulate_numerators`.
pub fn compute_line_coeffs(z: OodsPoint, v: QM31) -> (QM31, QM31) {
    let v_conj = v.conjugate();
    let z_y_conj = z.y.conjugate();
    let denom = z.y - z_y_conj;  // = 2 * z.y (imaginary part doubled)
    let b = (v - v_conj) * denom.inverse();
    let a = v - b * z.y;
    (a, b)
}

/// Compute the OODS vanishing denominator at a domain point p for sample point sp.
///
/// From the Circle STARK quotient formula:
///   D(sp, p) = (Re(sp.x) - p.x) * Im(sp.y) - (Re(sp.y) - p.y) * Im(sp.x)
///
/// where Re(q) = q.a (CM31 real part) and Im(q) = q.b (CM31 u-coefficient) for QM31 q.
/// p.x, p.y are M31 coordinates of the eval-domain point.
///
/// Returns a CM31 value. The full OODS quotient contribution at p for sample point sp is:
///   full_numer(p) * oods_denom(sp, p.x, p.y)^{-1}
#[inline]
pub fn oods_denom(sp: OodsPoint, px: M31, py: M31) -> CM31 {
    let prx = sp.x.a;  // Re(sp.x) — CM31
    let pry = sp.y.a;  // Re(sp.y) — CM31
    let pix = sp.x.b;  // Im(sp.x) — CM31
    let piy = sp.y.b;  // Im(sp.y) — CM31
    let term1 = (prx - CM31::new(px, M31::ZERO)) * piy;
    let term2 = (pry - CM31::new(py, M31::ZERO)) * pix;
    term1 - term2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::Channel;

    #[test]
    fn oods_point_on_circle() {
        let mut ch = Channel::new();
        ch.mix_digest(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let z = OodsPoint::from_channel(&mut ch);
        // Verify z.x^2 + z.y^2 == 1 over QM31
        let norm = z.x * z.x + z.y * z.y;
        assert_eq!(norm, QM31::ONE, "OODS point should satisfy x^2 + y^2 = 1");
    }

    #[test]
    fn oods_point_roundtrip() {
        let mut ch = Channel::new();
        let z = OodsPoint::from_channel(&mut ch);
        let arr = z.to_u32_array();
        let z2 = OodsPoint::from_u32_array(&arr);
        assert_eq!(z, z2);
    }

    #[test]
    fn double_x_qm31_consistent_with_m31() {
        // double_x over QM31 should be consistent with double_x over M31
        // when the input is in the M31 subfield (y=0, imaginary parts = 0)
        let x_m31 = M31(12345);
        let x_qm31 = qm31_from_m31(x_m31);
        let result = double_x_qm31(x_qm31);
        let expected_m31 = M31(2) * x_m31 * x_m31 - M31::ONE;
        let expected = qm31_from_m31(expected_m31);
        assert_eq!(result, expected, "double_x over QM31 should match M31 computation");
    }

    #[test]
    fn fold_constant_poly() {
        // A constant polynomial (all coefficients except the first are zero)
        // should evaluate to that constant at any point.
        let n = 4;
        let mut coeffs = vec![0u32; n];
        coeffs[0] = 42; // constant = 42
        let mut ch = Channel::new();
        let z = OodsPoint::from_channel(&mut ch);
        let result = eval_at_oods_from_coeffs(&coeffs, z);
        // For a constant polynomial c, fold([c, 0, 0, 0], [f2, f1, f0]) = c + f2*0 + ... = c
        assert_eq!(result, qm31_from_m31(M31(42)));
    }

    #[test]
    fn oods_next_step_on_circle() {
        let mut ch = Channel::new();
        let z = OodsPoint::from_channel(&mut ch);
        let step = CirclePoint::GENERATOR.repeated_double(11); // step for log_n=20
        let z_next = z.next_step(step);
        let norm = z_next.x * z_next.x + z_next.y * z_next.y;
        assert_eq!(norm, QM31::ONE, "z_next should also be on the QM31 circle");
    }

    #[test]
    fn batch_inverse_roundtrip() {
        let vals: Vec<QM31> = (1..=8).map(|i| qm31_from_m31(M31(i))).collect();
        let invs = qm31_batch_inverse(&vals);
        for (v, inv) in vals.iter().zip(invs.iter()) {
            let prod = *v * *inv;
            assert_eq!(prod, QM31::ONE, "batch_inverse: v * inv != 1");
        }
    }
}
