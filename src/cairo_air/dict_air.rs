//! Dict sub-AIR: execution-order and key-sorted access traces + LogUp interaction traces.
//!
//! ## Sub-AIR structure
//!
//! **Exec trace** (3 cols, `dict_n` rows): execution-order access log
//!   - col 0: key
//!   - col 1: prev_value
//!   - col 2: new_value
//!   Padded with (0,0,0) rows.
//!
//! **Sorted trace** (4 cols, `dict_n` rows): key-sorted access log
//!   - col 0: key
//!   - col 1: prev_value
//!   - col 2: new_value
//!   - col 3: is_first (1 if first access to this key group, 0 otherwise)
//!   Padded with (0,0,0,1) rows — padding groups are each treated as a new key group.
//!
//! **Interaction traces** (4 cols each, `dict_n` rows): LogUp running sums (QM31 SoA).
//!   Row i stores `S[i] = Σ_{j<i} δ(data[j])` — exclusive prefix sum.
//!   Row 0 is always S[0] = 0.
//!   `final_sum` = S[dict_n] = total sum over all dict_n rows (real + padding).
//!
//! ## Step-transition constraints (on sorted trace)
//!
//! Checked at every consecutive pair (i, i+1) for 0 ≤ i < dict_n − 1:
//!   - C0: `is_first[i] * (1 − is_first[i]) = 0` (binary)
//!   - C1: `(1 − is_first[i+1]) * (key[i+1] − key[i]) = 0` (same group → same key)
//!   - C2: `is_first[i+1] * prev[i+1] = 0` (new group → prev = 0)
//!   - C3: `(1 − is_first[i+1]) * (prev[i+1] − new[i]) = 0` (chain within group)
//!
//! ## Interaction trace step-transition
//!
//! At position i: `S[i+1] − S[i] = δ(data[i])`
//! where `δ(key, prev, new) = 1 / (z − (key + α·prev + α²·new))`.
//!
//! ## Statistical soundness
//!
//! Without FRI the constraints are checked at N_QUERIES random positions (same as the
//! main STARK's query set, mapped mod dict_n). For dict_n ≤ D and N_QUERIES = Q the
//! probability of NOT detecting a single constraint violation at a random row is
//! `(1 − 1/D)^Q`. See AUDIT.md GAP-1 for the full soundness discussion.

use crate::field::{M31, QM31};
use super::logup::qm31_from_m31;

/// Number of columns in the exec dict trace.
pub const EXEC_COLS: usize = 3;
/// Number of columns in the sorted dict trace.
pub const SORTED_COLS: usize = 4;

// M31 modulus
const P: u64 = 0x7FFF_FFFF;

// ---- Trace builders --------------------------------------------------------

/// Build raw exec and sorted dict trace columns from the execution-order access log.
///
/// Returns `(exec_cols, sorted_cols, dict_log_n)` where each `*_cols` is a
/// `Vec<Vec<u32>>` with `EXEC_COLS`/`SORTED_COLS` elements, each of length
/// `dict_n = 1 << dict_log_n`.
pub fn build_dict_raw_traces(
    accesses: &[(usize, u64, u64, u64)],
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, u32) {
    let n = accesses.len();
    assert!(n > 0, "build_dict_raw_traces called with empty access log");

    let dict_log_n = (n as f64).log2().ceil() as u32;
    let dict_n = 1usize << dict_log_n;

    // ---- Exec trace (execution order) ----
    let mut exec_key  = vec![0u32; dict_n];
    let mut exec_prev = vec![0u32; dict_n];
    let mut exec_new  = vec![0u32; dict_n];
    for (i, &(_step, k, p, nv)) in accesses.iter().enumerate() {
        exec_key[i]  = k  as u32;
        exec_prev[i] = p  as u32;
        exec_new[i]  = nv as u32;
    }

    // ---- Sorted trace (key-sorted) ----
    // Strip step index for sorting; sort by key (stable → within same key, execution order).
    let mut sorted: Vec<(u64, u64, u64)> = accesses.iter()
        .map(|&(_step, k, p, nv)| (k, p, nv))
        .collect();
    sorted.sort_by_key(|&(k, _, _)| k);

    let mut sorted_key      = vec![0u32; dict_n];
    let mut sorted_prev_col = vec![0u32; dict_n];
    let mut sorted_new_col  = vec![0u32; dict_n];
    // Padding rows beyond the real accesses use is_first=1.  This is safe: the verifier
    // checks ALL sorted step-transition constraints (C0-C3) for every pair (i, i+1).
    // For padding rows with is_first=1:
    //   C2: is_first[i+1] * prev[i+1] = 1 * 0 = 0  ✓ (prev=0 for any first-access row)
    //   C1: (1-is_first[i+1]) * (key[i+1]-key[i]) = 0 * ... = 0  ✓ trivially
    // So padding with is_first=1 cannot be exploited to forge dict accesses.
    let mut sorted_is_first = vec![1u32; dict_n];

    for (i, &(k, p, nv)) in sorted.iter().enumerate() {
        sorted_key[i]      = k  as u32;
        sorted_prev_col[i] = p  as u32;
        sorted_new_col[i]  = nv as u32;
        sorted_is_first[i] = if i == 0 || sorted[i - 1].0 != k { 1 } else { 0 };
    }

    let exec_cols   = vec![exec_key, exec_prev, exec_new];
    let sorted_cols = vec![sorted_key, sorted_prev_col, sorted_new_col, sorted_is_first];
    (exec_cols, sorted_cols, dict_log_n)
}

/// Build a LogUp interaction trace (exclusive prefix sums) for a 3-column data trace.
///
/// Each row `i` of the returned columns stores `S[i] = Σ_{j<i} δ(data[j])` (so
/// row 0 is S[0] = 0, row 1 is S[1] = δ(data[0]), ...).
///
/// Returns `(interact_cols, final_sum)` where `interact_cols` is a `Vec<Vec<u32>>`
/// with 4 elements (QM31 SoA layout), and `final_sum = S[dict_n]` is the total sum.
pub fn build_dict_interaction_trace(
    key_col:  &[u32],
    prev_col: &[u32],
    new_col:  &[u32],
    z:     QM31,
    alpha: QM31,
) -> (Vec<Vec<u32>>, QM31) {
    let n = key_col.len();
    assert!(n > 0 && n == prev_col.len() && n == new_col.len());

    let alpha_sq = alpha * alpha;
    let mut cols: Vec<Vec<u32>> = vec![vec![0u32; n]; 4]; // initialised to 0 = S[0]
    let mut s = QM31::ZERO;

    for i in 0..n {
        // Write S[i] into row i
        let arr = s.to_u32_array();
        cols[0][i] = arr[0];
        cols[1][i] = arr[1];
        cols[2][i] = arr[2];
        cols[3][i] = arr[3];

        // Advance: S[i+1] = S[i] + 1 / (z − entry(data[i]))
        let k  = M31(key_col[i]);
        let p  = M31(prev_col[i]);
        let nv = M31(new_col[i]);
        let entry = qm31_from_m31(k)
            + alpha    * qm31_from_m31(p)
            + alpha_sq * qm31_from_m31(nv);
        let denom = z - entry;
        assert!(denom != QM31::ZERO,
            "dict interact: LogUp denominator is zero at row {i} — Fiat-Shamir collision");
        s = s + denom.inverse();
    }
    // s == S[n] == final_sum
    (cols, s)
}

// ---- Constraint evaluator --------------------------------------------------

/// Evaluate sorted trace step-transition constraints at a consecutive row pair.
///
/// `curr = [key_i, prev_i, new_i, is_first_i]`
/// `next = [key_{i+1}, prev_{i+1}, new_{i+1}, is_first_{i+1}]`
///
/// Returns `[C0, C1, C2, C3]` — all should be 0 for a valid trace (mod P).
pub fn eval_sorted_constraints(curr: [u32; 4], next: [u32; 4]) -> [u32; 4] {
    let [ck, _cp, cn, cif] = curr.map(|x| x as u64);
    let [nk, np, _nn, nif] = next.map(|x| x as u64);

    // C0: is_first[i] * (1 − is_first[i]) = 0  (binary check)
    let c0 = mul_m31(cif, sub_m31(1, cif));
    // C1: (1 − is_first[i+1]) * (key[i+1] − key[i]) = 0  (same group → same key)
    let c1 = mul_m31(sub_m31(1, nif), sub_m31(nk, ck));
    // C2: is_first[i+1] * prev[i+1] = 0  (new group → prev = 0)
    let c2 = mul_m31(nif, np);
    // C3: (1 − is_first[i+1]) * (prev[i+1] − new[i]) = 0  (chain: prev = last new)
    let c3 = mul_m31(sub_m31(1, nif), sub_m31(np, cn));
    [c0 as u32, c1 as u32, c2 as u32, c3 as u32]
}

/// Evaluate the interact step-transition constraint at position `d`.
///
/// Returns `Ok(())` if `s_next − s_curr = δ(key, prev, new)`, else an error string.
pub fn verify_interact_step(
    key: u32, prev: u32, new_val: u32,
    s_curr: [u32; 4],
    s_next: [u32; 4],
    z: QM31,
    alpha: QM31,
) -> Result<(), String> {
    let alpha_sq = alpha * alpha;
    let k  = M31(key);
    let p  = M31(prev);
    let nv = M31(new_val);
    let entry = qm31_from_m31(k)
        + alpha    * qm31_from_m31(p)
        + alpha_sq * qm31_from_m31(nv);
    let denom = z - entry;
    if denom == QM31::ZERO {
        return Err("dict interact: denominator zero at query position".into());
    }
    let expected_delta = denom.inverse();
    let sc = QM31::from_u32_array(s_curr);
    let sn = QM31::from_u32_array(s_next);
    if sn - sc != expected_delta {
        return Err(format!(
            "dict interact step-transition failed: s_next-s_curr={:?} expected_delta={:?}",
            (sn - sc), expected_delta
        ));
    }
    Ok(())
}

// ---- M31 arithmetic helpers ------------------------------------------------

#[inline(always)]
fn sub_m31(a: u64, b: u64) -> u64 { (a + P - b % P) % P }

#[inline(always)]
fn mul_m31(a: u64, b: u64) -> u64 { (a * b) % P }

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::cm31::CM31;

    fn q(a: u32, b: u32, c: u32, d: u32) -> QM31 {
        QM31 { a: CM31 { a: M31(a), b: M31(b) }, b: CM31 { a: M31(c), b: M31(d) } }
    }

    #[test]
    fn test_build_raw_traces_single() {
        let accesses = [(0usize, 1u64, 0u64, 42u64)];
        let (exec, sorted, log_n) = build_dict_raw_traces(&accesses);
        assert_eq!(log_n, 0); // 1 access → 1 << 0 = 1 row
        assert_eq!(exec[0][0], 1);   // key
        assert_eq!(exec[1][0], 0);   // prev
        assert_eq!(exec[2][0], 42);  // new
        assert_eq!(sorted[3][0], 1); // is_first = 1 for first (only) access
    }

    #[test]
    fn test_build_raw_traces_two_keys() {
        let accesses = [(0usize, 1u64, 0u64, 10u64), (1, 2, 0, 20)];
        let (exec, sorted, log_n) = build_dict_raw_traces(&accesses);
        assert_eq!(log_n, 1); // 2 accesses → 1 << 1 = 2 rows
        // Exec: execution order
        assert_eq!(exec[0], &[1, 2]); // keys
        // Sorted: sorted by key, is_first=1 for each new key
        assert_eq!(sorted[0], &[1, 2]); // sorted keys
        assert_eq!(sorted[3], &[1, 1]); // both are first in their group
    }

    #[test]
    fn test_build_raw_traces_repeated_key() {
        // key=1 has two accesses: 0→42 then 42→99
        let accesses = [(0usize, 1u64, 0u64, 42u64), (1, 2, 0, 7), (2, 1, 42, 99)];
        let (_, sorted, log_n) = build_dict_raw_traces(&accesses);
        assert_eq!(log_n, 2); // 3 accesses → ceil(log2(3)) = 2, dict_n = 4

        // Sorted by key: key=1 (×2), key=2 (×1), padding (×1)
        // key=1 first access: is_first=1; second access: is_first=0
        // key=2 first access: is_first=1
        // padding row: is_first=1
        assert_eq!(sorted[0][0], 1); // sorted key[0] = 1
        assert_eq!(sorted[0][1], 1); // sorted key[1] = 1
        assert_eq!(sorted[0][2], 2); // sorted key[2] = 2
        assert_eq!(sorted[0][3], 0); // padding key = 0
        assert_eq!(sorted[3][0], 1); // is_first[0] = 1 (first access to key 1)
        assert_eq!(sorted[3][1], 0); // is_first[1] = 0 (second access to key 1)
        assert_eq!(sorted[3][2], 1); // is_first[2] = 1 (first access to key 2)
        assert_eq!(sorted[3][3], 1); // is_first[3] = 1 (padding)
    }

    #[test]
    fn test_interaction_trace_zero_start() {
        let keys  = vec![1u32, 2u32];
        let prevs = vec![0u32, 0u32];
        let news  = vec![10u32, 20u32];
        let z     = q(100, 200, 300, 400);
        let alpha = q(500, 600, 700, 800);

        let (cols, _final) = build_dict_interaction_trace(&keys, &prevs, &news, z, alpha);
        // Row 0 is S[0] = 0
        assert_eq!(cols[0][0], 0);
        assert_eq!(cols[1][0], 0);
        assert_eq!(cols[2][0], 0);
        assert_eq!(cols[3][0], 0);
    }

    #[test]
    fn test_interaction_trace_step_transitions() {
        let keys  = vec![1u32, 2u32, 1u32];
        let prevs = vec![0u32, 0u32, 10u32];
        let news  = vec![10u32, 20u32, 30u32];
        let z     = q(12345, 67890, 11111, 22222);
        let alpha = q(33333, 44444, 55555, 66666);

        let (cols, final_sum) = build_dict_interaction_trace(&keys, &prevs, &news, z, alpha);
        let n = keys.len();

        // Verify every step transition
        for i in 0..n - 1 {
            let s_curr = [cols[0][i], cols[1][i], cols[2][i], cols[3][i]];
            let s_next = [cols[0][i+1], cols[1][i+1], cols[2][i+1], cols[3][i+1]];
            verify_interact_step(keys[i], prevs[i], news[i], s_curr, s_next, z, alpha)
                .unwrap_or_else(|e| panic!("step {i}: {e}"));
        }

        // Verify last step: S[n-1] + delta(data[n-1]) = final_sum
        let s_last = [cols[0][n-1], cols[1][n-1], cols[2][n-1], cols[3][n-1]];
        let final_arr = final_sum.to_u32_array();
        verify_interact_step(
            keys[n-1], prevs[n-1], news[n-1],
            s_last, final_arr, z, alpha
        ).expect("final step transition");
    }

    #[test]
    fn test_exec_sorted_sums_equal() {
        // Both exec and sorted logs are permutations of the same multiset,
        // so their LogUp sums must be equal (permutation argument).
        let accesses = [(0usize, 1u64, 0u64, 10u64), (1, 2, 0, 20), (2, 1, 10, 30)];
        let (exec_cols, sorted_cols, _log_n) = build_dict_raw_traces(&accesses);
        let z     = q(98765, 43210, 11111, 99999);
        let alpha = q(22222, 33333, 44444, 55555);

        let (_, exec_final) = build_dict_interaction_trace(
            &exec_cols[0], &exec_cols[1], &exec_cols[2], z, alpha);
        let (_, sorted_final) = build_dict_interaction_trace(
            &sorted_cols[0], &sorted_cols[1], &sorted_cols[2], z, alpha);

        assert_eq!(exec_final, sorted_final,
            "exec and sorted LogUp sums must be equal (permutation argument)");
    }

    #[test]
    fn test_sorted_constraints_valid() {
        let accesses = [(0usize, 1u64, 0u64, 10u64), (1, 1, 10, 30), (2, 2, 0, 20)];
        let (_, sorted_cols, _) = build_dict_raw_traces(&accesses);
        let n = sorted_cols[0].len();
        for i in 0..n - 1 {
            let curr = [sorted_cols[0][i], sorted_cols[1][i],
                        sorted_cols[2][i], sorted_cols[3][i]];
            let next = [sorted_cols[0][i+1], sorted_cols[1][i+1],
                        sorted_cols[2][i+1], sorted_cols[3][i+1]];
            let cs = eval_sorted_constraints(curr, next);
            assert_eq!(cs, [0, 0, 0, 0],
                "sorted constraint violation at row {i}: {:?}", cs);
        }
    }

    #[test]
    fn test_sorted_constraint_c0_violation() {
        // is_first = 2 → C0 = 2*(1-2) = 2*(-1) ≠ 0
        let curr = [0u32, 0, 0, 2]; // is_first = 2 (not binary)
        let next = [0u32, 0, 0, 1];
        let cs = eval_sorted_constraints(curr, next);
        assert_ne!(cs[0], 0, "C0 should detect non-binary is_first");
    }

    #[test]
    fn test_sorted_constraint_c1_violation() {
        // Within same group (next.is_first=0), key should be same as curr.
        // If next.key ≠ curr.key → C1 ≠ 0.
        let curr = [5u32, 0, 10, 1];
        let next = [7u32, 0, 0, 0]; // same group but different key — violation
        let cs = eval_sorted_constraints(curr, next);
        assert_ne!(cs[1], 0, "C1 should detect key change within group");
    }

    #[test]
    fn test_sorted_constraint_c2_violation() {
        // New group (next.is_first=1) should have prev=0.
        let curr = [1u32, 0, 10, 1];
        let next = [2u32, 5, 0, 1]; // new group but prev=5 — violation
        let cs = eval_sorted_constraints(curr, next);
        assert_ne!(cs[2], 0, "C2 should detect non-zero prev at group start");
    }

    #[test]
    fn test_sorted_constraint_c3_violation() {
        // Within same group, next.prev should equal curr.new.
        let curr = [1u32, 0, 10, 1];
        let next = [1u32, 5, 20, 0]; // same group but prev=5 ≠ curr.new=10
        let cs = eval_sorted_constraints(curr, next);
        assert_ne!(cs[3], 0, "C3 should detect chain break within group");
    }
}
