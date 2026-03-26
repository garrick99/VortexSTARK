//! Dict consistency argument for Cairo felt252 dictionaries.
//!
//! # What this module provides
//!
//! **CPU-side chain verification** (`verify_chain`): checks that the ordered access log
//! produced by hint execution is internally consistent — for each key, each access's
//! `prev_value` matches the preceding access's `new_value`. Detects hint execution bugs
//! before they propagate into a proof.
//!
//! **LogUp helper functions** (`dict_logup_exec_sum`, `dict_logup_table_sum`): compute
//! the LogUp interaction sums over dict access triples. These are the building blocks for
//! a future STARK-level dict consistency argument.
//!
//! # Current soundness status
//!
//! The CPU chain check runs at prove time and will reject an inconsistent dict access log
//! with `ProveError::DictConsistencyViolation`. This protects against honest prover bugs.
//!
//! A *malicious* prover could still bypass this check (it runs outside the STARK). Full
//! STARK-level dict consistency requires dedicated trace columns (`dict_key`, `dict_prev`,
//! `dict_new`) committed before the Fiat-Shamir challenges, plus a LogUp argument wiring
//! those columns into the transcript. That is planned future work.
//!
//! # Access log format
//!
//! Each entry is `(key, prev_value, new_value)` in execution order as logged by
//! `Felt252DictEntryUpdate`. All values are u64 (M31-truncated felt252).

use std::collections::HashMap;
use crate::field::{M31, QM31};
use super::logup::qm31_from_m31;

/// Error type for dict consistency violations.
#[derive(Debug, Clone, PartialEq)]
pub enum DictConsistencyError {
    /// An access's prev_value does not match the preceding new_value for the same key.
    ChainViolation {
        key: u64,
        expected_prev: u64,
        actual_prev: u64,
    },
}

impl std::fmt::Display for DictConsistencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DictConsistencyError::ChainViolation { key, expected_prev, actual_prev } =>
                write!(f, "dict chain violation at key {key:#x}: \
                           expected prev_value={expected_prev:#x}, got {actual_prev:#x}"),
        }
    }
}

/// Verify that the dict access log forms a valid chain.
///
/// For each key, accesses must satisfy: `prev_value[i+1] == new_value[i]`.
/// The first access to any key must have `prev_value == 0` (dict default).
///
/// This is a CPU-side check only — it does not constrain anything in the STARK.
pub fn verify_chain(accesses: &[(usize, u64, u64, u64)]) -> Result<(), DictConsistencyError> {
    // last_new[key] = new_value of the most recent access to that key.
    // Absence = key never accessed before (expected prev = 0).
    let mut last_new: HashMap<u64, u64> = HashMap::new();

    for &(_step, key, prev_value, new_value) in accesses {
        let expected_prev = last_new.get(&key).copied().unwrap_or(0);
        if prev_value != expected_prev {
            return Err(DictConsistencyError::ChainViolation {
                key,
                expected_prev,
                actual_prev: prev_value,
            });
        }
        last_new.insert(key, new_value);
    }
    Ok(())
}

/// Compute the execution-side LogUp sum over the dict access log.
///
/// Each access triple (key, prev, new) contributes `+1 / (z - entry)` where
///   `entry = key + alpha * prev + alpha^2 * new`
///
/// The execution sum and table sum cancel iff the access log is a valid permutation
/// of the table entries. This is the building block for a STARK-level dict argument.
pub fn dict_logup_exec_sum(
    accesses: &[(usize, u64, u64, u64)],
    z: QM31,
    alpha: QM31,
) -> QM31 {
    let alpha_sq = alpha * alpha;
    let mut sum = QM31::ZERO;
    for &(_step, key, prev, new_val) in accesses {
        let entry = qm31_from_m31(M31(key as u32))
            + alpha * qm31_from_m31(M31(prev as u32))
            + alpha_sq * qm31_from_m31(M31(new_val as u32));
        let denom = z - entry;
        debug_assert!(denom != QM31::ZERO, "dict LogUp denominator zero — Fiat-Shamir collision");
        sum = sum + denom.inverse();
    }
    sum
}

/// Compute the table-side LogUp sum (negated, with multiplicities).
///
/// The table is the multiset of unique (key, prev, new) triples from the access log,
/// each weighted by its multiplicity. When added to the execution sum, the total is zero.
pub fn dict_logup_table_sum(
    accesses: &[(usize, u64, u64, u64)],
    z: QM31,
    alpha: QM31,
) -> QM31 {
    use std::collections::BTreeMap;
    let alpha_sq = alpha * alpha;

    // Count multiplicities of each unique (key, prev, new) triple (step is irrelevant for table).
    let mut counts: BTreeMap<(u64, u64, u64), u32> = BTreeMap::new();
    for &(_step, key, prev, new_val) in accesses {
        *counts.entry((key, prev, new_val)).or_insert(0) += 1;
    }

    let mut sum = QM31::ZERO;
    for ((key, prev, new_val), mult) in counts {
        let entry = qm31_from_m31(M31(key as u32))
            + alpha * qm31_from_m31(M31(prev as u32))
            + alpha_sq * qm31_from_m31(M31(new_val as u32));
        let denom = z - entry;
        debug_assert!(denom != QM31::ZERO, "dict table LogUp denominator zero — Fiat-Shamir collision");
        let mult_q = qm31_from_m31(M31(mult));
        sum = sum - mult_q * denom.inverse();
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::cm31::CM31;

    fn q(a: u32, b: u32, c: u32, d: u32) -> QM31 {
        QM31 { a: CM31 { a: M31(a), b: M31(b) }, b: CM31 { a: M31(c), b: M31(d) } }
    }

    #[test]
    fn test_verify_chain_empty() {
        assert!(verify_chain(&[]).is_ok());
    }

    #[test]
    fn test_verify_chain_single_key() {
        // key=1: first access prev=0→new=42, second access prev=42→new=99
        let accesses = [(0usize, 1u64, 0u64, 42u64), (1, 1, 42, 99)];
        assert!(verify_chain(&accesses).is_ok());
    }

    #[test]
    fn test_verify_chain_multi_key() {
        let accesses = [
            (0usize, 1u64, 0u64, 10u64),  // key=1: 0→10
            (1, 2, 0, 20),                 // key=2: 0→20
            (2, 1, 10, 30),                // key=1: 10→30
            (3, 2, 20, 40),                // key=2: 20→40
        ];
        assert!(verify_chain(&accesses).is_ok());
    }

    #[test]
    fn test_verify_chain_violation() {
        // key=1: first access prev=0→new=42, but second access claims prev=99 instead of 42
        let accesses = [(0usize, 1u64, 0u64, 42u64), (1, 1, 99, 100)];
        let err = verify_chain(&accesses).unwrap_err();
        assert!(matches!(err, DictConsistencyError::ChainViolation { key: 1, expected_prev: 42, actual_prev: 99 }));
    }

    #[test]
    fn test_verify_chain_wrong_initial_prev() {
        // First access to key=5 must have prev=0, but claims prev=7
        let accesses = [(0usize, 5u64, 7u64, 10u64)];
        let err = verify_chain(&accesses).unwrap_err();
        assert!(matches!(err, DictConsistencyError::ChainViolation { key: 5, expected_prev: 0, actual_prev: 7 }));
    }

    #[test]
    fn test_logup_cancellation() {
        // exec_sum + table_sum should equal zero for any consistent access log
        let accesses = [(0usize, 1u64, 0u64, 42u64), (1, 1, 42, 99), (2, 2, 0, 7)];
        let z     = q(12345, 67890, 11111, 22222);
        let alpha = q(33333, 44444, 55555, 66666);
        let exec_sum  = dict_logup_exec_sum(&accesses, z, alpha);
        let table_sum = dict_logup_table_sum(&accesses, z, alpha);
        assert_eq!(exec_sum + table_sum, QM31::ZERO,
            "dict LogUp sums should cancel: exec={exec_sum:?} table={table_sum:?}");
    }

    #[test]
    fn test_logup_cancellation_with_repeated_triple() {
        let accesses = [(0usize, 1u64, 0u64, 5u64), (1, 2, 0, 5), (2, 3, 0, 5)];
        let z     = q(98765, 43210, 11111, 99999);
        let alpha = q(22222, 33333, 44444, 55555);
        let exec_sum  = dict_logup_exec_sum(&accesses, z, alpha);
        let table_sum = dict_logup_table_sum(&accesses, z, alpha);
        assert_eq!(exec_sum + table_sum, QM31::ZERO,
            "dict LogUp should cancel for distinct keys with same new_value");
    }
}
