//! AIR (Algebraic Intermediate Representation) for the Fibonacci sequence.
//!
//! The Fibonacci AIR proves knowledge of a trace where:
//!   t[0] = a, t[1] = b (public inputs)
//!   t[i+2] = t[i] + t[i+1] for all i in [0, n-2)
//!
//! Constraints:
//!   1. Boundary: t[0] = a, t[1] = b
//!   2. Transition: t[i+2] - t[i+1] - t[i] = 0 for i in [0, n-2)

use crate::field::M31;

/// Fibonacci trace: one column of M31 values.
pub fn fibonacci_trace(a: M31, b: M31, log_n: u32) -> Vec<M31> {
    let n = 1usize << log_n;
    let mut trace = vec![M31::ZERO; n];
    trace[0] = a;
    trace[1] = b;
    for i in 2..n {
        trace[i] = trace[i - 1] + trace[i - 2];
    }
    trace
}

/// Fibonacci trace as raw u32 values (avoids M31 → u32 conversion step).
/// Uses branchless M31 reduction (no division): sum of two values < P fits in u32,
/// then subtract P if >= P.
pub fn fibonacci_trace_raw(a: M31, b: M31, log_n: u32) -> Vec<u32> {
    let n = 1usize << log_n;
    let p = crate::field::m31::P;
    let mut trace = Vec::with_capacity(n);
    // SAFETY: every element [0..n) is written before read.
    unsafe { trace.set_len(n) };
    trace[0] = a.0;
    trace[1] = b.0;
    for i in 2..n {
        // Both values < P = 2^31-1, so sum < 2^32-4, fits in u32.
        let sum = unsafe { *trace.get_unchecked(i - 1) + *trace.get_unchecked(i - 2) };
        let val = if sum >= p { sum - p } else { sum };
        unsafe { *trace.get_unchecked_mut(i) = val };
    }
    trace
}

/// Evaluate the transition constraint at a single point.
/// Returns t[i+2] - t[i+1] - t[i] (should be 0 for valid trace).
#[inline]
pub fn eval_transition(t_i: M31, t_i1: M31, t_i2: M31) -> M31 {
    t_i2 - t_i1 - t_i
}
