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

/// Evaluate the transition constraint at a single point.
/// Returns t[i+2] - t[i+1] - t[i] (should be 0 for valid trace).
#[inline]
pub fn eval_transition(t_i: M31, t_i1: M31, t_i2: M31) -> M31 {
    t_i2 - t_i1 - t_i
}
