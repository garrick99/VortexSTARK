//! AIR (Algebraic Intermediate Representation) for the Fibonacci sequence.
//!
//! The Fibonacci AIR proves knowledge of a trace where:
//!   t[0] = a, t[1] = b (public inputs)
//!   t[i+2] = t[i] + t[i+1] for all i in [0, n-2)
//!
//! Constraints:
//!   1. Boundary: t[0] = a, t[1] = b
//!   2. Transition: t[i+2] - t[i+1] - t[i] = 0 for i in [0, n-2)

use crate::cuda::ffi;
use crate::field::M31;
use std::ffi::c_void;

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

/// Generate Fibonacci trace directly into pinned host memory and upload.
/// Returns a DeviceBuffer containing the trace on GPU.
/// Pinned memory enables faster DMA transfer (~2x vs pageable).
pub fn fibonacci_trace_to_device(a: M31, b: M31, log_n: u32) -> crate::device::DeviceBuffer<u32> {
    let n = 1usize << log_n;
    let p = crate::field::m31::P;
    let bytes = n * std::mem::size_of::<u32>();

    // Allocate pinned host memory
    let mut pinned_ptr: *mut c_void = std::ptr::null_mut();
    let err = unsafe { ffi::cudaMallocHost(&mut pinned_ptr, bytes) };
    assert!(err == 0, "cudaMallocHost failed: {err}");
    let trace = pinned_ptr as *mut u32;

    // Generate Fibonacci trace into pinned buffer
    unsafe {
        *trace.add(0) = a.0;
        *trace.add(1) = b.0;
        for i in 2..n {
            let sum = *trace.add(i - 1) + *trace.add(i - 2);
            *trace.add(i) = if sum >= p { sum - p } else { sum };
        }
    }

    // Upload from pinned memory (faster DMA)
    let buf = unsafe { crate::device::DeviceBuffer::from_pinned(trace as *const u32, n) };

    // Free pinned host memory
    unsafe { ffi::cudaFreeHost(pinned_ptr) };

    buf
}

/// Compute the Fibonacci pair (F(k), F(k+1)) starting from (a, b) using matrix exponentiation.
/// Uses the doubling identity in M31:
///   F(2k)   = F(k) * (2*F(k+1) - F(k))
///   F(2k+1) = F(k)^2 + F(k+1)^2
/// O(log k) M31 multiplications.
fn fib_at(a: M31, b: M31, k: usize) -> (M31, M31) {
    if k == 0 {
        return (a, b);
    }
    // Use matrix form: [[1,1],[1,0]]^k applied to (b, a)
    // We'll track the matrix power and multiply with the initial vector at the end.
    // Matrix [[m00, m01], [m10, m11]] starts as identity.
    let mut m00 = M31::ONE;
    let mut m01 = M31::ZERO;
    let mut m10 = M31::ZERO;
    let mut m11 = M31::ONE;
    // Base matrix: [[1, 1], [1, 0]]
    let mut b00 = M31::ONE;
    let mut b01 = M31::ONE;
    let mut b10 = M31::ONE;
    let mut b11 = M31::ZERO;
    let mut exp = k;
    while exp > 0 {
        if exp & 1 == 1 {
            // M = M * B
            let t00 = m00 * b00 + m01 * b10;
            let t01 = m00 * b01 + m01 * b11;
            let t10 = m10 * b00 + m11 * b10;
            let t11 = m10 * b01 + m11 * b11;
            m00 = t00; m01 = t01; m10 = t10; m11 = t11;
        }
        // B = B * B
        let t00 = b00 * b00 + b01 * b10;
        let t01 = b00 * b01 + b01 * b11;
        let t10 = b10 * b00 + b11 * b10;
        let t11 = b10 * b01 + b11 * b11;
        b00 = t00; b01 = t01; b10 = t10; b11 = t11;
        exp >>= 1;
    }
    // Result: M * [b, a]^T = [m00*b + m01*a, m10*b + m11*a]
    // F(k) = m10*b + m11*a, F(k+1) = m00*b + m01*a
    // Wait, let me think about this more carefully.
    // F(0) = a, F(1) = b, F(n) = F(n-1) + F(n-2)
    // [F(n+1), F(n)] = [[1,1],[1,0]] * [F(n), F(n-1)]
    // So [[1,1],[1,0]]^k * [F(1), F(0)] = [F(k+1), F(k)]
    // = [[1,1],[1,0]]^k * [b, a]
    // After exponentiation, M = [[1,1],[1,0]]^k
    // [F(k+1), F(k)] = [m00*b + m01*a, m10*b + m11*a]
    let fk = m10 * b + m11 * a;
    let fk1 = m00 * b + m01 * a;
    (fk, fk1)
}

/// Generate Fibonacci trace into a pre-allocated pinned buffer using parallel chunking.
/// Each chunk computes its starting values via matrix exponentiation, then fills forward.
/// SAFETY: `out` must point to pinned memory of at least `n` u32 elements.
pub unsafe fn fibonacci_trace_parallel(a: M31, b: M31, log_n: u32, out: *mut u32) {
  // SAFETY: caller guarantees `out` points to pinned memory of >= (1<<log_n) u32s.
  // Each chunk writes to disjoint indices.
  unsafe {
    let n = 1usize << log_n;
    let p = crate::field::m31::P;

    if n <= 131072 {
        // Small: just do it sequentially (thread spawn overhead would dominate)
        *out.add(0) = a.0;
        *out.add(1) = b.0;
        for i in 2..n {
            let sum = *out.add(i - 1) + *out.add(i - 2);
            *out.add(i) = if sum >= p { sum - p } else { sum };
        }
        return;
    }

    // Split into chunks. Each chunk computes F(chunk_start) via matrix power,
    // then fills forward sequentially.
    let n_chunks = 8; // tuned for i9-285K: 8 threads balances parallelism vs spawn overhead
    let chunk_size = n / n_chunks;

    // Compute starting values for each chunk
    let starts: Vec<(usize, M31, M31)> = (0..n_chunks)
        .map(|c| {
            let start_idx = c * chunk_size;
            let (fk, fk1) = fib_at(a, b, start_idx);
            (start_idx, fk, fk1)
        })
        .collect();

    // Cast pointer to usize so it can be sent across threads.
    // SAFETY: each thread writes to a disjoint chunk of the output buffer.
    let out_addr = out as usize;

    // Fill each chunk in parallel using scoped threads
    std::thread::scope(|s| {
        for &(start_idx, fk, fk1) in &starts {
            let end = if start_idx + chunk_size >= n { n } else { start_idx + chunk_size };
            s.spawn(move || {
                let out = out_addr as *mut u32;
                *out.add(start_idx) = fk.0;
                if start_idx + 1 < end {
                    *out.add(start_idx + 1) = fk1.0;
                }
                for i in (start_idx + 2)..end {
                    let sum = *out.add(i - 1) + *out.add(i - 2);
                    *out.add(i) = if sum >= p { sum - p } else { sum };
                }
            });
        }
    });
}}

/// Evaluate the transition constraint at a single point.
/// Returns t[i+2] - t[i+1] - t[i] (should be 0 for valid trace).
#[inline]
pub fn eval_transition(t_i: M31, t_i1: M31, t_i2: M31) -> M31 {
    t_i2 - t_i1 - t_i
}
