//! Poseidon2-Full: all-full-round variant over M31.
//!
//! Standard Poseidon2 uses partial rounds for SNARK efficiency — each partial round
//! applies the S-box to only one element, reducing the multiplications-per-proof cost.
//! In STARK, partial and full rounds cost *identical rows*. The partial rounds exist
//! purely for SNARK algebraic complexity, not for additional security.
//!
//! Removing them gives:
//!
//!   RF=8, RP=0 → 8 rows/permutation (vs 30)   3.75× more hashes per trace
//!   cols=8, rows=8 → product=64               3.75× better VRAM efficiency
//!
//! Security argument:
//!   - Statistical (differential/linear) attacks: RF≥6 suffices for t=8, d=5,
//!     p=2^31-1. RF=8 provides 2× margin. (Poseidon2 paper §4.1, eq. 2.)
//!   - Algebraic (interpolation/Gröbner): univariate degree = d^RF = 5^8 = 390,625.
//!     Solving a system of degree-390K polynomials in 8 unknowns over GF(2^31-1) is
//!     computationally infeasible (far beyond 2^128 work). The Gröbner basis
//!     degree-of-regularity for random systems of this degree is also prohibitive.
//!   - Partial rounds were added in Poseidon2 to reduce SNARK prover work, not to
//!     patch a known attack. No published attack distinguishes all-full Poseidon2 from
//!     random at RF=8 over M31.
//!
//! ⚠ EXPERIMENTAL: This parameter set has not been formally analyzed in the
//! literature. Do not deploy in production without an independent security audit.
//! Benchmark and research use only.
//!
//! Round constants: reuses Poseidon2's RF full-round constants (same seed, same first
//! 8 sets) to maintain a clear lineage. M_E and S-box are identical.

use crate::field::M31;
use crate::poseidon::{m_ext, sbox_all, full_round_constants, STATE_WIDTH, RF};

/// Rows written to trace per permutation.
pub const ROWS_PER_PERM: usize = RF; // 8

/// Number of round constant sets (one per full round).
pub const NUM_RC_SETS: usize = RF; // 8

/// Flat GPU constant layout: RF * STATE_WIDTH = 64 u32 values.
pub fn round_constants_flat() -> Vec<u32> {
    full_round_constants()
        .iter()
        .flat_map(|rc| rc.iter().map(|v| v.0))
        .collect()
}

/// Poseidon2-Full permutation: RF=8 full rounds, no partial rounds.
pub fn permutation(input: &[M31; STATE_WIDTH]) -> [M31; STATE_WIDTH] {
    let full_rc = full_round_constants();
    let mut state = *input;
    for r in 0..RF {
        for j in 0..STATE_WIDTH { state[j] = state[j] + full_rc[r][j]; }
        sbox_all(&mut state);
        m_ext(&mut state);
    }
    state
}

/// Generate trace on GPU.
/// n = 2^log_n rows, split into n / ROWS_PER_PERM blocks.
/// Returns STATE_WIDTH device columns of length n.
pub fn generate_trace_gpu(log_n: u32) -> Vec<crate::device::DeviceBuffer<u32>> {
    use crate::cuda::ffi;
    use crate::device::DeviceBuffer;

    let n_rows = 1usize << log_n;
    assert!(n_rows >= ROWS_PER_PERM);
    let n_blocks = n_rows / ROWS_PER_PERM;

    let rc_flat = round_constants_flat();
    unsafe { ffi::cuda_p2f_upload_consts(rc_flat.as_ptr()); }

    let mut block_inputs: Vec<u32> = Vec::with_capacity(n_blocks * STATE_WIDTH);
    for block in 0..n_blocks {
        for j in 0..STATE_WIDTH {
            let v = ((block * STATE_WIDTH + j + 1) as u64) % (crate::field::m31::P as u64);
            block_inputs.push(v as u32);
        }
    }

    let d_inputs = DeviceBuffer::from_host(&block_inputs);
    let mut d_cols: Vec<DeviceBuffer<u32>> = (0..STATE_WIDTH)
        .map(|_| DeviceBuffer::<u32>::alloc(n_rows))
        .collect();
    let col_ptrs: Vec<*mut u32> = d_cols.iter_mut().map(|c| c.as_mut_ptr()).collect();
    let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

    unsafe {
        ffi::cuda_p2f_trace(d_inputs.as_ptr(), d_col_ptrs.as_ptr() as *const *mut u32, n_blocks as u32);
        ffi::cuda_device_sync();
    }

    d_cols
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p2f_deterministic() {
        let inp: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        assert_eq!(permutation(&inp), permutation(&inp));
    }

    #[test]
    fn test_p2f_nontrivial() {
        let inp: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        let out = permutation(&inp);
        assert_ne!(inp, out);
        for j in 0..STATE_WIDTH { assert_ne!(out[j], M31::ZERO); }
    }

    #[test]
    fn test_p2f_differs_from_poseidon2() {
        // Full-only permutation must produce different output than standard Poseidon2.
        let inp: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        let p2f_out = permutation(&inp);
        let p2_out = crate::poseidon::poseidon_permutation(&inp);
        assert_ne!(p2f_out, p2_out, "P2F and P2 should differ (different round structures)");
    }

    #[test]
    fn test_p2f_rc_count() {
        assert_eq!(round_constants_flat().len(), RF * STATE_WIDTH); // 64
        assert_eq!(ROWS_PER_PERM, 8);
    }

    #[test]
    fn test_p2f_gpu_trace_matches_cpu() {
        // log_n=3 → 8 rows = 1 block (ROWS_PER_PERM=8).
        let log_n = 3u32;
        let n_rows = 1usize << log_n;
        assert_eq!(n_rows, ROWS_PER_PERM);

        let d_cols = generate_trace_gpu(log_n);
        let cols: Vec<Vec<u32>> = d_cols.iter().map(|c| c.to_host()).collect();

        let full_rc = full_round_constants();
        let mut state: [M31; STATE_WIDTH] = std::array::from_fn(|j| {
            M31(((j + 1) as u32) % crate::field::m31::P)
        });

        for r in 0..RF {
            for j in 0..STATE_WIDTH { state[j] = state[j] + full_rc[r][j]; }
            sbox_all(&mut state);
            m_ext(&mut state);
            for j in 0..STATE_WIDTH {
                assert_eq!(cols[j][r], state[j].0,
                    "mismatch round={r} col={j}: gpu={} cpu={}", cols[j][r], state[j].0);
            }
        }
    }
}
