//! Poseidon2 hash function over M31 for STARK AIR.
//!
//! Parameters:
//! - State width: 8 (capacity 4, rate 4)
//! - S-box: x^5
//! - Full rounds: RF = 8 (RF_BEFORE=4 at start, RF_AFTER=4 at end)
//! - Partial rounds: RP = 22 (middle, only x[0] gets S-box)
//! - Total rounds: NUM_ROUNDS = 30 (down from old Poseidon's 22)
//! - S-box count: 8*8 + 22*1 = 86 (was 22*8 = 176 — 51% fewer)
//!
//! Linear layers:
//! - M_E (external, full rounds):   circ(3,1,1,1,1,1,1,1)
//!     s = sum(state); out[i] = 2*state[i] + s
//! - M_I (internal, partial rounds): circ(2,1,1,1,1,1,1,1)
//!     s = sum(state); out[i] = state[i] + s
//!
//! Trace layout per permutation block (NUM_ROUNDS = 30 rows):
//!   Rows 0..4:   first half full rounds (RF_BEFORE=4)
//!   Rows 4..26:  partial rounds (RP=22), only x[0] changes via S-box
//!   Rows 26..30: second half full rounds (RF_AFTER=4)
//!
//! Round constants:
//!   Full RC: RF * STATE_WIDTH = 64 values (8 rounds × 8 elements)
//!   Partial RC: RP = 22 values (one per partial round, applied to x[0] only)
//!   Flat GPU layout: [full_rcs[64], partial_rcs[22]] = 86 total values

use crate::field::M31;

/// Number of state elements.
pub const STATE_WIDTH: usize = 8;
/// Full rounds at start (before partial rounds).
pub const RF_BEFORE: usize = 4;
/// Partial rounds.
pub const RP: usize = 22;
/// Full rounds at end (after partial rounds).
pub const RF_AFTER: usize = 4;
/// Total full rounds (RF_BEFORE + RF_AFTER).
pub const RF: usize = RF_BEFORE + RF_AFTER;
/// Total rounds per permutation.
pub const NUM_ROUNDS: usize = RF + RP; // 30

/// M_E: external linear layer for full rounds.
/// circ(3,1,1,1,1,1,1,1): s = sum(state); out[i] = 2*state[i] + s.
/// Computable with additions only (2*x[i] = x[i]+x[i]).
#[inline]
pub fn m_ext(state: &mut [M31; STATE_WIDTH]) {
    let s = state.iter().copied().fold(M31::ZERO, |a, b| a + b);
    for x in state.iter_mut() {
        *x = *x + *x + s;
    }
}

/// M_I: internal linear layer for partial rounds.
/// circ(2,1,1,1,1,1,1,1): s = sum(state); out[i] = state[i] + s.
#[inline]
pub fn m_int(state: &mut [M31; STATE_WIDTH]) {
    let s = state.iter().copied().fold(M31::ZERO, |a, b| a + b);
    for x in state.iter_mut() {
        *x = *x + s;
    }
}

/// S-box: x → x^5 for one element.
#[inline]
pub fn sbox_one(x: M31) -> M31 {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x
}

/// S-box: x → x^5 for all 8 elements.
#[inline]
pub fn sbox_all(state: &mut [M31; STATE_WIDTH]) {
    for x in state.iter_mut() {
        *x = sbox_one(*x);
    }
}

/// Full round constants: RF rounds × STATE_WIDTH values.
/// Generated deterministically from a fixed seed.
pub fn full_round_constants() -> Vec<[M31; STATE_WIDTH]> {
    let mut rcs = Vec::with_capacity(RF);
    let mut seed = M31(0x12345678 % crate::field::m31::P);
    for r in 0..RF {
        let mut rc = [M31::ZERO; STATE_WIDTH];
        for j in 0..STATE_WIDTH {
            seed = seed * seed;
            seed = seed * seed;
            seed = M31(seed.0 ^ (((r * STATE_WIDTH + j + 1) as u32) % crate::field::m31::P));
            if seed.0 == 0 { seed = M31(1); }
            rc[j] = seed;
        }
        rcs.push(rc);
    }
    rcs
}

/// Partial round constants: RP values, applied to x[0] only.
pub fn partial_round_constants() -> Vec<M31> {
    let mut rcs = Vec::with_capacity(RP);
    let mut seed = M31(0xDEADBEEF % crate::field::m31::P);
    for r in 0..RP {
        seed = seed * seed;
        seed = seed * seed;
        seed = M31(seed.0 ^ (((r + RF * STATE_WIDTH + 1) as u32) % crate::field::m31::P));
        if seed.0 == 0 { seed = M31(1); }
        rcs.push(seed);
    }
    rcs
}

/// Flat round constants for GPU upload.
/// Layout: [full_rcs: RF*STATE_WIDTH=64, partial_rcs: RP=22] = 86 total u32s.
pub fn round_constants_flat() -> Vec<u32> {
    let full = full_round_constants();
    let partial = partial_round_constants();
    let mut out = Vec::with_capacity(RF * STATE_WIDTH + RP);
    for rc in &full {
        for v in rc { out.push(v.0); }
    }
    for v in &partial {
        out.push(v.0);
    }
    out
}

/// Poseidon2 permutation: 4 full → 22 partial → 4 full rounds.
pub fn poseidon_permutation(input: &[M31; STATE_WIDTH]) -> [M31; STATE_WIDTH] {
    let full_rc = full_round_constants();
    let partial_rc = partial_round_constants();
    let mut state = *input;

    for r in 0..RF_BEFORE {
        for j in 0..STATE_WIDTH { state[j] = state[j] + full_rc[r][j]; }
        sbox_all(&mut state);
        m_ext(&mut state);
    }
    for r in 0..RP {
        state[0] = state[0] + partial_rc[r];
        state[0] = sbox_one(state[0]);
        m_int(&mut state);
    }
    for r in 0..RF_AFTER {
        for j in 0..STATE_WIDTH { state[j] = state[j] + full_rc[RF_BEFORE + r][j]; }
        sbox_all(&mut state);
        m_ext(&mut state);
    }
    state
}

/// External MDS wrapper (for builtins.rs compatibility).
#[inline]
pub fn mds_apply(state: &[M31; STATE_WIDTH]) -> [M31; STATE_WIDTH] {
    let mut s = *state;
    m_ext(&mut s);
    s
}

/// Generate Poseidon2 hash chain trace on CPU.
/// Each block is one permutation (NUM_ROUNDS=30 rows).
/// Returns STATE_WIDTH columns of length n_blocks * NUM_ROUNDS.
pub fn generate_trace(log_n: u32) -> (Vec<Vec<u32>>, [M31; STATE_WIDTH], [M31; STATE_WIDTH]) {
    let n_rows = 1usize << log_n;
    assert!(n_rows >= NUM_ROUNDS, "trace too small for one permutation");
    let n_blocks = n_rows / NUM_ROUNDS;

    let full_rc = full_round_constants();
    let partial_rc = partial_round_constants();
    let mut columns: Vec<Vec<u32>> = (0..STATE_WIDTH).map(|_| Vec::with_capacity(n_rows)).collect();

    let first_input = {
        let mut inp = [M31::ZERO; STATE_WIDTH];
        for j in 0..STATE_WIDTH { inp[j] = M31(((j + 1) as u32) % crate::field::m31::P); }
        inp
    };
    let mut last_output = [M31::ZERO; STATE_WIDTH];

    for block in 0..n_blocks {
        let mut state = [M31::ZERO; STATE_WIDTH];
        for j in 0..STATE_WIDTH {
            let val = ((block * STATE_WIDTH + j + 1) as u64) % (crate::field::m31::P as u64);
            state[j] = M31(val as u32);
        }
        for r in 0..RF_BEFORE {
            for j in 0..STATE_WIDTH { state[j] = state[j] + full_rc[r][j]; }
            sbox_all(&mut state);
            m_ext(&mut state);
            for j in 0..STATE_WIDTH { columns[j].push(state[j].0); }
        }
        for r in 0..RP {
            state[0] = state[0] + partial_rc[r];
            state[0] = sbox_one(state[0]);
            m_int(&mut state);
            for j in 0..STATE_WIDTH { columns[j].push(state[j].0); }
        }
        for r in 0..RF_AFTER {
            for j in 0..STATE_WIDTH { state[j] = state[j] + full_rc[RF_BEFORE + r][j]; }
            sbox_all(&mut state);
            m_ext(&mut state);
            for j in 0..STATE_WIDTH { columns[j].push(state[j].0); }
        }
        last_output = state;
    }

    (columns, first_input, last_output)
}

/// Evaluate the Poseidon2 transition constraint at row `row`.
/// Determines round type from row index, applies the correct transformation,
/// returns STATE_WIDTH constraint values (zero iff trace is correct).
pub fn eval_constraints_at(
    columns: &[&[u32]],
    row: usize,
    eval_size: usize,
) -> [M31; STATE_WIDTH] {
    let full_rc = full_round_constants();
    let partial_rc = partial_round_constants();

    let round_in_block = row % NUM_ROUNDS;
    // next_round is the round being applied to produce state[row+1]
    let next_round = (round_in_block + 1) % NUM_ROUNDS;
    let next_row = (row + 1) % eval_size;

    let mut state: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31(columns[j][row]));
    let expected;

    let next_is_partial = next_round >= RF_BEFORE && next_round < RF_BEFORE + RP;

    if next_is_partial {
        let p = next_round - RF_BEFORE;
        state[0] = state[0] + partial_rc[p];
        state[0] = sbox_one(state[0]);
        m_int(&mut state);
        expected = state;
    } else {
        // full round
        let full_r = if next_round < RF_BEFORE { next_round } else { next_round - RP };
        for j in 0..STATE_WIDTH { state[j] = state[j] + full_rc[full_r][j]; }
        sbox_all(&mut state);
        m_ext(&mut state);
        expected = state;
    }

    let mut constraints = [M31::ZERO; STATE_WIDTH];
    for j in 0..STATE_WIDTH {
        let actual_next = M31(columns[j][next_row]);
        constraints[j] = actual_next - expected[j];
    }
    constraints
}

/// Generate Poseidon2 trace on GPU.
/// Block inputs are pre-computed on CPU (deterministic, O(1) per block),
/// then GPU-parallelizes the 30-round permutation work.
pub fn generate_trace_gpu(log_n: u32) -> (Vec<crate::device::DeviceBuffer<u32>>, [M31; STATE_WIDTH], [M31; STATE_WIDTH]) {
    use crate::cuda::ffi;
    use crate::device::DeviceBuffer;

    let n_rows = 1usize << log_n;
    assert!(n_rows >= NUM_ROUNDS);
    let n_blocks = n_rows / NUM_ROUNDS;

    // Upload round constants [full_rcs:64, partial_rcs:22]
    let rc_flat = round_constants_flat();
    unsafe { ffi::cuda_poseidon_upload_round_consts(rc_flat.as_ptr()); }

    let first_input = {
        let mut inp = [M31::ZERO; STATE_WIDTH];
        for j in 0..STATE_WIDTH { inp[j] = M31(((j + 1) as u32) % crate::field::m31::P); }
        inp
    };

    let mut block_inputs_flat: Vec<u32> = Vec::with_capacity(n_blocks * STATE_WIDTH);
    for block in 0..n_blocks {
        for j in 0..STATE_WIDTH {
            let val = ((block * STATE_WIDTH + j + 1) as u64) % (crate::field::m31::P as u64);
            block_inputs_flat.push(val as u32);
        }
    }

    // Compute last block output on CPU for public input
    let full_rc = full_round_constants();
    let partial_rc = partial_round_constants();
    let mut last_state = [M31::ZERO; STATE_WIDTH];
    for j in 0..STATE_WIDTH {
        last_state[j] = M31(block_inputs_flat[(n_blocks - 1) * STATE_WIDTH + j]);
    }
    for r in 0..RF_BEFORE {
        for j in 0..STATE_WIDTH { last_state[j] = last_state[j] + full_rc[r][j]; }
        sbox_all(&mut last_state);
        m_ext(&mut last_state);
    }
    for r in 0..RP {
        last_state[0] = last_state[0] + partial_rc[r];
        last_state[0] = sbox_one(last_state[0]);
        m_int(&mut last_state);
    }
    for r in 0..RF_AFTER {
        for j in 0..STATE_WIDTH { last_state[j] = last_state[j] + full_rc[RF_BEFORE + r][j]; }
        sbox_all(&mut last_state);
        m_ext(&mut last_state);
    }
    let last_output = last_state;

    let d_inputs = DeviceBuffer::from_host(&block_inputs_flat);
    let mut d_cols: Vec<DeviceBuffer<u32>> = (0..STATE_WIDTH)
        .map(|_| DeviceBuffer::<u32>::alloc(n_rows))
        .collect();
    let col_ptrs: Vec<*mut u32> = d_cols.iter_mut().map(|c| c.as_mut_ptr()).collect();
    let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

    unsafe {
        ffi::cuda_poseidon_trace(
            d_inputs.as_ptr(),
            d_col_ptrs.as_ptr() as *const *mut u32,
            n_blocks as u32,
        );
        ffi::cuda_device_sync();
    }

    (d_cols, first_input, last_output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poseidon2_permutation_deterministic() {
        let input: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        let out1 = poseidon_permutation(&input);
        let out2 = poseidon_permutation(&input);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_poseidon2_permutation_nontrivial() {
        let input: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        let output = poseidon_permutation(&input);
        assert_ne!(input, output);
        for j in 0..STATE_WIDTH {
            assert_ne!(output[j], M31::ZERO, "output[{j}] is zero");
        }
    }

    #[test]
    fn test_m_ext_nontrivial() {
        let mut state: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        let orig = state;
        m_ext(&mut state);
        assert_ne!(state, orig);
        // Verify formula: s = sum(orig); out[i] = 2*orig[i] + s
        let s: M31 = orig.iter().copied().fold(M31::ZERO, |a, b| a + b);
        for i in 0..STATE_WIDTH {
            assert_eq!(state[i], orig[i] + orig[i] + s);
        }
    }

    #[test]
    fn test_m_int_nontrivial() {
        let mut state: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        let orig = state;
        m_int(&mut state);
        // Verify formula: s = sum(orig); out[i] = orig[i] + s
        let s: M31 = orig.iter().copied().fold(M31::ZERO, |a, b| a + b);
        for i in 0..STATE_WIDTH {
            assert_eq!(state[i], orig[i] + s);
        }
    }

    #[test]
    fn test_sbox_is_x5() {
        let x = M31(42);
        assert_eq!(sbox_one(x), x * x * x * x * x);
    }

    #[test]
    fn test_trace_satisfies_constraints() {
        // 30 rounds per block, need log_n s.t. 2^log_n / 30 >= 2 blocks
        // 30*4 = 120 → log_n=7 (128 rows = 4 blocks with 8 unused rows)
        let log_n = 7;
        let (columns, _input, _output) = generate_trace(log_n);
        let n = columns[0].len();
        let col_refs: Vec<&[u32]> = columns.iter().map(|c| c.as_slice()).collect();

        let n_blocks = n / NUM_ROUNDS;
        for block in 0..n_blocks {
            let base = block * NUM_ROUNDS;
            for r in 0..(NUM_ROUNDS - 1) {
                let row = base + r;
                let c = eval_constraints_at(&col_refs, row, n);
                for j in 0..STATE_WIDTH {
                    assert_eq!(c[j], M31::ZERO,
                        "constraint violated at block {block}, round {r}, col {j}");
                }
            }
        }
    }

    #[test]
    fn test_gpu_trace_matches_cpu() {
        crate::cuda::ffi::init_memory_pool();
        let log_n = 7;
        let (cpu_cols, cpu_input, cpu_output) = generate_trace(log_n);
        let (gpu_cols, gpu_input, gpu_output) = generate_trace_gpu(log_n);

        assert_eq!(cpu_input, gpu_input, "inputs differ");
        assert_eq!(cpu_output, gpu_output, "outputs differ");

        let n_rows = 1 << log_n;
        let n_filled = (n_rows / NUM_ROUNDS) * NUM_ROUNDS;

        for c in 0..STATE_WIDTH {
            let gpu_host = gpu_cols[c].to_host();
            assert_eq!(&cpu_cols[c][..n_filled], &gpu_host[..n_filled], "column {c} differs");
        }
    }
}
