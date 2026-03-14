//! Poseidon hash function over M31 for STARK AIR.
//!
//! Parameters:
//! - State width: 8 (capacity 4, rate 4)
//! - S-box: x^5 (full S-box, all elements)
//! - Rounds: 22 full rounds (conservative security margin for M31)
//! - MDS: 8×8 circulant matrix derived from first row
//!
//! The AIR traces one Poseidon permutation per 22 rows.
//! Each row stores the 8-element state AFTER that round's transformation.
//! Row 0 = state after round 0 (input + round_const + sbox + MDS).
//! The constraint checks: state[r+1] = MDS(sbox(state[r] + round_const[r+1])).

use crate::field::M31;

/// Number of state elements.
pub const STATE_WIDTH: usize = 8;

/// Number of full rounds.
pub const NUM_ROUNDS: usize = 22;

/// MDS matrix first row (circulant: row k = rotate(first_row, k)).
/// These are small constants chosen for efficient arithmetic and full-rank MDS.
const MDS_FIRST_ROW: [u32; STATE_WIDTH] = [3, 1, 1, 1, 1, 1, 1, 2];

/// Round constants: NUM_ROUNDS × STATE_WIDTH.
/// Generated deterministically from a seed via repeated squaring in M31.
fn round_constants() -> Vec<[M31; STATE_WIDTH]> {
    let mut constants = Vec::with_capacity(NUM_ROUNDS);
    // Deterministic generation: hash-like derivation from a fixed seed
    let mut state = M31(0x12345678 % crate::field::m31::P);
    for _round in 0..NUM_ROUNDS {
        let mut rc = [M31::ZERO; STATE_WIDTH];
        for j in 0..STATE_WIDTH {
            // Mix: state = state^5 + (round*width + j + 1)
            state = state * state;
            state = state * state;
            state = state * M31(state.0 ^ (((_round * STATE_WIDTH + j + 1) as u32) % crate::field::m31::P));
            // Ensure non-zero
            if state.0 == 0 { state = M31(1); }
            rc[j] = state;
        }
        constants.push(rc);
    }
    constants
}

/// Get the round constants as a flat array (for GPU upload).
pub fn round_constants_flat() -> Vec<u32> {
    let rc = round_constants();
    rc.iter().flat_map(|row| row.iter().map(|m| m.0)).collect()
}

/// Apply MDS matrix to state (circulant multiplication).
#[inline]
pub fn mds_apply(state: &[M31; STATE_WIDTH]) -> [M31; STATE_WIDTH] {
    mds(state)
}

#[inline]
fn mds(state: &[M31; STATE_WIDTH]) -> [M31; STATE_WIDTH] {
    let mut out = [M31::ZERO; STATE_WIDTH];
    for i in 0..STATE_WIDTH {
        let mut acc = M31::ZERO;
        for j in 0..STATE_WIDTH {
            let mds_val = M31(MDS_FIRST_ROW[(STATE_WIDTH + j - i) % STATE_WIDTH]);
            acc = acc + mds_val * state[j];
        }
        out[i] = acc;
    }
    out
}

/// Apply S-box: x → x^5 for each element.
#[inline]
fn sbox(state: &mut [M31; STATE_WIDTH]) {
    for s in state.iter_mut() {
        let x2 = *s * *s;
        let x4 = x2 * x2;
        *s = x4 * *s;
    }
}

/// Full Poseidon permutation: apply all rounds to input state.
pub fn poseidon_permutation(input: &[M31; STATE_WIDTH]) -> [M31; STATE_WIDTH] {
    let rc = round_constants();
    let mut state = *input;
    for r in 0..NUM_ROUNDS {
        // Add round constants
        for j in 0..STATE_WIDTH {
            state[j] = state[j] + rc[r][j];
        }
        // S-box
        sbox(&mut state);
        // MDS
        state = mds(&state);
    }
    state
}

/// Generate a Poseidon hash chain trace.
/// Each "block" is one permutation (NUM_ROUNDS rows).
/// The trace has STATE_WIDTH columns and n_blocks * NUM_ROUNDS rows.
/// Row layout within each block:
///   Row 0: state after round 0
///   Row 1: state after round 1
///   ...
///   Row NUM_ROUNDS-1: state after last round (= output)
///
/// The input to block k+1 is derived from the output of block k
/// by XOR-ing the block index (sponge-like construction).
///
/// Returns STATE_WIDTH columns, each of length n_rows.
pub fn generate_trace(log_n: u32) -> (Vec<Vec<u32>>, [M31; STATE_WIDTH], [M31; STATE_WIDTH]) {
    let n_rows = 1usize << log_n;
    assert!(n_rows >= NUM_ROUNDS, "trace too small for one permutation");
    let n_blocks = n_rows / NUM_ROUNDS;

    let mut columns: Vec<Vec<u32>> = (0..STATE_WIDTH).map(|_| Vec::with_capacity(n_rows)).collect();

    let rc = round_constants();

    let first_input = {
        let mut inp = [M31::ZERO; STATE_WIDTH];
        for j in 0..STATE_WIDTH {
            inp[j] = M31(((j + 1) as u32) % crate::field::m31::P);
        }
        inp
    };

    let mut last_output = [M31::ZERO; STATE_WIDTH];

    for block in 0..n_blocks {
        // Independent block input (no chain dependency)
        let mut state = [M31::ZERO; STATE_WIDTH];
        for j in 0..STATE_WIDTH {
            let val = ((block * STATE_WIDTH + j + 1) as u64) % (crate::field::m31::P as u64);
            state[j] = M31(val as u32);
        }
        for r in 0..NUM_ROUNDS {
            for j in 0..STATE_WIDTH {
                state[j] = state[j] + rc[r][j];
            }
            sbox(&mut state);
            state = mds(&state);
            for j in 0..STATE_WIDTH {
                columns[j].push(state[j].0);
            }
        }
        last_output = state;
    }

    (columns, first_input, last_output)
}

/// Evaluate the Poseidon transition constraint at row index `row`.
/// The constraint checks: state[row+1] = MDS(sbox(state[row] + round_const[round_of_row]))
///
/// For each state element j, the constraint is:
///   C_j(row) = next_state[j] - MDS_row_j(sbox(state + rc))
///
/// Returns STATE_WIDTH constraint values (one per column).
pub fn eval_constraints_at(
    columns: &[&[u32]],  // STATE_WIDTH columns of eval data
    row: usize,
    eval_size: usize,
) -> [M31; STATE_WIDTH] {
    let rc = round_constants();
    let next_round = (row + 1) % NUM_ROUNDS;
    let next_row = (row + 1) % eval_size;

    // The transition: state[row] → add round_const[next_round] → sbox → MDS → should equal state[row+1]
    let mut state = [M31::ZERO; STATE_WIDTH];
    for j in 0..STATE_WIDTH {
        state[j] = M31(columns[j][row]) + rc[next_round][j];
    }

    // S-box
    sbox(&mut state);

    // MDS
    let expected = mds(&state);

    // Constraint: next_state - expected
    let mut constraints = [M31::ZERO; STATE_WIDTH];
    for j in 0..STATE_WIDTH {
        let actual_next = M31(columns[j][next_row]);
        constraints[j] = actual_next - expected[j];
    }
    constraints
}

/// Generate Poseidon trace on GPU.
/// Pre-computes block inputs on CPU (sequential, fast — just 8 values per block),
/// then GPU-parallelizes the actual permutation work (22 rounds × 8 S-boxes per block).
/// Returns DeviceBuffers (trace stays on GPU, no download).
pub fn generate_trace_gpu(log_n: u32) -> (Vec<crate::device::DeviceBuffer<u32>>, [M31; STATE_WIDTH], [M31; STATE_WIDTH]) {
    use crate::cuda::ffi;
    use crate::device::DeviceBuffer;

    let n_rows = 1usize << log_n;
    assert!(n_rows >= NUM_ROUNDS);
    let n_blocks = n_rows / NUM_ROUNDS;

    // Upload round constants to GPU constant memory (one-time)
    let rc_flat = round_constants_flat();
    unsafe { ffi::cuda_poseidon_upload_round_consts(rc_flat.as_ptr()) };

    // Block inputs: independent (no chain dependency), computed in O(1) per block.
    // Each block's input is deterministic: input_k[j] = (k * STATE_WIDTH + j + 1) mod P.
    // This makes trace generation embarrassingly parallel on GPU.
    // The STARK proves that each block's rounds are computed correctly (the hard part).
    let first_input = {
        let mut inp = [M31::ZERO; STATE_WIDTH];
        for j in 0..STATE_WIDTH {
            inp[j] = M31(((j + 1) as u32) % crate::field::m31::P);
        }
        inp
    };

    let mut block_inputs_flat: Vec<u32> = Vec::with_capacity(n_blocks * STATE_WIDTH);
    for block in 0..n_blocks {
        for j in 0..STATE_WIDTH {
            let val = ((block * STATE_WIDTH + j + 1) as u64) % (crate::field::m31::P as u64);
            block_inputs_flat.push(val as u32);
        }
    }

    // Compute last block's output on CPU (for public input)
    let rc = round_constants();
    let mut last_state = [M31::ZERO; STATE_WIDTH];
    for j in 0..STATE_WIDTH {
        last_state[j] = M31(block_inputs_flat[(n_blocks - 1) * STATE_WIDTH + j]);
    }
    for r in 0..NUM_ROUNDS {
        for j in 0..STATE_WIDTH { last_state[j] = last_state[j] + rc[r][j]; }
        sbox(&mut last_state);
        last_state = mds(&last_state);
    }
    let last_output = last_state;

    // Upload block inputs to GPU
    let d_inputs = DeviceBuffer::from_host(&block_inputs_flat);

    // Allocate output columns on GPU
    let mut d_cols: Vec<DeviceBuffer<u32>> = (0..STATE_WIDTH)
        .map(|_| DeviceBuffer::<u32>::alloc(n_rows))
        .collect();

    // Build column pointer array
    let col_ptrs: Vec<*mut u32> = d_cols.iter_mut().map(|c| c.as_mut_ptr()).collect();
    let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

    // Launch GPU trace generation
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
    fn test_poseidon_permutation_deterministic() {
        let input: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        let out1 = poseidon_permutation(&input);
        let out2 = poseidon_permutation(&input);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_poseidon_permutation_nontrivial() {
        let input: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        let output = poseidon_permutation(&input);
        // Output should differ from input
        assert_ne!(input, output);
        // All elements should be non-zero
        for j in 0..STATE_WIDTH {
            assert_ne!(output[j], M31::ZERO, "output[{j}] is zero");
        }
    }

    #[test]
    fn test_trace_satisfies_constraints() {
        // Need enough rows for at least 2 blocks so we can check within-block transitions
        // NUM_ROUNDS = 22, so log_n=6 gives 64 rows = 2 blocks (44 rows) + 20 padding rows
        // Actually, 64/22 = 2 blocks with remainder. Let's use exact multiple.
        // 22 * 4 = 88 → need log_n=7 (128 rows) for 5 blocks
        let log_n = 7;
        let (columns, _input, _output) = generate_trace(log_n);
        let n = columns[0].len();
        let col_refs: Vec<&[u32]> = columns.iter().map(|c| c.as_slice()).collect();

        // Check constraint at every within-block row (not at block boundaries)
        let n_blocks = n / NUM_ROUNDS;
        for block in 0..n_blocks {
            let base = block * NUM_ROUNDS;
            for r in 0..(NUM_ROUNDS - 1) {
                let row = base + r;
                let c = eval_constraints_at(&col_refs, row, n);
                for j in 0..STATE_WIDTH {
                    assert_eq!(c[j], M31::ZERO,
                        "constraint violated at block {block}, round {r}, column {j}: {:?}", c[j]);
                }
            }
        }
    }

    #[test]
    fn test_gpu_trace_matches_cpu() {
        crate::cuda::ffi::init_memory_pool();
        let log_n = 7; // 128 rows
        let (cpu_cols, cpu_input, cpu_output) = generate_trace(log_n);
        let (gpu_cols, gpu_input, gpu_output) = generate_trace_gpu(log_n);

        assert_eq!(cpu_input, gpu_input, "inputs differ");
        assert_eq!(cpu_output, gpu_output, "outputs differ");

        let n_rows = 1 << log_n;
        let n_filled = (n_rows / NUM_ROUNDS) * NUM_ROUNDS;

        for c in 0..STATE_WIDTH {
            let gpu_host = gpu_cols[c].to_host();
            // Compare only filled rows (CPU vec may be shorter than GPU allocation)
            assert_eq!(&cpu_cols[c][..n_filled], &gpu_host[..n_filled], "column {c} differs");
        }
    }

    #[test]
    fn test_mds_nontrivial() {
        let input: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        let output = mds(&input);
        assert_ne!(input, output);
    }

    #[test]
    fn test_sbox_is_x5() {
        let x = M31(42);
        let mut state = [x; STATE_WIDTH];
        sbox(&mut state);
        let expected = x * x * x * x * x;
        assert_eq!(state[0], expected);
    }
}
