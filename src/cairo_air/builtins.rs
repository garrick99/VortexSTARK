//! Cairo VM builtins: Poseidon, Pedersen, bitwise, etc.
//!
//! Builtins are cryptographic co-processors that execute alongside the Cairo VM.
//! Each builtin has its own sub-trace with dedicated columns and constraints.
//! The VM communicates with builtins through memory-mapped I/O:
//! - The VM writes input values to designated memory addresses
//! - The builtin reads those values, computes the result, writes output
//! - LogUp links the VM's memory accesses to the builtin's computation
//!
//! Architecture:
//! - Each builtin has its own trace columns (separate from VM's 27 columns)
//! - Builtin constraints are evaluated in a separate kernel (or fused with VM)
//! - The LogUp bus connects VM memory reads/writes to builtin I/O

use crate::field::M31;
use crate::poseidon::{self, STATE_WIDTH, NUM_ROUNDS};

/// Poseidon builtin segment in the execution trace.
/// Each invocation occupies NUM_ROUNDS rows in the builtin sub-trace.
/// Columns: 8 state elements (same as standalone Poseidon AIR).
pub struct PoseidonBuiltin {
    /// Input states for each invocation (set by the VM via memory writes).
    pub inputs: Vec<[M31; STATE_WIDTH]>,
    /// Output states (computed by the builtin).
    pub outputs: Vec<[M31; STATE_WIDTH]>,
}

impl PoseidonBuiltin {
    pub fn new() -> Self {
        Self { inputs: Vec::new(), outputs: Vec::new() }
    }

    /// Register a Poseidon invocation with the given input state.
    /// Returns the output (the permutation result).
    pub fn invoke(&mut self, input: [M31; STATE_WIDTH]) -> [M31; STATE_WIDTH] {
        let output = poseidon::poseidon_permutation(&input);
        self.inputs.push(input);
        self.outputs.push(output);
        output
    }

    /// Number of invocations.
    pub fn n_invocations(&self) -> usize {
        self.inputs.len()
    }

    /// Total rows in the builtin sub-trace (NUM_ROUNDS per invocation).
    pub fn n_rows(&self) -> usize {
        self.inputs.len() * NUM_ROUNDS
    }

    /// Generate the builtin sub-trace columns.
    /// Returns STATE_WIDTH columns, each of length n_rows.
    /// Each block of NUM_ROUNDS rows contains the intermediate states
    /// of one Poseidon permutation.
    pub fn generate_trace(&self, log_n: u32) -> Vec<Vec<u32>> {
        let n = 1usize << log_n;
        let n_rows = self.n_rows();
        assert!(n_rows <= n, "builtin trace too large for log_n={log_n}");

        let rc = poseidon::round_constants_flat();
        let rc_parsed: Vec<[M31; STATE_WIDTH]> = (0..NUM_ROUNDS)
            .map(|r| {
                let mut row = [M31::ZERO; STATE_WIDTH];
                for j in 0..STATE_WIDTH {
                    row[j] = M31(rc[r * STATE_WIDTH + j]);
                }
                row
            })
            .collect();

        let mut cols: Vec<Vec<u32>> = (0..STATE_WIDTH).map(|_| vec![0u32; n]).collect();

        for (inv_idx, input) in self.inputs.iter().enumerate() {
            let base_row = inv_idx * NUM_ROUNDS;
            let mut state = *input;

            for r in 0..NUM_ROUNDS {
                // Add round constants
                for j in 0..STATE_WIDTH {
                    state[j] = state[j] + rc_parsed[r][j];
                }
                // S-box: x^5
                for j in 0..STATE_WIDTH {
                    let x2 = state[j] * state[j];
                    let x4 = x2 * x2;
                    state[j] = x4 * state[j];
                }
                // MDS
                state = poseidon::mds_apply(&state);

                // Write to columns
                let row = base_row + r;
                if row < n {
                    for j in 0..STATE_WIDTH {
                        cols[j][row] = state[j].0;
                    }
                }
            }
        }

        cols
    }

    /// Generate LogUp entries linking VM memory to builtin I/O.
    /// Each invocation has STATE_WIDTH input reads and STATE_WIDTH output writes.
    /// Returns (address, value) pairs for the LogUp argument.
    pub fn logup_entries(&self, builtin_base_addr: u64) -> Vec<(M31, M31)> {
        let mut entries = Vec::new();
        let stride = STATE_WIDTH as u64 * 2; // input + output per invocation

        for (inv_idx, (input, output)) in self.inputs.iter().zip(&self.outputs).enumerate() {
            let base = builtin_base_addr + inv_idx as u64 * stride;
            // Input cells
            for j in 0..STATE_WIDTH {
                entries.push((M31((base + j as u64) as u32), input[j]));
            }
            // Output cells
            for j in 0..STATE_WIDTH {
                entries.push((M31((base + STATE_WIDTH as u64 + j as u64) as u32), output[j]));
            }
        }
        entries
    }
}

/// Memory-mapped address range for the Poseidon builtin.
/// The VM accesses these addresses to invoke the builtin.
pub const POSEIDON_BUILTIN_BASE: u64 = 0x4000_0000;

/// Invoke the Poseidon builtin from the Cairo VM.
/// The VM writes STATE_WIDTH input values to [base + inv*stride .. base + inv*stride + 8],
/// and reads STATE_WIDTH output values from [base + inv*stride + 8 .. base + inv*stride + 16].
pub fn vm_poseidon_invoke(
    memory: &mut super::vm::Memory,
    builtin: &mut PoseidonBuiltin,
    invocation_index: usize,
) -> [M31; STATE_WIDTH] {
    let stride = STATE_WIDTH as u64 * 2;
    let base = POSEIDON_BUILTIN_BASE + invocation_index as u64 * stride;

    // Read input from VM memory
    let mut input = [M31::ZERO; STATE_WIDTH];
    for j in 0..STATE_WIDTH {
        input[j] = M31(memory.get(base + j as u64) as u32);
    }

    // Compute Poseidon permutation
    let output = builtin.invoke(input);

    // Write output to VM memory
    for j in 0..STATE_WIDTH {
        memory.set(base + STATE_WIDTH as u64 + j as u64, output[j].0 as u64);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poseidon_builtin_invoke() {
        let mut builtin = PoseidonBuiltin::new();
        let input: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));

        let output = builtin.invoke(input);
        assert_eq!(builtin.n_invocations(), 1);
        assert_eq!(builtin.n_rows(), NUM_ROUNDS);

        // Output should match standalone Poseidon
        let expected = poseidon::poseidon_permutation(&input);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_poseidon_builtin_trace() {
        let mut builtin = PoseidonBuiltin::new();

        // 2 invocations
        let in1: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        let in2: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 10) as u32));
        builtin.invoke(in1);
        builtin.invoke(in2);

        assert_eq!(builtin.n_rows(), 2 * NUM_ROUNDS); // 44 rows

        // Generate trace (pad to next power of 2)
        let log_n = 6; // 64 rows
        let cols = builtin.generate_trace(log_n);
        assert_eq!(cols.len(), STATE_WIDTH);
        assert_eq!(cols[0].len(), 64);

        // Verify last row of first invocation matches output
        let out1 = poseidon::poseidon_permutation(&in1);
        for j in 0..STATE_WIDTH {
            assert_eq!(cols[j][NUM_ROUNDS - 1], out1[j].0,
                "first invocation output mismatch at column {j}");
        }
    }

    #[test]
    fn test_poseidon_builtin_logup() {
        let mut builtin = PoseidonBuiltin::new();
        let input: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        let output = builtin.invoke(input);

        let entries = builtin.logup_entries(POSEIDON_BUILTIN_BASE);
        // 8 inputs + 8 outputs = 16 entries
        assert_eq!(entries.len(), 16);

        // First 8 entries should be inputs
        for j in 0..STATE_WIDTH {
            assert_eq!(entries[j].1, input[j]);
        }
        // Last 8 entries should be outputs
        for j in 0..STATE_WIDTH {
            assert_eq!(entries[STATE_WIDTH + j].1, output[j]);
        }
    }

    #[test]
    fn test_vm_poseidon_invoke() {
        let mut mem = super::super::vm::Memory::with_capacity(0x5000_0000);
        let mut builtin = PoseidonBuiltin::new();

        // Write input to VM memory at builtin address
        let input: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 42) as u32));
        for j in 0..STATE_WIDTH {
            mem.set(POSEIDON_BUILTIN_BASE + j as u64, input[j].0 as u64);
        }

        // Invoke builtin
        let output = vm_poseidon_invoke(&mut mem, &mut builtin, 0);

        // Verify output is in memory
        for j in 0..STATE_WIDTH {
            let stored = mem.get(POSEIDON_BUILTIN_BASE + STATE_WIDTH as u64 + j as u64);
            assert_eq!(stored, output[j].0 as u64);
        }

        // Verify output matches standalone Poseidon
        let expected = poseidon::poseidon_permutation(&input);
        assert_eq!(output, expected);
    }
}
