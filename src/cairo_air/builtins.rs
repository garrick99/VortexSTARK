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
use crate::poseidon::{self, STATE_WIDTH, NUM_ROUNDS, RF_BEFORE, RP};

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
    /// Each block of NUM_ROUNDS (30) rows contains the intermediate states
    /// of one Poseidon2 permutation: 4 full → 22 partial → 4 full rounds.
    pub fn generate_trace(&self, log_n: u32) -> Vec<Vec<u32>> {
        let n = 1usize << log_n;
        let n_rows = self.n_rows();
        assert!(n_rows <= n, "builtin trace too large for log_n={log_n}");

        let full_rc = poseidon::full_round_constants();
        let partial_rc = poseidon::partial_round_constants();

        let mut cols: Vec<Vec<u32>> = (0..STATE_WIDTH).map(|_| vec![0u32; n]).collect();

        for (inv_idx, input) in self.inputs.iter().enumerate() {
            let base_row = inv_idx * NUM_ROUNDS;
            let mut state = *input;
            let mut row = base_row;

            // First RF_BEFORE full rounds
            for r in 0..RF_BEFORE {
                for j in 0..STATE_WIDTH { state[j] = state[j] + full_rc[r][j]; }
                poseidon::sbox_all(&mut state);
                poseidon::m_ext(&mut state);
                if row < n {
                    for j in 0..STATE_WIDTH { cols[j][row] = state[j].0; }
                }
                row += 1;
            }
            // RP partial rounds
            for r in 0..RP {
                state[0] = state[0] + partial_rc[r];
                state[0] = poseidon::sbox_one(state[0]);
                poseidon::m_int(&mut state);
                if row < n {
                    for j in 0..STATE_WIDTH { cols[j][row] = state[j].0; }
                }
                row += 1;
            }
            // Last RF_AFTER full rounds
            for r in 0..poseidon::RF_AFTER {
                for j in 0..STATE_WIDTH { state[j] = state[j] + full_rc[RF_BEFORE + r][j]; }
                poseidon::sbox_all(&mut state);
                poseidon::m_ext(&mut state);
                if row < n {
                    for j in 0..STATE_WIDTH { cols[j][row] = state[j].0; }
                }
                row += 1;
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
                let addr = base + j as u64;
                assert!(addr <= 0x7FFF_FFFF, "builtin address 0x{addr:x} overflows M31 (inv_idx={inv_idx})");
                entries.push((M31(addr as u32), input[j]));
            }
            // Output cells
            for j in 0..STATE_WIDTH {
                let addr = base + STATE_WIDTH as u64 + j as u64;
                assert!(addr <= 0x7FFF_FFFF, "builtin address 0x{addr:x} overflows M31 (inv_idx={inv_idx})");
                entries.push((M31(addr as u32), output[j]));
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

/// Memory-mapped address range for the Pedersen builtin.
pub const PEDERSEN_BUILTIN_BASE: u64 = super::pedersen::PEDERSEN_BUILTIN_BASE;

/// Invoke the Pedersen builtin from the Cairo VM.
/// The VM writes 2 Stark252 values (input_a, input_b) to memory-mapped addresses,
/// and reads 1 Stark252 result.
///
/// Memory layout per invocation (3 cells of Fp values stored as u64 pairs):
///   base + inv*6 + 0..1: input_a (low, high u64)
///   base + inv*6 + 2..3: input_b (low, high u64)
///   base + inv*6 + 4..5: output  (low, high u64)
///
/// The actual Pedersen computation is deferred to batch GPU execution.
/// During VM execution, we just record the (a, b) pair and write a placeholder.
/// After VM execution, `gpu_pedersen_trace()` computes all hashes on GPU.
pub fn vm_pedersen_invoke(
    memory: &mut super::vm::Memory,
    builtin: &mut super::pedersen::PedersenBuiltin,
    invocation_index: usize,
    a: super::pedersen::Stark252,
    b: super::pedersen::Stark252,
) -> super::pedersen::Stark252 {
    // Compute on CPU (for correctness / memory consistency)
    let output = builtin.invoke(a, b);

    // Write output to VM memory at builtin address range
    let stride = 6u64; // 2 cells per Fp (low + high u64) × 3 values
    let base = PEDERSEN_BUILTIN_BASE + invocation_index as u64 * stride;

    // Store first limb of each value for LogUp memory linking
    let fp_out = super::pedersen::stark252_to_fp(&output);
    memory.set(base + 0, super::pedersen::stark252_to_fp(&a).v[0]);
    memory.set(base + 1, super::pedersen::stark252_to_fp(&b).v[0]);
    memory.set(base + 2, fp_out.v[0]);

    output
}

/// Bitwise builtin.
///
/// Memory layout per invocation (5 cells):
///   base + inv*5 + 0: x
///   base + inv*5 + 1: y
///   base + inv*5 + 2: x & y  (AND)
///   base + inv*5 + 3: x ^ y  (XOR)
///   base + inv*5 + 4: x | y  (OR)
///
/// Algebraic constraints (2 per row, evaluated as M31 elements):
///   C0: xor + 2*and - x - y = 0   (follows from bit-level identity)
///   C1: or - and - xor = 0         (OR = AND + XOR, bitwise)
///
/// **Limitation:** These constraints hold as *integer* equalities and therefore
/// hold mod P for inputs x, y < 2^15. For inputs in the full M31 range (up to
/// 2^31-2), x+y may wrap around mod P, making the constraints satisfiable by
/// incorrect and/xor/or values. Full soundness requires bit-decomposition into
/// 16-bit chunks, which is not implemented here.
pub struct BitwiseBuiltin {
    /// Recorded (x, y) pairs in invocation order.
    pub inputs: Vec<(u32, u32)>,
    /// Computed (and, xor, or) results.
    pub outputs: Vec<(u32, u32, u32)>,
}

/// Memory-mapped base address for the Bitwise builtin.
pub const BITWISE_BUILTIN_BASE: u64 = 0x6000_0000;

impl BitwiseBuiltin {
    pub fn new() -> Self {
        Self { inputs: Vec::new(), outputs: Vec::new() }
    }

    /// Register one bitwise operation and return (and, xor, or).
    pub fn invoke(&mut self, x: u32, y: u32) -> (u32, u32, u32) {
        let and = x & y;
        let xor = x ^ y;
        let or  = x | y;
        self.inputs.push((x, y));
        self.outputs.push((and, xor, or));
        (and, xor, or)
    }

    pub fn n_invocations(&self) -> usize { self.inputs.len() }

    /// Generate 5 trace columns (x, y, and, xor, or), padded to `n = 1<<log_n`.
    pub fn generate_trace(&self, log_n: u32) -> Vec<Vec<u32>> {
        let n = 1usize << log_n;
        assert!(self.inputs.len() <= n, "bitwise trace too large for log_n={log_n}");
        let mut cols: Vec<Vec<u32>> = (0..5).map(|_| vec![0u32; n]).collect();
        for (i, ((x, y), (and, xor, or))) in self.inputs.iter().zip(&self.outputs).enumerate() {
            cols[0][i] = *x;
            cols[1][i] = *y;
            cols[2][i] = *and;
            cols[3][i] = *xor;
            cols[4][i] = *or;
        }
        cols
    }
}

/// Invoke the Bitwise builtin from the Cairo VM.
/// Reads x and y from the builtin memory segment, computes AND/XOR/OR,
/// and writes results back to memory.
pub fn vm_bitwise_invoke(
    memory: &mut super::vm::Memory,
    builtin: &mut BitwiseBuiltin,
    invocation_index: usize,
) -> (u32, u32, u32) {
    let stride = 5u64;
    let base = BITWISE_BUILTIN_BASE + invocation_index as u64 * stride;
    let x = memory.get(base) as u32;
    let y = memory.get(base + 1) as u32;
    let (and, xor, or) = builtin.invoke(x, y);
    memory.set(base + 2, and as u64);
    memory.set(base + 3, xor as u64);
    memory.set(base + 4, or as u64);
    (and, xor, or)
}

/// Generate Pedersen builtin trace columns on GPU.
/// Uses stored Fp inputs from invoke() — fused hash + trace on GPU.
/// Results never leave the GPU. Returns DeviceBuffers ready for NTT.
pub fn gpu_pedersen_builtin_trace(
    builtin: &super::pedersen::PedersenBuiltin,
    log_n: u32,
) -> Vec<crate::device::DeviceBuffer<u32>> {
    use super::pedersen::gpu_pedersen_trace;

    // Use stored Fp inputs directly — no Stark252→Fp conversion needed.
    // gpu_pedersen_trace fuses hash + M31 decompose on GPU.
    gpu_pedersen_trace(&builtin.fp_inputs_a, &builtin.fp_inputs_b, log_n)
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
    fn test_bitwise_builtin_basic() {
        let mut builtin = BitwiseBuiltin::new();
        let x: u32 = 0b1010_1010;
        let y: u32 = 0b1100_1100;
        let (and, xor, or) = builtin.invoke(x, y);
        assert_eq!(and, x & y);
        assert_eq!(xor, x ^ y);
        assert_eq!(or,  x | y);
        // Check algebraic constraints hold
        assert_eq!(xor.wrapping_add(2 * and), x.wrapping_add(y));
        assert_eq!(or, and + xor);
    }

    #[test]
    fn test_bitwise_builtin_trace() {
        let mut builtin = BitwiseBuiltin::new();
        builtin.invoke(0xFF, 0x0F);
        builtin.invoke(0xAA, 0x55);
        let cols = builtin.generate_trace(2); // n=4
        assert_eq!(cols.len(), 5);
        assert_eq!(cols[0][0], 0xFF);
        assert_eq!(cols[1][0], 0x0F);
        assert_eq!(cols[2][0], 0xFF & 0x0F);
        assert_eq!(cols[3][0], 0xFF ^ 0x0F);
        assert_eq!(cols[4][0], 0xFF | 0x0F);
        assert_eq!(cols[0][2], 0, "padding row should be zero");
    }

    #[test]
    fn test_vm_bitwise_invoke() {
        let mut mem = super::super::vm::Memory::with_capacity(0x1000_0000);
        let mut builtin = BitwiseBuiltin::new();
        let x: u64 = 0b1111_0000;
        let y: u64 = 0b1010_1010;
        let base = BITWISE_BUILTIN_BASE;
        mem.set(base, x);
        mem.set(base + 1, y);
        let (and, xor, or) = vm_bitwise_invoke(&mut mem, &mut builtin, 0);
        assert_eq!(mem.get(base + 2), and as u64);
        assert_eq!(mem.get(base + 3), xor as u64);
        assert_eq!(mem.get(base + 4), or as u64);
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
