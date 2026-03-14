//! LogUp memory consistency argument for Cairo VM.
//!
//! Proves that all memory reads in the execution trace are consistent:
//! every (address, value) pair read during execution corresponds to a valid
//! memory entry.
//!
//! Protocol:
//! 1. Commit execution trace (27 columns)
//! 2. Draw random challenges z, alpha (QM31)
//! 3. For each memory access (addr, value) at row i:
//!    - denom = z - (addr + alpha * value)  [QM31]
//!    - entry = multiplicity / denom        [QM31]
//! 4. Running sum S[i] = S[i-1] + sum_of_entries_at_row_i
//! 5. Commit interaction trace (running sum columns)
//! 6. Constraint: S transitions correctly, S wraps to claimed_sum
//!
//! The execution trace has 3 memory accesses per row:
//! - Instruction fetch: (pc, instruction)
//! - dst operand: (dst_addr, dst)
//! - op0/op1 operands: (op0_addr, op0), (op1_addr, op1)
//!
//! Each access contributes +1/denom to the running sum.
//! The memory table (sorted unique entries) contributes -multiplicity/denom.

use crate::field::{M31, QM31};
use crate::field::cm31::CM31;

/// Number of memory accesses per execution step.
/// Instruction fetch + dst + op0 + op1 = 4 accesses.
pub const ACCESSES_PER_ROW: usize = 4;

/// Compute LogUp interaction values for a single row.
/// Returns the sum of 1/(z - (addr_j + alpha * val_j)) for each access j.
pub fn logup_row_contribution(
    z: QM31,
    alpha: QM31,
    accesses: &[(M31, M31)], // [(addr, value)] pairs
) -> QM31 {
    let mut sum = QM31::ZERO;
    for &(addr, value) in accesses {
        // denom = z - (addr + alpha * value)
        let addr_qm31 = qm31_from_m31(addr);
        let val_qm31 = qm31_from_m31(value);
        let entry = addr_qm31 + alpha * val_qm31;
        let denom = z - entry;
        // 1 / denom
        let inv = denom.inverse();
        sum = sum + inv;
    }
    sum
}

/// Compute the full LogUp interaction trace (running sum) on CPU.
/// Returns:
/// - interaction_cols: 4 columns (QM31 stored as SoA M31) of running sum
/// - claimed_sum: final running sum value (should equal memory table's contribution)
///
/// The running sum accumulates: S[i] = S[i-1] + logup_contribution(row_i)
pub fn compute_interaction_trace(
    trace_cols: &[Vec<u32>],  // 27 trace columns
    n: usize,
    z: QM31,
    alpha: QM31,
) -> ([Vec<u32>; 4], QM31) {
    use super::trace::*;

    let mut running_sum = QM31::ZERO;
    let mut interaction: [Vec<u32>; 4] = std::array::from_fn(|_| vec![0u32; n]);

    for i in 0..n {
        // Collect memory accesses at this row
        let pc = M31(trace_cols[COL_PC][i]);
        let inst_lo = M31(trace_cols[COL_INST_LO][i]);
        let dst_addr = M31(trace_cols[COL_DST_ADDR][i]);
        let dst = M31(trace_cols[COL_DST][i]);
        let op0_addr = M31(trace_cols[COL_OP0_ADDR][i]);
        let op0 = M31(trace_cols[COL_OP0][i]);
        let op1_addr = M31(trace_cols[COL_OP1_ADDR][i]);
        let op1 = M31(trace_cols[COL_OP1][i]);

        let accesses = [
            (pc, inst_lo),         // instruction fetch (simplified: use inst_lo as value)
            (dst_addr, dst),       // dst operand
            (op0_addr, op0),       // op0 operand
            (op1_addr, op1),       // op1 operand
        ];

        let contribution = logup_row_contribution(z, alpha, &accesses);
        running_sum = running_sum + contribution;

        let arr = running_sum.to_u32_array();
        for c in 0..4 {
            interaction[c][i] = arr[c];
        }
    }

    (interaction, running_sum)
}

/// Compute the LogUp contribution from the memory table.
/// The memory table has one entry per unique (addr, value) pair,
/// with multiplicity = number of times it's accessed.
/// Its LogUp contribution is: sum_j -multiplicity_j / (z - (addr_j + alpha * value_j))
///
/// For the overall argument to hold: execution_sum + memory_table_sum = 0
pub fn compute_memory_table_sum(
    memory_entries: &[(M31, M31, u32)],  // (addr, value, multiplicity)
    z: QM31,
    alpha: QM31,
) -> QM31 {
    let mut sum = QM31::ZERO;
    for &(addr, value, mult) in memory_entries {
        let addr_qm31 = qm31_from_m31(addr);
        let val_qm31 = qm31_from_m31(value);
        let entry = addr_qm31 + alpha * val_qm31;
        let denom = z - entry;
        let inv = denom.inverse();
        let mult_qm31 = qm31_from_m31(M31(mult));
        sum = sum - mult_qm31 * inv;
    }
    sum
}

/// Extract memory table from an execution trace.
/// Returns sorted unique (addr, value, multiplicity) entries.
pub fn extract_memory_table(
    trace_cols: &[Vec<u32>],
    n_steps: usize,  // actual execution steps (may be < trace length due to padding)
) -> Vec<(M31, M31, u32)> {
    use std::collections::BTreeMap;
    use super::trace::*;

    let mut mem_counts: BTreeMap<(u32, u32), u32> = BTreeMap::new();

    for i in 0..n_steps {
        let accesses = [
            (trace_cols[COL_PC][i], trace_cols[COL_INST_LO][i]),
            (trace_cols[COL_DST_ADDR][i], trace_cols[COL_DST][i]),
            (trace_cols[COL_OP0_ADDR][i], trace_cols[COL_OP0][i]),
            (trace_cols[COL_OP1_ADDR][i], trace_cols[COL_OP1][i]),
        ];

        for (addr, val) in accesses {
            *mem_counts.entry((addr, val)).or_insert(0) += 1;
        }
    }

    mem_counts.into_iter()
        .map(|((addr, val), mult)| (M31(addr), M31(val), mult))
        .collect()
}

/// Helper: embed M31 into QM31.
pub fn qm31_from_m31(x: M31) -> QM31 {
    QM31 {
        a: CM31 { a: x, b: M31::ZERO },
        b: CM31::ZERO,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cairo_air::{decode::Instruction, vm::{Memory, execute}, trace};

    #[test]
    fn test_qm31_inverse() {
        let x = qm31_from_m31(M31(42));
        let inv = x.inverse();
        let product = x * inv;
        assert_eq!(product, QM31::ONE, "QM31 inverse failed");
    }

    #[test]
    fn test_qm31_inverse_complex() {
        let x = QM31 {
            a: CM31 { a: M31(7), b: M31(13) },
            b: CM31 { a: M31(42), b: M31(99) },
        };
        let inv = x.inverse();
        let product = x * inv;
        assert_eq!(product, QM31::ONE, "complex QM31 inverse failed");
    }

    #[test]
    fn test_logup_fibonacci_consistency() {
        // Fibonacci program — more complex memory access patterns
        let mut mem = Memory::new();
        let i = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        mem.set(0, i.encode());
        mem.set(1, 1);
        mem.set(2, i.encode());
        mem.set(3, 1);

        let add = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        for addr in 4..12 {
            mem.set(addr, add.encode());
        }

        let n_steps = 10;
        let vm_trace = execute(&mut mem, n_steps);
        let log_n = 4; // pad to 16
        let cols = trace::trace_to_columns(&vm_trace, log_n);

        let z = QM31 {
            a: CM31 { a: M31(98765), b: M31(43210) },
            b: CM31 { a: M31(11111), b: M31(22222) },
        };
        let alpha = QM31 {
            a: CM31 { a: M31(33333), b: M31(44444) },
            b: CM31 { a: M31(55555), b: M31(66666) },
        };

        let (_, exec_sum) = compute_interaction_trace(&cols, n_steps, z, alpha);
        let mem_table = extract_memory_table(&cols, n_steps);
        let mem_sum = compute_memory_table_sum(&mem_table, z, alpha);

        let total = exec_sum + mem_sum;
        assert_eq!(total, QM31::ZERO,
            "LogUp sums don't cancel for Fibonacci trace!\n  {} memory entries",
            mem_table.len());
    }

    #[test]
    fn test_logup_memory_consistency() {
        // Build a simple program and verify LogUp sums cancel
        let mut mem = Memory::new();
        let i = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        mem.set(0, i.encode());
        mem.set(1, 42);
        mem.set(2, i.encode());
        mem.set(3, 99);

        let vm_trace = execute(&mut mem, 2);
        let log_n = 1; // 2 rows (no padding for this test)
        let cols = trace::trace_to_columns(&vm_trace, log_n);

        // Draw random challenges
        let z = QM31 {
            a: CM31 { a: M31(12345), b: M31(67890) },
            b: CM31 { a: M31(11111), b: M31(22222) },
        };
        let alpha = QM31 {
            a: CM31 { a: M31(33333), b: M31(44444) },
            b: CM31 { a: M31(55555), b: M31(66666) },
        };

        // Compute execution trace LogUp sum
        let (interaction, exec_sum) = compute_interaction_trace(&cols, 2, z, alpha);

        // Compute memory table LogUp sum
        let mem_table = extract_memory_table(&cols, 2);
        let mem_sum = compute_memory_table_sum(&mem_table, z, alpha);

        // The sums should cancel: exec_sum + mem_sum = 0
        let total = exec_sum + mem_sum;
        assert_eq!(total, QM31::ZERO,
            "LogUp sums don't cancel!\n  exec_sum = {:?}\n  mem_sum = {:?}\n  total = {:?}",
            exec_sum, mem_sum, total);
    }
}
