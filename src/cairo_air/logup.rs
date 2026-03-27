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
use rayon::prelude::*;

/// Number of memory accesses per execution step.
/// Instruction fetch + dst + op0 + op1 = 4 accesses.
pub const ACCESSES_PER_ROW: usize = 4;

/// Compute LogUp interaction values for a single row.
/// Returns the sum of 1/(z - (addr_j + alpha * val_j)) for each access j.
/// Access 0 is the instruction fetch: uses extended denominator
///   z - (pc + alpha * inst_lo + alpha^2 * inst_hi)
/// to fully bind both halves of the 63-bit instruction word.
pub fn logup_row_contribution(
    z: QM31,
    alpha: QM31,
    alpha_sq: QM31,
    accesses: &[(M31, M31)], // [(addr, value)] pairs; access 0 = (pc, inst_lo)
    inst_hi: M31,            // upper 31 bits of instruction (for access 0 only)
) -> QM31 {
    let mut sum = QM31::ZERO;
    for (j, &(addr, value)) in accesses.iter().enumerate() {
        let addr_qm31 = qm31_from_m31(addr);
        let val_qm31 = qm31_from_m31(value);
        let entry = if j == 0 {
            // Instruction fetch: bind inst_hi via alpha^2 term so the full
            // 63-bit word is committed — not just the lower 31 bits.
            addr_qm31 + alpha * val_qm31 + alpha_sq * qm31_from_m31(inst_hi)
        } else {
            addr_qm31 + alpha * val_qm31
        };
        let denom = z - entry;
        assert!(denom != QM31::ZERO, "LogUp denominator is zero — Fiat-Shamir collision (negligible probability; retry with different inputs)");
        sum = sum + denom.inverse();
    }
    sum
}

/// Compute the full LogUp interaction trace (running sum) on CPU.
/// Returns:
/// - interaction_cols: 4 columns (QM31 stored as SoA M31) of running sum
/// - claimed_sum: final running sum value (should equal memory table's contribution)
///
/// The running sum accumulates: S[i] = S[i-1] + logup_contribution(row_i)
///
/// The instruction fetch access uses the extended denominator
///   z - (pc + alpha * inst_lo + alpha^2 * inst_hi)
/// so both halves of the 63-bit instruction are authenticated.
pub fn compute_interaction_trace(
    trace_cols: &[Vec<u32>],  // 31 trace columns
    n: usize,
    z: QM31,
    alpha: QM31,
) -> ([Vec<u32>; 4], QM31) {
    use super::trace::*;

    let alpha_sq = alpha * alpha;
    let mut running_sum = QM31::ZERO;
    let mut interaction: [Vec<u32>; 4] = std::array::from_fn(|_| vec![0u32; n]);

    for i in 0..n {
        let pc       = M31(trace_cols[COL_PC][i]);
        let inst_lo  = M31(trace_cols[COL_INST_LO][i]);
        let inst_hi  = M31(trace_cols[COL_INST_HI][i]);
        let dst_addr = M31(trace_cols[COL_DST_ADDR][i]);
        let dst      = M31(trace_cols[COL_DST][i]);
        let op0_addr = M31(trace_cols[COL_OP0_ADDR][i]);
        let op0      = M31(trace_cols[COL_OP0][i]);
        let op1_addr = M31(trace_cols[COL_OP1_ADDR][i]);
        let op1      = M31(trace_cols[COL_OP1][i]);

        // Access 0 is instruction fetch; inst_hi is passed separately so
        // logup_row_contribution can apply the alpha^2 * inst_hi extension.
        let accesses = [
            (pc, inst_lo),
            (dst_addr, dst),
            (op0_addr, op0),
            (op1_addr, op1),
        ];

        let contribution = logup_row_contribution(z, alpha, alpha_sq, &accesses, inst_hi);
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
/// For instruction fetch entries the denominator is extended:
///   z - (pc + alpha * inst_lo + alpha^2 * inst_hi)
/// to match the execution-side denominator. Pass inst_hi_override = Some(inst_hi)
/// for instruction entries; None for all other memory accesses.
///
/// For the overall argument to hold: execution_sum + memory_table_sum = 0
pub fn compute_memory_table_sum(
    memory_entries: &[(M31, M31, u32)],  // (addr, value_lo, multiplicity)
    instr_entries: &[(M31, M31, M31, u32)], // (pc, inst_lo, inst_hi, multiplicity)
    z: QM31,
    alpha: QM31,
) -> QM31 {
    let alpha_sq = alpha * alpha;

    // Parallel reduction — each term is independent (pure function of its entry).
    let data_sum: QM31 = memory_entries.par_iter()
        .map(|&(addr, value, mult)| {
            let entry = qm31_from_m31(addr) + alpha * qm31_from_m31(value);
            let denom = z - entry;
            debug_assert!(denom != QM31::ZERO, "LogUp table denominator is zero — Fiat-Shamir collision");
            -(qm31_from_m31(M31(mult)) * denom.inverse())
        })
        .reduce(|| QM31::ZERO, |a, b| a + b);

    let instr_sum: QM31 = instr_entries.par_iter()
        .map(|&(pc, inst_lo, inst_hi, mult)| {
            let entry = qm31_from_m31(pc)
                + alpha * qm31_from_m31(inst_lo)
                + alpha_sq * qm31_from_m31(inst_hi);
            let denom = z - entry;
            debug_assert!(denom != QM31::ZERO, "LogUp instr denominator is zero — Fiat-Shamir collision");
            -(qm31_from_m31(M31(mult)) * denom.inverse())
        })
        .reduce(|| QM31::ZERO, |a, b| a + b);

    data_sum + instr_sum
}

/// Extract memory table from an execution trace.
/// Returns:
/// - data_entries: sorted unique (addr, value, multiplicity) for dst/op0/op1 accesses
/// - instr_entries: sorted unique (pc, inst_lo, inst_hi, multiplicity) for instruction fetches
///
/// Instruction fetches use a separate entry type so the extended denominator
/// (with inst_hi) can be applied during the memory table sum computation.
pub fn extract_memory_table(
    trace_cols: &[Vec<u32>],
    n_steps: usize,
) -> (Vec<(M31, M31, u32)>, Vec<(M31, M31, M31, u32)>) {
    use super::trace::*;

    // Collect raw pairs then sort+dedup — cache-friendly and O(n log n),
    // much faster than BTreeMap for large n (avoids tree pointer chasing).
    let mut data_pairs: Vec<(u32, u32)> = Vec::with_capacity(n_steps * 3);
    let mut instr_triples: Vec<(u32, u32, u32)> = Vec::with_capacity(n_steps);

    for i in 0..n_steps {
        instr_triples.push((
            trace_cols[COL_PC][i],
            trace_cols[COL_INST_LO][i],
            trace_cols[COL_INST_HI][i],
        ));
        data_pairs.push((trace_cols[COL_DST_ADDR][i], trace_cols[COL_DST][i]));
        data_pairs.push((trace_cols[COL_OP0_ADDR][i], trace_cols[COL_OP0][i]));
        data_pairs.push((trace_cols[COL_OP1_ADDR][i], trace_cols[COL_OP1][i]));
    }

    data_pairs.par_sort_unstable();
    let data: Vec<(M31, M31, u32)> = {
        let mut out = Vec::new();
        let mut i = 0;
        while i < data_pairs.len() {
            let key = data_pairs[i];
            let start = i;
            while i < data_pairs.len() && data_pairs[i] == key { i += 1; }
            out.push((M31(key.0), M31(key.1), (i - start) as u32));
        }
        out
    };

    instr_triples.par_sort_unstable();
    let instrs: Vec<(M31, M31, M31, u32)> = {
        let mut out = Vec::new();
        let mut i = 0;
        while i < instr_triples.len() {
            let key = instr_triples[i];
            let start = i;
            while i < instr_triples.len() && instr_triples[i] == key { i += 1; }
            out.push((M31(key.0), M31(key.1), M31(key.2), (i - start) as u32));
        }
        out
    };

    (data, instrs)
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
        let (data_table, instr_table) = extract_memory_table(&cols, n_steps);
        let mem_sum = compute_memory_table_sum(&data_table, &instr_table, z, alpha);

        let total = exec_sum + mem_sum;
        assert_eq!(total, QM31::ZERO,
            "LogUp sums don't cancel for Fibonacci trace!\n  {} data + {} instr entries",
            data_table.len(), instr_table.len());
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
        let (_interaction, exec_sum) = compute_interaction_trace(&cols, 2, z, alpha);

        // Compute memory table LogUp sum
        let (data_table, instr_table) = extract_memory_table(&cols, 2);
        let mem_sum = compute_memory_table_sum(&data_table, &instr_table, z, alpha);

        // The sums should cancel: exec_sum + mem_sum = 0
        let total = exec_sum + mem_sum;
        assert_eq!(total, QM31::ZERO,
            "LogUp sums don't cancel!\n  exec_sum = {:?}\n  mem_sum = {:?}\n  total = {:?}",
            exec_sum, mem_sum, total);
    }
}
