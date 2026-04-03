//! Range check argument for Cairo VM offsets.
//!
//! Cairo instructions contain three 16-bit offsets (off0, off1, off2),
//! each biased by 2^15. The raw values are in [0, 2^16).
//!
//! The range check proves that every offset in the execution trace
//! is a valid 16-bit value by using a LogUp permutation argument
//! against a precomputed table of all values 0..2^16.
//!
//! This reuses the same LogUp infrastructure as memory consistency:
//! - Execution trace contributes +1/(z_rc - offset) per offset per row
//! - Range check table contributes -multiplicity/(z_rc - value) per table entry
//! - The sums cancel if and only if all offsets are in range
//!
//! Each row has 3 offsets → 3 LogUp contributions per row.

use crate::field::{M31, QM31};
use super::logup::qm31_from_m31;

/// Number of offset values per row (off0, off1, off2).
pub const OFFSETS_PER_ROW: usize = 3;

/// Range check table size: all values 0..2^16.
pub const RC_TABLE_SIZE: usize = 1 << 16;

/// Column indices for the offsets in the trace.
/// off0, off1, off2 are stored as the lower 16 bits of instruction encoding.
/// We extract them from the instruction columns or store separately.
/// For now, we extract from the raw trace data.

/// Compute range check LogUp contribution for a single row.
/// Returns sum of 1/(z_rc - offset_j) for each of the 3 offsets.
pub fn rc_row_contribution(z_rc: QM31, offsets: &[M31; 3]) -> QM31 {
    let mut sum = QM31::ZERO;
    for &off in offsets {
        let off_qm31 = qm31_from_m31(off);
        let denom = z_rc - off_qm31;
        assert!(denom != QM31::ZERO, "RangeCheck exec denominator is zero — Fiat-Shamir collision (negligible probability; retry with different inputs)");
        sum = sum + denom.inverse();
    }
    sum
}

/// Compute the full range check interaction trace (running sum).
/// Returns 4 columns (QM31 SoA) and the claimed sum.
pub fn compute_rc_interaction_trace(
    offsets: &[[M31; 3]],  // n rows, each with 3 offsets
    n: usize,
    z_rc: QM31,
) -> ([Vec<u32>; 4], QM31) {
    let mut running_sum = QM31::ZERO;
    let mut interaction: [Vec<u32>; 4] = std::array::from_fn(|_| vec![0u32; n]);

    for i in 0..n {
        let contribution = rc_row_contribution(z_rc, &offsets[i]);
        running_sum = running_sum + contribution;

        let arr = running_sum.to_u32_array();
        for c in 0..4 {
            interaction[c][i] = arr[c];
        }
    }

    (interaction, running_sum)
}

/// Compute the range check table's LogUp contribution.
/// The table has entries 0..2^16, each with its multiplicity from the trace.
/// Returns -sum(multiplicity / (z_rc - value)) for all table entries.
pub fn compute_rc_table_sum(
    offset_counts: &[u32; RC_TABLE_SIZE],  // multiplicity of each value 0..2^16
    z_rc: QM31,
) -> QM31 {
    let mut sum = QM31::ZERO;
    for (val, &mult) in offset_counts.iter().enumerate() {
        if mult == 0 { continue; }
        let val_qm31 = qm31_from_m31(M31(val as u32));
        let denom = z_rc - val_qm31;
        assert!(denom != QM31::ZERO, "RangeCheck table denominator is zero — Fiat-Shamir collision (negligible probability; retry with different inputs)");
        let inv = denom.inverse();
        let mult_qm31 = qm31_from_m31(M31(mult));
        sum = sum - mult_qm31 * inv;
    }
    sum
}

/// Extract offsets from an execution trace and count multiplicities.
/// Returns (offsets_per_row, multiplicity_table).
pub fn extract_offsets(
    trace_cols: &[Vec<u32>],
    n_steps: usize,
) -> (Vec<[M31; 3]>, [u32; RC_TABLE_SIZE]) {
    use super::trace::{COL_OFF0, COL_OFF1, COL_OFF2};

    let mut offsets_per_row = Vec::with_capacity(n_steps);
    let mut counts = [0u32; RC_TABLE_SIZE];

    for i in 0..n_steps {
        // Read offsets directly from dedicated trace columns (off0/off1/off2 are
        // stored as raw 16-bit values; do NOT reconstruct from inst_lo/inst_hi,
        // because inst_hi is stored mod P and reconstruction would corrupt bit 31).
        let off0 = trace_cols[COL_OFF0][i];
        let off1 = trace_cols[COL_OFF1][i];
        let off2 = trace_cols[COL_OFF2][i];

        offsets_per_row.push([M31(off0), M31(off1), M31(off2)]);

        // Count multiplicities
        counts[off0 as usize] += 1;
        counts[off1 as usize] += 1;
        counts[off2 as usize] += 1;
    }

    (offsets_per_row, counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::cm31::CM31;
    use crate::cairo_air::{decode::Instruction, vm::{Memory, execute}, trace};

    #[test]
    fn test_range_check_simple() {
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
        let cols = trace::trace_to_columns(&vm_trace, 1);

        let z_rc = QM31 {
            a: CM31 { a: M31(77777), b: M31(88888) },
            b: CM31 { a: M31(11111), b: M31(22222) },
        };

        let (offsets, counts) = extract_offsets(&cols, 2);
        assert_eq!(offsets.len(), 2);

        // All offsets should be valid 16-bit values
        for row in &offsets {
            for off in row {
                assert!(off.0 < RC_TABLE_SIZE as u32, "offset {} out of range", off.0);
            }
        }

        // LogUp sums should cancel
        let (_, exec_sum) = compute_rc_interaction_trace(&offsets, 2, z_rc);
        let table_sum = compute_rc_table_sum(&counts, z_rc);
        let total = exec_sum + table_sum;
        assert_eq!(total, QM31::ZERO, "range check LogUp sums don't cancel");
    }

    #[test]
    fn test_range_check_fibonacci() {
        let mut mem = Memory::new();
        let assert_imm = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        mem.set(0, assert_imm.encode());
        mem.set(1, 1);
        mem.set(2, assert_imm.encode());
        mem.set(3, 1);

        let add = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        for addr in 4..20 {
            mem.set(addr, add.encode());
        }

        let n_steps = 10;
        let vm_trace = execute(&mut mem, n_steps);
        let cols = trace::trace_to_columns(&vm_trace, 4); // pad to 16

        let z_rc = QM31 {
            a: CM31 { a: M31(54321), b: M31(12345) },
            b: CM31 { a: M31(99999), b: M31(77777) },
        };

        let (offsets, counts) = extract_offsets(&cols, n_steps);

        // Verify offsets are in range
        for (row_idx, row) in offsets.iter().enumerate() {
            for (j, off) in row.iter().enumerate() {
                assert!(off.0 < RC_TABLE_SIZE as u32,
                    "offset out of range at row {row_idx}, offset {j}: {}", off.0);
            }
        }

        // Verify LogUp cancellation
        let (_, exec_sum) = compute_rc_interaction_trace(&offsets, n_steps, z_rc);
        let table_sum = compute_rc_table_sum(&counts, z_rc);
        let total = exec_sum + table_sum;
        assert_eq!(total, QM31::ZERO,
            "range check LogUp sums don't cancel for Fibonacci trace");
    }

    #[test]
    fn test_offset_extraction() {
        // Verify that extracted offsets match the instruction encoding
        let instr = Instruction {
            off0: 0x8000, off1: 0x7FFE, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        let encoded = instr.encode();
        let off0 = (encoded & 0xFFFF) as u32;
        let off1 = ((encoded >> 16) & 0xFFFF) as u32;
        let off2 = ((encoded >> 32) & 0xFFFF) as u32;

        assert_eq!(off0, 0x8000);
        assert_eq!(off1, 0x7FFE);
        assert_eq!(off2, 0x8001);
    }
}
