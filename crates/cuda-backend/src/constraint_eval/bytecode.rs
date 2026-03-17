//! Bytecode representation for GPU constraint evaluation.
//!
//! The bytecode is a flat sequence of [`BytecodeOp`] values that describe how to evaluate
//! all constraints for a single component row. The program is the same for every row;
//! only the trace values (fetched via `PushTraceVal`) change.
//!
//! The interpreter uses an explicit value stack. Values on the stack are either M31
//! (base field) or QM31 (secure extension field). The `Wide*` variants operate on
//! QM31 values; the unprefixed variants operate on M31.

use std::fmt;

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;

/// A single bytecode instruction.
///
/// The design is intentionally simple: no register allocation, no optimization passes.
/// This is meant to be a faithful recording of the operations that `FrameworkEval::evaluate`
/// performs, suitable for replay on the GPU with different trace data per row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BytecodeOp {
    // ── Stack loads ──────────────────────────────────────────────────────

    /// Push an M31 constant onto the stack.
    PushBaseField(u32),

    /// Push a QM31 constant onto the stack (4 × M31 limbs).
    PushSecureField([u32; 4]),

    /// Push a trace value: `trace[interaction][col_idx]` at the current row.
    /// For non-zero offsets the kernel must handle index arithmetic.
    PushTraceVal {
        interaction: u16,
        col_idx: u16,
        offset: i16,
    },

    // ── M31 (base field) arithmetic ─────────────────────────────────────

    /// Pop two M31 values, push their sum.
    Add,
    /// Pop two M31 values, push (a - b) where a was below b on the stack.
    Sub,
    /// Pop two M31 values, push their product.
    Mul,
    /// Pop one M31 value, push its negation.
    Neg,
    /// Pop one M31 value, add an M31 constant, push the result.
    AddConst(u32),
    /// Pop one M31 value, multiply by an M31 constant, push the result.
    MulConst(u32),

    // ── QM31 (secure extension field) arithmetic ────────────────────────

    /// Pop two QM31 values, push their sum.
    WideAdd,
    /// Pop two QM31 values, push (a - b).
    WideSub,
    /// Pop two QM31 values, push their product.
    WideMul,
    /// Pop one QM31 value, push its negation.
    WideNeg,
    /// Pop one QM31, add a QM31 constant, push.
    WideAddConst([u32; 4]),
    /// Pop one QM31, multiply by a QM31 constant, push.
    WideMulConst([u32; 4]),

    // ── Mixed-width arithmetic ──────────────────────────────────────────

    /// Pop QM31 and M31, push QM31 + M31 (M31 extended to QM31 first).
    WideAddBase,
    /// Pop QM31 and M31, push QM31 * M31.
    WideMulBase,
    /// Pop M31, add SecureField constant, push QM31.
    BaseAddSecureConst([u32; 4]),
    /// Pop M31, multiply by SecureField constant, push QM31.
    BaseMulSecureConst([u32; 4]),

    // ── Widening ────────────────────────────────────────────────────────

    /// Pop M31 value, widen to QM31 (embed in first coordinate), push.
    Widen,
    /// Pop 4 M31 values, combine into a single QM31, push.
    CombineEF,

    // ── Constraint accumulation ─────────────────────────────────────────

    /// Pop a constraint value (M31 or QM31 — the tracer widens M31 to QM31 before
    /// emitting this op). Multiply by `random_coeff_powers[constraint_idx]` and
    /// accumulate into the row result. Increments the internal constraint counter.
    AddConstraint,
}

/// A compiled bytecode program for a single component.
#[derive(Clone, Debug)]
pub struct BytecodeProgram {
    /// The flat instruction stream.
    pub ops: Vec<BytecodeOp>,
    /// Number of constraints (= number of `AddConstraint` ops).
    pub n_constraints: usize,
    /// Number of distinct trace column accesses. Each `PushTraceVal` with a unique
    /// `(interaction, col_idx, offset)` triple counts once.
    pub n_trace_accesses: usize,
    /// Maximum stack depth reached during recording.
    pub max_stack_depth: usize,
}

impl BytecodeProgram {
    /// Encode the bytecode program as a flat `Vec<u32>` for GPU consumption.
    ///
    /// Encoding scheme per instruction:
    ///   - **Header word**: `opcode << 24 | operand_24bit`
    ///   - **Extended words**: for values that don't fit in 24 bits
    ///
    /// Single-word ops (operand fits in 24 bits):
    ///   `[opcode:8 | operand:24]`
    ///
    /// `PushBaseField(v)`: If v < 2^24, single word. Otherwise header has bit 23
    /// set as an "extended" flag, followed by the full u32 value.
    ///
    /// `PushSecureField([a,b,c,d])`: header + 4 data words.
    /// `PushTraceVal{interaction, col_idx, offset}`: single word with packed operand.
    /// `WideAddConst`, `WideMulConst`, etc.: header + 4 data words.
    /// `AddConst(v)`, `MulConst(v)`: same as PushBaseField encoding.
    /// `BaseAddSecureConst`, `BaseMulSecureConst`: header + 4 data words.
    ///
    /// Returns the encoded words and a mapping from instruction index to word offset
    /// (useful for debugging).
    pub fn encode(&self) -> Vec<u32> {
        let mut words = Vec::with_capacity(self.ops.len() * 2);

        for op in &self.ops {
            let opcode = op.opcode_id() as u32;

            match op {
                BytecodeOp::PushBaseField(v) => {
                    if *v < (1 << 24) {
                        words.push((opcode << 24) | *v);
                    } else {
                        // Extended: set bit 23 in operand field as flag
                        words.push((opcode << 24) | (1 << 23));
                        words.push(*v);
                    }
                }
                BytecodeOp::PushSecureField(limbs) => {
                    words.push(opcode << 24);
                    words.extend_from_slice(limbs);
                }
                BytecodeOp::PushTraceVal {
                    interaction,
                    col_idx,
                    offset,
                } => {
                    // Pack: [interaction:4 | col_idx:10 | offset_sign:1 | offset_abs:9]
                    // This supports interaction 0..15, col_idx 0..1023, offset -511..511
                    let inter = (*interaction as u32) & 0xF;
                    let col = (*col_idx as u32) & 0x3FF;
                    let (sign, abs_off) = if *offset < 0 {
                        (1u32, (-*offset) as u32)
                    } else {
                        (0u32, *offset as u32)
                    };
                    let operand = (inter << 20) | (col << 10) | (sign << 9) | (abs_off & 0x1FF);
                    words.push((opcode << 24) | operand);
                }
                // Simple no-operand ops
                BytecodeOp::Add
                | BytecodeOp::Sub
                | BytecodeOp::Mul
                | BytecodeOp::Neg
                | BytecodeOp::WideAdd
                | BytecodeOp::WideSub
                | BytecodeOp::WideMul
                | BytecodeOp::WideNeg
                | BytecodeOp::WideAddBase
                | BytecodeOp::WideMulBase
                | BytecodeOp::Widen
                | BytecodeOp::CombineEF
                | BytecodeOp::AddConstraint => {
                    words.push(opcode << 24);
                }
                BytecodeOp::AddConst(v) | BytecodeOp::MulConst(v) => {
                    if *v < (1 << 24) {
                        words.push((opcode << 24) | *v);
                    } else {
                        words.push((opcode << 24) | (1 << 23));
                        words.push(*v);
                    }
                }
                BytecodeOp::WideAddConst(limbs)
                | BytecodeOp::WideMulConst(limbs)
                | BytecodeOp::BaseAddSecureConst(limbs)
                | BytecodeOp::BaseMulSecureConst(limbs) => {
                    words.push(opcode << 24);
                    words.extend_from_slice(limbs);
                }
            }
        }

        words
    }

    /// Verify the program's stack balance by simulating execution.
    /// Returns `true` if the stack never goes negative (i.e., the program is valid
    /// for a stack-based interpreter). Returns `false` if Clone-induced stack
    /// imbalance is detected.
    pub fn verify_stack_balance(&self) -> bool {
        let mut sp: i32 = 0;
        for op in &self.ops {
            match op {
                BytecodeOp::PushBaseField(_)
                | BytecodeOp::PushSecureField(_)
                | BytecodeOp::PushTraceVal { .. } => {
                    sp += 1;
                }
                BytecodeOp::Add | BytecodeOp::Sub | BytecodeOp::Mul => {
                    sp -= 2;
                    if sp < 0 { return false; }
                    sp += 1;
                }
                BytecodeOp::Neg | BytecodeOp::AddConst(_) | BytecodeOp::MulConst(_) => {
                    if sp < 1 { return false; }
                    // pops 1, pushes 1 = net 0
                }
                BytecodeOp::WideAdd | BytecodeOp::WideSub | BytecodeOp::WideMul => {
                    sp -= 2;
                    if sp < 0 { return false; }
                    sp += 1;
                }
                BytecodeOp::WideNeg | BytecodeOp::WideAddConst(_) | BytecodeOp::WideMulConst(_) => {
                    if sp < 1 { return false; }
                }
                BytecodeOp::WideAddBase | BytecodeOp::WideMulBase => {
                    sp -= 2;
                    if sp < 0 { return false; }
                    sp += 1;
                }
                BytecodeOp::BaseAddSecureConst(_) | BytecodeOp::BaseMulSecureConst(_) => {
                    if sp < 1 { return false; }
                }
                BytecodeOp::Widen => {
                    if sp < 1 { return false; }
                }
                BytecodeOp::CombineEF => {
                    sp -= 4;
                    if sp < 0 { return false; }
                    sp += 1;
                }
                BytecodeOp::AddConstraint => {
                    sp -= 1;
                    if sp < 0 { return false; }
                }
            }
        }
        true
    }

    /// Pretty-print the bytecode as a numbered instruction listing.
    pub fn dump(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "BytecodeProgram: {} ops, {} constraints, max_stack={}\n",
            self.ops.len(),
            self.n_constraints,
            self.max_stack_depth
        ));
        for (i, op) in self.ops.iter().enumerate() {
            out.push_str(&format!("  {:4}: {}\n", i, op));
        }
        out
    }
}

/// Encode a bytecode op as a packed u32 word for GPU consumption.
///
/// Encoding scheme (first milestone is the recording, so this is forward-looking):
///   bits [31:24] = opcode
///   bits [23:0]  = operand (meaning depends on opcode)
///
/// Multi-word ops (e.g. QM31 constants) use the opcode word followed by extra data words.
impl BytecodeOp {
    /// Returns a compact opcode identifier (for future GPU encoding).
    pub fn opcode_id(&self) -> u8 {
        match self {
            BytecodeOp::PushBaseField(_) => 0x01,
            BytecodeOp::PushSecureField(_) => 0x02,
            BytecodeOp::PushTraceVal { .. } => 0x03,
            BytecodeOp::Add => 0x10,
            BytecodeOp::Sub => 0x11,
            BytecodeOp::Mul => 0x12,
            BytecodeOp::Neg => 0x13,
            BytecodeOp::AddConst(_) => 0x14,
            BytecodeOp::MulConst(_) => 0x15,
            BytecodeOp::WideAdd => 0x20,
            BytecodeOp::WideSub => 0x21,
            BytecodeOp::WideMul => 0x22,
            BytecodeOp::WideNeg => 0x23,
            BytecodeOp::WideAddConst(_) => 0x24,
            BytecodeOp::WideMulConst(_) => 0x25,
            BytecodeOp::WideAddBase => 0x28,
            BytecodeOp::WideMulBase => 0x29,
            BytecodeOp::BaseAddSecureConst(_) => 0x2A,
            BytecodeOp::BaseMulSecureConst(_) => 0x2B,
            BytecodeOp::Widen => 0x30,
            BytecodeOp::CombineEF => 0x31,
            BytecodeOp::AddConstraint => 0x40,
        }
    }
}

impl fmt::Display for BytecodeOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BytecodeOp::PushBaseField(v) => write!(f, "push_base  {v}"),
            BytecodeOp::PushSecureField(v) => {
                write!(f, "push_secure  ({}, {}, {}, {})", v[0], v[1], v[2], v[3])
            }
            BytecodeOp::PushTraceVal {
                interaction,
                col_idx,
                offset,
            } => write!(f, "push_trace  [inter={interaction}, col={col_idx}, off={offset}]"),
            BytecodeOp::Add => write!(f, "add"),
            BytecodeOp::Sub => write!(f, "sub"),
            BytecodeOp::Mul => write!(f, "mul"),
            BytecodeOp::Neg => write!(f, "neg"),
            BytecodeOp::AddConst(v) => write!(f, "add_const  {v}"),
            BytecodeOp::MulConst(v) => write!(f, "mul_const  {v}"),
            BytecodeOp::WideAdd => write!(f, "wide_add"),
            BytecodeOp::WideSub => write!(f, "wide_sub"),
            BytecodeOp::WideMul => write!(f, "wide_mul"),
            BytecodeOp::WideNeg => write!(f, "wide_neg"),
            BytecodeOp::WideAddConst(v) => {
                write!(f, "wide_add_const  ({}, {}, {}, {})", v[0], v[1], v[2], v[3])
            }
            BytecodeOp::WideMulConst(v) => {
                write!(f, "wide_mul_const  ({}, {}, {}, {})", v[0], v[1], v[2], v[3])
            }
            BytecodeOp::WideAddBase => write!(f, "wide_add_base"),
            BytecodeOp::WideMulBase => write!(f, "wide_mul_base"),
            BytecodeOp::BaseAddSecureConst(v) => {
                write!(
                    f,
                    "base_add_secure_const  ({}, {}, {}, {})",
                    v[0], v[1], v[2], v[3]
                )
            }
            BytecodeOp::BaseMulSecureConst(v) => {
                write!(
                    f,
                    "base_mul_secure_const  ({}, {}, {}, {})",
                    v[0], v[1], v[2], v[3]
                )
            }
            BytecodeOp::Widen => write!(f, "widen"),
            BytecodeOp::CombineEF => write!(f, "combine_ef"),
            BytecodeOp::AddConstraint => write!(f, "add_constraint"),
        }
    }
}

// ── Helper conversions ──────────────────────────────────────────────────

/// Convert a `BaseField` to its raw u32 representation for embedding in bytecode.
pub(crate) fn base_to_raw(v: BaseField) -> u32 {
    v.0
}

/// Convert a `SecureField` to its 4 raw u32 limbs for embedding in bytecode.
pub(crate) fn secure_to_raw(v: SecureField) -> [u32; 4] {
    let arr = v.to_m31_array();
    [arr[0].0, arr[1].0, arr[2].0, arr[3].0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_simple_ops() {
        let prog = BytecodeProgram {
            ops: vec![
                BytecodeOp::PushBaseField(42),
                BytecodeOp::PushBaseField(100),
                BytecodeOp::Add,
                BytecodeOp::AddConstraint,
            ],
            n_constraints: 1,
            n_trace_accesses: 0,
            max_stack_depth: 2,
        };
        let words = prog.encode();
        // PushBaseField(42): 0x01 << 24 | 42
        assert_eq!(words[0], (0x01 << 24) | 42);
        // PushBaseField(100): 0x01 << 24 | 100
        assert_eq!(words[1], (0x01 << 24) | 100);
        // Add: 0x10 << 24
        assert_eq!(words[2], 0x10 << 24);
        // AddConstraint: 0x40 << 24
        assert_eq!(words[3], 0x40 << 24);
    }

    #[test]
    fn test_encode_extended_basefield() {
        // Value > 2^24 needs 2-word encoding
        let big_val = (1 << 24) + 7;
        let prog = BytecodeProgram {
            ops: vec![BytecodeOp::PushBaseField(big_val)],
            n_constraints: 0,
            n_trace_accesses: 0,
            max_stack_depth: 1,
        };
        let words = prog.encode();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], (0x01 << 24) | (1 << 23)); // extended flag
        assert_eq!(words[1], big_val);
    }

    #[test]
    fn test_encode_secure_field() {
        let prog = BytecodeProgram {
            ops: vec![BytecodeOp::PushSecureField([10, 20, 30, 40])],
            n_constraints: 0,
            n_trace_accesses: 0,
            max_stack_depth: 1,
        };
        let words = prog.encode();
        assert_eq!(words.len(), 5);
        assert_eq!(words[0], 0x02 << 24);
        assert_eq!(words[1], 10);
        assert_eq!(words[2], 20);
        assert_eq!(words[3], 30);
        assert_eq!(words[4], 40);
    }

    #[test]
    fn test_encode_trace_val() {
        let prog = BytecodeProgram {
            ops: vec![BytecodeOp::PushTraceVal {
                interaction: 1,
                col_idx: 5,
                offset: -1,
            }],
            n_constraints: 0,
            n_trace_accesses: 1,
            max_stack_depth: 1,
        };
        let words = prog.encode();
        assert_eq!(words.len(), 1);
        let word = words[0];
        let opcode = word >> 24;
        assert_eq!(opcode, 0x03);
        let operand = word & 0xFFFFFF;
        let interaction = (operand >> 20) & 0xF;
        let col_idx = (operand >> 10) & 0x3FF;
        let sign = (operand >> 9) & 1;
        let abs_off = operand & 0x1FF;
        assert_eq!(interaction, 1);
        assert_eq!(col_idx, 5);
        assert_eq!(sign, 1); // negative
        assert_eq!(abs_off, 1);
    }

    #[test]
    fn test_encode_roundtrip_deterministic() {
        let prog = BytecodeProgram {
            ops: vec![
                BytecodeOp::PushTraceVal { interaction: 0, col_idx: 0, offset: 0 },
                BytecodeOp::PushBaseField(1),
                BytecodeOp::Sub,
                BytecodeOp::PushTraceVal { interaction: 0, col_idx: 0, offset: 0 },
                BytecodeOp::Mul,
                BytecodeOp::AddConstraint,
            ],
            n_constraints: 1,
            n_trace_accesses: 1,
            max_stack_depth: 2,
        };
        let w1 = prog.encode();
        let w2 = prog.encode();
        assert_eq!(w1, w2);
    }
}
