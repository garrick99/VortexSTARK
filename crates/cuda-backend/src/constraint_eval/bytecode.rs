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
