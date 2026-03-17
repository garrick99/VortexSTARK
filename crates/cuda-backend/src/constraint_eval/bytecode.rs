//! Register-based bytecode representation for GPU constraint evaluation.
//!
//! Each traced value is assigned a virtual register index. Operations read from
//! source registers and write to a destination register. This eliminates the
//! Clone problem that plagued the stack-based VM: Clone just copies a register
//! index — no bytecode emitted, no stack imbalance possible.
//!
//! ## Encoding format
//!
//! Each instruction is encoded as 1 or more u32 words:
//!
//! **3-register ops** (Add, Sub, Mul, WideAdd, ...):
//!   `[opcode:8 | dst:8 | src1:8 | src2:8]` — 1 word
//!
//! **2-register ops** (Neg, WideNeg, Widen):
//!   `[opcode:8 | dst:8 | src:8 | 0:8]` — 1 word
//!
//! **LoadTrace**:
//!   `[opcode:8 | dst:8 | flat_col:14 | sign:1 | abs_offset:1]` — 1 word (after remap)
//!   Pre-remap: `[opcode:8 | dst:8 | interaction:4 | col_idx:10 | sign:1 | abs_offset:1]`
//!   For offsets > 1: 2 words — header + packed (flat_col:16 | sign:1 | abs_offset:15)
//!
//! **LoadConst**:
//!   Small (< 2^16): `[opcode:8 | dst:8 | value:16]` — 1 word
//!   Large: `[opcode:8 | dst:8 | 0xFFFF:16]` + full u32 — 2 words
//!
//! **LoadSecureConst**: header `[opcode:8 | dst:8 | 0:16]` + 4 data words — 5 words
//!
//! **ConstOps** (AddConst, MulConst, WideAddConst, etc.):
//!   Similar to LoadConst/LoadSecureConst but with src register in place of dst.
//!
//! **AddConstraint**: `[opcode:8 | src:8 | 0:16]` — 1 word
//!
//! **CombineEF**: `[opcode:8 | dst:8 | src0:8 | src1:8]` + `[src2:8 | src3:8 | 0:16]` — 2 words

use std::fmt;

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;

/// A single register-based bytecode instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BytecodeOp {
    // ── Register loads ───────────────────────────────────────────────────

    /// Load an M31 constant into a register.
    LoadConst { dst: u16, value: u32 },

    /// Load a QM31 constant into a register (4 × M31 limbs).
    LoadSecureConst { dst: u16, value: [u32; 4] },

    /// Load a trace value into a register.
    LoadTrace {
        dst: u16,
        interaction: u16,
        col_idx: u16,
        offset: i16,
    },

    // ── M31 (base field) arithmetic ──────────────────────────────────────

    /// dst = src1 + src2 (M31)
    Add { dst: u16, src1: u16, src2: u16 },
    /// dst = src1 - src2 (M31)
    Sub { dst: u16, src1: u16, src2: u16 },
    /// dst = src1 * src2 (M31)
    Mul { dst: u16, src1: u16, src2: u16 },
    /// dst = -src (M31)
    Neg { dst: u16, src: u16 },
    /// dst = src + const (M31)
    AddConst { dst: u16, src: u16, value: u32 },
    /// dst = src * const (M31)
    MulConst { dst: u16, src: u16, value: u32 },

    // ── QM31 (secure extension field) arithmetic ─────────────────────────

    /// dst = src1 + src2 (QM31)
    WideAdd { dst: u16, src1: u16, src2: u16 },
    /// dst = src1 - src2 (QM31)
    WideSub { dst: u16, src1: u16, src2: u16 },
    /// dst = src1 * src2 (QM31)
    WideMul { dst: u16, src1: u16, src2: u16 },
    /// dst = -src (QM31)
    WideNeg { dst: u16, src: u16 },
    /// dst = src + QM31_const
    WideAddConst { dst: u16, src: u16, value: [u32; 4] },
    /// dst = src * QM31_const
    WideMulConst { dst: u16, src: u16, value: [u32; 4] },

    // ── Mixed-width arithmetic ───────────────────────────────────────────

    /// dst = wide_src + base_src (base widened to QM31)
    WideAddBase { dst: u16, wide: u16, base: u16 },
    /// dst = wide_src * base_src
    WideMulBase { dst: u16, wide: u16, base: u16 },
    /// dst(QM31) = base_src(M31) + QM31_const
    BaseAddSecureConst { dst: u16, src: u16, value: [u32; 4] },
    /// dst(QM31) = base_src(M31) * QM31_const
    BaseMulSecureConst { dst: u16, src: u16, value: [u32; 4] },

    // ── Widening ─────────────────────────────────────────────────────────

    /// dst = widen(src) — M31 to QM31 (embed in first coordinate)
    Widen { dst: u16, src: u16 },
    /// dst(QM31) = combine(src0, src1, src2, src3) — 4 M31 registers into 1 QM31
    CombineEF { dst: u16, src: [u16; 4] },

    // ── Constraint accumulation ──────────────────────────────────────────

    /// Accumulate constraint value from src register.
    AddConstraint { src: u16 },
}

/// A compiled register-based bytecode program for a single component.
#[derive(Clone, Debug)]
pub struct BytecodeProgram {
    /// The flat instruction stream.
    pub ops: Vec<BytecodeOp>,
    /// Number of constraints (= number of `AddConstraint` ops).
    pub n_constraints: usize,
    /// Number of distinct trace column accesses.
    pub n_trace_accesses: usize,
    /// Number of virtual registers used.
    pub n_registers: u16,
}

// Opcode IDs for GPU encoding.
// These must match the #defines in constraint_eval.cu.
const OP_LOAD_CONST: u8 = 0x01;
const OP_LOAD_SECURE_CONST: u8 = 0x02;
const OP_LOAD_TRACE: u8 = 0x03;
const OP_ADD: u8 = 0x10;
const OP_SUB: u8 = 0x11;
const OP_MUL: u8 = 0x12;
const OP_NEG: u8 = 0x13;
const OP_ADD_CONST: u8 = 0x14;
const OP_MUL_CONST: u8 = 0x15;
const OP_WIDE_ADD: u8 = 0x20;
const OP_WIDE_SUB: u8 = 0x21;
const OP_WIDE_MUL: u8 = 0x22;
const OP_WIDE_NEG: u8 = 0x23;
const OP_WIDE_ADD_CONST: u8 = 0x24;
const OP_WIDE_MUL_CONST: u8 = 0x25;
const OP_WIDE_ADD_BASE: u8 = 0x28;
const OP_WIDE_MUL_BASE: u8 = 0x29;
const OP_BASE_ADD_SECURE_CONST: u8 = 0x2A;
const OP_BASE_MUL_SECURE_CONST: u8 = 0x2B;
const OP_WIDEN: u8 = 0x30;
const OP_COMBINE_EF: u8 = 0x31;
const OP_ADD_CONSTRAINT: u8 = 0x40;

impl BytecodeProgram {
    /// Encode the bytecode program as a flat `Vec<u32>` for GPU consumption.
    ///
    /// Register-based encoding: each op encodes dst/src register indices inline.
    /// All register indices are u8 (max 255 registers).
    pub fn encode(&self) -> Vec<u32> {
        let mut words = Vec::with_capacity(self.ops.len() * 2);

        for op in &self.ops {
            match op {
                BytecodeOp::LoadConst { dst, value } => {
                    let d = *dst as u32 & 0xFF;
                    if *value < (1 << 16) {
                        // Single word: [opcode:8 | dst:8 | value:16]
                        words.push(((OP_LOAD_CONST as u32) << 24) | (d << 16) | *value);
                    } else {
                        // Two words: header with 0xFFFF marker, then full value
                        words.push(((OP_LOAD_CONST as u32) << 24) | (d << 16) | 0xFFFF);
                        words.push(*value);
                    }
                }
                BytecodeOp::LoadSecureConst { dst, value } => {
                    let d = *dst as u32 & 0xFF;
                    words.push(((OP_LOAD_SECURE_CONST as u32) << 24) | (d << 16));
                    words.extend_from_slice(value);
                }
                BytecodeOp::LoadTrace { dst, interaction, col_idx, offset } => {
                    // Pre-remap encoding (Rust side sees interaction/col_idx).
                    // component_prover.rs will remap to flat col indices.
                    // Encoding: word1 = [opcode:8 | dst:8 | interaction:4 | col_idx:10 | sign:1 | abs_offset:1]
                    // For |offset| > 1: word1 marker + word2 with full offset
                    let d = *dst as u32 & 0xFF;
                    let inter = (*interaction as u32) & 0xF;
                    let col = (*col_idx as u32) & 0x3FF;
                    let (sign, abs_off) = if *offset < 0 {
                        (1u32, (-*offset) as u32)
                    } else {
                        (0u32, *offset as u32)
                    };
                    if abs_off <= 1 {
                        let operand = (inter << 12) | (col << 2) | (sign << 1) | (abs_off & 1);
                        words.push(((OP_LOAD_TRACE as u32) << 24) | (d << 16) | operand);
                    } else {
                        // Extended: marker bit in operand, second word has full info
                        let operand = (inter << 12) | (col << 2) | 0x3; // 0x3 = both bits set = marker
                        words.push(((OP_LOAD_TRACE as u32) << 24) | (d << 16) | operand);
                        words.push((sign << 31) | abs_off);
                    }
                }

                // 3-register arithmetic: [opcode:8 | dst:8 | src1:8 | src2:8]
                BytecodeOp::Add { dst, src1, src2 } => {
                    words.push(encode_3reg(OP_ADD, *dst, *src1, *src2));
                }
                BytecodeOp::Sub { dst, src1, src2 } => {
                    words.push(encode_3reg(OP_SUB, *dst, *src1, *src2));
                }
                BytecodeOp::Mul { dst, src1, src2 } => {
                    words.push(encode_3reg(OP_MUL, *dst, *src1, *src2));
                }
                BytecodeOp::WideAdd { dst, src1, src2 } => {
                    words.push(encode_3reg(OP_WIDE_ADD, *dst, *src1, *src2));
                }
                BytecodeOp::WideSub { dst, src1, src2 } => {
                    words.push(encode_3reg(OP_WIDE_SUB, *dst, *src1, *src2));
                }
                BytecodeOp::WideMul { dst, src1, src2 } => {
                    words.push(encode_3reg(OP_WIDE_MUL, *dst, *src1, *src2));
                }
                BytecodeOp::WideAddBase { dst, wide, base } => {
                    words.push(encode_3reg(OP_WIDE_ADD_BASE, *dst, *wide, *base));
                }
                BytecodeOp::WideMulBase { dst, wide, base } => {
                    words.push(encode_3reg(OP_WIDE_MUL_BASE, *dst, *wide, *base));
                }

                // 2-register ops: [opcode:8 | dst:8 | src:8 | 0:8]
                BytecodeOp::Neg { dst, src } => {
                    words.push(encode_2reg(OP_NEG, *dst, *src));
                }
                BytecodeOp::WideNeg { dst, src } => {
                    words.push(encode_2reg(OP_WIDE_NEG, *dst, *src));
                }
                BytecodeOp::Widen { dst, src } => {
                    words.push(encode_2reg(OP_WIDEN, *dst, *src));
                }

                // Const ops: [opcode:8 | dst:8 | src:8 | 0:8] + value word(s)
                BytecodeOp::AddConst { dst, src, value } => {
                    words.push(encode_2reg(OP_ADD_CONST, *dst, *src));
                    words.push(*value);
                }
                BytecodeOp::MulConst { dst, src, value } => {
                    words.push(encode_2reg(OP_MUL_CONST, *dst, *src));
                    words.push(*value);
                }
                BytecodeOp::WideAddConst { dst, src, value } => {
                    words.push(encode_2reg(OP_WIDE_ADD_CONST, *dst, *src));
                    words.extend_from_slice(value);
                }
                BytecodeOp::WideMulConst { dst, src, value } => {
                    words.push(encode_2reg(OP_WIDE_MUL_CONST, *dst, *src));
                    words.extend_from_slice(value);
                }
                BytecodeOp::BaseAddSecureConst { dst, src, value } => {
                    words.push(encode_2reg(OP_BASE_ADD_SECURE_CONST, *dst, *src));
                    words.extend_from_slice(value);
                }
                BytecodeOp::BaseMulSecureConst { dst, src, value } => {
                    words.push(encode_2reg(OP_BASE_MUL_SECURE_CONST, *dst, *src));
                    words.extend_from_slice(value);
                }

                // CombineEF: 2 words
                // word1: [opcode:8 | dst:8 | src0:8 | src1:8]
                // word2: [src2:8 | src3:8 | 0:16]
                BytecodeOp::CombineEF { dst, src } => {
                    words.push(encode_3reg(OP_COMBINE_EF, *dst, src[0], src[1]));
                    words.push(((src[2] as u32 & 0xFF) << 24) | ((src[3] as u32 & 0xFF) << 16));
                }

                // AddConstraint: [opcode:8 | src:8 | 0:16]
                BytecodeOp::AddConstraint { src } => {
                    words.push(((OP_ADD_CONSTRAINT as u32) << 24) | ((*src as u32 & 0xFF) << 16));
                }
            }
        }

        words
    }

    /// Pretty-print the bytecode as a numbered instruction listing.
    pub fn dump(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "BytecodeProgram (register-based): {} ops, {} constraints, {} registers\n",
            self.ops.len(),
            self.n_constraints,
            self.n_registers
        ));
        for (i, op) in self.ops.iter().enumerate() {
            out.push_str(&format!("  {:4}: {}\n", i, op));
        }
        out
    }
}

/// Encode a 3-register op: [opcode:8 | dst:8 | src1:8 | src2:8]
fn encode_3reg(opcode: u8, dst: u16, src1: u16, src2: u16) -> u32 {
    ((opcode as u32) << 24)
        | ((dst as u32 & 0xFF) << 16)
        | ((src1 as u32 & 0xFF) << 8)
        | (src2 as u32 & 0xFF)
}

/// Encode a 2-register op: [opcode:8 | dst:8 | src:8 | 0:8]
fn encode_2reg(opcode: u8, dst: u16, src: u16) -> u32 {
    ((opcode as u32) << 24)
        | ((dst as u32 & 0xFF) << 16)
        | ((src as u32 & 0xFF) << 8)
}

impl BytecodeOp {
    /// Returns a compact opcode identifier for GPU encoding.
    pub fn opcode_id(&self) -> u8 {
        match self {
            BytecodeOp::LoadConst { .. } => OP_LOAD_CONST,
            BytecodeOp::LoadSecureConst { .. } => OP_LOAD_SECURE_CONST,
            BytecodeOp::LoadTrace { .. } => OP_LOAD_TRACE,
            BytecodeOp::Add { .. } => OP_ADD,
            BytecodeOp::Sub { .. } => OP_SUB,
            BytecodeOp::Mul { .. } => OP_MUL,
            BytecodeOp::Neg { .. } => OP_NEG,
            BytecodeOp::AddConst { .. } => OP_ADD_CONST,
            BytecodeOp::MulConst { .. } => OP_MUL_CONST,
            BytecodeOp::WideAdd { .. } => OP_WIDE_ADD,
            BytecodeOp::WideSub { .. } => OP_WIDE_SUB,
            BytecodeOp::WideMul { .. } => OP_WIDE_MUL,
            BytecodeOp::WideNeg { .. } => OP_WIDE_NEG,
            BytecodeOp::WideAddConst { .. } => OP_WIDE_ADD_CONST,
            BytecodeOp::WideMulConst { .. } => OP_WIDE_MUL_CONST,
            BytecodeOp::WideAddBase { .. } => OP_WIDE_ADD_BASE,
            BytecodeOp::WideMulBase { .. } => OP_WIDE_MUL_BASE,
            BytecodeOp::BaseAddSecureConst { .. } => OP_BASE_ADD_SECURE_CONST,
            BytecodeOp::BaseMulSecureConst { .. } => OP_BASE_MUL_SECURE_CONST,
            BytecodeOp::Widen { .. } => OP_WIDEN,
            BytecodeOp::CombineEF { .. } => OP_COMBINE_EF,
            BytecodeOp::AddConstraint { .. } => OP_ADD_CONSTRAINT,
        }
    }

    /// Number of u32 words this op occupies when encoded.
    pub fn encoded_len(&self) -> usize {
        match self {
            BytecodeOp::LoadConst { value, .. } => {
                if *value < (1 << 16) { 1 } else { 2 }
            }
            BytecodeOp::LoadSecureConst { .. } => 5,
            BytecodeOp::LoadTrace { offset, .. } => {
                let abs_off = if *offset < 0 { -(*offset) as u32 } else { *offset as u32 };
                if abs_off <= 1 { 1 } else { 2 }
            }
            BytecodeOp::Add { .. }
            | BytecodeOp::Sub { .. }
            | BytecodeOp::Mul { .. }
            | BytecodeOp::Neg { .. }
            | BytecodeOp::WideAdd { .. }
            | BytecodeOp::WideSub { .. }
            | BytecodeOp::WideMul { .. }
            | BytecodeOp::WideNeg { .. }
            | BytecodeOp::WideAddBase { .. }
            | BytecodeOp::WideMulBase { .. }
            | BytecodeOp::Widen { .. }
            | BytecodeOp::AddConstraint { .. } => 1,
            BytecodeOp::AddConst { .. } | BytecodeOp::MulConst { .. } => 2,
            BytecodeOp::WideAddConst { .. }
            | BytecodeOp::WideMulConst { .. }
            | BytecodeOp::BaseAddSecureConst { .. }
            | BytecodeOp::BaseMulSecureConst { .. } => 5,
            BytecodeOp::CombineEF { .. } => 2,
        }
    }
}

impl fmt::Display for BytecodeOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BytecodeOp::LoadConst { dst, value } => write!(f, "r{dst} = load_const {value}"),
            BytecodeOp::LoadSecureConst { dst, value } => {
                write!(f, "r{dst} = load_secure ({}, {}, {}, {})", value[0], value[1], value[2], value[3])
            }
            BytecodeOp::LoadTrace { dst, interaction, col_idx, offset } => {
                write!(f, "r{dst} = load_trace [inter={interaction}, col={col_idx}, off={offset}]")
            }
            BytecodeOp::Add { dst, src1, src2 } => write!(f, "r{dst} = r{src1} + r{src2}"),
            BytecodeOp::Sub { dst, src1, src2 } => write!(f, "r{dst} = r{src1} - r{src2}"),
            BytecodeOp::Mul { dst, src1, src2 } => write!(f, "r{dst} = r{src1} * r{src2}"),
            BytecodeOp::Neg { dst, src } => write!(f, "r{dst} = -r{src}"),
            BytecodeOp::AddConst { dst, src, value } => write!(f, "r{dst} = r{src} + {value}"),
            BytecodeOp::MulConst { dst, src, value } => write!(f, "r{dst} = r{src} * {value}"),
            BytecodeOp::WideAdd { dst, src1, src2 } => write!(f, "r{dst} = r{src1} w+ r{src2}"),
            BytecodeOp::WideSub { dst, src1, src2 } => write!(f, "r{dst} = r{src1} w- r{src2}"),
            BytecodeOp::WideMul { dst, src1, src2 } => write!(f, "r{dst} = r{src1} w* r{src2}"),
            BytecodeOp::WideNeg { dst, src } => write!(f, "r{dst} = w-r{src}"),
            BytecodeOp::WideAddConst { dst, src, value } => {
                write!(f, "r{dst} = r{src} w+ ({}, {}, {}, {})", value[0], value[1], value[2], value[3])
            }
            BytecodeOp::WideMulConst { dst, src, value } => {
                write!(f, "r{dst} = r{src} w* ({}, {}, {}, {})", value[0], value[1], value[2], value[3])
            }
            BytecodeOp::WideAddBase { dst, wide, base } => write!(f, "r{dst} = r{wide} w+ widen(r{base})"),
            BytecodeOp::WideMulBase { dst, wide, base } => write!(f, "r{dst} = r{wide} w* r{base}"),
            BytecodeOp::BaseAddSecureConst { dst, src, value } => {
                write!(f, "r{dst} = widen(r{src}) + ({}, {}, {}, {})", value[0], value[1], value[2], value[3])
            }
            BytecodeOp::BaseMulSecureConst { dst, src, value } => {
                write!(f, "r{dst} = r{src} * ({}, {}, {}, {})", value[0], value[1], value[2], value[3])
            }
            BytecodeOp::Widen { dst, src } => write!(f, "r{dst} = widen(r{src})"),
            BytecodeOp::CombineEF { dst, src } => {
                write!(f, "r{dst} = combine(r{}, r{}, r{}, r{})", src[0], src[1], src[2], src[3])
            }
            BytecodeOp::AddConstraint { src } => write!(f, "add_constraint r{src}"),
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
    fn test_encode_3reg_op() {
        let prog = BytecodeProgram {
            ops: vec![
                BytecodeOp::LoadConst { dst: 0, value: 42 },
                BytecodeOp::LoadConst { dst: 1, value: 100 },
                BytecodeOp::Add { dst: 2, src1: 0, src2: 1 },
                BytecodeOp::AddConstraint { src: 2 },
            ],
            n_constraints: 1,
            n_trace_accesses: 0,
            n_registers: 3,
        };
        let words = prog.encode();
        // LoadConst(dst=0, val=42): [0x01:8 | 0:8 | 42:16]
        assert_eq!(words[0], (0x01 << 24) | (0 << 16) | 42);
        // LoadConst(dst=1, val=100): [0x01:8 | 1:8 | 100:16]
        assert_eq!(words[1], (0x01 << 24) | (1 << 16) | 100);
        // Add(dst=2, src1=0, src2=1): [0x10:8 | 2:8 | 0:8 | 1:8]
        assert_eq!(words[2], (0x10 << 24) | (2 << 16) | (0 << 8) | 1);
        // AddConstraint(src=2): [0x40:8 | 2:8 | 0:16]
        assert_eq!(words[3], (0x40 << 24) | (2 << 16));
    }

    #[test]
    fn test_encode_extended_const() {
        let big_val = (1 << 24) + 7;
        let prog = BytecodeProgram {
            ops: vec![BytecodeOp::LoadConst { dst: 0, value: big_val }],
            n_constraints: 0,
            n_trace_accesses: 0,
            n_registers: 1,
        };
        let words = prog.encode();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], (0x01 << 24) | (0 << 16) | 0xFFFF);
        assert_eq!(words[1], big_val);
    }

    #[test]
    fn test_encode_secure_field() {
        let prog = BytecodeProgram {
            ops: vec![BytecodeOp::LoadSecureConst { dst: 5, value: [10, 20, 30, 40] }],
            n_constraints: 0,
            n_trace_accesses: 0,
            n_registers: 6,
        };
        let words = prog.encode();
        assert_eq!(words.len(), 5);
        assert_eq!(words[0], (0x02 << 24) | (5 << 16));
        assert_eq!(words[1], 10);
        assert_eq!(words[2], 20);
        assert_eq!(words[3], 30);
        assert_eq!(words[4], 40);
    }

    #[test]
    fn test_encode_trace_val() {
        let prog = BytecodeProgram {
            ops: vec![BytecodeOp::LoadTrace {
                dst: 3,
                interaction: 1,
                col_idx: 5,
                offset: -1,
            }],
            n_constraints: 0,
            n_trace_accesses: 1,
            n_registers: 4,
        };
        let words = prog.encode();
        assert_eq!(words.len(), 1);
        let word = words[0];
        let opcode = word >> 24;
        assert_eq!(opcode, 0x03);
        let dst = (word >> 16) & 0xFF;
        assert_eq!(dst, 3);
        let operand = word & 0xFFFF;
        // interaction:4=1 | col_idx:10=5 | sign:1=1 | abs_offset:1=1
        let interaction = (operand >> 12) & 0xF;
        let col_idx = (operand >> 2) & 0x3FF;
        let sign = (operand >> 1) & 1;
        let abs_off = operand & 1;
        assert_eq!(interaction, 1);
        assert_eq!(col_idx, 5);
        assert_eq!(sign, 1);
        assert_eq!(abs_off, 1);
    }

    #[test]
    fn test_encode_roundtrip_deterministic() {
        let prog = BytecodeProgram {
            ops: vec![
                BytecodeOp::LoadTrace { dst: 0, interaction: 0, col_idx: 0, offset: 0 },
                BytecodeOp::LoadConst { dst: 1, value: 1 },
                BytecodeOp::Sub { dst: 2, src1: 1, src2: 0 },
                BytecodeOp::Mul { dst: 3, src1: 0, src2: 2 },
                BytecodeOp::AddConstraint { src: 3 },
            ],
            n_constraints: 1,
            n_trace_accesses: 1,
            n_registers: 4,
        };
        let w1 = prog.encode();
        let w2 = prog.encode();
        assert_eq!(w1, w2);
    }
}
