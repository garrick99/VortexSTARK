//! Register-based bytecode representation for GPU constraint evaluation.
//!
//! Each traced value is assigned a virtual register index. Operations read from
//! source registers and write to a destination register. This eliminates the
//! Clone problem that plagued the stack-based VM: Clone just copies a register
//! index — no bytecode emitted, no stack imbalance possible.
//!
//! ## Encoding format
//!
//! All register indices are 16-bit (u16), supporting up to 65536 virtual registers.
//! Every instruction begins with a header word `[opcode:8 | 0:8 | dst:16]`.
//!
//! **3-register ops** (Add, Sub, Mul, WideAdd, ...):
//!   word1: `[opcode:8 | 0:8 | dst:16]`
//!   word2: `[src1:16 | src2:16]`
//!
//! **2-register ops** (Neg, WideNeg, Widen):
//!   word1: `[opcode:8 | 0:8 | dst:16]`
//!   word2: `[src:16 | 0:16]`
//!
//! **LoadConst**:
//!   word1: `[opcode:8 | 0:8 | dst:16]`
//!   word2: `[value:32]`
//!
//! **LoadSecureConst**:
//!   word1: `[opcode:8 | 0:8 | dst:16]`
//!   words 2–5: value[0..3]
//!
//! **LoadTrace** (pre-remap):
//!   word1: `[opcode:8 | 0:8 | dst:16]`
//!   word2: `[interaction:16 | col_idx:16]`
//!   word3: `[sign:1 | abs_offset:31]`
//!   After remap by component_prover.rs:
//!   word2: `[flat_col:32]`  (word3 unchanged)
//!
//! **ConstOps** (AddConst, MulConst):
//!   word1: `[opcode:8 | 0:8 | dst:16]`
//!   word2: `[src:16 | 0:16]`
//!   word3: `[value:32]`
//!
//! **SecureConstOps** (WideAddConst, WideMulConst, BaseAddSecureConst, BaseMulSecureConst):
//!   word1: `[opcode:8 | 0:8 | dst:16]`
//!   word2: `[src:16 | 0:16]`
//!   words 3–6: value[0..3]
//!
//! **CombineEF**:
//!   word1: `[opcode:8 | 0:8 | dst:16]`
//!   word2: `[src0:16 | src1:16]`
//!   word3: `[src2:16 | src3:16]`
//!
//! **AddConstraint**: `[opcode:8 | 0:8 | src:16]` — 1 word (no dst)

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
    /// All register indices are 16-bit (max 65535 registers).
    pub fn encode(&self) -> Vec<u32> {
        let mut words = Vec::with_capacity(self.ops.len() * 3);

        for op in &self.ops {
            match op {
                BytecodeOp::LoadConst { dst, value } => {
                    // 2 words: header + value
                    words.push(hdr(OP_LOAD_CONST, *dst));
                    words.push(*value);
                }
                BytecodeOp::LoadSecureConst { dst, value } => {
                    // 5 words: header + 4 data words
                    words.push(hdr(OP_LOAD_SECURE_CONST, *dst));
                    words.extend_from_slice(value);
                }
                BytecodeOp::LoadTrace { dst, interaction, col_idx, offset } => {
                    // 3 words: header, [interaction:16|col_idx:16], [sign:1|abs_off:31]
                    // component_prover.rs will remap word2 to flat_col.
                    let (sign, abs_off) = if *offset < 0 {
                        (1u32, (-*offset) as u32)
                    } else {
                        (0u32, *offset as u32)
                    };
                    words.push(hdr(OP_LOAD_TRACE, *dst));
                    words.push(((*interaction as u32) << 16) | (*col_idx as u32));
                    words.push((sign << 31) | abs_off);
                }

                // 3-register arithmetic: 2 words
                BytecodeOp::Add { dst, src1, src2 } => {
                    encode_3reg(&mut words, OP_ADD, *dst, *src1, *src2);
                }
                BytecodeOp::Sub { dst, src1, src2 } => {
                    encode_3reg(&mut words, OP_SUB, *dst, *src1, *src2);
                }
                BytecodeOp::Mul { dst, src1, src2 } => {
                    encode_3reg(&mut words, OP_MUL, *dst, *src1, *src2);
                }
                BytecodeOp::WideAdd { dst, src1, src2 } => {
                    encode_3reg(&mut words, OP_WIDE_ADD, *dst, *src1, *src2);
                }
                BytecodeOp::WideSub { dst, src1, src2 } => {
                    encode_3reg(&mut words, OP_WIDE_SUB, *dst, *src1, *src2);
                }
                BytecodeOp::WideMul { dst, src1, src2 } => {
                    encode_3reg(&mut words, OP_WIDE_MUL, *dst, *src1, *src2);
                }
                BytecodeOp::WideAddBase { dst, wide, base } => {
                    encode_3reg(&mut words, OP_WIDE_ADD_BASE, *dst, *wide, *base);
                }
                BytecodeOp::WideMulBase { dst, wide, base } => {
                    encode_3reg(&mut words, OP_WIDE_MUL_BASE, *dst, *wide, *base);
                }

                // 2-register ops: 2 words
                BytecodeOp::Neg { dst, src } => {
                    encode_2reg(&mut words, OP_NEG, *dst, *src);
                }
                BytecodeOp::WideNeg { dst, src } => {
                    encode_2reg(&mut words, OP_WIDE_NEG, *dst, *src);
                }
                BytecodeOp::Widen { dst, src } => {
                    encode_2reg(&mut words, OP_WIDEN, *dst, *src);
                }

                // M31 const ops: 3 words (header + src word + value)
                BytecodeOp::AddConst { dst, src, value } => {
                    words.push(hdr(OP_ADD_CONST, *dst));
                    words.push((*src as u32) << 16);
                    words.push(*value);
                }
                BytecodeOp::MulConst { dst, src, value } => {
                    words.push(hdr(OP_MUL_CONST, *dst));
                    words.push((*src as u32) << 16);
                    words.push(*value);
                }

                // QM31 const ops: 6 words (header + src word + 4 data words)
                BytecodeOp::WideAddConst { dst, src, value } => {
                    words.push(hdr(OP_WIDE_ADD_CONST, *dst));
                    words.push((*src as u32) << 16);
                    words.extend_from_slice(value);
                }
                BytecodeOp::WideMulConst { dst, src, value } => {
                    words.push(hdr(OP_WIDE_MUL_CONST, *dst));
                    words.push((*src as u32) << 16);
                    words.extend_from_slice(value);
                }
                BytecodeOp::BaseAddSecureConst { dst, src, value } => {
                    words.push(hdr(OP_BASE_ADD_SECURE_CONST, *dst));
                    words.push((*src as u32) << 16);
                    words.extend_from_slice(value);
                }
                BytecodeOp::BaseMulSecureConst { dst, src, value } => {
                    words.push(hdr(OP_BASE_MUL_SECURE_CONST, *dst));
                    words.push((*src as u32) << 16);
                    words.extend_from_slice(value);
                }

                // CombineEF: 3 words
                BytecodeOp::CombineEF { dst, src } => {
                    words.push(hdr(OP_COMBINE_EF, *dst));
                    words.push(((src[0] as u32) << 16) | (src[1] as u32));
                    words.push(((src[2] as u32) << 16) | (src[3] as u32));
                }

                // AddConstraint: 1 word, src in low 16 bits
                BytecodeOp::AddConstraint { src } => {
                    words.push(((OP_ADD_CONSTRAINT as u32) << 24) | (*src as u32));
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

/// Build a header word: `[opcode:8 | 0:8 | dst:16]`
#[inline]
fn hdr(opcode: u8, dst: u16) -> u32 {
    ((opcode as u32) << 24) | (dst as u32)
}

/// Encode a 3-register op: 2 words.
/// word1: `[opcode:8 | 0:8 | dst:16]`, word2: `[src1:16 | src2:16]`
fn encode_3reg(words: &mut Vec<u32>, opcode: u8, dst: u16, src1: u16, src2: u16) {
    words.push(hdr(opcode, dst));
    words.push(((src1 as u32) << 16) | (src2 as u32));
}

/// Encode a 2-register op: 2 words.
/// word1: `[opcode:8 | 0:8 | dst:16]`, word2: `[src:16 | 0:16]`
fn encode_2reg(words: &mut Vec<u32>, opcode: u8, dst: u16, src: u16) {
    words.push(hdr(opcode, dst));
    words.push((src as u32) << 16);
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
            // 2 words: header + value
            BytecodeOp::LoadConst { .. } => 2,
            // 5 words: header + 4 data
            BytecodeOp::LoadSecureConst { .. } => 5,
            // 3 words: header + interaction/col word + sign/abs_off word
            BytecodeOp::LoadTrace { .. } => 3,
            // 2 words: header + src1/src2 word
            BytecodeOp::Add { .. }
            | BytecodeOp::Sub { .. }
            | BytecodeOp::Mul { .. }
            | BytecodeOp::WideAdd { .. }
            | BytecodeOp::WideSub { .. }
            | BytecodeOp::WideMul { .. }
            | BytecodeOp::WideAddBase { .. }
            | BytecodeOp::WideMulBase { .. } => 2,
            // 2 words: header + src word
            BytecodeOp::Neg { .. }
            | BytecodeOp::WideNeg { .. }
            | BytecodeOp::Widen { .. } => 2,
            // 1 word: opcode + src
            BytecodeOp::AddConstraint { .. } => 1,
            // 3 words: header + src word + value
            BytecodeOp::AddConst { .. } | BytecodeOp::MulConst { .. } => 3,
            // 6 words: header + src word + 4 data
            BytecodeOp::WideAddConst { .. }
            | BytecodeOp::WideMulConst { .. }
            | BytecodeOp::BaseAddSecureConst { .. }
            | BytecodeOp::BaseMulSecureConst { .. } => 6,
            // 3 words: header + src0/src1 word + src2/src3 word
            BytecodeOp::CombineEF { .. } => 3,
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
        // LoadConst(dst=0, val=42): word[0]=[0x01:8|0:8|0:16], word[1]=42
        assert_eq!(words[0], (0x01u32 << 24) | 0);
        assert_eq!(words[1], 42);
        // LoadConst(dst=1, val=100): word[2]=[0x01:8|0:8|1:16], word[3]=100
        assert_eq!(words[2], (0x01u32 << 24) | 1);
        assert_eq!(words[3], 100);
        // Add(dst=2, src1=0, src2=1): word[4]=[0x10:8|0:8|2:16], word[5]=[0:16|1:16]
        assert_eq!(words[4], (0x10u32 << 24) | 2);
        assert_eq!(words[5], (0u32 << 16) | 1);
        // AddConstraint(src=2): word[6]=[0x40:8|0:8|2:16]
        assert_eq!(words[6], (0x40u32 << 24) | 2);
        assert_eq!(words.len(), 7);
    }

    #[test]
    fn test_encode_load_const_always_2_words() {
        // Small value
        let prog = BytecodeProgram {
            ops: vec![BytecodeOp::LoadConst { dst: 0, value: 42 }],
            n_constraints: 0,
            n_trace_accesses: 0,
            n_registers: 1,
        };
        let words = prog.encode();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], (0x01u32 << 24) | 0);
        assert_eq!(words[1], 42);

        // Large value
        let big_val = (1 << 24) + 7;
        let prog2 = BytecodeProgram {
            ops: vec![BytecodeOp::LoadConst { dst: 0, value: big_val }],
            n_constraints: 0,
            n_trace_accesses: 0,
            n_registers: 1,
        };
        let words2 = prog2.encode();
        assert_eq!(words2.len(), 2);
        assert_eq!(words2[0], (0x01u32 << 24) | 0);
        assert_eq!(words2[1], big_val);
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
        assert_eq!(words[0], (0x02u32 << 24) | 5);
        assert_eq!(words[1], 10);
        assert_eq!(words[2], 20);
        assert_eq!(words[3], 30);
        assert_eq!(words[4], 40);
    }

    #[test]
    fn test_encode_load_trace_3_words() {
        // offset=-1: 3 words, word3 has sign=1 | abs=1
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
        assert_eq!(words.len(), 3, "LoadTrace always 3 words");
        // word1: [0x03:8 | 0:8 | 3:16]
        assert_eq!(words[0], (0x03u32 << 24) | 3);
        // word2: [interaction:16 | col_idx:16]
        assert_eq!(words[1], (1u32 << 16) | 5);
        // word3: [sign:1 | abs_off:31] = 1<<31 | 1
        assert_eq!(words[2], (1u32 << 31) | 1);
    }

    #[test]
    fn test_encode_load_trace_positive_offset() {
        // offset=+1: 3 words, word3 has sign=0 | abs=1
        let prog = BytecodeProgram {
            ops: vec![BytecodeOp::LoadTrace {
                dst: 2,
                interaction: 0,
                col_idx: 3,
                offset: 1,
            }],
            n_constraints: 0,
            n_trace_accesses: 1,
            n_registers: 3,
        };
        let words = prog.encode();
        assert_eq!(words.len(), 3, "LoadTrace always 3 words");
        assert_eq!(words[0], (0x03u32 << 24) | 2);
        assert_eq!(words[1], (0u32 << 16) | 3);
        assert_eq!(words[2], 1u32); // sign=0, abs=1
    }

    #[test]
    fn test_encode_load_trace_zero_offset() {
        let prog = BytecodeProgram {
            ops: vec![BytecodeOp::LoadTrace {
                dst: 0,
                interaction: 0,
                col_idx: 0,
                offset: 0,
            }],
            n_constraints: 0,
            n_trace_accesses: 1,
            n_registers: 1,
        };
        let words = prog.encode();
        assert_eq!(words.len(), 3);
        assert_eq!(words[0], (0x03u32 << 24) | 0);
        assert_eq!(words[1], 0u32); // interaction=0, col_idx=0
        assert_eq!(words[2], 0u32); // sign=0, abs=0
    }

    #[test]
    fn test_encode_high_register_indices() {
        // Verify 16-bit register indices work (> 255)
        let prog = BytecodeProgram {
            ops: vec![
                BytecodeOp::LoadConst { dst: 300, value: 1 },
                BytecodeOp::LoadConst { dst: 500, value: 2 },
                BytecodeOp::Add { dst: 1000, src1: 300, src2: 500 },
                BytecodeOp::AddConstraint { src: 1000 },
            ],
            n_constraints: 1,
            n_trace_accesses: 0,
            n_registers: 1001,
        };
        let words = prog.encode();
        // LoadConst dst=300
        assert_eq!(words[0], (0x01u32 << 24) | 300);
        // LoadConst dst=500
        assert_eq!(words[2], (0x01u32 << 24) | 500);
        // Add dst=1000, src1=300, src2=500
        assert_eq!(words[4], (0x10u32 << 24) | 1000);
        assert_eq!(words[5], (300u32 << 16) | 500);
        // AddConstraint src=1000
        assert_eq!(words[6], (0x40u32 << 24) | 1000);
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

    #[test]
    fn test_encoded_len_matches_encode() {
        // Verify encoded_len() matches actual encode() word count for every op type.
        let ops: Vec<BytecodeOp> = vec![
            BytecodeOp::LoadConst { dst: 0, value: 42 },
            BytecodeOp::LoadSecureConst { dst: 1, value: [1, 2, 3, 4] },
            BytecodeOp::LoadTrace { dst: 2, interaction: 0, col_idx: 0, offset: 0 },
            BytecodeOp::LoadTrace { dst: 3, interaction: 1, col_idx: 2, offset: -3 },
            BytecodeOp::Add { dst: 4, src1: 0, src2: 1 },
            BytecodeOp::Sub { dst: 5, src1: 0, src2: 1 },
            BytecodeOp::Mul { dst: 6, src1: 0, src2: 1 },
            BytecodeOp::Neg { dst: 7, src: 0 },
            BytecodeOp::AddConst { dst: 8, src: 0, value: 7 },
            BytecodeOp::MulConst { dst: 9, src: 0, value: 3 },
            BytecodeOp::WideAdd { dst: 10, src1: 0, src2: 1 },
            BytecodeOp::WideSub { dst: 11, src1: 0, src2: 1 },
            BytecodeOp::WideMul { dst: 12, src1: 0, src2: 1 },
            BytecodeOp::WideNeg { dst: 13, src: 0 },
            BytecodeOp::WideAddConst { dst: 14, src: 0, value: [1, 2, 3, 4] },
            BytecodeOp::WideMulConst { dst: 15, src: 0, value: [1, 2, 3, 4] },
            BytecodeOp::WideAddBase { dst: 16, wide: 0, base: 1 },
            BytecodeOp::WideMulBase { dst: 17, wide: 0, base: 1 },
            BytecodeOp::BaseAddSecureConst { dst: 18, src: 0, value: [1, 2, 3, 4] },
            BytecodeOp::BaseMulSecureConst { dst: 19, src: 0, value: [1, 2, 3, 4] },
            BytecodeOp::Widen { dst: 20, src: 0 },
            BytecodeOp::CombineEF { dst: 21, src: [0, 1, 2, 3] },
            BytecodeOp::AddConstraint { src: 4 },
        ];
        for op in &ops {
            let prog = BytecodeProgram {
                ops: vec![op.clone()],
                n_constraints: if matches!(op, BytecodeOp::AddConstraint { .. }) { 1 } else { 0 },
                n_trace_accesses: if matches!(op, BytecodeOp::LoadTrace { .. }) { 1 } else { 0 },
                n_registers: 30,
            };
            let encoded = prog.encode();
            assert_eq!(
                encoded.len(),
                op.encoded_len(),
                "encoded_len mismatch for {:?}",
                op
            );
        }
    }
}
