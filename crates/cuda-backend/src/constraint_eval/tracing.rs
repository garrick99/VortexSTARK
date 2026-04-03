//! A "tracing" implementation of `EvalAtRow` that records operations as register-based bytecode.
//!
//! Instead of computing actual field values, `TracingEvalAtRow` captures the sequence
//! of operations into a [`BytecodeProgram`]. The resulting bytecode is identical for
//! every row of the trace — only the values loaded by `LoadTrace` differ.
//!
//! # Register-based Design
//!
//! Each traced value holds a **register index** (`u16`) that identifies where its
//! value lives in the virtual register file. Operations allocate a new destination
//! register and emit an instruction that reads from source registers.
//!
//! **Clone becomes free**: cloning a traced value just copies its register index.
//! Multiple consumers of a cloned value read from the same register. No bytecode
//! is emitted, no duplication occurs, and no stack imbalance is possible.
//!
//! This fixes the 32/67 stwo-cairo components that broke the stack-based VM.

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};
use std::rc::Rc;

use num_traits::{One, Zero};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::fields::FieldExpOps;
use stwo::core::Fraction;
use stwo_constraint_framework::{EvalAtRow, INTERACTION_TRACE_IDX};

use super::bytecode::{base_to_raw, secure_to_raw, BytecodeOp, BytecodeProgram};

// ═══════════════════════════════════════════════════════════════════════
//  Shared recording state
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug)]
struct RecorderInner {
    ops: Vec<BytecodeOp>,
    n_constraints: usize,
    n_trace_accesses: usize,
    next_reg: u16,
    /// True if this is the main recorder (attached to the evaluator).
    /// False for detached recorders (standalone constants).
    is_main: bool,
}

impl RecorderInner {
    fn new(is_main: bool) -> Self {
        Self {
            ops: Vec::new(),
            n_constraints: 0,
            n_trace_accesses: 0,
            next_reg: 0,
            is_main,
        }
    }

    fn alloc_reg(&mut self) -> u16 {
        let r = self.next_reg;
        self.next_reg += 1;
        r
    }

    fn emit(&mut self, op: BytecodeOp) {
        self.ops.push(op);
    }
}

type Recorder = Rc<RefCell<RecorderInner>>;

fn new_main_recorder() -> Recorder {
    Rc::new(RefCell::new(RecorderInner::new(true)))
}

fn new_detached_recorder() -> Recorder {
    Rc::new(RefCell::new(RecorderInner::new(false)))
}

/// Merge two recorders when a binary operation combines values from different recorders.
///
/// If both share the same Rc, return it (no merge needed).
/// Otherwise, merge the detached recorder into the main one. If both are detached,
/// merge b into a. If both are main (shouldn't happen normally), merge b into a.
///
/// The key invariant: the main recorder always absorbs the detached one, never
/// the other way around. This ensures the evaluator's instruction stream stays
/// in a single, coherent recorder.
fn merge_recorders(a: &Recorder, b: &Recorder) -> Recorder {
    if Rc::ptr_eq(a, b) {
        return a.clone();
    }

    let a_is_main = a.borrow().is_main;
    let b_is_main = b.borrow().is_main;

    match (a_is_main, b_is_main) {
        (true, false) => merge_into(a, b),
        (false, true) => merge_into(b, a),
        // Both detached or both main: merge b into a
        _ => merge_into(a, b),
    }
}

/// Merge `src` recorder's ops into `dst` recorder, remapping register indices.
/// Returns `dst`.
fn merge_into(dst: &Recorder, src: &Recorder) -> Recorder {
    let src_inner = src.borrow();
    if src_inner.ops.is_empty() {
        return dst.clone();
    }

    let mut dst_inner = dst.borrow_mut();
    let reg_offset = dst_inner.next_reg;

    // Remap all register indices in src ops by adding reg_offset.
    for op in &src_inner.ops {
        dst_inner.ops.push(remap_op(op, reg_offset));
    }
    dst_inner.next_reg += src_inner.next_reg;
    dst_inner.n_trace_accesses += src_inner.n_trace_accesses;

    drop(src_inner);
    drop(dst_inner);
    dst.clone()
}

/// Remap all register indices in an op by adding an offset.
fn remap_op(op: &BytecodeOp, offset: u16) -> BytecodeOp {
    match op {
        BytecodeOp::LoadConst { dst, value } => BytecodeOp::LoadConst {
            dst: dst + offset,
            value: *value,
        },
        BytecodeOp::LoadSecureConst { dst, value } => BytecodeOp::LoadSecureConst {
            dst: dst + offset,
            value: *value,
        },
        BytecodeOp::LoadTrace { dst, interaction, col_idx, offset: off } => BytecodeOp::LoadTrace {
            dst: dst + offset,
            interaction: *interaction,
            col_idx: *col_idx,
            offset: *off,
        },
        BytecodeOp::Add { dst, src1, src2 } => BytecodeOp::Add {
            dst: dst + offset,
            src1: src1 + offset,
            src2: src2 + offset,
        },
        BytecodeOp::Sub { dst, src1, src2 } => BytecodeOp::Sub {
            dst: dst + offset,
            src1: src1 + offset,
            src2: src2 + offset,
        },
        BytecodeOp::Mul { dst, src1, src2 } => BytecodeOp::Mul {
            dst: dst + offset,
            src1: src1 + offset,
            src2: src2 + offset,
        },
        BytecodeOp::Neg { dst, src } => BytecodeOp::Neg {
            dst: dst + offset,
            src: src + offset,
        },
        BytecodeOp::AddConst { dst, src, value } => BytecodeOp::AddConst {
            dst: dst + offset,
            src: src + offset,
            value: *value,
        },
        BytecodeOp::MulConst { dst, src, value } => BytecodeOp::MulConst {
            dst: dst + offset,
            src: src + offset,
            value: *value,
        },
        BytecodeOp::WideAdd { dst, src1, src2 } => BytecodeOp::WideAdd {
            dst: dst + offset,
            src1: src1 + offset,
            src2: src2 + offset,
        },
        BytecodeOp::WideSub { dst, src1, src2 } => BytecodeOp::WideSub {
            dst: dst + offset,
            src1: src1 + offset,
            src2: src2 + offset,
        },
        BytecodeOp::WideMul { dst, src1, src2 } => BytecodeOp::WideMul {
            dst: dst + offset,
            src1: src1 + offset,
            src2: src2 + offset,
        },
        BytecodeOp::WideNeg { dst, src } => BytecodeOp::WideNeg {
            dst: dst + offset,
            src: src + offset,
        },
        BytecodeOp::WideAddConst { dst, src, value } => BytecodeOp::WideAddConst {
            dst: dst + offset,
            src: src + offset,
            value: *value,
        },
        BytecodeOp::WideMulConst { dst, src, value } => BytecodeOp::WideMulConst {
            dst: dst + offset,
            src: src + offset,
            value: *value,
        },
        BytecodeOp::WideAddBase { dst, wide, base } => BytecodeOp::WideAddBase {
            dst: dst + offset,
            wide: wide + offset,
            base: base + offset,
        },
        BytecodeOp::WideMulBase { dst, wide, base } => BytecodeOp::WideMulBase {
            dst: dst + offset,
            wide: wide + offset,
            base: base + offset,
        },
        BytecodeOp::BaseAddSecureConst { dst, src, value } => BytecodeOp::BaseAddSecureConst {
            dst: dst + offset,
            src: src + offset,
            value: *value,
        },
        BytecodeOp::BaseMulSecureConst { dst, src, value } => BytecodeOp::BaseMulSecureConst {
            dst: dst + offset,
            src: src + offset,
            value: *value,
        },
        BytecodeOp::Widen { dst, src } => BytecodeOp::Widen {
            dst: dst + offset,
            src: src + offset,
        },
        BytecodeOp::CombineEF { dst, src } => BytecodeOp::CombineEF {
            dst: dst + offset,
            src: [src[0] + offset, src[1] + offset, src[2] + offset, src[3] + offset],
        },
        BytecodeOp::AddConstraint { src } => BytecodeOp::AddConstraint {
            src: src + offset,
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  TracedBaseField — holds a register index
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct TracedBaseField {
    /// Virtual register index holding this value.
    reg: u16,
    /// The shared recorder (for emitting new ops and allocating new regs).
    rec: Recorder,
}

impl Debug for TracedBaseField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TracedBaseField(r{})", self.reg)
    }
}

impl Zero for TracedBaseField {
    fn zero() -> Self {
        let rec = new_detached_recorder();
        let reg = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::LoadConst { dst: reg, value: 0 });
        Self { reg, rec }
    }

    fn is_zero(&self) -> bool {
        false
    }
}

impl One for TracedBaseField {
    fn one() -> Self {
        let rec = new_detached_recorder();
        let reg = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::LoadConst { dst: reg, value: 1 });
        Self { reg, rec }
    }
}

impl From<BaseField> for TracedBaseField {
    fn from(value: BaseField) -> Self {
        let rec = new_detached_recorder();
        let reg = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::LoadConst {
            dst: reg,
            value: base_to_raw(value),
        });
        Self { reg, rec }
    }
}

impl Neg for TracedBaseField {
    type Output = Self;
    fn neg(self) -> Self {
        let dst = self.rec.borrow_mut().alloc_reg();
        self.rec.borrow_mut().emit(BytecodeOp::Neg { dst, src: self.reg });
        Self { reg: dst, rec: self.rec }
    }
}

impl Add for TracedBaseField {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let rec = merge_recorders(&self.rec, &rhs.rec);
        // After merge, self.reg and rhs.reg might have been remapped if they were
        // in the "src" recorder. We need to figure out the correct register indices.
        // The trick: if self.rec was kept as dst, self.reg stays the same.
        // If rhs.rec was kept as dst, rhs.reg stays the same and self.reg is remapped.
        //
        // Actually, we need a different approach. The merge remaps the "src" recorder's
        // registers. We need to track which recorder was the source and remap accordingly.
        //
        // Simpler approach: since we always merge_into the bigger one, and the Rc
        // identity tells us which was kept, we can check Rc::ptr_eq.
        let (r1, r2) = resolve_regs_after_merge(&self.rec, self.reg, &rhs.rec, rhs.reg, &rec);
        let dst = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::Add { dst, src1: r1, src2: r2 });
        Self { reg: dst, rec }
    }
}

impl Sub for TracedBaseField {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let rec = merge_recorders(&self.rec, &rhs.rec);
        let (r1, r2) = resolve_regs_after_merge(&self.rec, self.reg, &rhs.rec, rhs.reg, &rec);
        let dst = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::Sub { dst, src1: r1, src2: r2 });
        Self { reg: dst, rec }
    }
}

impl Mul for TracedBaseField {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let rec = merge_recorders(&self.rec, &rhs.rec);
        let (r1, r2) = resolve_regs_after_merge(&self.rec, self.reg, &rhs.rec, rhs.reg, &rec);
        let dst = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::Mul { dst, src1: r1, src2: r2 });
        Self { reg: dst, rec }
    }
}

impl Mul<BaseField> for TracedBaseField {
    type Output = Self;
    fn mul(self, rhs: BaseField) -> Self {
        let dst = self.rec.borrow_mut().alloc_reg();
        self.rec.borrow_mut().emit(BytecodeOp::MulConst {
            dst,
            src: self.reg,
            value: base_to_raw(rhs),
        });
        Self { reg: dst, rec: self.rec }
    }
}

impl AddAssign for TracedBaseField {
    fn add_assign(&mut self, rhs: Self) {
        let rec = merge_recorders(&self.rec, &rhs.rec);
        let (r1, r2) = resolve_regs_after_merge(&self.rec, self.reg, &rhs.rec, rhs.reg, &rec);
        let dst = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::Add { dst, src1: r1, src2: r2 });
        self.reg = dst;
        self.rec = rec;
    }
}

impl AddAssign<BaseField> for TracedBaseField {
    fn add_assign(&mut self, rhs: BaseField) {
        let dst = self.rec.borrow_mut().alloc_reg();
        self.rec.borrow_mut().emit(BytecodeOp::AddConst {
            dst,
            src: self.reg,
            value: base_to_raw(rhs),
        });
        self.reg = dst;
    }
}

impl MulAssign for TracedBaseField {
    fn mul_assign(&mut self, rhs: Self) {
        let rec = merge_recorders(&self.rec, &rhs.rec);
        let (r1, r2) = resolve_regs_after_merge(&self.rec, self.reg, &rhs.rec, rhs.reg, &rec);
        let dst = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::Mul { dst, src1: r1, src2: r2 });
        self.reg = dst;
        self.rec = rec;
    }
}

impl FieldExpOps for TracedBaseField {
    fn inverse(&self) -> Self {
        panic!("TracedBaseField::inverse() not supported in bytecode recording");
    }
}

impl Add<SecureField> for TracedBaseField {
    type Output = TracedSecureField;
    fn add(self, rhs: SecureField) -> TracedSecureField {
        let dst = self.rec.borrow_mut().alloc_reg();
        self.rec.borrow_mut().emit(BytecodeOp::BaseAddSecureConst {
            dst,
            src: self.reg,
            value: secure_to_raw(rhs),
        });
        TracedSecureField { reg: dst, rec: self.rec }
    }
}

impl Mul<SecureField> for TracedBaseField {
    type Output = TracedSecureField;
    fn mul(self, rhs: SecureField) -> TracedSecureField {
        let dst = self.rec.borrow_mut().alloc_reg();
        self.rec.borrow_mut().emit(BytecodeOp::BaseMulSecureConst {
            dst,
            src: self.reg,
            value: secure_to_raw(rhs),
        });
        TracedSecureField { reg: dst, rec: self.rec }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  TracedSecureField — holds a register index for QM31 value
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct TracedSecureField {
    reg: u16,
    rec: Recorder,
}

impl Debug for TracedSecureField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TracedSecureField(r{})", self.reg)
    }
}

impl Zero for TracedSecureField {
    fn zero() -> Self {
        let rec = new_detached_recorder();
        let reg = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::LoadSecureConst {
            dst: reg,
            value: [0, 0, 0, 0],
        });
        Self { reg, rec }
    }

    fn is_zero(&self) -> bool {
        false
    }
}

impl One for TracedSecureField {
    fn one() -> Self {
        let rec = new_detached_recorder();
        let reg = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::LoadSecureConst {
            dst: reg,
            value: [1, 0, 0, 0],
        });
        Self { reg, rec }
    }
}

impl From<SecureField> for TracedSecureField {
    fn from(value: SecureField) -> Self {
        let rec = new_detached_recorder();
        let reg = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::LoadSecureConst {
            dst: reg,
            value: secure_to_raw(value),
        });
        Self { reg, rec }
    }
}

impl From<TracedBaseField> for TracedSecureField {
    fn from(value: TracedBaseField) -> Self {
        let dst = value.rec.borrow_mut().alloc_reg();
        value.rec.borrow_mut().emit(BytecodeOp::Widen {
            dst,
            src: value.reg,
        });
        TracedSecureField { reg: dst, rec: value.rec }
    }
}

impl Neg for TracedSecureField {
    type Output = Self;
    fn neg(self) -> Self {
        let dst = self.rec.borrow_mut().alloc_reg();
        self.rec.borrow_mut().emit(BytecodeOp::WideNeg { dst, src: self.reg });
        Self { reg: dst, rec: self.rec }
    }
}

impl Add for TracedSecureField {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let rec = merge_recorders(&self.rec, &rhs.rec);
        let (r1, r2) = resolve_regs_after_merge(&self.rec, self.reg, &rhs.rec, rhs.reg, &rec);
        let dst = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::WideAdd { dst, src1: r1, src2: r2 });
        Self { reg: dst, rec }
    }
}

impl Sub for TracedSecureField {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let rec = merge_recorders(&self.rec, &rhs.rec);
        let (r1, r2) = resolve_regs_after_merge(&self.rec, self.reg, &rhs.rec, rhs.reg, &rec);
        let dst = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::WideSub { dst, src1: r1, src2: r2 });
        Self { reg: dst, rec }
    }
}

impl Mul for TracedSecureField {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let rec = merge_recorders(&self.rec, &rhs.rec);
        let (r1, r2) = resolve_regs_after_merge(&self.rec, self.reg, &rhs.rec, rhs.reg, &rec);
        let dst = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::WideMul { dst, src1: r1, src2: r2 });
        Self { reg: dst, rec }
    }
}

impl AddAssign for TracedSecureField {
    fn add_assign(&mut self, rhs: Self) {
        let rec = merge_recorders(&self.rec, &rhs.rec);
        let (r1, r2) = resolve_regs_after_merge(&self.rec, self.reg, &rhs.rec, rhs.reg, &rec);
        let dst = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::WideAdd { dst, src1: r1, src2: r2 });
        self.reg = dst;
        self.rec = rec;
    }
}

impl MulAssign for TracedSecureField {
    fn mul_assign(&mut self, rhs: Self) {
        let rec = merge_recorders(&self.rec, &rhs.rec);
        let (r1, r2) = resolve_regs_after_merge(&self.rec, self.reg, &rhs.rec, rhs.reg, &rec);
        let dst = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::WideMul { dst, src1: r1, src2: r2 });
        self.reg = dst;
        self.rec = rec;
    }
}

impl Add<BaseField> for TracedSecureField {
    type Output = Self;
    fn add(self, rhs: BaseField) -> Self {
        let dst = self.rec.borrow_mut().alloc_reg();
        self.rec.borrow_mut().emit(BytecodeOp::WideAddConst {
            dst,
            src: self.reg,
            value: [base_to_raw(rhs), 0, 0, 0],
        });
        Self { reg: dst, rec: self.rec }
    }
}

impl Mul<BaseField> for TracedSecureField {
    type Output = Self;
    fn mul(self, rhs: BaseField) -> Self {
        let dst = self.rec.borrow_mut().alloc_reg();
        self.rec.borrow_mut().emit(BytecodeOp::WideMulConst {
            dst,
            src: self.reg,
            value: [base_to_raw(rhs), 0, 0, 0],
        });
        Self { reg: dst, rec: self.rec }
    }
}

impl Add<SecureField> for TracedSecureField {
    type Output = Self;
    fn add(self, rhs: SecureField) -> Self {
        let dst = self.rec.borrow_mut().alloc_reg();
        self.rec.borrow_mut().emit(BytecodeOp::WideAddConst {
            dst,
            src: self.reg,
            value: secure_to_raw(rhs),
        });
        Self { reg: dst, rec: self.rec }
    }
}

impl Sub<SecureField> for TracedSecureField {
    type Output = Self;
    fn sub(self, rhs: SecureField) -> Self {
        let neg = -rhs;
        let dst = self.rec.borrow_mut().alloc_reg();
        self.rec.borrow_mut().emit(BytecodeOp::WideAddConst {
            dst,
            src: self.reg,
            value: secure_to_raw(neg),
        });
        Self { reg: dst, rec: self.rec }
    }
}

impl Mul<SecureField> for TracedSecureField {
    type Output = Self;
    fn mul(self, rhs: SecureField) -> Self {
        let dst = self.rec.borrow_mut().alloc_reg();
        self.rec.borrow_mut().emit(BytecodeOp::WideMulConst {
            dst,
            src: self.reg,
            value: secure_to_raw(rhs),
        });
        Self { reg: dst, rec: self.rec }
    }
}

impl Add<TracedBaseField> for TracedSecureField {
    type Output = Self;
    fn add(self, rhs: TracedBaseField) -> Self {
        let rec = merge_recorders(&self.rec, &rhs.rec);
        let (r1, r2) = resolve_regs_after_merge(&self.rec, self.reg, &rhs.rec, rhs.reg, &rec);
        let dst = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::WideAddBase { dst, wide: r1, base: r2 });
        Self { reg: dst, rec }
    }
}

impl Mul<TracedBaseField> for TracedSecureField {
    type Output = Self;
    fn mul(self, rhs: TracedBaseField) -> Self {
        let rec = merge_recorders(&self.rec, &rhs.rec);
        let (r1, r2) = resolve_regs_after_merge(&self.rec, self.reg, &rhs.rec, rhs.reg, &rec);
        let dst = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::WideMulBase { dst, wide: r1, base: r2 });
        Self { reg: dst, rec }
    }
}

impl FieldExpOps for TracedSecureField {
    fn inverse(&self) -> Self {
        panic!("TracedSecureField::inverse() not supported in bytecode recording");
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Register resolution after merge
// ═══════════════════════════════════════════════════════════════════════

/// After merging two recorders, resolve the register indices for both operands.
///
/// If a_rec was the one kept (merged_rec points to same Rc), a_reg stays the same
/// and b_reg needs to be offset by the old next_reg of a_rec (before merge).
/// Vice versa if b_rec was kept.
///
/// If both point to the same Rc (already shared), both regs are unchanged.
fn resolve_regs_after_merge(
    a_rec: &Recorder,
    a_reg: u16,
    b_rec: &Recorder,
    b_reg: u16,
    merged_rec: &Recorder,
) -> (u16, u16) {
    if Rc::ptr_eq(a_rec, b_rec) {
        // Same recorder — no remapping needed.
        return (a_reg, b_reg);
    }

    if Rc::ptr_eq(a_rec, merged_rec) {
        // a was the destination, b was merged into a.
        // b's registers were remapped by adding a's old next_reg.
        // But we don't have a's old next_reg anymore... we need to compute it.
        //
        // After merge, merged_rec.next_reg = a_old_next_reg + b.next_reg
        // So a_old_next_reg = merged_rec.next_reg - b.next_reg
        //
        // But b_rec might have been consumed... Actually in our merge_into,
        // we don't modify src_inner. We just read it. So b_rec still has its
        // original state. But wait — b_rec is an Rc<RefCell>, and we only
        // borrow it immutably in merge_into.
        //
        // a_old_next_reg = merged_rec.borrow().next_reg - b_rec.borrow().next_reg
        let b_n_regs = b_rec.borrow().next_reg;
        let merged_next = merged_rec.borrow().next_reg;
        let a_old_next = merged_next - b_n_regs;
        return (a_reg, b_reg + a_old_next);
    }

    if Rc::ptr_eq(b_rec, merged_rec) {
        // b was the destination, a was merged into b.
        let a_n_regs = a_rec.borrow().next_reg;
        let merged_next = merged_rec.borrow().next_reg;
        let b_old_next = merged_next - a_n_regs;
        return (a_reg + b_old_next, b_reg);
    }

    // Neither matches — shouldn't happen with our merge logic.
    // Fallback: no remapping.
    (a_reg, b_reg)
}

// ═══════════════════════════════════════════════════════════════════════
//  TracingEvalAtRow
// ═══════════════════════════════════════════════════════════════════════

/// An `EvalAtRow` implementation that records all operations as register-based bytecode.
pub struct TracingEvalAtRow {
    rec: Recorder,
    column_index_per_interaction: Vec<usize>,
    logup_interaction: usize,
    logup_cumsum_shift: SecureField,
    logup_fracs: Vec<Fraction<TracedSecureField, TracedSecureField>>,
    logup_is_finalized: bool,
}

impl TracingEvalAtRow {
    pub fn new(claimed_sum: SecureField, log_size: u32, n_interactions: usize) -> Self {
        let rec = new_main_recorder();
        let cumsum_shift = claimed_sum / BaseField::from_u32_unchecked(1 << log_size);
        Self {
            rec,
            column_index_per_interaction: vec![0; n_interactions],
            logup_interaction: INTERACTION_TRACE_IDX,
            logup_cumsum_shift: cumsum_shift,
            logup_fracs: Vec::new(),
            logup_is_finalized: true,
        }
    }

    pub fn into_program(self) -> BytecodeProgram {
        let inner = self.rec.borrow();
        BytecodeProgram {
            ops: inner.ops.clone(),
            n_constraints: inner.n_constraints,
            n_trace_accesses: inner.n_trace_accesses,
            n_registers: inner.next_reg,
        }
    }
}

impl EvalAtRow for TracingEvalAtRow {
    type F = TracedBaseField;
    type EF = TracedSecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        if self.column_index_per_interaction.len() <= interaction {
            self.column_index_per_interaction.resize(interaction + 1, 0);
        }

        let col_idx = self.column_index_per_interaction[interaction];
        self.column_index_per_interaction[interaction] += 1;

        let mut regs = [0u16; N];
        {
            let mut inner = self.rec.borrow_mut();
            for i in 0..N {
                let dst = inner.alloc_reg();
                inner.emit(BytecodeOp::LoadTrace {
                    dst,
                    interaction: interaction as u16,
                    col_idx: col_idx as u16,
                    offset: offsets[i] as i16,
                });
                inner.n_trace_accesses += 1;
                regs[i] = dst;
            }
        }

        std::array::from_fn(|i| TracedBaseField {
            reg: regs[i],
            rec: self.rec.clone(),
        })
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF> + From<G>,
    {
        // Convert constraint to TracedSecureField to get its register index.
        // G implements Into<TracedSecureField> (via From<G> for Self::EF).
        let traced: TracedSecureField = constraint.into();
        // Merge the constraint's recorder into ours if from a detached source.
        // If traced.rec is detached and gets merged into self.rec, the registers
        // from traced.rec are remapped by an offset. We must apply that same
        // offset to traced.reg.
        let merged = merge_recorders(&self.rec, &traced.rec);
        let src = if Rc::ptr_eq(&self.rec, &traced.rec) {
            // Same recorder — no remapping needed.
            traced.reg
        } else if Rc::ptr_eq(&self.rec, &merged) {
            // traced.rec was merged into self.rec.
            // traced's registers were offset by self.rec's old next_reg.
            let traced_n_regs = traced.rec.borrow().next_reg;
            let merged_next = merged.borrow().next_reg;
            let self_old_next = merged_next - traced_n_regs;
            traced.reg + self_old_next
        } else {
            // self.rec was merged into traced.rec (shouldn't normally happen
            // since self.rec is the main recorder, but handle it for safety).
            traced.reg
        };
        let mut inner = self.rec.borrow_mut();
        inner.emit(BytecodeOp::AddConstraint { src });
        inner.n_constraints += 1;
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        let rec = values[0].rec.clone();
        let dst = rec.borrow_mut().alloc_reg();
        rec.borrow_mut().emit(BytecodeOp::CombineEF {
            dst,
            src: [values[0].reg, values[1].reg, values[2].reg, values[3].reg],
        });
        TracedSecureField { reg: dst, rec }
    }

    fn write_logup_frac(&mut self, fraction: Fraction<Self::EF, Self::EF>) {
        if self.logup_fracs.is_empty() {
            self.logup_is_finalized = false;
        }
        self.logup_fracs.push(fraction);
    }

    fn finalize_logup_batched(&mut self, batching: &Vec<usize>) {
        assert!(!self.logup_is_finalized, "LogupAtRow was already finalized");
        assert_eq!(
            batching.len(),
            self.logup_fracs.len(),
            "Batching must be same length as number of entries"
        );

        let last_batch = *batching.iter().max().unwrap();

        let mut fracs_by_batch: std::collections::HashMap<
            usize,
            Vec<Fraction<Self::EF, Self::EF>>,
        > = std::collections::HashMap::new();
        for (batch, frac) in batching.iter().zip(self.logup_fracs.drain(..)) {
            fracs_by_batch.entry(*batch).or_default().push(frac);
        }

        // prev_col_cumsum = QM31::zero
        let zero_reg = self.rec.borrow_mut().alloc_reg();
        self.rec.borrow_mut().emit(BytecodeOp::LoadSecureConst {
            dst: zero_reg,
            value: [0, 0, 0, 0],
        });
        let mut prev_col_cumsum = TracedSecureField {
            reg: zero_reg,
            rec: self.rec.clone(),
        };

        for batch_id in 0..last_batch {
            let cur_frac = sum_fractions(fracs_by_batch.remove(&batch_id).unwrap());
            let [cur_cumsum] =
                self.next_extension_interaction_mask(self.logup_interaction, [0]);
            let diff = cur_cumsum.clone() - prev_col_cumsum.clone();
            prev_col_cumsum = cur_cumsum;
            self.add_constraint(diff * cur_frac.denominator - cur_frac.numerator);
        }

        let frac = sum_fractions(fracs_by_batch.remove(&last_batch).unwrap());
        let [prev_row_cumsum, cur_cumsum] =
            self.next_extension_interaction_mask(self.logup_interaction, [-1, 0]);

        let diff = cur_cumsum - prev_row_cumsum - prev_col_cumsum;
        // Emit cumsum_shift
        let shift_reg = self.rec.borrow_mut().alloc_reg();
        self.rec.borrow_mut().emit(BytecodeOp::LoadSecureConst {
            dst: shift_reg,
            value: secure_to_raw(self.logup_cumsum_shift),
        });
        let shift = TracedSecureField {
            reg: shift_reg,
            rec: self.rec.clone(),
        };
        let shifted_diff = diff + shift;

        self.add_constraint(shifted_diff * frac.denominator - frac.numerator);

        self.logup_is_finalized = true;
    }

    fn finalize_logup(&mut self) {
        let batches: Vec<usize> = (0..self.logup_fracs.len()).collect();
        self.finalize_logup_batched(&batches);
    }

    fn finalize_logup_in_pairs(&mut self) {
        let batches: Vec<usize> = (0..self.logup_fracs.len()).map(|n| n / 2).collect();
        self.finalize_logup_batched(&batches);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════

fn sum_fractions(
    fracs: Vec<Fraction<TracedSecureField, TracedSecureField>>,
) -> Fraction<TracedSecureField, TracedSecureField> {
    assert!(!fracs.is_empty());
    let mut iter = fracs.into_iter();
    let first = iter.next().unwrap();
    iter.fold(first, |acc, frac| {
        let new_num = acc.numerator.clone() * frac.denominator.clone()
            + frac.numerator.clone() * acc.denominator.clone();
        let new_den = acc.denominator * frac.denominator;
        Fraction::new(new_num, new_den)
    })
}

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Record the bytecode for a `FrameworkEval` component.
pub fn record_bytecode<E: stwo_constraint_framework::FrameworkEval>(
    eval: &E,
    claimed_sum: SecureField,
) -> BytecodeProgram {
    let log_size = eval.log_size();
    let n_interactions = 3;
    let tracer = TracingEvalAtRow::new(claimed_sum, log_size, n_interactions);
    let tracer = eval.evaluate(tracer);
    tracer.into_program()
}

#[cfg(test)]
mod tests {
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo_constraint_framework::{EvalAtRow, FrameworkEval};

    use super::*;
    use crate::constraint_eval::bytecode::BytecodeOp;

    // ── flag * (1 - flag) = 0 ────────────────────────────────────────────

    struct FlagConstraintEval;

    impl FrameworkEval for FlagConstraintEval {
        fn log_size(&self) -> u32 {
            8
        }
        fn max_constraint_log_degree_bound(&self) -> u32 {
            9
        }
        fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
            let flag = eval.next_trace_mask();
            let one = E::F::from(BaseField::from_u32_unchecked(1));
            let constraint = flag.clone() * (one - flag);
            eval.add_constraint(constraint);
            eval
        }
    }

    #[test]
    fn test_flag_constraint_bytecode() {
        let program = record_bytecode(&FlagConstraintEval, SecureField::default());
        println!("{}", program.dump());

        assert_eq!(program.n_constraints, 1);
        assert_eq!(program.n_trace_accesses, 1);

        // Should have LoadTrace, LoadConst(1), Sub, Mul, AddConstraint
        let has_load_trace = program
            .ops
            .iter()
            .any(|op| matches!(op, BytecodeOp::LoadTrace { .. }));
        assert!(has_load_trace, "Should have LoadTrace");

        let has_load_const_1 = program
            .ops
            .iter()
            .any(|op| matches!(op, BytecodeOp::LoadConst { value: 1, .. }));
        assert!(has_load_const_1, "Should have LoadConst(1)");

        let has_sub = program.ops.iter().any(|op| matches!(op, BytecodeOp::Sub { .. }));
        assert!(has_sub, "Should have Sub for (1 - flag)");

        let has_mul = program.ops.iter().any(|op| matches!(op, BytecodeOp::Mul { .. }));
        assert!(has_mul, "Should have Mul for flag * (1-flag)");

        let has_add_constraint = program
            .ops
            .iter()
            .any(|op| matches!(op, BytecodeOp::AddConstraint { .. }));
        assert!(has_add_constraint, "Should have AddConstraint");
    }

    // ── Clone correctness: flag used twice should work ───────────────────

    #[test]
    fn test_clone_produces_no_extra_ops() {
        let program = record_bytecode(&FlagConstraintEval, SecureField::default());
        println!("{}", program.dump());

        // In the register VM, flag.clone() just copies the register index.
        // There should be exactly 1 LoadTrace (not 2).
        let load_trace_count = program
            .ops
            .iter()
            .filter(|op| matches!(op, BytecodeOp::LoadTrace { .. }))
            .count();
        assert_eq!(load_trace_count, 1, "Clone should NOT emit extra LoadTrace ops");

        // The Sub and Mul should reference the same source register (the flag's register)
        // as one of their operands.
    }

    // ── a + b - c = 0 ───────────────────────────────────────────────────

    struct LinearCombinationEval;

    impl FrameworkEval for LinearCombinationEval {
        fn log_size(&self) -> u32 {
            10
        }
        fn max_constraint_log_degree_bound(&self) -> u32 {
            11
        }
        fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
            let a = eval.next_trace_mask();
            let b = eval.next_trace_mask();
            let c = eval.next_trace_mask();
            eval.add_constraint(a + b - c);
            eval
        }
    }

    #[test]
    fn test_linear_combination_bytecode() {
        let program = record_bytecode(&LinearCombinationEval, SecureField::default());
        println!("{}", program.dump());

        assert_eq!(program.n_constraints, 1);
        assert_eq!(program.n_trace_accesses, 3);
    }

    // ── Multiple constraints with Clone ──────────────────────────────────

    struct MultiConstraintEval;

    impl FrameworkEval for MultiConstraintEval {
        fn log_size(&self) -> u32 {
            8
        }
        fn max_constraint_log_degree_bound(&self) -> u32 {
            10
        }
        fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
            let flag = eval.next_trace_mask();
            let a = eval.next_trace_mask();
            let b = eval.next_trace_mask();
            let c = eval.next_trace_mask();

            let one = E::F::from(BaseField::from_u32_unchecked(1));

            // flag is cloned twice here — this broke the stack VM!
            eval.add_constraint(flag.clone() * (one - flag.clone()));
            eval.add_constraint(a + b * flag - c);

            eval
        }
    }

    #[test]
    fn test_multi_constraint_bytecode() {
        let program = record_bytecode(&MultiConstraintEval, SecureField::default());
        println!("{}", program.dump());

        assert_eq!(program.n_constraints, 2);
        assert_eq!(program.n_trace_accesses, 4);

        let add_constraint_count = program
            .ops
            .iter()
            .filter(|op| matches!(op, BytecodeOp::AddConstraint { .. }))
            .count();
        assert_eq!(add_constraint_count, 2);
    }

    // ── Constant multiplication ──────────────────────────────────────────

    struct ConstMulEval;

    impl FrameworkEval for ConstMulEval {
        fn log_size(&self) -> u32 {
            8
        }
        fn max_constraint_log_degree_bound(&self) -> u32 {
            9
        }
        fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
            let x = eval.next_trace_mask();
            let y = eval.next_trace_mask();
            let three = BaseField::from_u32_unchecked(3);
            eval.add_constraint(x * three - y);
            eval
        }
    }

    #[test]
    fn test_const_mul_bytecode() {
        let program = record_bytecode(&ConstMulEval, SecureField::default());
        println!("{}", program.dump());

        assert_eq!(program.n_constraints, 1);
        let has_mul_const = program
            .ops
            .iter()
            .any(|op| matches!(op, BytecodeOp::MulConst { value: 3, .. }));
        assert!(has_mul_const, "Should contain MulConst(3)");
    }

    // ── Determinism ──────────────────────────────────────────────────────

    #[test]
    fn test_bytecode_deterministic() {
        let p1 = record_bytecode(&FlagConstraintEval, SecureField::default());
        let p2 = record_bytecode(&FlagConstraintEval, SecureField::default());
        assert_eq!(p1.ops, p2.ops);
        assert_eq!(p1.n_constraints, p2.n_constraints);
        assert_eq!(p1.n_trace_accesses, p2.n_trace_accesses);
    }

    // ── Next-row access ──────────────────────────────────────────────────

    struct NextRowEval;

    impl FrameworkEval for NextRowEval {
        fn log_size(&self) -> u32 {
            8
        }
        fn max_constraint_log_degree_bound(&self) -> u32 {
            9
        }
        fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
            let [curr, next] = eval.next_interaction_mask(1, [0, 1]);
            let one = E::F::from(BaseField::from_u32_unchecked(1));
            eval.add_constraint(next - curr - one);
            eval
        }
    }

    #[test]
    fn test_next_row_access() {
        let program = record_bytecode(&NextRowEval, SecureField::default());
        println!("{}", program.dump());

        let trace_ops: Vec<_> = program
            .ops
            .iter()
            .filter(|op| matches!(op, BytecodeOp::LoadTrace { .. }))
            .collect();
        assert_eq!(trace_ops.len(), 2);
        assert!(matches!(
            trace_ops[0],
            BytecodeOp::LoadTrace { offset: 0, .. }
        ));
        assert!(matches!(
            trace_ops[1],
            BytecodeOp::LoadTrace { offset: 1, .. }
        ));
    }
}
