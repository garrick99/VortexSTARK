//! A "tracing" implementation of `EvalAtRow` that records operations as bytecode.
//!
//! Instead of computing actual field values, `TracingEvalAtRow` captures the sequence
//! of operations into a [`BytecodeProgram`]. The resulting bytecode is identical for
//! every row of the trace — only the values loaded by `PushTraceVal` differ.
//!
//! # Design
//!
//! There is one "main" recorder created by `TracingEvalAtRow`. Values produced by
//! the evaluator (from `next_interaction_mask`, logup, etc.) hold a reference to
//! this main recorder and emit ops directly into it.
//!
//! Standalone values created by `From<BaseField>`, `Zero::zero()`, etc. hold a
//! "detached" recorder. When a detached value participates in arithmetic with a
//! main-recorder value, the detached recorder's ops are merged into the main
//! recorder first. This ensures the full computation is captured.
//!
//! `Clone` on a traced value copies the `Rc` reference, so cloned values share
//! the same recorder. This is correct because cloning doesn't create a new
//! computation — it just reuses the existing value.

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
    /// True if this is the main recorder attached to the evaluator.
    is_main: bool,
}

impl RecorderInner {
    fn new(is_main: bool) -> Self {
        Self {
            ops: Vec::new(),
            n_constraints: 0,
            n_trace_accesses: 0,
            is_main,
        }
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

/// Merge two recorders such that a detached recorder's ops flow into the main one.
/// If both are main (same Rc), this is a no-op.
/// If both are detached, merge src into dst.
/// If one is main, merge the detached one into the main one and return the main Rc.
fn merge_and_get_main(a: &Recorder, b: &Recorder) -> Recorder {
    if Rc::ptr_eq(a, b) {
        return a.clone();
    }

    let a_is_main = a.borrow().is_main;
    let b_is_main = b.borrow().is_main;

    match (a_is_main, b_is_main) {
        (true, false) => {
            // Merge b (detached) into a (main).
            let src = b.borrow();
            let mut dst = a.borrow_mut();
            for op in &src.ops {
                dst.ops.push(op.clone());
            }
            dst.n_trace_accesses += src.n_trace_accesses;
            drop(src);
            drop(dst);
            a.clone()
        }
        (false, true) => {
            // Merge a (detached) into b (main).
            let src = a.borrow();
            let mut dst = b.borrow_mut();
            for op in &src.ops {
                dst.ops.push(op.clone());
            }
            dst.n_trace_accesses += src.n_trace_accesses;
            drop(src);
            drop(dst);
            b.clone()
        }
        (false, false) => {
            // Both detached — merge b into a.
            let src = b.borrow();
            let mut dst = a.borrow_mut();
            for op in &src.ops {
                dst.ops.push(op.clone());
            }
            dst.n_trace_accesses += src.n_trace_accesses;
            drop(src);
            drop(dst);
            a.clone()
        }
        (true, true) => {
            // Both main but different Rc — shouldn't happen in normal use.
            // Merge b into a.
            let src = b.borrow();
            let mut dst = a.borrow_mut();
            for op in &src.ops {
                dst.ops.push(op.clone());
            }
            dst.n_trace_accesses += src.n_trace_accesses;
            dst.n_constraints += src.n_constraints;
            drop(src);
            drop(dst);
            a.clone()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  TracedBaseField — phantom type for EvalAtRow::F
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct TracedBaseField {
    rec: Recorder,
}

impl Debug for TracedBaseField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TracedBaseField")
    }
}

impl Zero for TracedBaseField {
    fn zero() -> Self {
        let rec = new_detached_recorder();
        rec.borrow_mut().emit(BytecodeOp::PushBaseField(0));
        Self { rec }
    }

    fn is_zero(&self) -> bool {
        false
    }
}

impl One for TracedBaseField {
    fn one() -> Self {
        let rec = new_detached_recorder();
        rec.borrow_mut().emit(BytecodeOp::PushBaseField(1));
        Self { rec }
    }
}

impl From<BaseField> for TracedBaseField {
    fn from(value: BaseField) -> Self {
        let rec = new_detached_recorder();
        rec.borrow_mut()
            .emit(BytecodeOp::PushBaseField(base_to_raw(value)));
        Self { rec }
    }
}

impl Neg for TracedBaseField {
    type Output = Self;
    fn neg(self) -> Self {
        self.rec.borrow_mut().emit(BytecodeOp::Neg);
        self
    }
}

impl Add for TracedBaseField {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let rec = merge_and_get_main(&self.rec, &rhs.rec);
        rec.borrow_mut().emit(BytecodeOp::Add);
        Self { rec }
    }
}

impl Sub for TracedBaseField {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let rec = merge_and_get_main(&self.rec, &rhs.rec);
        rec.borrow_mut().emit(BytecodeOp::Sub);
        Self { rec }
    }
}

impl Mul for TracedBaseField {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let rec = merge_and_get_main(&self.rec, &rhs.rec);
        rec.borrow_mut().emit(BytecodeOp::Mul);
        Self { rec }
    }
}

impl Mul<BaseField> for TracedBaseField {
    type Output = Self;
    fn mul(self, rhs: BaseField) -> Self {
        self.rec
            .borrow_mut()
            .emit(BytecodeOp::MulConst(base_to_raw(rhs)));
        self
    }
}

impl AddAssign for TracedBaseField {
    fn add_assign(&mut self, rhs: Self) {
        let rec = merge_and_get_main(&self.rec, &rhs.rec);
        rec.borrow_mut().emit(BytecodeOp::Add);
        self.rec = rec;
    }
}

impl AddAssign<BaseField> for TracedBaseField {
    fn add_assign(&mut self, rhs: BaseField) {
        self.rec
            .borrow_mut()
            .emit(BytecodeOp::AddConst(base_to_raw(rhs)));
    }
}

impl MulAssign for TracedBaseField {
    fn mul_assign(&mut self, rhs: Self) {
        let rec = merge_and_get_main(&self.rec, &rhs.rec);
        rec.borrow_mut().emit(BytecodeOp::Mul);
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
        self.rec
            .borrow_mut()
            .emit(BytecodeOp::BaseAddSecureConst(secure_to_raw(rhs)));
        TracedSecureField { rec: self.rec }
    }
}

impl Mul<SecureField> for TracedBaseField {
    type Output = TracedSecureField;
    fn mul(self, rhs: SecureField) -> TracedSecureField {
        self.rec
            .borrow_mut()
            .emit(BytecodeOp::BaseMulSecureConst(secure_to_raw(rhs)));
        TracedSecureField { rec: self.rec }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  TracedSecureField — phantom type for EvalAtRow::EF
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct TracedSecureField {
    rec: Recorder,
}

impl Debug for TracedSecureField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TracedSecureField")
    }
}

impl Zero for TracedSecureField {
    fn zero() -> Self {
        let rec = new_detached_recorder();
        rec.borrow_mut()
            .emit(BytecodeOp::PushSecureField([0, 0, 0, 0]));
        Self { rec }
    }

    fn is_zero(&self) -> bool {
        false
    }
}

impl One for TracedSecureField {
    fn one() -> Self {
        let rec = new_detached_recorder();
        rec.borrow_mut()
            .emit(BytecodeOp::PushSecureField([1, 0, 0, 0]));
        Self { rec }
    }
}

impl From<SecureField> for TracedSecureField {
    fn from(value: SecureField) -> Self {
        let rec = new_detached_recorder();
        rec.borrow_mut()
            .emit(BytecodeOp::PushSecureField(secure_to_raw(value)));
        Self { rec }
    }
}

impl From<TracedBaseField> for TracedSecureField {
    fn from(value: TracedBaseField) -> Self {
        value.rec.borrow_mut().emit(BytecodeOp::Widen);
        TracedSecureField { rec: value.rec }
    }
}

impl Neg for TracedSecureField {
    type Output = Self;
    fn neg(self) -> Self {
        self.rec.borrow_mut().emit(BytecodeOp::WideNeg);
        self
    }
}

impl Add for TracedSecureField {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let rec = merge_and_get_main(&self.rec, &rhs.rec);
        rec.borrow_mut().emit(BytecodeOp::WideAdd);
        Self { rec }
    }
}

impl Sub for TracedSecureField {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let rec = merge_and_get_main(&self.rec, &rhs.rec);
        rec.borrow_mut().emit(BytecodeOp::WideSub);
        Self { rec }
    }
}

impl Mul for TracedSecureField {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let rec = merge_and_get_main(&self.rec, &rhs.rec);
        rec.borrow_mut().emit(BytecodeOp::WideMul);
        Self { rec }
    }
}

impl AddAssign for TracedSecureField {
    fn add_assign(&mut self, rhs: Self) {
        let rec = merge_and_get_main(&self.rec, &rhs.rec);
        rec.borrow_mut().emit(BytecodeOp::WideAdd);
        self.rec = rec;
    }
}

impl MulAssign for TracedSecureField {
    fn mul_assign(&mut self, rhs: Self) {
        let rec = merge_and_get_main(&self.rec, &rhs.rec);
        rec.borrow_mut().emit(BytecodeOp::WideMul);
        self.rec = rec;
    }
}

impl Add<BaseField> for TracedSecureField {
    type Output = Self;
    fn add(self, rhs: BaseField) -> Self {
        self.rec
            .borrow_mut()
            .emit(BytecodeOp::WideAddConst([base_to_raw(rhs), 0, 0, 0]));
        self
    }
}

impl Mul<BaseField> for TracedSecureField {
    type Output = Self;
    fn mul(self, rhs: BaseField) -> Self {
        self.rec
            .borrow_mut()
            .emit(BytecodeOp::WideMulConst([base_to_raw(rhs), 0, 0, 0]));
        self
    }
}

impl Add<SecureField> for TracedSecureField {
    type Output = Self;
    fn add(self, rhs: SecureField) -> Self {
        self.rec
            .borrow_mut()
            .emit(BytecodeOp::WideAddConst(secure_to_raw(rhs)));
        self
    }
}

impl Sub<SecureField> for TracedSecureField {
    type Output = Self;
    fn sub(self, rhs: SecureField) -> Self {
        let neg = -rhs;
        self.rec
            .borrow_mut()
            .emit(BytecodeOp::WideAddConst(secure_to_raw(neg)));
        self
    }
}

impl Mul<SecureField> for TracedSecureField {
    type Output = Self;
    fn mul(self, rhs: SecureField) -> Self {
        self.rec
            .borrow_mut()
            .emit(BytecodeOp::WideMulConst(secure_to_raw(rhs)));
        self
    }
}

impl Add<TracedBaseField> for TracedSecureField {
    type Output = Self;
    fn add(self, rhs: TracedBaseField) -> Self {
        let rec = merge_and_get_main(&self.rec, &rhs.rec);
        rec.borrow_mut().emit(BytecodeOp::WideAddBase);
        Self { rec }
    }
}

impl Mul<TracedBaseField> for TracedSecureField {
    type Output = Self;
    fn mul(self, rhs: TracedBaseField) -> Self {
        let rec = merge_and_get_main(&self.rec, &rhs.rec);
        rec.borrow_mut().emit(BytecodeOp::WideMulBase);
        Self { rec }
    }
}

impl FieldExpOps for TracedSecureField {
    fn inverse(&self) -> Self {
        panic!("TracedSecureField::inverse() not supported in bytecode recording");
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  TracingEvalAtRow
// ═══════════════════════════════════════════════════════════════════════

/// An `EvalAtRow` implementation that records all operations as bytecode.
///
/// Usage:
/// ```ignore
/// let tracer = TracingEvalAtRow::new(claimed_sum, log_size, 3);
/// let tracer = component_eval.evaluate(tracer);
/// let program = tracer.into_program();
/// println!("{}", program.dump());
/// ```
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
            max_stack_depth: 0,
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

        {
            let mut inner = self.rec.borrow_mut();
            for i in 0..N {
                inner.emit(BytecodeOp::PushTraceVal {
                    interaction: interaction as u16,
                    col_idx: col_idx as u16,
                    offset: offsets[i] as i16,
                });
                inner.n_trace_accesses += 1;
            }
        }

        std::array::from_fn(|_| TracedBaseField {
            rec: self.rec.clone(),
        })
    }

    fn add_constraint<G>(&mut self, _constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF> + From<G>,
    {
        let mut inner = self.rec.borrow_mut();
        inner.emit(BytecodeOp::AddConstraint);
        inner.n_constraints += 1;
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        // All 4 values come from the main recorder.
        let rec = values[0].rec.clone();
        rec.borrow_mut().emit(BytecodeOp::CombineEF);
        TracedSecureField { rec }
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

        // prev_col_cumsum = QM31::zero — emit into main recorder directly.
        self.rec
            .borrow_mut()
            .emit(BytecodeOp::PushSecureField([0, 0, 0, 0]));
        let mut prev_col_cumsum = TracedSecureField {
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
        // Emit cumsum_shift into main recorder.
        self.rec.borrow_mut().emit(BytecodeOp::PushSecureField(
            secure_to_raw(self.logup_cumsum_shift),
        ));
        let shift = TracedSecureField {
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

        let has_push_base_1 = program
            .ops
            .iter()
            .any(|op| matches!(op, BytecodeOp::PushBaseField(1)));
        assert!(has_push_base_1, "Should have PushBaseField(1)");

        let has_sub = program.ops.iter().any(|op| matches!(op, BytecodeOp::Sub));
        assert!(has_sub, "Should have Sub for (1 - flag)");

        let has_mul = program.ops.iter().any(|op| matches!(op, BytecodeOp::Mul));
        assert!(has_mul, "Should have Mul for flag * (1-flag)");

        let has_add_constraint = program
            .ops
            .iter()
            .any(|op| matches!(op, BytecodeOp::AddConstraint));
        assert!(has_add_constraint, "Should have AddConstraint");
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

    // ── Multiple constraints ─────────────────────────────────────────────

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
            .filter(|op| matches!(op, BytecodeOp::AddConstraint))
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
            .any(|op| matches!(op, BytecodeOp::MulConst(3)));
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
            .filter(|op| matches!(op, BytecodeOp::PushTraceVal { .. }))
            .collect();
        assert_eq!(trace_ops.len(), 2);
        assert!(matches!(
            trace_ops[0],
            BytecodeOp::PushTraceVal { offset: 0, .. }
        ));
        assert!(matches!(
            trace_ops[1],
            BytecodeOp::PushTraceVal { offset: 1, .. }
        ));
    }
}
