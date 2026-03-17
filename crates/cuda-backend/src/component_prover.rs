//! ComponentProver<CudaBackend> for stwo's constraint framework.
//!
//! Uses a newtype wrapper `CudaFrameworkComponent<E>` to satisfy Rust's orphan rule.
//!
//! The primary path records constraint evaluation as bytecode, then runs the
//! bytecode interpreter kernel on the GPU. If bytecode recording fails for any
//! reason (unsupported operation, panic in the tracer, etc.), it falls back to
//! the CPU path: download trace, evaluate on CPU, upload results.
//!
//! Profiling: Each phase has a tracing span so `prove --trace` output shows
//! exactly where time is spent.

use std::borrow::Cow;
use std::ops::Deref;

use itertools::Itertools;
use stwo::core::air::accumulation::PointEvaluationAccumulator;
use stwo::core::air::Component;
use stwo::core::circle::CirclePoint;
use stwo::core::constraints::coset_vanishing;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse;
use stwo::core::ColumnVec;
use stwo::prover::backend::{Column, CpuBackend};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::{ComponentProver, DomainEvaluationAccumulator, EvaluationMode, Poly, Trace};
use stwo_constraint_framework::{
    CpuDomainEvaluator, FrameworkComponent, FrameworkEval, PREPROCESSED_TRACE_IDX,
};
use tracing::{span, Level};
use vortexstark::cuda::ffi;
use vortexstark::device::DeviceBuffer;

use super::CudaBackend;
use super::constraint_eval::bytecode::BytecodeOp;
use super::constraint_eval::tracing::record_bytecode;

/// A wrapper around `FrameworkComponent<E>` that implements `ComponentProver<CudaBackend>`.
///
/// This newtype is necessary because Rust's orphan rule prevents implementing
/// a foreign trait (`ComponentProver`) for a foreign type (`FrameworkComponent`)
/// even when the type parameter (`CudaBackend`) is local.
///
/// Usage:
/// ```ignore
/// let component = FrameworkComponent::new(&mut allocator, eval, claimed_sum);
/// let cuda_component = CudaFrameworkComponent(component);
/// // cuda_component implements ComponentProver<CudaBackend>
/// ```
pub struct CudaFrameworkComponent<E: FrameworkEval>(pub FrameworkComponent<E>);

/// A borrowing wrapper around `&FrameworkComponent<E>` that implements
/// `ComponentProver<CudaBackend>`.
///
/// Use this when you have a reference to a `FrameworkComponent` (e.g. borrowed
/// from a larger struct) and need to pass it as a `&dyn ComponentProver<CudaBackend>`.
pub struct CudaFrameworkComponentRef<'a, E: FrameworkEval>(pub &'a FrameworkComponent<E>);

impl<E: FrameworkEval> Deref for CudaFrameworkComponent<E> {
    type Target = FrameworkComponent<E>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, E: FrameworkEval> Deref for CudaFrameworkComponentRef<'a, E> {
    type Target = FrameworkComponent<E>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<E: FrameworkEval> Component for CudaFrameworkComponent<E> {
    fn n_constraints(&self) -> usize {
        self.0.n_constraints()
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.0.max_constraint_log_degree_bound()
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        self.0.trace_log_degree_bounds()
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
        max_log_degree_bound: u32,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        self.0.mask_points(point, max_log_degree_bound)
    }

    fn preprocessed_column_indices(&self) -> ColumnVec<usize> {
        self.0.preprocessed_column_indices().to_vec()
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        max_log_degree_bound: u32,
    ) {
        self.0.evaluate_constraint_quotients_at_point(
            point,
            mask,
            evaluation_accumulator,
            max_log_degree_bound,
        );
    }
}

impl<'a, E: FrameworkEval> Component for CudaFrameworkComponentRef<'a, E> {
    fn n_constraints(&self) -> usize {
        self.0.n_constraints()
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.0.max_constraint_log_degree_bound()
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        self.0.trace_log_degree_bounds()
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
        max_log_degree_bound: u32,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        self.0.mask_points(point, max_log_degree_bound)
    }

    fn preprocessed_column_indices(&self) -> ColumnVec<usize> {
        self.0.preprocessed_column_indices().to_vec()
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        max_log_degree_bound: u32,
    ) {
        self.0.evaluate_constraint_quotients_at_point(
            point,
            mask,
            evaluation_accumulator,
            max_log_degree_bound,
        );
    }
}

/// Shared implementation for both CudaFrameworkComponent and CudaFrameworkComponentRef.
///
/// Strategy:
/// 1. Try to record bytecode from the FrameworkEval.
/// 2. If successful, run the GPU bytecode interpreter kernel.
/// 3. If recording fails (panic, unsupported op), fall back to CPU evaluation.
fn evaluate_constraint_quotients_impl<E: FrameworkEval + Sync>(
    component: &FrameworkComponent<E>,
    trace: &Trace<'_, CudaBackend>,
    evaluation_accumulator: &mut DomainEvaluationAccumulator<CudaBackend>,
) {
    let n_constraints = component.n_constraints();
    if n_constraints == 0 {
        return;
    }

    if component.is_disabled() {
        evaluation_accumulator.skip_coeffs(n_constraints);
        return;
    }

    let eval_domain =
        CanonicCoset::new(component.max_constraint_log_degree_bound()).circle_domain();
    let trace_domain = CanonicCoset::new(component.log_size());

    // Extract this component's polynomial columns from the full trace.
    let mut component_polys = trace.polys.sub_tree(component.trace_locations());
    component_polys[PREPROCESSED_TRACE_IDX] = component
        .preprocessed_column_indices()
        .iter()
        .map(|idx| &trace.polys[PREPROCESSED_TRACE_IDX][*idx])
        .collect();

    // Phase 1: GPU polynomial extension to eval domain.
    let gpu_trace_evals = get_constraint_quotients_inputs_cuda(
        eval_domain,
        component_polys,
        evaluation_accumulator.evaluation_mode(),
    );

    // Denom inverses (CPU computation, small -- O(blowup_factor) elements).
    let log_expand = eval_domain.log_size() - trace_domain.log_size();
    let mut denom_inv = (0..1 << log_expand)
        .map(|i| coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse())
        .collect_vec();
    bit_reverse(&mut denom_inv);

    // Get accumulator column (GPU-resident).
    let [mut accum] =
        evaluation_accumulator.columns([(eval_domain.log_size(), n_constraints)]);
    accum.random_coeff_powers.reverse();

    // GPU bytecode path — disabled pending integration debugging.
    // Unit tests pass but real stwo-cairo components cause illegal memory access.
    // TODO: Debug operand encoding / trace column remapping for 60+ component types.
    let gpu_success = false && try_gpu_bytecode_eval(
        component,
        &gpu_trace_evals,
        eval_domain.log_size(),
        trace_domain.log_size(),
        log_expand,
        &denom_inv,
        &accum.random_coeff_powers,
        &mut accum.col,
    );

    if gpu_success {
        return;
    }

    // ── CPU fallback path ──────────────────────────────────────────────
    tracing::warn!("GPU bytecode eval failed, falling back to CPU");

    // Phase 2: Download all GPU trace evaluations to CPU.
    let cpu_trace_evals: TreeVec<Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>;
    {
        let _span = span!(
            Level::INFO,
            "GPU→CPU trace download (fallback)",
            log_size = eval_domain.log_size(),
        )
        .entered();
        cpu_trace_evals = gpu_trace_evals.map_cols(|gpu_eval| {
            let cpu_values = gpu_eval.values.to_cpu();
            CircleEvaluation::new(gpu_eval.domain, cpu_values)
        });
    }
    let cpu_trace_refs = cpu_trace_evals.as_cols_ref();

    // Download current accumulator state to CPU.
    let cpu_accum;
    {
        let _span = span!(Level::INFO, "GPU→CPU accum download (fallback)").entered();
        cpu_accum = accum.col.to_cpu();
    }

    // Phase 3: Run point-wise constraint evaluation on CPU.
    let cpu_result;
    {
        let _span = span!(
            Level::INFO,
            "Constraint eval (CPU serial fallback)",
            log_eval_size = eval_domain.log_size(),
            log_trace_size = trace_domain.log_size(),
            n_constraints = n_constraints,
        )
        .entered();
        cpu_result = accumulate_pointwise_cpu(
            component,
            cpu_trace_refs,
            eval_domain.log_size(),
            trace_domain.log_size(),
            denom_inv,
            &accum.random_coeff_powers,
            &cpu_accum,
        );
    }

    // Phase 4: Upload result back to GPU.
    {
        let _span = span!(Level::INFO, "CPU→GPU result upload (fallback)").entered();
        *accum.col = SecureColumnByCoords {
            columns: std::array::from_fn(|i| {
                cpu_result.columns[i].iter().copied().collect()
            }),
        };
    }
}

/// Try to evaluate constraints on GPU using the bytecode interpreter.
///
/// Returns `true` if successful, `false` if recording failed and caller
/// should fall back to CPU.
fn try_gpu_bytecode_eval<E: FrameworkEval>(
    component: &FrameworkComponent<E>,
    gpu_trace_evals: &TreeVec<Vec<Cow<'_, CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>>>,
    eval_log_size: u32,
    trace_log_size: u32,
    log_expand: u32,
    denom_inv: &[BaseField],
    random_coeff_powers: &[SecureField],
    accum_col: &mut SecureColumnByCoords<CudaBackend>,
) -> bool {
    // Phase A: Record bytecode.
    let program = {
        let _span = span!(Level::INFO, "Bytecode recording").entered();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            record_bytecode(component.deref(), component.claimed_sum())
        }));
        match result {
            Ok(prog) => prog,
            Err(e) => {
                tracing::warn!("Bytecode recording panicked: {:?}", e.downcast_ref::<&str>());
                return false;
            }
        }
    };

    if program.ops.is_empty() || program.n_constraints == 0 {
        // Empty program — nothing to do (constraints are trivially satisfied).
        return true;
    }

    let _span = span!(
        Level::INFO,
        "GPU bytecode constraint eval",
        n_ops = program.ops.len(),
        n_constraints = program.n_constraints,
        eval_log_size = eval_log_size,
        trace_log_size = trace_log_size,
    )
    .entered();

    // Phase B: Build flat column pointer array and remap bytecode.
    //
    // The bytecode PushTraceVal ops encode (interaction, col_idx, offset).
    // We need to remap them to flat column indices matching the order of
    // columns in gpu_trace_evals.
    //
    // Build a mapping: (interaction, col_idx) -> flat_index, then patch
    // the encoded bytecode.

    // Count columns per interaction and build the cumulative offset.
    let n_interactions = gpu_trace_evals.len();
    let mut interaction_offsets = Vec::with_capacity(n_interactions + 1);
    interaction_offsets.push(0usize);
    for interaction_cols in gpu_trace_evals.iter() {
        interaction_offsets.push(interaction_offsets.last().unwrap() + interaction_cols.len());
    }
    let total_cols = *interaction_offsets.last().unwrap();

    // Collect device pointers for all columns.
    let mut col_ptrs: Vec<*const u32> = Vec::with_capacity(total_cols);
    for interaction_cols in gpu_trace_evals.iter() {
        for eval in interaction_cols {
            col_ptrs.push(eval.values.device_ptr());
        }
    }

    // Encode bytecode and remap PushTraceVal to use flat column indices.
    let mut encoded = program.encode();

    // Walk through the encoded bytecode and patch PushTraceVal operands.
    // We need to match the encoding format from BytecodeProgram::encode():
    //   PushTraceVal: single word with (interaction:4 | col_idx:10 | sign:1 | abs_offset:9)
    // After remapping: (flat_col:14 | sign:1 | abs_offset:9)
    {
        let mut pc = 0;
        let mut op_idx = 0;
        while op_idx < program.ops.len() {
            let op = &program.ops[op_idx];
            match op {
                BytecodeOp::PushTraceVal { interaction, col_idx, offset } => {
                    let flat_idx = interaction_offsets[*interaction as usize] + *col_idx as usize;
                    if flat_idx >= total_cols {
                        tracing::warn!(
                            "PushTraceVal flat_idx {} >= total_cols {} (interaction={}, col_idx={})",
                            flat_idx, total_cols, interaction, col_idx
                        );
                        return false;
                    }
                    // Rebuild the word with flat index
                    let (sign, abs_off) = if *offset < 0 {
                        (1u32, (-*offset) as u32)
                    } else {
                        (0u32, *offset as u32)
                    };
                    let operand = ((flat_idx as u32) << 10) | (sign << 9) | (abs_off & 0x1FF);
                    encoded[pc] = (0x03u32 << 24) | operand;
                    pc += 1;
                }
                BytecodeOp::PushBaseField(v) => {
                    pc += if *v >= (1 << 24) { 2 } else { 1 };
                }
                BytecodeOp::PushSecureField(_) => {
                    pc += 5; // header + 4 data words
                }
                BytecodeOp::AddConst(v) | BytecodeOp::MulConst(v) => {
                    pc += if *v >= (1 << 24) { 2 } else { 1 };
                }
                BytecodeOp::WideAddConst(_) | BytecodeOp::WideMulConst(_)
                | BytecodeOp::BaseAddSecureConst(_) | BytecodeOp::BaseMulSecureConst(_) => {
                    pc += 5; // header + 4 data words
                }
                // All other ops are single-word, no patching needed
                _ => {
                    pc += 1;
                }
            }
            op_idx += 1;
        }
    }

    let n_rows = 1u32 << eval_log_size;
    let trace_n_rows = 1u32 << trace_log_size;

    // Phase C: Upload data to GPU.

    // Upload encoded bytecode.
    let d_bytecode = DeviceBuffer::from_host(&encoded);

    // Upload column pointer array (array of device pointers on device).
    let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

    // Column sizes (all the same for this domain).
    let col_sizes: Vec<u32> = vec![n_rows; total_cols];
    let d_col_sizes = DeviceBuffer::from_host(&col_sizes);

    // Upload random_coeff_powers as flat [n_constraints * 4] u32 array.
    let mut coeff_flat: Vec<u32> = Vec::with_capacity(random_coeff_powers.len() * 4);
    for coeff in random_coeff_powers {
        let arr = coeff.to_m31_array();
        coeff_flat.push(arr[0].0);
        coeff_flat.push(arr[1].0);
        coeff_flat.push(arr[2].0);
        coeff_flat.push(arr[3].0);
    }
    let d_coeff = DeviceBuffer::from_host(&coeff_flat);

    // Upload denom_inv.
    let denom_inv_raw: Vec<u32> = denom_inv.iter().map(|v| v.0).collect();
    let d_denom_inv = DeviceBuffer::from_host(&denom_inv_raw);

    // Phase D: Launch kernel.
    {
        let _span = span!(Level::INFO, "CUDA bytecode kernel launch", n_rows = n_rows).entered();
        unsafe {
            ffi::cuda_bytecode_constraint_eval(
                d_bytecode.as_ptr(),
                encoded.len() as u32,
                d_col_ptrs.as_ptr() as *const *const u32,
                d_col_sizes.as_ptr(),
                total_cols as u32,
                n_rows,
                trace_n_rows,
                d_coeff.as_ptr(),
                d_denom_inv.as_ptr(),
                log_expand,
                accum_col.columns[0].buf.as_mut_ptr(),
                accum_col.columns[1].buf.as_mut_ptr(),
                accum_col.columns[2].buf.as_mut_ptr(),
                accum_col.columns[3].buf.as_mut_ptr(),
            );
            ffi::cuda_device_sync();
        }
    }

    // Check for CUDA errors.
    let err = unsafe { ffi::cudaGetLastError() };
    if err != 0 {
        tracing::error!("CUDA bytecode kernel error: {}", err);
        return false;
    }

    true
}

impl<'a, E: FrameworkEval + Sync> ComponentProver<CudaBackend> for CudaFrameworkComponentRef<'a, E> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, CudaBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CudaBackend>,
    ) {
        evaluate_constraint_quotients_impl(self.0, trace, evaluation_accumulator);
    }
}

impl<E: FrameworkEval + Sync> ComponentProver<CudaBackend> for CudaFrameworkComponent<E> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, CudaBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CudaBackend>,
    ) {
        evaluate_constraint_quotients_impl(&self.0, trace, evaluation_accumulator);
    }
}

/// Prepares trace evaluations for constraint quotient computation on CudaBackend.
/// Either borrows committed evaluations directly (subdomain mode) or extends them
/// to the evaluation domain.
fn get_constraint_quotients_inputs_cuda<'a>(
    eval_domain: stwo::core::poly::circle::CircleDomain,
    component_polys: TreeVec<Vec<&'a &Poly<CudaBackend>>>,
    mode: EvaluationMode,
) -> TreeVec<Vec<Cow<'a, CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>>> {
    match mode {
        EvaluationMode::SubDomain { log_expansion: 0 } => {
            component_polys.map_cols(|c| Cow::Borrowed(&c.evals))
        }
        EvaluationMode::SubDomain { log_expansion: _ } => {
            unimplemented!("SubDomain with log_expansion > 0 not yet supported")
        }
        EvaluationMode::ExtendToEvalDomain => {
            let _span = span!(Level::INFO, "CUDA Constraint Extension").entered();
            let twiddles = CudaBackend::precompute_twiddles(eval_domain.half_coset);
            component_polys.as_cols_ref().map_cols(|col| {
                Cow::Owned(col.get_evaluation_on_domain(eval_domain, &twiddles))
            })
        }
    }
}

/// Point-wise CPU constraint evaluation, mirroring the private function in
/// stwo-constraint-framework. Downloads trace data and evaluates constraints
/// row-by-row using `CpuDomainEvaluator`.
fn accumulate_pointwise_cpu<E: FrameworkEval>(
    component: &FrameworkComponent<E>,
    trace_cols: TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>,
    eval_log_size: u32,
    trace_log_size: u32,
    denom_inv: Vec<BaseField>,
    random_coeff_powers: &[SecureField],
    accum: &SecureColumnByCoords<CpuBackend>,
) -> SecureColumnByCoords<CpuBackend> {
    let mut res = SecureColumnByCoords::zeros(1 << eval_log_size);
    for row in 0..(1 << eval_log_size) {
        let eval = CpuDomainEvaluator::new(
            &trace_cols,
            row,
            random_coeff_powers,
            trace_log_size,
            eval_log_size,
            component.log_size(),
            component.claimed_sum(),
        );
        let row_res = component.evaluate(eval).row_res;

        let row_denom_inv = denom_inv[row >> trace_log_size];
        res.set(row, accum.at(row) + row_res * row_denom_inv);
    }
    res
}
