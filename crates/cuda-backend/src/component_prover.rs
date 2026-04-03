//! ComponentProver<CudaBackend> for stwo's constraint framework.
//!
//! Uses a newtype wrapper `CudaFrameworkComponent<E>` to satisfy Rust's orphan rule.
//!
//! The primary path records constraint evaluation as register-based bytecode,
//! then runs the bytecode interpreter kernel on the GPU. The register-based VM
//! eliminates the Clone problem: Clone just copies a register index, so all 67
//! stwo-cairo components work without stack imbalance issues.
//!
//! If bytecode recording fails for any reason (unsupported operation, panic in
//! the tracer, etc.), it falls back to the CPU path: download trace, evaluate
//! on CPU, upload results.

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
pub struct CudaFrameworkComponent<E: FrameworkEval>(pub FrameworkComponent<E>);

/// A borrowing wrapper around `&FrameworkComponent<E>` that implements
/// `ComponentProver<CudaBackend>`.
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

    // GPU register-based bytecode constraint eval.
    let gpu_success = try_gpu_bytecode_eval(
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

/// Try to evaluate constraints on GPU using the register-based bytecode interpreter.
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
        return true;
    }

    // (debug logging moved after register count check)

    // Register-based VM has no stack balance issue. No verify_stack_balance() needed.
    // 16-bit register indices support up to 65535 registers; GPU MAX_REGS=1024.
    // Components with >1024 registers use the warp-cooperative kernel.
    // Components with >8192 registers fall back to CPU (unlikely in practice).
    if program.n_registers > 8192 {
        tracing::warn!(
            "Bytecode uses {} registers (max 8192) — falling back to CPU",
            program.n_registers
        );
        return false;
    }

    let _span = span!(
        Level::INFO,
        "GPU register bytecode constraint eval",
        n_ops = program.ops.len(),
        n_constraints = program.n_constraints,
        n_registers = program.n_registers,
        eval_log_size = eval_log_size,
        trace_log_size = trace_log_size,
    )
    .entered();

    // Phase B: Build flat column pointer array and remap bytecode.
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

    // Encode bytecode and remap LoadTrace to use flat column indices.
    let mut encoded = program.encode();

    // Walk through ops and encoded words in parallel to remap LoadTrace operands.
    // LoadTrace is 3 words: word1=header (dst), word2=flat_col, word3=sign|abs_off.
    // We overwrite word2 with the computed flat column index.
    {
        let mut pc = 0;
        for op in &program.ops {
            match op {
                BytecodeOp::LoadTrace { interaction, col_idx, .. } => {
                    let flat_idx = interaction_offsets[*interaction as usize] + *col_idx as usize;
                    if flat_idx >= total_cols {
                        tracing::warn!(
                            "LoadTrace flat_idx {} >= total_cols {} (interaction={}, col_idx={})",
                            flat_idx, total_cols, interaction, col_idx
                        );
                        return false;
                    }
                    // word1 (pc+0): already has correct [opcode | dst] from encode()
                    // word2 (pc+1): replace pre-remap interaction/col_idx with flat_col
                    encoded[pc + 1] = flat_idx as u32;
                    // word3 (pc+2): sign|abs_off already correct from encode()
                    pc += 3;
                }
                _ => {
                    pc += op.encoded_len();
                }
            }
        }
    }

    let n_rows = 1u32 << eval_log_size;
    let trace_n_rows = 1u32 << trace_log_size;

    // Validate all flat column indices are in bounds.
    {
        let mut pc = 0;
        for op in &program.ops {
            if let BytecodeOp::LoadTrace { .. } = op {
                // LoadTrace: 3 words. word2 (pc+1) holds flat_col after remap.
                let flat_col = encoded[pc + 1];
                if flat_col as usize >= total_cols {
                    tracing::error!(
                        "Bytecode word[{}]: flat_col={flat_col} >= total_cols={total_cols}",
                        pc + 1
                    );
                    return false;
                }
            }
            pc += op.encoded_len();
        }
    }

    // Validate column pointers.
    for (i, col_ptr) in col_ptrs.iter().enumerate() {
        if col_ptr.is_null() {
            tracing::error!("Column pointer {i} is null!");
            return false;
        }
    }

    // Phase C: Upload data to GPU.
    let d_bytecode = DeviceBuffer::from_host(&encoded);
    let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);
    let col_sizes: Vec<u32> = vec![n_rows; total_cols];
    let d_col_sizes = DeviceBuffer::from_host(&col_sizes);

    let mut coeff_flat: Vec<u32> = Vec::with_capacity(random_coeff_powers.len() * 4);
    for coeff in random_coeff_powers {
        let arr = coeff.to_m31_array();
        coeff_flat.push(arr[0].0);
        coeff_flat.push(arr[1].0);
        coeff_flat.push(arr[2].0);
        coeff_flat.push(arr[3].0);
    }
    let d_coeff = DeviceBuffer::from_host(&coeff_flat);

    let denom_inv_raw: Vec<u32> = denom_inv.iter().map(|v| v.0).collect();
    let d_denom_inv = DeviceBuffer::from_host(&denom_inv_raw);

    // Sync GPU before kernel launch.
    unsafe { ffi::cuda_device_sync(); }
    let pre_err = unsafe { ffi::cudaGetLastError() };
    if pre_err != 0 {
        tracing::error!("CUDA context already has error {} before bytecode kernel", pre_err);
        return false;
    }

    // Phase D: Launch kernel.
    // Use warp-cooperative kernel for high-register components (>1024 regs)
    // to avoid local memory spilling. The warp kernel distributes the register
    // file across 32 lanes, reducing per-thread memory 32x.
    {
        let use_warp = program.n_registers > 1024;
        let _span = span!(Level::INFO, "CUDA bytecode kernel launch",
            n_rows = n_rows, n_ops = encoded.len(), n_cols = total_cols,
            n_constraints = program.n_constraints, n_registers = program.n_registers,
            warp_cooperative = use_warp,
        ).entered();
        unsafe {
            if use_warp {
                ffi::cuda_warp_bytecode_constraint_eval(
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
                    program.n_registers as u32,
                );
            } else {
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
                    program.n_registers as u32,
                );
            }
            ffi::cuda_device_sync();
        }
    }

    // Check for CUDA errors.
    let err = unsafe { ffi::cudaGetLastError() };
    if err != 0 {
        tracing::error!(
            "CUDA bytecode kernel error {err}: n_ops={}, n_rows={n_rows}, \
             n_cols={total_cols}, n_constraints={}, n_registers={}",
            encoded.len(), program.n_constraints, program.n_registers
        );
        for i in 0..4 {
            accum_col.columns[i].buf.zero();
        }
        unsafe { ffi::cudaGetLastError(); }
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
            // Extend polynomial evaluations to the larger sub-domain via GPU NTT.
            // Same path as ExtendToEvalDomain: interpolate → evaluate at eval_domain.
            let _span = span!(Level::INFO, "CUDA SubDomain Extension").entered();
            let twiddles = CudaBackend::precompute_twiddles(eval_domain.half_coset);
            component_polys.as_cols_ref().map_cols(|col| {
                Cow::Owned(col.get_evaluation_on_domain(eval_domain, &twiddles))
            })
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

/// Point-wise CPU constraint evaluation fallback.
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
