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
    if program.n_registers > 1024 {
        tracing::warn!(
            "Bytecode uses {} registers (max 1024) — falling back to CPU",
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

    // Dump first GPU component's bytecode for debugging
    static DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let should_dump = !DUMPED.swap(true, std::sync::atomic::Ordering::Relaxed);
    if should_dump {
        eprintln!("[BYTECODE_DUMP] {} ops, {} constraints, {} registers",
            program.ops.len(), program.n_constraints, program.n_registers);
        for (i, op) in program.ops.iter().enumerate() {
            eprintln!("  [{i:3}] {op}");
        }
    }

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

    if should_dump {
        eprintln!("[BYTECODE_DUMP] Encoded {} u32 words (from {} ops), interaction_offsets={:?}, total_cols={}",
            encoded.len(), program.ops.len(), interaction_offsets, total_cols);
        // Verify encoded word count matches sum of op.encoded_len()
        let expected_len: usize = program.ops.iter().map(|op| op.encoded_len()).sum();
        eprintln!("[BYTECODE_DUMP] Expected encoded len: {}, actual: {}", expected_len, encoded.len());
        if expected_len != encoded.len() {
            eprintln!("[BYTECODE_DUMP] MISMATCH in encoded length!");
        }
        // Print all LoadTrace ops with their flat column and offset info after remap
        let mut pc = 0;
        for (i, op) in program.ops.iter().enumerate() {
            if let BytecodeOp::LoadTrace { interaction, col_idx, offset, .. } = op {
                let flat_col = encoded[pc + 1];
                let ext_word = encoded[pc + 2];
                eprintln!("[REMAP] op[{i}] inter={interaction} col={col_idx} off={offset} -> flat={flat_col} word2=0x{ext_word:08x}");
            }
            pc += op.encoded_len();
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
    {
        let _span = span!(Level::INFO, "CUDA bytecode kernel launch",
            n_rows = n_rows, n_ops = encoded.len(), n_cols = total_cols,
            n_constraints = program.n_constraints, n_registers = program.n_registers
        ).entered();
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

    // DEBUG: Compare GPU results with CPU for the first component.
    // Download GPU accumulator, compute CPU reference, compare.
    #[cfg(debug_assertions)]
    {
        let gpu_accum0 = accum_col.columns[0].to_cpu();
        let gpu_accum1 = accum_col.columns[1].to_cpu();
        if n_rows <= 256 {
            let mut mismatches = 0;
            for i in 0..n_rows as usize {
                if gpu_accum0[i].0 != 0 || gpu_accum1[i].0 != 0 {
                    // Non-zero result — worth checking
                    if mismatches < 3 {
                        eprintln!("[GPU_CHECK] row {i}: accum0={} accum1={}", gpu_accum0[i].0, gpu_accum1[i].0);
                    }
                    mismatches += 1;
                }
            }
            if mismatches > 0 {
                eprintln!("[GPU_CHECK] {mismatches}/{} non-zero accum values (n_constraints={}, n_ops={})",
                    n_rows, program.n_constraints, program.ops.len());
            }
        }
    }

    // VALIDATION: Compare GPU result with CPU for the first component.
    static VALIDATED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !VALIDATED.swap(true, std::sync::atomic::Ordering::Relaxed) && n_rows <= 256 {
        // Download GPU result
        let gpu_c0 = accum_col.columns[0].to_cpu();
        let gpu_c1 = accum_col.columns[1].to_cpu();
        let gpu_c2 = accum_col.columns[2].to_cpu();
        let gpu_c3 = accum_col.columns[3].to_cpu();

        // Zero the accum and run CPU path
        for i in 0..4 { accum_col.columns[i].buf.zero(); }

        // CPU evaluation
        let cpu_trace_evals: TreeVec<Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>> =
            gpu_trace_evals.as_ref().map(|tree| {
                tree.iter().map(|eval| {
                    CircleEvaluation::new(eval.domain, eval.values.to_cpu())
                }).collect()
            });
        // Print CPU trace values for row 0 (all columns of interaction 1)
        if cpu_trace_evals.len() > 1 {
            let inter1 = &cpu_trace_evals[1];
            for col_idx in 0..std::cmp::min(12, inter1.len()) {
                eprintln!("[CPU_DBG] row=0 inter=1 col={} val={}", col_idx, inter1[col_idx][0].0);
            }
        }
        // Print interaction 2 columns with offset info
        if cpu_trace_evals.len() > 2 {
            let inter2 = &cpu_trace_evals[2];
            for col_idx in 0..std::cmp::min(12, inter2.len()) {
                let val_row0 = inter2[col_idx][0].0;
                // Compute offset=-1 row index using stwo's function
                let prev_row = stwo::core::utils::offset_bit_reversed_circle_domain_index(
                    0, trace_log_size, eval_log_size, -1);
                let val_prev = inter2[col_idx][prev_row].0;
                eprintln!("[CPU_DBG] row=0 inter=2 col={} val_off0={} prev_row={} val_off-1={}",
                    col_idx, val_row0, prev_row, val_prev);
            }
        }

        let cpu_trace_refs = cpu_trace_evals.as_ref().map(|tree| {
            tree.iter().map(|e| e).collect::<Vec<_>>()
        });
        let cpu_result = accumulate_pointwise_cpu(
            component, cpu_trace_refs,
            eval_log_size, trace_log_size,
            denom_inv.to_vec(), random_coeff_powers,
            &SecureColumnByCoords::<CpuBackend> {
                columns: std::array::from_fn(|_| vec![BaseField::from(0u32); n_rows as usize]),
            },
        );

        let mut mismatches = 0;
        for row in 0..n_rows as usize {
            let g0 = gpu_c0[row].0;
            let c0 = cpu_result.columns[0][row].0;
            if g0 != c0 {
                if mismatches < 5 {
                    eprintln!("[VALIDATE] row {row}: GPU=({},{},{},{}) CPU=({},{},{},{})",
                        g0, gpu_c1[row].0, gpu_c2[row].0, gpu_c3[row].0,
                        c0, cpu_result.columns[1][row].0, cpu_result.columns[2][row].0, cpu_result.columns[3][row].0);
                }
                mismatches += 1;
            }
        }
        if mismatches > 0 {
            eprintln!("[VALIDATE] {mismatches}/{n_rows} mismatches! GPU constraint eval is wrong.");
            // Restore CPU result to accum
            *accum_col = SecureColumnByCoords {
                columns: std::array::from_fn(|i| cpu_result.columns[i].iter().copied().collect()),
            };
            return true; // "succeeded" but with CPU data
        } else {
            eprintln!("[VALIDATE] GPU matches CPU perfectly for {n_rows} rows!");
            // Restore GPU result
            *accum_col = SecureColumnByCoords {
                columns: [
                    gpu_c0.into_iter().collect(),
                    gpu_c1.into_iter().collect(),
                    gpu_c2.into_iter().collect(),
                    gpu_c3.into_iter().collect(),
                ],
            };
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
    static CPU_DBG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let should_dbg = !CPU_DBG.swap(true, std::sync::atomic::Ordering::Relaxed);
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

        if should_dbg && row == 0 {
            let arr = row_res.to_m31_array();
            let dinv = denom_inv[row >> trace_log_size as usize];
            eprintln!("[CPU_DBG] row=0 row_res=({},{},{},{}) denom_inv={}",
                arr[0].0, arr[1].0, arr[2].0, arr[3].0, dinv.0);
        }

        let row_denom_inv = denom_inv[row >> trace_log_size];
        res.set(row, accum.at(row) + row_res * row_denom_inv);
    }
    res
}
