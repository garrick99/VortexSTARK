//! ComponentProver<CudaBackend> for stwo's constraint framework.
//!
//! Uses a newtype wrapper `CudaFrameworkComponent<E>` to satisfy Rust's orphan rule.
//! The implementation downloads GPU trace columns to CPU, evaluates constraints
//! using `CpuDomainEvaluator`, and uploads the result back to the GPU.

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

use super::CudaBackend;

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

impl<E: FrameworkEval> Deref for CudaFrameworkComponent<E> {
    type Target = FrameworkComponent<E>;

    fn deref(&self) -> &Self::Target {
        &self.0
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

impl<E: FrameworkEval + Sync> ComponentProver<CudaBackend> for CudaFrameworkComponent<E> {
    /// Evaluates constraint quotients by downloading GPU trace data to CPU,
    /// running point-wise constraint evaluation, and uploading results back.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, CudaBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CudaBackend>,
    ) {
        let n_constraints = self.n_constraints();
        if n_constraints == 0 {
            return;
        }

        if self.is_disabled() {
            evaluation_accumulator.skip_coeffs(n_constraints);
            return;
        }

        let eval_domain =
            CanonicCoset::new(self.max_constraint_log_degree_bound()).circle_domain();
        let trace_domain = CanonicCoset::new(self.log_size());

        // Extract this component's polynomial columns from the full trace.
        let mut component_polys = trace.polys.sub_tree(self.trace_locations());
        component_polys[PREPROCESSED_TRACE_IDX] = self
            .preprocessed_column_indices()
            .iter()
            .map(|idx| &trace.polys[PREPROCESSED_TRACE_IDX][*idx])
            .collect();

        // Get trace evaluations on the eval domain (GPU-resident).
        let gpu_trace_evals = get_constraint_quotients_inputs_cuda(
            eval_domain,
            component_polys,
            evaluation_accumulator.evaluation_mode(),
        );

        // Download all GPU trace evaluations to CPU.
        let cpu_trace_evals: TreeVec<Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>> =
            gpu_trace_evals.map_cols(|gpu_eval| {
                let cpu_values = gpu_eval.values.to_cpu();
                CircleEvaluation::new(gpu_eval.domain, cpu_values)
            });
        let cpu_trace_refs = cpu_trace_evals.as_cols_ref();

        // Denom inverses (CPU computation, small).
        let log_expand = eval_domain.log_size() - trace_domain.log_size();
        let mut denom_inv = (0..1 << log_expand)
            .map(|i| coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse())
            .collect_vec();
        bit_reverse(&mut denom_inv);

        // Get accumulator column (GPU-resident).
        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), n_constraints)]);
        accum.random_coeff_powers.reverse();

        let _span = span!(
            Level::INFO,
            "CUDA Constraint point-wise eval (CPU fallback)",
            class = "ConstraintEval"
        )
        .entered();

        // Download current accumulator state to CPU.
        let cpu_accum = accum.col.to_cpu();

        // Run point-wise constraint evaluation on CPU.
        let cpu_result = accumulate_pointwise_cpu(
            &self.0,
            cpu_trace_refs,
            eval_domain.log_size(),
            trace_domain.log_size(),
            denom_inv,
            &accum.random_coeff_powers,
            &cpu_accum,
        );

        // Upload result back to GPU.
        *accum.col = SecureColumnByCoords {
            columns: std::array::from_fn(|i| {
                cpu_result.columns[i].iter().copied().collect()
            }),
        };
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
