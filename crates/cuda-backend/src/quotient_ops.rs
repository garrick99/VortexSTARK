//! QuotientOps: DEEP quotient accumulation.
//!
//! CPU fallback implementation — downloads column data to CPU, runs CPU
//! quotient logic, uploads results back to GPU.
//!
//! TODO: GPU kernel for row-parallel quotient accumulation.

use stwo::prover::backend::Column;
use stwo::core::fields::m31::BaseField;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use stwo::prover::poly::BitReversedOrder;
use stwo::core::pcs::quotients::ColumnSampleBatch;

use super::CudaBackend;

// Note: QuotientOps is re-exported from stwo::prover::QuotientOps.
// AccumulatedNumerators is in the same module but not re-exported.
// We implement the trait by delegating to CpuBackend.
impl stwo::prover::QuotientOps for CudaBackend {
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<stwo::prover::AccumulatedNumerators<Self>>,
    ) {
        // CPU fallback: download columns, run CPU logic, upload results.
        let cpu_columns: Vec<Vec<BaseField>> = columns.iter()
            .map(|col| col.values.to_cpu())
            .collect();
        let cpu_evals: Vec<CircleEvaluation<
            stwo::prover::backend::CpuBackend, BaseField, BitReversedOrder
        >> = columns.iter().zip(cpu_columns.iter())
            .map(|(col, cpu_vals)| {
                CircleEvaluation::new(col.domain, cpu_vals.clone())
            })
            .collect();
        let cpu_eval_refs: Vec<&CircleEvaluation<
            stwo::prover::backend::CpuBackend, BaseField, BitReversedOrder
        >> = cpu_evals.iter().collect();

        let mut cpu_acc_vec: Vec<stwo::prover::AccumulatedNumerators<
            stwo::prover::backend::CpuBackend
        >> = Vec::new();
        <stwo::prover::backend::CpuBackend as stwo::prover::QuotientOps>::accumulate_numerators(
            &cpu_eval_refs, sample_batches, &mut cpu_acc_vec,
        );

        // Convert CPU results to GPU
        for cpu_acc in cpu_acc_vec {
            let gpu_cols = SecureColumnByCoords {
                columns: std::array::from_fn(|i| {
                    cpu_acc.partial_numerators_acc.columns[i].iter().copied().collect()
                }),
            };
            accumulated_numerators_vec.push(stwo::prover::AccumulatedNumerators {
                sample_point: cpu_acc.sample_point,
                partial_numerators_acc: gpu_cols,
                first_linear_term_acc: cpu_acc.first_linear_term_acc,
            });
        }
    }

    fn compute_quotients_and_combine(
        accumulations: Vec<stwo::prover::AccumulatedNumerators<Self>>,
        lifting_log_size: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        // CPU fallback: convert GPU data to CPU, compute, upload back.
        let cpu_accs: Vec<stwo::prover::AccumulatedNumerators<
            stwo::prover::backend::CpuBackend
        >> = accumulations
            .into_iter()
            .map(|acc| stwo::prover::AccumulatedNumerators {
                sample_point: acc.sample_point,
                partial_numerators_acc: acc.partial_numerators_acc.to_cpu(),
                first_linear_term_acc: acc.first_linear_term_acc,
            })
            .collect();

        let cpu_result = <stwo::prover::backend::CpuBackend as stwo::prover::QuotientOps>::compute_quotients_and_combine(
            cpu_accs, lifting_log_size,
        );

        // Upload to GPU
        let gpu_cols = SecureColumnByCoords {
            columns: std::array::from_fn(|i| {
                cpu_result.values.columns[i].iter().copied().collect()
            }),
        };
        SecureEvaluation::new(cpu_result.domain, gpu_cols)
    }
}
