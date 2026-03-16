//! QuotientOps: DEEP quotient accumulation.
//!
//! CPU fallback — delegates to CpuBackend after downloading columns.

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::{Column, CpuBackend};
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{AccumulatedNumerators, ColumnSampleBatch, QuotientOps};

use super::CudaBackend;
use super::column::CudaColumn;

impl QuotientOps for CudaBackend {
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
    ) {
        // Download BaseField columns to CPU
        let cpu_columns: Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> =
            columns.iter().map(|col| {
                let cpu_vals = col.values.to_cpu();
                CircleEvaluation::new(col.domain, cpu_vals)
            }).collect();
        let cpu_col_refs: Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> =
            cpu_columns.iter().collect();

        let mut cpu_accs = Vec::new();
        CpuBackend::accumulate_numerators(&cpu_col_refs, sample_batches, &mut cpu_accs);

        // Upload results to GPU
        for acc in cpu_accs {
            let gpu_coords = SecureColumnByCoords {
                columns: std::array::from_fn(|i| {
                    acc.partial_numerators_acc.columns[i].iter().copied().collect()
                }),
            };
            accumulated_numerators_vec.push(AccumulatedNumerators {
                sample_point: acc.sample_point,
                partial_numerators_acc: gpu_coords,
                first_linear_term_acc: acc.first_linear_term_acc,
            });
        }
    }

    fn compute_quotients_and_combine(
        accs: Vec<AccumulatedNumerators<Self>>,
        lifting_log_size: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        // Download to CPU
        let cpu_accs: Vec<AccumulatedNumerators<CpuBackend>> = accs
            .into_iter()
            .map(|acc| {
                let cpu_coords = SecureColumnByCoords {
                    columns: std::array::from_fn(|i| {
                        acc.partial_numerators_acc.columns[i].to_cpu()
                    }),
                };
                AccumulatedNumerators {
                    sample_point: acc.sample_point,
                    partial_numerators_acc: cpu_coords,
                    first_linear_term_acc: acc.first_linear_term_acc,
                }
            })
            .collect();

        let cpu_result = CpuBackend::compute_quotients_and_combine(cpu_accs, lifting_log_size);

        // Upload to GPU
        let gpu_cols = SecureColumnByCoords {
            columns: std::array::from_fn(|i| {
                cpu_result.values.columns[i].iter().copied().collect()
            }),
        };
        SecureEvaluation::new(cpu_result.domain, gpu_cols)
    }
}
