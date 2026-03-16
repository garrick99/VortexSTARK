//! QuotientOps: DEEP quotient accumulation.
//!
//! CPU implementation for now — the DEEP quotient involves complex conjugate
//! line coefficients and per-point denominator inverses that need a custom
//! CUDA kernel. The computation is O(domain_size × n_samples) which is
//! significant at large sizes.
//!
//! TODO: GPU kernel for row-parallel quotient accumulation.

use stwo_prover::core::backend::Column;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;
use stwo_prover::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
use stwo_prover::core::poly::circle::{CircleDomain, CircleEvaluation, SecureEvaluation};
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::utils::bit_reverse_index;

use super::CudaBackend;
use super::column::CudaColumn;

impl QuotientOps for CudaBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        _log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        // Download columns to CPU for quotient evaluation
        let cpu_columns: Vec<Vec<BaseField>> = columns.iter()
            .map(|col| col.values.to_cpu())
            .collect();

        let quotient_constants = stwo_prover::core::backend::cpu::quotients::quotient_constants(
            sample_batches, random_coeff,
        );

        let values = unsafe { SecureColumnByCoords::<CudaBackend>::uninitialized(domain.size()) };

        // Evaluate quotients row by row on CPU, then upload
        let n = domain.size();
        let mut result_vals = Vec::with_capacity(n);
        for row in 0..n {
            let domain_point = domain.at(bit_reverse_index(row, domain.log_size()));
            let query_values: Vec<BaseField> = cpu_columns.iter()
                .map(|col| col[row])
                .collect();
            let row_value = stwo_prover::core::backend::cpu::quotients::accumulate_row_quotients(
                sample_batches,
                &query_values,
                &quotient_constants,
                domain_point,
            );
            result_vals.push(row_value);
        }

        // Upload results to GPU as SecureColumnByCoords (4 coordinate columns)
        let mut cols: [Vec<u32>; 4] = [Vec::with_capacity(n), Vec::with_capacity(n),
                                        Vec::with_capacity(n), Vec::with_capacity(n)];
        for v in &result_vals {
            let arr = v.to_m31_array();
            for c in 0..4 {
                cols[c].push(arr[c].0);
            }
        }

        let result = SecureColumnByCoords {
            columns: std::array::from_fn(|c| {
                CudaColumn::from_device_buffer(
                    vortexstark::device::DeviceBuffer::from_host(&cols[c]),
                    n,
                )
            }),
        };

        SecureEvaluation::new(domain, result)
    }
}
