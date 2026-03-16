//! QuotientOps: DEEP quotient accumulation on GPU.

use stwo_prover::core::pcs::quotients::{QuotientOps, ColumnSampleBatch};
use stwo_prover::core::poly::circle::{CircleDomain, CircleEvaluation, SecureEvaluation};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::poly::BitReversedOrder;

use super::CudaBackend;

impl QuotientOps for CudaBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        todo!("QuotientOps::accumulate_quotients")
    }
}
