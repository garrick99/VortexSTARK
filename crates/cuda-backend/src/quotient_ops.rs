//! QuotientOps: DEEP quotient accumulation on GPU.

use stwo_prover::core::pcs::quotients::QuotientOps;
use stwo_prover::core::circle::Coset;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::pcs::quotients::ColumnSampleBatch;
use stwo_prover::core::poly::circle::SecureEvaluation;
use stwo_prover::core::poly::BitReversedOrder;

use super::CudaBackend;
use super::column::CudaColumn;

impl QuotientOps for CudaBackend {
    fn accumulate_quotients(
        domain: Coset,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        _log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        todo!("QuotientOps::accumulate_quotients — DEEP quotient on GPU")
    }
}

// Bring in types needed by the signature
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::poly::circle::CircleEvaluation;
