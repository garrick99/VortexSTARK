//! FriOps: FRI folding on GPU.
//!
//! Maps to VortexSTARK's fri.cu kernels (fold_line, fold_circle_into_line).

use stwo_prover::core::fri::FriOps;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::poly::line::LineEvaluation;
use stwo_prover::core::poly::circle::SecureEvaluation;
use stwo_prover::core::poly::BitReversedOrder;

use super::CudaBackend;

impl FriOps for CudaBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &super::poly_ops::CudaTwiddles,
    ) -> LineEvaluation<Self> {
        todo!("FriOps::fold_line — wire to cuda_fri_fold_line")
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &super::poly_ops::CudaTwiddles,
    ) {
        todo!("FriOps::fold_circle_into_line — wire to cuda_fri_fold_circle_into_line")
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        todo!("FriOps::decompose — extract even/odd components")
    }
}
