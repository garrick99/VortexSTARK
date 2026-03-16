//! FriOps: FRI folding on GPU.

use stwo_prover::core::fri::FriOps;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::poly::line::LineEvaluation;
use stwo_prover::core::poly::circle::SecureEvaluation;
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::poly::twiddles::TwiddleTree;

use super::CudaBackend;

impl FriOps for CudaBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        todo!("FriOps::fold_line")
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        todo!("FriOps::fold_circle_into_line")
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        todo!("FriOps::decompose")
    }
}
