//! GkrOps: GKR lookup protocol on GPU.
//!
//! The GKR (Goldwasser-Kalai-Rothblum) protocol is used for efficient
//! lookup arguments. This is the newest addition to stwo's trait surface.

use stwo_prover::core::lookups::gkr_prover::{GkrOps, Layer};
use stwo_prover::core::lookups::mle::{Mle, MleOps};
use stwo_prover::core::lookups::utils::UnivariatePoly;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;

use super::CudaBackend;
use super::column::CudaColumn;

impl MleOps<BaseField> for CudaBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        todo!("MleOps<BaseField>::fix_first_variable — GPU MLE evaluation")
    }
}

impl MleOps<SecureField> for CudaBackend {
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        todo!("MleOps<SecureField>::fix_first_variable — GPU MLE evaluation")
    }
}

impl GkrOps for CudaBackend {
    fn gen_eq_evals(
        y: &[SecureField],
        v: SecureField,
    ) -> Mle<Self, SecureField> {
        todo!("GkrOps::gen_eq_evals — generate equality evaluations on GPU")
    }

    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        todo!("GkrOps::next_layer — compute next GKR layer on GPU")
    }

    fn sum_as_poly_in_first_variable(
        h: &Layer<Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        todo!("GkrOps::sum_as_poly_in_first_variable — sumcheck polynomial")
    }
}
