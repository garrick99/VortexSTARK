//! GkrOps: GKR lookup protocol on GPU.

use stwo_prover::core::lookups::gkr_prover::{GkrOps, GkrMultivariatePolyOracle, Layer};
use stwo_prover::core::lookups::mle::{Mle, MleOps};
use stwo_prover::core::lookups::utils::UnivariatePoly;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;

use super::CudaBackend;

impl MleOps<BaseField> for CudaBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        todo!("MleOps<BaseField>::fix_first_variable")
    }
}

impl MleOps<SecureField> for CudaBackend {
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        todo!("MleOps<SecureField>::fix_first_variable")
    }
}

impl GkrOps for CudaBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        todo!("GkrOps::gen_eq_evals")
    }

    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        todo!("GkrOps::next_layer")
    }

    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        todo!("GkrOps::sum_as_poly_in_first_variable")
    }
}
