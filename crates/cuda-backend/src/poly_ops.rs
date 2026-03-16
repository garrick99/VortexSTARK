//! PolyOps: Circle NTT forward/inverse on GPU.

use stwo_prover::core::poly::circle::PolyOps;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::poly::twiddles::TwiddleTree;
use stwo_prover::core::circle::CirclePoint;

use super::CudaBackend;
use super::column::CudaColumn;

impl PolyOps for CudaBackend {
    type Twiddles = ();

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: CudaColumn<BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        todo!("PolyOps::new_canonical_ordered")
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        itwiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        todo!("PolyOps::interpolate")
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
        todo!("PolyOps::eval_at_point")
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        todo!("PolyOps::extend")
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        todo!("PolyOps::evaluate")
    }

    fn precompute_twiddles(coset: stwo_prover::core::circle::Coset) -> TwiddleTree<Self> {
        todo!("PolyOps::precompute_twiddles")
    }
}
