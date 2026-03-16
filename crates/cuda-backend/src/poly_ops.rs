//! PolyOps: Circle NTT forward/inverse on GPU.
//!
//! Maps to VortexSTARK's circle_ntt.cu kernels.
//! This is the most performance-critical trait — all polynomial evaluation
//! and interpolation flows through here.

use stwo_prover::core::poly::circle::PolyOps;
use stwo_prover::core::circle::Coset;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::poly::circle::{CircleEvaluation, CirclePoly};
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::poly::twiddles::TwiddleTree;

use super::CudaBackend;

/// Twiddle factors stored on GPU for NTT.
pub struct CudaTwiddles {
    // TODO: wrap VortexSTARK's ForwardTwiddleCache / InverseTwiddleCache
    _placeholder: (),
}

impl PolyOps for CudaBackend {
    type Twiddles = CudaTwiddles;

    fn new_canonical_ordered(
        coset: Coset,
        values: super::column::CudaColumn<BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        todo!("PolyOps::new_canonical_ordered — wire to GPU bit-reverse + reorder")
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        itwiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        todo!("PolyOps::interpolate — wire to cuda_circle_ntt_interpolate")
    }

    fn eval_at_point(
        poly: &CirclePoly<Self>,
        point: stwo::core::circle::CirclePoint<SecureField>,
    ) -> SecureField {
        todo!("PolyOps::eval_at_point — Horner evaluation on GPU or CPU")
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        todo!("PolyOps::extend — zero-pad coefficients")
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: Coset,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        todo!("PolyOps::evaluate — wire to cuda_circle_ntt_evaluate")
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        todo!("PolyOps::precompute_twiddles — wire to VortexSTARK twiddle cache")
    }
}
