//! PolyOps: Circle polynomial operations.
//!
//! NTT evaluate/interpolate uses CPU fallback with stwo's twiddle format
//! to ensure correctness. GPU Merkle + GPU accumulation handle the
//! commitment and field operations.
//!
//! TODO: Port GPU NTT to use stwo's twiddle format for 200-400x speedup.

use num_traits::Zero;
use stwo::core::circle::{CirclePoint, Coset as StwoCoset};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::{CanonicCoset, CircleDomain};
use stwo::prover::backend::{Column, CpuBackend};
use stwo::prover::poly::circle::{
    CircleCoefficients, CircleEvaluation, PolyOps,
};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::poly::BitReversedOrder;

use vortexstark::circle::Coset as VortexCoset;
use vortexstark::device::DeviceBuffer;
use vortexstark::ntt::TwiddleCache;

use super::CudaBackend;
use super::column::CudaColumn;

/// GPU twiddle factors (placeholder — not used for NTT yet).
pub struct CudaTwiddles {
    pub cache: TwiddleCache,
}

/// Convert stwo Coset → VortexSTARK Coset.
pub fn convert_coset(coset: &StwoCoset) -> VortexCoset {
    VortexCoset {
        initial: vortexstark::circle::CirclePoint {
            x: vortexstark::field::M31(coset.initial.x.0),
            y: vortexstark::field::M31(coset.initial.y.0),
        },
        step: vortexstark::circle::CirclePoint {
            x: vortexstark::field::M31(coset.step.x.0),
            y: vortexstark::field::M31(coset.step.y.0),
        },
        log_size: coset.log_size,
    }
}

/// Build a GPU twiddle cache (used for precompute_twiddles).
pub fn twiddle_cache_for_coset(coset: &StwoCoset) -> TwiddleCache {
    TwiddleCache::new(&convert_coset(coset))
}

impl PolyOps for CudaBackend {
    type Twiddles = CudaTwiddles;

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        _twiddles: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self> {
        // CPU NTT for correctness — stwo's twiddle format differs from our GPU kernel
        let cpu_vals = eval.values.to_cpu();
        let cpu_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
            eval.domain, cpu_vals,
        );
        let cpu_twiddles = CpuBackend::precompute_twiddles(eval.domain.half_coset);
        let cpu_poly = CpuBackend::interpolate(cpu_eval, &cpu_twiddles);
        CircleCoefficients::new(cpu_poly.coeffs.into_iter().collect())
    }

    fn eval_at_point(poly: &CircleCoefficients<Self>, point: CirclePoint<SecureField>) -> SecureField {
        let coeffs = poly.coeffs.to_cpu();
        if coeffs.is_empty() { return SecureField::zero(); }
        if coeffs.len() == 1 { return coeffs[0].into(); }
        let mut mappings = vec![point.y];
        let mut x = point.x;
        for _ in 1..poly.log_size() {
            mappings.push(x);
            x = CirclePoint::double_x(x);
        }
        mappings.reverse();
        stwo::core::poly::utils::fold(&coeffs, &mappings)
    }

    fn barycentric_weights(coset: CanonicCoset, p: CirclePoint<SecureField>) -> CudaColumn<SecureField> {
        CpuBackend::barycentric_weights(coset, p).into_iter().collect()
    }

    fn barycentric_eval_at_point(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        weights: &CudaColumn<SecureField>,
    ) -> SecureField {
        let cpu_evals = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
            evals.domain, evals.values.to_cpu(),
        );
        let cpu_weights = weights.to_cpu();
        CpuBackend::barycentric_eval_at_point(&cpu_evals, &cpu_weights)
    }

    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        _twiddles: &TwiddleTree<Self>,
    ) -> SecureField {
        let cpu_evals = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
            evals.domain, evals.values.to_cpu(),
        );
        let cpu_twiddles = CpuBackend::precompute_twiddles(evals.domain.half_coset);
        CpuBackend::eval_at_point_by_folding(&cpu_evals, point, &cpu_twiddles)
    }

    fn split_at_mid(
        poly: CircleCoefficients<Self>,
    ) -> (CircleCoefficients<Self>, CircleCoefficients<Self>) {
        let (left, right) = poly.coeffs.split_at_mid();
        (CircleCoefficients::new(left), CircleCoefficients::new(right))
    }

    fn extend(poly: &CircleCoefficients<Self>, log_size: u32) -> CircleCoefficients<Self> {
        assert!(log_size >= poly.log_size());
        let old_len = poly.coeffs.len();
        let new_len = 1usize << log_size;
        if old_len == new_len {
            return CircleCoefficients::new(poly.coeffs.clone());
        }
        let mut new_buf = DeviceBuffer::<u32>::alloc(new_len);
        new_buf.zero();
        unsafe {
            vortexstark::cuda::ffi::cudaMemcpy(
                new_buf.as_mut_ptr() as *mut std::ffi::c_void,
                poly.coeffs.buf.as_ptr() as *const std::ffi::c_void,
                old_len * 4,
                vortexstark::cuda::ffi::MEMCPY_D2D,
            );
        }
        CircleCoefficients::new(CudaColumn::from_device_buffer(new_buf, new_len))
    }

    fn evaluate(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        _twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // CPU NTT for correctness
        let cpu_coeffs = poly.coeffs.to_cpu();
        let cpu_poly = CircleCoefficients::<CpuBackend>::new(cpu_coeffs);
        let cpu_twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
        let cpu_eval = CpuBackend::evaluate(&cpu_poly, domain, &cpu_twiddles);
        CircleEvaluation::new(domain, cpu_eval.values.into_iter().collect())
    }

    fn evaluate_into(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        _twiddles: &TwiddleTree<Self>,
        _buffer: CudaColumn<BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // CPU fallback (ignores buffer, creates new)
        Self::evaluate(poly, domain, _twiddles)
    }

    fn precompute_twiddles(coset: StwoCoset) -> TwiddleTree<Self> {
        let fwd = twiddle_cache_for_coset(&coset);
        let inv = twiddle_cache_for_coset(&coset);
        TwiddleTree {
            root_coset: coset,
            twiddles: CudaTwiddles { cache: fwd },
            itwiddles: CudaTwiddles { cache: inv },
        }
    }
}
