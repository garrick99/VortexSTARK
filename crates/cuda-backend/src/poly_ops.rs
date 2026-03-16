//! PolyOps: Circle NTT forward/inverse on GPU.
//!
//! Maps stwo's polynomial operations to VortexSTARK's circle_ntt.cu kernels.
//! This is the most performance-critical trait — all polynomial evaluation
//! and interpolation flows through the GPU NTT.

use num_traits::Zero;
use stwo::prover::backend::{Col, Column};
use stwo::core::circle::{CirclePoint, Coset as StwoCoset};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::{CanonicCoset, CircleDomain};
use stwo::prover::poly::circle::{
    CircleEvaluation, CircleCoefficients, PolyOps,
};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::poly::BitReversedOrder;

use vortexstark::circle::Coset as VortexCoset;
use vortexstark::device::DeviceBuffer;

use super::CudaBackend;
use super::column::CudaColumn;

/// GPU twiddle factors: forward + inverse, stored on device.
pub struct CudaTwiddles {
    pub cache: vortexstark::ntt::TwiddleCache,
}

/// Convert stwo Coset → VortexSTARK Coset for twiddle computation.
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

impl PolyOps for CudaBackend {
    type Twiddles = CudaTwiddles;

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self> {
        // CPU fallback: download, interpolate on CPU, upload.
        // TODO: restore GPU NTT after fixing the twiddle/domain compatibility issue.
        let cpu_vals = eval.values.to_cpu();
        let cpu_eval = stwo::prover::poly::circle::CircleEvaluation::<
            stwo::prover::backend::CpuBackend, BaseField, BitReversedOrder
        >::new(eval.domain, cpu_vals);
        let cpu_twiddles = stwo::prover::backend::CpuBackend::precompute_twiddles(eval.domain.half_coset);
        let cpu_poly = stwo::prover::backend::CpuBackend::interpolate(cpu_eval, &cpu_twiddles);
        let gpu_coeffs: CudaColumn<BaseField> = cpu_poly.coeffs.into_iter().collect();
        CircleCoefficients::new(gpu_coeffs)
    }

    fn eval_at_point(poly: &CircleCoefficients<Self>, point: CirclePoint<SecureField>) -> SecureField {
        // Single-point evaluation via Horner's method — small, CPU is fine.
        let coeffs = poly.coeffs.to_cpu();
        if coeffs.is_empty() {
            return SecureField::zero();
        }
        if coeffs.len() == 1 {
            return coeffs[0].into();
        }

        let mut mappings = vec![point.y];
        let mut x = point.x;
        for _ in 1..poly.log_size() {
            mappings.push(x);
            x = CirclePoint::double_x(x);
        }
        mappings.reverse();

        stwo::core::poly::utils::fold(&coeffs, &mappings)
    }

    fn barycentric_weights(
        coset: CanonicCoset,
        p: CirclePoint<SecureField>,
    ) -> Col<Self, SecureField> {
        // CPU fallback — small computation.
        let cpu_weights = stwo::prover::backend::CpuBackend::barycentric_weights(coset, p);
        cpu_weights.into_iter().collect()
    }

    fn barycentric_eval_at_point(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        weights: &Col<Self, SecureField>,
    ) -> SecureField {
        // CPU fallback — small computation.
        let cpu_evals_vals = evals.values.to_cpu();
        let cpu_weights = weights.to_cpu();
        let cpu_evals = CircleEvaluation::<stwo::prover::backend::CpuBackend, BaseField, BitReversedOrder>::new(
            evals.domain, cpu_evals_vals,
        );
        stwo::prover::backend::CpuBackend::barycentric_eval_at_point(&cpu_evals, &cpu_weights)
    }

    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureField {
        // CPU fallback — single-point evaluation.
        let cpu_vals = evals.values.to_cpu();
        let cpu_evals = CircleEvaluation::<stwo::prover::backend::CpuBackend, BaseField, BitReversedOrder>::new(
            evals.domain, cpu_vals,
        );
        let cpu_twiddles = stwo::prover::backend::CpuBackend::precompute_twiddles(
            twiddles.root_coset,
        );
        stwo::prover::backend::CpuBackend::eval_at_point_by_folding(&cpu_evals, point, &cpu_twiddles)
    }

    fn extend(poly: &CircleCoefficients<Self>, log_size: u32) -> CircleCoefficients<Self> {
        assert!(log_size >= poly.log_size());
        let old_len = poly.coeffs.len();
        let new_len = 1usize << log_size;
        if old_len == new_len {
            return CircleCoefficients::new(poly.coeffs.clone());
        }
        // Zero-pad on GPU
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
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // CPU fallback: download, evaluate on CPU, upload.
        let cpu_coeffs = poly.coeffs.to_cpu();
        let cpu_poly = stwo::prover::poly::circle::CircleCoefficients::<
            stwo::prover::backend::CpuBackend
        >::new(cpu_coeffs);
        let cpu_twiddles = stwo::prover::backend::CpuBackend::precompute_twiddles(domain.half_coset);
        let cpu_eval = stwo::prover::backend::CpuBackend::evaluate(&cpu_poly, domain, &cpu_twiddles);
        let gpu_vals: CudaColumn<BaseField> = cpu_eval.values.into_iter().collect();
        CircleEvaluation::new(domain, gpu_vals)
    }

    fn evaluate_into(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
        buffer: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // Copy coefficients into the provided buffer, zero-pad the rest.
        let poly_len = poly.coeffs.len();
        let buf_len = buffer.len;
        assert_eq!(buf_len, domain.size());

        let mut values = buffer;
        // Zero the buffer
        values.buf.zero();
        // Copy coefficient data
        if poly_len > 0 {
            unsafe {
                vortexstark::cuda::ffi::cudaMemcpy(
                    values.buf.as_mut_ptr() as *mut std::ffi::c_void,
                    poly.coeffs.buf.as_ptr() as *const std::ffi::c_void,
                    poly_len * 4,
                    vortexstark::cuda::ffi::MEMCPY_D2D,
                );
            }
        }

        // CPU fallback for evaluate_into
        let cpu_coeffs = values.to_cpu();
        let n = 1usize << domain.log_size();
        let mut padded = cpu_coeffs;
        padded.resize(n, BaseField::from(0u32));
        let cpu_poly = stwo::prover::poly::circle::CircleCoefficients::<
            stwo::prover::backend::CpuBackend
        >::new(padded);
        let cpu_twiddles = stwo::prover::backend::CpuBackend::precompute_twiddles(domain.half_coset);
        let cpu_eval = stwo::prover::backend::CpuBackend::evaluate(&cpu_poly, domain, &cpu_twiddles);
        let gpu_vals: CudaColumn<BaseField> = cpu_eval.values.into_iter().collect();
        CircleEvaluation::new(domain, gpu_vals)
    }

    fn precompute_twiddles(coset: StwoCoset) -> TwiddleTree<Self> {
        let vortex_coset = convert_coset(&coset);
        // Both forward and inverse twiddles live in the same TwiddleCache.
        // We wrap the same cache in both slots — stwo only uses twiddles for
        // evaluate and itwiddles for interpolate, and our cache has both.
        let fwd = vortexstark::ntt::TwiddleCache::new(&vortex_coset);
        let inv = vortexstark::ntt::TwiddleCache::new(&vortex_coset);

        TwiddleTree {
            root_coset: coset,
            twiddles: CudaTwiddles { cache: fwd },
            itwiddles: CudaTwiddles { cache: inv },
        }
    }

    fn split_at_mid(
        poly: CircleCoefficients<Self>,
    ) -> (CircleCoefficients<Self>, CircleCoefficients<Self>) {
        let (left, right) = poly.coeffs.split_at_mid();
        (CircleCoefficients::new(left), CircleCoefficients::new(right))
    }
}
