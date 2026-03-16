//! PolyOps: Circle NTT forward/inverse on GPU.
//!
//! Maps stwo's polynomial operations to VortexSTARK's circle_ntt.cu kernels.
//! This is the most performance-critical trait — all polynomial evaluation
//! and interpolation flows through the GPU NTT.

use num_traits::Zero;
use stwo_prover::core::backend::{Column, ColumnOps};
use stwo_prover::core::circle::{CirclePoint, Coset as StwoCoset};
use stwo_prover::core::fields::m31::{BaseField, M31};
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::FieldExpOps;
use stwo_prover::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use stwo_prover::core::poly::twiddles::TwiddleTree;
use stwo_prover::core::poly::BitReversedOrder;

use vortexstark::circle::Coset as VortexCoset;
use vortexstark::device::DeviceBuffer;
use vortexstark::ntt::{self, TwiddleCache};

use super::CudaBackend;
use super::column::CudaColumn;

/// GPU twiddle factors: forward + inverse, stored on device.
pub struct CudaTwiddles {
    pub cache: TwiddleCache,
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

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: CudaColumn<BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let domain = coset.circle_domain();
        assert_eq!(values.len(), domain.size());
        // Reorder from canonic order to circle domain order, then bit-reverse.
        // For now, do this on CPU. TODO: GPU kernel for reordering.
        let cpu_vals = values.to_cpu();
        let reordered = coset_order_to_circle_domain_order(&cpu_vals);
        let mut col: CudaColumn<BaseField> = reordered.into_iter().collect();
        <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut col);
        CircleEvaluation::new(domain, col)
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        let mut values = eval.values;
        let log_n = eval.domain.log_size();
        let n = values.len();

        // GPU inverse NTT (in-place)
        ntt::interpolate(&mut values.buf, &twiddles.itwiddles.cache);

        // Divide all values by n (the IFFT normalization)
        // TODO: fuse this into the kernel
        let inv = M31::from(n as u32).inverse();
        let mut host = values.buf.to_host();
        let inv_val = inv.0;
        let p = vortexstark::field::m31::P;
        for v in &mut host {
            *v = ((*v as u64 * inv_val as u64) % p as u64) as u32;
        }
        values.buf = DeviceBuffer::from_host(&host);

        CirclePoly::new(values)
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
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

        stwo_prover::core::poly::utils::fold(&coeffs, &mappings)
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        assert!(log_size >= poly.log_size());
        let old_len = poly.coeffs.len();
        let new_len = 1usize << log_size;
        if old_len == new_len {
            return CirclePoly::new(poly.coeffs.clone());
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
        CirclePoly::new(CudaColumn::from_device_buffer(new_buf, new_len))
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let mut values = Self::extend(poly, domain.log_size()).coeffs;

        // GPU forward NTT (in-place)
        ntt::evaluate(&mut values.buf, &twiddles.twiddles.cache);

        CircleEvaluation::new(domain, values)
    }

    fn precompute_twiddles(coset: StwoCoset) -> TwiddleTree<Self> {
        let vortex_coset = convert_coset(&coset);
        // Both forward and inverse twiddles live in the same TwiddleCache.
        // We wrap the same cache in both slots — stwo only uses twiddles for
        // evaluate and itwiddles for interpolate, and our cache has both.
        let fwd = TwiddleCache::new(&vortex_coset);
        let inv = TwiddleCache::new(&vortex_coset);

        TwiddleTree {
            root_coset: coset,
            twiddles: CudaTwiddles { cache: fwd },
            itwiddles: CudaTwiddles { cache: inv },
        }
    }
}

/// Reorder from canonic coset order to circle domain order.
/// Takes even-indexed elements as first half, odd-indexed reversed as second half.
fn coset_order_to_circle_domain_order(values: &[BaseField]) -> Vec<BaseField> {
    let n = values.len();
    let half = n / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..half {
        out.push(values[i << 1]);
    }
    for i in 0..half {
        out.push(values[n - 1 - (i << 1)]);
    }
    out
}
