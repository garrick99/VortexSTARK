//! FriOps: FRI folding on GPU.
//!
//! Stwo's SecureColumnByCoords already stores QM31 in SoA layout
//! (4 separate BaseField columns), which matches our CUDA kernel layout exactly.

use stwo_prover::core::backend::Column;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;
use stwo_prover::core::fields::FieldExpOps;
use stwo_prover::core::fri::FriOps;
use stwo_prover::core::poly::circle::SecureEvaluation;
use stwo_prover::core::poly::line::LineEvaluation;
use stwo_prover::core::poly::twiddles::TwiddleTree;
use stwo_prover::core::poly::BitReversedOrder;

use vortexstark::cuda::ffi;
use vortexstark::device::DeviceBuffer;

use super::CudaBackend;
use super::column::CudaColumn;

impl FriOps for CudaBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let n = eval.len();
        assert!(n >= 2);
        let half_n = n / 2;

        // SecureColumnByCoords already has 4 separate BaseField columns = SoA
        let cols = &eval.values.columns;

        // Compute fold twiddles
        let domain = eval.domain();
        let vortex_coset = super::poly_ops::convert_coset(&domain.coset());
        let d_twiddles = vortexstark::fri::compute_fold_twiddles_on_demand(
            &vortex_coset, false,
        );

        let alpha_arr = qm31_to_arr(alpha);

        let mut o0 = DeviceBuffer::<u32>::alloc(half_n);
        let mut o1 = DeviceBuffer::<u32>::alloc(half_n);
        let mut o2 = DeviceBuffer::<u32>::alloc(half_n);
        let mut o3 = DeviceBuffer::<u32>::alloc(half_n);

        unsafe {
            ffi::cuda_fold_line_soa(
                cols[0].buf.as_ptr(), cols[1].buf.as_ptr(),
                cols[2].buf.as_ptr(), cols[3].buf.as_ptr(),
                d_twiddles.as_ptr(),
                o0.as_mut_ptr(), o1.as_mut_ptr(), o2.as_mut_ptr(), o3.as_mut_ptr(),
                alpha_arr.as_ptr(),
                half_n as u32,
            );
        }

        let result = SecureColumnByCoords {
            columns: [
                CudaColumn::from_device_buffer(o0, half_n),
                CudaColumn::from_device_buffer(o1, half_n),
                CudaColumn::from_device_buffer(o2, half_n),
                CudaColumn::from_device_buffer(o3, half_n),
            ],
        };
        let result_domain = domain.double();
        LineEvaluation::new(result_domain, result)
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        let n = src.len();
        let half_n = n / 2;

        let src_cols = &src.values.columns;
        let dst_cols = &mut dst.values.columns;

        let domain = src.domain;
        let vortex_coset = super::poly_ops::convert_coset(&domain.half_coset);
        let d_twiddles = vortexstark::fri::compute_fold_twiddles_on_demand(
            &vortex_coset, true,
        );

        let alpha_arr = qm31_to_arr(alpha);
        let alpha_sq = alpha * alpha;
        let alpha_sq_arr = qm31_to_arr(alpha_sq);

        unsafe {
            ffi::cuda_fold_circle_into_line_soa(
                dst_cols[0].buf.as_mut_ptr(), dst_cols[1].buf.as_mut_ptr(),
                dst_cols[2].buf.as_mut_ptr(), dst_cols[3].buf.as_mut_ptr(),
                src_cols[0].buf.as_ptr(), src_cols[1].buf.as_ptr(),
                src_cols[2].buf.as_ptr(), src_cols[3].buf.as_ptr(),
                d_twiddles.as_ptr(),
                alpha_arr.as_ptr(),
                alpha_sq_arr.as_ptr(),
                half_n as u32,
            );
        }
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        // Decompose: f(P) = g(P) + lambda * alternating(P)
        // Small enough for CPU fallback.
        let n = eval.len();

        // Sum all values to get lambda = sum / n
        let mut sum = SecureField::from(BaseField::from(0u32));
        for i in 0..n {
            sum = sum + eval.values.at(i);
        }
        let n_inv: SecureField = BaseField::from(n as u32).inverse().into();
        let lambda = sum * n_inv;

        // g(P) = f(P) - lambda * alternating(P)
        let mut new_cols: [CudaColumn<BaseField>; 4] = std::array::from_fn(|_| {
            CudaColumn::from_device_buffer(DeviceBuffer::<u32>::alloc(0), 0)
        });

        // Download each coordinate column, subtract lambda contribution, re-upload
        for c in 0..4 {
            let mut host = eval.values.columns[c].buf.to_host();
            let lambda_arr = lambda.to_m31_array();
            let lc = lambda_arr[c].0;
            let p = vortexstark::field::m31::P;
            for i in 0..n {
                let sign: u32 = if i % 2 == 0 { lc } else { p - lc };
                let v = host[i] as u64;
                let s = sign as u64;
                host[i] = ((v + p as u64 - s) % p as u64) as u32;
            }
            new_cols[c] = CudaColumn::from_device_buffer(DeviceBuffer::from_host(&host), n);
        }

        let result = SecureColumnByCoords { columns: new_cols };
        (SecureEvaluation::new(eval.domain, result), lambda)
    }
}

fn qm31_to_arr(v: SecureField) -> [u32; 4] {
    let arr = v.to_m31_array();
    [arr[0].0, arr[1].0, arr[2].0, arr[3].0]
}
