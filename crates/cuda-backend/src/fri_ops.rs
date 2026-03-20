//! FriOps: FRI folding on GPU.
//!
//! Fold twiddles computed on CPU from stwo's domain iteration (matching
//! the CPU FRI exactly), uploaded as flat arrays for GPU kernels.

use stwo::core::fields::qm31::SecureField;
use stwo::core::utils::bit_reverse_index;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::backend::{Column, CpuBackend};
use stwo::prover::fri::FriOps;
use stwo::prover::poly::circle::SecureEvaluation;
use stwo::prover::line::LineEvaluation;
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::poly::BitReversedOrder;

use vortexstark::cuda::ffi;
use vortexstark::device::DeviceBuffer;

use super::CudaBackend;
use super::column::CudaColumn;

impl FriOps for CudaBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
        fold_step: u32,
    ) -> LineEvaluation<Self> {
        assert!(fold_step >= 1);

        let mut folding_alpha = alpha;
        let mut res = gpu_fold_line_single(eval, folding_alpha);
        for _ in 0..fold_step - 1 {
            folding_alpha = folding_alpha * folding_alpha;
            res = gpu_fold_line_single(&res, folding_alpha);
        }
        res
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) {
        let n = src.len();
        let half_n = n / 2;
        let domain = src.domain;

        // Compute fold twiddles: 1/y for each pair's domain point
        let mut inv_y_vals = Vec::with_capacity(half_n);
        for i in 0..half_n {
            let p = domain.at(bit_reverse_index(i << 1, domain.log_size()));
            inv_y_vals.push(p.y.inverse().0);
        }
        let d_twiddles = DeviceBuffer::from_host(&inv_y_vals);

        let src_cols = &src.values.columns;
        let dst_cols = &mut dst.values.columns;

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
            ffi::cuda_device_sync();
        }
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        // CPU — small computation
        let cpu_eval = secure_eval_to_cpu(eval);
        let (cpu_result, lambda) = CpuBackend::decompose(&cpu_eval);
        (secure_eval_from_cpu(&cpu_result), lambda)
    }
}

/// GPU fold_line: compute twiddles from domain iteration (matching CPU exactly).
fn gpu_fold_line_single(
    eval: &LineEvaluation<CudaBackend>,
    alpha: SecureField,
) -> LineEvaluation<CudaBackend> {
    let n = eval.len();
    assert!(n >= 2);
    let half_n = n / 2;
    let domain = eval.domain();

    // Compute fold twiddles: 1/x for each pair's domain point
    let mut inv_x_vals = Vec::with_capacity(half_n);
    for i in 0..half_n {
        let x = domain.at(bit_reverse_index(i << 1, domain.log_size()));
        inv_x_vals.push(x.inverse().0);
    }
    let d_twiddles = DeviceBuffer::from_host(&inv_x_vals);

    let cols = &eval.values.columns;
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
        ffi::cuda_device_sync();
    }

    let result = SecureColumnByCoords {
        columns: [
            CudaColumn::from_device_buffer(o0, half_n),
            CudaColumn::from_device_buffer(o1, half_n),
            CudaColumn::from_device_buffer(o2, half_n),
            CudaColumn::from_device_buffer(o3, half_n),
        ],
    };
    LineEvaluation::new(domain.double(), result)
}

fn secure_eval_to_cpu(eval: &SecureEvaluation<CudaBackend, BitReversedOrder>) -> SecureEvaluation<CpuBackend, BitReversedOrder> {
    let cpu_coords = SecureColumnByCoords {
        columns: std::array::from_fn(|i| eval.values.columns[i].to_cpu()),
    };
    SecureEvaluation::new(eval.domain, cpu_coords)
}

fn secure_eval_from_cpu(eval: &SecureEvaluation<CpuBackend, BitReversedOrder>) -> SecureEvaluation<CudaBackend, BitReversedOrder> {
    let gpu_coords = SecureColumnByCoords {
        columns: std::array::from_fn(|i| eval.values.columns[i].iter().copied().collect()),
    };
    SecureEvaluation::new(eval.domain, gpu_coords)
}

fn qm31_to_arr(v: SecureField) -> [u32; 4] {
    let arr = v.to_m31_array();
    [arr[0].0, arr[1].0, arr[2].0, arr[3].0]
}
