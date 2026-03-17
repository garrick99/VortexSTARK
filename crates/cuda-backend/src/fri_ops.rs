//! FriOps: FRI folding — CPU fallback with cached twiddles.
//!
//! GPU FRI fold kernels use VortexSTARK's twiddle format which differs from
//! stwo's coset conventions. Using CPU fallback until the fold twiddle
//! computation is aligned (same approach as the NTT twiddle fix).
//! Twiddle precomputation is cached globally to avoid redundant work.

use stwo::core::fields::qm31::SecureField;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::backend::{Column, CpuBackend};
use stwo::prover::fri::FriOps;
use stwo::prover::poly::circle::SecureEvaluation;
use stwo::prover::line::LineEvaluation;
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::poly::BitReversedOrder;

use super::CudaBackend;
use super::column::CudaColumn;
use super::poly_ops::cached_cpu_twiddle_tree;

impl FriOps for CudaBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
        fold_step: u32,
    ) -> LineEvaluation<Self> {
        let cpu_eval = line_eval_to_cpu(eval);
        let cpu_twiddles = cached_cpu_twiddle_tree(&eval.domain().coset());
        let cpu_result = CpuBackend::fold_line(&cpu_eval, alpha, &cpu_twiddles, fold_step);
        line_eval_from_cpu(&cpu_result)
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) {
        let mut cpu_dst = line_eval_to_cpu(dst);
        let cpu_src = secure_eval_to_cpu(src);
        let cpu_twiddles = cached_cpu_twiddle_tree(&src.domain.half_coset);
        CpuBackend::fold_circle_into_line(&mut cpu_dst, &cpu_src, alpha, &cpu_twiddles);
        *dst = line_eval_from_cpu(&cpu_dst);
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        let cpu_eval = secure_eval_to_cpu(eval);
        let (cpu_result, lambda) = CpuBackend::decompose(&cpu_eval);
        (secure_eval_from_cpu(&cpu_result), lambda)
    }
}

fn line_eval_to_cpu(eval: &LineEvaluation<CudaBackend>) -> LineEvaluation<CpuBackend> {
    let cpu_coords = SecureColumnByCoords {
        columns: std::array::from_fn(|i| eval.values.columns[i].to_cpu()),
    };
    LineEvaluation::new(eval.domain(), cpu_coords)
}

fn line_eval_from_cpu(eval: &LineEvaluation<CpuBackend>) -> LineEvaluation<CudaBackend> {
    let gpu_coords = SecureColumnByCoords {
        columns: std::array::from_fn(|i| eval.values.columns[i].iter().copied().collect()),
    };
    LineEvaluation::new(eval.domain(), gpu_coords)
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
