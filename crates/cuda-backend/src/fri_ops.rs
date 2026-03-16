//! FriOps: FRI folding.
//!
//! TEMPORARY: CPU fallback while investigating stale GPU pointer bug in FRI commit.
//! TODO: Restore GPU path after fixing the DeviceBuffer lifetime issue.

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::backend::{Column, CpuBackend};
use stwo::prover::fri::FriOps;
use stwo::prover::poly::circle::{SecureEvaluation, PolyOps};
use stwo::prover::line::LineEvaluation;
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::poly::BitReversedOrder;

use super::CudaBackend;
use super::column::CudaColumn;

impl FriOps for CudaBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
        fold_step: u32,
    ) -> LineEvaluation<Self> {
        // CPU fallback: download, fold on CPU, upload
        let cpu_eval = line_eval_to_cpu(eval);
        let cpu_twiddles = twiddles_to_cpu(twiddles);
        let cpu_result = CpuBackend::fold_line(&cpu_eval, alpha, &cpu_twiddles, fold_step);
        line_eval_from_cpu(&cpu_result)
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        let mut cpu_dst = line_eval_to_cpu(dst);
        let cpu_src = secure_eval_to_cpu(src);
        let cpu_twiddles = twiddles_to_cpu(twiddles);
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

// ---- Conversion helpers: CudaBackend ↔ CpuBackend ----

fn line_eval_to_cpu(eval: &LineEvaluation<CudaBackend>) -> LineEvaluation<CpuBackend> {
    let n = eval.values.columns[0].len();
    eprintln!("[FRI] line_eval_to_cpu: n={n}, domain={:?}", eval.domain());
    let cpu_coords = SecureColumnByCoords {
        columns: std::array::from_fn(|i| {
            let col = eval.values.columns[i].to_cpu();
            assert_eq!(col.len(), n, "column {i} length mismatch");
            col
        }),
    };
    // Verify a few values roundtrip
    if n > 0 {
        let v = cpu_coords.at(0);
        eprintln!("[FRI]   first value: {:?}", v.to_m31_array().map(|m| m.0));
    }
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

fn twiddles_to_cpu(twiddles: &TwiddleTree<CudaBackend>) -> TwiddleTree<CpuBackend> {
    // CPU twiddles are precomputed from the coset
    CpuBackend::precompute_twiddles(twiddles.root_coset)
}
