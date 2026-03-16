//! GkrOps: GKR lookup protocol.
//!
//! CPU fallback implementation — downloads MLE data, runs GKR logic on CPU,
//! uploads results. GKR is not the proving bottleneck (NTT/FRI/Merkle are),
//! so CPU is acceptable for v1.
//!
//! TODO: GPU kernels for gen_eq_evals and next_layer when profiling shows need.

use stwo::prover::backend::{Column, CpuBackend};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::lookups::gkr_prover::{GkrMultivariatePolyOracle, GkrOps, Layer};
use stwo::prover::lookups::mle::{Mle, MleOps};
use stwo::prover::lookups::utils::UnivariatePoly;

use super::CudaBackend;

impl MleOps<BaseField> for CudaBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        // Download to CPU, run CPU MleOps, upload back
        let cpu_data = mle.to_cpu();
        let cpu_mle = Mle::new(cpu_data);
        let cpu_result = CpuBackend::fix_first_variable(cpu_mle, assignment);
        let result_data: Vec<SecureField> = cpu_result.to_cpu();
        Mle::new(result_data.into_iter().collect())
    }
}

impl MleOps<SecureField> for CudaBackend {
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        let cpu_data = mle.to_cpu();
        let cpu_mle = Mle::new(cpu_data);
        let cpu_result = CpuBackend::fix_first_variable(cpu_mle, assignment);
        let result_data: Vec<SecureField> = cpu_result.to_cpu();
        Mle::new(result_data.into_iter().collect())
    }
}

impl GkrOps for CudaBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        // CPU generation, upload to GPU
        let cpu_result = CpuBackend::gen_eq_evals(y, v);
        let data: Vec<SecureField> = cpu_result.to_cpu();
        Mle::new(data.into_iter().collect())
    }

    fn next_layer(_layer: &Layer<Self>) -> Layer<Self> {
        // Convert GPU layer → CPU layer, compute, convert back
        // This requires downloading MLE data and re-uploading.
        // For now, use todo!() — this needs per-variant conversion logic.
        todo!("GkrOps::next_layer — needs Layer<CudaBackend> → Layer<CpuBackend> conversion")
    }

    fn sum_as_poly_in_first_variable(
        _h: &GkrMultivariatePolyOracle<'_, Self>,
        _claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        // This operates on GkrMultivariatePolyOracle which references Layer<CudaBackend>.
        // Converting the oracle to CPU form requires deep access to its internals.
        todo!("GkrOps::sum_as_poly_in_first_variable — needs oracle conversion")
    }
}
