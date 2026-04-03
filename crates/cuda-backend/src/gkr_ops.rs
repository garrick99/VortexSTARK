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

/// Convert a `Layer<CpuBackend>` to `Layer<CudaBackend>` by uploading MLE data to GPU.
fn layer_to_cuda(layer: Layer<CpuBackend>) -> Layer<CudaBackend> {
    match layer {
        Layer::GrandProduct(mle) => {
            let data: Vec<SecureField> = mle.to_cpu();
            Layer::GrandProduct(Mle::new(data.into_iter().collect()))
        }
        Layer::LogUpGeneric { numerators, denominators } => {
            let num: Vec<SecureField> = numerators.to_cpu();
            let den: Vec<SecureField> = denominators.to_cpu();
            Layer::LogUpGeneric {
                numerators:   Mle::new(num.into_iter().collect()),
                denominators: Mle::new(den.into_iter().collect()),
            }
        }
        Layer::LogUpMultiplicities { numerators, denominators } => {
            let num: Vec<BaseField>   = numerators.to_cpu();
            let den: Vec<SecureField> = denominators.to_cpu();
            Layer::LogUpMultiplicities {
                numerators:   Mle::new(num.into_iter().collect()),
                denominators: Mle::new(den.into_iter().collect()),
            }
        }
        Layer::LogUpSingles { denominators } => {
            let den: Vec<SecureField> = denominators.to_cpu();
            Layer::LogUpSingles {
                denominators: Mle::new(den.into_iter().collect()),
            }
        }
    }
}

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

    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        // CPU fallback: download MLE data, run CPU GKR logic, upload result.
        // GKR is called once per layer reduction (O(log n) times), so the
        // download/upload overhead is acceptable compared to NTT/FRI cost.
        let cpu_layer = layer.to_cpu();
        let next_cpu = CpuBackend::next_layer(&cpu_layer);
        layer_to_cuda(next_cpu)
    }

    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        // CPU fallback: GkrMultivariatePolyOracle::to_cpu() downloads eq_evals
        // and input_layer, then CpuBackend evaluates the sum polynomial.
        let cpu_oracle = h.to_cpu();
        CpuBackend::sum_as_poly_in_first_variable(&cpu_oracle, claim)
    }
}
