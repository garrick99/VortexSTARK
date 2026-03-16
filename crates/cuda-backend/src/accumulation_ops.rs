//! AccumulationOps: column accumulation on GPU.

use stwo_prover::core::air::accumulation::AccumulationOps;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;

use super::CudaBackend;

impl AccumulationOps for CudaBackend {
    fn accumulate(
        column: &mut SecureColumnByCoords<Self>,
        other: &SecureColumnByCoords<Self>,
    ) {
        todo!("AccumulationOps::accumulate — element-wise QM31 addition on GPU")
    }

    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
        // This is small (n_powers ~ 100), CPU is fine.
        let mut powers = Vec::with_capacity(n_powers);
        let mut current = SecureField::one();
        for _ in 0..n_powers {
            powers.push(current);
            current = current * felt;
        }
        powers
    }
}
