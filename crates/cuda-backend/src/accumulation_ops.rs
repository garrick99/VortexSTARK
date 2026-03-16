//! AccumulationOps: column accumulation on GPU.

use num_traits::One;
use stwo_prover::core::air::accumulation::AccumulationOps;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;

use vortexstark::cuda::ffi;
use super::CudaBackend;

impl AccumulationOps for CudaBackend {
    fn accumulate(
        column: &mut SecureColumnByCoords<Self>,
        other: &SecureColumnByCoords<Self>,
    ) {
        // Element-wise M31 addition on each of the 4 coordinate columns.
        // SecureColumnByCoords has 4 × Col<Self, BaseField> = 4 × CudaColumn<BaseField>.
        for i in 0..4 {
            assert_eq!(column.columns[i].len, other.columns[i].len);
            let n = column.columns[i].len;
            if n > 0 {
                unsafe {
                    ffi::cuda_m31_add(
                        column.columns[i].buf.as_ptr(),
                        other.columns[i].buf.as_ptr(),
                        column.columns[i].buf.as_mut_ptr(),
                        n as u32,
                    );
                }
            }
        }
        unsafe { ffi::cuda_device_sync(); }
    }

    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
        // Small (n_powers ~ 100), CPU is fine.
        let mut powers = Vec::with_capacity(n_powers);
        let mut current = SecureField::one();
        for _ in 0..n_powers {
            powers.push(current);
            current = current * felt;
        }
        powers
    }
}
