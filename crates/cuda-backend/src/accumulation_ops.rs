//! AccumulationOps: column accumulation on GPU.

use num_traits::One;
use stwo::prover::AccumulationOps;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::secure_column::SecureColumnByCoords;
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

    fn lift_and_accumulate(
        cols: Vec<SecureColumnByCoords<Self>>,
    ) -> Option<SecureColumnByCoords<Self>> {
        if cols.is_empty() {
            return None;
        }

        // GPU lift_and_accumulate.
        // Algorithm (matches stwo CPU impl):
        //   curr = zeros(2)
        //   for each col (ascending size):
        //       log_ratio = col.log_size - curr.log_size
        //       for i in 0..col.len(): col[i] += curr[src_idx(i, log_ratio)]
        //       curr = col

        const INITIAL_SIZE: usize = 2;
        // Build initial curr: 4 channels of zeros(2) = SecureColumnByCoords::zeros(2)
        let mut curr: SecureColumnByCoords<Self> = SecureColumnByCoords::zeros(INITIAL_SIZE);

        for mut col in cols.into_iter() {
            let col_n = col.columns[0].len;
            let curr_n = curr.columns[0].len;
            assert!(col_n >= INITIAL_SIZE);
            let log_ratio = (col_n.ilog2() - curr_n.ilog2()) as u32;

            // Process each of the 4 coordinate channels independently
            for c in 0..4 {
                unsafe {
                    ffi::cuda_accumulate_lift(
                        col.columns[c].buf.as_mut_ptr(),
                        curr.columns[c].buf.as_ptr(),
                        col_n as u32,
                        log_ratio,
                    );
                }
            }
            unsafe { ffi::cuda_device_sync(); }

            curr = col;
        }

        Some(curr)
    }
}
