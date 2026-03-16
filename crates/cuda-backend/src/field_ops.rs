//! FieldOps: batch inverse for BaseField and SecureField on GPU.

use num_traits::{One, Zero};
use stwo_prover::core::fields::FieldOps;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::FieldExpOps;
use vortexstark::cuda::ffi;

use super::CudaBackend;
use super::column::CudaColumn;

impl FieldOps<BaseField> for CudaBackend {
    fn batch_inverse(column: &CudaColumn<BaseField>, dst: &mut CudaColumn<BaseField>) {
        assert_eq!(column.len, dst.len);
        unsafe {
            ffi::cuda_batch_inverse_m31(
                column.buf.as_ptr(),
                dst.buf.as_mut_ptr(),
                column.len as u32,
            );
            ffi::cuda_device_sync();
        }
    }
}

impl FieldOps<SecureField> for CudaBackend {
    fn batch_inverse(column: &CudaColumn<SecureField>, dst: &mut CudaColumn<SecureField>) {
        assert_eq!(column.len, dst.len);
        let n = column.len;

        // Download, Montgomery's trick on CPU, upload
        // TODO: QM31 batch inverse GPU kernel
        let cpu_vals = column.buf.to_host();
        let vals: Vec<SecureField> = cpu_vals.chunks_exact(4).map(|c| {
            SecureField::from_m31_array(std::array::from_fn(|i| {
                BaseField::from_u32_unchecked(c[i])
            }))
        }).collect();

        let mut prefix = Vec::with_capacity(n);
        let mut running = SecureField::one();
        for v in &vals {
            running = running * *v;
            prefix.push(running);
        }

        let mut inv = running.inverse();
        let mut inverses = vec![SecureField::zero(); n];
        for i in (0..n).rev() {
            if i > 0 {
                inverses[i] = inv * prefix[i - 1];
            } else {
                inverses[i] = inv;
            }
            inv = inv * vals[i];
        }

        let flat: Vec<u32> = inverses.iter().flat_map(|f| {
            let arr = f.to_m31_array();
            [arr[0].0, arr[1].0, arr[2].0, arr[3].0]
        }).collect();
        dst.buf = vortexstark::device::DeviceBuffer::from_host(&flat);
    }
}
