//! MerkleOps: GPU-accelerated Merkle tree commitment via Blake2s.
//!
//! Implements both the legacy MerkleOps (for vcs) and the new MerkleOpsLifted
//! (for vcs_lifted) + PackLeavesOps traits.
//!
//! The leaf layer (largest, most expensive) runs on GPU via cuda_merkle_hash_leaves.
//! Internal layers use CPU fallback since they halve each level and are small.

use stwo::prover::backend::{Col, Column};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SECURE_EXTENSION_DEGREE;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasherGeneric;
use stwo::core::vcs_lifted::verifier::PACKED_LEAF_SIZE;
use stwo::prover::vcs_lifted::ops::{MerkleOpsLifted, PackLeavesOps};

use vortexstark::cuda::ffi;
use vortexstark::device::DeviceBuffer;

use super::CudaBackend;
use super::column::CudaColumn;

impl<const IS_M31_OUTPUT: bool> MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>> for CudaBackend {
    fn build_leaves(
        columns: &[&Col<Self, BaseField>],
        lifting_log_size: u32,
    ) -> Col<Self, Blake2sHash> {
        // CPU fallback: download, compute, upload.
        eprintln!("[CUDA] build_leaves: {} columns, lifting_log_size={lifting_log_size}", columns.len());
        let cpu_columns: Vec<Vec<BaseField>> = columns.iter().map(|c| c.to_cpu()).collect();
        let cpu_col_refs: Vec<&Vec<BaseField>> = cpu_columns.iter().collect();
        let cpu_result = <stwo::prover::backend::CpuBackend as MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>>::build_leaves(
            &cpu_col_refs, lifting_log_size,
        );
        eprintln!("[CUDA] build_leaves done: {} hashes, uploading to GPU...", cpu_result.len());
        let result: CudaColumn<Blake2sHash> = cpu_result.into_iter().collect();
        eprintln!("[CUDA] build_leaves upload done");
        result
    }

    fn build_next_layer(prev_layer: &Col<Self, Blake2sHash>) -> Col<Self, Blake2sHash> {
        let n = prev_layer.len() / 2;
        if n == 0 {
            return CudaColumn::from_device_buffer(DeviceBuffer::<u32>::alloc(0), 0);
        }

        // GPU merge path: hash pairs of children
        let mut d_parents = DeviceBuffer::<u32>::alloc(n * 8);
        unsafe {
            ffi::cuda_merkle_hash_nodes(
                prev_layer.buf.as_ptr(),
                d_parents.as_mut_ptr(),
                n as u32,
            );
            ffi::cuda_device_sync();
        }
        CudaColumn::from_device_buffer(d_parents, n)

    }
}

impl PackLeavesOps for CudaBackend {
    fn pack_leaves_input(
        values: &[Col<Self, BaseField>; SECURE_EXTENSION_DEGREE],
    ) -> [Col<Self, BaseField>; SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE] {
        // CPU fallback: download, pack, upload.
        let cpu_values: [Vec<BaseField>; SECURE_EXTENSION_DEGREE] =
            std::array::from_fn(|i| values[i].to_cpu());
        let cpu_result = <stwo::prover::backend::CpuBackend as PackLeavesOps>::pack_leaves_input(&cpu_values);
        std::array::from_fn(|i| cpu_result[i].iter().copied().collect())
    }
}

// Poseidon252 MerkleOpsLifted: CPU fallback (Poseidon is not GPU-accelerated yet).
#[cfg(not(target_arch = "wasm32"))]
mod poseidon_merkle {
    use stwo::prover::backend::{Col, Column};
    use stwo::core::fields::m31::BaseField;
    use stwo::core::vcs_lifted::poseidon252_merkle::Poseidon252MerkleHasher;
    use stwo::prover::vcs_lifted::ops::MerkleOpsLifted;
    use starknet_ff::FieldElement as FieldElement252;

    use super::CudaBackend;
    use super::CudaColumn;

    impl MerkleOpsLifted<Poseidon252MerkleHasher> for CudaBackend {
        fn build_leaves(
            columns: &[&Col<Self, BaseField>],
            lifting_log_size: u32,
        ) -> Col<Self, FieldElement252> {
            // CPU fallback: download, compute, upload.
            let cpu_columns: Vec<Vec<BaseField>> = columns.iter().map(|c| c.to_cpu()).collect();
            let cpu_col_refs: Vec<&Vec<BaseField>> = cpu_columns.iter().collect();
            let cpu_result = <stwo::prover::backend::CpuBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_leaves(
                &cpu_col_refs, lifting_log_size,
            );
            cpu_result.into_iter().collect()
        }

        fn build_next_layer(prev_layer: &Col<Self, FieldElement252>) -> Col<Self, FieldElement252> {
            // CPU fallback: download, compute, upload.
            let cpu_data: Vec<FieldElement252> = prev_layer.to_cpu();
            let cpu_col: Vec<FieldElement252> = cpu_data;
            let cpu_result = <stwo::prover::backend::CpuBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_next_layer(&cpu_col);
            cpu_result.into_iter().collect()
        }
    }
}

// Legacy MerkleOps removed — using MerkleOpsLifted exclusively.
// The old non-lifted VCS path is not used by stwo-cairo.

/// GPU leaf hashing: hash column values into leaf nodes.
fn commit_leaves_gpu(
    n: usize,
    columns: &[&CudaColumn<BaseField>],
) -> CudaColumn<Blake2sHash> {
    let n_cols = columns.len() as u32;

    // Build array of column device pointers
    let col_ptrs: Vec<*const u32> = columns.iter().map(|c| c.buf.as_ptr()).collect();
    let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

    // Output: n hashes × 8 u32s each
    let mut d_hashes = DeviceBuffer::<u32>::alloc(n * 8);

    unsafe {
        ffi::cuda_merkle_hash_leaves(
            d_col_ptrs.as_ptr() as *const *const u32,
            d_hashes.as_mut_ptr(),
            n_cols,
            n as u32,
        );
        ffi::cuda_device_sync();
    }

    CudaColumn::from_device_buffer(d_hashes, n)
}

/// GPU merge: hash pairs of children into parent nodes.
fn commit_merge_gpu(
    n: usize,
    prev_layer: &CudaColumn<Blake2sHash>,
) -> CudaColumn<Blake2sHash> {
    let mut d_parents = DeviceBuffer::<u32>::alloc(n * 8);

    unsafe {
        ffi::cuda_merkle_hash_nodes(
            prev_layer.buf.as_ptr(),
            d_parents.as_mut_ptr(),
            n as u32,
        );
        ffi::cuda_device_sync();
    }

    CudaColumn::from_device_buffer(d_parents, n)
}

// Legacy commit_cpu removed — MerkleOpsLifted handles all cases via build_leaves/build_next_layer.
