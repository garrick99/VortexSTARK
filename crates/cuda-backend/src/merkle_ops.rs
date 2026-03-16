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
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
use stwo::core::vcs_lifted::verifier::PACKED_LEAF_SIZE;
use stwo::prover::vcs_lifted::ops::{MerkleOpsLifted, PackLeavesOps};

use vortexstark::cuda::ffi;
use vortexstark::device::DeviceBuffer;

use super::CudaBackend;
use super::column::CudaColumn;

impl MerkleOpsLifted<Blake2sMerkleHasher> for CudaBackend {
    fn build_leaves(
        columns: &[&Col<Self, BaseField>],
        lifting_log_size: u32,
    ) -> Col<Self, Blake2sHash> {
        // CPU fallback: download, compute, upload.
        let cpu_columns: Vec<Vec<BaseField>> = columns.iter().map(|c| c.to_cpu()).collect();
        let cpu_col_refs: Vec<&Vec<BaseField>> = cpu_columns.iter().collect();
        let cpu_result = <stwo::prover::backend::CpuBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
            &cpu_col_refs, lifting_log_size,
        );
        cpu_result.into_iter().collect()
    }

    fn build_next_layer(prev_layer: &Col<Self, Blake2sHash>) -> Col<Self, Blake2sHash> {
        let n = prev_layer.len() / 2;
        if n == 0 {
            return CudaColumn::from_device_buffer(DeviceBuffer::<u32>::alloc(0), 0);
        }

        // Try GPU merge path for pure hash pairing
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

// Also keep the legacy MerkleOps for the non-lifted VCS path, which may still be
// referenced by some code paths.
use stwo::core::vcs::blake2_merkle::Blake2sMerkleHasherGeneric;
use stwo::core::vcs::MerkleHasher;
use stwo::prover::vcs::ops::MerkleOps;

impl<const IS_M31_OUTPUT: bool> MerkleOps<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>> for CudaBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, Blake2sHash> {
        let n = 1usize << log_size;

        if prev_layer.is_none() && !columns.is_empty() {
            // Leaf layer — GPU path
            return commit_leaves_gpu(n, columns);
        }

        if let Some(prev) = prev_layer {
            if columns.is_empty() {
                // Pure merge layer (no column data) — GPU path
                return commit_merge_gpu(n, prev);
            }
        }

        // General case: children + columns — CPU fallback
        commit_cpu::<IS_M31_OUTPUT>(n, prev_layer, columns)
    }
}

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

/// CPU fallback for internal layers with both children and column data.
fn commit_cpu<const IS_M31_OUTPUT: bool>(
    n: usize,
    prev_layer: Option<&CudaColumn<Blake2sHash>>,
    columns: &[&CudaColumn<BaseField>],
) -> CudaColumn<Blake2sHash> {
    // Download everything to CPU
    let prev_cpu: Option<Vec<Blake2sHash>> = prev_layer.map(|p| p.to_cpu());
    let cols_cpu: Vec<Vec<BaseField>> = columns.iter().map(|c| c.to_cpu()).collect();

    let hashes: Vec<Blake2sHash> = (0..n)
        .map(|i| {
            let children = prev_cpu.as_ref().map(|p| (p[2 * i], p[2 * i + 1]));
            let col_vals: Vec<BaseField> = cols_cpu.iter().map(|c| c[i]).collect();
            Blake2sMerkleHasherGeneric::<IS_M31_OUTPUT>::hash_node(children, &col_vals)
        })
        .collect();

    hashes.into_iter().collect()
}
