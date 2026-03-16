//! MerkleOps: GPU-accelerated Merkle tree commitment via Blake2s.
//!
//! The leaf layer (largest, most expensive) runs on GPU via cuda_merkle_hash_leaves.
//! Internal layers use CPU fallback since they halve each level and are small.
//! TODO: GPU kernel for internal layers with column data.

use stwo_prover::core::backend::{Col, Column};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::vcs::blake2_hash::Blake2sHash;
use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use stwo_prover::core::vcs::ops::{MerkleHasher, MerkleOps};

use vortexstark::cuda::ffi;
use vortexstark::device::DeviceBuffer;

use super::CudaBackend;
use super::column::CudaColumn;

impl MerkleOps<Blake2sMerkleHasher> for CudaBackend {
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
        // (internal layers are small, this is rarely the bottleneck)
        commit_cpu(n, prev_layer, columns)
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
fn commit_cpu(
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
            Blake2sMerkleHasher::hash_node(children, &col_vals)
        })
        .collect();

    hashes.into_iter().collect()
}
