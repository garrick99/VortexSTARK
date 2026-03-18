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
        let t0 = std::time::Instant::now();
        let n_leaves = 1u32 << lifting_log_size;

        if columns.is_empty() {
            // Empty columns: return default hashes (CPU, trivial)
            let cpu_result = <stwo::prover::backend::CpuBackend as MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>>::build_leaves(
                &[], lifting_log_size,
            );
            return cpu_result.into_iter().collect();
        }

        // Build the hash schedule: group columns by log_size (ascending),
        // chunk by 16 (one Blake2s compression block per chunk).
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct LeafHashChunk {
            col_indices: [u32; 16],
            n_cols: u32,
            log_size: u32,
        }

        // Columns must be sorted by ascending log_size (stwo guarantees this).
        let mut schedule: Vec<LeafHashChunk> = Vec::new();
        let mut col_idx = 0usize;

        while col_idx < columns.len() {
            let log_size = columns[col_idx].len().ilog2();
            // Collect all columns at this log_size, chunked by 16.
            let group_start = col_idx;
            while col_idx < columns.len() && columns[col_idx].len().ilog2() == log_size {
                col_idx += 1;
            }
            let group = &columns[group_start..col_idx];

            for chunk in group.chunks(16) {
                let mut entry = LeafHashChunk {
                    col_indices: [0; 16],
                    n_cols: chunk.len() as u32,
                    log_size,
                };
                for (i, _) in chunk.iter().enumerate() {
                    entry.col_indices[i] = (group_start + (col_idx - group_start - group.len()) + i + (chunk.as_ptr() as usize - group.as_ptr() as usize) / std::mem::size_of::<&Col<Self, BaseField>>()) as u32;
                }
                schedule.push(entry);
            }
        }

        // Fix col_indices: we need absolute indices into the columns array.
        // Rebuild cleanly:
        schedule.clear();
        let mut abs_idx = 0usize;
        let mut i = 0;
        while i < columns.len() {
            let log_size = columns[i].len().ilog2();
            let group_start = i;
            while i < columns.len() && columns[i].len().ilog2() == log_size {
                i += 1;
            }
            for chunk_start in (group_start..i).step_by(16) {
                let chunk_end = std::cmp::min(chunk_start + 16, i);
                let mut entry = LeafHashChunk {
                    col_indices: [0; 16],
                    n_cols: (chunk_end - chunk_start) as u32,
                    log_size,
                };
                for j in 0..(chunk_end - chunk_start) {
                    entry.col_indices[j] = (chunk_start + j) as u32;
                }
                schedule.push(entry);
            }
        }

        eprintln!("[LEAVES-GPU] {} cols, lift={lifting_log_size}, {} chunks, {} leaves",
            columns.len(), schedule.len(), n_leaves);

        // Fall back to CPU for very large leaf counts (>16M).
        // The preprocessed trace (67M leaves) is a one-time cost.
        // TODO: fix the Blake2s slow-tests heap corruption (pre-existing Windows bug).
        if n_leaves > 16 * 1024 * 1024 {
            eprintln!("[LEAVES-GPU] Large leaf count, using CPU fallback");
            let cpu_columns: Vec<Vec<BaseField>> = columns.iter().map(|c| c.to_cpu()).collect();
            let cpu_col_refs: Vec<&Vec<BaseField>> = cpu_columns.iter().collect();
            let cpu_result = <stwo::prover::backend::CpuBackend as MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>>::build_leaves(
                &cpu_col_refs, lifting_log_size,
            );
            let result: CudaColumn<Blake2sHash> = cpu_result.into_iter().collect();
            eprintln!("[LEAVES-GPU] CPU fallback done in {:.3}s", t0.elapsed().as_secs_f64());
            return result;
        }

        // Collect device pointers for all columns.
        let col_ptrs: Vec<*const u32> = columns.iter().map(|c| c.buf.as_ptr()).collect();
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

        // Upload schedule to GPU as u32 words (18 words per chunk = 72 bytes).
        let schedule_words: Vec<u32> = unsafe {
            let ptr = schedule.as_ptr() as *const u32;
            let n_words = schedule.len() * (std::mem::size_of::<LeafHashChunk>() / 4);
            std::slice::from_raw_parts(ptr, n_words).to_vec()
        };
        let d_schedule = DeviceBuffer::<u32>::from_host(&schedule_words);

        // Allocate output: n_leaves * 8 u32s (Blake2s hash = 32 bytes = 8 words).
        let mut d_hashes = DeviceBuffer::<u32>::alloc(n_leaves as usize * 8);

        unsafe {
            ffi::cuda_build_leaves_lifted(
                d_col_ptrs.as_ptr() as *const *const u32,
                d_schedule.as_ptr() as *const u8,
                schedule.len() as u32,
                lifting_log_size,
                d_hashes.as_mut_ptr(),
                n_leaves,
            );
            ffi::cuda_device_sync();
            let err = ffi::cudaGetLastError();
            if err != 0 {
                eprintln!("[LEAVES-GPU] CUDA error {err} after build_leaves_lifted! Falling back to CPU.");
                let cpu_columns: Vec<Vec<BaseField>> = columns.iter().map(|c| c.to_cpu()).collect();
                let cpu_col_refs: Vec<&Vec<BaseField>> = cpu_columns.iter().collect();
                let cpu_result = <stwo::prover::backend::CpuBackend as MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>>::build_leaves(
                    &cpu_col_refs, lifting_log_size,
                );
                return cpu_result.into_iter().collect();
            }
        }

        // Keep schedule and col_ptrs alive until after GPU sync.
        drop(d_schedule);
        drop(d_col_ptrs);

        let result = CudaColumn::from_device_buffer(d_hashes, n_leaves as usize);

        eprintln!("[LEAVES-GPU] done: {} hashes in {:.3}s", n_leaves, t0.elapsed().as_secs_f64());
        result
    }

    fn build_next_layer(prev_layer: &Col<Self, Blake2sHash>) -> Col<Self, Blake2sHash> {
        let n = prev_layer.len() / 2;
        eprintln!("[CUDA] build_next_layer: n={n}");
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
