//! GPU Merkle tree commitment using Blake2s.
//!
//! Builds a complete binary Merkle tree entirely on the GPU:
//! 1. Hash leaves (column values → leaf hashes) in parallel
//! 2. Hash internal nodes layer by layer (bottom-up)
//!
//! The tree is stored as a flat array: layer 0 = root, layer log_n = leaves.

use crate::cuda::ffi;
use crate::device::DeviceBuffer;

/// 32-byte Blake2s hash, stored as 8 × u32.
pub const HASH_WORDS: usize = 8;

/// A Merkle tree committed on GPU. Stores all layers.
pub struct MerkleTree {
    /// Tree layers, from root (index 0) to leaves (index log_n).
    /// Each layer is a DeviceBuffer of HASH_WORDS * num_nodes u32.
    pub layers: Vec<DeviceBuffer<u32>>,
    pub log_n_leaves: u32,
}

impl MerkleTree {
    /// Build a Merkle tree from column data on GPU.
    ///
    /// `columns` are device buffers, each containing `n_leaves` M31 values.
    /// Returns the complete tree with all layers on GPU.
    pub fn commit(columns: &[DeviceBuffer<u32>], log_n_leaves: u32) -> Self {
        let n_leaves = 1u32 << log_n_leaves;
        let n_cols = columns.len() as u32;

        // Build device pointer array for leaf hashing kernel
        let col_ptrs: Vec<*const u32> = columns.iter().map(|c| c.as_ptr()).collect();
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

        // Allocate leaf hashes
        let mut d_leaves = DeviceBuffer::<u32>::alloc((n_leaves as usize) * HASH_WORDS);

        // Hash leaves on GPU
        unsafe {
            ffi::cuda_merkle_hash_leaves(
                d_col_ptrs.as_ptr() as *const *const u32,
                d_leaves.as_mut_ptr(),
                n_cols,
                n_leaves,
            );
        }

        // Build tree layers bottom-up
        let mut layers = Vec::with_capacity(log_n_leaves as usize + 1);

        // We'll collect layers in reverse (leaves first, then shrink to root)
        let mut current = d_leaves;
        let mut current_size = n_leaves;
        let mut all_layers = vec![];

        while current_size > 1 {
            let parent_size = current_size / 2;
            let mut d_parents = DeviceBuffer::<u32>::alloc((parent_size as usize) * HASH_WORDS);

            unsafe {
                ffi::cuda_merkle_hash_nodes(
                    current.as_ptr(),
                    d_parents.as_mut_ptr(),
                    parent_size,
                );
            }

            all_layers.push(current);
            current = d_parents;
            current_size = parent_size;
        }

        // No explicit sync needed — root() calls to_host() which syncs via cudaMemcpy D2H

        // Current is the root (1 hash)
        // Layers: root first, leaves last
        layers.push(current); // root
        for layer in all_layers.into_iter().rev() {
            layers.push(layer);
        }
        // Wait, we want layers[0] = root, layers[log_n] = leaves
        // all_layers was collected leaves→root direction, we reversed it
        // Actually let me restructure: all_layers has [leaves, n/2 nodes, n/4 nodes, ...]
        // After reversal: [..., n/4, n/2, leaves]
        // Then we push root first, then reversed = [root, n/4, ..., n/2, leaves]
        // That's wrong. Let me fix the ordering.

        // Redo: collect properly
        // layers[0] = root (1 hash), layers[1] = 2 hashes, ..., layers[log_n] = n_leaves hashes
        // We built: all_layers = [leaves, layer_{log_n-1}, ..., layer_1]
        // current = root
        // Want: [root, layer_1, ..., layer_{log_n-1}, leaves]
        // = [current] + all_layers.reverse()

        // The code above already does this correctly since we reverse.
        // layers = [root] + reversed(all_layers)
        // = [root, layer_1, ..., layer_{log_n-1}, leaves]

        Self {
            layers,
            log_n_leaves,
        }
    }

    /// Get the root hash (downloaded to host).
    pub fn root(&self) -> [u32; HASH_WORDS] {
        let host = self.layers[0].to_host();
        let mut root = [0u32; HASH_WORDS];
        root.copy_from_slice(&host[..HASH_WORDS]);
        root
    }

    /// Build a Merkle tree and return only the root hash (no layer storage).
    /// Skips storing intermediate layers, freeing memory sooner.
    pub fn commit_root_only<C: AsRef<DeviceBuffer<u32>>>(columns: &[C], log_n_leaves: u32) -> [u32; HASH_WORDS] {
        let n_leaves = 1u32 << log_n_leaves;
        let n_cols = columns.len() as u32;

        let col_ptrs: Vec<*const u32> = columns.iter().map(|c| c.as_ref().as_ptr()).collect();
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

        let mut current = DeviceBuffer::<u32>::alloc((n_leaves as usize) * HASH_WORDS);
        unsafe {
            ffi::cuda_merkle_hash_leaves(
                d_col_ptrs.as_ptr() as *const *const u32,
                current.as_mut_ptr(),
                n_cols,
                n_leaves,
            );
        }

        let mut current_size = n_leaves;
        while current_size > 1 {
            let parent_size = current_size / 2;
            let mut parents = DeviceBuffer::<u32>::alloc((parent_size as usize) * HASH_WORDS);
            unsafe {
                ffi::cuda_merkle_hash_nodes(
                    current.as_ptr(),
                    parents.as_mut_ptr(),
                    parent_size,
                );
            }
            current = parents;
            current_size = parent_size;
        }

        // cudaMemcpy D2H in to_host() implicitly syncs the default stream
        let host = current.to_host();
        let mut root = [0u32; HASH_WORDS];
        root.copy_from_slice(&host[..HASH_WORDS]);
        root
    }

    /// Build Merkle root from 4-column SoA data (SecureColumn / QM31).
    /// Uses single-kernel shared-memory reduction for small trees (≤ 2048 leaves),
    /// or fused leaf+merge + multi-launch tree for larger trees.
    pub fn commit_root_soa4(
        col0: &DeviceBuffer<u32>,
        col1: &DeviceBuffer<u32>,
        col2: &DeviceBuffer<u32>,
        col3: &DeviceBuffer<u32>,
        log_n_leaves: u32,
    ) -> [u32; HASH_WORDS] {
        let n_leaves = 1u32 << log_n_leaves;

        // Small tree: single kernel does everything (leaf hash + full tree reduction)
        if n_leaves <= 2048 {
            let mut d_root = DeviceBuffer::<u32>::alloc(HASH_WORDS);
            unsafe {
                ffi::cuda_merkle_commit_small_soa4(
                    col0.as_ptr(), col1.as_ptr(),
                    col2.as_ptr(), col3.as_ptr(),
                    d_root.as_mut_ptr(),
                    n_leaves,
                );
            }
            let host = d_root.to_host();
            let mut root = [0u32; HASH_WORDS];
            root.copy_from_slice(&host[..HASH_WORDS]);
            return root;
        }

        // Large tree: fused leaf+merge kernel + multi-launch node hashing
        let mut current_size = n_leaves / 2;
        let mut current = DeviceBuffer::<u32>::alloc((current_size as usize) * HASH_WORDS);
        unsafe {
            ffi::cuda_merkle_hash_leaves_merge_soa4(
                col0.as_ptr(), col1.as_ptr(),
                col2.as_ptr(), col3.as_ptr(),
                current.as_mut_ptr(),
                n_leaves,
            );
        }

        // Continue hashing up the tree
        while current_size > 1 {
            let parent_size = current_size / 2;
            let mut parents = DeviceBuffer::<u32>::alloc((parent_size as usize) * HASH_WORDS);
            unsafe {
                ffi::cuda_merkle_hash_nodes(
                    current.as_ptr(),
                    parents.as_mut_ptr(),
                    parent_size,
                );
            }
            current = parents;
            current_size = parent_size;
        }

        let host = current.to_host();
        let mut root = [0u32; HASH_WORDS];
        root.copy_from_slice(&host[..HASH_WORDS]);
        root
    }

    /// Generate a Merkle authentication path for leaf at `index`.
    /// Returns log_n sibling hashes (each HASH_WORDS u32).
    pub fn auth_path(&self, index: usize) -> Vec<[u32; HASH_WORDS]> {
        let mut path = Vec::with_capacity(self.log_n_leaves as usize);
        let mut idx = index;

        // Walk from leaves to root
        for layer_idx in (1..=self.log_n_leaves as usize).rev() {
            let sibling_idx = idx ^ 1;
            let layer_host = self.layers[layer_idx].to_host();
            let mut hash = [0u32; HASH_WORDS];
            let start = sibling_idx * HASH_WORDS;
            hash.copy_from_slice(&layer_host[start..start + HASH_WORDS]);
            path.push(hash);
            idx /= 2;
        }

        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::m31::P;

    #[test]
    fn test_merkle_commit_deterministic() {
        let log_n = 4u32;
        let n = 1usize << log_n;

        // Create some column data
        let col1: Vec<u32> = (0..n).map(|i| (i as u32 * 7 + 3) % P).collect();
        let col2: Vec<u32> = (0..n).map(|i| (i as u32 * 13 + 5) % P).collect();

        let d_col1 = DeviceBuffer::from_host(&col1);
        let d_col2 = DeviceBuffer::from_host(&col2);

        let tree1 = MerkleTree::commit(&[d_col1, d_col2], log_n);
        let root1 = tree1.root();

        // Same data should give same root
        let d_col1b = DeviceBuffer::from_host(&col1);
        let d_col2b = DeviceBuffer::from_host(&col2);
        let tree2 = MerkleTree::commit(&[d_col1b, d_col2b], log_n);
        let root2 = tree2.root();

        assert_eq!(root1, root2, "Merkle tree is not deterministic");
    }

    #[test]
    fn test_merkle_different_data_different_root() {
        let log_n = 4u32;
        let n = 1usize << log_n;

        let col1: Vec<u32> = (0..n).map(|i| (i as u32) % P).collect();
        let col2: Vec<u32> = (0..n).map(|i| (i as u32 + 1) % P).collect();

        let d1 = DeviceBuffer::from_host(&col1);
        let d2 = DeviceBuffer::from_host(&col2);

        let tree1 = MerkleTree::commit(&[d1], log_n);
        let tree2 = MerkleTree::commit(&[d2], log_n);

        assert_ne!(tree1.root(), tree2.root(), "Different data should give different roots");
    }

    #[test]
    fn test_merkle_auth_path_length() {
        let log_n = 6u32;
        let n = 1usize << log_n;

        let col: Vec<u32> = (0..n).map(|i| (i as u32 * 3) % P).collect();
        let d_col = DeviceBuffer::from_host(&col);

        let tree = MerkleTree::commit(&[d_col], log_n);
        let path = tree.auth_path(0);

        assert_eq!(path.len(), log_n as usize, "Auth path should have log_n elements");
    }

    #[test]
    fn test_merkle_tree_layers() {
        let log_n = 4u32;
        let n = 1usize << log_n;

        let col: Vec<u32> = (0..n).map(|i| (i as u32) % P).collect();
        let d_col = DeviceBuffer::from_host(&col);

        let tree = MerkleTree::commit(&[d_col], log_n);

        // Should have log_n + 1 layers
        assert_eq!(tree.layers.len(), (log_n + 1) as usize);

        // Root layer has 1 hash (8 words)
        assert_eq!(tree.layers[0].len(), HASH_WORDS);

        // Leaf layer has n hashes
        assert_eq!(tree.layers[log_n as usize].len(), n * HASH_WORDS);
    }
}
