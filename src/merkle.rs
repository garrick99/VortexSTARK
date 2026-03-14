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
    /// Uses tiled processing for large trees to avoid allocating a full leaf hash buffer.
    pub fn commit_root_only<C: AsRef<DeviceBuffer<u32>>>(columns: &[C], log_n_leaves: u32) -> [u32; HASH_WORDS] {
        let n_leaves = 1u32 << log_n_leaves;
        let n_cols = columns.len() as u32;

        let col_ptrs: Vec<*const u32> = columns.iter().map(|c| c.as_ref().as_ptr()).collect();
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

        // Tiled path: process 1024 leaves per block in shared memory,
        // producing n_leaves/1024 subtree roots instead of n_leaves full hashes.
        // Reduces peak allocation from O(n × 32B) to O(n/1024 × 32B).
        const TILE_SIZE: u32 = 1024;
        if n_leaves >= TILE_SIZE {
            let n_subtrees = n_leaves / TILE_SIZE;
            let mut current = DeviceBuffer::<u32>::alloc((n_subtrees as usize) * HASH_WORDS);
            unsafe {
                ffi::cuda_merkle_tiled_generic(
                    d_col_ptrs.as_ptr() as *const *const u32,
                    current.as_mut_ptr(),
                    n_cols,
                    n_leaves,
                );
            }
            drop(d_col_ptrs);

            let mut current_size = n_subtrees;
            while current_size > 1024 {
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

            if current_size > 1 {
                let mut d_root = DeviceBuffer::<u32>::alloc(HASH_WORDS);
                unsafe {
                    ffi::cuda_merkle_reduce_to_root(
                        current.as_ptr(),
                        d_root.as_mut_ptr(),
                        current_size,
                    );
                }
                let host = d_root.to_host();
                let mut root = [0u32; HASH_WORDS];
                root.copy_from_slice(&host[..HASH_WORDS]);
                return root;
            } else {
                let host = current.to_host();
                let mut root = [0u32; HASH_WORDS];
                root.copy_from_slice(&host[..HASH_WORDS]);
                return root;
            }
        }

        // Small tree fallback: full leaf hash buffer (< 1024 leaves = 32KB, negligible)
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

        // Large tree: tiled kernel processes 1024 leaves/block in shared memory,
        // producing n_leaves/1024 subtree roots. Then reduce those to final root.
        // This replaces fused leaf+merge + ~9 hash_nodes launches with 1 tiled + 1-2 launches.
        const TILE_SIZE: u32 = 1024;
        let n_subtrees = n_leaves / TILE_SIZE;
        let mut current = DeviceBuffer::<u32>::alloc((n_subtrees as usize) * HASH_WORDS);
        unsafe {
            ffi::cuda_merkle_tiled_soa4(
                col0.as_ptr(), col1.as_ptr(),
                col2.as_ptr(), col3.as_ptr(),
                current.as_mut_ptr(),
                n_leaves,
            );
        }

        let mut current_size = n_subtrees;
        // Hash remaining node levels until ≤ 1024 for tail reduction
        while current_size > 1024 {
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

        if current_size > 1 {
            let mut d_root = DeviceBuffer::<u32>::alloc(HASH_WORDS);
            unsafe {
                ffi::cuda_merkle_reduce_to_root(
                    current.as_ptr(),
                    d_root.as_mut_ptr(),
                    current_size,
                );
            }
            let host = d_root.to_host();
            let mut root = [0u32; HASH_WORDS];
            root.copy_from_slice(&host[..HASH_WORDS]);
            root
        } else {
            let host = current.to_host();
            let mut root = [0u32; HASH_WORDS];
            root.copy_from_slice(&host[..HASH_WORDS]);
            root
        }
    }

    /// Build Merkle root from 4-column SoA data and also return the tile subtree roots.
    /// The subtree roots can be used later for targeted auth path computation without
    /// re-hashing all leaves.
    pub fn commit_root_soa4_with_subtrees(
        col0: &DeviceBuffer<u32>,
        col1: &DeviceBuffer<u32>,
        col2: &DeviceBuffer<u32>,
        col3: &DeviceBuffer<u32>,
        log_n_leaves: u32,
    ) -> ([u32; HASH_WORDS], Vec<[u32; HASH_WORDS]>) {
        let n_leaves = 1u32 << log_n_leaves;
        const TILE_SIZE: u32 = 1024;

        if n_leaves < TILE_SIZE {
            let root = Self::commit_root_soa4(col0, col1, col2, col3, log_n_leaves);
            return (root, vec![root]);
        }

        let n_subtrees = n_leaves / TILE_SIZE;
        let mut d_subtrees = DeviceBuffer::<u32>::alloc((n_subtrees as usize) * HASH_WORDS);
        unsafe {
            ffi::cuda_merkle_tiled_soa4(
                col0.as_ptr(), col1.as_ptr(),
                col2.as_ptr(), col3.as_ptr(),
                d_subtrees.as_mut_ptr(),
                n_leaves,
            );
        }

        // Download subtree roots to host BEFORE reducing to final root
        let subtree_host = d_subtrees.to_host();
        let subtree_roots: Vec<[u32; HASH_WORDS]> = (0..n_subtrees as usize)
            .map(|i| {
                let mut h = [0u32; HASH_WORDS];
                h.copy_from_slice(&subtree_host[i * HASH_WORDS..(i + 1) * HASH_WORDS]);
                h
            })
            .collect();

        // Reduce to final root
        let mut current = d_subtrees;
        let mut current_size = n_subtrees;
        while current_size > 1024 {
            let parent_size = current_size / 2;
            let mut parents = DeviceBuffer::<u32>::alloc((parent_size as usize) * HASH_WORDS);
            unsafe {
                ffi::cuda_merkle_hash_nodes(current.as_ptr(), parents.as_mut_ptr(), parent_size);
            }
            current = parents;
            current_size = parent_size;
        }

        let root = if current_size > 1 {
            let mut d_root = DeviceBuffer::<u32>::alloc(HASH_WORDS);
            unsafe {
                ffi::cuda_merkle_reduce_to_root(current.as_ptr(), d_root.as_mut_ptr(), current_size);
            }
            let host = d_root.to_host();
            let mut r = [0u32; HASH_WORDS];
            r.copy_from_slice(&host[..HASH_WORDS]);
            r
        } else {
            let host = current.to_host();
            let mut r = [0u32; HASH_WORDS];
            r.copy_from_slice(&host[..HASH_WORDS]);
            r
        };

        (root, subtree_roots)
    }

    /// Same as commit_root_only but also returns tile subtree roots for targeted auth paths.
    pub fn commit_root_only_with_subtrees<C: AsRef<DeviceBuffer<u32>>>(
        columns: &[C], log_n_leaves: u32,
    ) -> ([u32; HASH_WORDS], Vec<[u32; HASH_WORDS]>) {
        let n_leaves = 1u32 << log_n_leaves;
        let n_cols = columns.len() as u32;
        const TILE_SIZE: u32 = 1024;

        if n_leaves < TILE_SIZE {
            let root = Self::commit_root_only(columns, log_n_leaves);
            return (root, vec![root]);
        }

        let col_ptrs: Vec<*const u32> = columns.iter().map(|c| c.as_ref().as_ptr()).collect();
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

        let n_subtrees = n_leaves / TILE_SIZE;
        let mut d_subtrees = DeviceBuffer::<u32>::alloc((n_subtrees as usize) * HASH_WORDS);
        unsafe {
            ffi::cuda_merkle_tiled_generic(
                d_col_ptrs.as_ptr() as *const *const u32,
                d_subtrees.as_mut_ptr(),
                n_cols,
                n_leaves,
            );
        }
        drop(d_col_ptrs);

        let subtree_host = d_subtrees.to_host();
        let subtree_roots: Vec<[u32; HASH_WORDS]> = (0..n_subtrees as usize)
            .map(|i| {
                let mut h = [0u32; HASH_WORDS];
                h.copy_from_slice(&subtree_host[i * HASH_WORDS..(i + 1) * HASH_WORDS]);
                h
            })
            .collect();

        let mut current = d_subtrees;
        let mut current_size = n_subtrees;
        while current_size > 1024 {
            let parent_size = current_size / 2;
            let mut parents = DeviceBuffer::<u32>::alloc((parent_size as usize) * HASH_WORDS);
            unsafe {
                ffi::cuda_merkle_hash_nodes(current.as_ptr(), parents.as_mut_ptr(), parent_size);
            }
            current = parents;
            current_size = parent_size;
        }

        let root = if current_size > 1 {
            let mut d_root = DeviceBuffer::<u32>::alloc(HASH_WORDS);
            unsafe {
                ffi::cuda_merkle_reduce_to_root(current.as_ptr(), d_root.as_mut_ptr(), current_size);
            }
            let host = d_root.to_host();
            let mut r = [0u32; HASH_WORDS];
            r.copy_from_slice(&host[..HASH_WORDS]);
            r
        } else {
            let host = current.to_host();
            let mut r = [0u32; HASH_WORDS];
            r.copy_from_slice(&host[..HASH_WORDS]);
            r
        };

        (root, subtree_roots)
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

    /// Generate authentication paths for multiple indices efficiently.
    /// Downloads each layer once rather than once per query.
    pub fn batch_auth_paths(&self, indices: &[usize]) -> Vec<Vec<[u32; HASH_WORDS]>> {
        let depth = self.log_n_leaves as usize;
        let mut paths: Vec<Vec<[u32; HASH_WORDS]>> = indices
            .iter()
            .map(|_| Vec::with_capacity(depth))
            .collect();
        let mut current_indices: Vec<usize> = indices.to_vec();

        // Walk from leaves to root, downloading each layer once
        for layer_idx in (1..=depth).rev() {
            let layer_host = self.layers[layer_idx].to_host();
            for (q, idx) in current_indices.iter_mut().enumerate() {
                let sibling_idx = *idx ^ 1;
                let mut hash = [0u32; HASH_WORDS];
                let start = sibling_idx * HASH_WORDS;
                hash.copy_from_slice(&layer_host[start..start + HASH_WORDS]);
                paths[q].push(hash);
                *idx /= 2;
            }
        }

        paths
    }

    /// Download leaf values at given indices from the leaf layer.
    /// Returns raw u32 words: HASH_WORDS per leaf (the Blake2s hash).
    pub fn leaf_hashes_at(&self, indices: &[usize]) -> Vec<[u32; HASH_WORDS]> {
        let leaf_layer = &self.layers[self.log_n_leaves as usize];
        let host = leaf_layer.to_host();
        indices
            .iter()
            .map(|&idx| {
                let mut hash = [0u32; HASH_WORDS];
                let start = idx * HASH_WORDS;
                hash.copy_from_slice(&host[start..start + HASH_WORDS]);
                hash
            })
            .collect()
    }

    /// Verify a Merkle auth path against a known root.
    /// `leaf_hash` is the Blake2s hash of the leaf data (HASH_WORDS u32).
    /// `index` is the leaf position in the tree.
    /// `path` contains sibling hashes from leaf to root.
    /// Returns true if the computed root matches `expected_root`.
    pub fn verify_auth_path(
        expected_root: &[u32; HASH_WORDS],
        leaf_hash: &[u32; HASH_WORDS],
        mut index: usize,
        path: &[[u32; HASH_WORDS]],
    ) -> bool {
        use crate::channel::blake2s_hash;
        let mut current = *leaf_hash;

        for sibling in path {
            // Determine order: if index is even, current is left child
            let mut input = [0u8; 64];
            if index % 2 == 0 {
                for (i, &w) in current.iter().enumerate() {
                    input[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
                }
                for (i, &w) in sibling.iter().enumerate() {
                    input[32 + i * 4..32 + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
                }
            } else {
                for (i, &w) in sibling.iter().enumerate() {
                    input[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
                }
                for (i, &w) in current.iter().enumerate() {
                    input[32 + i * 4..32 + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
                }
            }
            let hash_bytes = blake2s_hash(&input);
            for i in 0..HASH_WORDS {
                current[i] = u32::from_le_bytes([
                    hash_bytes[i * 4],
                    hash_bytes[i * 4 + 1],
                    hash_bytes[i * 4 + 2],
                    hash_bytes[i * 4 + 3],
                ]);
            }
            index /= 2;
        }

        current == *expected_root
    }

    /// Compute the Blake2s leaf hash for a single value with n_cols columns.
    /// This matches the GPU leaf hashing in blake2s.cu.
    pub fn hash_leaf(values: &[u32]) -> [u32; HASH_WORDS] {
        use crate::channel::blake2s_hash;
        let n_cols = values.len();
        let mut input = [0u8; 64]; // max 16 columns × 4 bytes
        for (i, &v) in values.iter().enumerate() {
            input[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let hash_bytes = blake2s_hash(&input[..n_cols * 4]);
        let mut out = [0u32; HASH_WORDS];
        for i in 0..HASH_WORDS {
            out[i] = u32::from_le_bytes([
                hash_bytes[i * 4],
                hash_bytes[i * 4 + 1],
                hash_bytes[i * 4 + 2],
                hash_bytes[i * 4 + 3],
            ]);
        }
        out
    }

    /// Build a CPU-side Merkle tree from 4-column SoA host data and extract auth paths.
    /// Produces the same hashes as the GPU Merkle tree (commit_soa4).
    /// This avoids storing the full tree on GPU.
    pub fn cpu_merkle_auth_paths_soa4(
        host_cols: &[Vec<u32>; 4],
        indices: &[usize],
    ) -> Vec<Vec<[u32; HASH_WORDS]>> {
        let n = host_cols[0].len();
        assert!(n.is_power_of_two() && n >= 1);

        let hash_leaf = |i: usize| -> [u32; HASH_WORDS] {
            Self::hash_leaf(&[host_cols[0][i], host_cols[1][i], host_cols[2][i], host_cols[3][i]])
        };

        Self::targeted_auth_paths(n, indices, &hash_leaf)
    }

    /// Build a CPU-side Merkle tree from single-column host data and extract auth paths.
    /// Produces the same hashes as the GPU Merkle tree (commit with 1 column).
    pub fn cpu_merkle_auth_paths_single(
        host_col: &[u32],
        indices: &[usize],
    ) -> Vec<Vec<[u32; HASH_WORDS]>> {
        let n = host_col.len();
        assert!(n.is_power_of_two() && n >= 1);

        let hash_leaf = |i: usize| -> [u32; HASH_WORDS] {
            Self::hash_leaf(&[host_col[i]])
        };

        Self::targeted_auth_paths(n, indices, &hash_leaf)
    }

    /// Compute auth paths by only building the subtrees that contain queried indices.
    /// For 100 queries across 268M leaves, this touches ~200 tiles of 1024 instead of all 262K.
    /// The upper tree (subtree roots → final root) is always built in full since it's small.
    fn targeted_auth_paths(
        n: usize,
        indices: &[usize],
        hash_leaf: &dyn Fn(usize) -> [u32; HASH_WORDS],
    ) -> Vec<Vec<[u32; HASH_WORDS]>> {
        use std::collections::HashMap;

        const TILE_SIZE: usize = 1024;

        // For small trees, just build the full tree directly (no tiling overhead)
        if n <= TILE_SIZE * 4 {
            let leaf_hashes: Vec<[u32; HASH_WORDS]> = (0..n).map(hash_leaf).collect();
            let layers = Self::build_cpu_tree_layers(leaf_hashes);
            return Self::extract_paths_from_layers(&layers, indices);
        }

        let n_tiles = n / TILE_SIZE;

        // Collect which tiles we need (from query indices)
        let mut needed_tiles: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for &qi in indices {
            needed_tiles.insert(qi / TILE_SIZE);
        }

        // Compute ALL tile roots (needed for upper tree correctness).
        // Only build full subtrees for the ~200 tiles containing queries.
        let mut tile_roots: Vec<[u32; HASH_WORDS]> = Vec::with_capacity(n_tiles);
        let mut tile_subtrees: HashMap<usize, Vec<Vec<[u32; HASH_WORDS]>>> = HashMap::new();

        for tile_idx in 0..n_tiles {
            let base = tile_idx * TILE_SIZE;
            let leaf_hashes: Vec<[u32; HASH_WORDS]> = (base..base + TILE_SIZE)
                .map(hash_leaf)
                .collect();

            if needed_tiles.contains(&tile_idx) {
                let layers = Self::build_cpu_tree_layers(leaf_hashes);
                let root = layers.last().unwrap()[0];
                tile_roots.push(root);
                tile_subtrees.insert(tile_idx, layers);
            } else {
                let root = Self::reduce_to_root(leaf_hashes);
                tile_roots.push(root);
            }
        }

        // Build upper tree from tile roots
        let upper_layers = Self::build_cpu_tree_layers(tile_roots);

        // Extract auth paths: intra-tile path + upper tree path
        indices
            .iter()
            .map(|&qi| {
                let tile_idx = qi / TILE_SIZE;
                let intra_idx = qi % TILE_SIZE;
                let mut path = Vec::new();

                // Intra-tile path
                let subtree = &tile_subtrees[&tile_idx];
                let mut idx = intra_idx;
                for layer in &subtree[..subtree.len() - 1] {
                    path.push(layer[idx ^ 1]);
                    idx /= 2;
                }

                // Upper tree path
                let mut idx = tile_idx;
                for layer in &upper_layers[..upper_layers.len() - 1] {
                    path.push(layer[idx ^ 1]);
                    idx /= 2;
                }

                path
            })
            .collect()
    }

    /// Compute auth paths using pre-computed tile roots from the GPU tiled kernel.
    /// Only builds the ~200 subtrees containing queried indices on CPU.
    /// The upper tree is built from the pre-computed tile roots.
    /// This is O(n_queries * 1024) instead of O(n).
    pub fn targeted_auth_paths_with_tile_roots(
        tile_roots: &[[u32; HASH_WORDS]],
        n_leaves: usize,
        indices: &[usize],
        hash_leaf: &dyn Fn(usize) -> [u32; HASH_WORDS],
    ) -> Vec<Vec<[u32; HASH_WORDS]>> {
        use std::collections::HashMap;

        const TILE_SIZE: usize = 1024;

        // Collect which tiles we need
        let mut needed_tiles: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for &qi in indices {
            needed_tiles.insert(qi / TILE_SIZE);
        }

        // Build only the needed subtrees on CPU
        let mut tile_subtrees: HashMap<usize, Vec<Vec<[u32; HASH_WORDS]>>> = HashMap::new();
        for &tile_idx in &needed_tiles {
            let base = tile_idx * TILE_SIZE;
            let leaf_hashes: Vec<[u32; HASH_WORDS]> = (base..base + TILE_SIZE)
                .map(hash_leaf)
                .collect();
            tile_subtrees.insert(tile_idx, Self::build_cpu_tree_layers(leaf_hashes));
        }

        // Build upper tree from ALL tile roots (already computed by GPU)
        let upper_layers = Self::build_cpu_tree_layers(tile_roots.to_vec());

        // Extract auth paths: intra-tile path + upper tree path
        indices
            .iter()
            .map(|&qi| {
                let tile_idx = qi / TILE_SIZE;
                let intra_idx = qi % TILE_SIZE;
                let mut path = Vec::new();

                let subtree = &tile_subtrees[&tile_idx];
                let mut idx = intra_idx;
                for layer in &subtree[..subtree.len() - 1] {
                    path.push(layer[idx ^ 1]);
                    idx /= 2;
                }

                let mut idx = tile_idx;
                for layer in &upper_layers[..upper_layers.len() - 1] {
                    path.push(layer[idx ^ 1]);
                    idx /= 2;
                }

                path
            })
            .collect()
    }

    /// Build CPU Merkle tree layers from leaf hashes. Returns [leaves, parents, ..., root].
    fn build_cpu_tree_layers(leaf_hashes: Vec<[u32; HASH_WORDS]>) -> Vec<Vec<[u32; HASH_WORDS]>> {
        use crate::channel::blake2s_hash;
        let mut layers = vec![leaf_hashes];
        while layers.last().unwrap().len() > 1 {
            let prev = layers.last().unwrap();
            let parents: Vec<[u32; HASH_WORDS]> = (0..prev.len() / 2)
                .map(|i| Self::hash_pair(&prev[2 * i], &prev[2 * i + 1]))
                .collect();
            layers.push(parents);
        }
        layers
    }

    /// Reduce leaf hashes to a single root without storing intermediate layers.
    fn reduce_to_root(mut hashes: Vec<[u32; HASH_WORDS]>) -> [u32; HASH_WORDS] {
        while hashes.len() > 1 {
            hashes = (0..hashes.len() / 2)
                .map(|i| Self::hash_pair(&hashes[2 * i], &hashes[2 * i + 1]))
                .collect();
        }
        hashes[0]
    }

    /// Hash two sibling nodes into a parent.
    fn hash_pair(left: &[u32; HASH_WORDS], right: &[u32; HASH_WORDS]) -> [u32; HASH_WORDS] {
        use crate::channel::blake2s_hash;
        let mut input = [0u8; 64];
        for (j, &w) in left.iter().enumerate() {
            input[j * 4..j * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        for (j, &w) in right.iter().enumerate() {
            input[32 + j * 4..32 + j * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        let h = blake2s_hash(&input);
        let mut out = [0u32; HASH_WORDS];
        for k in 0..HASH_WORDS {
            out[k] = u32::from_le_bytes([h[k * 4], h[k * 4 + 1], h[k * 4 + 2], h[k * 4 + 3]]);
        }
        out
    }

    /// Extract auth paths from pre-built tree layers.
    fn extract_paths_from_layers(
        layers: &[Vec<[u32; HASH_WORDS]>],
        indices: &[usize],
    ) -> Vec<Vec<[u32; HASH_WORDS]>> {
        indices
            .iter()
            .map(|&qi| {
                let mut path = Vec::new();
                let mut idx = qi;
                for layer in &layers[..layers.len() - 1] {
                    path.push(layer[idx ^ 1]);
                    idx /= 2;
                }
                path
            })
            .collect()
    }

    /// Build a full Merkle tree from 4-column SoA data, storing all layers.
    /// Uses the same hashing as commit_root_soa4 but retains the tree.
    pub fn commit_soa4(
        col0: &DeviceBuffer<u32>,
        col1: &DeviceBuffer<u32>,
        col2: &DeviceBuffer<u32>,
        col3: &DeviceBuffer<u32>,
        log_n_leaves: u32,
    ) -> Self {
        let n_leaves = 1u32 << log_n_leaves;

        // Use fused leaf+merge to get n/2 parent hashes
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

        // Also compute full leaf hashes for the leaf layer (needed for auth paths)
        // Build leaf hashes from the 4 SoA columns
        let col_ptrs: Vec<*const u32> = vec![
            col0.as_ptr(), col1.as_ptr(), col2.as_ptr(), col3.as_ptr(),
        ];
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);
        let mut d_leaves = DeviceBuffer::<u32>::alloc((n_leaves as usize) * HASH_WORDS);
        unsafe {
            ffi::cuda_merkle_hash_leaves(
                d_col_ptrs.as_ptr() as *const *const u32,
                d_leaves.as_mut_ptr(),
                4,
                n_leaves,
            );
        }

        // Build tree layers bottom-up, keeping all layers
        let mut all_layers = vec![current]; // first entry = n/2 nodes
        let mut sz = current_size;
        while sz > 1 {
            let parent_size = sz / 2;
            let mut parents = DeviceBuffer::<u32>::alloc((parent_size as usize) * HASH_WORDS);
            unsafe {
                ffi::cuda_merkle_hash_nodes(
                    all_layers.last().unwrap().as_ptr(),
                    parents.as_mut_ptr(),
                    parent_size,
                );
            }
            all_layers.push(parents);
            sz = parent_size;
        }

        // Arrange as [root, ..., n/2 nodes, leaves]
        let root = all_layers.pop().unwrap();
        let mut layers = vec![root];
        for layer in all_layers.into_iter().rev() {
            layers.push(layer);
        }
        layers.push(d_leaves); // leaf layer

        Self {
            layers,
            log_n_leaves,
        }
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
