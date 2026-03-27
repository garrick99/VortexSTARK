//! CPU Merkle tree for Stark252 field elements.
//!
//! Each leaf = Blake2s(fp_to_bytes(value)) with domain 0x00.
//! Each internal node = blake2s_hash_node(left_child || right_child) with domain 0x01.
//!
//! Auth paths are provided bottom-up (leaf sibling first, root last).

use super::field::{Fp, fp_to_u32x8};
use crate::channel::{blake2s_hash, blake2s_hash_node};

pub type Digest = [u32; 8];

/// Hash a single Fp element as a leaf node.
fn hash_leaf(x: &Fp) -> Digest {
    let words = fp_to_u32x8(x);
    let mut bytes = [0u8; 32];
    for (i, &w) in words.iter().enumerate() {
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
    }
    let h = blake2s_hash(&bytes);
    let mut out = [0u32; 8];
    for (i, chunk) in h.chunks_exact(4).enumerate() {
        out[i] = u32::from_le_bytes(chunk.try_into().unwrap());
    }
    out
}

/// Hash two child digests into a parent node.
fn hash_node(left: &Digest, right: &Digest) -> Digest {
    let mut bytes = [0u8; 64];
    for (i, &w) in left.iter().enumerate() {
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
    }
    for (i, &w) in right.iter().enumerate() {
        bytes[32 + i * 4..32 + (i + 1) * 4].copy_from_slice(&w.to_le_bytes());
    }
    let h = blake2s_hash_node(&bytes);
    let mut out = [0u32; 8];
    for (i, chunk) in h.chunks_exact(4).enumerate() {
        out[i] = u32::from_le_bytes(chunk.try_into().unwrap());
    }
    out
}

/// A committed Merkle tree over a column of Fp elements.
pub struct MerkleTree252 {
    /// Tree nodes: `nodes[0]` = root, `nodes[i]` = parent of `nodes[2i+1]` and `nodes[2i+2]`.
    /// The leaf level starts at index `n - 1` (for n leaves).
    /// Total nodes: 2n - 1.
    nodes: Vec<Digest>,
    pub log_n: u32,
}

impl MerkleTree252 {
    /// Commit a slice of Fp values. Length must be a power of two.
    pub fn commit(values: &[Fp]) -> Self {
        let n = values.len();
        assert!(n.is_power_of_two(), "Merkle tree requires power-of-two leaf count");
        let log_n = n.trailing_zeros();

        let total = 2 * n - 1;
        let mut nodes = vec![[0u32; 8]; total];

        // Fill leaf level (last n nodes in the array, offset n-1)
        for (i, v) in values.iter().enumerate() {
            nodes[n - 1 + i] = hash_leaf(v);
        }

        // Build internal nodes bottom-up
        for i in (0..n - 1).rev() {
            nodes[i] = hash_node(&nodes[2 * i + 1], &nodes[2 * i + 2]);
        }

        MerkleTree252 { nodes, log_n }
    }

    /// Return the Merkle root.
    pub fn root(&self) -> Digest {
        self.nodes[0]
    }

    /// Return the authentication path for leaf index `idx`.
    ///
    /// The path is ordered from leaf-level sibling up to just below the root.
    /// Length = log_n.
    pub fn auth_path(&self, idx: usize) -> Vec<Digest> {
        let n = 1usize << self.log_n;
        let mut path = Vec::with_capacity(self.log_n as usize);
        let mut node_idx = n - 1 + idx;
        while node_idx > 0 {
            // Sibling: if current is left child (odd), sibling is right child (+1), and vice versa
            let sibling = if node_idx % 2 == 1 { node_idx + 1 } else { node_idx - 1 };
            path.push(self.nodes[sibling]);
            node_idx = (node_idx - 1) / 2;
        }
        path
    }

    /// Return the leaf hash for index `idx`.
    pub fn leaf(&self, idx: usize) -> Digest {
        let n = 1usize << self.log_n;
        self.nodes[n - 1 + idx]
    }
}

/// Verify a Merkle auth path.
///
/// `leaf_value` — the Fp element at the queried index.
/// `auth_path`  — the auth path returned by `MerkleTree252::auth_path`.
/// `root`       — the committed Merkle root.
/// `idx`        — the leaf index.
/// `log_n`      — log2 of the number of leaves.
pub fn verify_auth_path(
    leaf_value: &Fp,
    auth_path: &[Digest],
    root: &Digest,
    mut idx: usize,
    log_n: u32,
) -> bool {
    assert_eq!(auth_path.len(), log_n as usize);
    let mut current = hash_leaf(leaf_value);
    for sibling in auth_path {
        current = if idx % 2 == 0 {
            hash_node(&current, sibling)
        } else {
            hash_node(sibling, &current)
        };
        idx >>= 1;
    }
    &current == root
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cairo_air::stark252_field::Fp;

    #[test]
    fn test_merkle_roundtrip() {
        let leaves: Vec<Fp> = (1u64..=8).map(Fp::from_u64).collect();
        let tree = MerkleTree252::commit(&leaves);
        let root = tree.root();

        for i in 0..8 {
            let path = tree.auth_path(i);
            assert!(
                verify_auth_path(&leaves[i], &path, &root, i, 3),
                "Auth path verification failed for leaf {i}"
            );
        }
    }

    #[test]
    fn test_merkle_tamper_detected() {
        let leaves: Vec<Fp> = (1u64..=4).map(Fp::from_u64).collect();
        let tree = MerkleTree252::commit(&leaves);
        let root = tree.root();
        let path = tree.auth_path(0);

        // Tamper: wrong leaf value
        let wrong = Fp::from_u64(999);
        assert!(
            !verify_auth_path(&wrong, &path, &root, 0, 2),
            "Tampered leaf should be rejected"
        );
    }
}
