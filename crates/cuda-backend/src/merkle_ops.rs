//! MerkleOps: GPU-accelerated Merkle tree commitment via Blake2s.

use stwo_prover::core::backend::{Col, ColumnOps};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use stwo_prover::core::vcs::ops::MerkleOps;

use super::CudaBackend;

impl MerkleOps<Blake2sMerkleHasher> for CudaBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, <Blake2sMerkleHasher as stwo_prover::core::vcs::ops::MerkleHasher>::Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, <Blake2sMerkleHasher as stwo_prover::core::vcs::ops::MerkleHasher>::Hash> {
        todo!("MerkleOps::commit_on_layer — wire to cuda_merkle_hash_leaves / cuda_blake2s")
    }
}
