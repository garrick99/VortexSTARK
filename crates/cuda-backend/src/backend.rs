//! The CudaBackend marker type and Backend trait impl.

use serde::{Serialize, Deserialize};
use stwo::prover::backend::Backend;

/// GPU-accelerated backend for stwo Circle STARK proving.
/// All heavy operations (NTT, FRI, Merkle, field ops) run on NVIDIA CUDA.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct CudaBackend;

impl Backend for CudaBackend {}

impl stwo::prover::backend::BackendForChannel<
    stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel
> for CudaBackend {}
