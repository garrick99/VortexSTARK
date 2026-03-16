//! The CudaBackend marker type and Backend trait impl.

use serde::{Serialize, Deserialize};
use stwo_prover::core::backend::Backend;

/// GPU-accelerated backend for stwo Circle STARK proving.
/// All heavy operations (NTT, FRI, Merkle, field ops) run on NVIDIA CUDA.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct CudaBackend;

impl Backend for CudaBackend {}
