//! VortexSTARK CUDA backend for stwo.
//!
//! Implements the stwo `Backend` trait using GPU-accelerated kernels
//! for Circle NTT, FRI folding, Merkle commitment, and field operations.
//! All heavy computation runs on NVIDIA GPUs via CUDA.

mod column;
mod backend;
mod field_ops;
mod poly_ops;
mod fri_ops;
mod quotient_ops;
mod accumulation_ops;
mod gkr_ops;
mod merkle_ops;

pub use backend::CudaBackend;
pub use column::CudaColumn;
