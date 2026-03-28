//! Stark252-native STARK prover — standard FRI over the 252-bit STARK prime.
//!
//! This module implements a proof system where field elements ARE Stark252 (Fp252)
//! elements, enabling correct Cairo/Starknet arithmetic without M31 truncation.
//!
//! # Field
//! P = 2^251 + 17·2^192 + 1 (the Stark field used by Cairo / Starknet).
//! P ≡ 1 (mod 4), so Circle STARK does not apply; we use standard FRI.
//!
//! # Protocol
//! - Evaluation domain: {ω_{4N}^i} (multiplicative coset, 4× blowup)
//! - FRI fold: f(x) → (f(x)+f(-x))/2 + α·(f(x)-f(-x))/(2x)
//! - Fibonacci STARK: degree-1 quotient (transition constraint only)
//!
//! # Current status
//! Fibonacci STARK: fully proven and verified.
//! Cairo VM STARK over Stark252: TODO (see cairo_air submodule).

pub mod field;
pub mod ntt;
pub mod merkle;
pub mod prover;
pub mod verifier;
pub mod fri;
pub mod stark;
pub mod cairo_air;
pub mod multi_stark;
pub mod logup;
pub mod range_check;
pub mod cairo_proof;

pub use field::{Fp, ntt_root_of_unity, batch_inverse, Channel252,
                fp_to_u32x8, fp_from_u32x8};
pub use prover::{prove, Stark252Proof};
pub use verifier::verify;
pub use fri::{FriProof, fri_commit, fri_build_proof, fri_verify};
pub use stark::{StarkAir, StarkProof, prove as prove_general, verify_with_air};
pub use multi_stark::{MultiColumnAir, MultiProof, prove_multi, verify_multi};
pub use logup::{MemoryAccess, MemoryLogupProof, prove_memory_logup, verify_memory_logup};
pub use range_check::{RangeCheckProof, prove_range_check, verify_range_check};
pub use cairo_proof::{CairoProof, prove_cairo, verify_cairo};
