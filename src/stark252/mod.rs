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

pub use field::{Fp, ntt_root_of_unity, batch_inverse, Channel252,
                fp_to_u32x8, fp_from_u32x8};
pub use prover::{prove, Stark252Proof};
pub use verifier::verify;
