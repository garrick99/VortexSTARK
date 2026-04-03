//! Cairo VM AIR for VortexSTARK.
//!
//! Implements the full Cairo CPU constraint system:
//! - Instruction decoding (15 flag bits)
//! - Operand resolution (dst, op0, op1, res)
//! - Register updates (pc, ap, fp)
//! - Memory consistency (via permutation argument)
//! - Range checks on offsets
//!
//! Architecture: single CUDA kernel evaluates all constraints per row,
//! combining opcode-specific branches via warp divergence.

pub mod decode;
pub mod vm;
pub mod trace;
pub mod logup;
pub mod range_check;
pub mod builtins;
pub mod pedersen;
pub mod bitwise;
pub mod stark252_field;
pub mod prover;
pub mod ec_constraint;
pub mod casm_loader;
pub mod hints;
pub mod dict_consistency;
pub mod dict_air;
pub mod starknet_rpc;
