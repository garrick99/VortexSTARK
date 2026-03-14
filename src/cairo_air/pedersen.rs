//! Pedersen hash builtin for Cairo VM.
//!
//! The Pedersen hash operates on the STARK curve (NOT M31):
//!   y² = x³ + x + β  over  F_p  where  p = 2^251 + 17·2^192 + 1
//!
//! H(a, b) = [P₀ + a_low·P₁ + a_high·P₂ + b_low·P₃ + b_high·P₄]_x
//!
//! where a_low = lowest 248 bits, a_high = highest 4 bits,
//! and P₀..P₄ are fixed curve points derived from digits of π.
//!
//! STARK AIR representation:
//! The 252-bit field elements are decomposed into limbs of M31-sized values
//! for the constraint system. Each Stark252 element requires ~9 M31 limbs
//! (252 bits / 31 bits per limb ≈ 9 limbs).
//!
//! The EC point addition and scalar multiplication are expressed as
//! polynomial constraints over these M31 limbs.
//!
//! Architecture (matching stwo-cairo):
//! - pedersen_builtin: 3 trace columns (input_a_id, input_b_id, output_id)
//!   Links to memory via LogUp
//! - pedersen_aggregator: 206 columns
//!   Aggregates the windowed scalar multiplication results
//! - partial_ec_mul: 297 columns (windowed, 18-bit windows)
//!   Performs the actual elliptic curve scalar multiplication
//!
//! Total: ~500+ columns for the full Pedersen AIR
//!
//! For our implementation, we take a pragmatic approach:
//! 1. The COMPUTATION runs natively (Stark252 arithmetic on CPU/GPU)
//! 2. The PROOF uses a lookup-based approach: the Pedersen builtin
//!    produces (input_a, input_b, output) tuples, and LogUp proves
//!    the VM's memory accesses match these tuples.
//! 3. A separate "Pedersen table" component proves the tuples are
//!    valid Pedersen hashes (via the EC arithmetic constraints).

use crate::field::M31;

/// STARK curve prime: p = 2^251 + 17·2^192 + 1
pub const STARK_PRIME_HEX: &str =
    "0800000000000011000000000000000000000000000000000000000000000001";

/// STARK curve parameters: y² = x³ + αx + β where α = 1
pub const CURVE_ALPHA: u64 = 1;

/// β = 3141592653589793238462643383279502884197169399375105820974944592307816406665
pub const CURVE_BETA_HEX: &str =
    "06f21413efbe40de150e596d72f7a8c5609ad26c15c915c1f4cdfcb99cee9e89";

/// Number of M31 limbs to represent a Stark252 field element.
/// 252 bits / 31 bits = 9 limbs (with the top limb < 2^(252-8*31) = 2^4 = 16)
pub const N_LIMBS: usize = 9;

/// A 252-bit field element represented as 9 M31 limbs (little-endian).
/// limb[0] is the least significant 31 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stark252 {
    pub limbs: [u32; N_LIMBS],
}

impl Stark252 {
    pub const ZERO: Self = Self { limbs: [0; N_LIMBS] };

    /// Create from a u64 (for small values).
    pub fn from_u64(v: u64) -> Self {
        let mut limbs = [0u32; N_LIMBS];
        limbs[0] = (v & 0x7FFF_FFFF) as u32;
        limbs[1] = ((v >> 31) & 0x7FFF_FFFF) as u32;
        limbs[2] = ((v >> 62) & 0x3) as u32; // top 2 bits
        Self { limbs }
    }

    /// Create from a hex string (big-endian).
    pub fn from_hex(s: &str) -> Self {
        let s = s.trim_start_matches("0x").trim_start_matches("0X");
        let bytes = hex_to_bytes(s);
        let mut limbs = [0u32; N_LIMBS];
        // Convert big-endian bytes to little-endian 31-bit limbs
        let mut bit_pos = 0usize;
        for &byte in bytes.iter().rev() {
            for bit in 0..8 {
                if byte & (1 << bit) != 0 {
                    let limb_idx = bit_pos / 31;
                    let limb_bit = bit_pos % 31;
                    if limb_idx < N_LIMBS {
                        limbs[limb_idx] |= 1 << limb_bit;
                    }
                }
                bit_pos += 1;
            }
        }
        Self { limbs }
    }

    /// Convert to M31 limbs (for trace columns).
    pub fn to_m31_limbs(&self) -> [M31; N_LIMBS] {
        std::array::from_fn(|i| M31(self.limbs[i]))
    }
}

/// A point on the STARK curve.
#[derive(Clone, Copy, Debug, Default)]
pub struct StarkPoint {
    pub x: Stark252,
    pub y: Stark252,
}

/// The 5 Pedersen hash constant points (from digits of π).
pub fn pedersen_points() -> [StarkPoint; 5] {
    [
        StarkPoint {
            x: Stark252::from_hex("49ee3eba8c1600700ee1b87eb599f16716b0b1022947733551fde4050ca6804"),
            y: Stark252::from_hex("3ca0cfe4b3bc6ddf346d49d06ea0ed34e621062c0e056c1d0405d266e10268a"),
        },
        StarkPoint {
            x: Stark252::from_hex("234287dcbaffe7f969c748655fca9e58fa8120b6d56eb0c1080d17957ebe47b"),
            y: Stark252::from_hex("3b056f100f96fb21e889527d41f4e39940135dd7a6c7e6c2f8116572f578e85"),
        },
        StarkPoint {
            x: Stark252::from_hex("4fa56f376c83db33f9dab2656558f3399099ec1de5e3018b7571f510a2c7768"),
            y: Stark252::from_hex("3f42a042e45b8a3e3821a7133325bfa989e2bc26485dbe63ac6eadc28fc2fad"),
        },
        StarkPoint {
            x: Stark252::from_hex("4ba4cc166be8dec764910f75b45f74b40642ad9b32d50d8865e3e7caa740577"),
            y: Stark252::from_hex("00416a975392d0e71777ab65e5e7e4c54daee0efbb7d00b8d2ccacfefa2d8e1c"),
        },
        StarkPoint {
            x: Stark252::from_hex("54302dcb0e6cc1c6e44cca8f61a63bb2ca65048d53fb325d36ff12c49a58202"),
            y: Stark252::from_hex("01b77b3e37d13504b348046268d8ae25ce98ad783c25561a879dcc77e99c2426"),
        },
    ]
}

/// Pedersen builtin for Cairo VM.
/// Manages invocations and generates trace data.
pub struct PedersenBuiltin {
    /// (input_a, input_b, output) tuples
    pub entries: Vec<(Stark252, Stark252, Stark252)>,
}

impl PedersenBuiltin {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Invoke Pedersen hash using real EC arithmetic on the STARK curve.
    pub fn invoke(&mut self, a: Stark252, b: Stark252) -> Stark252 {
        use super::stark252_field::{Fp, pedersen_hash};
        let fp_a = stark252_to_fp(&a);
        let fp_b = stark252_to_fp(&b);
        let fp_out = pedersen_hash(fp_a, fp_b);
        let out = fp_to_stark252(&fp_out);
        self.entries.push((a, b, out));
        out
    }

    pub fn n_invocations(&self) -> usize {
        self.entries.len()
    }

    /// Generate trace columns for the Pedersen builtin.
    /// Each invocation produces one row with 3 * N_LIMBS = 27 columns:
    /// [a_limbs(9), b_limbs(9), output_limbs(9)]
    pub fn generate_trace(&self, log_n: u32) -> Vec<Vec<u32>> {
        let n = 1usize << log_n;
        let n_cols = 3 * N_LIMBS; // 27 columns
        let mut cols: Vec<Vec<u32>> = (0..n_cols).map(|_| vec![0u32; n]).collect();

        for (i, (a, b, out)) in self.entries.iter().enumerate() {
            if i >= n { break; }
            for j in 0..N_LIMBS {
                cols[j][i] = a.limbs[j];
                cols[N_LIMBS + j][i] = b.limbs[j];
                cols[2 * N_LIMBS + j][i] = out.limbs[j];
            }
        }

        cols
    }

    /// Generate LogUp entries for memory consistency.
    pub fn logup_entries(&self, base_addr: u64) -> Vec<(M31, M31)> {
        let mut entries = Vec::new();
        let stride = 3u64; // 3 memory cells per invocation (a_id, b_id, out_id)

        for (inv_idx, (a, b, out)) in self.entries.iter().enumerate() {
            let base = base_addr + inv_idx as u64 * stride;
            // For simplicity, use first limb as the memory value
            entries.push((M31((base) as u32), M31(a.limbs[0])));
            entries.push((M31((base + 1) as u32), M31(b.limbs[0])));
            entries.push((M31((base + 2) as u32), M31(out.limbs[0])));
        }
        entries
    }
}

/// Convert Stark252 (9×31-bit limbs) to Fp (4×64-bit limbs).
pub fn stark252_to_fp(s: &Stark252) -> super::stark252_field::Fp {
    use super::stark252_field::Fp;
    // Reassemble 252 bits from 31-bit limbs into 64-bit limbs
    let mut bits = [0u8; 256];
    for i in 0..N_LIMBS {
        let val = s.limbs[i];
        let n_bits = if i == N_LIMBS - 1 { 252 - 31 * (N_LIMBS - 1) } else { 31 };
        for b in 0..n_bits {
            bits[i * 31 + b] = ((val >> b) & 1) as u8;
        }
    }
    let mut v = [0u64; 4];
    for i in 0..252 {
        if bits[i] == 1 {
            v[i / 64] |= 1u64 << (i % 64);
        }
    }
    Fp { v }
}

/// Convert Fp (4×64-bit limbs) to Stark252 (9×31-bit limbs).
pub fn fp_to_stark252(fp: &super::stark252_field::Fp) -> Stark252 {
    let mut bits = [0u8; 256];
    for i in 0..252 {
        if fp.v[i / 64] & (1u64 << (i % 64)) != 0 {
            bits[i] = 1;
        }
    }
    let mut limbs = [0u32; N_LIMBS];
    for i in 0..N_LIMBS {
        let n_bits = if i == N_LIMBS - 1 { 252 - 31 * (N_LIMBS - 1) } else { 31 };
        for b in 0..n_bits {
            if bits[i * 31 + b] == 1 {
                limbs[i] |= 1 << b;
            }
        }
    }
    Stark252 { limbs }
}

pub const PEDERSEN_BUILTIN_BASE: u64 = 0x5000_0000;

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    let hex = if hex.len() % 2 == 1 {
        format!("0{hex}")
    } else {
        hex.to_string()
    };
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap_or(0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stark252_from_u64() {
        let val = Stark252::from_u64(42);
        assert_eq!(val.limbs[0], 42);
        assert_eq!(val.limbs[1], 0);
    }

    #[test]
    fn test_stark252_from_hex() {
        let val = Stark252::from_hex("ff");
        assert_eq!(val.limbs[0], 0xFF);
        assert_eq!(val.limbs[1], 0);
    }

    #[test]
    fn test_stark252_large() {
        // 2^31 should be in limb[1]
        let val = Stark252::from_hex("80000000"); // 2^31
        assert_eq!(val.limbs[0], 0);
        assert_eq!(val.limbs[1], 1);
    }

    #[test]
    fn test_pedersen_points_loaded() {
        let points = pedersen_points();
        // P0.x should be non-zero
        assert_ne!(points[0].x, Stark252::ZERO);
        assert_ne!(points[0].y, Stark252::ZERO);
    }

    #[test]
    fn test_pedersen_builtin_invoke() {
        let mut builtin = PedersenBuiltin::new();
        let a = Stark252::from_u64(42);
        let b = Stark252::from_u64(99);
        let out = builtin.invoke(a, b);

        assert_eq!(builtin.n_invocations(), 1);
        assert_ne!(out, Stark252::ZERO);

        // Deterministic
        let mut builtin2 = PedersenBuiltin::new();
        let out2 = builtin2.invoke(a, b);
        assert_eq!(out, out2);
    }

    #[test]
    fn test_pedersen_trace() {
        let mut builtin = PedersenBuiltin::new();
        builtin.invoke(Stark252::from_u64(1), Stark252::from_u64(2));
        builtin.invoke(Stark252::from_u64(3), Stark252::from_u64(4));

        let cols = builtin.generate_trace(2); // 4 rows
        assert_eq!(cols.len(), 3 * N_LIMBS); // 27 columns
        assert_eq!(cols[0].len(), 4);

        // First invocation: a.limbs[0] = 1
        assert_eq!(cols[0][0], 1);
        // Second invocation: a.limbs[0] = 3
        assert_eq!(cols[0][1], 3);
    }
}
