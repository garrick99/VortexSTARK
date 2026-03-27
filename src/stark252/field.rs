//! Stark252 field utilities for the standard FRI-based STARK prover.
//!
//! Reexports Fp from cairo_air::stark252_field and adds:
//! - Serialization helpers (fp_to_u32x8, fp_from_u32x8)
//! - NTT domain: ntt_root_of_unity, batch_inverse
//! - Channel252: Fiat-Shamir channel that draws Fp252 elements

pub use crate::cairo_air::stark252_field::{Fp, PRIME};
pub use crate::channel::{blake2s_hash, blake2s_hash_node, Channel};

// ─────────────────────────────────────────────
// Serialization
// ─────────────────────────────────────────────

/// Serialize an Fp element as 8 × u32 (LE: each u64 limb split into lo/hi u32).
/// Layout: [limb0_lo, limb0_hi, limb1_lo, limb1_hi, limb2_lo, limb2_hi, limb3_lo, limb3_hi]
pub fn fp_to_u32x8(x: &Fp) -> [u32; 8] {
    let mut out = [0u32; 8];
    for (k, &limb) in x.v.iter().enumerate() {
        out[2 * k]     = limb as u32;
        out[2 * k + 1] = (limb >> 32) as u32;
    }
    out
}

/// Deserialize an Fp element from 8 × u32 (LE). Reduces mod P.
pub fn fp_from_u32x8(w: &[u32; 8]) -> Fp {
    let mut v = [0u64; 4];
    for k in 0..4 {
        v[k] = (w[2 * k] as u64) | ((w[2 * k + 1] as u64) << 32);
    }
    Fp::new(v)
}

// ─────────────────────────────────────────────
// NTT domain
// ─────────────────────────────────────────────

/// P − 1 = 2^192 × (2^59 + 17). The odd part of P−1.
const ODD_PART: u64 = (1u64 << 59) + 17; // 576460752303423505

/// Generator of F_p* (standard choice for Stark252).
pub const GENERATOR: Fp = Fp { v: [3, 0, 0, 0] };

/// Return a primitive 2^log_n-th root of unity for the Stark252 field.
///
/// Algorithm:
///   ω_max = g^{ODD_PART}  (order 2^192)
///   ω_k   = ω_max^{2^{192-k}}  (order 2^k)
///
/// Requires log_n ≤ 192 (the 2-adicity of P−1).
pub fn ntt_root_of_unity(log_n: u32) -> Fp {
    assert!(log_n <= 192, "NTT size 2^{log_n} exceeds 2-adicity of Stark252 (192)");
    let g = Fp { v: [3, 0, 0, 0] };
    // Compute g^{ODD_PART} (a primitive 2^192-th root of unity).
    let odd_exp = Fp { v: [ODD_PART, 0, 0, 0] };
    let mut omega = g.pow_fp(odd_exp);
    // Square (192 − log_n) times to get a primitive 2^log_n-th root.
    for _ in 0..(192 - log_n) {
        omega = omega.mul(omega);
    }
    omega
}

/// Compute g^k for a u64 exponent (convenience wrapper around pow_fp).
pub fn fp_pow_u64(base: Fp, exp: u64) -> Fp {
    base.pow_fp(Fp { v: [exp, 0, 0, 0] })
}

/// Batch modular inverse via Montgomery's trick.
/// Returns `[1/a[0], 1/a[1], ..., 1/a[n-1]]`.
/// Panics if any element is zero.
pub fn batch_inverse(a: &[Fp]) -> Vec<Fp> {
    if a.is_empty() {
        return Vec::new();
    }
    let n = a.len();
    // Forward pass: prefix[i] = a[0] * a[1] * … * a[i]
    let mut prefix = Vec::with_capacity(n);
    prefix.push(a[0]);
    for i in 1..n {
        prefix.push(prefix[i - 1].mul(a[i]));
    }
    // Single inversion
    let mut inv = prefix[n - 1].inverse();
    // Backward pass
    let mut result = vec![Fp::ZERO; n];
    for i in (1..n).rev() {
        result[i] = inv.mul(prefix[i - 1]);
        inv = inv.mul(a[i]);
    }
    result[0] = inv;
    result
}

// ─────────────────────────────────────────────
// Channel wrapper for Fp252 draws
// ─────────────────────────────────────────────

/// Fiat-Shamir channel that draws Stark252 field elements.
/// Wraps the existing Blake2s `Channel`.
pub struct Channel252(pub Channel);

impl Channel252 {
    pub fn new() -> Self {
        Self(Channel::new())
    }

    /// Mix a Merkle root (8 × u32) into the transcript.
    pub fn mix_digest(&mut self, d: &[u32; 8]) {
        self.0.mix_digest(d);
    }

    /// Mix a field element into the transcript.
    pub fn mix_fp(&mut self, x: &Fp) {
        self.0.mix_digest(&fp_to_u32x8(x));
    }

    /// Mix a u64 (e.g., PoW nonce) into the transcript.
    pub fn mix_u64(&mut self, v: u64) {
        self.0.mix_u64(v);
    }

    /// Draw a random Fp252 element.
    ///
    /// Squeezes 32 bytes, interprets as 4 × u64 (LE), and reduces mod P.
    /// Bias ≈ 2^252/2^256 ≈ 1/16, acceptable for Fiat-Shamir.
    pub fn draw_fp(&mut self) -> Fp {
        let bytes = self.0.squeeze_raw();
        let mut v = [0u64; 4];
        for i in 0..4 {
            v[i] = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
        }
        Fp::new(v)
    }

    /// Draw a random index in `[0, bound)`.
    pub fn draw_number(&mut self, bound: usize) -> usize {
        self.0.draw_number(bound)
    }

    /// Return the current state as 8 × u32 (for PoW prefix computation).
    pub fn state_words(&self) -> [u32; 8] {
        self.0.state_words()
    }

    /// Grind a PoW nonce: find `n` such that Blake2s(state || n) has `pow_bits` trailing zeros.
    pub fn grind_pow(&mut self, pow_bits: u32) -> u64 {
        if pow_bits == 0 {
            return 0;
        }
        let prefix = self.state_words();
        let target_mask: u32 = (1u32 << pow_bits) - 1;
        for nonce in 0u64.. {
            // Hash: Blake2s(channel_state_u8 || nonce_le_u8)
            let mut input = [0u8; 40];
            for (i, &w) in prefix.iter().enumerate() {
                input[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
            }
            input[32..40].copy_from_slice(&nonce.to_le_bytes());
            let h = blake2s_hash(&input);
            let lo = u32::from_le_bytes([h[0], h[1], h[2], h[3]]);
            if lo & target_mask == 0 {
                return nonce;
            }
        }
        unreachable!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp_roundtrip_u32x8() {
        let x = Fp { v: [1234567890123u64, 999999999u64, 0, 1] };
        let w = fp_to_u32x8(&x);
        let y = fp_from_u32x8(&w);
        assert_eq!(x, y);
    }

    #[test]
    fn test_ntt_root_order() {
        // ω_k^{2^k} == 1 and ω_k^{2^{k-1}} == P-1 (i.e., -1)
        for log_k in [2u32, 4, 8, 16] {
            let omega = ntt_root_of_unity(log_k);
            // Square log_k times → should give 1
            let mut x = omega;
            for _ in 0..log_k {
                x = x.mul(x);
            }
            assert_eq!(x, Fp::ONE, "ω_{log_k}^{{2^{log_k}}} != 1");
            // Square (log_k-1) times → should give P-1 = -1
            let mut y = omega;
            for _ in 0..(log_k - 1) {
                y = y.mul(y);
            }
            let minus_one = PRIME.sub(Fp::ONE);
            assert_eq!(y, minus_one, "ω_{log_k}^{{2^{{log_k-1}}}} != -1");
        }
    }

    #[test]
    fn test_batch_inverse() {
        let vals: Vec<Fp> = (1u64..=8).map(Fp::from_u64).collect();
        let invs = batch_inverse(&vals);
        for (v, inv) in vals.iter().zip(invs.iter()) {
            let prod = v.mul(*inv);
            assert_eq!(prod, Fp::ONE, "v * inv != 1");
        }
    }
}
