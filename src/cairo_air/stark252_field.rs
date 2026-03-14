//! Stark252 field arithmetic: modular operations over the STARK prime.
//!
//! p = 2^251 + 17·2^192 + 1
//!   = 0x0800000000000011000000000000000000000000000000000000000000000001
//!
//! Elements stored as 4 × u64 limbs (little-endian, 256 bits total).
//! All arithmetic reduces mod p.

use std::ops::{Add, Sub, Mul, Neg};

/// The STARK prime p = 2^251 + 17·2^192 + 1
pub const P: [u64; 4] = [
    0x0000000000000001, // limb 0 (least significant)
    0x0000000000000000, // limb 1
    0x0000000000000011, // limb 2 (17·2^192 → 17 in this limb since limb2 = bits 128..191)
    0x0800000000000000, // limb 3 (2^251 → bit 251 in the top limb)
];

// Wait — let me recalculate the limb layout.
// p = 2^251 + 17·2^192 + 1
// In 4 × 64-bit limbs (little-endian):
// limb[0] = bits 0..63
// limb[1] = bits 64..127
// limb[2] = bits 128..191
// limb[3] = bits 192..255
//
// 2^251: bit 251 is in limb[3] (bits 192..255), position 251-192=59. So limb[3] has 1<<59.
// Wait, that's wrong. 251-192=59? No: 3*64=192. Bit 251 is in limb 3 at position 251-192=59.
// Hmm: 2^251 = 2^(3*64+59) → limb[3] bit 59 → limb[3] = 1 << 59 = 0x0800000000000000. ✓
//
// 17·2^192: 2^192 = 2^(3*64) → this is the lowest bit of limb[3].
// So 17·2^192 = 17 << (3*64 - 0) ... wait.
// 2^192 in 4×64 limbs: limb[0]=0, limb[1]=0, limb[2]=0, limb[3]=1.
// So 17·2^192: limb[3] = 17.
// But we already have 2^251 in limb[3] = 0x0800000000000000.
// Total limb[3] = 0x0800000000000000 + 17 = 0x0800000000000011.
//
// Plus 1 in limb[0].
// p = [1, 0, 0, 0x0800000000000011]

/// A 252-bit field element mod p, stored as 4 × u64 (little-endian).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp {
    pub v: [u64; 4],
}

/// The STARK prime.
pub const PRIME: Fp = Fp { v: [1, 0, 0, 0x0800000000000011] };

impl Fp {
    pub const ZERO: Self = Self { v: [0, 0, 0, 0] };
    pub const ONE: Self = Self { v: [1, 0, 0, 0] };

    pub fn new(v: [u64; 4]) -> Self {
        let mut r = Self { v };
        r.reduce();
        r
    }

    pub fn from_u64(x: u64) -> Self {
        Self { v: [x, 0, 0, 0] }
    }

    pub fn from_hex(s: &str) -> Self {
        let s = s.trim_start_matches("0x").trim_start_matches("0X");
        let s = format!("{:0>64}", s); // pad to 64 hex chars = 256 bits
        let v3 = u64::from_str_radix(&s[0..16], 16).unwrap_or(0);
        let v2 = u64::from_str_radix(&s[16..32], 16).unwrap_or(0);
        let v1 = u64::from_str_radix(&s[32..48], 16).unwrap_or(0);
        let v0 = u64::from_str_radix(&s[48..64], 16).unwrap_or(0);
        let mut r = Self { v: [v0, v1, v2, v3] };
        r.reduce();
        r
    }

    /// Reduce mod p.
    fn reduce(&mut self) {
        while self.ge_prime() {
            self.sub_prime();
        }
    }

    /// self >= p?
    fn ge_prime(&self) -> bool {
        for i in (0..4).rev() {
            if self.v[i] > PRIME.v[i] { return true; }
            if self.v[i] < PRIME.v[i] { return false; }
        }
        true // equal
    }

    /// self -= p (assumes self >= p)
    fn sub_prime(&mut self) {
        let mut borrow: u64 = 0;
        for i in 0..4 {
            let (r, b1) = self.v[i].overflowing_sub(PRIME.v[i]);
            let (r, b2) = r.overflowing_sub(borrow);
            self.v[i] = r;
            borrow = (b1 as u64) + (b2 as u64);
        }
    }

    /// Modular addition.
    pub fn add(self, rhs: Self) -> Self {
        let mut r = [0u64; 4];
        let mut carry = 0u128;
        for i in 0..4 {
            carry += self.v[i] as u128 + rhs.v[i] as u128;
            r[i] = carry as u64;
            carry >>= 64;
        }
        let mut result = Self { v: r };
        result.reduce();
        result
    }

    /// Modular subtraction.
    pub fn sub(self, rhs: Self) -> Self {
        if self.ge_rhs(&rhs) {
            let mut r = [0u64; 4];
            let mut borrow: u64 = 0;
            for i in 0..4 {
                let (r1, b1) = self.v[i].overflowing_sub(rhs.v[i]);
                let (r2, b2) = r1.overflowing_sub(borrow);
                r[i] = r2;
                borrow = (b1 as u64) + (b2 as u64);
            }
            Self { v: r }
        } else {
            // self < rhs: compute p + self - rhs
            let p_plus_self = PRIME.add(self);
            let mut r = [0u64; 4];
            let mut borrow: u64 = 0;
            for i in 0..4 {
                let (r1, b1) = p_plus_self.v[i].overflowing_sub(rhs.v[i]);
                let (r2, b2) = r1.overflowing_sub(borrow);
                r[i] = r2;
                borrow = (b1 as u64) + (b2 as u64);
            }
            let mut result = Self { v: r };
            result.reduce();
            result
        }
    }

    fn ge_rhs(&self, rhs: &Self) -> bool {
        for i in (0..4).rev() {
            if self.v[i] > rhs.v[i] { return true; }
            if self.v[i] < rhs.v[i] { return false; }
        }
        true
    }

    /// Modular multiplication using schoolbook 256×256→512 then Barrett-like reduction.
    pub fn mul(self, rhs: Self) -> Self {
        // Full 512-bit product
        let mut prod = [0u128; 8];
        for i in 0..4 {
            let mut carry = 0u128;
            for j in 0..4 {
                prod[i + j] += self.v[i] as u128 * rhs.v[j] as u128 + carry;
                carry = prod[i + j] >> 64;
                prod[i + j] &= 0xFFFF_FFFF_FFFF_FFFF;
            }
            if i + 4 < 8 {
                prod[i + 4] += carry;
            }
        }

        // Convert to u64 limbs
        let mut full = [0u64; 8];
        for i in 0..8 {
            full[i] = prod[i] as u64;
        }

        // Reduce mod p using repeated subtraction of p·2^k
        // More efficient: use the identity p = 2^251 + 17·2^192 + 1
        // So x mod p = x - floor(x/p) * p
        // For simplicity, use iterative reduction:
        reduce_512(&full)
    }

    /// Modular inverse via Fermat's little theorem: a^(p-2) mod p
    /// p - 2 = 0x080000000000001_0FFFFFFFFFFFFFFFE_FFFFFFFFFFFFFFFF_FFFFFFFFFFFFFFFF
    pub fn inverse(self) -> Self {
        // p - 2 hardcoded (verified: p = 2^251 + 17*2^192 + 1)
        let pm2 = Fp { v: [
            0xFFFF_FFFF_FFFF_FFFF, // p[0]-2 with borrow chain
            0xFFFF_FFFF_FFFF_FFFF,
            0xFFFF_FFFF_FFFF_FFFF,
            0x0800_0000_0000_0010, // p[3] - 1 from borrow
        ]};
        self.pow_fp(pm2)
    }

    /// Modular exponentiation (square-and-multiply).
    pub fn pow_fp(self, exp: Fp) -> Self {
        // Collect all bits of the exponent
        let mut bits = Vec::with_capacity(256);
        for limb_idx in 0..4 {
            let mut e = exp.v[limb_idx];
            let n_bits = if limb_idx == 3 { 60 } else { 64 }; // 252 bits total
            for _ in 0..n_bits {
                bits.push(e & 1);
                e >>= 1;
            }
        }
        // Trim trailing zeros
        while bits.last() == Some(&0) { bits.pop(); }

        // Square-and-multiply from MSB to LSB (more efficient)
        let mut result = Fp::ONE;
        for &bit in bits.iter().rev() {
            result = result.mul(result);
            if bit == 1 {
                result = result.mul(self);
            }
        }
        result
    }

    /// Negation: p - self
    pub fn neg(self) -> Self {
        if self == Self::ZERO { return Self::ZERO; }
        PRIME.sub(self)
    }

    /// Check if zero.
    pub fn is_zero(&self) -> bool {
        self.v == [0, 0, 0, 0]
    }
}

/// Reduce a 512-bit value mod p using signed i128 limbs for safe borrow propagation.
fn reduce_512(full: &[u64; 8]) -> Fp {
    // Use signed arithmetic to handle borrows cleanly
    let mut w = [0i128; 9];
    for i in 0..8 { w[i] = full[i] as i128; }

    // Reduce high limbs by subtracting q * p shifted
    let p3 = PRIME.v[3] as i128;
    for top in (4..8).rev() {
        while w[top] > 0 {
            let q = w[top] / p3; // quotient estimate
            let q = if q == 0 { 1 } else { q }; // at least 1 to make progress
            let shift = (top - 3) as usize;
            // Subtract q * p shifted
            for i in 0..4 {
                let idx = i + shift;
                if idx < 9 {
                    w[idx] -= q * PRIME.v[i] as i128;
                }
            }
            // Propagate borrows
            for i in 0..8 {
                if w[i] < 0 {
                    let borrow = ((-w[i]) + ((1i128 << 64) - 1)) >> 64;
                    w[i] += borrow << 64;
                    w[i + 1] -= borrow;
                }
                if w[i] >= (1i128 << 64) {
                    let carry = w[i] >> 64;
                    w[i] -= carry << 64;
                    w[i + 1] += carry;
                }
            }
        }
    }

    // Add p until non-negative
    while w[3] < 0 || w[4] != 0 {
        if w[3] < 0 || w[4] < 0 {
            for i in 0..4 { w[i] += PRIME.v[i] as i128; }
            for i in 0..4 {
                if w[i] >= (1i128 << 64) {
                    let carry = w[i] >> 64;
                    w[i] -= carry << 64;
                    w[i + 1] += carry;
                }
            }
        } else {
            break;
        }
    }

    let mut result = Fp { v: [w[0] as u64, w[1] as u64, w[2] as u64, w[3] as u64] };
    result.reduce();
    result
}

impl Add for Fp {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Fp::add(self, rhs) }
}

impl Sub for Fp {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Fp::sub(self, rhs) }
}

impl Mul for Fp {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self { Fp::mul(self, rhs) }
}

impl Neg for Fp {
    type Output = Self;
    fn neg(self) -> Self { Fp::neg(self) }
}

/// Point on the STARK elliptic curve: y² = x³ + x + β
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurvePoint {
    Infinity,
    Affine(Fp, Fp), // (x, y)
}

/// Curve parameter β
pub fn curve_beta() -> Fp {
    Fp::from_hex("06f21413efbe40de150e596d72f7a8c5609ad26c15c915c1f4cdfcb99cee9e89")
}

impl CurvePoint {
    /// Point addition on the STARK curve.
    pub fn add(self, rhs: Self) -> Self {
        match (self, rhs) {
            (CurvePoint::Infinity, p) | (p, CurvePoint::Infinity) => p,
            (CurvePoint::Affine(x1, y1), CurvePoint::Affine(x2, y2)) => {
                if x1 == x2 {
                    if y1 == y2 {
                        // Point doubling
                        if y1.is_zero() { return CurvePoint::Infinity; }
                        // λ = (3x² + a) / (2y),  a = 1 for STARK curve
                        let x1_sq = x1 * x1;
                        let three_x1_sq = x1_sq + x1_sq + x1_sq;
                        let num = three_x1_sq + Fp::ONE; // +a where a=1
                        let denom = y1 + y1;
                        let lambda = num * denom.inverse();
                        let x3 = lambda * lambda - x1 - x2;
                        let y3 = lambda * (x1 - x3) - y1;
                        CurvePoint::Affine(x3, y3)
                    } else {
                        // x1 == x2, y1 != y2: point at infinity
                        CurvePoint::Infinity
                    }
                } else {
                    // Standard addition
                    let lambda = (y2 - y1) * (x2 - x1).inverse();
                    let x3 = lambda * lambda - x1 - x2;
                    let y3 = lambda * (x1 - x3) - y1;
                    CurvePoint::Affine(x3, y3)
                }
            }
        }
    }

    /// Scalar multiplication via double-and-add.
    /// Scalar is a 252-bit value.
    pub fn scalar_mul(self, scalar: Fp) -> Self {
        let mut result = CurvePoint::Infinity;
        let mut base = self;

        for limb_idx in 0..4 {
            let mut s = scalar.v[limb_idx];
            let bits = if limb_idx == 3 { 60 } else { 64 }; // 252 bits total
            for _ in 0..bits {
                if s & 1 == 1 {
                    result = result.add(base);
                }
                base = base.add(base); // double
                s >>= 1;
            }
        }
        result
    }
}

/// The 5 Pedersen constant points.
pub fn pedersen_points() -> [CurvePoint; 5] {
    [
        CurvePoint::Affine(
            Fp::from_hex("49ee3eba8c1600700ee1b87eb599f16716b0b1022947733551fde4050ca6804"),
            Fp::from_hex("3ca0cfe4b3bc6ddf346d49d06ea0ed34e621062c0e056c1d0405d266e10268a"),
        ),
        CurvePoint::Affine(
            Fp::from_hex("234287dcbaffe7f969c748655fca9e58fa8120b6d56eb0c1080d17957ebe47b"),
            Fp::from_hex("3b056f100f96fb21e889527d41f4e39940135dd7a6c7e6c2f8116572f578e85"),
        ),
        CurvePoint::Affine(
            Fp::from_hex("4fa56f376c83db33f9dab2656558f3399099ec1de5e3018b7571f510a2c7768"),
            Fp::from_hex("3f42a042e45b8a3e3821a7133325bfa989e2bc26485dbe63ac6eadc28fc2fad"),
        ),
        CurvePoint::Affine(
            Fp::from_hex("4ba4cc166be8dec764910f75b45f74b40642ad9b32d50d8865e3e7caa740577"),
            Fp::from_hex("00416a975392d0e71777ab65e5e7e4c54daee0efbb7d00b8d2ccacfefa2d8e1c"),
        ),
        CurvePoint::Affine(
            Fp::from_hex("54302dcb0e6cc1c6e44cca8f61a63bb2ca65048d53fb325d36ff12c49a58202"),
            Fp::from_hex("01b77b3e37d13504b348046268d8ae25ce98ad783c25561a879dcc77e99c2426"),
        ),
    ]
}

/// Compute Pedersen hash H(a, b).
/// H(a, b) = [P₀ + a_low·P₁ + a_high·P₂ + b_low·P₃ + b_high·P₄]_x
pub fn pedersen_hash(a: Fp, b: Fp) -> Fp {
    let points = pedersen_points();

    // Decompose a: a_low = lowest 248 bits, a_high = top 4 bits
    let a_low = mask_bits(&a, 248);
    let a_high = shift_right(&a, 248);

    // Decompose b similarly
    let b_low = mask_bits(&b, 248);
    let b_high = shift_right(&b, 248);

    // H = P₀ + a_low·P₁ + a_high·P₂ + b_low·P₃ + b_high·P₄
    let mut result = points[0];
    result = result.add(points[1].scalar_mul(a_low));
    result = result.add(points[2].scalar_mul(a_high));
    result = result.add(points[3].scalar_mul(b_low));
    result = result.add(points[4].scalar_mul(b_high));

    match result {
        CurvePoint::Affine(x, _) => x,
        CurvePoint::Infinity => Fp::ZERO,
    }
}

/// Mask the lowest `n_bits` of a field element.
fn mask_bits(val: &Fp, n_bits: usize) -> Fp {
    let mut result = *val;
    let full_limbs = n_bits / 64;
    let remaining = n_bits % 64;

    for i in (full_limbs + 1)..4 {
        result.v[i] = 0;
    }
    if full_limbs < 4 && remaining > 0 {
        result.v[full_limbs] &= (1u64 << remaining) - 1;
    } else if full_limbs < 4 {
        result.v[full_limbs] = 0;
    }
    result
}

/// Shift right by `n_bits`.
fn shift_right(val: &Fp, n_bits: usize) -> Fp {
    let limb_shift = n_bits / 64;
    let bit_shift = n_bits % 64;
    let mut result = Fp::ZERO;

    for i in 0..4 {
        let src = i + limb_shift;
        if src < 4 {
            result.v[i] = val.v[src] >> bit_shift;
            if bit_shift > 0 && src + 1 < 4 {
                result.v[i] |= val.v[src + 1] << (64 - bit_shift);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp_add() {
        let a = Fp::from_u64(42);
        let b = Fp::from_u64(58);
        let c = a + b;
        assert_eq!(c, Fp::from_u64(100));
    }

    #[test]
    fn test_fp_sub() {
        let a = Fp::from_u64(100);
        let b = Fp::from_u64(42);
        let c = a - b;
        assert_eq!(c, Fp::from_u64(58));
    }

    #[test]
    fn test_fp_sub_underflow() {
        let a = Fp::from_u64(1);
        let b = Fp::from_u64(2);
        let c = a - b; // should be p - 1
        assert_eq!(c, PRIME - Fp::from_u64(1));
    }

    #[test]
    fn test_fp_mul() {
        let a = Fp::from_u64(7);
        let b = Fp::from_u64(6);
        let c = a * b;
        assert_eq!(c, Fp::from_u64(42));
    }

    #[test]
    fn test_fp_mul_large() {
        let a = Fp::from_u64(1u64 << 32);
        let b = Fp::from_u64(1u64 << 32);
        let c = a * b;
        assert_eq!(c, Fp::from_u64(1u64 << 63).add(Fp::from_u64(1u64 << 63)));
        // 2^32 * 2^32 = 2^64, which spans two limbs
    }

    #[test]
    fn test_fp_pow_small() {
        let a = Fp::from_u64(3);
        let a_cubed = a.pow_fp(Fp::from_u64(3)); // 3^3 = 27
        assert_eq!(a_cubed, Fp::from_u64(27), "3^3 should be 27");
    }

    #[test]
    fn test_fp_pow_larger() {
        let a = Fp::from_u64(2);
        let a_10 = a.pow_fp(Fp::from_u64(10)); // 2^10 = 1024
        assert_eq!(a_10, Fp::from_u64(1024));
    }

    #[test]
    fn test_fp_mul_mod() {
        // Test that (p-1) * 2 = p - 2 (since 2*(p-1) = 2p - 2 ≡ -2 mod p ≡ p - 2)
        let pm1 = Fp { v: [0, 0, 0, 0x0800000000000011] }; // p - 1... wait
        // p = [1, 0, 0, 0x0800000000000011]
        // p-1 = [0, 0, 0, 0x0800000000000011] — NO, that's wrong
        // p-1: limb[0] = 1-1 = 0, rest same
        let pm1 = Fp { v: [0, 0, 0, 0x0800000000000011] };
        let result = pm1 * Fp::from_u64(2);
        // 2*(p-1) = 2p - 2 ≡ -2 ≡ p-2 mod p
        let expected = Fp { v: [
            0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF,
            0xFFFF_FFFF_FFFF_FFFF, 0x0800000000000010
        ]};
        assert_eq!(result, expected, "2*(p-1) should equal p-2");
    }

    #[test]
    fn test_fp_p_minus_2() {
        // Verify p - 2 is correct
        let pm2 = PRIME.sub(Fp::from_u64(2));
        let check = pm2.add(Fp::from_u64(2));
        assert_eq!(check, PRIME, "p - 2 + 2 should equal p (which reduces to 0)");
        // p - 2 + 2 reduces to 0 since p ≡ 0 mod p
        assert_eq!(check, Fp::ZERO, "p should equal 0 in Fp");
    }

    #[test]
    fn test_fp_mul_near_prime() {
        // (p-1)^2 mod p = (-1)^2 mod p = 1
        let pm1 = Fp { v: [0, 0, 0, 0x0800000000000011] };
        let result = pm1 * pm1;
        assert_eq!(result, Fp::ONE, "(p-1)^2 should be 1");
    }

    #[test]
    fn test_fermat_little() {
        // 2^(p-1) ≡ 1 mod p (Fermat's little theorem)
        let two = Fp::from_u64(2);
        let pm1 = Fp { v: [0, 0, 0, 0x0800000000000011] }; // p - 1
        let result = two.pow_fp(pm1);
        assert_eq!(result, Fp::ONE, "2^(p-1) should be 1 mod p");
    }

    #[test]
    fn test_fp_inverse_via_small_exp() {
        // Test with a small prime where we can verify
        // In F_p, 2^(p-1) ≡ 1 (Fermat's little theorem)
        // So 2^(p-2) ≡ 2^(-1) ≡ (p+1)/2
        let two = Fp::from_u64(2);
        let inv_two = two.inverse();
        // 2 * inv_two should be 1
        let product = two * inv_two;
        assert_eq!(product, Fp::ONE, "2 * 2^(-1) should be 1");
    }

    #[test]
    fn test_fp_inverse() {
        let a = Fp::from_u64(42);
        let inv = a.inverse();
        let product = a * inv;
        assert_eq!(product, Fp::ONE, "42 * 42^(-1) should be 1");
    }

    #[test]
    fn test_fp_inverse_large() {
        let a = Fp::from_hex("deadbeef");
        let inv = a.inverse();
        let product = a * inv;
        assert_eq!(product, Fp::ONE);
    }

    #[test]
    fn test_fp_neg() {
        let a = Fp::from_u64(42);
        let neg_a = -a;
        let sum = a + neg_a;
        assert_eq!(sum, Fp::ZERO);
    }

    #[test]
    fn test_curve_point_add_double() {
        let p = pedersen_points()[0];
        let doubled = p.add(p);
        // Just verify it doesn't crash and produces a valid point
        match doubled {
            CurvePoint::Affine(x, y) => {
                assert!(!x.is_zero() || !y.is_zero(), "doubled point shouldn't be zero");
            }
            CurvePoint::Infinity => panic!("doubled point shouldn't be infinity"),
        }
    }

    #[test]
    fn test_scalar_mul_identity() {
        let p = pedersen_points()[1];
        let result = p.scalar_mul(Fp::ONE);
        assert_eq!(result, p, "P * 1 should equal P");
    }

    #[test]
    fn test_scalar_mul_two() {
        let p = pedersen_points()[1];
        let doubled = p.add(p);
        let result = p.scalar_mul(Fp::from_u64(2));
        assert_eq!(result, doubled, "P * 2 should equal P + P");
    }

    #[test]
    fn test_pedersen_hash_deterministic() {
        let a = Fp::from_u64(100);
        let b = Fp::from_u64(200);
        let h1 = pedersen_hash(a, b);
        let h2 = pedersen_hash(a, b);
        assert_eq!(h1, h2, "Pedersen hash should be deterministic");
        assert!(!h1.is_zero(), "hash should be non-zero");
    }

    #[test]
    fn test_pedersen_hash_different_inputs() {
        let h1 = pedersen_hash(Fp::from_u64(1), Fp::from_u64(2));
        let h2 = pedersen_hash(Fp::from_u64(3), Fp::from_u64(4));
        assert_ne!(h1, h2, "different inputs should give different hashes");
    }

    #[test]
    fn test_point_on_curve() {
        // Verify P₀ is on the curve: y² = x³ + x + β
        let p0 = pedersen_points()[0];
        if let CurvePoint::Affine(x, y) = p0 {
            let y_sq = y * y;
            let x_cu = x * x * x;
            let rhs = x_cu + x + curve_beta();
            assert_eq!(y_sq, rhs, "P₀ should be on the STARK curve");
        }
    }
}
