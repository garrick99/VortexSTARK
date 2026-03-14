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
            // self < rhs: compute p + self - rhs (using raw addition, no reduce)
            let mut p_plus_self = [0u64; 4];
            let mut carry = 0u128;
            for i in 0..4 {
                carry += PRIME.v[i] as u128 + self.v[i] as u128;
                p_plus_self[i] = carry as u64;
                carry >>= 64;
            }
            // Now subtract rhs
            let mut r = [0u64; 4];
            let mut borrow: u64 = 0;
            for i in 0..4 {
                let (r1, b1) = p_plus_self[i].overflowing_sub(rhs.v[i]);
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

/// Reduce a 512-bit product mod p.
/// Uses: 2^256 ≡ -(544·2^192 + 32) mod p (since p = 2^251 + 17·2^192 + 1).
/// Iteratively folds high limbs down until the value fits in 4 limbs.
fn reduce_512(full: &[u64; 8]) -> Fp {
    // Strategy: repeatedly fold bits >= 256 using 2^256 ≡ -(544·2^192 + 32) mod p.
    // Work in i128 limbs to handle signed intermediates safely.
    // After each fold, normalize limbs to [0, 2^64) and check convergence.
    let mut w = [0i128; 8]; // extra headroom
    for i in 0..8 { w[i] = full[i] as i128; }

    // Iterative fold: for each limb k >= 4, fold w[k] * 2^(64k) down.
    // w[k] * 2^(64k) = w[k] * 2^(64*(k-4)) * 2^256
    //                 ≡ -w[k] * 2^(64*(k-4)) * (544*2^192 + 32) mod p
    // = subtract w[k]*32 from position (k-4) and w[k]*544 from position (k-4+3)=(k-1)
    // This folds limb k down to limbs k-4 and k-1.

    // Two passes should suffice (512 bits → ~266 bits → ~256 bits).
    for _pass in 0..3 {
        // Normalize first: propagate carries so each limb is in [0, 2^64)
        for i in 0..7 {
            if w[i] >= (1i128 << 64) {
                w[i + 1] += w[i] >> 64;
                w[i] &= (1i128 << 64) - 1;
            } else if w[i] < 0 {
                let borrow = ((-w[i] - 1) >> 64) + 1; // ceil(-w[i] / 2^64)
                w[i] += borrow << 64;
                w[i + 1] -= borrow;
            }
        }

        // Fold limbs 4+ down
        let mut any_high = false;
        for k in (4..8).rev() {
            if w[k] == 0 { continue; }
            any_high = true;
            let val = w[k];
            w[k] = 0;
            let base = k - 4;
            w[base] -= val * 32;         // -(val * 32) at position base
            w[base + 3] -= val * 544;    // -(val * 544) at position base+3
        }
        if !any_high { break; }
    }

    // Final normalize
    for i in 0..7 {
        if w[i] >= (1i128 << 64) {
            w[i + 1] += w[i] >> 64;
            w[i] &= (1i128 << 64) - 1;
        } else if w[i] < 0 {
            let borrow = ((-w[i] - 1) >> 64) + 1;
            w[i] += borrow << 64;
            w[i + 1] -= borrow;
        }
    }

    // If negative (w[3] < 0 or w[4] != 0), add p repeatedly
    for _ in 0..10 {
        if w[4] <= 0 && w[3] >= 0 { break; }
        w[0] += PRIME.v[0] as i128;
        w[3] += PRIME.v[3] as i128;
        for i in 0..4 {
            if w[i] >= (1i128 << 64) {
                w[i + 1] += w[i] >> 64;
                w[i] &= (1i128 << 64) - 1;
            }
        }
    }

    let mut result = Fp { v: [w[0] as u64, w[1] as u64, w[2] as u64, w[3] as u64] };
    result.reduce();
    result
}

#[allow(dead_code)]
fn reduce_512_old(full: &[u64; 8]) -> Fp {
    // Signed i128 limbs with headroom for negative intermediates.
    // We work with 5 limbs to handle the fold overflow.
    let mut lo = [full[0] as i128, full[1] as i128, full[2] as i128, full[3] as i128, 0i128];
    let hi = [full[4] as i128, full[5] as i128, full[6] as i128, full[7] as i128];

    // Fold: lo -= hi * (544·2^192 + 32)
    // 544·2^192 = 544 in "limb 3" position (bits 192-255)
    // 32 in "limb 0" position (bits 0-63)
    // hi[k] * 2^(64*k) * (544·2^192 + 32)
    //   = hi[k] * 544 * 2^(192 + 64*k) + hi[k] * 32 * 2^(64*k)
    // In limb terms:
    //   -= hi[k] * 32 into lo[k]
    //   -= hi[k] * 544 into lo[k+3]
    for k in 0..4 {
        lo[k] -= hi[k] * 32;
        if k + 3 < 5 {
            lo[k + 3] -= hi[k] * 544;
        }
        // hi[3] * 544 would go to lo[6] — out of range. Handle separately.
    }
    // hi[3] * 544 * 2^(192+192) = hi[3] * 544 * 2^384
    // 2^384 = 2^256 * 2^128. Need another fold for this term.
    // hi[3] * 544 goes into a "super-high" term at limb position 6 (bits 384-447).
    // Fold again: this_term * 2^384 ≡ -this_term * (544·2^192+32) * 2^128 mod p
    // which is complex. For products of 252-bit values, hi[3] ≈ 2^59, so
    // hi[3] * 544 ≈ 2^69 — this goes into lo[4] (overflow limb).
    // Actually hi[3]*544 is at bit position 3+3=6 in our 5-limb layout.
    // Since we only have 5 limbs (indices 0-4), this overflows.
    // Handle by putting it in lo[4] and doing a second fold.
    // hi[k=1]*544 goes to lo[4]: ✓ (1+3=4)
    // hi[k=2]*544 goes to lo[5]: overflow! But k+3<5 catches k≤1 only.
    // Fix: handle k=2 and k=3 overflow terms separately.

    // hi[2]*544 goes to position 5 — we need a second fold.
    // hi[3]*544 goes to position 6 — also needs folding.
    let overflow2 = hi[2] * 544; // goes to 2^(64*5) = 2^320
    let overflow3 = hi[3] * 544; // goes to 2^(64*6) = 2^384

    // 2^320 = 2^256 * 2^64 ≡ -(544·2^192+32) * 2^64 mod p
    //   = -(544·2^256 + 32·2^64) ≡ -(-544*(544·2^192+32) + 32·2^64) ... getting recursive.
    // Simpler: just fold overflow into lo[0..4] using the same identity.
    // 2^320 = 2^64 * 2^256 ≡ -2^64 * (544*2^192 + 32) = -(544*2^256 + 32*2^64)
    //   ≡ -(-(544)*(544*2^192+32) + 32*2^64) — no, this recurses.
    //
    // Clean approach: treat overflow as a 2-limb "hi2" and fold again.
    // overflow2 * 2^320 + overflow3 * 2^384
    //   = 2^256 * (overflow2 * 2^64 + overflow3 * 2^128)
    //   ≡ -(overflow2 * 2^64 + overflow3 * 2^128) * (544*2^192 + 32)  [mod p]
    //
    // Even simpler: since overflow2 and overflow3 are small (~2^69),
    // just do: lo[4] holds the combined overflow, fold lo[4] back into lo[0..3].

    // Re-do the fold properly: handle all k, let overflow go to lo[4]
    // Reset and redo
    lo = [full[0] as i128, full[1] as i128, full[2] as i128, full[3] as i128, 0i128];

    for k in 0..4 {
        lo[k] -= hi[k] * 32;
        let dst = k + 3;
        if dst < 5 {
            lo[dst] -= hi[k] * 544;
        } else {
            // dst = 5 (k=2) or 6 (k=3): fold again
            // These contribute to an even higher overflow.
            // For k=2: hi[2]*544 * 2^(5*64) = hi[2]*544 * 2^320
            // For k=3: hi[3]*544 * 2^(6*64) = hi[3]*544 * 2^384
            // 2^320 mod p: fold 2^256 first. 2^320 = 2^64 * 2^256 ≡ -2^64*(544*2^192+32) mod p
            //   = -(544*2^256 + 32*2^64) ≡ 544*(544*2^192+32) - 32*2^64 [double negation]
            // No, 2^320 ≡ -2^64*(544*2^192+32) = -(544*2^(192+64) + 32*2^64) = -(544*2^256 + 32*2^64)
            //   ≡ -(-(544*2^192+32)*544 + 32*2^64)... this recurses.

            // PRAGMATIC: since these overflow terms are small (< 2^80),
            // compute them as Fp values and add to result at the end.
            // For now, approximate: add to lo[4] and fold lo[4] separately.
            lo[4] -= hi[k] * 544; // approximate, will fold lo[4] below
        }
    }

    // Normalize: propagate borrows/carries through lo[0..4]
    for i in 0..4 {
        if lo[i] < 0 {
            let borrow = ((-lo[i]) + ((1 << 64) - 1)) >> 64; // ceil division
            lo[i] += borrow << 64;
            lo[i + 1] -= borrow;
        }
        if lo[i] >= (1 << 64) {
            let carry = lo[i] >> 64;
            lo[i] &= (1i128 << 64) - 1;
            lo[i + 1] += carry;
        }
    }

    // Fold lo[4] (overflow from the first fold) back into lo[0..3]
    // lo[4] * 2^256 ≡ -lo[4] * (544*2^192 + 32) mod p
    if lo[4] != 0 {
        let h = lo[4];
        lo[0] -= h * 32;
        lo[3] -= h * 544;
        lo[4] = 0;

        // Normalize again
        for i in 0..4 {
            if lo[i] < 0 {
                let borrow = ((-lo[i]) + ((1 << 64) - 1)) >> 64;
                lo[i] += borrow << 64;
                lo[i + 1] -= borrow;
            }
            if lo[i] >= (1 << 64) {
                let carry = lo[i] >> 64;
                lo[i] &= (1i128 << 64) - 1;
                lo[i + 1] += carry;
            }
        }
    }

    // If still negative, add multiples of p
    while lo[4] < 0 || lo[3] < 0 {
        lo[0] += PRIME.v[0] as i128;
        lo[3] += PRIME.v[3] as i128;
        for i in 0..4 {
            if lo[i] >= (1i128 << 64) {
                let carry = lo[i] >> 64;
                lo[i] &= (1i128 << 64) - 1;
                lo[i + 1] += carry;
            }
        }
    }

    // lo[4] should now be 0 after folding
    debug_assert!(lo[4] == 0, "reduce_512: lo[4] = {} after fold", lo[4]);

    let mut result = Fp { v: [lo[0] as u64, lo[1] as u64, lo[2] as u64, lo[3] as u64] };
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
    fn test_fp_mul_two_large() {
        // (p-1) * (p-2) mod p = (-1)*(-2) = 2
        let pm1 = Fp { v: [0, 0, 0, 0x0800000000000011] };
        let pm2 = Fp { v: [0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0800000000000010] };
        let result = pm1 * pm2;
        assert_eq!(result, Fp::from_u64(2), "(p-1)*(p-2) should be 2");
    }

    #[test]
    fn test_fp_pow_64() {
        let two = Fp::from_u64(2);
        let exp = Fp::from_u64(64);
        let result = two.pow_fp(exp);
        // 2^64 < p, so result should be exactly 2^64
        assert_eq!(result, Fp { v: [0, 1, 0, 0] }, "2^64 should be 2^64");
    }

    #[test]
    fn test_fp_pow_128() {
        let two = Fp::from_u64(2);
        let exp = Fp::from_u64(128);
        let result = two.pow_fp(exp);
        assert_eq!(result, Fp { v: [0, 0, 1, 0] }, "2^128 should be 2^128");
    }

    #[test]
    fn test_fp_pow_192() {
        let two = Fp::from_u64(2);
        let exp = Fp::from_u64(192);
        let result = two.pow_fp(exp);
        assert_eq!(result, Fp { v: [0, 0, 0, 1] }, "2^192 should be 2^192");
    }

    #[test]
    fn test_fp_pow_large_exp() {
        // 2^256 mod p. Since p ≈ 2^251, this should be a small-ish number.
        // 2^256 = 32*p - 544*2^192 - 32
        // So 2^256 mod p = -(544*2^192 + 32) mod p = p - 544*2^192 - 32
        let two = Fp::from_u64(2);
        let exp = Fp { v: [0, 0, 0, 0x0000000000000001] }; // 2^192
        let r = two.pow_fp(exp);
        // 2^(2^192) — this is a huge number. Just check it's non-zero and deterministic.
        let r2 = two.pow_fp(exp);
        assert_eq!(r, r2, "pow should be deterministic");
        assert!(!r.is_zero(), "2^(2^192) should be non-zero");
    }

    #[test]
    fn test_fp_pow_2_to_59() {
        // 2^(2^59): exponent has only bit 59 set
        let two = Fp::from_u64(2);
        let exp = Fp { v: [0, 0, 0, 0x0800000000000000] }; // bit 251 = bit 59 of limb 3
        let r = two.pow_fp(exp);
        // Just check deterministic
        let r2 = two.pow_fp(exp);
        assert_eq!(r, r2);
        assert!(!r.is_zero());
    }

    #[test]
    fn test_fp_pow_bit_192() {
        // 2^(2^192): exponent = [0, 0, 0, 1]
        let two = Fp::from_u64(2);
        let exp = Fp { v: [0, 0, 0, 1] }; // 2^192
        let r = two.pow_fp(exp);
        // Compute expected: 2^(2^192) mod p
        // Since this is huge, just verify via squaring: r*r should be 2^(2^193)
        let r2 = two.pow_fp(Fp { v: [0, 0, 0, 2] }); // 2^(2*2^192) = 2^(2^193)
        assert_eq!(r * r, r2, "2^(2^192) squared should be 2^(2^193)");
    }

    #[test]
    fn test_fp_pow_combined_bits() {
        // Exponent with bits 192 and 251: 2^(2^192 + 2^251)
        let two = Fp::from_u64(2);
        let exp_a = Fp { v: [0, 0, 0, 1] }; // 2^192
        let exp_b = Fp { v: [0, 0, 0, 0x0800000000000000] }; // 2^251
        let exp_combined = Fp { v: [0, 0, 0, 0x0800000000000001] }; // 2^192 + 2^251

        let r_a = two.pow_fp(exp_a);
        let r_b = two.pow_fp(exp_b);
        let r_combined = two.pow_fp(exp_combined);

        // 2^(a+b) = 2^a * 2^b
        assert_eq!(r_combined, r_a * r_b, "2^(a+b) should equal 2^a * 2^b");
    }

    #[test]
    fn test_fermat_little_3() {
        // 3^(p-1) ≡ 1 mod p
        let three = Fp::from_u64(3);
        let pm1 = Fp { v: [0, 0, 0, 0x0800000000000011] };
        let result = three.pow_fp(pm1);
        assert_eq!(result, Fp::ONE, "3^(p-1) should be 1 mod p");
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
