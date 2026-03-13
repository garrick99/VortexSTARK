//! M31: Mersenne-31 prime field (p = 2^31 - 1).

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

pub const P: u32 = 0x7FFF_FFFF;

/// A field element in GF(2^31 - 1), stored as u32 in [0, P).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct M31(pub u32);

impl M31 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    #[inline]
    pub fn new(val: u32) -> Self {
        debug_assert!(val < P, "M31 value out of range: {val}");
        Self(val)
    }

    #[inline]
    pub fn reduce(val: u64) -> Self {
        let lo = (val & P as u64) as u32;
        let hi = (val >> 31) as u32;
        let r = lo + hi;
        Self(if r >= P { r - P } else { r })
    }

    #[inline]
    pub fn inverse(self) -> Self {
        // Fermat: a^(p-2) mod p
        self.pow(P - 2)
    }

    #[inline]
    pub fn pow(self, mut exp: u32) -> Self {
        let mut result = Self::ONE;
        let mut base = self;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }
        result
    }

    #[inline]
    pub fn double(self) -> Self {
        let r = self.0 << 1;
        Self(if r >= P { r - P } else { r })
    }
}

impl Add for M31 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let r = self.0 + rhs.0;
        Self(if r >= P { r - P } else { r })
    }
}

impl Sub for M31 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(if self.0 >= rhs.0 {
            self.0 - rhs.0
        } else {
            self.0 + P - rhs.0
        })
    }
}

impl Mul for M31 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::reduce(self.0 as u64 * rhs.0 as u64)
    }
}

impl Neg for M31 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(if self.0 == 0 { 0 } else { P - self.0 })
    }
}

impl fmt::Debug for M31 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "M31({})", self.0)
    }
}

impl fmt::Display for M31 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for M31 {
    fn from(val: u32) -> Self {
        Self::reduce(val as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(M31(5) + M31(3), M31(8));
        assert_eq!(M31(P - 1) + M31(1), M31(0));
        assert_eq!(M31(P - 1) + M31(2), M31(1));
    }

    #[test]
    fn test_sub() {
        assert_eq!(M31(5) - M31(3), M31(2));
        assert_eq!(M31(0) - M31(1), M31(P - 1));
    }

    #[test]
    fn test_mul() {
        assert_eq!(M31(5) * M31(3), M31(15));
        let a = M31(1 << 16);
        assert_eq!(a * a, M31::reduce((1u64 << 32)));
    }

    #[test]
    fn test_inverse() {
        let a = M31(12345);
        let inv = a.inverse();
        assert_eq!(a * inv, M31::ONE);
    }

    #[test]
    fn test_neg() {
        assert_eq!(-M31::ZERO, M31::ZERO);
        assert_eq!(-M31(1), M31(P - 1));
        let a = M31(42);
        assert_eq!(a + (-a), M31::ZERO);
    }
}
