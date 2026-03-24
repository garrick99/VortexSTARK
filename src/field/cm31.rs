//! CM31: Complex extension of M31. Elements are a + bi where i^2 = -1.

use super::m31::M31;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// CM31 = M31[i] / (i^2 + 1)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub struct CM31 {
    pub a: M31, // real
    pub b: M31, // imaginary
}

impl CM31 {
    pub const ZERO: Self = Self {
        a: M31::ZERO,
        b: M31::ZERO,
    };
    pub const ONE: Self = Self {
        a: M31::ONE,
        b: M31::ZERO,
    };

    #[inline]
    pub fn new(a: M31, b: M31) -> Self {
        Self { a, b }
    }

    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            a: self.a,
            b: -self.b,
        }
    }

    /// Norm: a^2 + b^2 (since i^2 = -1)
    #[inline]
    pub fn norm(self) -> M31 {
        self.a * self.a + self.b * self.b
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let inv_norm = self.norm().inverse();
        let conj = self.conjugate();
        Self {
            a: conj.a * inv_norm,
            b: conj.b * inv_norm,
        }
    }
}

impl Add for CM31 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl Sub for CM31 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

// (a+bi)(c+di) = (ac-bd) + (ad+bc)i
impl Mul for CM31 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            a: self.a * rhs.a - self.b * rhs.b,
            b: self.a * rhs.b + self.b * rhs.a,
        }
    }
}

impl Mul<M31> for CM31 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: M31) -> Self {
        Self {
            a: self.a * rhs,
            b: self.b * rhs,
        }
    }
}

impl Neg for CM31 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            a: -self.a,
            b: -self.b,
        }
    }
}

impl fmt::Debug for CM31 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CM31({} + {}i)", self.a, self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul() {
        let a = CM31::new(M31(3), M31(4));
        let b = CM31::new(M31(5), M31(6));
        let c = a * b;
        // (3+4i)(5+6i) = 15-24 + (18+20)i = -9 + 38i
        // -9 mod P = P - 9
        assert_eq!(c.a, M31(super::super::m31::P - 9));
        assert_eq!(c.b, M31(38));
    }

    #[test]
    fn test_inverse() {
        let a = CM31::new(M31(123), M31(456));
        let inv = a.inverse();
        let prod = a * inv;
        assert_eq!(prod, CM31::ONE);
    }
}
