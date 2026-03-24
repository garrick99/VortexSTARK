//! QM31: Secure field extension. Elements are a + bu where u^2 = 2 + i.
//! QM31 = CM31[u] / (u^2 - (2+i))

use super::cm31::CM31;
use super::m31::M31;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// QM31 = CM31 + CM31*u, stored as (a, b) where element = a + b*u.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub struct QM31 {
    pub a: CM31, // "real" CM31 part
    pub b: CM31, // "u" CM31 part
}

impl QM31 {
    pub const ZERO: Self = Self {
        a: CM31::ZERO,
        b: CM31::ZERO,
    };
    pub const ONE: Self = Self {
        a: CM31::ONE,
        b: CM31::ZERO,
    };

    #[inline]
    pub fn new(a: CM31, b: CM31) -> Self {
        Self { a, b }
    }

    /// Create from four M31 components: v0 + v1*i + v2*u + v3*i*u
    #[inline]
    pub fn from_m31_array(v: [M31; 4]) -> Self {
        Self {
            a: CM31::new(v[0], v[1]),
            b: CM31::new(v[2], v[3]),
        }
    }

    /// Extract as four M31 components.
    #[inline]
    pub fn to_m31_array(self) -> [M31; 4] {
        [self.a.a, self.a.b, self.b.a, self.b.b]
    }

    /// Extract as four raw u32 values (for GPU upload).
    #[inline]
    pub fn to_u32_array(self) -> [u32; 4] {
        [self.a.a.0, self.a.b.0, self.b.a.0, self.b.b.0]
    }

    #[inline]
    pub fn from_u32_array(v: [u32; 4]) -> Self {
        Self::from_m31_array([M31(v[0]), M31(v[1]), M31(v[2]), M31(v[3])])
    }

    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            a: self.a,
            b: -self.b,
        }
    }

    /// Norm in CM31: a^2 - b^2*(2+i)
    #[inline]
    pub fn norm(self) -> CM31 {
        let a2 = self.a * self.a;
        let b2 = self.b * self.b;
        // b2 * (2+i): (p+qi)(2+i) = (2p-q) + (p+2q)i
        let b2_u2 = CM31::new(
            b2.a.double() - b2.b,
            b2.a + b2.b.double(),
        );
        a2 - b2_u2
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let norm = self.norm();
        let inv_norm = norm.inverse();
        let conj = self.conjugate();
        // conj * inv_norm (CM31 scalar multiply on each component)
        Self {
            a: conj.a * inv_norm,
            b: conj.b * inv_norm,
        }
    }
}

impl Add for QM31 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl Sub for QM31 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

// (a + bu)(c + du) = (ac + bd*(2+i)) + (ad + bc)u
impl Mul for QM31 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let ac = self.a * rhs.a;
        let bd = self.b * rhs.b;
        let ad = self.a * rhs.b;
        let bc = self.b * rhs.a;
        // bd * (2+i)
        let bd_u2 = CM31::new(
            bd.a.double() - bd.b,
            bd.a + bd.b.double(),
        );
        Self {
            a: ac + bd_u2,
            b: ad + bc,
        }
    }
}

impl Mul<M31> for QM31 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: M31) -> Self {
        Self {
            a: self.a * rhs,
            b: self.b * rhs,
        }
    }
}

impl Mul<CM31> for QM31 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: CM31) -> Self {
        Self {
            a: self.a * rhs,
            b: self.b * rhs,
        }
    }
}

impl Neg for QM31 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            a: -self.a,
            b: -self.b,
        }
    }
}

impl fmt::Debug for QM31 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "QM31({} + {}i + {}u + {}iu)",
            self.a.a, self.a.b, self.b.a, self.b.b
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_sub() {
        let a = QM31::from_m31_array([M31(1), M31(2), M31(3), M31(4)]);
        let b = QM31::from_m31_array([M31(5), M31(6), M31(7), M31(8)]);
        let sum = a + b;
        assert_eq!(sum, QM31::from_m31_array([M31(6), M31(8), M31(10), M31(12)]));
        assert_eq!(sum - b, a);
    }

    #[test]
    fn test_inverse() {
        let a = QM31::from_m31_array([M31(111), M31(222), M31(333), M31(444)]);
        let inv = a.inverse();
        let prod = a * inv;
        assert_eq!(prod, QM31::ONE);
    }

    #[test]
    fn test_u32_roundtrip() {
        let vals = [42u32, 99, 1234, 5678];
        let q = QM31::from_u32_array(vals);
        assert_eq!(q.to_u32_array(), vals);
    }
}
