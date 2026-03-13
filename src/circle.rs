//! Circle group and domain types over M31.
//!
//! The circle group is { (x,y) : x^2 + y^2 = 1 } over M31.
//! The generator has order 2^31, matching the M31 field structure.

use crate::field::M31;

/// A point on the circle x^2 + y^2 = 1 over M31.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CirclePoint {
    pub x: M31,
    pub y: M31,
}

impl CirclePoint {
    /// Circle group identity: (1, 0).
    pub const IDENTITY: Self = Self {
        x: M31::ONE,
        y: M31::ZERO,
    };

    /// Generator of order 2^31 on the M31 circle group.
    /// G = (2, 1268011823) where G.x^2 + G.y^2 = 1 mod P.
    pub const GENERATOR: Self = Self {
        x: M31(2),
        y: M31(1268011823),
    };

    /// Circle group operation: (x1,y1) * (x2,y2) = (x1*x2 - y1*y2, x1*y2 + y1*x2)
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x - self.y * rhs.y,
            y: self.x * rhs.y + self.y * rhs.x,
        }
    }

    /// Inverse: conjugate (x, -y) since |point| = 1.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            x: self.x,
            y: -self.y,
        }
    }

    /// Double: p * p.
    #[inline]
    pub fn double(self) -> Self {
        self.mul(self)
    }

    /// Repeated squaring: self^(2^n).
    #[inline]
    pub fn repeated_double(self, n: u32) -> Self {
        let mut p = self;
        for _ in 0..n {
            p = p.double();
        }
        p
    }

    /// Scalar multiplication by repeated doubling.
    pub fn mul_scalar(self, mut k: u32) -> Self {
        let mut result = Self::IDENTITY;
        let mut base = self;
        while k > 0 {
            if k & 1 == 1 {
                result = result.mul(base);
            }
            base = base.double();
            k >>= 1;
        }
        result
    }

    /// Antipodal point: (-x, -y).
    #[inline]
    pub fn antipode(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

/// A coset of the circle group: { initial * G^(step*i) : i = 0..size }
/// where G is the circle group generator.
#[derive(Clone, Copy, Debug)]
pub struct Coset {
    pub initial: CirclePoint,
    pub step: CirclePoint,
    pub log_size: u32,
}

impl Coset {
    /// Create a standard power-of-2 coset.
    /// The subgroup of order 2^log_size has generator G^(2^(31-log_size)).
    pub fn subgroup(log_size: u32) -> Self {
        assert!(log_size <= 31);
        let step = CirclePoint::GENERATOR.repeated_double(31 - log_size);
        Self {
            initial: CirclePoint::IDENTITY,
            step,
            log_size,
        }
    }

    /// Half-coset: standard domain used for polynomial evaluation.
    /// Coset = G^(2^(30-log_size)) * subgroup(log_size).
    pub fn half_coset(log_size: u32) -> Self {
        assert!(log_size <= 30);
        let step = CirclePoint::GENERATOR.repeated_double(31 - log_size);
        let initial = CirclePoint::GENERATOR.repeated_double(30 - log_size);
        Self {
            initial,
            step,
            log_size,
        }
    }

    pub fn size(&self) -> usize {
        1 << self.log_size
    }

    /// Get the i-th point of the coset.
    pub fn at(&self, i: usize) -> CirclePoint {
        self.initial.mul(self.step.mul_scalar(i as u32))
    }
}

/// Compute twiddle factors for the Circle NTT.
/// Returns (line_twiddles, circle_twiddles, layer_offsets, layer_sizes).
///
/// Line twiddles: for each layer l (l = n_line_layers-1 down to 0),
///   twiddles[offset[l]..offset[l]+size[l]] = x-coordinates of subgroup points.
///
/// Circle twiddles: y-coordinates for the circle butterfly (layer 0).
pub fn compute_twiddles(
    coset: &Coset,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let log_n = coset.log_size;
    let n = coset.size();
    let n_line_layers = if log_n > 0 { log_n - 1 } else { 0 };

    let mut line_twiddles = Vec::new();
    let mut layer_offsets = Vec::new();
    let mut layer_sizes = Vec::new();

    // Compute all domain points
    let points: Vec<CirclePoint> = (0..n).map(|i| coset.at(i)).collect();

    // Circle twiddles: y-coordinates of first half of points
    let half_n = n / 2;
    let circle_twids: Vec<u32> = (0..half_n).map(|i| points[i].y.0).collect();

    // Line layers: process from the perspective of the butterfly structure
    // After the circle layer, we work with x-coordinates only.
    // For each line layer, twiddles are the x-coordinates at the appropriate stride.

    // Build line twiddles layer by layer
    // Layer l has 2^(log_n - l - 2) twiddle values (half the pairs at that level)
    let mut xs: Vec<M31> = points.iter().map(|p| p.x).collect();

    for _layer in 0..n_line_layers as usize {
        let layer_size = xs.len() / 2;
        let offset = line_twiddles.len();
        layer_offsets.push(offset as u32);
        layer_sizes.push(layer_size as u32);

        // Twiddle for this layer: x-coordinates of the first element of each pair
        for i in 0..layer_size {
            line_twiddles.push(xs[2 * i].0);
        }

        // Squash: x' = 2*x^2 - 1 (circle doubling x-coordinate)
        let new_xs: Vec<M31> = (0..layer_size)
            .map(|i| {
                let x = xs[2 * i];
                x * x + x * x - M31::ONE
            })
            .collect();
        xs = new_xs;
    }

    (line_twiddles, circle_twids, layer_offsets, layer_sizes)
}

/// Compute inverse twiddle factors.
/// For iNTT, twiddles need to be the inverses of the forward twiddles.
pub fn compute_itwiddles(
    coset: &Coset,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let (mut line_twids, mut circle_twids, offsets, sizes) = compute_twiddles(coset);

    // Invert all twiddle values
    for t in &mut line_twids {
        *t = M31(*t).inverse().0;
    }
    for t in &mut circle_twids {
        *t = M31(*t).inverse().0;
    }

    (line_twids, circle_twids, offsets, sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_on_circle() {
        let g = CirclePoint::GENERATOR;
        // x^2 + y^2 should equal 1 (mod P)
        let sum = g.x * g.x + g.y * g.y;
        assert_eq!(sum, M31::ONE, "Generator not on circle");
    }

    #[test]
    fn test_generator_order() {
        // G^(2^31) should be the identity
        let g = CirclePoint::GENERATOR;
        let result = g.repeated_double(31);
        assert_eq!(result, CirclePoint::IDENTITY, "Generator order is not 2^31");
    }

    #[test]
    fn test_subgroup() {
        // Subgroup of order 4: G^(2^29)
        let coset = Coset::subgroup(2);
        assert_eq!(coset.size(), 4);
        // The 4th power should return to identity
        let p = coset.step.mul_scalar(4);
        assert_eq!(p, CirclePoint::IDENTITY);
    }

    #[test]
    fn test_identity() {
        let g = CirclePoint::GENERATOR;
        assert_eq!(g.mul(CirclePoint::IDENTITY), g);
    }

    #[test]
    fn test_conjugate() {
        let g = CirclePoint::GENERATOR;
        let prod = g.mul(g.conjugate());
        assert_eq!(prod, CirclePoint::IDENTITY);
    }
}
