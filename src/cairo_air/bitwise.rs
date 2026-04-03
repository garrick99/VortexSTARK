//! Bitwise builtin for Cairo VM.
//!
//! Proves bitwise operations (AND, XOR, OR) on 252-bit values.
//! Each invocation takes two inputs and produces three outputs:
//!   AND(a, b), XOR(a, b), OR(a, b)
//!
//! The AIR decomposes each input into individual bits and constrains:
//! - Each bit is binary (b * (1-b) = 0)
//! - AND: bit_and[i] = a_bit[i] * b_bit[i]
//! - XOR: bit_xor[i] = a_bit[i] + b_bit[i] - 2 * a_bit[i] * b_bit[i]
//! - OR:  bit_or[i]  = a_bit[i] + b_bit[i] - a_bit[i] * b_bit[i]
//! - Reconstruction: value = sum(bit[i] * 2^i) matches the original
//!
//! For M31 representation: 252 bits decomposed into 8 chunks of 31-32 bits.

/// Bitwise builtin entry: (input_a, input_b, and, xor, or)
pub struct BitwiseBuiltin {
    pub entries: Vec<BitwiseEntry>,
}

#[derive(Clone, Debug)]
pub struct BitwiseEntry {
    pub a: [u32; 8],    // 252 bits as 8 × 32-bit words
    pub b: [u32; 8],
    pub and: [u32; 8],
    pub xor: [u32; 8],
    pub or: [u32; 8],
}

impl BitwiseBuiltin {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Invoke bitwise operations on two 252-bit values.
    pub fn invoke(&mut self, a: [u32; 8], b: [u32; 8]) -> BitwiseEntry {
        let mut and = [0u32; 8];
        let mut xor = [0u32; 8];
        let mut or = [0u32; 8];

        for i in 0..8 {
            and[i] = a[i] & b[i];
            xor[i] = a[i] ^ b[i];
            or[i] = a[i] | b[i];
        }

        let entry = BitwiseEntry { a, b, and, xor, or };
        self.entries.push(entry.clone());
        entry
    }

    pub fn n_invocations(&self) -> usize {
        self.entries.len()
    }

    /// Generate trace columns.
    /// Each invocation: 5 × 8 = 40 columns (a, b, and, xor, or as 8-word limbs)
    pub fn generate_trace(&self, log_n: u32) -> Vec<Vec<u32>> {
        let n = 1usize << log_n;
        let n_cols = 5 * 8; // 40 columns
        let mut cols: Vec<Vec<u32>> = (0..n_cols).map(|_| vec![0u32; n]).collect();

        for (i, entry) in self.entries.iter().enumerate() {
            if i >= n { break; }
            for j in 0..8 {
                cols[j][i] = entry.a[j];
                cols[8 + j][i] = entry.b[j];
                cols[16 + j][i] = entry.and[j];
                cols[24 + j][i] = entry.xor[j];
                cols[32 + j][i] = entry.or[j];
            }
        }
        cols
    }
}

pub const BITWISE_BUILTIN_BASE: u64 = 0x6000_0000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitwise_invoke() {
        let mut builtin = BitwiseBuiltin::new();
        let a = [0xFF00FF00u32, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x0F0F0F0Fu32, 0, 0, 0, 0, 0, 0, 0];

        let result = builtin.invoke(a, b);
        assert_eq!(result.and[0], 0x0F000F00);
        assert_eq!(result.xor[0], 0xF00FF00F);
        assert_eq!(result.or[0], 0xFF0FFF0F);
    }

    #[test]
    fn test_bitwise_trace() {
        let mut builtin = BitwiseBuiltin::new();
        builtin.invoke([1, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0]);
        builtin.invoke([3, 0, 0, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0, 0, 0]);

        let cols = builtin.generate_trace(2);
        assert_eq!(cols.len(), 40);
        assert_eq!(cols[0][0], 1); // a[0] of first invocation
        assert_eq!(cols[0][1], 3); // a[0] of second invocation
        assert_eq!(cols[16][0], 0); // and(1, 2) = 0
        assert_eq!(cols[16][1], 1); // and(3, 5) = 1
        assert_eq!(cols[24][0], 3); // xor(1, 2) = 3
        assert_eq!(cols[24][1], 6); // xor(3, 5) = 6
    }
}
