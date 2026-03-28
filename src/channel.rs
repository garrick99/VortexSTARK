//! Fiat-Shamir channel for non-interactive proof generation.
//!
//! Uses Blake2s as the hash function. The prover commits data by mixing it
//! into the channel state, then draws random field elements as challenges.
//! Lightweight — runs on CPU since it's just a few hashes.

use crate::field::QM31;

/// Simple Blake2s-based Fiat-Shamir channel.
///
/// Squeeze convention matches stwo's Blake2sChannel:
///   draw input = state[32] || n_draws_le_u32[4] || 0x00[1]  (37 bytes)
/// This ensures proofs are verifiable by an stwo-compatible verifier.
#[derive(Clone)]
pub struct Channel {
    state: [u8; 32],
    /// Draw counter — u32, matching stwo's n_draws field.
    n_draws: u32,
}

impl Channel {
    pub fn new() -> Self {
        Self {
            state: [0u8; 32],
            n_draws: 0,
        }
    }

    /// Mix a 32-byte commitment (Merkle root) into the channel.
    pub fn mix_digest(&mut self, digest: &[u32; 8]) {
        let mut input = [0u8; 64];
        input[..32].copy_from_slice(&self.state);
        for (i, &w) in digest.iter().enumerate() {
            input[32 + i * 4..32 + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        self.state = blake2s_hash(&input);
        self.n_draws = 0;
    }

    /// Mix QM31 field elements into the channel.
    ///
    /// Matches stwo's Channel::mix_felts: all felts are appended to the current digest
    /// bytes in a single Blake2s call (not one hash per felt).
    /// Input = state[32] || f0_u0..f0_u3 || f1_u0..f1_u3 || ...  (each u32 LE, 4 bytes)
    /// Uses the `blake2` crate for correct multi-block hashing (input > 64 bytes).
    pub fn mix_felts(&mut self, felts: &[QM31]) {
        if felts.is_empty() { return; }
        use blake2::{Blake2s256, Digest as _};
        let mut hasher = Blake2s256::new();
        hasher.update(&self.state);
        for f in felts {
            for &w in &f.to_u32_array() {
                hasher.update(&w.to_le_bytes());
            }
        }
        let result: [u8; 32] = hasher.finalize().into();
        self.state = result;
        self.n_draws = 0;
    }

    /// Draw a random QM31 element from the channel.
    ///
    /// Matches stwo's draw_secure_felt: rejection-samples the 8 u32s from one
    /// squeeze until all are in [0, 2P), then reduces. Uses first 4 of the 8.
    /// Rejection probability per round ≈ 2^{-28}; practically never retries.
    pub fn draw_felt(&mut self) -> QM31 {
        const P: u32 = crate::field::m31::P;
        loop {
            let bytes = self.squeeze();
            let u32s: [u32; 8] = std::array::from_fn(|i| {
                u32::from_le_bytes(bytes[i * 4..(i + 1) * 4].try_into().unwrap())
            });
            if u32s.iter().all(|&x| x < 2 * P) {
                let v = [
                    u32s[0] % P,
                    u32s[1] % P,
                    u32s[2] % P,
                    u32s[3] % P,
                ];
                return QM31::from_u32_array(v);
            }
        }
    }

    /// Draw multiple random QM31 elements.
    pub fn draw_felts(&mut self, n: usize) -> Vec<QM31> {
        (0..n).map(|_| self.draw_felt()).collect()
    }

    /// Draw a random number in [0, bound).
    pub fn draw_number(&mut self, bound: usize) -> usize {
        let bytes = self.squeeze();
        let raw = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        (raw % bound as u64) as usize
    }

    /// Mix a u64 (PoW nonce) into the channel state.
    /// Matches stwo: mix_u32s(&[lo, hi]) = Blake2s(state || lo_le || hi_le).
    pub fn mix_u64(&mut self, value: u64) {
        let lo = value as u32;
        let hi = (value >> 32) as u32;
        let mut input = [0u8; 40];
        input[..32].copy_from_slice(&self.state);
        input[32..36].copy_from_slice(&lo.to_le_bytes());
        input[36..40].copy_from_slice(&hi.to_le_bytes());
        self.state = blake2s_hash(&input);
        self.n_draws = 0;
    }

    /// Return the current channel state as 8 u32 words (for GPU PoW prefix computation).
    pub fn state_words(&self) -> [u32; 8] {
        let mut out = [0u32; 8];
        for (i, chunk) in self.state.chunks_exact(4).enumerate() {
            out[i] = u32::from_le_bytes(chunk.try_into().unwrap());
        }
        out
    }

    /// Verify that Blake2s(prefix_digest || nonce_le) has >= pow_bits trailing zeros
    /// in its first 16 bytes (u128 LE), matching stwo's Blake2s PoW convention.
    ///
    /// prefix_digest = Blake2s(0x12345678_LE || [0u8;12] || self.state || pow_bits_LE)
    pub fn verify_pow_nonce(&self, pow_bits: u32, nonce: u64) -> bool {
        // Compute prefix_digest (52-byte input)
        let mut prefix_input = [0u8; 52];
        prefix_input[0..4].copy_from_slice(&0x12345678u32.to_le_bytes());
        // bytes 4..16 are zero padding
        prefix_input[16..48].copy_from_slice(&self.state);
        prefix_input[48..52].copy_from_slice(&pow_bits.to_le_bytes());
        let prefix_digest = blake2s_hash(&prefix_input);

        // Compute Blake2s(prefix_digest || nonce_le) — 40-byte input
        let mut input = [0u8; 40];
        input[..32].copy_from_slice(&prefix_digest);
        input[32..40].copy_from_slice(&nonce.to_le_bytes());
        let result = blake2s_hash(&input);

        // Check trailing zeros of first 16 bytes (u128 LE)
        let lo = u64::from_le_bytes(result[0..8].try_into().unwrap());
        let hi = u64::from_le_bytes(result[8..16].try_into().unwrap());
        let tz = if lo != 0 {
            lo.trailing_zeros()
        } else {
            64 + hi.trailing_zeros()
        };
        tz >= pow_bits
    }

    /// Expose raw 32-byte squeeze for Stark252 field element generation.
    pub fn squeeze_raw(&mut self) -> [u8; 32] { self.squeeze() }

    /// Squeeze 32 bytes from the channel.
    ///
    /// Input = state[32] || n_draws_le_u32[4] || 0x00[1]  (37 bytes total)
    /// Matches stwo's Blake2sChannel::draw_u32s with domain byte 0x00.
    fn squeeze(&mut self) -> [u8; 32] {
        let mut input = [0u8; 37];
        input[..32].copy_from_slice(&self.state);
        input[32..36].copy_from_slice(&self.n_draws.to_le_bytes());
        input[36] = 0x00; // domain separator (matches stwo)
        self.n_draws += 1;
        blake2s_hash(&input)
    }
}

/// Commitment hash for an arbitrary-length u32 slice.
///
/// Processes data in 8-word (32-byte) chunks using `Channel::mix_digest` in sequence,
/// then returns the final channel state as [u32; 8].
///
/// This is used to commit large data arrays (memory table, RC counts) into the
/// Fiat-Shamir transcript before FRI challenges are drawn.
pub fn hash_words(words: &[u32]) -> [u32; 8] {
    let mut ch = Channel::new();
    for chunk in words.chunks(8) {
        let mut buf = [0u32; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        ch.mix_digest(&buf);
    }
    let mut out = [0u32; 8];
    for (i, b) in ch.state.chunks_exact(4).enumerate() {
        out[i] = u32::from_le_bytes(b.try_into().unwrap());
    }
    out
}

/// Blake2s hash for internal Merkle nodes.
/// Uses standard Blake2s (no domain byte), matching stwo's MerkleHasherLifted::hash_children.
pub fn blake2s_hash_node(input: &[u8]) -> [u8; 32] {
    blake2s_hash(input)
}

/// Minimal Blake2s hash (single block, up to 64 bytes input).
/// Used for leaf hashing and Fiat-Shamir (domain = 0x00, no personalization change).
pub fn blake2s_hash(input: &[u8]) -> [u8; 32] {
    blake2s_hash_domain(input, 0x00)
}

/// Blake2s hash with domain separation via personalization byte.
/// `domain` is XORed into h[6] (first byte of Blake2s personalization field).
fn blake2s_hash_domain(input: &[u8], domain: u8) -> [u8; 32] {
    const IV: [u32; 8] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    ];

    const SIGMA: [[usize; 16]; 10] = [
        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15],
        [14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3],
        [11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4],
        [ 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8],
        [ 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13],
        [ 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9],
        [12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11],
        [13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10],
        [ 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5],
        [10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0],
    ];

    #[inline]
    fn g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(12);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(8);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(7);
    }

    // Pad input to 64 bytes
    let mut m_bytes = [0u8; 64];
    let len = input.len().min(64);
    m_bytes[..len].copy_from_slice(&input[..len]);

    let mut m = [0u32; 16];
    for i in 0..16 {
        m[i] = u32::from_le_bytes([
            m_bytes[i * 4],
            m_bytes[i * 4 + 1],
            m_bytes[i * 4 + 2],
            m_bytes[i * 4 + 3],
        ]);
    }

    let mut h = IV;
    h[0] ^= 0x01010020; // digest_length=32, fanout=1, depth=1
    h[6] ^= domain as u32; // domain separation (personalization byte)

    let mut v = [0u32; 16];
    v[..8].copy_from_slice(&h);
    v[8..16].copy_from_slice(&IV);
    v[12] ^= len as u32; // t0
    v[14] ^= 0xFFFF_FFFF; // final block

    for r in 0..10 {
        g(&mut v, 0, 4,  8, 12, m[SIGMA[r][ 0]], m[SIGMA[r][ 1]]);
        g(&mut v, 1, 5,  9, 13, m[SIGMA[r][ 2]], m[SIGMA[r][ 3]]);
        g(&mut v, 2, 6, 10, 14, m[SIGMA[r][ 4]], m[SIGMA[r][ 5]]);
        g(&mut v, 3, 7, 11, 15, m[SIGMA[r][ 6]], m[SIGMA[r][ 7]]);
        g(&mut v, 0, 5, 10, 15, m[SIGMA[r][ 8]], m[SIGMA[r][ 9]]);
        g(&mut v, 1, 6, 11, 12, m[SIGMA[r][10]], m[SIGMA[r][11]]);
        g(&mut v, 2, 7,  8, 13, m[SIGMA[r][12]], m[SIGMA[r][13]]);
        g(&mut v, 3, 4,  9, 14, m[SIGMA[r][14]], m[SIGMA[r][15]]);
    }

    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }

    let mut out = [0u8; 32];
    for (i, &w) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_deterministic() {
        let mut ch1 = Channel::new();
        let mut ch2 = Channel::new();

        let digest = [1u32, 2, 3, 4, 5, 6, 7, 8];
        ch1.mix_digest(&digest);
        ch2.mix_digest(&digest);

        let f1 = ch1.draw_felt();
        let f2 = ch2.draw_felt();
        assert_eq!(f1, f2, "Channel is not deterministic");
    }

    #[test]
    fn test_channel_different_inputs() {
        let mut ch1 = Channel::new();
        let mut ch2 = Channel::new();

        ch1.mix_digest(&[1, 2, 3, 4, 5, 6, 7, 8]);
        ch2.mix_digest(&[9, 10, 11, 12, 13, 14, 15, 16]);

        let f1 = ch1.draw_felt();
        let f2 = ch2.draw_felt();
        assert_ne!(f1, f2, "Different inputs should give different outputs");
    }

    #[test]
    fn test_channel_draw_multiple() {
        let mut ch = Channel::new();
        ch.mix_digest(&[42; 8]);
        let felts = ch.draw_felts(10);
        assert_eq!(felts.len(), 10);
        // Most should be different (allow some collisions from M31 range reduction)
        let mut unique = felts.clone();
        unique.sort_by_key(|f| f.to_u32_array());
        unique.dedup();
        assert!(unique.len() >= 7, "Too many duplicate felts: {} unique out of 10", unique.len());
    }

    /// Verify mix_felts batches all felts into one Blake2s call, matching stwo's protocol.
    #[test]
    fn test_mix_felts_matches_stwo_batched() {
        use blake2::{Blake2s256, Digest as _};
        use crate::field::QM31;

        let felts: Vec<QM31> = (1u32..=36)
            .collect::<Vec<_>>()
            .chunks(4)
            .map(|c| QM31::from_u32_array([c[0], c[1], c[2], c[3]]))
            .collect();
        // 9 QM31 values = 36 M31 words = 144 bytes after state → multi-block territory (>64B).

        // Compute expected value the same way stwo would:
        // new_state = Blake2s(state || f0_bytes || f1_bytes || ...)
        let initial_state = [0u8; 32];
        let mut hasher = Blake2s256::new();
        hasher.update(initial_state);
        for f in &felts {
            for &w in &f.to_u32_array() {
                hasher.update(w.to_le_bytes());
            }
        }
        let expected: [u8; 32] = hasher.finalize().into();

        let mut ch = Channel::new(); // state = [0; 32]
        ch.mix_felts(&felts);
        // Access state via state_words (state_words returns u32 words from self.state).
        let got: [u8; 32] = {
            let mut out = [0u8; 32];
            for (i, w) in ch.state_words().iter().enumerate() {
                out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            out
        };

        assert_eq!(got, expected, "mix_felts does not match stwo's batched Blake2s hashing");
    }
}
