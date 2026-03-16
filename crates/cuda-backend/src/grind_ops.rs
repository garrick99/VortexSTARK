//! GrindOps: proof-of-work nonce search.
//!
//! Uses parallel CPU threads to search the nonce space.
//! Each thread tests a disjoint range. With 24 cores this is 24x faster
//! than the serial CPU backend.
//!
//! TODO: Full GPU implementation — launch thousands of CUDA threads each
//! testing a nonce range, with Blake2s computed entirely on device.
//! The 5090's 21,504 cores would make this near-instant even at high pow_bits.

use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::proof_of_work::GrindOps;

use super::CudaBackend;

impl GrindOps<Blake2sChannel> for CudaBackend {
    fn grind(channel: &Blake2sChannel, pow_bits: u32) -> u64 {
        let n_threads = std::thread::available_parallelism()
            .map_or(8, |n| n.get());

        // Each thread searches a disjoint nonce range
        let found = std::sync::atomic::AtomicU64::new(u64::MAX);

        std::thread::scope(|s| {
            for t in 0..n_threads {
                let found = &found;
                let channel = channel.clone();
                s.spawn(move || {
                    let mut nonce = t as u64;
                    let stride = n_threads as u64;
                    loop {
                        // Early exit if another thread found it
                        if found.load(std::sync::atomic::Ordering::Relaxed) != u64::MAX {
                            return;
                        }
                        if channel.verify_pow_nonce(pow_bits, nonce) {
                            found.store(nonce, std::sync::atomic::Ordering::Relaxed);
                            return;
                        }
                        nonce += stride;
                    }
                });
            }
        });

        found.load(std::sync::atomic::Ordering::Relaxed)
    }
}
