//! GrindOps: proof-of-work nonce search on GPU.
//!
//! Precomputes the Blake2s prefix digest on CPU, then launches millions of
//! GPU threads to search in parallel. The 5090's 21,504 CUDA cores make
//! this near-instant even at high pow_bits.
//!
//! For IS_M31_OUTPUT=true channels, falls back to parallel CPU grinding since
//! the M31 reduction on hash output changes bit patterns and would require a
//! separate GPU kernel. This case is rare in practice.

use std::ffi::c_void;

use blake2::{Blake2s256, Digest};
use stwo::core::channel::{Blake2sChannelGeneric, Channel};
use stwo::core::proof_of_work::GrindOps;

use super::CudaBackend;

/// Number of nonces to try per GPU batch. 2^24 = 16M threads per launch.
/// On an RTX 5090 this completes in ~1ms per batch, so even pow_bits=30
/// (expected ~1B nonces) finishes in ~60 batches = ~60ms.
const BATCH_SIZE: u32 = 1 << 24;

/// CPU parallel grind fallback (used for Poseidon252 and IS_M31_OUTPUT=true).
fn grind_cpu_parallel<C: Channel + Clone + Send>(channel: &C, pow_bits: u32) -> u64 {
    let n_threads = std::thread::available_parallelism()
        .map_or(8, |n| n.get());

    let found = std::sync::atomic::AtomicU64::new(u64::MAX);

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let found = &found;
            let channel = channel.clone();
            s.spawn(move || {
                let mut nonce = t as u64;
                let stride = n_threads as u64;
                loop {
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

/// GPU grind for standard Blake2s (IS_M31_OUTPUT=false).
fn grind_gpu_blake2s(channel_digest: &[u8; 32], pow_bits: u32) -> u64 {
    // Step 1: Precompute prefixed_digest on CPU.
    // prefixed_digest = Blake2s(POW_PREFIX_LE || [0u8; 12] || channel_digest || pow_bits_LE)
    let pow_prefix: u32 = 0x12345678; // Blake2sChannelGeneric::POW_PREFIX
    let mut hasher = Blake2s256::new();
    blake2::Digest::update(&mut hasher, &pow_prefix.to_le_bytes());
    blake2::Digest::update(&mut hasher, &[0u8; 12]);
    blake2::Digest::update(&mut hasher, &channel_digest[..]);
    blake2::Digest::update(&mut hasher, &pow_bits.to_le_bytes());
    let prefixed_hash: [u8; 32] = hasher.finalize().into();

    // Convert to u32 words (little-endian) for the GPU kernel
    let mut prefixed_words = [0u32; 8];
    for i in 0..8 {
        prefixed_words[i] = u32::from_le_bytes(
            prefixed_hash[i * 4..(i + 1) * 4].try_into().unwrap(),
        );
    }

    // Step 2: Upload prefixed_digest and result buffer to GPU
    let digest_bytes = std::mem::size_of::<[u32; 8]>();
    let mut d_digest: *mut c_void = std::ptr::null_mut();
    let mut d_result: *mut c_void = std::ptr::null_mut();

    unsafe {
        let err = vortexstark::cuda::ffi::cudaMalloc(&mut d_digest, digest_bytes);
        assert!(err == 0, "cudaMalloc for grind digest failed: {err}");

        let err = vortexstark::cuda::ffi::cudaMalloc(&mut d_result, std::mem::size_of::<u64>());
        assert!(err == 0, "cudaMalloc for grind result failed: {err}");

        // Upload digest
        let err = vortexstark::cuda::ffi::cudaMemcpy(
            d_digest,
            prefixed_words.as_ptr() as *const c_void,
            digest_bytes,
            vortexstark::cuda::ffi::MEMCPY_H2D,
        );
        assert!(err == 0, "cudaMemcpy H2D for grind digest failed: {err}");

        // Initialize result to u64::MAX
        let init_val: u64 = u64::MAX;
        let err = vortexstark::cuda::ffi::cudaMemcpy(
            d_result,
            &init_val as *const u64 as *const c_void,
            std::mem::size_of::<u64>(),
            vortexstark::cuda::ffi::MEMCPY_H2D,
        );
        assert!(err == 0, "cudaMemcpy H2D for grind result failed: {err}");
    }

    // Step 3: Launch batches until we find a nonce
    let mut batch_offset: u64 = 0;
    let found_nonce: u64;

    loop {
        unsafe {
            vortexstark::cuda::ffi::cuda_grind_pow(
                d_digest as *const u32,
                d_result as *mut u64,
                pow_bits,
                batch_offset,
                BATCH_SIZE,
            );
            vortexstark::cuda::ffi::cudaDeviceSynchronize();
        }

        // Read result
        let mut result: u64 = u64::MAX;
        unsafe {
            let err = vortexstark::cuda::ffi::cudaMemcpy(
                &mut result as *mut u64 as *mut c_void,
                d_result as *const c_void,
                std::mem::size_of::<u64>(),
                vortexstark::cuda::ffi::MEMCPY_D2H,
            );
            assert!(err == 0, "cudaMemcpy D2H for grind result failed: {err}");
        }

        if result != u64::MAX {
            found_nonce = result;
            break;
        }

        batch_offset += BATCH_SIZE as u64;

        // Safety valve: if we've searched 2^40 nonces something is very wrong
        assert!(
            batch_offset < (1u64 << 40),
            "GPU grind failed: searched {} nonces without finding pow_bits={pow_bits}",
            batch_offset
        );
    }

    // Cleanup
    unsafe {
        vortexstark::cuda::ffi::cudaFree(d_digest);
        vortexstark::cuda::ffi::cudaFree(d_result);
    }

    found_nonce
}

impl GrindOps<Blake2sChannelGeneric<false>> for CudaBackend {
    fn grind(channel: &Blake2sChannelGeneric<false>, pow_bits: u32) -> u64 {
        if pow_bits == 0 {
            return 0;
        }

        let nonce = grind_gpu_blake2s(&channel.digest().0, pow_bits);

        // Verify on CPU (defense in depth -- only in debug builds)
        debug_assert!(
            channel.verify_pow_nonce(pow_bits, nonce),
            "GPU grind returned invalid nonce {nonce} for pow_bits={pow_bits}"
        );

        nonce
    }
}

impl GrindOps<Blake2sChannelGeneric<true>> for CudaBackend {
    fn grind(channel: &Blake2sChannelGeneric<true>, pow_bits: u32) -> u64 {
        // M31-output channels require reducing hash output mod P before checking
        // trailing zeros. Fall back to CPU parallel grinding for this rare case.
        // TODO: GPU kernel variant that applies M31 reduction to hash output.
        if pow_bits == 0 {
            return 0;
        }
        grind_cpu_parallel(channel, pow_bits)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl GrindOps<stwo::core::channel::Poseidon252Channel> for CudaBackend {
    fn grind(channel: &stwo::core::channel::Poseidon252Channel, pow_bits: u32) -> u64 {
        // Poseidon252 grinding stays on CPU -- no GPU kernel for Poseidon hash.
        if pow_bits == 0 {
            return 0;
        }
        grind_cpu_parallel(channel, pow_bits)
    }
}
