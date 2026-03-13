//! Raw FFI bindings to CUDA runtime and kraken-stark kernels.

use std::ffi::c_void;

// CUDA runtime
unsafe extern "C" {
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(dev_ptr: *mut c_void) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    pub fn cudaMemset(dev_ptr: *mut c_void, value: i32, count: usize) -> i32;
    pub fn cudaDeviceSynchronize() -> i32;
    pub fn cudaGetLastError() -> i32;
}

// cudaMemcpyKind
pub const MEMCPY_H2D: i32 = 1;
pub const MEMCPY_D2H: i32 = 2;
pub const MEMCPY_D2D: i32 = 3;

// Field operation kernels
unsafe extern "C" {
    pub fn cuda_m31_add(a: *const u32, b: *const u32, out: *mut u32, n: u32);
    pub fn cuda_m31_mul(a: *const u32, b: *const u32, out: *mut u32, n: u32);
    pub fn cuda_device_sync();
}

// Circle NTT kernels
unsafe extern "C" {
    pub fn cuda_circle_ntt_evaluate(
        d_data: *mut u32,
        d_twiddles: *const u32,
        d_circle_twids: *const u32,
        h_layer_offsets: *const u32,
        h_layer_sizes: *const u32,
        n_line_layers: u32,
        n: u32,
    );

    pub fn cuda_circle_ntt_interpolate(
        d_data: *mut u32,
        d_itwiddles: *const u32,
        d_circle_itwids: *const u32,
        h_layer_offsets: *const u32,
        h_layer_sizes: *const u32,
        n_line_layers: u32,
        n: u32,
    );

    pub fn cuda_circle_ntt_evaluate_batch(
        d_columns: *mut *mut u32,
        d_twiddles: *const u32,
        d_circle_twids: *const u32,
        h_layer_offsets: *const u32,
        h_layer_sizes: *const u32,
        n_line_layers: u32,
        n: u32,
        n_cols: u32,
    );

    pub fn cuda_circle_ntt_interpolate_batch(
        d_columns: *mut *mut u32,
        d_itwiddles: *const u32,
        d_circle_itwids: *const u32,
        h_layer_offsets: *const u32,
        h_layer_sizes: *const u32,
        n_line_layers: u32,
        n: u32,
        n_cols: u32,
    );

    pub fn cuda_bit_reverse_m31(data: *mut u32, log_n: u32);

    pub fn cuda_eval_at_point(
        d_coeffs: *const u32,
        d_folding_factors: *const u32,
        h_result: *mut u32,
        n: u32,
        d_scratch1: *mut u32,
        d_scratch2: *mut u32,
    );
}

// FRI fold kernels (SoA layout)
unsafe extern "C" {
    pub fn cuda_fold_line_soa(
        in0: *const u32, in1: *const u32, in2: *const u32, in3: *const u32,
        twiddles: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha: *const u32, // [4] on host
        half_n: u32,
    );

    pub fn cuda_fold_circle_into_line_soa(
        dst0: *mut u32, dst1: *mut u32, dst2: *mut u32, dst3: *mut u32,
        src0: *const u32, src1: *const u32, src2: *const u32, src3: *const u32,
        twiddles: *const u32,
        alpha: *const u32,     // [4] on host
        alpha_sq: *const u32,  // [4] on host
        half_n: u32,
    );
}

// Constraint evaluation kernels
unsafe extern "C" {
    pub fn cuda_fibonacci_quotient(
        trace: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha: *const u32, // [4] on host
        n: u32,
    );

    pub fn cuda_zero_pad(
        src: *const u32,
        dst: *mut u32,
        src_n: u32,
        dst_n: u32,
    );
}

// Blake2s Merkle tree kernels
unsafe extern "C" {
    pub fn cuda_merkle_hash_leaves(
        columns: *const *const u32,
        hashes: *mut u32,
        n_cols: u32,
        n_leaves: u32,
    );

    pub fn cuda_merkle_hash_nodes(
        children: *const u32,
        parents: *mut u32,
        n_parents: u32,
    );
}
