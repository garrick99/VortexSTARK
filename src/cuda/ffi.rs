//! Raw FFI bindings to CUDA runtime and kraken-stark kernels.

use std::ffi::c_void;

// CUDA runtime
unsafe extern "C" {
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(dev_ptr: *mut c_void) -> i32;
    pub fn cudaMallocAsync(dev_ptr: *mut *mut c_void, size: usize, stream: *mut c_void) -> i32;
    pub fn cudaFreeAsync(dev_ptr: *mut c_void, stream: *mut c_void) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: *mut c_void) -> i32;
    pub fn cudaMemset(dev_ptr: *mut c_void, value: i32, count: usize) -> i32;
    pub fn cudaDeviceSynchronize() -> i32;
    pub fn cudaGetLastError() -> i32;
    pub fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFreeHost(ptr: *mut c_void) -> i32;
    pub fn cudaDeviceGetDefaultMemPool(pool: *mut *mut c_void, device: i32) -> i32;
    pub fn cudaMemPoolSetAttribute(pool: *mut c_void, attr: i32, value: *const c_void) -> i32;
}

// CUDA streams
unsafe extern "C" {
    pub fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    pub fn cudaStreamDestroy(stream: *mut c_void) -> i32;
}

/// RAII wrapper for a CUDA stream.
pub struct CudaStream {
    pub ptr: *mut c_void,
}

impl CudaStream {
    pub fn new() -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { cudaStreamCreate(&mut ptr) };
        assert!(err == 0, "cudaStreamCreate failed: {err}");
        Self { ptr }
    }

    pub fn sync(&self) {
        let err = unsafe { cudaStreamSynchronize(self.ptr) };
        assert!(err == 0, "cudaStreamSynchronize failed: {err}");
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaStreamDestroy(self.ptr) };
        }
    }
}

// cudaMemcpyKind
pub const MEMCPY_H2D: i32 = 1;
pub const MEMCPY_D2H: i32 = 2;
pub const MEMCPY_D2D: i32 = 3;

// cudaMemPoolAttr
pub const MEMPOOL_ATTR_RELEASE_THRESHOLD: i32 = 4;

/// Initialize CUDA memory pool for async allocation. Call once at startup.
/// Sets the default pool's release threshold to MAX so freed memory stays in the pool.
pub fn init_memory_pool() {
    unsafe {
        let mut pool: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = cudaDeviceGetDefaultMemPool(&mut pool, 0);
        assert!(err == 0, "cudaDeviceGetDefaultMemPool failed: {err}");
        let threshold: u64 = u64::MAX;
        let err = cudaMemPoolSetAttribute(
            pool,
            MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &threshold as *const u64 as *const std::ffi::c_void,
        );
        assert!(err == 0, "cudaMemPoolSetAttribute failed: {err}");
    }
}

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

    pub fn cuda_circle_ntt_layer(
        d_data: *mut u32,
        d_twiddles: *const u32,
        layer_idx: u32,
        n: u32,
        forward: i32,
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

// Twiddle computation kernels
unsafe extern "C" {
    pub fn cuda_compute_fold_twiddle_sources(
        initial_x: u32, initial_y: u32,
        step_x: u32, step_y: u32,
        output: *mut u32,
        n: u32, log_n: u32,
        extract_y: i32,
    );

    pub fn cuda_batch_inverse_m31(input: *const u32, output: *mut u32, n: u32);

    pub fn cuda_compute_fold_twiddle_sources_stream(
        initial_x: u32, initial_y: u32,
        step_x: u32, step_y: u32,
        output: *mut u32,
        n: u32, log_n: u32,
        extract_y: i32,
        stream: *mut c_void,
    );

    pub fn cuda_batch_inverse_m31_stream(input: *const u32, output: *mut u32, n: u32, stream: *mut c_void);

    pub fn cuda_compute_coset_points(
        initial_x: u32, initial_y: u32,
        step_x: u32, step_y: u32,
        output_x: *mut u32, output_y: *mut u32,
        n: u32,
    );

    pub fn cuda_squash_x(input: *const u32, output: *mut u32, out_n: u32);

    pub fn cuda_extract_and_squash(
        input: *const u32,
        twiddle_out: *mut u32,
        squash_out: *mut u32,
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

    pub fn cuda_fibonacci_quotient_chunk(
        trace: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha: *const u32, // [4] on host
        offset: u32,
        chunk_n: u32,
        global_n: u32,
    );

    pub fn cuda_fibonacci_quotient_chunk_stream(
        trace: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha: *const u32,
        offset: u32, chunk_n: u32, global_n: u32,
        stream: *mut std::ffi::c_void,
    );

    // LogUp interaction kernels
    /// Chunked LogUp: process one (addr, value) pair, accumulate into running sum.
    /// Call 4 times (once per memory access) with is_first=1 for the first call.
    pub fn cuda_logup_accumulate_pair(
        col_addr: *const u32, col_val: *const u32,
        acc0: *mut u32, acc1: *mut u32, acc2: *mut u32, acc3: *mut u32,
        z: *const u32, alpha: *const u32,
        n: u32, is_first: u32,
    );

    pub fn cuda_logup_memory_fused(
        col_pc: *const u32, col_inst_lo: *const u32,
        col_dst_addr: *const u32, col_dst: *const u32,
        col_op0_addr: *const u32, col_op0: *const u32,
        col_op1_addr: *const u32, col_op1: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        z: *const u32, alpha: *const u32,
        n: u32,
    );

    pub fn cuda_logup_memory_denoms(
        col_pc: *const u32, col_inst_lo: *const u32,
        col_dst_addr: *const u32, col_dst: *const u32,
        col_op0_addr: *const u32, col_op0: *const u32,
        col_op1_addr: *const u32, col_op1: *const u32,
        denom0: *mut u32, denom1: *mut u32, denom2: *mut u32, denom3: *mut u32,
        z: *const u32, alpha: *const u32,
        n: u32,
    );

    pub fn cuda_logup_memory_combine(
        col_pc: *const u32, col_inst_lo: *const u32,
        col_dst_addr: *const u32, col_dst: *const u32,
        col_op0_addr: *const u32, col_op0: *const u32,
        col_op1_addr: *const u32, col_op1: *const u32,
        inv0: *const u32, inv1: *const u32, inv2: *const u32, inv3: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        z: *const u32, alpha: *const u32,
        n: u32,
    );

    pub fn cuda_qm31_inverse(
        in0: *const u32, in1: *const u32, in2: *const u32, in3: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        n: u32,
    );

    pub fn cuda_qm31_block_scan(
        c0: *mut u32, c1: *mut u32, c2: *mut u32, c3: *mut u32,
        block_sums0: *mut u32, block_sums1: *mut u32,
        block_sums2: *mut u32, block_sums3: *mut u32,
        n: u32, block_size: u32,
    );

    pub fn cuda_qm31_add_block_prefix(
        c0: *mut u32, c1: *mut u32, c2: *mut u32, c3: *mut u32,
        prefix0: *const u32, prefix1: *const u32,
        prefix2: *const u32, prefix3: *const u32,
        n: u32, block_size: u32,
    );

    pub fn cuda_cairo_quotient(
        trace_cols: *const *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha_coeffs: *const u32,
        n: u32,
    );

    pub fn cuda_cairo_quotient_chunk(
        trace_cols: *const *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha_coeffs: *const u32,
        offset: u32, chunk_n: u32, global_n: u32,
    );

    // Fp252 test
    pub fn cuda_fp252_test(results: *mut u64);

    // Pedersen GPU
    pub fn cuda_pedersen_test_double(
        px: *const u64, py: *const u64,
        out_x: *mut u64, out_y: *mut u64, out_z: *mut u64,
    );

    pub fn cuda_pedersen_upload_points(px: *const u64, py: *const u64);

    // Upload precomputed windowed tables + P0 in Montgomery form
    pub fn cuda_pedersen_upload_tables(
        table_x: *const u64, table_y: *const u64, table_z: *const u64, // [4][16] each
        p0_x: *const u64, p0_y: *const u64, p0_z: *const u64, // single point
    );
    pub fn cuda_pedersen_hash_batch(
        inputs_a: *const u64, inputs_b: *const u64,
        out_x: *mut u64, out_zz: *mut u64, n: u32,
    );

    pub fn cuda_pedersen_hash_batch_stream(
        inputs_a: *const u64, inputs_b: *const u64,
        out_x: *mut u64, out_zz: *mut u64, n: u32,
        stream: *mut c_void,
    );

    /// Decompose pre-computed Fp252 values into 27 M31 trace columns (no hashing).
    pub fn cuda_pedersen_decompose(
        vals_a: *const u64, vals_b: *const u64, vals_out: *const u64,
        trace_cols: *mut *mut u32,
        n: u32,
        stream: *mut c_void,
    );

    /// EC trace generation: outputs intermediate Jacobian points per step.
    pub fn cuda_pedersen_ec_trace(
        inputs_a: *const u64, inputs_b: *const u64,
        ec_trace: *mut u64, ec_ops: *mut u32,
        n: u32, stream: *mut c_void,
    );

    /// Decompose raw EC trace (u64 Jacobian) into M31 SoA columns.
    pub fn cuda_ec_trace_decompose(
        ec_trace: *const u64, ec_ops: *const u32,
        trace_cols: *mut *mut u32,
        n_rows: u32, stream: *mut c_void,
    );

    /// Fused Pedersen hash + trace column generation.
    /// Hashes (a, b) pairs and decomposes results into 27 M31 trace columns on GPU.
    /// trace_cols: device pointer to array of 27 device pointers (one per column).
    pub fn cuda_pedersen_trace(
        inputs_a: *const u64, inputs_b: *const u64,
        trace_cols: *mut *mut u32,
        n: u32,
        stream: *mut c_void,
    );

    pub fn cuda_poseidon_upload_round_consts(host_rc: *const u32);

    pub fn cuda_poseidon_trace(
        block_inputs: *const u32,
        trace_cols: *const *mut u32,
        n_blocks: u32,
    );

    pub fn cuda_poseidon_quotient(
        trace_cols: *const *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        round_consts: *const u32,
        alpha_coeffs: *const u32,
        n: u32,
    );

    pub fn cuda_poseidon_quotient_chunk(
        trace_cols: *const *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        round_consts: *const u32,
        alpha_coeffs: *const u32,
        offset: u32, chunk_n: u32, global_n: u32,
    );

    pub fn cuda_interleave_u32(
        even: *const u32,
        odd: *const u32,
        output: *mut u32,
        half_n: u32,
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

    pub fn cuda_merkle_reduce_to_root(
        nodes: *const u32,
        root_out: *mut u32,
        n_nodes: u32,
    );

    pub fn cuda_merkle_commit_small_soa4(
        col0: *const u32, col1: *const u32,
        col2: *const u32, col3: *const u32,
        root_out: *mut u32,
        n_leaves: u32,
    );

    pub fn cuda_merkle_hash_leaves_merge_soa4(
        col0: *const u32, col1: *const u32,
        col2: *const u32, col3: *const u32,
        parents: *mut u32,
        n_leaves: u32,
    );

    pub fn cuda_merkle_tiled_soa4(
        col0: *const u32, col1: *const u32,
        col2: *const u32, col3: *const u32,
        subtree_roots: *mut u32,
        n_leaves: u32,
    );

    pub fn cuda_merkle_tiled_soa4_stream(
        col0: *const u32, col1: *const u32,
        col2: *const u32, col3: *const u32,
        subtree_roots: *mut u32,
        n_leaves: u32,
        stream: *mut std::ffi::c_void,
    );

    pub fn cuda_merkle_tiled_generic(
        columns: *const *const u32,
        subtree_roots: *mut u32,
        n_cols: u32,
        n_leaves: u32,
    );
}
