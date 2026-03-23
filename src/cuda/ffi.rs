//! Raw FFI bindings to CUDA runtime and VortexSTARK kernels.

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
    pub fn cudaMemPoolTrimTo(pool: *mut c_void, min_bytes_to_keep: usize) -> i32;
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

// CUDA memory info
unsafe extern "C" {
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
}

/// Query current VRAM state. Returns (free_bytes, total_bytes).
pub fn vram_query() -> (usize, usize) {
    let mut free: usize = 0;
    let mut total: usize = 0;
    let err = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    assert!(err == 0, "cudaMemGetInfo failed: {err}");
    (free, total)
}

/// VRAM safety check via nvidia-smi. Must be called before CUDA context init
/// (before detect_wsl2_and_configure) so readings reflect external processes only.
pub fn vram_preflight_check() {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"])
        .output();

    if let Ok(out) = output {
        let s = String::from_utf8_lossy(&out.stdout);
        let parts: Vec<&str> = s.trim().split(',').collect();
        if parts.len() == 3 {
            let used_mb: usize = parts[0].trim().parse().unwrap_or(0);
            let free_mb: usize = parts[1].trim().parse().unwrap_or(0);
            let total_mb: usize = parts[2].trim().parse().unwrap_or(0);
            eprintln!("[VRAM] {used_mb} MB used / {total_mb} MB total ({free_mb} MB free)");
            if used_mb > 512 {
                eprintln!("[VRAM] WARNING: {used_mb} MB already in use by another process.");
                eprintln!("[VRAM] Another GPU workload may be running.");
                eprintln!("[VRAM] VortexSTARK needs up to 28 GB for large proofs (log_n>=27).");
                eprintln!("[VRAM] Proceeding, but large proofs may OOM. Stop other GPU processes first.");
            }
            return;
        }
    }
    // Fallback to cudaMemGetInfo if nvidia-smi unavailable (note: requires active CUDA context)
    let (free, total) = vram_query();
    let used = total - free;
    eprintln!("[VRAM] {} MB used / {} MB total ({} MB free)",
        used / (1024*1024), total / (1024*1024), free / (1024*1024));
}

/// Flush the CUDA memory pool back to the OS. Call after each job completes
/// to release VRAM that's no longer needed. Keeps `keep_bytes` in the pool
/// for fast reuse on the next job (0 = release everything).
pub fn vram_release(keep_bytes: usize) {
    if crate::device::buffer::use_sync() {
        // Sync mode: no pool to trim, just sync the device
        unsafe { cudaDeviceSynchronize(); }
        return;
    }
    unsafe {
        cudaDeviceSynchronize();
        let mut pool: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = cudaDeviceGetDefaultMemPool(&mut pool, 0);
        if err != 0 { return; }
        cudaMemPoolTrimTo(pool, keep_bytes);
    }
}

/// Initialize CUDA memory pool for async allocation. Call once at startup.
///
/// Runs a VRAM preflight check, then configures the default memory pool.
/// The pool release threshold is set to 2 GB — freed buffers stay cached
/// up to that limit for fast reuse, but anything beyond 2 GB is returned
/// to the OS between jobs so other processes can share the GPU.
pub fn init_memory_pool() {
    // Check VRAM before CUDA context is initialized (nvidia-smi sees only external processes)
    vram_preflight_check();

    // Detect WSL2 and switch to sync malloc if needed
    crate::device::buffer::detect_wsl2_and_configure();

    // Sanity check: verify GPU is accessible with a tiny allocation
    unsafe {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = cudaMalloc(&mut ptr, 64);
        if err != 0 {
            panic!("[CUDA] GPU sanity check failed: cudaMalloc(64) returned error {err}. \
                    Is the CUDA driver working? Try: nvidia-smi");
        }
        cudaFree(ptr);
        let err = cudaDeviceSynchronize();
        if err != 0 {
            panic!("[CUDA] GPU sync failed after sanity check: error {err}");
        }
        eprintln!("[CUDA] GPU sanity check passed");
    }

    // Skip pool configuration on WSL2 (not using async alloc)
    if crate::device::buffer::use_sync() {
        return;
    }

    unsafe {
        let mut pool: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = cudaDeviceGetDefaultMemPool(&mut pool, 0);
        assert!(err == 0, "cudaDeviceGetDefaultMemPool failed: {err}");
        // Keep 2 GB cached in the pool for fast reuse; release the rest.
        let threshold: u64 = 2 * 1024 * 1024 * 1024;
        let err = cudaMemPoolSetAttribute(
            pool,
            MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &threshold as *const u64 as *const std::ffi::c_void,
        );
        assert!(err == 0, "cudaMemPoolSetAttribute failed: {err}");
    }
}

/// Initialize with greedy pool (never releases memory). Use only for
/// back-to-back benchmarks where you know nothing else needs the GPU.
pub fn init_memory_pool_greedy() {
    // Check VRAM before CUDA context is initialized (nvidia-smi sees only external processes)
    vram_preflight_check();
    crate::device::buffer::detect_wsl2_and_configure();

    if crate::device::buffer::use_sync() {
        return;
    }

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

// Stwo-compatible Circle NTT kernels (flat twiddle format)
unsafe extern "C" {
    /// Forward NTT using stwo's flat twiddle buffer.
    pub fn cuda_stwo_ntt_evaluate(d_data: *mut u32, d_twiddles: *const u32, n: u32);
    /// Inverse NTT using stwo's flat twiddle buffer (includes 1/n scaling).
    pub fn cuda_stwo_ntt_interpolate(d_data: *mut u32, d_itwiddles: *const u32, n: u32);
}

// Circle NTT kernels (original VortexSTARK format)
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

    pub fn cuda_rpo_upload_constants(host_mds: *const u32, host_rc: *const u32);

    pub fn cuda_rpo_trace(
        block_inputs: *const u32,
        trace_cols: *const *mut u32,
        n_blocks: u32,
    );

    pub fn cuda_p2f_upload_consts(host_rc: *const u32);

    pub fn cuda_p2f_trace(
        block_inputs: *const u32,
        trace_cols: *const *mut u32,
        n_blocks: u32,
    );
}

// Blake2s PoW grinding kernel
unsafe extern "C" {
    /// Launch GPU grind kernel. Each of n_threads threads tries one nonce
    /// starting from batch_offset. Result is atomicMin'd into result[0]
    /// (must be initialized to u64::MAX by caller).
    pub fn cuda_grind_pow(
        prefixed_digest: *const u32, // device ptr, [8] words
        result: *mut u64,            // device ptr, [1] word
        pow_bits: u32,
        batch_offset: u64,
        n_threads: u32,
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

// Bytecode constraint evaluation kernel
unsafe extern "C" {
    /// Execute a bytecode constraint evaluation program on the GPU.
    ///
    /// Each GPU thread interprets the same bytecode program for one row,
    /// evaluating all constraints and accumulating the results into the
    /// output columns (SoA QM31 format).
    ///
    /// - `bytecode`: device pointer to encoded u32 instruction stream
    /// - `n_words`: number of u32 words in the bytecode
    /// - `trace_cols`: device pointer to array of device pointers (one per column)
    /// - `trace_col_sizes`: device pointer to array of column sizes (unused, reserved)
    /// - `n_trace_cols`: number of columns in trace_cols
    /// - `n_rows`: number of rows in the evaluation domain
    /// - `trace_n_rows`: number of rows in the trace domain (power of 2)
    /// - `random_coeff_powers`: device pointer to [n_constraints * 4] QM31 values (reversed)
    /// - `denom_inv`: device pointer to [1 << log_expand] M31 denominator inverses
    /// - `log_expand`: eval_log_size - trace_log_size
    /// - `accum0..3`: device pointers to output accumulator columns (SoA QM31)
    pub fn cuda_bytecode_constraint_eval(
        bytecode: *const u32,
        n_words: u32,
        trace_cols: *const *const u32,
        trace_col_sizes: *const u32,
        n_trace_cols: u32,
        n_rows: u32,
        trace_n_rows: u32,
        random_coeff_powers: *const u32,
        denom_inv: *const u32,
        log_expand: u32,
        accum0: *mut u32,
        accum1: *mut u32,
        accum2: *mut u32,
        accum3: *mut u32,
        n_registers: u32,
    );

    /// Warp-cooperative bytecode constraint eval kernel.
    /// Uses one warp (32 threads) per row, distributing the register file
    /// across warp lanes. Each thread holds ceil(n_registers/32) registers.
    /// Same interface as cuda_bytecode_constraint_eval.
    pub fn cuda_warp_bytecode_constraint_eval(
        bytecode: *const u32,
        n_words: u32,
        trace_cols: *const *const u32,
        trace_col_sizes: *const u32,
        n_trace_cols: u32,
        n_rows: u32,
        trace_n_rows: u32,
        random_coeff_powers: *const u32,
        denom_inv: *const u32,
        log_expand: u32,
        accum0: *mut u32,
        accum1: *mut u32,
        accum2: *mut u32,
        accum3: *mut u32,
        n_registers: u32,
    );

    // ── GPU lifted Merkle leaf hashing ──────────────────────────────────

    /// Build Merkle leaves on GPU using the lifted hashing algorithm.
    /// Columns of different sizes are handled via the lifted row index formula.
    ///
    /// - `col_ptrs`: device pointer to array of column device pointers
    /// - `schedule`: device pointer to array of LeafHashChunk structs
    /// - `n_chunks`: number of chunks in the schedule
    /// - `lifting_log_size`: log2 of output leaf count
    /// - `output_hashes`: device pointer to [n_leaves * 8] u32s (Blake2s hashes)
    /// - `n_leaves`: 2^lifting_log_size
    pub fn cuda_build_leaves_lifted(
        col_ptrs: *const *const u32,
        schedule: *const u8,  // LeafHashChunk array (passed as raw bytes)
        n_chunks: u32,
        lifting_log_size: u32,
        output_hashes: *mut u32,
        n_leaves: u32,
    );

    // ── Barycentric evaluation ──────────────────────────────────────────

    /// Compute result = sum_i(evals[i] * weights[i]) using a parallel reduction.
    ///
    /// - `evals`: device pointer to n M31 values (1 u32 each)
    /// - `weights`: device pointer to n QM31 values in AoS layout (4 u32 per element)
    /// - `n`: number of elements
    /// - `out`: device pointer to output buffer of `n_blocks * 4` u32s (QM31 partial sums)
    /// - `n_blocks`: number of parallel reduction blocks (caller allocates out and
    ///   must CPU-reduce the partial sums after the call)
    pub fn cuda_barycentric_eval(
        evals: *const u32,
        weights: *const u32,
        n: u32,
        out: *mut u32,
        n_blocks: u32,
    );

    // ── FRI Quotient kernels ────────────────────────────────────────────

    /// Accumulate partial numerators for a single sample batch.
    ///
    /// For each row: result = sum_i (c_i * col[col_idx_i][row] - b_i)
    /// where (b_i, c_i) are QM31 line coefficients.
    ///
    /// - `col_ptrs`: device array of pointers to M31 column data
    /// - `col_indices`: device array of column indices to use [n_batch_cols]
    /// - `b_coeffs`: device array of QM31 b-coefficients [n_batch_cols * 4] M31 limbs
    /// - `c_coeffs`: device array of QM31 c-coefficients [n_batch_cols * 4] M31 limbs
    /// - `n_batch_cols`: number of columns in this batch
    /// - `n_rows`: number of rows
    /// - `out0..3`: output SoA QM31 accumulator [n_rows] each
    pub fn cuda_accumulate_numerators(
        col_ptrs: *const *const u32,
        col_indices: *const u32,
        b_coeffs: *const u32,
        c_coeffs: *const u32,
        n_batch_cols: u32,
        n_rows: u32,
        out0: *mut u32,
        out1: *mut u32,
        out2: *mut u32,
        out3: *mut u32,
    );

    /// Compute FRI quotients and combine across sample points.
    ///
    /// For each row:
    ///   domain_point = (domain_xs[row], domain_ys[row])
    ///   quotient[row] = sum_j (numer_j[lifted_idx] - a_acc_j * y) * den_inv_j
    ///
    /// - `sample_points_x/y`: device arrays of QM31 sample point coords [n_accs * 4]
    /// - `first_linear_acc`: device array of QM31 a-coefficients [n_accs * 4]
    /// - `numer_ptrs0..3`: device arrays of pointers to partial numerator SoA columns
    /// - `acc_log_sizes`: device array of log2(size) per accumulation [n_accs]
    /// - `n_accs`: number of accumulations
    /// - `domain_xs/ys`: device arrays of M31 domain point coords [n_rows]
    /// - `lifting_log_size`: log2 of the lifting domain size
    /// - `n_rows`: number of output rows
    /// - `out0..3`: output SoA QM31 [n_rows] each
    pub fn cuda_compute_quotients_combine(
        sample_points_x: *const u32,
        sample_points_y: *const u32,
        first_linear_acc: *const u32,
        numer_ptrs0: *const *const u32,
        numer_ptrs1: *const *const u32,
        numer_ptrs2: *const *const u32,
        numer_ptrs3: *const *const u32,
        acc_log_sizes: *const u32,
        n_accs: u32,
        domain_xs: *const u32,
        domain_ys: *const u32,
        lifting_log_size: u32,
        n_rows: u32,
        out0: *mut u32,
        out1: *mut u32,
        out2: *mut u32,
        out3: *mut u32,
    );
}
