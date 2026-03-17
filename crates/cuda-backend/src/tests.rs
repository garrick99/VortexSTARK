//! Tests for CudaBackend against stwo's own test patterns.
//! Validates GPU results match CPU results exactly.

#[cfg(test)]
mod tests {
    use stwo::prover::backend::{Column, ColumnOps, CpuBackend};
    use stwo::core::fields::m31::{BaseField, M31};
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::fields::FieldExpOps;
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};
    use stwo::prover::poly::twiddles::TwiddleTree;
    use num_traits::{One, Zero};
    use std::time::Instant;

    use crate::CudaBackend;
    use crate::column::CudaColumn;

    fn init_gpu() {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            vortexstark::cuda::ffi::init_memory_pool();
        });
    }

    // ---- Column tests ----

    #[test]
    fn test_column_roundtrip_base() {
        init_gpu();
        let values: Vec<BaseField> = (0..1024).map(|i| M31::from(i as u32)).collect();
        let col: CudaColumn<BaseField> = values.iter().copied().collect();
        let back = col.to_cpu();
        assert_eq!(values, back);
    }

    #[test]
    fn test_column_roundtrip_secure() {
        init_gpu();
        let values: Vec<SecureField> = (0..256).map(|i| {
            SecureField::from_m31_array([
                M31::from(i as u32), M31::from(i as u32 + 100),
                M31::from(i as u32 + 200), M31::from(i as u32 + 300),
            ])
        }).collect();
        let col: CudaColumn<SecureField> = values.iter().copied().collect();
        let back = col.to_cpu();
        assert_eq!(values, back);
    }

    #[test]
    fn test_column_at_set() {
        init_gpu();
        let mut col = CudaColumn::<BaseField>::zeros(64);
        col.set(42, M31::from(12345));
        assert_eq!(col.at(42), M31::from(12345));
        assert_eq!(col.at(0), M31::zero());
    }

    #[test]
    fn test_column_split_at_mid_base() {
        init_gpu();
        let values: Vec<BaseField> = (0..8).map(|i| M31::from(i as u32)).collect();
        let col: CudaColumn<BaseField> = values.iter().copied().collect();
        let (left, right) = col.split_at_mid();
        let left_cpu = left.to_cpu();
        let right_cpu = right.to_cpu();
        assert_eq!(left_cpu, &values[..4]);
        assert_eq!(right_cpu, &values[4..]);
    }

    // ---- Bit-reverse tests ----

    #[test]
    fn test_bit_reverse_base() {
        init_gpu();
        let values: Vec<BaseField> = (0..16).map(|i| M31::from(i)).collect();
        let mut gpu_col: CudaColumn<BaseField> = values.iter().copied().collect();
        <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut gpu_col);
        let gpu_result = gpu_col.to_cpu();

        let mut cpu_col = values.clone();
        CpuBackend::bit_reverse_column(&mut cpu_col);

        assert_eq!(gpu_result, cpu_col, "GPU bit-reverse must match CPU");
    }

    // ---- NTT evaluate + interpolate roundtrip ----

    #[test]
    fn test_ntt_roundtrip_small() {
        init_gpu();
        ntt_roundtrip(4); // 16 elements
    }

    #[test]
    fn test_ntt_roundtrip_medium() {
        init_gpu();
        ntt_roundtrip(10); // 1024 elements
    }

    #[test]
    fn test_ntt_roundtrip_large() {
        init_gpu();
        ntt_roundtrip(16); // 65536 elements
    }

    fn ntt_roundtrip(log_size: u32) {
        let n = 1usize << log_size;
        let coeffs: Vec<BaseField> = (1..=n as u32).map(M31::from).collect();

        // Build polynomial on GPU
        let gpu_col: CudaColumn<BaseField> = coeffs.iter().copied().collect();
        let poly = CircleCoefficients::<CudaBackend>::new(gpu_col);

        // Evaluate on a domain
        let domain = CanonicCoset::new(log_size + 1).circle_domain();
        let coset = CanonicCoset::new(log_size + 1).half_coset();
        let twiddles = CudaBackend::precompute_twiddles(coset);
        let eval = CudaBackend::evaluate(&poly, domain, &twiddles);

        // Interpolate back
        let recovered = CudaBackend::interpolate(eval, &twiddles);
        let recovered_coeffs = recovered.coeffs.to_cpu();

        // The recovered polynomial should match original (padded to domain size)
        for i in 0..n {
            assert_eq!(recovered_coeffs[i], coeffs[i],
                "Coefficient mismatch at index {i}");
        }
        // Extended coefficients should be zero
        for i in n..recovered_coeffs.len() {
            assert_eq!(recovered_coeffs[i], M31::zero(),
                "Extended coefficient at {i} should be zero");
        }
    }

    // ---- Eval at point (GPU poly, CPU eval) ----

    // ---- GPU vs CPU NTT compatibility ----

    #[test]
    fn test_gpu_vs_cpu_evaluate_small() {
        init_gpu();
        gpu_vs_cpu_evaluate(3); // 8 elements
    }

    #[test]
    fn test_gpu_vs_cpu_evaluate_medium() {
        init_gpu();
        gpu_vs_cpu_evaluate(6); // 64 elements
    }

    #[test]
    fn test_gpu_vs_cpu_evaluate_large() {
        init_gpu();
        gpu_vs_cpu_evaluate(14); // 16384 elements
    }

    /// Test a single NTT layer in isolation to verify twiddle indexing.
    #[test]
    fn test_single_layer_gpu_vs_cpu() {
        init_gpu();
        // Use log_size=2 for a tiny test (4 coefficients, domain size 8)
        let log_size = 2u32;
        let n = 1usize << (log_size + 1); // domain size = 8
        let coeffs: Vec<BaseField> = (1..=n as u32).map(M31::from).collect();
        let domain = CanonicCoset::new(log_size + 1).circle_domain();
        let coset = CanonicCoset::new(log_size + 1).half_coset();

        let cpu_twiddles = CpuBackend::precompute_twiddles(coset);

        eprintln!("Test: log_size={log_size}, domain_size={n}, coset_log_size={}", coset.log_size);
        eprintln!("Twiddle buffer ({} values):", cpu_twiddles.twiddles.len());
        for (i, t) in cpu_twiddles.twiddles.iter().enumerate() {
            eprintln!("  [{i}] = {}", t.0);
        }

        // CPU evaluate
        let cpu_poly = stwo::prover::backend::cpu::CpuCirclePoly::new(coeffs[..4].to_vec());
        let cpu_eval = CpuBackend::evaluate(&cpu_poly, domain, &cpu_twiddles);
        eprintln!("CPU evaluate result:");
        for (i, v) in cpu_eval.values.iter().enumerate() {
            eprintln!("  [{i}] = {}", v.0);
        }

        // GPU: upload coefficients, pad to domain size, run ONE layer at a time
        let mut gpu_data: Vec<u32> = coeffs[..4].iter().map(|m| m.0).collect();
        gpu_data.resize(n, 0); // zero-pad
        let mut d_data = vortexstark::device::DeviceBuffer::from_host(&gpu_data);

        let twid_u32: Vec<u32> = cpu_twiddles.twiddles.iter().map(|t| t.0).collect();
        let d_twid = vortexstark::device::DeviceBuffer::from_host(&twid_u32);

        // Run the full stwo NTT
        unsafe {
            vortexstark::cuda::ffi::cuda_stwo_ntt_evaluate(
                d_data.as_mut_ptr(), d_twid.as_ptr(), n as u32,
            );
        }
        let gpu_result = d_data.to_host();
        eprintln!("GPU evaluate result:");
        for (i, v) in gpu_result.iter().enumerate() {
            eprintln!("  [{i}] = {v}");
        }

        // Compare
        for i in 0..n {
            assert_eq!(gpu_result[i], cpu_eval.values[i].0,
                "Mismatch at {i}: GPU={} CPU={}", gpu_result[i], cpu_eval.values[i].0);
        }
    }

    #[test]
    fn test_twiddle_comparison() {
        init_gpu();
        let log_size = 3u32;
        let domain = CanonicCoset::new(log_size + 1).circle_domain();
        let coset = CanonicCoset::new(log_size + 1).half_coset();

        // CPU twiddles
        let cpu_twiddles = CpuBackend::precompute_twiddles(coset);
        eprintln!("CPU twiddles ({} values):", cpu_twiddles.twiddles.len());
        for (i, t) in cpu_twiddles.twiddles.iter().enumerate().take(20) {
            eprintln!("  [{i}] = {}", t.0);
        }
        eprintln!("CPU itwiddles ({} values):", cpu_twiddles.itwiddles.len());
        for (i, t) in cpu_twiddles.itwiddles.iter().enumerate().take(20) {
            eprintln!("  [{i}] = {}", t.0);
        }
        eprintln!("CPU root_coset: initial=({}, {}), step=({}, {}), log_size={}",
            coset.initial.x.0, coset.initial.y.0,
            coset.step.x.0, coset.step.y.0, coset.log_size);

        // GPU twiddles
        let cache = crate::poly_ops::twiddle_cache_for_coset(&coset);
        let gpu_line_twiddles = cache.d_twiddles.to_host();
        let gpu_circle_twiddles = cache.d_circle_twids.to_host();
        eprintln!("\nGPU line twiddles ({} values):", gpu_line_twiddles.len());
        for (i, t) in gpu_line_twiddles.iter().enumerate().take(20) {
            eprintln!("  [{i}] = {t}");
        }
        eprintln!("GPU circle twiddles ({} values):", gpu_circle_twiddles.len());
        for (i, t) in gpu_circle_twiddles.iter().enumerate().take(20) {
            eprintln!("  [{i}] = {t}");
        }
        eprintln!("GPU cache: log_n={}, offsets={:?}, sizes={:?}",
            cache.log_n, cache.layer_offsets, cache.layer_sizes);
    }

    fn gpu_vs_cpu_evaluate(log_size: u32) {
        let n = 1usize << log_size;
        let coeffs: Vec<BaseField> = (1..=n as u32).map(M31::from).collect();
        let domain = CanonicCoset::new(log_size + 1).circle_domain();
        let coset = CanonicCoset::new(log_size + 1).half_coset();

        // CPU evaluate
        let cpu_poly = stwo::prover::backend::cpu::CpuCirclePoly::new(coeffs.clone());
        let cpu_twiddles = CpuBackend::precompute_twiddles(coset);
        let cpu_eval = CpuBackend::evaluate(&cpu_poly, domain, &cpu_twiddles);
        let cpu_vals = cpu_eval.values.clone();

        // GPU evaluate (using our GPU NTT)
        let gpu_col: CudaColumn<BaseField> = coeffs.iter().copied().collect();
        let gpu_poly = CircleCoefficients::<CudaBackend>::new(gpu_col);

        // GPU evaluate via stwo-compatible NTT kernel
        let extended = CudaBackend::extend(&gpu_poly, domain.log_size());
        let mut gpu_vals_buf = extended.coeffs;
        let d_twiddles = vortexstark::device::DeviceBuffer::from_host(
            &cpu_twiddles.twiddles.iter().map(|t| t.0).collect::<Vec<u32>>()
        );
        unsafe {
            vortexstark::cuda::ffi::cuda_stwo_ntt_evaluate(
                gpu_vals_buf.buf.as_mut_ptr(), d_twiddles.as_ptr(),
                (1u32 << (log_size + 1)),
            );
        }
        let gpu_vals = gpu_vals_buf.to_cpu();

        // Compare
        let eval_size = 1 << (log_size + 1);
        let mut mismatches = 0;
        for i in 0..eval_size {
            if gpu_vals[i] != cpu_vals[i] {
                if mismatches < 5 {
                    eprintln!("MISMATCH at {i}: GPU={} CPU={}", gpu_vals[i].0, cpu_vals[i].0);
                }
                mismatches += 1;
            }
        }
        if mismatches > 0 {
            panic!("{mismatches}/{eval_size} mismatches in evaluate at log_size={log_size}");
        }
    }

    #[test]
    fn test_gpu_vs_cpu_interpolate_small() {
        init_gpu();
        gpu_vs_cpu_interpolate(3);
    }

    #[test]
    fn test_gpu_vs_cpu_interpolate_medium() {
        init_gpu();
        gpu_vs_cpu_interpolate(10);
    }

    fn gpu_vs_cpu_interpolate(log_size: u32) {
        let n = 1usize << log_size;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let coset = domain.half_coset;

        // Create evaluation values
        let vals: Vec<BaseField> = (1..=n as u32).map(M31::from).collect();

        // CPU interpolate
        let cpu_eval = stwo::prover::poly::circle::CircleEvaluation::<
            CpuBackend, BaseField, stwo::prover::poly::BitReversedOrder
        >::new(domain, vals.clone());
        let cpu_twiddles = CpuBackend::precompute_twiddles(coset);
        let cpu_poly = CpuBackend::interpolate(cpu_eval, &cpu_twiddles);

        // GPU interpolate via stwo-compatible NTT kernel
        let gpu_col: CudaColumn<BaseField> = vals.iter().copied().collect();
        let mut gpu_buf = gpu_col;
        let d_itwiddles = vortexstark::device::DeviceBuffer::from_host(
            &cpu_twiddles.itwiddles.iter().map(|t| t.0).collect::<Vec<u32>>()
        );
        unsafe {
            vortexstark::cuda::ffi::cuda_stwo_ntt_interpolate(
                gpu_buf.buf.as_mut_ptr(), d_itwiddles.as_ptr(), n as u32,
            );
        }
        let gpu_coeffs = gpu_buf.to_cpu();

        // Compare
        let mut mismatches = 0;
        for i in 0..n {
            if gpu_coeffs[i] != cpu_poly.coeffs[i] {
                if mismatches < 5 {
                    eprintln!("MISMATCH at {i}: GPU={} CPU={}", gpu_coeffs[i].0, cpu_poly.coeffs[i].0);
                }
                mismatches += 1;
            }
        }
        if mismatches > 0 {
            panic!("{mismatches}/{n} mismatches in interpolate at log_size={log_size}");
        }
    }

    #[test]
    fn test_eval_at_point() {
        init_gpu();
        // Polynomial 1 + 2y + 3x + 4xy
        let coeffs = [1, 3, 2, 4].map(|v| M31::from(v as u32));
        let gpu_col: CudaColumn<BaseField> = coeffs.iter().copied().collect();
        let poly = CircleCoefficients::<CudaBackend>::new(gpu_col);

        let cpu_poly = stwo::prover::backend::cpu::CpuCirclePoly::new(coeffs.to_vec());

        let x: SecureField = M31::from(5).into();
        let y: SecureField = M31::from(8).into();
        let point = stwo::core::circle::CirclePoint { x, y };

        let gpu_eval = CudaBackend::eval_at_point(&poly, point);
        let cpu_eval = CpuBackend::eval_at_point(&cpu_poly, point);

        assert_eq!(gpu_eval, cpu_eval, "eval_at_point must match CPU");
    }

    // ---- GPU register-based bytecode constraint evaluation kernel tests ----

    /// Test the GPU register bytecode constraint eval kernel directly with a hand-crafted
    /// bytecode program and known trace data.
    ///
    /// Program: flag * (1 - flag) = 0
    /// Register bytecode:
    ///   r0 = load_trace[flat_col=0, offset=0]    -- flag
    ///   r1 = load_const(1)                        -- 1
    ///   r2 = r1 - r0                              -- (1 - flag)
    ///   r3 = r0 * r2                              -- flag * (1 - flag)
    ///   add_constraint r3
    ///
    /// Note: flag is used twice (r0 in both Sub and Mul) — this is the Clone pattern
    /// that broke the stack-based VM. In the register VM it just works.
    ///
    /// Trace: 4 rows with values [0, 1, 0, 1] (all valid flags).
    /// Expected: all constraint values are 0, so accumulator stays zero.
    #[test]
    fn test_gpu_bytecode_kernel_zero_constraints() {
        use vortexstark::device::DeviceBuffer;
        use vortexstark::cuda::ffi;

        init_gpu();

        // Trace: single column, 4 rows, values [0, 1, 0, 1]
        let trace_data: Vec<u32> = vec![0, 1, 0, 1];
        let d_trace_col = DeviceBuffer::from_host(&trace_data);
        let trace_col_ptr = d_trace_col.as_ptr();
        let d_col_ptrs = DeviceBuffer::from_host(&[trace_col_ptr]);
        let d_col_sizes = DeviceBuffer::from_host(&[4u32]);

        // Register-based bytecode: flag * (1 - flag), one constraint
        // Encoding format:
        //   LoadTrace:  [0x03:8 | dst:8 | flat_col:14 | sign:1 | abs_offset:1]
        //   LoadConst:  [0x01:8 | dst:8 | value:16]
        //   Sub:        [0x11:8 | dst:8 | src1:8 | src2:8]
        //   Mul:        [0x12:8 | dst:8 | src1:8 | src2:8]
        //   AddConstr:  [0x40:8 | src:8 | 0:16]
        let bytecode: Vec<u32> = vec![
            // r0 = load_trace[flat_col=0, offset=0]: operand = (0 << 2) | (0 << 1) | 0 = 0
            (0x03u32 << 24) | (0 << 16) | 0,
            // r1 = load_const(1): small value fits in 16 bits
            (0x01u32 << 24) | (1 << 16) | 1,
            // r2 = r1 - r0: [0x11 | dst=2 | src1=1 | src2=0]
            (0x11u32 << 24) | (2 << 16) | (1 << 8) | 0,
            // r3 = r0 * r2: [0x12 | dst=3 | src1=0 | src2=2]
            (0x12u32 << 24) | (3 << 16) | (0 << 8) | 2,
            // add_constraint r3: [0x40 | src=3 | 0]
            (0x40u32 << 24) | (3 << 16),
        ];
        let d_bytecode = DeviceBuffer::from_host(&bytecode);

        // Random coeff powers: 1 constraint, so 4 u32s (one QM31)
        let coeff: Vec<u32> = vec![1, 0, 0, 0];
        let d_coeff = DeviceBuffer::from_host(&coeff);

        let denom_inv: Vec<u32> = vec![1];
        let d_denom_inv = DeviceBuffer::from_host(&denom_inv);

        // Output accumulators: 4 rows, initialized to zero
        let mut d_accum0 = DeviceBuffer::<u32>::alloc(4);
        let mut d_accum1 = DeviceBuffer::<u32>::alloc(4);
        let mut d_accum2 = DeviceBuffer::<u32>::alloc(4);
        let mut d_accum3 = DeviceBuffer::<u32>::alloc(4);
        d_accum0.zero();
        d_accum1.zero();
        d_accum2.zero();
        d_accum3.zero();

        unsafe {
            ffi::cuda_bytecode_constraint_eval(
                d_bytecode.as_ptr(),
                bytecode.len() as u32,
                d_col_ptrs.as_ptr() as *const *const u32,
                d_col_sizes.as_ptr(),
                1, // n_trace_cols
                4, // n_rows
                4, // trace_n_rows
                d_coeff.as_ptr(),
                d_denom_inv.as_ptr(),
                0, // log_expand
                d_accum0.as_mut_ptr(),
                d_accum1.as_mut_ptr(),
                d_accum2.as_mut_ptr(),
                d_accum3.as_mut_ptr(),
            );
            ffi::cuda_device_sync();
        }

        let err = unsafe { ffi::cudaGetLastError() };
        assert_eq!(err, 0, "CUDA kernel error: {err}");

        // Download results
        let out0 = d_accum0.to_host();
        let out1 = d_accum1.to_host();
        let out2 = d_accum2.to_host();
        let out3 = d_accum3.to_host();

        // flag*(1-flag) = 0 for all rows (flag=0 or flag=1), so output should be all zeros
        for row in 0..4 {
            assert_eq!(out0[row], 0, "accum0[{row}] should be 0, got {}", out0[row]);
            assert_eq!(out1[row], 0, "accum1[{row}] should be 0, got {}", out1[row]);
            assert_eq!(out2[row], 0, "accum2[{row}] should be 0, got {}", out2[row]);
            assert_eq!(out3[row], 0, "accum3[{row}] should be 0, got {}", out3[row]);
        }
    }

    /// Test with a constraint that has non-zero output.
    /// Program: x - 5 = 0 (constraint value = x - 5)
    /// Register bytecode:
    ///   r0 = load_trace[flat_col=0, offset=0]    -- x
    ///   r1 = r0 + const(-5 mod p)                -- x - 5
    ///   add_constraint r1
    ///
    /// Trace: [3, 5, 7, 5] -> constraint values: [-2, 0, 2, 0] in M31
    #[test]
    fn test_gpu_bytecode_kernel_nonzero_constraints() {
        use vortexstark::device::DeviceBuffer;
        use vortexstark::cuda::ffi;

        init_gpu();

        let p = 0x7FFFFFFFu32; // M31 modulus

        let trace_data: Vec<u32> = vec![3, 5, 7, 5];
        let d_trace_col = DeviceBuffer::from_host(&trace_data);
        let trace_col_ptr = d_trace_col.as_ptr();
        let d_col_ptrs = DeviceBuffer::from_host(&[trace_col_ptr]);
        let d_col_sizes = DeviceBuffer::from_host(&[4u32]);

        // Register-based bytecode: r0 = load_trace, r1 = r0 + const(p-5), add_constraint r1
        // AddConst encoding: [0x14:8 | dst:8 | src:8 | 0:8] + value_word
        let neg5 = p - 5;
        let bytecode: Vec<u32> = vec![
            // r0 = load_trace[flat_col=0, offset=0]
            (0x03u32 << 24) | (0 << 16) | 0,
            // r1 = r0 + const(neg5): header + value word
            (0x14u32 << 24) | (1 << 16) | (0 << 8),
            neg5,
            // add_constraint r1
            (0x40u32 << 24) | (1 << 16),
        ];
        let d_bytecode = DeviceBuffer::from_host(&bytecode);

        let coeff: Vec<u32> = vec![1, 0, 0, 0];
        let d_coeff = DeviceBuffer::from_host(&coeff);

        let denom_inv: Vec<u32> = vec![1];
        let d_denom_inv = DeviceBuffer::from_host(&denom_inv);

        let mut d_accum0 = DeviceBuffer::<u32>::alloc(4);
        let mut d_accum1 = DeviceBuffer::<u32>::alloc(4);
        let mut d_accum2 = DeviceBuffer::<u32>::alloc(4);
        let mut d_accum3 = DeviceBuffer::<u32>::alloc(4);
        d_accum0.zero();
        d_accum1.zero();
        d_accum2.zero();
        d_accum3.zero();

        unsafe {
            ffi::cuda_bytecode_constraint_eval(
                d_bytecode.as_ptr(),
                bytecode.len() as u32,
                d_col_ptrs.as_ptr() as *const *const u32,
                d_col_sizes.as_ptr(),
                1, 4, 4,
                d_coeff.as_ptr(),
                d_denom_inv.as_ptr(),
                0,
                d_accum0.as_mut_ptr(),
                d_accum1.as_mut_ptr(),
                d_accum2.as_mut_ptr(),
                d_accum3.as_mut_ptr(),
            );
            ffi::cuda_device_sync();
        }

        let err = unsafe { ffi::cudaGetLastError() };
        assert_eq!(err, 0, "CUDA kernel error: {err}");

        let out0 = d_accum0.to_host();

        // Expected: (3-5) mod p = p-2, (5-5) = 0, (7-5) = 2, (5-5) = 0
        assert_eq!(out0[0], p - 2, "row 0: 3-5 = -2 mod p");
        assert_eq!(out0[1], 0,     "row 1: 5-5 = 0");
        assert_eq!(out0[2], 2,     "row 2: 7-5 = 2");
        assert_eq!(out0[3], 0,     "row 3: 5-5 = 0");
    }

    /// Test the full encode -> GPU path using the register-based bytecode encoder.
    #[test]
    fn test_gpu_bytecode_via_encoder() {
        use vortexstark::device::DeviceBuffer;
        use vortexstark::cuda::ffi;
        use crate::constraint_eval::bytecode::{BytecodeOp, BytecodeProgram};

        init_gpu();

        // Build a register-based bytecode program: a + b - c = 0
        // r0 = trace[0,0], r1 = trace[0,1], r2 = r0 + r1, r3 = trace[0,2], r4 = r2 - r3
        let prog = BytecodeProgram {
            ops: vec![
                BytecodeOp::LoadTrace { dst: 0, interaction: 0, col_idx: 0, offset: 0 },
                BytecodeOp::LoadTrace { dst: 1, interaction: 0, col_idx: 1, offset: 0 },
                BytecodeOp::Add { dst: 2, src1: 0, src2: 1 },
                BytecodeOp::LoadTrace { dst: 3, interaction: 0, col_idx: 2, offset: 0 },
                BytecodeOp::Sub { dst: 4, src1: 2, src2: 3 },
                BytecodeOp::AddConstraint { src: 4 },
            ],
            n_constraints: 1,
            n_trace_accesses: 3,
            n_registers: 5,
        };

        let encoded = prog.encode();

        // Remap LoadTrace: (interaction=0, col_idx=N) -> flat_col=N
        // For interaction=0, flat_col = col_idx. The register-based encoding
        // puts flat_col in bits [15:2] of the operand. Since interaction=0
        // and the encoding uses interaction:4|col_idx:10|sign:1|abs:1,
        // and flat_col replaces interaction:col_idx in the same bit positions,
        // the encoding is already correct for interaction=0.

        // Trace: 3 columns, 4 rows
        let col_a: Vec<u32> = vec![1, 2, 3, 4];
        let col_b: Vec<u32> = vec![10, 20, 30, 40];
        let col_c: Vec<u32> = vec![11, 22, 33, 44];

        let d_col_a = DeviceBuffer::from_host(&col_a);
        let d_col_b = DeviceBuffer::from_host(&col_b);
        let d_col_c = DeviceBuffer::from_host(&col_c);

        let col_ptrs: Vec<*const u32> = vec![
            d_col_a.as_ptr(),
            d_col_b.as_ptr(),
            d_col_c.as_ptr(),
        ];
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);
        let d_col_sizes = DeviceBuffer::from_host(&[4u32, 4, 4]);

        let d_bytecode = DeviceBuffer::from_host(&encoded);
        let coeff: Vec<u32> = vec![1, 0, 0, 0];
        let d_coeff = DeviceBuffer::from_host(&coeff);
        let denom_inv: Vec<u32> = vec![1];
        let d_denom_inv = DeviceBuffer::from_host(&denom_inv);

        let mut d_accum0 = DeviceBuffer::<u32>::alloc(4);
        let mut d_accum1 = DeviceBuffer::<u32>::alloc(4);
        let mut d_accum2 = DeviceBuffer::<u32>::alloc(4);
        let mut d_accum3 = DeviceBuffer::<u32>::alloc(4);
        d_accum0.zero();
        d_accum1.zero();
        d_accum2.zero();
        d_accum3.zero();

        unsafe {
            ffi::cuda_bytecode_constraint_eval(
                d_bytecode.as_ptr(),
                encoded.len() as u32,
                d_col_ptrs.as_ptr() as *const *const u32,
                d_col_sizes.as_ptr(),
                3, 4, 4,
                d_coeff.as_ptr(),
                d_denom_inv.as_ptr(),
                0,
                d_accum0.as_mut_ptr(),
                d_accum1.as_mut_ptr(),
                d_accum2.as_mut_ptr(),
                d_accum3.as_mut_ptr(),
            );
            ffi::cuda_device_sync();
        }

        let err = unsafe { ffi::cudaGetLastError() };
        assert_eq!(err, 0, "CUDA kernel error: {err}");

        let out0 = d_accum0.to_host();

        // Expected: a + b - c = [1+10-11, 2+20-22, 3+30-33, 4+40-44] = [0, 0, 0, 0]
        for row in 0..4 {
            assert_eq!(out0[row], 0, "row {row}: a+b-c should be 0, got {}", out0[row]);
        }
    }
}
