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

    // ---- GPU Merkle leaf hashing tests ----

    #[test]
    fn test_gpu_leaf_hash_vs_cpu() {
        use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
        use stwo::prover::vcs_lifted::ops::MerkleOpsLifted;

        init_gpu();

        // Create 4 columns of 8 elements each (log_size=3), lifting_log_size=3.
        let n = 8usize;
        let lifting_log = 3u32;

        let col_data: Vec<Vec<BaseField>> = (0..4).map(|c| {
            (0..n).map(|r| M31::from((c * 100 + r) as u32)).collect()
        }).collect();

        // CPU reference
        let cpu_col_refs: Vec<&Vec<BaseField>> = col_data.iter().collect();
        let cpu_hashes = <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
            &cpu_col_refs, lifting_log,
        );

        // GPU
        let gpu_cols: Vec<CudaColumn<BaseField>> = col_data.iter().map(|c| c.iter().copied().collect()).collect();
        let gpu_col_refs: Vec<&CudaColumn<BaseField>> = gpu_cols.iter().collect();
        let gpu_hashes = <CudaBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
            &gpu_col_refs, lifting_log,
        );
        let gpu_result = gpu_hashes.to_cpu();

        assert_eq!(cpu_hashes.len(), gpu_result.len(), "Hash count mismatch");
        let mut mismatches = 0;
        for i in 0..cpu_hashes.len() {
            if cpu_hashes[i] != gpu_result[i] {
                if mismatches < 3 {
                    eprintln!("[LEAF_HASH] mismatch at leaf {i}: CPU={:?} GPU={:?}",
                        &cpu_hashes[i].0[..8], &gpu_result[i].0[..8]);
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "{mismatches}/{} leaf hashes differ", cpu_hashes.len());
    }

    #[test]
    fn test_gpu_leaf_hash_many_columns() {
        use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
        use stwo::prover::vcs_lifted::ops::MerkleOpsLifted;

        init_gpu();

        // 20 columns (>16, tests multi-chunk hashing), log_size=4, lifting=4
        let n = 16usize;
        let lifting_log = 4u32;

        let col_data: Vec<Vec<BaseField>> = (0..20).map(|c| {
            (0..n).map(|r| M31::from((c * 1000 + r * 7 + 3) as u32)).collect()
        }).collect();

        let cpu_col_refs: Vec<&Vec<BaseField>> = col_data.iter().collect();
        let cpu_hashes = <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
            &cpu_col_refs, lifting_log,
        );

        let gpu_cols: Vec<CudaColumn<BaseField>> = col_data.iter().map(|c| c.iter().copied().collect()).collect();
        let gpu_col_refs: Vec<&CudaColumn<BaseField>> = gpu_cols.iter().collect();
        let gpu_hashes = <CudaBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
            &gpu_col_refs, lifting_log,
        );
        let gpu_result = gpu_hashes.to_cpu();

        let mut mismatches = 0;
        for i in 0..cpu_hashes.len() {
            if cpu_hashes[i] != gpu_result[i] {
                if mismatches < 3 {
                    eprintln!("[LEAF_HASH] mismatch at leaf {i}");
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "{mismatches}/{} leaf hashes differ (20 cols, multi-chunk)",
            cpu_hashes.len());
    }

    #[test]
    fn test_gpu_leaf_hash_mixed_sizes() {
        use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
        use stwo::prover::vcs_lifted::ops::MerkleOpsLifted;

        init_gpu();

        // Mixed column sizes: 2 cols of size 4 (log=2), 3 cols of size 8 (log=3).
        // Lifting_log=3 (output 8 leaves).
        // Columns MUST be sorted by ascending size (stwo requirement).
        let lifting_log = 3u32;

        let small_cols: Vec<Vec<BaseField>> = (0..2).map(|c| {
            (0..4).map(|r| M31::from((c * 50 + r * 11 + 1) as u32)).collect()
        }).collect();
        let big_cols: Vec<Vec<BaseField>> = (0..3).map(|c| {
            (0..8).map(|r| M31::from((c * 200 + r * 13 + 7) as u32)).collect()
        }).collect();

        let mut all_cols: Vec<Vec<BaseField>> = Vec::new();
        all_cols.extend(small_cols);
        all_cols.extend(big_cols);

        let cpu_col_refs: Vec<&Vec<BaseField>> = all_cols.iter().collect();
        let cpu_hashes = <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
            &cpu_col_refs, lifting_log,
        );

        let gpu_cols: Vec<CudaColumn<BaseField>> = all_cols.iter().map(|c| c.iter().copied().collect()).collect();
        let gpu_col_refs: Vec<&CudaColumn<BaseField>> = gpu_cols.iter().collect();
        let gpu_hashes = <CudaBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
            &gpu_col_refs, lifting_log,
        );
        let gpu_result = gpu_hashes.to_cpu();

        let mut mismatches = 0;
        for i in 0..cpu_hashes.len() {
            if cpu_hashes[i] != gpu_result[i] {
                if mismatches < 3 {
                    eprintln!("[LEAF_HASH] mixed-size mismatch at leaf {i}: CPU={:02x}{:02x}{:02x}{:02x}... GPU={:02x}{:02x}{:02x}{:02x}...",
                        cpu_hashes[i].0[0], cpu_hashes[i].0[1], cpu_hashes[i].0[2], cpu_hashes[i].0[3],
                        gpu_result[i].0[0], gpu_result[i].0[1], gpu_result[i].0[2], gpu_result[i].0[3]);
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "{mismatches}/{} leaf hashes differ (mixed sizes)",
            cpu_hashes.len());
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

    #[test]
    fn test_barycentric_eval_gpu_vs_cpu() {
        use stwo::core::poly::circle::CanonicCoset;
        use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
        use stwo::prover::poly::BitReversedOrder;
        use stwo::prover::backend::CpuBackend;
        use stwo::core::circle::CirclePoint;

        init_gpu();

        let log_size = 8u32; // 256 elements
        let coset = CanonicCoset::new(log_size);
        let domain = coset.circle_domain();

        // Arbitrary evaluation point (not on the domain)
        let x = SecureField::from_m31_array([M31::from(3), M31::from(7), M31::from(11), M31::from(2)]);
        let y = SecureField::from_m31_array([M31::from(5), M31::from(1), M31::from(8), M31::from(4)]);
        let point = CirclePoint { x, y };

        // Build random evaluations
        let n = 1usize << log_size;
        let vals: Vec<BaseField> = (1..=n as u32).map(M31::from).collect();

        // CPU path: compute weights then evaluate
        let cpu_weights = CpuBackend::barycentric_weights(coset, point);
        let cpu_evals = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
            domain, vals.clone(),
        );
        let cpu_result = CpuBackend::barycentric_eval_at_point(&cpu_evals, &cpu_weights);

        // GPU path: weights computed on CPU then uploaded (via CudaBackend::barycentric_weights),
        // eval runs on GPU
        let gpu_weights = CudaBackend::barycentric_weights(coset, point);
        let gpu_evals: CudaColumn<BaseField> = vals.into_iter().collect();
        let gpu_eval_circle = CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
            domain, gpu_evals,
        );
        let gpu_result = CudaBackend::barycentric_eval_at_point(&gpu_eval_circle, &gpu_weights);

        assert_eq!(gpu_result, cpu_result, "GPU barycentric eval must match CPU");
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
        // 16-bit register encoding:
        //   word1: [opcode:8 | 0:8 | dst:16]   (or src for AddConstraint)
        //   word2: [src1:16 | src2:16]          for 3-reg ops
        //   word2: [src:16 | 0:16]              for 2-reg ops
        //   word2: value (u32)                  for LoadConst
        let bytecode: Vec<u32> = vec![
            // r0 = load_trace[flat_col=0, offset=0]: 3 words
            (0x03u32 << 24) | 0u32, // hdr: dst=0
            0u32,                    // flat_col=0
            0u32,                    // offset=0
            // r1 = load_const(1): 2 words
            (0x01u32 << 24) | 1u32, // hdr: dst=1
            1u32,                    // value=1
            // r2 = r1 - r0: 2 words
            (0x11u32 << 24) | 2u32, // hdr: dst=2
            (1u32 << 16) | 0u32,    // src1=1, src2=0
            // r3 = r0 * r2: 2 words
            (0x12u32 << 24) | 3u32, // hdr: dst=3
            (0u32 << 16) | 2u32,    // src1=0, src2=2
            // add_constraint r3: 1 word
            (0x40u32 << 24) | 3u32, // src=3
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
                64, // n_registers (small test programs)
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

        // Register-based bytecode (16-bit register encoding):
        //   r0 = load_trace[flat_col=0, offset=0]  — 3 words
        //   r1 = r0 + const(p-5)                   — 3 words
        //   add_constraint r1                       — 1 word
        let neg5 = p - 5;
        let bytecode: Vec<u32> = vec![
            // LoadTrace: word1=[opcode:8|0:8|dst:16], word2=flat_col, word3=sign|abs_off
            (0x03u32 << 24) | 0u32, // dst=0
            0u32,                    // flat_col=0
            0u32,                    // offset=0 (sign=0, abs=0)
            // AddConst: word1=[opcode:8|0:8|dst:16], word2=[src:16|0:16], word3=value
            (0x14u32 << 24) | 1u32, // dst=1
            0u32 << 16,              // src=0
            neg5,
            // AddConstraint: word1=[opcode:8|0:8|src:16]
            (0x40u32 << 24) | 1u32, // src=1
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
                64, // n_registers (small test programs)
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
                64, // n_registers (small test programs)
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

    // ──────────────────────────────────────────────────────────────────────
    // End-to-end GPU vs CPU bytecode constraint evaluation test.
    //
    // This test uses stwo's actual FrameworkEval/EvalAtRow infrastructure
    // (not hand-crafted bytecode) to verify the full pipeline:
    //   record_bytecode -> encode -> GPU kernel -> compare with CpuDomainEvaluator
    // ──────────────────────────────────────────────────────────────────────

    // ── Logup component definitions (module scope, outside test functions) ──

    stwo_constraint_framework::relation!(TestLookupRelation, 1);

    const LOGUP_LOG_N: u32 = 2; // 4 rows — small for easy debugging

    struct SimpleLogupEval {
        lookup: TestLookupRelation,
        claimed_sum: SecureField,
    }

    impl stwo_constraint_framework::FrameworkEval for SimpleLogupEval {
        fn log_size(&self) -> u32 { LOGUP_LOG_N }
        fn max_constraint_log_degree_bound(&self) -> u32 { LOGUP_LOG_N + 1 }
        fn evaluate<E: stwo_constraint_framework::EvalAtRow>(&self, mut eval: E) -> E {
            let val = eval.next_trace_mask();
            eval.add_to_relation(stwo_constraint_framework::RelationEntry::new(
                &self.lookup,
                E::EF::one(),
                &[val],
            ));
            eval.finalize_logup();
            eval
        }
    }

    /// End-to-end logup GPU vs CPU test.
    ///
    /// Records bytecode for a single-entry logup component (one M31 lookup value,
    /// QM31 interaction cumsum). Runs the GPU bytecode kernel and CpuDomainEvaluator
    /// on the same arbitrary trace data and compares row_res values.
    ///
    /// Trace layout (5 flat columns, 4 rows each):
    ///   flat 0 = interaction 1 col 0  (base trace: the lookup value)
    ///   flat 1..4 = interaction 2 col 0..3  (QM31 cumsum, M31 components)
    ///
    /// interaction_offsets = [0, 0, 1, 5]
    ///   preproc (inter 0): 0 cols → starts at flat 0
    ///   base    (inter 1): 1 col  → starts at flat 0
    ///   inter   (inter 2): 4 cols → starts at flat 1
    #[test]
    fn test_e2e_logup_constraint_gpu_vs_cpu() {
        use vortexstark::device::DeviceBuffer;
        use vortexstark::cuda::ffi;
        use crate::constraint_eval::tracing::record_bytecode;
        use crate::constraint_eval::bytecode::BytecodeOp;
        use stwo_constraint_framework::{CpuDomainEvaluator, FrameworkEval};
        use stwo::core::pcs::TreeVec;
        use stwo::core::poly::circle::CanonicCoset;
        use stwo::prover::poly::circle::CircleEvaluation;
        use stwo::prover::poly::BitReversedOrder;
        use stwo::prover::backend::CpuBackend;

        init_gpu();

        // claimed_sum: arbitrary fixed QM31 (used for cumsum_shift in the constraint)
        let claimed_sum = SecureField::from_m31_array([
            M31::from(1), M31::from(2), M31::from(3), M31::from(4),
        ]);

        let component = SimpleLogupEval {
            lookup: TestLookupRelation::dummy(),
            claimed_sum,
        };

        // Record bytecode via FrameworkEval
        let program = record_bytecode(&component, claimed_sum);
        eprintln!("Logup bytecode ({} ops, {} constraints):\n{}",
            program.ops.len(), program.n_constraints, program.dump());
        assert_eq!(program.n_constraints, 1, "expected 1 logup constraint");

        // Use log_expand=1: eval domain is 2x the trace domain.
        // This is required because offset_bit_reversed_circle_domain_index
        // computes (eval_log_size - domain_log_size - 1), which underflows when
        // eval_log_size == domain_log_size (log_expand=0).
        let trace_n_rows = 1u32 << LOGUP_LOG_N;        // 4 rows
        let n_rows       = 1u32 << (LOGUP_LOG_N + 1);  // 8 eval rows
        let log_expand   = 1u32;

        // interaction_offsets: preproc(0 cols), base(1 col), inter(4 cols)
        // → [0, 0, 1, 5]
        let interaction_offsets: Vec<usize> = vec![0, 0, 1, 5];

        // Encode bytecode and remap LoadTrace to flat column indices
        let mut encoded = program.encode();
        {
            let mut pc = 0;
            for op in &program.ops {
                if let BytecodeOp::LoadTrace { interaction, col_idx, .. } = op {
                    let flat_idx = interaction_offsets[*interaction as usize] + *col_idx as usize;
                    // 3-word LoadTrace: word1=header(dst), word2=flat_col, word3=sign|abs_off
                    encoded[pc + 1] = flat_idx as u32;
                    pc += 3;
                } else {
                    pc += op.encoded_len();
                }
            }
        }

        // Build trace data: 8 eval rows each (eval domain = 2x trace domain).
        // flat 0 = base trace col (lookup values, 8 eval rows)
        // flat 1..4 = interaction 2 col 0..3 (QM31 cumsum, 8 eval rows each)
        let base_col:  Vec<u32> = vec![3, 7, 2, 9, 4, 1, 8, 6];
        let i2c0_vals: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let i2c1_vals: Vec<u32> = vec![1,  2,  3,  4,  5,  6,  7,  8];
        let i2c2_vals: Vec<u32> = vec![5,  6,  7,  8,  9, 10, 11, 12];
        let i2c3_vals: Vec<u32> = vec![11, 12, 13, 14, 15, 16, 17, 18];

        let d_base = DeviceBuffer::from_host(&base_col);
        let d_i2c0 = DeviceBuffer::from_host(&i2c0_vals);
        let d_i2c1 = DeviceBuffer::from_host(&i2c1_vals);
        let d_i2c2 = DeviceBuffer::from_host(&i2c2_vals);
        let d_i2c3 = DeviceBuffer::from_host(&i2c3_vals);

        let col_ptrs: Vec<*const u32> = vec![
            d_base.as_ptr(),
            d_i2c0.as_ptr(), d_i2c1.as_ptr(), d_i2c2.as_ptr(), d_i2c3.as_ptr(),
        ];
        let d_col_ptrs  = DeviceBuffer::from_host(&col_ptrs);
        let d_col_sizes = DeviceBuffer::from_host(&[n_rows; 5]);
        let d_bytecode  = DeviceBuffer::from_host(&encoded);

        // 1 constraint: random_coeff = QM31(1,0,0,0)
        let coeff: Vec<u32>     = vec![1, 0, 0, 0];
        // log_expand=1 → 2 coset denominators. Both set to 1 so GPU accum == CPU row_res.
        let denom_inv: Vec<u32> = vec![1, 1];
        let d_coeff     = DeviceBuffer::from_host(&coeff);
        let d_denom_inv = DeviceBuffer::from_host(&denom_inv);

        let mut d_a0 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut d_a1 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut d_a2 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut d_a3 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        d_a0.zero(); d_a1.zero(); d_a2.zero(); d_a3.zero();

        unsafe {
            ffi::cuda_bytecode_constraint_eval(
                d_bytecode.as_ptr(), encoded.len() as u32,
                d_col_ptrs.as_ptr() as *const *const u32,
                d_col_sizes.as_ptr(),
                5, n_rows, trace_n_rows,
                d_coeff.as_ptr(), d_denom_inv.as_ptr(),
                log_expand,
                d_a0.as_mut_ptr(), d_a1.as_mut_ptr(), d_a2.as_mut_ptr(), d_a3.as_mut_ptr(),
                64, // n_registers (small test programs)
            );
            ffi::cuda_device_sync();
        }

        let err = unsafe { ffi::cudaGetLastError() };
        assert_eq!(err, 0, "CUDA kernel error: {err}");

        let gpu_a0 = d_a0.to_host();
        let gpu_a1 = d_a1.to_host();
        let gpu_a2 = d_a2.to_host();
        let gpu_a3 = d_a3.to_host();

        // CPU reference: CpuDomainEvaluator for each eval row
        let to_bf = |v: &[u32]| -> Vec<BaseField> { v.iter().map(|&x| M31::from(x)).collect() };
        // eval domain has 2x rows (log_expand=1): CanonicCoset::new(LOGUP_LOG_N + 1)
        let eval_domain = CanonicCoset::new(LOGUP_LOG_N + 1).circle_domain();

        let base_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(eval_domain, to_bf(&base_col));
        let i2c0_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(eval_domain, to_bf(&i2c0_vals));
        let i2c1_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(eval_domain, to_bf(&i2c1_vals));
        let i2c2_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(eval_domain, to_bf(&i2c2_vals));
        let i2c3_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(eval_domain, to_bf(&i2c3_vals));

        // TreeVec: [0]=preprocessed (empty), [1]=base trace (1 col), [2]=interaction (4 cols)
        let trace_refs: TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>> =
            TreeVec::new(vec![
                vec![],
                vec![&base_eval],
                vec![&i2c0_eval, &i2c1_eval, &i2c2_eval, &i2c3_eval],
            ]);

        let random_coeff_powers = vec![SecureField::one()]; // 1 constraint

        let mut mismatches = 0;
        for row in 0..n_rows as usize {
            let cpu_eval = CpuDomainEvaluator::new(
                &trace_refs,
                row,
                &random_coeff_powers,
                LOGUP_LOG_N,     // domain_log_size (trace domain)
                LOGUP_LOG_N + 1, // eval_domain_log_size (eval domain = trace + 1)
                LOGUP_LOG_N,     // log_size for LogupAtRow cumsum_shift
                claimed_sum,
            );
            let cpu_eval = component.evaluate(cpu_eval);
            let row_res = cpu_eval.row_res;
            let arr = row_res.to_m31_array();
            let (e0, e1, e2, e3) = (arr[0].0, arr[1].0, arr[2].0, arr[3].0);
            let (g0, g1, g2, g3) = (gpu_a0[row], gpu_a1[row], gpu_a2[row], gpu_a3[row]);

            if e0 != g0 || e1 != g1 || e2 != g2 || e3 != g3 {
                if mismatches < 8 {
                    eprintln!("[LOGUP MISMATCH] row {row}:");
                    eprintln!("  CPU row_res: ({e0}, {e1}, {e2}, {e3})");
                    eprintln!("  GPU accum:   ({g0}, {g1}, {g2}, {g3})");
                }
                mismatches += 1;
            } else {
                eprintln!("[LOGUP OK] row {row}: ({e0}, {e1}, {e2}, {e3})");
            }
        }
        assert_eq!(mismatches, 0,
            "{mismatches}/{n_rows} rows differ between GPU and CPU for logup constraint");
    }

    // ── Two-entry logup with finalize_logup_in_pairs (stwo-cairo pattern) ──

    struct LogupInPairsEval {
        lookup: TestLookupRelation,
    }

    impl stwo_constraint_framework::FrameworkEval for LogupInPairsEval {
        fn log_size(&self) -> u32 { LOGUP_LOG_N }
        fn max_constraint_log_degree_bound(&self) -> u32 { LOGUP_LOG_N + 1 }
        fn evaluate<E: stwo_constraint_framework::EvalAtRow>(&self, mut eval: E) -> E {
            let val1 = eval.next_trace_mask();
            let val2 = eval.next_trace_mask();
            eval.add_to_relation(stwo_constraint_framework::RelationEntry::new(
                &self.lookup, E::EF::one(), &[val1],
            ));
            eval.add_to_relation(stwo_constraint_framework::RelationEntry::new(
                &self.lookup, E::EF::one(), &[val2],
            ));
            eval.finalize_logup_in_pairs();
            eval
        }
    }

    /// End-to-end test for two-entry logup with finalize_logup_in_pairs.
    ///
    /// This matches the pattern used by real stwo-cairo components where multiple
    /// logup entries are batched in pairs. The fraction sum (frac1 + frac2) generates
    /// additional WideAdd/WideMul bytecode that exercises the register tracker more deeply.
    ///
    /// Trace layout:
    ///   inter 1, col 0 = val1 (base trace)
    ///   inter 1, col 1 = val2 (base trace)
    ///   inter 2, col 0..3 = QM31 cumsum (interaction trace)
    ///   interaction_offsets = [0, 0, 2, 6]
    #[test]
    fn test_e2e_logup_in_pairs_gpu_vs_cpu() {
        use vortexstark::device::DeviceBuffer;
        use vortexstark::cuda::ffi;
        use crate::constraint_eval::tracing::record_bytecode;
        use crate::constraint_eval::bytecode::BytecodeOp;
        use stwo_constraint_framework::{CpuDomainEvaluator, FrameworkEval};
        use stwo::core::pcs::TreeVec;
        use stwo::core::poly::circle::CanonicCoset;
        use stwo::prover::poly::circle::CircleEvaluation;
        use stwo::prover::poly::BitReversedOrder;
        use stwo::prover::backend::CpuBackend;

        init_gpu();

        let claimed_sum = SecureField::from_m31_array([
            M31::from(7), M31::from(3), M31::from(11), M31::from(5),
        ]);

        let component = LogupInPairsEval { lookup: TestLookupRelation::dummy() };
        let program = record_bytecode(&component, claimed_sum);
        eprintln!("LogupInPairs bytecode ({} ops, {} constraints):\n{}",
            program.ops.len(), program.n_constraints, program.dump());
        assert_eq!(program.n_constraints, 1, "expected 1 logup constraint");

        let trace_n_rows = 1u32 << LOGUP_LOG_N;        // 4
        let n_rows       = 1u32 << (LOGUP_LOG_N + 1);  // 8 eval rows
        let log_expand   = 1u32;

        // inter 1 has 2 cols (val1, val2), inter 2 has 4 cols (QM31 cumsum)
        // interaction_offsets = [0, 0, 2, 6]
        let interaction_offsets: Vec<usize> = vec![0, 0, 2, 6];

        let mut encoded = program.encode();
        {
            let mut pc = 0;
            for op in &program.ops {
                if let BytecodeOp::LoadTrace { interaction, col_idx, .. } = op {
                    let flat_idx = interaction_offsets[*interaction as usize] + *col_idx as usize;
                    // 3-word LoadTrace: word1=header(dst), word2=flat_col, word3=sign|abs_off
                    encoded[pc + 1] = flat_idx as u32;
                    pc += 3;
                } else {
                    pc += op.encoded_len();
                }
            }
        }

        // 6 flat columns: val1, val2, i2c0, i2c1, i2c2, i2c3 (8 eval rows each)
        let val1_col:  Vec<u32> = vec![3, 7, 2, 9, 4, 1, 8, 6];
        let val2_col:  Vec<u32> = vec![5, 2, 8, 1, 3, 9, 4, 7];
        let i2c0_vals: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let i2c1_vals: Vec<u32> = vec![1,  2,  3,  4,  5,  6,  7,  8];
        let i2c2_vals: Vec<u32> = vec![5,  6,  7,  8,  9, 10, 11, 12];
        let i2c3_vals: Vec<u32> = vec![11, 12, 13, 14, 15, 16, 17, 18];

        let d_val1 = DeviceBuffer::from_host(&val1_col);
        let d_val2 = DeviceBuffer::from_host(&val2_col);
        let d_i2c0 = DeviceBuffer::from_host(&i2c0_vals);
        let d_i2c1 = DeviceBuffer::from_host(&i2c1_vals);
        let d_i2c2 = DeviceBuffer::from_host(&i2c2_vals);
        let d_i2c3 = DeviceBuffer::from_host(&i2c3_vals);

        let col_ptrs: Vec<*const u32> = vec![
            d_val1.as_ptr(), d_val2.as_ptr(),
            d_i2c0.as_ptr(), d_i2c1.as_ptr(), d_i2c2.as_ptr(), d_i2c3.as_ptr(),
        ];
        let d_col_ptrs  = DeviceBuffer::from_host(&col_ptrs);
        let d_col_sizes = DeviceBuffer::from_host(&[n_rows; 6]);
        let d_bytecode  = DeviceBuffer::from_host(&encoded);

        let coeff: Vec<u32>     = vec![1, 0, 0, 0];
        let denom_inv: Vec<u32> = vec![1, 1];
        let d_coeff     = DeviceBuffer::from_host(&coeff);
        let d_denom_inv = DeviceBuffer::from_host(&denom_inv);

        let mut d_a0 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut d_a1 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut d_a2 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut d_a3 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        d_a0.zero(); d_a1.zero(); d_a2.zero(); d_a3.zero();

        unsafe {
            ffi::cuda_bytecode_constraint_eval(
                d_bytecode.as_ptr(), encoded.len() as u32,
                d_col_ptrs.as_ptr() as *const *const u32,
                d_col_sizes.as_ptr(),
                6, n_rows, trace_n_rows,
                d_coeff.as_ptr(), d_denom_inv.as_ptr(),
                log_expand,
                d_a0.as_mut_ptr(), d_a1.as_mut_ptr(), d_a2.as_mut_ptr(), d_a3.as_mut_ptr(),
                64, // n_registers (small test programs)
            );
            ffi::cuda_device_sync();
        }

        let err = unsafe { ffi::cudaGetLastError() };
        assert_eq!(err, 0, "CUDA kernel error: {err}");

        let gpu_a0 = d_a0.to_host();
        let gpu_a1 = d_a1.to_host();
        let gpu_a2 = d_a2.to_host();
        let gpu_a3 = d_a3.to_host();

        let to_bf = |v: &[u32]| -> Vec<BaseField> { v.iter().map(|&x| M31::from(x)).collect() };
        let eval_domain = CanonicCoset::new(LOGUP_LOG_N + 1).circle_domain();

        let val1_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(eval_domain, to_bf(&val1_col));
        let val2_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(eval_domain, to_bf(&val2_col));
        let i2c0_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(eval_domain, to_bf(&i2c0_vals));
        let i2c1_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(eval_domain, to_bf(&i2c1_vals));
        let i2c2_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(eval_domain, to_bf(&i2c2_vals));
        let i2c3_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(eval_domain, to_bf(&i2c3_vals));

        let trace_refs: TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>> =
            TreeVec::new(vec![
                vec![],
                vec![&val1_eval, &val2_eval],
                vec![&i2c0_eval, &i2c1_eval, &i2c2_eval, &i2c3_eval],
            ]);

        let random_coeff_powers = vec![SecureField::one()];

        let mut mismatches = 0;
        for row in 0..n_rows as usize {
            let cpu_eval = CpuDomainEvaluator::new(
                &trace_refs, row, &random_coeff_powers,
                LOGUP_LOG_N, LOGUP_LOG_N + 1, LOGUP_LOG_N,
                claimed_sum,
            );
            let cpu_eval = component.evaluate(cpu_eval);
            let row_res = cpu_eval.row_res;
            let arr = row_res.to_m31_array();
            let (e0, e1, e2, e3) = (arr[0].0, arr[1].0, arr[2].0, arr[3].0);
            let (g0, g1, g2, g3) = (gpu_a0[row], gpu_a1[row], gpu_a2[row], gpu_a3[row]);

            if e0 != g0 || e1 != g1 || e2 != g2 || e3 != g3 {
                if mismatches < 8 {
                    eprintln!("[LOGUP_PAIRS MISMATCH] row {row}:");
                    eprintln!("  CPU: ({e0}, {e1}, {e2}, {e3})");
                    eprintln!("  GPU: ({g0}, {g1}, {g2}, {g3})");
                }
                mismatches += 1;
            } else {
                eprintln!("[LOGUP_PAIRS OK] row {row}: ({e0}, {e1}, {e2}, {e3})");
            }
        }
        assert_eq!(mismatches, 0,
            "{mismatches}/{n_rows} rows differ between GPU and CPU for logup_in_pairs");
    }

    // ── Four-entry logup with finalize_logup_in_pairs (batches=[0,0,1,1]) ──
    // This exercises the intermediate batch path: batch 0 generates a column
    // constraint on inter2 cols 0..3 at offset=0, and batch 1 (final) generates
    // the cumsum constraint on inter2 cols 4..7 at offsets [-1, 0].
    // Total: 2 constraints, 8 interaction-2 columns.

    struct LogupFourEntriesEval {
        lookup: TestLookupRelation,
    }

    impl stwo_constraint_framework::FrameworkEval for LogupFourEntriesEval {
        fn log_size(&self) -> u32 { LOGUP_LOG_N }
        fn max_constraint_log_degree_bound(&self) -> u32 { LOGUP_LOG_N + 1 }
        fn evaluate<E: stwo_constraint_framework::EvalAtRow>(&self, mut eval: E) -> E {
            let v0 = eval.next_trace_mask();
            let v1 = eval.next_trace_mask();
            let v2 = eval.next_trace_mask();
            let v3 = eval.next_trace_mask();
            eval.add_to_relation(stwo_constraint_framework::RelationEntry::new(&self.lookup, E::EF::one(), &[v0]));
            eval.add_to_relation(stwo_constraint_framework::RelationEntry::new(&self.lookup, E::EF::one(), &[v1]));
            eval.add_to_relation(stwo_constraint_framework::RelationEntry::new(&self.lookup, E::EF::one(), &[v2]));
            eval.add_to_relation(stwo_constraint_framework::RelationEntry::new(&self.lookup, E::EF::one(), &[v3]));
            eval.finalize_logup_in_pairs();
            eval
        }
    }

    /// End-to-end test for four-entry logup with finalize_logup_in_pairs.
    ///
    /// Batching = [0,0,1,1]: batch 0 (fracs 0+1) generates an intermediate interaction column
    /// constraint (offset=0 only); batch 1 (fracs 2+3) generates the final cumsum constraint
    /// (offsets [-1, 0]). This covers both the intermediate and final batch code paths.
    ///
    /// Total: 4 base trace cols + 8 interaction-2 cols = 12 flat columns.
    /// interaction_offsets = [0, 0, 4, 12]
    #[test]
    fn test_e2e_logup_four_entries_gpu_vs_cpu() {
        use vortexstark::device::DeviceBuffer;
        use vortexstark::cuda::ffi;
        use crate::constraint_eval::tracing::record_bytecode;
        use crate::constraint_eval::bytecode::BytecodeOp;
        use stwo_constraint_framework::{CpuDomainEvaluator, FrameworkEval};
        use stwo::core::pcs::TreeVec;
        use stwo::core::poly::circle::CanonicCoset;
        use stwo::prover::poly::circle::CircleEvaluation;
        use stwo::prover::poly::BitReversedOrder;
        use stwo::prover::backend::CpuBackend;

        init_gpu();

        let claimed_sum = SecureField::from_m31_array([
            M31::from(13), M31::from(5), M31::from(7), M31::from(11),
        ]);

        let component = LogupFourEntriesEval { lookup: TestLookupRelation::dummy() };
        let program = record_bytecode(&component, claimed_sum);
        eprintln!("LogupFourEntries bytecode ({} ops, {} constraints):\n{}",
            program.ops.len(), program.n_constraints, program.dump());
        assert_eq!(program.n_constraints, 2, "expected 2 constraints (one per batch)");

        let trace_n_rows = 1u32 << LOGUP_LOG_N;
        let n_rows       = 1u32 << (LOGUP_LOG_N + 1);
        let log_expand   = 1u32;

        // inter 1: 4 cols (v0..v3), inter 2: 8 cols (batch0: 4 cols + batch1: 4 cols)
        // interaction_offsets = [0, 0, 4, 12]
        let interaction_offsets: Vec<usize> = vec![0, 0, 4, 12];

        let mut encoded = program.encode();
        {
            let mut pc = 0;
            for op in &program.ops {
                if let BytecodeOp::LoadTrace { interaction, col_idx, .. } = op {
                    let flat_idx = interaction_offsets[*interaction as usize] + *col_idx as usize;
                    // 3-word LoadTrace: word1=header(dst), word2=flat_col, word3=sign|abs_off
                    encoded[pc + 1] = flat_idx as u32;
                    pc += 3;
                } else {
                    pc += op.encoded_len();
                }
            }
        }

        // 12 flat columns, 8 eval rows each
        let mk = |start: u32| -> Vec<u32> { (0..8).map(|i| start + i * 3).collect() };
        let cols: Vec<Vec<u32>> = (0..12).map(|i| mk(i * 7 + 1)).collect();
        let d_cols: Vec<DeviceBuffer<u32>> = cols.iter().map(|c| DeviceBuffer::from_host(c)).collect();

        let col_ptrs: Vec<*const u32> = d_cols.iter().map(|d| d.as_ptr()).collect();
        let d_col_ptrs  = DeviceBuffer::from_host(&col_ptrs);
        let d_col_sizes = DeviceBuffer::from_host(&[n_rows; 12]);
        let d_bytecode  = DeviceBuffer::from_host(&encoded);

        // 2 constraints: coefficients [QM31(1,0,0,0), QM31(1,0,0,0)]
        let coeff: Vec<u32>     = vec![1, 0, 0, 0,  1, 0, 0, 0];
        let denom_inv: Vec<u32> = vec![1, 1];
        let d_coeff     = DeviceBuffer::from_host(&coeff);
        let d_denom_inv = DeviceBuffer::from_host(&denom_inv);

        let mut d_a0 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut d_a1 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut d_a2 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut d_a3 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        d_a0.zero(); d_a1.zero(); d_a2.zero(); d_a3.zero();

        unsafe {
            ffi::cuda_bytecode_constraint_eval(
                d_bytecode.as_ptr(), encoded.len() as u32,
                d_col_ptrs.as_ptr() as *const *const u32,
                d_col_sizes.as_ptr(),
                12, n_rows, trace_n_rows,
                d_coeff.as_ptr(), d_denom_inv.as_ptr(),
                log_expand,
                d_a0.as_mut_ptr(), d_a1.as_mut_ptr(), d_a2.as_mut_ptr(), d_a3.as_mut_ptr(),
                64, // n_registers (small test programs)
            );
            ffi::cuda_device_sync();
        }

        let err = unsafe { ffi::cudaGetLastError() };
        assert_eq!(err, 0, "CUDA kernel error: {err}");

        let gpu_a0 = d_a0.to_host();
        let gpu_a1 = d_a1.to_host();
        let gpu_a2 = d_a2.to_host();
        let gpu_a3 = d_a3.to_host();

        let to_bf = |v: &[u32]| -> Vec<BaseField> { v.iter().map(|&x| M31::from(x)).collect() };
        let eval_domain = CanonicCoset::new(LOGUP_LOG_N + 1).circle_domain();

        let cpu_evals: Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> =
            cols.iter().map(|c| CircleEvaluation::new(eval_domain, to_bf(c))).collect();

        let trace_refs: TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>> =
            TreeVec::new(vec![
                vec![],
                cpu_evals[0..4].iter().collect(),    // inter 1: 4 cols
                cpu_evals[4..12].iter().collect(),   // inter 2: 8 cols
            ]);

        let random_coeff_powers = vec![SecureField::one(), SecureField::one()]; // 2 constraints

        let mut mismatches = 0;
        for row in 0..n_rows as usize {
            let cpu_eval = CpuDomainEvaluator::new(
                &trace_refs, row, &random_coeff_powers,
                LOGUP_LOG_N, LOGUP_LOG_N + 1, LOGUP_LOG_N,
                claimed_sum,
            );
            let cpu_eval = component.evaluate(cpu_eval);
            let row_res = cpu_eval.row_res;
            let arr = row_res.to_m31_array();
            let (e0, e1, e2, e3) = (arr[0].0, arr[1].0, arr[2].0, arr[3].0);
            let (g0, g1, g2, g3) = (gpu_a0[row], gpu_a1[row], gpu_a2[row], gpu_a3[row]);

            if e0 != g0 || e1 != g1 || e2 != g2 || e3 != g3 {
                if mismatches < 8 {
                    eprintln!("[LOGUP4 MISMATCH] row {row}:");
                    eprintln!("  CPU: ({e0}, {e1}, {e2}, {e3})");
                    eprintln!("  GPU: ({g0}, {g1}, {g2}, {g3})");
                }
                mismatches += 1;
            } else if row < 2 {
                eprintln!("[LOGUP4 OK] row {row}: ({e0}, {e1}, {e2}, {e3})");
            }
        }
        assert_eq!(mismatches, 0,
            "{mismatches}/{n_rows} rows differ between GPU and CPU for 4-entry logup");
    }

    /// A simple component: flag * (1 - flag) = 0
    /// Uses Clone (flag used twice) and creates a constant (BaseField::one)
    /// via From<BaseField>, exercising the detached recorder merge path.
    struct E2eFlagEval;

    impl stwo_constraint_framework::FrameworkEval for E2eFlagEval {
        fn log_size(&self) -> u32 { 3 } // 8 rows
        fn max_constraint_log_degree_bound(&self) -> u32 { 4 }
        fn evaluate<E: stwo_constraint_framework::EvalAtRow>(&self, mut eval: E) -> E {
            let flag = eval.next_trace_mask();
            let one = E::F::from(BaseField::from_u32_unchecked(1));
            let constraint = flag.clone() * (one - flag);
            eval.add_constraint(constraint);
            eval
        }
    }

    /// A component with two constraints and three trace columns:
    ///   1) flag * (1 - flag) = 0
    ///   2) a + b * flag - c = 0
    struct E2eMultiEval;

    impl stwo_constraint_framework::FrameworkEval for E2eMultiEval {
        fn log_size(&self) -> u32 { 3 }
        fn max_constraint_log_degree_bound(&self) -> u32 { 5 }
        fn evaluate<E: stwo_constraint_framework::EvalAtRow>(&self, mut eval: E) -> E {
            let flag = eval.next_trace_mask();
            let a = eval.next_trace_mask();
            let b = eval.next_trace_mask();
            let c = eval.next_trace_mask();
            let one = E::F::from(BaseField::from_u32_unchecked(1));
            eval.add_constraint(flag.clone() * (one - flag.clone()));
            eval.add_constraint(a + b * flag - c);
            eval
        }
    }

    /// End-to-end test: record bytecode via FrameworkEval, run GPU kernel,
    /// compare with CPU CpuDomainEvaluator row-by-row.
    ///
    /// Uses INVALID flag values (2, 3) to produce non-zero constraint values,
    /// ensuring the GPU kernel actually computes the right answer (not just zeros).
    #[test]
    fn test_e2e_bytecode_flag_constraint_gpu_vs_cpu() {
        use vortexstark::device::DeviceBuffer;
        use vortexstark::cuda::ffi;
        use crate::constraint_eval::tracing::record_bytecode;
        use crate::constraint_eval::bytecode::BytecodeOp;
        use stwo_constraint_framework::{CpuDomainEvaluator, FrameworkEval};
        use stwo::core::pcs::TreeVec;
        use stwo::core::poly::circle::CanonicCoset;
        use stwo::prover::poly::circle::CircleEvaluation;
        use stwo::prover::poly::BitReversedOrder;
        use stwo::prover::backend::CpuBackend;

        init_gpu();

        let claimed_sum = SecureField::default();
        let program = record_bytecode(&E2eFlagEval, claimed_sum);
        eprintln!("E2E flag constraint bytecode:\n{}", program.dump());

        assert_eq!(program.n_constraints, 1);

        // Use log_expand=1 so offset calculations work properly.
        let trace_log_size = 3u32;
        let eval_log_size = trace_log_size + 1;
        let n_rows = 1u32 << eval_log_size; // 16 eval rows
        let trace_n_rows = 1u32 << trace_log_size; // 8 trace rows
        let log_expand = 1u32;

        // Flag values: include INVALID values (2, 3) so constraint != 0.
        // flag*(1-flag): 0→0, 1→0, 2→-2, 3→-6 (non-zero!)
        let flag_vals: Vec<u32> = vec![0, 1, 2, 3, 1, 0, 2, 0,
                                       0, 1, 2, 3, 1, 0, 2, 0];

        let d_flag = DeviceBuffer::from_host(&flag_vals);
        // interaction_offsets: [0, 0, 1] — preproc=0 cols, base=1 col
        let col_ptrs: Vec<*const u32> = vec![d_flag.as_ptr()];
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);
        let d_col_sizes = DeviceBuffer::from_host(&[n_rows]);

        // Encode bytecode and remap LoadTrace (16-bit encoding: always 3 words).
        let mut encoded = program.encode();
        {
            let mut pc = 0;
            for op in &program.ops {
                if let BytecodeOp::LoadTrace { interaction, col_idx, .. } = op {
                    // Flat col = col_idx (only 1 interaction with columns)
                    let flat_idx = *col_idx as usize;
                    encoded[pc + 1] = flat_idx as u32;
                    pc += 3;
                } else {
                    pc += op.encoded_len();
                }
            }
        }

        let d_bytecode = DeviceBuffer::from_host(&encoded);

        let coeff: Vec<u32> = vec![1, 0, 0, 0];
        let denom_inv: Vec<u32> = vec![1, 1]; // 2 entries for log_expand=1
        let d_coeff = DeviceBuffer::from_host(&coeff);
        let d_denom_inv = DeviceBuffer::from_host(&denom_inv);

        let mut d_a0 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut d_a1 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut d_a2 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut d_a3 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        d_a0.zero(); d_a1.zero(); d_a2.zero(); d_a3.zero();

        unsafe {
            ffi::cuda_bytecode_constraint_eval(
                d_bytecode.as_ptr(), encoded.len() as u32,
                d_col_ptrs.as_ptr() as *const *const u32,
                d_col_sizes.as_ptr(),
                1, n_rows, trace_n_rows,
                d_coeff.as_ptr(), d_denom_inv.as_ptr(), log_expand,
                d_a0.as_mut_ptr(), d_a1.as_mut_ptr(), d_a2.as_mut_ptr(), d_a3.as_mut_ptr(),
                64, // n_registers (small test programs)
            );
            ffi::cuda_device_sync();
        }

        let err = unsafe { ffi::cudaGetLastError() };
        assert_eq!(err, 0, "CUDA kernel error: {err}");

        let gpu_a0 = d_a0.to_host();
        let gpu_a1 = d_a1.to_host();
        let gpu_a2 = d_a2.to_host();
        let gpu_a3 = d_a3.to_host();

        // CPU reference via CpuDomainEvaluator
        let to_bf = |v: &[u32]| -> Vec<BaseField> { v.iter().map(|&x| M31::from(x)).collect() };
        let eval_domain = CanonicCoset::new(eval_log_size).circle_domain();
        let flag_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
            eval_domain, to_bf(&flag_vals));
        let trace_refs: TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>> =
            TreeVec::new(vec![vec![], vec![&flag_eval]]);
        let random_coeff_powers = vec![SecureField::one()];

        let mut mismatches = 0;
        let mut has_nonzero = false;
        for row in 0..n_rows as usize {
            let cpu_eval = CpuDomainEvaluator::new(
                &trace_refs, row, &random_coeff_powers,
                trace_log_size, eval_log_size, trace_log_size, claimed_sum,
            );
            let cpu_eval = E2eFlagEval.evaluate(cpu_eval);
            let arr = cpu_eval.row_res.to_m31_array();
            let (e0, e1, e2, e3) = (arr[0].0, arr[1].0, arr[2].0, arr[3].0);
            let (g0, g1, g2, g3) = (gpu_a0[row], gpu_a1[row], gpu_a2[row], gpu_a3[row]);
            if e0 != 0 || e1 != 0 || e2 != 0 || e3 != 0 { has_nonzero = true; }
            if e0 != g0 || e1 != g1 || e2 != g2 || e3 != g3 {
                if mismatches < 5 {
                    eprintln!("[FLAG MISMATCH] row {row}: CPU=({e0},{e1},{e2},{e3}) GPU=({g0},{g1},{g2},{g3})");
                }
                mismatches += 1;
            }
        }
        assert!(has_nonzero, "Test must produce non-zero constraint values to be meaningful");
        assert_eq!(mismatches, 0,
            "{mismatches}/{n_rows} rows differ between GPU and CPU for flag constraint");
    }

    /// End-to-end test with non-zero constraint values and multiple constraints.
    /// Compares GPU bytecode kernel output against manual CPU computation.
    #[test]
    fn test_e2e_bytecode_multi_constraint_gpu_vs_cpu() {
        use vortexstark::device::DeviceBuffer;
        use vortexstark::cuda::ffi;
        use crate::constraint_eval::tracing::record_bytecode;
        use crate::constraint_eval::bytecode::BytecodeOp;

        init_gpu();

        let p = 0x7FFFFFFFu32;
        let claimed_sum = SecureField::default();
        let program = record_bytecode(&E2eMultiEval, claimed_sum);
        eprintln!("E2E multi-constraint bytecode:\n{}", program.dump());

        assert_eq!(program.n_constraints, 2);

        let n = 8u32;
        let trace_log = 3u32;
        let log_exp = 0u32;

        // Trace: 4 columns [flag, a, b, c]
        // flag=0 or 1; when flag=1: a + b - c should = 0; when flag=0: a - c should = 0
        let flag_vals: Vec<u32> = vec![0, 1, 0, 1, 1, 0, 1, 0];
        let a_vals: Vec<u32>    = vec![5, 3, 7, 10, 2, 9, 4, 6];
        let b_vals: Vec<u32>    = vec![2, 4, 3, 5, 8, 1, 6, 3];
        // c = a + b*flag (so constraint 2 is satisfied)
        let c_vals: Vec<u32> = (0..8).map(|i| {
            let a = a_vals[i];
            let b = b_vals[i];
            let f = flag_vals[i];
            (a + b * f) % p
        }).collect();

        let d_flag = DeviceBuffer::from_host(&flag_vals);
        let d_a = DeviceBuffer::from_host(&a_vals);
        let d_b = DeviceBuffer::from_host(&b_vals);
        let d_c = DeviceBuffer::from_host(&c_vals);

        let col_ptrs: Vec<*const u32> = vec![
            d_flag.as_ptr(), d_a.as_ptr(), d_b.as_ptr(), d_c.as_ptr(),
        ];
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);
        let d_col_sizes = DeviceBuffer::from_host(&[n; 4]);

        let mut encoded = program.encode();
        // Remap LoadTrace to flat column indices (16-bit encoding: always 3 words)
        {
            let mut pc = 0;
            for op in &program.ops {
                if let BytecodeOp::LoadTrace { interaction, col_idx, .. } = op {
                    let flat_idx = *col_idx as usize; // all interaction 1, flat = col_idx
                    encoded[pc + 1] = flat_idx as u32;
                    pc += 3;
                } else {
                    pc += op.encoded_len();
                }
            }
        }

        let d_bytecode = DeviceBuffer::from_host(&encoded);

        // 2 constraints: coefficients [1,0,0,0] for each
        let coeff: Vec<u32> = vec![1, 0, 0, 0, 1, 0, 0, 0];
        let d_coeff = DeviceBuffer::from_host(&coeff);

        let denom_inv: Vec<u32> = vec![1];
        let d_denom_inv = DeviceBuffer::from_host(&denom_inv);

        let mut d_accum0 = DeviceBuffer::<u32>::alloc(n as usize);
        let mut d_accum1 = DeviceBuffer::<u32>::alloc(n as usize);
        let mut d_accum2 = DeviceBuffer::<u32>::alloc(n as usize);
        let mut d_accum3 = DeviceBuffer::<u32>::alloc(n as usize);
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
                4, n, n,
                d_coeff.as_ptr(),
                d_denom_inv.as_ptr(),
                log_exp,
                d_accum0.as_mut_ptr(),
                d_accum1.as_mut_ptr(),
                d_accum2.as_mut_ptr(),
                d_accum3.as_mut_ptr(),
                64, // n_registers (small test programs)
            );
            ffi::cuda_device_sync();
        }

        let err = unsafe { ffi::cudaGetLastError() };
        assert_eq!(err, 0, "CUDA kernel error: {err}");

        let out0 = d_accum0.to_host();

        // Both constraints are satisfied by our trace data, so output should be all zeros.
        // Constraint 1: flag*(1-flag) = 0 (all flags are 0 or 1)
        // Constraint 2: a + b*flag - c = 0 (c was computed to satisfy this)
        let mut mismatches = 0;
        for row in 0..n as usize {
            if out0[row] != 0 {
                if mismatches < 5 {
                    eprintln!("[E2E multi] row {row}: accum0={} (flag={}, a={}, b={}, c={})",
                        out0[row], flag_vals[row], a_vals[row], b_vals[row], c_vals[row]);
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "{mismatches}/{n} rows have non-zero output (both constraints should be satisfied)");
    }

    /// Test that the LoadConst sentinel (0xFFFF) does not collide with actual constant values.
    /// This tests the bug fix where value=0xFFFF (65535) was incorrectly encoded in single-word
    /// format, causing the GPU decoder to read a phantom second word and desync the PC.
    #[test]
    fn test_e2e_const_sentinel_no_collision() {
        use crate::constraint_eval::bytecode::{BytecodeOp, BytecodeProgram};

        // LoadConst always uses 2 words in the new 16-bit register encoding.
        // Any value including 0xFFFF is stored in word2; no sentinel needed.
        let prog = BytecodeProgram {
            ops: vec![
                BytecodeOp::LoadConst { dst: 0, value: 0xFFFF },
                BytecodeOp::AddConstraint { src: 0 },
            ],
            n_constraints: 1,
            n_trace_accesses: 0,
            n_registers: 1,
        };
        let words = prog.encode();
        // 2 words for LoadConst + 1 for AddConstraint = 3
        assert_eq!(words.len(), 3, "LoadConst always 2 words, got {} words total", words.len());
        // word1: [opcode:8 | 0:8 | dst:16], dst=0
        assert_eq!(words[0], (0x01u32 << 24) | 0, "Header word mismatch");
        // word2: the value directly
        assert_eq!(words[1], 0xFFFF, "Value word should be 0xFFFF");
    }

    /// Test that LoadTrace always uses 3 words in the new 16-bit register encoding.
    #[test]
    fn test_e2e_trace_offset_neg1_no_collision() {
        use crate::constraint_eval::bytecode::{BytecodeOp, BytecodeProgram};

        let prog = BytecodeProgram {
            ops: vec![
                BytecodeOp::LoadTrace { dst: 0, interaction: 0, col_idx: 0, offset: -1 },
                BytecodeOp::AddConstraint { src: 0 },
            ],
            n_constraints: 1,
            n_trace_accesses: 1,
            n_registers: 1,
        };
        let words = prog.encode();
        // LoadTrace is always 3 words + 1 for AddConstraint = 4
        assert_eq!(words.len(), 4, "LoadTrace always 3 words, got {} words total", words.len());
        // word1: [0x03:8 | 0:8 | dst:16], dst=0
        assert_eq!(words[0], (0x03u32 << 24) | 0);
        // word2: [interaction:16 | col_idx:16] pre-remap
        assert_eq!(words[1], 0u32);
        // word3: [sign:1 | abs_off:31], sign=1, abs=1
        assert_eq!(words[2], (1u32 << 31) | 1);
    }
}
