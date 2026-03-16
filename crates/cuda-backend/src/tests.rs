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

        // Re-enable GPU NTT for this test
        let gpu_twiddles = CudaBackend::precompute_twiddles(coset);

        // GPU evaluate: extend + NTT
        let extended = CudaBackend::extend(&gpu_poly, domain.log_size());
        let mut gpu_vals_buf = extended.coeffs;
        let cache = crate::poly_ops::twiddle_cache_for_coset(&domain.half_coset);
        vortexstark::ntt::evaluate(&mut gpu_vals_buf.buf, &cache);
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

        // GPU interpolate
        let gpu_col: CudaColumn<BaseField> = vals.iter().copied().collect();
        let gpu_eval = stwo::prover::poly::circle::CircleEvaluation::<
            CudaBackend, BaseField, stwo::prover::poly::BitReversedOrder
        >::new(domain, gpu_col);
        let cache = crate::poly_ops::twiddle_cache_for_coset(&coset);
        let mut gpu_buf = gpu_eval.values;
        vortexstark::ntt::interpolate(&mut gpu_buf.buf, &cache);
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
}
