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
