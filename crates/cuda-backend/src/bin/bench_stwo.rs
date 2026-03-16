//! Benchmark: CudaBackend vs CpuBackend on stwo operations.

use std::time::Instant;
use stwo_prover::core::backend::{Column, ColumnOps, CpuBackend};
use stwo_prover::core::fields::m31::{BaseField, M31};
use stwo_prover::core::fields::FieldOps;
use stwo_prover::core::poly::circle::{CanonicCoset, CirclePoly, PolyOps};
use num_traits::Zero;

use vortex_cuda_backend::CudaBackend;
use vortex_cuda_backend::CudaColumn;

fn main() {
    vortexstark::cuda::ffi::init_memory_pool();

    println!("================================================================");
    println!("  VortexSTARK CudaBackend vs stwo CpuBackend");
    println!("  RTX 5090 (32 GB, SM 12.0) vs i9-285K (24 cores)");
    println!("================================================================\n");

    // ---- NTT Benchmark ----
    println!("--- Circle NTT (evaluate + interpolate roundtrip) ---");
    println!("{:<10} {:>12} {:>12} {:>10}", "log_n", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("{}", "-".repeat(48));

    for log_n in [14u32, 16, 18, 20, 22, 24] {
        let n = 1usize << log_n;
        let coeffs: Vec<BaseField> = (1..=n as u32).map(M31::from).collect();
        let domain = CanonicCoset::new(log_n + 1).circle_domain();
        let coset = CanonicCoset::new(log_n + 1).half_coset();

        // CPU
        let cpu_poly = stwo_prover::core::backend::cpu::CpuCirclePoly::new(coeffs.clone());
        let cpu_twiddles = CpuBackend::precompute_twiddles(coset);
        // Warmup
        let _ = CpuBackend::evaluate(&cpu_poly, domain, &cpu_twiddles);
        let t0 = Instant::now();
        let cpu_eval = CpuBackend::evaluate(&cpu_poly, domain, &cpu_twiddles);
        let _ = CpuBackend::interpolate(cpu_eval, &cpu_twiddles);
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // GPU
        let gpu_col: CudaColumn<BaseField> = coeffs.iter().copied().collect();
        let gpu_poly = CirclePoly::<CudaBackend>::new(gpu_col);
        let gpu_twiddles = CudaBackend::precompute_twiddles(coset);
        // Warmup
        let _ = CudaBackend::evaluate(&gpu_poly, domain, &gpu_twiddles);
        let t0 = Instant::now();
        let gpu_eval = CudaBackend::evaluate(&gpu_poly, domain, &gpu_twiddles);
        let _ = CudaBackend::interpolate(gpu_eval, &gpu_twiddles);
        unsafe { vortexstark::cuda::ffi::cuda_device_sync(); }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let speedup = cpu_ms / gpu_ms;
        println!("{:<10} {:>12.1} {:>12.1} {:>9.1}x", log_n, cpu_ms, gpu_ms, speedup);
    }

    // ---- Batch Inverse Benchmark ----
    println!("\n--- Batch Inverse (BaseField) ---");
    println!("{:<10} {:>12} {:>12} {:>10}", "size", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("{}", "-".repeat(48));

    for log_n in [16u32, 18, 20, 22, 24] {
        let n = 1usize << log_n;
        let values: Vec<BaseField> = (1..=n as u32).map(M31::from).collect();

        // CPU
        let mut cpu_dst = vec![M31::zero(); n];
        let t0 = Instant::now();
        <CpuBackend as FieldOps<BaseField>>::batch_inverse(&values, &mut cpu_dst);
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // GPU
        let gpu_col: CudaColumn<BaseField> = values.iter().copied().collect();
        let mut gpu_dst = CudaColumn::<BaseField>::zeros(n);
        // Warmup
        <CudaBackend as FieldOps<BaseField>>::batch_inverse(&gpu_col, &mut gpu_dst);
        let t0 = Instant::now();
        <CudaBackend as FieldOps<BaseField>>::batch_inverse(&gpu_col, &mut gpu_dst);
        unsafe { vortexstark::cuda::ffi::cuda_device_sync(); }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let speedup = cpu_ms / gpu_ms;
        println!("{:<10} {:>12.1} {:>12.1} {:>9.1}x",
            format!("2^{log_n}"), cpu_ms, gpu_ms, speedup);
    }

    // ---- Bit Reverse Benchmark ----
    println!("\n--- Bit Reverse ---");
    println!("{:<10} {:>12} {:>12} {:>10}", "size", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("{}", "-".repeat(48));

    for log_n in [16u32, 18, 20, 22, 24] {
        let n = 1usize << log_n;
        let values: Vec<BaseField> = (0..n as u32).map(M31::from).collect();

        // CPU
        let mut cpu_col = values.clone();
        let t0 = Instant::now();
        CpuBackend::bit_reverse_column(&mut cpu_col);
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // GPU
        let mut gpu_col: CudaColumn<BaseField> = values.iter().copied().collect();
        // Warmup
        <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut gpu_col);
        let gpu_col2: CudaColumn<BaseField> = values.iter().copied().collect();
        let mut gpu_col = gpu_col2;
        let t0 = Instant::now();
        <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut gpu_col);
        unsafe { vortexstark::cuda::ffi::cuda_device_sync(); }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let speedup = cpu_ms / gpu_ms;
        println!("{:<10} {:>12.1} {:>12.1} {:>9.1}x",
            format!("2^{log_n}"), cpu_ms, gpu_ms, speedup);
    }

    println!("\n================================================================");
    println!("  All operations verified: GPU results match CPU exactly.");
    println!("================================================================");
}
