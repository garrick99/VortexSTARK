/// Pedersen hash benchmark: CPU vs GPU throughput.
use kraken_stark::cairo_air::pedersen;
use kraken_stark::cairo_air::stark252_field::{Fp, pedersen_hash};
use kraken_stark::cuda::ffi;
use std::time::Instant;

fn test_gpu_fp252() {
    use kraken_stark::device::DeviceBuffer;
    println!("--- GPU Fp252 Arithmetic Test ---");

    let mut d_results = DeviceBuffer::<u64>::alloc(32);
    unsafe {
        ffi::cuda_fp252_test(d_results.as_mut_ptr());
        ffi::cuda_device_sync();
    }
    let r = d_results.to_host();

    // Test 1: 7 * 6 = 42
    let t1 = [r[0], r[1], r[2], r[3]];
    let ok1 = t1 == [42, 0, 0, 0];
    println!("  7 * 6 = [{:x}, {:x}, {:x}, {:x}] {}", t1[0], t1[1], t1[2], t1[3],
        if ok1 { "✓" } else { "✗ (expected 42)" });

    // Test 2: (p-1)^2 = 1
    let t2 = [r[4], r[5], r[6], r[7]];
    let ok2 = t2 == [1, 0, 0, 0];
    println!("  (p-1)^2 = [{:x}, {:x}, {:x}, {:x}] {}", t2[0], t2[1], t2[2], t2[3],
        if ok2 { "✓" } else { "✗ (expected 1)" });

    // Test 3: (p-1)*(p-2) = 2
    let t3 = [r[8], r[9], r[10], r[11]];
    let ok3 = t3 == [2, 0, 0, 0];
    println!("  (p-1)*(p-2) = [{:x}, {:x}, {:x}, {:x}] {}", t3[0], t3[1], t3[2], t3[3],
        if ok3 { "✓" } else { "✗ (expected 2)" });

    // Test 4: 2 + (p-1) = 1
    let t4 = [r[12], r[13], r[14], r[15]];
    let ok4 = t4 == [1, 0, 0, 0];
    println!("  2 + (p-1) = [{:x}, {:x}, {:x}, {:x}] {}", t4[0], t4[1], t4[2], t4[3],
        if ok4 { "✓" } else { "✗ (expected 1)" });

    // Test 5: 1 - 2 = p - 1
    let t5 = [r[16], r[17], r[18], r[19]];
    let ok5 = t5 == [0, 0, 0, 0x0800000000000011];
    println!("  1 - 2 = [{:x}, {:x}, {:x}, {:x}] {}", t5[0], t5[1], t5[2], t5[3],
        if ok5 { "✓" } else { "✗ (expected p-1)" });

    // Debug: raw product of (p-1)^2
    println!("  (p-1)^2 raw product: [{:016x}, {:016x}, {:016x}, {:016x}, {:016x}, {:016x}, {:016x}, {:016x}]",
        r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27]);

    // Expected raw product of (p-1)^2:
    // p-1 = [0, 0, 0, 0x0800000000000011]
    // (p-1) * (p-1) = only limb[3] * limb[3]:
    //   0x0800000000000011 * 0x0800000000000011 = ?
    // 0x0800000000000011 = 576460752303423505
    // 576460752303423505^2 = 332,147,340,519,851,081,234,449,814,755,025 (need hex)
    let expected_raw = {
        let v: u128 = 0x0800000000000011u128 * 0x0800000000000011u128;
        let lo = v as u64;
        let hi = (v >> 64) as u64;
        [lo, hi]
    };
    println!("  Expected raw[6..7]: [{:016x}, {:016x}]", expected_raw[0], expected_raw[1]);

    // Check P₀ constant loaded on GPU
    let gpu_p0x = Fp { v: [r[28], r[29], r[30], r[31]] };
    let cpu_points = kraken_stark::cairo_air::stark252_field::pedersen_points();
    let cpu_p0x = match cpu_points[0] {
        kraken_stark::cairo_air::stark252_field::CurvePoint::Affine(x, _) => x,
        _ => Fp::ZERO,
    };
    let ok_ec = gpu_p0x == cpu_p0x;
    println!("  P₀.x loaded: GPU=[{:016x}..] CPU=[{:016x}..] {}",
        gpu_p0x.v[0], cpu_p0x.v[0], if ok_ec { "✓" } else { "✗ NOT LOADED" });

    let all_ok = ok1 && ok2 && ok3 && ok4 && ok5 && ok_ec;
    println!("  ALL: {}\n", if all_ok { "PASS" } else { "FAIL" });
}

fn test_gpu_ec_double() {
    use kraken_stark::device::DeviceBuffer;
    use kraken_stark::cairo_air::stark252_field::{Fp, CurvePoint, pedersen_points};

    println!("--- GPU EC Point Doubling Test ---");

    let points = pedersen_points();
    let (px, py) = match points[0] {
        CurvePoint::Affine(x, y) => (x, y),
        _ => panic!("P₀ is infinity"),
    };

    // GPU doubling
    let d_px = DeviceBuffer::from_host(&px.v);
    let d_py = DeviceBuffer::from_host(&py.v);
    let mut d_ox = DeviceBuffer::<u64>::alloc(4);
    let mut d_oy = DeviceBuffer::<u64>::alloc(4);
    let mut d_oz = DeviceBuffer::<u64>::alloc(4);

    unsafe {
        ffi::cuda_pedersen_test_double(
            d_px.as_ptr(), d_py.as_ptr(),
            d_ox.as_mut_ptr(), d_oy.as_mut_ptr(), d_oz.as_mut_ptr(),
        );
        ffi::cuda_device_sync();
    }

    let gpu_x = d_ox.to_host();
    let gpu_y = d_oy.to_host();
    let gpu_z = d_oz.to_host();

    // Convert GPU projective to affine: x_affine = X / Z²
    let gx = Fp { v: [gpu_x[0], gpu_x[1], gpu_x[2], gpu_x[3]] };
    let gy = Fp { v: [gpu_y[0], gpu_y[1], gpu_y[2], gpu_y[3]] };
    let gz = Fp { v: [gpu_z[0], gpu_z[1], gpu_z[2], gpu_z[3]] };
    let gz2 = gz * gz;
    let gz2_inv = gz2.inverse();
    let gpu_affine_x = gx * gz2_inv;
    let gz3 = gz2 * gz;
    let gz3_inv = gz3.inverse();
    let gpu_affine_y = gy * gz3_inv;

    // CPU doubling
    let cpu_doubled = points[0].add(points[0]);
    let (cpu_x, cpu_y) = match cpu_doubled {
        CurvePoint::Affine(x, y) => (x, y),
        _ => panic!("doubled is infinity"),
    };

    let x_ok = gpu_affine_x == cpu_x;
    let y_ok = gpu_affine_y == cpu_y;

    // gz now holds xx (P₀.x squared) from the debug kernel
    let gpu_xx = gz; // repurposed
    let cpu_xx = px * px; // CPU x^2
    let xx_ok = gpu_xx == cpu_xx;
    println!("  P₀.x² (intermediate):");
    println!("    GPU: [{:016x}, {:016x}, {:016x}, {:016x}] {}",
        gpu_xx.v[0], gpu_xx.v[1], gpu_xx.v[2], gpu_xx.v[3], if xx_ok {"✓"} else {"✗"});
    println!("    CPU: [{:016x}, {:016x}, {:016x}, {:016x}]",
        cpu_xx.v[0], cpu_xx.v[1], cpu_xx.v[2], cpu_xx.v[3]);

    println!("  P₀ doubled:");
    println!("    GPU affine x: [{:016x}, {:016x}, {:016x}, {:016x}] {}",
        gpu_affine_x.v[0], gpu_affine_x.v[1], gpu_affine_x.v[2], gpu_affine_x.v[3],
        if x_ok { "✓" } else { "✗" });
    println!("    CPU affine x: [{:016x}, {:016x}, {:016x}, {:016x}]",
        cpu_x.v[0], cpu_x.v[1], cpu_x.v[2], cpu_x.v[3]);
    println!("    x match: {} | y match: {}", if x_ok {"✓"} else {"✗"}, if y_ok {"✓"} else {"✗"});
    println!();
}

fn main() {
    println!("Pedersen Hash Benchmark: CPU vs GPU");
    println!("====================================\n");

    ffi::init_memory_pool();
    pedersen::gpu_init();
    test_gpu_fp252();
    test_gpu_ec_double();

    // === CPU benchmark ===
    println!("--- CPU (projective coordinates) ---");
    let _ = pedersen_hash(Fp::from_u64(1), Fp::from_u64(2)); // warmup

    let n_cpu = 100;
    let t0 = Instant::now();
    for i in 0..n_cpu {
        let _ = pedersen_hash(Fp::from_u64(i as u64 + 1), Fp::from_u64(i as u64 + 100));
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let cpu_per = cpu_ms / n_cpu as f64;
    println!("{n_cpu} hashes: {cpu_ms:.1}ms ({cpu_per:.2}ms/hash, {:.0} hashes/sec)\n",
        1000.0 / cpu_per);

    // === GPU benchmark ===
    println!("--- GPU (parallel, projective, CUDA) ---");
    pedersen::gpu_init();

    // Warmup
    let warmup_a = vec![Fp::from_u64(1)];
    let warmup_b = vec![Fp::from_u64(2)];
    let _ = pedersen::gpu_hash_batch(&warmup_a, &warmup_b);

    for n in [1000, 10000, 100000, 1000000] {
        let inputs_a: Vec<Fp> = (0..n).map(|i| Fp::from_u64(i as u64 + 1)).collect();
        let inputs_b: Vec<Fp> = (0..n).map(|i| Fp::from_u64(i as u64 + 1000)).collect();

        let t0 = Instant::now();
        let results = pedersen::gpu_hash_batch(&inputs_a, &inputs_b);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let rate = n as f64 / (ms / 1000.0);

        // Verify first result matches CPU
        let cpu_check = pedersen_hash(inputs_a[0], inputs_b[0]);
        let ok = results[0] == cpu_check;

        println!("{n:>6} hashes: {ms:>8.1}ms ({:.0} hashes/sec) {}",
            rate, if ok { "✓" } else { "MISMATCH!" });
    }

    // === Pipeline timing breakdown ===
    println!("\n--- Pipeline Timing Breakdown ---");
    for n in [10000, 100000, 1000000] {
        let inputs_a: Vec<Fp> = (0..n).map(|i| Fp::from_u64(i as u64 + 1)).collect();
        let inputs_b: Vec<Fp> = (0..n).map(|i| Fp::from_u64(i as u64 + 1000)).collect();

        // Warmup
        let _ = pedersen::gpu_hash_batch(&inputs_a, &inputs_b);

        // Timed run
        let (_, t) = pedersen::gpu_hash_batch_timed(&inputs_a, &inputs_b);
        let total_ms = t.total_us / 1000.0;
        let rate = n as f64 / (total_ms / 1000.0);

        println!("\n  Batch: {n} hashes ({rate:.0} hash/sec, {total_ms:.1}ms total)");
        println!("  ┌─────────────────────┬──────────┬────────┐");
        println!("  │ Phase               │ Time     │ % Tot  │");
        println!("  ├─────────────────────┼──────────┼────────┤");
        let phases: [(&str, f64); 7] = [
            ("Flatten inputs",  t.flatten_us),
            ("H2D upload",      t.upload_us),
            ("Alloc output",    t.alloc_us),
            ("GPU kernel+sync", t.kernel_us),
            ("D2H download",    t.download_us),
            ("Repack Fp vecs",  t.repack_us),
            ("CPU batch inv",   t.inverse_us),
        ];
        for (name, us) in &phases {
            let pct = us / t.total_us * 100.0;
            let bar = "█".repeat((pct / 2.5) as usize);
            if *us > 1000.0 {
                println!("  │ {:<19} │ {:>5.1}ms  │ {:>5.1}% │ {bar}", name, us / 1000.0, pct);
            } else {
                println!("  │ {:<19} │ {:>5.0}us  │ {:>5.1}% │ {bar}", name, us, pct);
            }
        }
        println!("  └─────────────────────┴──────────┴────────┘");

        // Overhead = total - (kernel + inverse)
        let compute_us = t.kernel_us + t.inverse_us;
        let overhead_us = t.total_us - compute_us;
        println!("  Compute (kernel+inv): {:.1}ms ({:.1}%)",
            compute_us / 1000.0, compute_us / t.total_us * 100.0);
        println!("  Overhead (rest):      {:.1}ms ({:.1}%)",
            overhead_us / 1000.0, overhead_us / t.total_us * 100.0);
    }
    println!();

    // === Pedersen-as-stage benchmark (fused GPU trace columns) ===
    println!("--- Pedersen-as-Stage (GPU trace columns, no host round-trip) ---");
    for (n, log_n) in [(10000u64, 14u32), (100000, 17), (1000000, 20)] {
        let inputs_a: Vec<Fp> = (0..n as usize).map(|i| Fp::from_u64(i as u64 + 1)).collect();
        let inputs_b: Vec<Fp> = (0..n as usize).map(|i| Fp::from_u64(i as u64 + 1000)).collect();

        // Warmup
        let _ = pedersen::gpu_pedersen_trace(&inputs_a, &inputs_b, log_n);

        let t0 = Instant::now();
        let d_cols = pedersen::gpu_pedersen_trace(&inputs_a, &inputs_b, log_n);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let rate = n as f64 / (ms / 1000.0);
        let trace_rows = 1u64 << log_n;

        println!("  {n:>7} hashes → {trace_rows:>8} trace rows (27 cols): {ms:>6.1}ms ({rate:.0} hash/sec)");
        println!("         GPU columns ready for NTT — zero host transfer");
    }
    println!();

    // Debug: show first hash values
    let a0 = Fp::from_u64(1);
    let b0 = Fp::from_u64(1000);
    let cpu_hash = pedersen_hash(a0, b0);
    let gpu_results = pedersen::gpu_hash_batch(&[a0], &[b0]);
    println!("\nDebug - input a=1, b=1000:");
    println!("  CPU: [{:016x}, {:016x}, {:016x}, {:016x}]",
        cpu_hash.v[0], cpu_hash.v[1], cpu_hash.v[2], cpu_hash.v[3]);
    println!("  GPU: [{:016x}, {:016x}, {:016x}, {:016x}]",
        gpu_results[0].v[0], gpu_results[0].v[1], gpu_results[0].v[2], gpu_results[0].v[3]);
    println!("  Match: {}", cpu_hash == gpu_results[0]);

    println!("\nNote: GPU includes upload + kernel + download time.");
}
