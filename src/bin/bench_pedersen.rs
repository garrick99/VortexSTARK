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

fn main() {
    println!("Pedersen Hash Benchmark: CPU vs GPU");
    println!("====================================\n");

    ffi::init_memory_pool();
    pedersen::gpu_init(); // upload constant points before test
    test_gpu_fp252();

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

    for n in [10, 100, 1000] {
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
