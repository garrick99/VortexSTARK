use kraken_stark::device::DeviceBuffer;
use kraken_stark::cuda::ffi;
use std::time::Instant;

fn main() {
    println!("kraken-stark GPU smoke test");
    println!("===========================");

    // Test 1: DeviceBuffer roundtrip
    let n = 1 << 20; // 1M elements
    let host_data: Vec<u32> = (0..n).map(|i| i % 0x7FFF_FFFF).collect();

    let t0 = Instant::now();
    let d_buf = DeviceBuffer::from_host(&host_data);
    let upload_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    let result = d_buf.to_host();
    let download_ms = t0.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(host_data, result);
    println!(
        "[OK] Buffer roundtrip: {n} u32s, upload {upload_ms:.2}ms, download {download_ms:.2}ms"
    );

    // Test 2: GPU M31 add
    let a: Vec<u32> = (0..n).map(|i| i % 0x7FFF_FFFF).collect();
    let b: Vec<u32> = (0..n).map(|i| (i * 3) % 0x7FFF_FFFF).collect();
    let d_a = DeviceBuffer::from_host(&a);
    let d_b = DeviceBuffer::from_host(&b);
    let mut d_out = DeviceBuffer::<u32>::alloc(n as usize);

    let t0 = Instant::now();
    unsafe {
        ffi::cuda_m31_add(d_a.as_ptr(), d_b.as_ptr(), d_out.as_mut_ptr(), n);
        ffi::cuda_device_sync();
    }
    let add_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let out = d_out.to_host();
    // Verify a few
    let p = 0x7FFF_FFFFu64;
    for i in [0usize, 1, n as usize / 2, n as usize - 1] {
        let expected = ((a[i] as u64 + b[i] as u64) % p) as u32;
        assert_eq!(out[i], expected, "M31 add mismatch at {i}");
    }
    println!("[OK] M31 add: {n} elements in {add_ms:.2}ms");

    // Test 3: GPU M31 mul
    let t0 = Instant::now();
    unsafe {
        ffi::cuda_m31_mul(d_a.as_ptr(), d_b.as_ptr(), d_out.as_mut_ptr(), n);
        ffi::cuda_device_sync();
    }
    let mul_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let out = d_out.to_host();
    for i in [0usize, 1, 100, n as usize - 1] {
        // M31 reduce: (x & P) + (x >> 31), then subtract P if needed
        let x = a[i] as u64 * b[i] as u64;
        let lo = (x & 0x7FFF_FFFF) as u32;
        let hi = (x >> 31) as u32;
        let r = lo + hi;
        let expected = if r >= 0x7FFF_FFFF { r - 0x7FFF_FFFF } else { r };
        assert_eq!(out[i], expected, "M31 mul mismatch at {i}");
    }
    println!("[OK] M31 mul: {n} elements in {mul_ms:.2}ms");

    println!("\nAll tests passed!");
}
