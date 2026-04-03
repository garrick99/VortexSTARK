// GPU Fp252 arithmetic verification kernel.
// Tests basic operations against known CPU-computed values.

#include "include/fp252.cuh"

// Note: PEDERSEN_PX/PY are in pedersen_gpu.cu's constant memory.
// Can't access from here. EC tests moved to pedersen_gpu.cu.

__global__ void fp252_test_kernel(uint64_t* results) {
    // Test 1: 7 * 6 = 42
    Fp252 a = {{7, 0, 0, 0}};
    Fp252 b = {{6, 0, 0, 0}};
    Fp252 c = fp_mul(a, b);
    results[0] = c.v[0]; results[1] = c.v[1]; results[2] = c.v[2]; results[3] = c.v[3];

    // Test 1b: verify schoolbook directly: 7*6 raw product
    uint64_t raw[8];
    schoolbook_4x4(a, b, raw);
    // raw should be [42, 0, 0, 0, 0, 0, 0, 0]

    // Test 2: (p-1)^2 = 1
    Fp252 pm1 = {{0, 0, 0, FP_P3}};
    // Verify schoolbook of (p-1)*(p-1)
    uint64_t raw2[8];
    schoolbook_4x4(pm1, pm1, raw2);
    // Store raw product in results[20..27] for debugging
    for (int i = 0; i < 8; i++) results[20+i] = raw2[i];

    Fp252 sq = fp_mul(pm1, pm1);
    results[4] = sq.v[0]; results[5] = sq.v[1]; results[6] = sq.v[2]; results[7] = sq.v[3];

    // Test 3: (p-1) * (p-2) = 2
    Fp252 pm2 = {{0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x0800000000000010ULL}};
    Fp252 prod = fp_mul(pm1, pm2);
    results[8] = prod.v[0]; results[9] = prod.v[1]; results[10] = prod.v[2]; results[11] = prod.v[3];

    // Test 3b: basic EC test — double a known point
    // Use P = (2, y) where y² = 8 + 2 + β. Skip — constant memory not available here.
    results[28] = 0xDEAD; results[29] = 0; results[30] = 0; results[31] = 0;

    // Test 4: 2 + (p-1) = 1 (mod p)
    Fp252 two = {{2, 0, 0, 0}};
    Fp252 sum = fp_add(two, pm1);
    results[12] = sum.v[0]; results[13] = sum.v[1]; results[14] = sum.v[2]; results[15] = sum.v[3];

    // Test 5: 1 - 2 = p - 1
    Fp252 one = {{1, 0, 0, 0}};
    Fp252 diff = fp_sub(one, two);
    results[16] = diff.v[0]; results[17] = diff.v[1]; results[18] = diff.v[2]; results[19] = diff.v[3];
}

extern "C" {
void cuda_fp252_test(uint64_t* results) {
    fp252_test_kernel<<<1, 1>>>(results);
}
}
