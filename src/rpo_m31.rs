//! RPO-M31 hash function for Circle STARKs over M31.
//!
//! Paper: "RPO-M31 and XHash-M31: Efficient Hash Functions for Circle STARKs"
//! (eprint 2024/1635, Ashur & Tariq). MIT-licensed reference: AbdelStark/rpo-xhash-m31.
//!
//! Parameters:
//! - State width: 24 (capacity 8, rate 16)
//! - Rounds: RPO_ROUNDS = 7, each round = FM half-round + BM half-round
//! - Rows per permutation: ROWS_PER_PERM = 14 (RPO_ROUNDS * 2; CLS is linear, no row)
//! - Forward S-box: x^5  (3 mults: x->x²->x⁴->x⁵)
//! - Inverse S-box: x^(5⁻¹ mod p-1) = x^1717986917 (~45 mults)
//! - MDS: 24×24 circulant, first row = hard-coded 24 M31 values
//!
//! Round structure (per full round):
//!   FM: state = x^5(MDS(state) + RC_FM)    → write row 2r
//!   BM: state = x^(1/5)(MDS(state) + RC_BM)→ write row 2r+1
//! Final CLS: state = MDS(state) + RC_CLS    (no trace row; produces final output)
//!
//! Compared to Poseidon2 (30 rows/perm): 2.14x more permutations per trace,
//! but higher arithmetic intensity (24×24 MDS vs O(n) additions).

use crate::field::M31;
use std::sync::OnceLock;

pub const STATE_WIDTH: usize = 24;
pub const RPO_ROUNDS: usize = 7;
/// Rows written to trace per permutation (FM + BM per round; CLS writes no row).
pub const ROWS_PER_PERM: usize = RPO_ROUNDS * 2; // 14
/// Total round-constant steps (FM × 7 + BM × 7 + CLS × 1).
pub const NUM_STEPS: usize = RPO_ROUNDS * 2 + 1; // 15
/// Inverse quintic exponent: 5⁻¹ mod (p-1) where p = 2^31 - 1.
pub const INV_QUINTIC_EXP: u32 = 1_717_986_917;

// ─────────────────────────────────────────────────────────────────────────────
// MDS matrix — 24×24 circulant built from the 32-element root-of-unity sequence.
// M[row][col] = HARD_FIRST_ROW[(col + 32 - row) % 32], where hard[24..32] = 0.
// Source: Appendix A.3 of eprint 2024/1635.
// ─────────────────────────────────────────────────────────────────────────────

const HARD_FIRST_ROW: [u32; 24] = [
    185870542, 2144994796, 1696461115,  215190769,  930115258,  766567118,
   2003379079, 1770558586, 1779722644,  434368282,  289154277, 1979813463,
   1436360233, 1342944808,   63026005,  903393155, 1512525948,  105409451,
   1072974295,  979558870,  436105640, 2126764826, 1981550821,  636196459,
];

static MDS_MATRIX: OnceLock<[[M31; STATE_WIDTH]; STATE_WIDTH]> = OnceLock::new();

fn get_mds() -> &'static [[M31; STATE_WIDTH]; STATE_WIDTH] {
    MDS_MATRIX.get_or_init(|| {
        let mut m = [[M31::ZERO; STATE_WIDTH]; STATE_WIDTH];
        const N: usize = 32;
        for row in 0..STATE_WIDTH {
            for col in 0..STATE_WIDTH {
                let idx = (col + N - row) % N;
                m[row][col] = if idx < STATE_WIDTH {
                    M31(HARD_FIRST_ROW[idx])
                } else {
                    M31::ZERO
                };
            }
        }
        m
    })
}

/// MDS matrix as flat [STATE_WIDTH * STATE_WIDTH] u32 array for GPU upload.
pub fn mds_flat() -> Vec<u32> {
    let m = get_mds();
    m.iter().flat_map(|row| row.iter().map(|v| v.0)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Round constants — 360 values, deterministically derived via SHAKE-256.
// Tag (exact UTF-8): "RPO\xe2\x80\x91M31:p=2147483647,m=24,c=8,n=7"
// Layout: [FM_0(24), BM_0(24), FM_1(24), BM_1(24), ..., FM_6(24), BM_6(24), CLS(24)]
// Pre-computed from the reference implementation (AbdelStark/rpo-xhash-m31).
// ─────────────────────────────────────────────────────────────────────────────

const RPO_ROUND_CONSTANTS: [u32; NUM_STEPS * STATE_WIDTH] = [
    // FM_0
    1044571043, 1317517494,  249487400, 1712152640,  936790368, 1798470765,
      75987472,  923861688, 1748282650, 1458636302, 2125307428,  627234082,
    2094513831,  392286449, 1205004590, 1646616632,  146852791, 1084936660,
    1213837505, 1508526896, 2088113386,  438801021,  620341898, 1747829738,
    // BM_0
    1684110208,  194902933, 2034327328, 1605185266,  776772368,  877927050,
    1561964863,  962611746,  188953594, 1836996574,  856531682,  749945640,
     751634965,  187573186, 1378231507, 1231694826, 1943461010, 2139818866,
    1719290980, 1471403545,  304901028, 1593911519, 2121163118, 1042419745,
    // FM_1
     430714545, 1802534386, 1411432650, 1975589510,  251312259,  546614365,
     861319596,  433939148, 1152132416,  899019929,  561968644, 1115375421,
    1681855333,  136300369,  384504971,  467924243, 1603505046,  653434233,
    1505420243, 1284859439,  236632659,  976134411,  494176167,  679268153,
    // BM_1
     874463401,  564716317, 1001251281, 1386232188, 1228044904, 1465124466,
     877364151,  898604478, 1544100063,  915337253, 1950629822, 1906906106,
     757979146, 2002835449, 1214662108, 1434697933, 1662108020, 1936844077,
     542769444,  204190061, 1012010535, 1430350110, 1398293548,  886809152,
    // FM_2
     790804361,  367584450, 1165352909,    2233920,  955253877,  621800289,
    1485836247, 1656386665, 1985508482, 1032665723,  722925139, 2020384718,
    1115877501,   20610385,  180841501,  395256351, 1552537471, 2110092113,
    1058543121,   81808142,  949977919,  715499236,  943686204, 1422121034,
    // BM_2
     745980516, 1913889045, 1782769571, 1165159750, 1914385059, 1809338068,
    1177671488,  598847504, 1285095643, 1935029517,  390198635,  116743474,
    1719389913,  154253453, 1239199897,  592659166,  139837822,  953279054,
     918427924, 1262684657,  849026136, 1491875427,  687067407,  286614573,
    // FM_3
    1938265077,  764410813,  966180870,  841434815,  535875336,  292916763,
    1320939155, 1414337386, 1928600283,  293987357, 1001539892, 1425158027,
    1127542804, 1390392454, 1535562128,  469264898,  184249372,  716590694,
    1228813361, 1780157382, 1949536017,   73510583, 1934750976, 2140723487,
    // BM_3
     245425607, 1414863040, 1951927839,  196132533,  416495472, 1407191283,
      42873232,  639964293, 1013217774,  519899172, 1334120767, 1853855471,
     663063513, 1693625049, 1567218755, 2123277830, 1243733868, 1783823199,
     769587054,   48044813,  324092972, 1795938818, 1134458730, 1328939602,
    // FM_4
    1230851342,  267767087,  522693111,   57086192,  107676261,  442564081,
     716638036, 1755526725, 2125501390,  271091103,  671689141,  956120501,
     660403697, 1300430489,   53310533,  535970544, 2068115302, 2133603187,
      62017252, 1539180651, 1160889845,  141875767, 1652116664,  679606033,
    // BM_4
     359733143,  212711718,  588071278, 1766076774, 1262041434, 1727953856,
     567579954, 1441364608,  439540796, 1326545544,   67193305,  534841932,
     737345739,  694226368,  973220856, 2017274086,  750471678,  138369658,
    1904221552,  720786848,  951516589,  921310073,  389800747,  952921749,
    // FM_5
     282292561, 1208095394, 1247331148, 1245102222,  859543152, 1462120451,
     799775307,  968355629, 1612145162, 1749315342,  293380594,  145527966,
    1419068342, 2031957997,  888083721, 1382614377,  916920974, 1296916153,
    2011230971,  697722432,  386062754, 1900046699, 1585545260,  708361486,
    // BM_5
     178122392, 1311864996, 1352288535, 1283109748, 1379051687, 2090838053,
     628779636,  971729603, 1004705254, 2047415761,  714775169, 1188845145,
    1401204922,  674825691,  416212167, 1843107919,  596301862, 1915939922,
    1618162956, 1450092627,  107386172,  506445401, 1921830619, 2086845529,
    // FM_6
     724994202,  198168589,  169256380,  920601077,  342498620, 1886429554,
    1189428740,  392408933, 1592337489, 1746180144, 1460751591,  444800367,
    1210701826,  638380860,  295249870, 1546224051, 2066701216,   61687557,
    2032029179,  464348718,  445515901, 1448021165,  312825948, 2073211063,
    // BM_6
     972419809,  167003533,   73895973, 1814963636,  848867698,  360586058,
    1902061290, 1181548648, 1091022720, 1505503585, 2073616608, 1076682089,
    1964736989,  768226089,  269778254, 2045559121, 2025368936, 1061136486,
     597010591, 1642802834, 1214396177,  989902115,  155452933,   12002769,
    // CLS
      95012232, 1536513128,  159034207,  856068314, 1739694228, 1349483389,
    1055805398, 1000132196, 1164649946, 1162020698, 2054206931, 1025425448,
    1748283578, 1657857317,  303425122,  688113491, 1562492500, 1148182238,
    1650552995, 1591611216, 1676998327, 1600694694,  549083144,  376686657,
];

/// Round constants as a 2D view: [NUM_STEPS][STATE_WIDTH].
pub fn round_constants() -> &'static [[u32; STATE_WIDTH]] {
    // Safety: the flat array has exactly NUM_STEPS * STATE_WIDTH elements.
    unsafe {
        std::slice::from_raw_parts(
            RPO_ROUND_CONSTANTS.as_ptr() as *const [u32; STATE_WIDTH],
            NUM_STEPS,
        )
    }
}

/// Flat round constants for GPU upload (already flat, just return a reference).
pub fn round_constants_flat() -> &'static [u32] {
    &RPO_ROUND_CONSTANTS
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU permutation
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the 24×24 MDS matrix in-place.
#[inline]
fn mds_apply(state: &mut [M31; STATE_WIDTH]) {
    let mds = get_mds();
    let mut tmp = [M31::ZERO; STATE_WIDTH];
    for i in 0..STATE_WIDTH {
        tmp[i] = mds[i].iter().zip(state.iter()).fold(M31::ZERO, |acc, (&m, &s)| acc + m * s);
    }
    *state = tmp;
}

/// Forward S-box: x → x^5.
#[inline(always)]
pub fn sbox_fwd(x: M31) -> M31 {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x
}

/// Inverse S-box: x → x^(5⁻¹ mod p-1) = x^1717986917.
#[inline(always)]
pub fn sbox_inv(x: M31) -> M31 {
    let mut result = M31::ONE;
    let mut base = x;
    let mut exp = INV_QUINTIC_EXP;
    while exp > 0 {
        if exp & 1 != 0 {
            result = result * base;
        }
        base = base * base;
        exp >>= 1;
    }
    result
}

/// RPO-M31 permutation: 7 × (FM + BM) rounds, then CLS.
pub fn rpo_permutation(input: &[M31; STATE_WIDTH]) -> [M31; STATE_WIDTH] {
    let rc = round_constants();
    let mut state = *input;
    let mut idx = 0usize;

    for _ in 0..RPO_ROUNDS {
        // FM step: MDS → add RC → x^5
        mds_apply(&mut state);
        for j in 0..STATE_WIDTH { state[j] = state[j] + M31(rc[idx][j]); }
        for j in 0..STATE_WIDTH { state[j] = sbox_fwd(state[j]); }
        idx += 1;

        // BM step: MDS → add RC → x^(1/5)
        mds_apply(&mut state);
        for j in 0..STATE_WIDTH { state[j] = state[j] + M31(rc[idx][j]); }
        for j in 0..STATE_WIDTH { state[j] = sbox_inv(state[j]); }
        idx += 1;
    }

    // CLS: MDS → add RC (no S-box)
    mds_apply(&mut state);
    for j in 0..STATE_WIDTH { state[j] = state[j] + M31(rc[idx][j]); }

    state
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU trace generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate RPO-M31 trace on GPU.
/// Trace has n = 2^log_n rows, split into n / ROWS_PER_PERM permutation blocks.
/// Returns STATE_WIDTH device columns, each of length n.
pub fn generate_trace_gpu(log_n: u32) -> Vec<crate::device::DeviceBuffer<u32>> {
    use crate::cuda::ffi;
    use crate::device::DeviceBuffer;

    let n_rows = 1usize << log_n;
    assert!(n_rows >= ROWS_PER_PERM, "trace too small for one RPO-M31 permutation");
    let n_blocks = n_rows / ROWS_PER_PERM;

    // Upload MDS (576 u32) and round constants (360 u32) to GPU constant memory.
    let mds = mds_flat();
    let rc = round_constants_flat();
    unsafe {
        ffi::cuda_rpo_upload_constants(mds.as_ptr(), rc.as_ptr());
    }

    // Build block input array: each block gets a deterministic 24-element state.
    let mut block_inputs: Vec<u32> = Vec::with_capacity(n_blocks * STATE_WIDTH);
    for block in 0..n_blocks {
        for j in 0..STATE_WIDTH {
            let val = ((block * STATE_WIDTH + j + 1) as u64) % (crate::field::m31::P as u64);
            block_inputs.push(val as u32);
        }
    }

    let d_inputs = DeviceBuffer::from_host(&block_inputs);
    let mut d_cols: Vec<DeviceBuffer<u32>> = (0..STATE_WIDTH)
        .map(|_| DeviceBuffer::<u32>::alloc(n_rows))
        .collect();
    let col_ptrs: Vec<*mut u32> = d_cols.iter_mut().map(|c| c.as_mut_ptr()).collect();
    let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

    unsafe {
        ffi::cuda_rpo_trace(
            d_inputs.as_ptr(),
            d_col_ptrs.as_ptr() as *const *mut u32,
            n_blocks as u32,
        );
        ffi::cuda_device_sync();
    }

    d_cols
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpo_permutation_deterministic() {
        let input: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        let out1 = rpo_permutation(&input);
        let out2 = rpo_permutation(&input);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_rpo_permutation_nontrivial() {
        let input: [M31; STATE_WIDTH] = std::array::from_fn(|j| M31((j + 1) as u32));
        let output = rpo_permutation(&input);
        assert_ne!(input[..], output[..]);
        for j in 0..STATE_WIDTH {
            assert_ne!(output[j], M31::ZERO, "output[{j}] is zero (suspicious)");
        }
    }

    #[test]
    fn test_sbox_roundtrip() {
        for v in [1u32, 42, 12345, 2_000_000_000, crate::field::m31::P - 1] {
            let x = M31(v);
            assert_eq!(sbox_inv(sbox_fwd(x)), x, "sbox_inv(sbox_fwd(x)) != x for x={v}");
            assert_eq!(sbox_fwd(sbox_inv(x)), x, "sbox_fwd(sbox_inv(x)) != x for x={v}");
        }
    }

    #[test]
    fn test_mds_deterministic() {
        let mds1 = get_mds();
        let mds2 = get_mds();
        assert!(std::ptr::eq(mds1, mds2), "MDS OnceLock should return same pointer");
    }

    #[test]
    fn test_round_constant_count() {
        assert_eq!(RPO_ROUND_CONSTANTS.len(), NUM_STEPS * STATE_WIDTH);
        assert_eq!(NUM_STEPS, 15);
        assert_eq!(ROWS_PER_PERM, 14);
    }

    #[test]
    #[cfg(not(feature = "no_gpu"))]
    fn test_rpo_gpu_trace_matches_cpu() {
        // Generate trace on GPU at small log_n, verify first block matches CPU.
        let log_n = 4u32; // 16 rows = 1 permutation block (ROWS_PER_PERM=14, pad to 16)
        // Actually need at least ROWS_PER_PERM rows. Use log_n=4 = 16 rows.
        // n_blocks = 16 / 14 = 1 block.
        let n_rows = 1usize << log_n;
        assert!(n_rows >= ROWS_PER_PERM);
        let n_blocks = n_rows / ROWS_PER_PERM;
        assert_eq!(n_blocks, 1);

        let d_cols = generate_trace_gpu(log_n);

        // Bring GPU columns back to CPU.
        let cols: Vec<Vec<u32>> = d_cols.iter().map(|c| c.to_host()).collect();

        // Compute CPU trace for block 0.
        let mut state: [M31; STATE_WIDTH] = std::array::from_fn(|j| {
            M31(((j + 1) as u32) % crate::field::m31::P)
        });
        let rc = round_constants();
        let mut idx = 0usize;

        for r in 0..RPO_ROUNDS {
            // FM
            mds_apply(&mut state);
            for j in 0..STATE_WIDTH { state[j] = state[j] + M31(rc[idx][j]); }
            for j in 0..STATE_WIDTH { state[j] = sbox_fwd(state[j]); }
            idx += 1;
            let fm_row = 2 * r;
            for j in 0..STATE_WIDTH {
                assert_eq!(cols[j][fm_row], state[j].0,
                    "FM mismatch round={r} col={j}: gpu={} cpu={}", cols[j][fm_row], state[j].0);
            }

            // BM
            mds_apply(&mut state);
            for j in 0..STATE_WIDTH { state[j] = state[j] + M31(rc[idx][j]); }
            for j in 0..STATE_WIDTH { state[j] = sbox_inv(state[j]); }
            idx += 1;
            let bm_row = 2 * r + 1;
            for j in 0..STATE_WIDTH {
                assert_eq!(cols[j][bm_row], state[j].0,
                    "BM mismatch round={r} col={j}: gpu={} cpu={}", cols[j][bm_row], state[j].0);
            }
        }
    }

    #[test]
    fn test_rpo_zero_input() {
        let input = [M31::ZERO; STATE_WIDTH];
        let output = rpo_permutation(&input);
        // With zero input, all-zero should still produce non-zero output (from round constants)
        assert!(output.iter().any(|&x| x != M31::ZERO));
    }
}
