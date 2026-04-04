//! PolyOps: Circle polynomial operations.
//!
//! NTT evaluate/interpolate uses stwo's twiddle format on GPU kernels.
//! A global twiddle cache avoids recomputing and re-uploading twiddles
//! for repeated coset sizes (hundreds of columns share the same cosets).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use num_traits::Zero;
use stwo::core::circle::{CirclePoint, Coset as StwoCoset};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::{CanonicCoset, CircleDomain};
use stwo::prover::backend::{Column, CpuBackend};
use stwo::prover::poly::circle::{
    CircleCoefficients, CircleEvaluation, PolyOps,
};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::poly::BitReversedOrder;

use vortexstark::circle::Coset as VortexCoset;
use vortexstark::device::DeviceBuffer;
use vortexstark::ntt::TwiddleCache;

use super::CudaBackend;
use super::column::CudaColumn;

/// GPU twiddle factors (placeholder — not used for NTT yet).
pub struct CudaTwiddles {
    pub cache: TwiddleCache,
}

// ---------------------------------------------------------------------------
// Global twiddle cache
// ---------------------------------------------------------------------------

/// Cache key: (initial.x, initial.y, step.x, step.y, log_size) as raw u32s.
type CosetKey = (u32, u32, u32, u32, u32);

fn coset_key(coset: &StwoCoset) -> CosetKey {
    (coset.initial.x.0, coset.initial.y.0, coset.step.x.0, coset.step.y.0, coset.log_size)
}

/// Cached GPU twiddle pair: (forward twiddles, inverse twiddles) on device.
struct GpuTwiddlePair {
    twiddles: Arc<DeviceBuffer<u32>>,
    itwiddles: Arc<DeviceBuffer<u32>>,
}

/// Cached CPU twiddle pair: raw u32 vectors (for FRI fallback and other CPU paths).
pub(crate) struct CpuTwiddlePair {
    pub twiddles: Arc<Vec<u32>>,
    pub itwiddles: Arc<Vec<u32>>,
    pub root_coset: StwoCoset,
}

fn gpu_cache() -> &'static Mutex<HashMap<CosetKey, GpuTwiddlePair>> {
    static CACHE: OnceLock<Mutex<HashMap<CosetKey, GpuTwiddlePair>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cpu_cache() -> &'static Mutex<HashMap<CosetKey, CpuTwiddlePair>> {
    static CACHE: OnceLock<Mutex<HashMap<CosetKey, CpuTwiddlePair>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get cached GPU twiddle device buffers for a coset, computing + uploading if needed.
fn cached_gpu_twiddles(coset: &StwoCoset) -> (Arc<DeviceBuffer<u32>>, Arc<DeviceBuffer<u32>>) {
    let key = coset_key(coset);
    {
        let cache = gpu_cache().lock().unwrap();
        if let Some(pair) = cache.get(&key) {
            return (pair.twiddles.clone(), pair.itwiddles.clone());
        }
    }
    // Compute on CPU, upload to GPU
    let cpu_tw = CpuBackend::precompute_twiddles(*coset);
    let d_twiddles = Arc::new(DeviceBuffer::from_host(
        &cpu_tw.twiddles.iter().map(|t| t.0).collect::<Vec<u32>>()
    ));
    let d_itwiddles = Arc::new(DeviceBuffer::from_host(
        &cpu_tw.itwiddles.iter().map(|t| t.0).collect::<Vec<u32>>()
    ));
    let result = (d_twiddles.clone(), d_itwiddles.clone());
    {
        let mut cache = gpu_cache().lock().unwrap();
        cache.entry(key).or_insert(GpuTwiddlePair {
            twiddles: d_twiddles,
            itwiddles: d_itwiddles,
        });
    }
    result
}

/// Get cached CPU twiddle data for a coset (used by FRI fallback).
/// Returns (twiddles, itwiddles) as raw u32 vecs plus the root coset.
pub(crate) fn cached_cpu_twiddles(coset: &StwoCoset) -> CpuTwiddlePair {
    let key = coset_key(coset);
    {
        let cache = cpu_cache().lock().unwrap();
        if let Some(pair) = cache.get(&key) {
            return CpuTwiddlePair {
                twiddles: pair.twiddles.clone(),
                itwiddles: pair.itwiddles.clone(),
                root_coset: pair.root_coset,
            };
        }
    }
    let cpu_tw = CpuBackend::precompute_twiddles(*coset);
    let twiddles = Arc::new(cpu_tw.twiddles.iter().map(|t| t.0).collect::<Vec<u32>>());
    let itwiddles = Arc::new(cpu_tw.itwiddles.iter().map(|t| t.0).collect::<Vec<u32>>());
    let result = CpuTwiddlePair {
        twiddles: twiddles.clone(),
        itwiddles: itwiddles.clone(),
        root_coset: cpu_tw.root_coset,
    };
    {
        let mut cache = cpu_cache().lock().unwrap();
        cache.entry(key).or_insert(CpuTwiddlePair {
            twiddles,
            itwiddles,
            root_coset: cpu_tw.root_coset,
        });
    }
    result
}

/// Fold twiddle cache: batch-inverted domain coordinates for GPU fold kernels.
fn fold_twiddle_cache() -> &'static Mutex<HashMap<(CosetKey, bool), Arc<DeviceBuffer<u32>>>> {
    static CACHE: OnceLock<Mutex<HashMap<(CosetKey, bool), Arc<DeviceBuffer<u32>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get GPU fold twiddles for a VortexSTARK coset (cached).
/// `extract_y=true` for circle fold (y-coords), `false` for line fold (x-coords).
fn cached_fold_twiddles_vc(coset: &VortexCoset, extract_y: bool) -> Arc<DeviceBuffer<u32>> {
    let key = (
        (coset.initial.x.0, coset.initial.y.0, coset.step.x.0, coset.step.y.0, coset.log_size),
        extract_y,
    );
    {
        let cache = fold_twiddle_cache().lock().unwrap();
        if let Some(buf) = cache.get(&key) {
            return buf.clone();
        }
    }
    let buf = Arc::new(vortexstark::fri::compute_fold_twiddles_on_demand(coset, extract_y));
    {
        let mut cache = fold_twiddle_cache().lock().unwrap();
        cache.entry(key).or_insert(buf.clone());
    }
    buf
}

/// Get GPU fold twiddles for a stwo coset (cached), converting via `convert_coset`.
fn cached_fold_twiddles(coset: &StwoCoset, extract_y: bool) -> Arc<DeviceBuffer<u32>> {
    cached_fold_twiddles_vc(&convert_coset(coset), extract_y)
}

/// Reconstruct a `TwiddleTree<CpuBackend>` from cached CPU twiddle data.
pub(crate) fn cached_cpu_twiddle_tree(coset: &StwoCoset) -> TwiddleTree<CpuBackend> {
    let pair = cached_cpu_twiddles(coset);
    TwiddleTree {
        root_coset: pair.root_coset,
        twiddles: pair.twiddles.iter().map(|&v| BaseField::from_u32_unchecked(v)).collect(),
        itwiddles: pair.itwiddles.iter().map(|&v| BaseField::from_u32_unchecked(v)).collect(),
    }
}

// ---------------------------------------------------------------------------
// Fold workspace cache
// ---------------------------------------------------------------------------
//
// eval_at_point_by_folding allocates large ping/pong buffers per call.
// For log_n=28 that's ~5GB of cudaMalloc/cudaFree every call. With 32 FRI
// queries that's 160GB of allocations — the dominant cost.
//
// Solution: cache one workspace per element-count n. The workspace holds
// two ping-pong buffers (each 4×(n/2) u32s) and a permanent zeros buffer
// (n u32s). Ping is re-zeroed via cudaMemset at the start of each call
// (~1ms at 1.79 TB/s). Zeros stays zero permanently.

struct FoldWorkspace {
    /// Accumulation target for circle fold; alternating source for line folds.
    ping: [DeviceBuffer<u32>; 4],
    /// Alternating destination for line folds.
    pong: [DeviceBuffer<u32>; 4],
    /// Zero-filled buffer of length n (imaginary QM31 channels for M31 input).
    zeros: DeviceBuffer<u32>,
}

fn fold_workspace_cache() -> &'static Mutex<HashMap<usize, Arc<Mutex<FoldWorkspace>>>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<Mutex<FoldWorkspace>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_fold_workspace(n: usize) -> Arc<Mutex<FoldWorkspace>> {
    {
        let cache = fold_workspace_cache().lock().unwrap();
        if let Some(ws) = cache.get(&n) {
            return Arc::clone(ws);
        }
    }
    let half_n = n / 2;
    let ws = FoldWorkspace {
        ping:  std::array::from_fn(|_| DeviceBuffer::<u32>::alloc(half_n)),
        pong:  std::array::from_fn(|_| DeviceBuffer::<u32>::alloc(half_n)),
        zeros: { let mut b = DeviceBuffer::<u32>::alloc(n); b.zero(); b },
    };
    let arc = Arc::new(Mutex::new(ws));
    {
        let mut cache = fold_workspace_cache().lock().unwrap();
        cache.entry(n).or_insert(Arc::clone(&arc));
    }
    arc
}

/// Convert stwo Coset → VortexSTARK Coset.
pub fn convert_coset(coset: &StwoCoset) -> VortexCoset {
    VortexCoset {
        initial: vortexstark::circle::CirclePoint {
            x: vortexstark::field::M31(coset.initial.x.0),
            y: vortexstark::field::M31(coset.initial.y.0),
        },
        step: vortexstark::circle::CirclePoint {
            x: vortexstark::field::M31(coset.step.x.0),
            y: vortexstark::field::M31(coset.step.y.0),
        },
        log_size: coset.log_size,
    }
}

/// Build a GPU twiddle cache (used for precompute_twiddles).
pub fn twiddle_cache_for_coset(coset: &StwoCoset) -> TwiddleCache {
    TwiddleCache::new(&convert_coset(coset))
}

impl PolyOps for CudaBackend {
    type Twiddles = CudaTwiddles;

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        _twiddles: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self> {
        let mut values = eval.values;
        let n = values.len() as u32;

        // Cached GPU itwiddles (avoids recomputing for repeated coset sizes)
        let (_d_twiddles, d_itwiddles) = cached_gpu_twiddles(&eval.domain.half_coset);

        // GPU IFFT using stwo twiddle format
        unsafe {
            vortexstark::cuda::ffi::cuda_stwo_ntt_interpolate(
                values.buf.as_mut_ptr(), d_itwiddles.as_ptr(), n,
            );
        }

        CircleCoefficients::new(values)
    }

    fn eval_at_point(poly: &CircleCoefficients<Self>, point: CirclePoint<SecureField>) -> SecureField {
        use stwo::core::fields::m31::M31;

        let n = poly.coeffs.len();
        if n == 0 { return SecureField::zero(); }
        if n == 1 {
            return poly.coeffs.at(0).into();
        }

        // Compute folding factors: [y, x, double_x(x), ...] reversed
        let mut mappings = vec![point.y];
        let mut x = point.x;
        for _ in 1..poly.log_size() {
            mappings.push(x);
            x = CirclePoint::double_x(x);
        }
        mappings.reverse();

        // Upload folding factors to GPU as QM31 (4 u32s each)
        let mut factors_flat: Vec<u32> = Vec::with_capacity(mappings.len() * 4);
        for m in &mappings {
            let arr = m.to_m31_array();
            factors_flat.extend_from_slice(&[arr[0].0, arr[1].0, arr[2].0, arr[3].0]);
        }
        let d_factors = DeviceBuffer::from_host(&factors_flat);

        // Scratch buffers for GPU fold (QM31 = 4 u32s per element)
        let half_n = n / 2;
        let d_scratch1 = DeviceBuffer::<u32>::alloc(half_n * 4);
        let d_scratch2 = DeviceBuffer::<u32>::alloc(half_n * 4);

        // GPU fold: reduces n M31 coefficients to 1 QM31 result
        let mut result = [0u32; 4];
        unsafe {
            vortexstark::cuda::ffi::cuda_eval_at_point(
                poly.coeffs.buf.as_ptr(),
                d_factors.as_ptr(),
                result.as_mut_ptr(),
                n as u32,
                d_scratch1.as_ptr() as *mut u32,
                d_scratch2.as_ptr() as *mut u32,
            );
        }

        SecureField::from_m31_array(std::array::from_fn(|i| M31::from_u32_unchecked(result[i])))
    }

    fn barycentric_weights(coset: CanonicCoset, p: CirclePoint<SecureField>) -> CudaColumn<SecureField> {
        CpuBackend::barycentric_weights(coset, p).into_iter().collect()
    }

    fn barycentric_eval_at_point(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        weights: &CudaColumn<SecureField>,
    ) -> SecureField {
        use vortexstark::cuda::ffi;
        use vortexstark::device::DeviceBuffer;

        let n = evals.values.len() as u32;
        if n == 0 {
            return SecureField::default();
        }

        // Launch at most 256 blocks of 256 threads each.
        // CPU final-reduces the partial sums (at most 256 QM31 values).
        const MAX_BLOCKS: u32 = 256;
        let n_blocks = ((n + 255) / 256).min(MAX_BLOCKS);

        let d_out = DeviceBuffer::<u32>::alloc((n_blocks * 4) as usize);
        unsafe {
            ffi::cuda_barycentric_eval(
                evals.values.device_ptr(),
                weights.device_ptr(),
                n,
                d_out.as_ptr() as *mut u32,
                n_blocks,
            );
        }

        // Download partial sums and reduce on CPU.
        let partial = d_out.to_host();
        let m31_add = |a: u32, b: u32| -> u32 {
            let s = a + b;
            if s >= 0x7FFF_FFFFu32 { s - 0x7FFF_FFFFu32 } else { s }
        };
        let mut result = [0u32; 4];
        for b in 0..n_blocks as usize {
            for j in 0..4 {
                result[j] = m31_add(result[j], partial[b * 4 + j]);
            }
        }
        use stwo::core::fields::m31::M31;
        SecureField::from_m31_array([
            M31::from_u32_unchecked(result[0]),
            M31::from_u32_unchecked(result[1]),
            M31::from_u32_unchecked(result[2]),
            M31::from_u32_unchecked(result[3]),
        ])
    }

    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        _twiddles: &TwiddleTree<Self>,
    ) -> SecureField {
        use vortexstark::cuda::ffi;
        use stwo::core::poly::utils::get_folding_alphas;

        let log_size = evals.domain.log_size();
        let n = evals.values.len();

        // For small polys, CPU is faster (kernel launch overhead dominates).
        if n < (1 << 10) {
            let cpu_evals = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
                evals.domain, evals.values.to_cpu(),
            );
            let cpu_twiddles = cached_cpu_twiddle_tree(&evals.domain.half_coset);
            return CpuBackend::eval_at_point_by_folding(&cpu_evals, point, &cpu_twiddles);
        }

        // Compute folding alphas (log_size values, CPU work O(log_n)).
        // alphas[log_size-1] = point.y  →  used for circle fold first
        // alphas[k] = (double_x applied) point.x  →  line folds
        let mut folding_alphas: Vec<SecureField> = get_folding_alphas(point, log_size as usize);

        let qm31_to_arr = |v: SecureField| -> [u32; 4] {
            let m = v.to_m31_array();
            [m[0].0, m[1].0, m[2].0, m[3].0]
        };

        // ── Workspace ────────────────────────────────────────────────────────
        // Get (or create) cached ping-pong workspace for this n.
        // Eliminates O(log_n) large cudaMalloc calls per invocation.
        let half_n = n / 2;
        let ws_arc = get_fold_workspace(n);
        let mut ws = ws_arc.lock().unwrap();

        // Zero ping: fold_circle_into_line_soa accumulates into dst.
        // cudaMemset at ~1.79 TB/s: ~1ms for log_n=28.
        for ch in 0..4 { ws.ping[ch].zero(); }

        // ── Step 1: fold_circle_into_line ─────────────────────────────────────
        let alpha     = folding_alphas.pop().unwrap();  // point.y
        let alpha_sq  = alpha * alpha;
        let alpha_u32    = qm31_to_arr(alpha);
        let alpha_sq_u32 = qm31_to_arr(alpha_sq);

        let d_circle_tw = cached_fold_twiddles(&evals.domain.half_coset, true);

        unsafe {
            ffi::cuda_fold_circle_into_line_soa(
                ws.ping[0].as_mut_ptr(), ws.ping[1].as_mut_ptr(),
                ws.ping[2].as_mut_ptr(), ws.ping[3].as_mut_ptr(),
                evals.values.buf.as_ptr(),
                ws.zeros.as_ptr(), ws.zeros.as_ptr(), ws.zeros.as_ptr(),
                d_circle_tw.as_ptr(),
                alpha_u32.as_ptr(),
                alpha_sq_u32.as_ptr(),
                half_n as u32,
            );
            ffi::cuda_device_sync();
        }

        // ── Steps 2+: fold_line (log_size-1 iterations, ping-pong) ───────────
        // use_ping: true = current result in ping (src), fold ping→pong next.
        let mut use_ping = true;
        let mut cur_log = log_size - 1;

        while cur_log > 0 {
            let new_n = (1usize << cur_log) / 2;
            let alpha  = folding_alphas.pop().unwrap();
            let alpha_u32 = qm31_to_arr(alpha);

            let lc = vortexstark::circle::Coset::half_odds(cur_log);
            let d_line_tw = cached_fold_twiddles_vc(&lc, false);

            unsafe {
                if use_ping {
                    ffi::cuda_fold_line_soa(
                        ws.ping[0].as_ptr(), ws.ping[1].as_ptr(),
                        ws.ping[2].as_ptr(), ws.ping[3].as_ptr(),
                        d_line_tw.as_ptr(),
                        ws.pong[0].as_mut_ptr(), ws.pong[1].as_mut_ptr(),
                        ws.pong[2].as_mut_ptr(), ws.pong[3].as_mut_ptr(),
                        alpha_u32.as_ptr(),
                        new_n as u32,
                    );
                } else {
                    ffi::cuda_fold_line_soa(
                        ws.pong[0].as_ptr(), ws.pong[1].as_ptr(),
                        ws.pong[2].as_ptr(), ws.pong[3].as_ptr(),
                        d_line_tw.as_ptr(),
                        ws.ping[0].as_mut_ptr(), ws.ping[1].as_mut_ptr(),
                        ws.ping[2].as_mut_ptr(), ws.ping[3].as_mut_ptr(),
                        alpha_u32.as_ptr(),
                        new_n as u32,
                    );
                }
                ffi::cuda_device_sync();
            }

            use_ping = !use_ping;
            cur_log -= 1;
        }

        // Download 1 QM31 result (16 bytes) from the active buffer.
        // After the circle fold use_ping=true; each line fold toggles it.
        // use_ping=true → last write was ping→pong → result in pong (use_ping was false going in).
        // Wait: if use_ping at loop start, we fold ping→pong, then toggle to false.
        // After toggle: use_ping=false → result is in pong.
        // Conversely: if use_ping=true after the loop, last fold was pong→ping → result in ping.
        let result_ch: Vec<u32> = if use_ping {
            ws.ping.iter_mut().map(|b| b.to_host()[0]).collect()
        } else {
            ws.pong.iter_mut().map(|b| b.to_host()[0]).collect()
        };
        let result_raw = SecureField::from_m31_array(
            std::array::from_fn(|i| BaseField::from_u32_unchecked(result_ch[i]))
        );

        // Divide by 2^log_size (matches CPU: result / 2^log_size).
        let two_pow = 1u32 << log_size;
        result_raw / SecureField::from(BaseField::from_u32_unchecked(two_pow))
    }

    fn split_at_mid(
        poly: CircleCoefficients<Self>,
    ) -> (CircleCoefficients<Self>, CircleCoefficients<Self>) {
        let (left, right) = poly.coeffs.split_at_mid();
        (CircleCoefficients::new(left), CircleCoefficients::new(right))
    }

    fn extend(poly: &CircleCoefficients<Self>, log_size: u32) -> CircleCoefficients<Self> {
        assert!(log_size >= poly.log_size());
        let old_len = poly.coeffs.len();
        let new_len = 1usize << log_size;
        if old_len == new_len {
            return CircleCoefficients::new(poly.coeffs.clone());
        }
        let mut new_buf = DeviceBuffer::<u32>::alloc(new_len);
        new_buf.zero();
        unsafe {
            vortexstark::cuda::ffi::cudaMemcpy(
                new_buf.as_mut_ptr() as *mut std::ffi::c_void,
                poly.coeffs.buf.as_ptr() as *const std::ffi::c_void,
                old_len * 4,
                vortexstark::cuda::ffi::MEMCPY_D2D,
            );
        }
        CircleCoefficients::new(CudaColumn::from_device_buffer(new_buf, new_len))
    }

    fn evaluate(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        _twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let mut values = Self::extend(poly, domain.log_size()).coeffs;
        let n = values.len() as u32;

        // Cached GPU twiddles (avoids recomputing for repeated coset sizes)
        let (d_twiddles, _d_itwiddles) = cached_gpu_twiddles(&domain.half_coset);

        // GPU FFT using stwo twiddle format
        unsafe {
            vortexstark::cuda::ffi::cuda_stwo_ntt_evaluate(
                values.buf.as_mut_ptr(), d_twiddles.as_ptr(), n,
            );
        }

        CircleEvaluation::new(domain, values)
    }

    fn evaluate_into(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        _twiddles: &TwiddleTree<Self>,
        _buffer: CudaColumn<BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // CPU fallback (ignores buffer, creates new)
        Self::evaluate(poly, domain, _twiddles)
    }

    fn precompute_twiddles(coset: StwoCoset) -> TwiddleTree<Self> {
        let fwd = twiddle_cache_for_coset(&coset);
        let inv = twiddle_cache_for_coset(&coset);
        TwiddleTree {
            root_coset: coset,
            twiddles: CudaTwiddles { cache: fwd },
            itwiddles: CudaTwiddles { cache: inv },
        }
    }
}
