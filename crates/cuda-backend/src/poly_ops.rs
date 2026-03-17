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

/// Reconstruct a `TwiddleTree<CpuBackend>` from cached CPU twiddle data.
pub(crate) fn cached_cpu_twiddle_tree(coset: &StwoCoset) -> TwiddleTree<CpuBackend> {
    let pair = cached_cpu_twiddles(coset);
    TwiddleTree {
        root_coset: pair.root_coset,
        twiddles: pair.twiddles.iter().map(|&v| BaseField::from_u32_unchecked(v)).collect(),
        itwiddles: pair.itwiddles.iter().map(|&v| BaseField::from_u32_unchecked(v)).collect(),
    }
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
        // CPU fallback — barycentric eval is a weighted dot product, small
        // TODO: GPU dot product kernel
        let cpu_evals = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
            evals.domain, evals.values.to_cpu(),
        );
        let cpu_weights = weights.to_cpu();
        CpuBackend::barycentric_eval_at_point(&cpu_evals, &cpu_weights)
    }

    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        _twiddles: &TwiddleTree<Self>,
    ) -> SecureField {
        // CPU path — the GPU IFFT + fold per-polynomial has too much per-call overhead
        // for the hundreds of small polynomials in OODS evaluation.
        // The real fix is batching all OODS evaluations into one kernel launch.
        // TODO: GPU batch OODS evaluation.
        let cpu_evals = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
            evals.domain, evals.values.to_cpu(),
        );
        let cpu_twiddles = cached_cpu_twiddle_tree(&evals.domain.half_coset);
        CpuBackend::eval_at_point_by_folding(&cpu_evals, point, &cpu_twiddles)
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
