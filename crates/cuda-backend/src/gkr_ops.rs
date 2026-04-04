//! GkrOps: GKR lookup protocol — GPU-accelerated.
//!
//! All kernels are in cuda/gkr.cu.  The only CPU work left is:
//!   - Accumulating the partial sums from sum_as_poly reductions
//!   - Calling correct_sum_as_poly_in_first_variable (O(1) work)

use stwo::prover::backend::{Column, CpuBackend};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::lookups::gkr_prover::{
    correct_sum_as_poly_in_first_variable, GkrMultivariatePolyOracle, GkrOps, Layer,
};
use stwo::prover::lookups::sumcheck::MultivariatePolyOracle;
use stwo::prover::lookups::mle::{Mle, MleOps};
use stwo::prover::lookups::utils::UnivariatePoly;

use vortexstark::cuda::ffi;
use vortexstark::device::DeviceBuffer;

use super::CudaBackend;
use super::column::CudaColumn;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Upload a SecureField value to a small device buffer (4 u32s).
fn upload_qm31(v: SecureField) -> DeviceBuffer<u32> {
    let arr = v.to_m31_array();
    DeviceBuffer::from_host(&[arr[0].0, arr[1].0, arr[2].0, arr[3].0])
}

/// Convert a partial-sums buffer (8 u32s per block) to (eval_at_0, eval_at_2) SecureField.
fn reduce_partial_sums(psums: &[u32]) -> (SecureField, SecureField) {
    let n_blocks = psums.len() / 8;
    let mut acc0 = [0u32; 4];
    let mut acc2 = [0u32; 4];
    for b in 0..n_blocks {
        let base = b * 8;
        for c in 0..4 {
            acc0[c] = m31_add_cpu(acc0[c], psums[base + c]);
            acc2[c] = m31_add_cpu(acc2[c], psums[base + 4 + c]);
        }
    }
    (u32s_to_qm31(acc0), u32s_to_qm31(acc2))
}

fn m31_add_cpu(a: u32, b: u32) -> u32 {
    const P: u32 = 0x7FFF_FFFF;
    let r = a.wrapping_add(b);
    if r >= P { r - P } else { r }
}

fn u32s_to_qm31(w: [u32; 4]) -> SecureField {
    SecureField::from_m31_array(std::array::from_fn(|i| {
        BaseField::from_u32_unchecked(w[i])
    }))
}

// ─── MleOps<BaseField> ───────────────────────────────────────────────────────

impl MleOps<BaseField> for CudaBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        let n = mle.len();
        if n < 2 {
            // Fallback for trivially small MLEs
            let cpu_data = mle.to_cpu();
            let cpu_mle = Mle::new(cpu_data);
            let cpu_result = CpuBackend::fix_first_variable(cpu_mle, assignment);
            let result_data: Vec<SecureField> = cpu_result.to_cpu();
            return Mle::new(result_data.into_iter().collect());
        }

        let half_n = n / 2;
        let d_r = upload_qm31(assignment);
        // Output: half_n SecureField elements = half_n * 4 u32s
        let mut d_out = DeviceBuffer::<u32>::alloc(half_n * 4);

        unsafe {
            ffi::cuda_gkr_fix_first_variable_base(
                mle.buf.as_ptr(),
                d_out.as_mut_ptr(),
                d_r.as_ptr(),
                n as u32,
            );
            ffi::cuda_device_sync();
        }

        Mle::new(CudaColumn::from_device_buffer(d_out, half_n))
    }
}

impl MleOps<SecureField> for CudaBackend {
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        let n = mle.len();
        if n < 2 {
            let cpu_data = mle.to_cpu();
            let cpu_mle = Mle::new(cpu_data);
            let cpu_result = CpuBackend::fix_first_variable(cpu_mle, assignment);
            let result_data: Vec<SecureField> = cpu_result.to_cpu();
            return Mle::new(result_data.into_iter().collect());
        }

        let half_n = n / 2;
        let d_r = upload_qm31(assignment);
        let mut d_out = DeviceBuffer::<u32>::alloc(half_n * 4);

        unsafe {
            ffi::cuda_gkr_fix_first_variable_secure(
                mle.buf.as_ptr(),
                d_out.as_mut_ptr(),
                d_r.as_ptr(),
                n as u32,
            );
            ffi::cuda_device_sync();
        }

        Mle::new(CudaColumn::from_device_buffer(d_out, half_n))
    }
}

// ─── GkrOps ──────────────────────────────────────────────────────────────────

impl GkrOps for CudaBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        let k = y.len();
        let n = 1usize << k;  // output size: 2^k SecureField elements

        if k == 0 {
            // Just [v]
            return Mle::new([v].into_iter().collect());
        }

        // Allocate full output buffer (4n u32s)
        let mut d_buf = DeviceBuffer::<u32>::alloc(n * 4);

        // Init: set buf[0..3] = v
        let d_v = upload_qm31(v);
        unsafe {
            ffi::cuda_gkr_gen_eq_evals_init(d_buf.as_mut_ptr(), d_v.as_ptr());
            ffi::cuda_device_sync();
        }

        // Doubling passes: y.iter().rev() = y[k-1], y[k-2], ..., y[0]
        // Pass i (0-indexed): y_i = y[k-1-i], cur_size = 2^i
        for i in 0..k {
            let y_i = y[k - 1 - i];
            let cur_size = 1u32 << i;
            let d_yi = upload_qm31(y_i);
            unsafe {
                ffi::cuda_gkr_gen_eq_evals_pass(d_buf.as_mut_ptr(), d_yi.as_ptr(), cur_size);
                ffi::cuda_device_sync();
            }
        }

        Mle::new(CudaColumn::from_device_buffer(d_buf, n))
    }

    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        match layer {
            Layer::GrandProduct(mle) => {
                let n = mle.len();
                if n < 2 {
                    return next_layer_cpu(layer);
                }
                let half_n = n / 2;
                let mut d_out = DeviceBuffer::<u32>::alloc(half_n * 4);
                unsafe {
                    ffi::cuda_gkr_next_layer_grand_product(
                        mle.buf.as_ptr(), d_out.as_mut_ptr(), n as u32,
                    );
                    ffi::cuda_device_sync();
                }
                Layer::GrandProduct(Mle::new(CudaColumn::from_device_buffer(d_out, half_n)))
            }

            Layer::LogUpGeneric { numerators, denominators } => {
                let n = denominators.len();
                if n < 2 {
                    return next_layer_cpu(layer);
                }
                let half_n = n / 2;
                let mut d_out_num = DeviceBuffer::<u32>::alloc(half_n * 4);
                let mut d_out_den = DeviceBuffer::<u32>::alloc(half_n * 4);
                unsafe {
                    ffi::cuda_gkr_next_layer_logup_generic(
                        numerators.buf.as_ptr(),
                        denominators.buf.as_ptr(),
                        d_out_num.as_mut_ptr(),
                        d_out_den.as_mut_ptr(),
                        n as u32,
                    );
                    ffi::cuda_device_sync();
                }
                Layer::LogUpGeneric {
                    numerators:   Mle::new(CudaColumn::from_device_buffer(d_out_num, half_n)),
                    denominators: Mle::new(CudaColumn::from_device_buffer(d_out_den, half_n)),
                }
            }

            Layer::LogUpMultiplicities { numerators, denominators } => {
                let n = denominators.len();
                if n < 2 {
                    return next_layer_cpu(layer);
                }
                let half_n = n / 2;
                let mut d_out_num = DeviceBuffer::<u32>::alloc(half_n * 4);
                let mut d_out_den = DeviceBuffer::<u32>::alloc(half_n * 4);
                unsafe {
                    // numerators: M31 (1 u32/element), denominators: QM31 (4 u32/element)
                    ffi::cuda_gkr_next_layer_logup_mult(
                        numerators.buf.as_ptr(),
                        denominators.buf.as_ptr(),
                        d_out_num.as_mut_ptr(),
                        d_out_den.as_mut_ptr(),
                        n as u32,
                    );
                    ffi::cuda_device_sync();
                }
                // Output is LogUpGeneric (numerators become QM31)
                Layer::LogUpGeneric {
                    numerators:   Mle::new(CudaColumn::from_device_buffer(d_out_num, half_n)),
                    denominators: Mle::new(CudaColumn::from_device_buffer(d_out_den, half_n)),
                }
            }

            Layer::LogUpSingles { denominators } => {
                let n = denominators.len();
                if n < 2 {
                    return next_layer_cpu(layer);
                }
                let half_n = n / 2;
                let mut d_out_num = DeviceBuffer::<u32>::alloc(half_n * 4);
                let mut d_out_den = DeviceBuffer::<u32>::alloc(half_n * 4);
                unsafe {
                    ffi::cuda_gkr_next_layer_logup_singles(
                        denominators.buf.as_ptr(),
                        d_out_num.as_mut_ptr(),
                        d_out_den.as_mut_ptr(),
                        n as u32,
                    );
                    ffi::cuda_device_sync();
                }
                Layer::LogUpGeneric {
                    numerators:   Mle::new(CudaColumn::from_device_buffer(d_out_num, half_n)),
                    denominators: Mle::new(CudaColumn::from_device_buffer(d_out_den, half_n)),
                }
            }
        }
    }

    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        let n_variables = h.n_variables();
        assert!(n_variables != 0);

        let n_terms = 1usize << (n_variables - 1);

        // Allocate partial sums buffer: ceil(n_terms/256) * 8 u32s
        let n_blocks = (n_terms + 255) / 256;
        let mut d_partial = DeviceBuffer::<u32>::alloc(n_blocks * 8);

        // eq_evals device pointer (via Deref to CudaColumn<SecureField>)
        let eq_ptr = h.eq_evals.buf.as_ptr();

        let lambda_words = upload_qm31(h.lambda);

        let blocks_used = unsafe {
            match &h.input_layer {
                Layer::GrandProduct(mle) => {
                    ffi::cuda_gkr_sum_poly_grand_product(
                        eq_ptr,
                        mle.buf.as_ptr(),
                        d_partial.as_mut_ptr(),
                        n_terms as u32,
                    )
                }
                Layer::LogUpGeneric { numerators, denominators } => {
                    ffi::cuda_gkr_sum_poly_logup_generic(
                        eq_ptr,
                        numerators.buf.as_ptr(),
                        denominators.buf.as_ptr(),
                        lambda_words.as_ptr(),
                        d_partial.as_mut_ptr(),
                        n_terms as u32,
                    )
                }
                Layer::LogUpMultiplicities { numerators, denominators } => {
                    ffi::cuda_gkr_sum_poly_logup_mult(
                        eq_ptr,
                        numerators.buf.as_ptr(),
                        denominators.buf.as_ptr(),
                        lambda_words.as_ptr(),
                        d_partial.as_mut_ptr(),
                        n_terms as u32,
                    )
                }
                Layer::LogUpSingles { denominators } => {
                    ffi::cuda_gkr_sum_poly_logup_singles(
                        eq_ptr,
                        denominators.buf.as_ptr(),
                        lambda_words.as_ptr(),
                        d_partial.as_mut_ptr(),
                        n_terms as u32,
                    )
                }
            }
        };

        unsafe { ffi::cuda_device_sync(); }

        // Download and accumulate partial sums
        let psums = d_partial.to_host();
        let (mut eval_at_0, mut eval_at_2) =
            reduce_partial_sums(&psums[..blocks_used as usize * 8]);

        // Apply eq_fixed_var_correction (CPU-side scalar multiplication)
        eval_at_0 = eval_at_0 * h.eq_fixed_var_correction;
        eval_at_2 = eval_at_2 * h.eq_fixed_var_correction;

        let y = h.eq_evals.y();
        correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, n_variables)
    }
}

/// CPU fallback for trivially small layers (n < 2).
fn next_layer_cpu(layer: &Layer<CudaBackend>) -> Layer<CudaBackend> {
    let cpu_layer = layer.to_cpu();
    let next_cpu = CpuBackend::next_layer(&cpu_layer);
    layer_to_cuda(next_cpu)
}

fn layer_to_cuda(layer: Layer<CpuBackend>) -> Layer<CudaBackend> {
    match layer {
        Layer::GrandProduct(mle) => {
            let data: Vec<SecureField> = mle.to_cpu();
            Layer::GrandProduct(Mle::new(data.into_iter().collect()))
        }
        Layer::LogUpGeneric { numerators, denominators } => {
            let num: Vec<SecureField> = numerators.to_cpu();
            let den: Vec<SecureField> = denominators.to_cpu();
            Layer::LogUpGeneric {
                numerators:   Mle::new(num.into_iter().collect()),
                denominators: Mle::new(den.into_iter().collect()),
            }
        }
        Layer::LogUpMultiplicities { numerators, denominators } => {
            let num: Vec<BaseField>   = numerators.to_cpu();
            let den: Vec<SecureField> = denominators.to_cpu();
            Layer::LogUpMultiplicities {
                numerators:   Mle::new(num.into_iter().collect()),
                denominators: Mle::new(den.into_iter().collect()),
            }
        }
        Layer::LogUpSingles { denominators } => {
            let den: Vec<SecureField> = denominators.to_cpu();
            Layer::LogUpSingles {
                denominators: Mle::new(den.into_iter().collect()),
            }
        }
    }
}
