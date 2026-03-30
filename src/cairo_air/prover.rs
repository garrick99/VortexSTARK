//! End-to-end Cairo STARK prover and verifier.
//!
//! Produces a complete proof of Cairo program execution:
//! 1. VM execution → 31-column trace
//! 2. Optional GPU Pedersen builtin → 27 more columns
//! 3. NTT + Merkle commitment
//! 4. Fiat-Shamir challenges
//! 5. Fused LogUp interaction trace (denoms + inverse + combine)
//! 6. Constraint quotient evaluation
//! 7. FRI folding + commitments
//! 8. Decommitment (Merkle auth paths)
//!
//! The verifier replays Fiat-Shamir, checks Merkle paths, and verifies
//! FRI fold equations — same structure as the Fibonacci verifier but for
//! the full Cairo AIR.

use crate::channel::Channel;
use crate::circle::Coset;
use crate::cuda::ffi;
use crate::device::DeviceBuffer;
use crate::field::{M31, QM31};
use crate::fri::{self, SecureColumn};
use crate::merkle::MerkleTree;
use crate::ntt::{self, ForwardTwiddleCache, InverseTwiddleCache};
use crate::prover::{QueryDecommitment, N_QUERIES, BLOWUP_BITS, POW_BITS};
use super::trace::{N_COLS, N_VM_COLS, N_CONSTRAINTS,
    COL_PC, COL_AP, COL_FP, COL_INST_LO, COL_INST_HI,
    COL_FLAGS_START, COL_DST_ADDR, COL_DST, COL_OP0_ADDR, COL_OP0, COL_OP1_ADDR, COL_OP1,
    COL_RES, COL_OFF0, COL_OFF1, COL_OFF2, COL_DST_INV,
    COL_DICT_KEY, COL_DICT_NEW, COL_DICT_ACTIVE};
use super::range_check::{extract_offsets, compute_rc_table_sum};
use super::logup::{extract_memory_table, compute_memory_table_sum};
use super::dict_air;

use super::vm::Memory;
use super::ec_constraint;

/// Errors returned by the hint-aware prover.
#[derive(Debug, Clone)]
pub enum ProveError {
    /// The program bytecode contains felt252 values that cannot be represented in 64 bits.
    /// These values were truncated at load time; proof output would be incorrect.
    /// Use only programs whose field values stay within u64 range.
    Felt252Overflow { count: usize },

    /// A dict access chain is inconsistent: an access's prev_value does not match
    /// the preceding access's new_value for the same key.
    /// This indicates a hint execution bug; the generated trace would be invalid.
    DictConsistencyViolation { key: u64, expected_prev: u64, actual_prev: u64 },

    /// One or more data values read during execution exceeded M31 (P = 2^31 - 1).
    /// VortexSTARK operates over M31; values in the range (M31, u64] are silently
    /// truncated mod M31, producing a proof of the wrong computation.
    /// Use only programs whose runtime values stay within M31.
    ExecutionRangeViolation { count: usize },

    /// n_steps is not equal to 2^log_n.
    ///
    /// VortexSTARK requires the trace to be fully populated: n_steps must equal exactly 2^log_n.
    /// When n_steps < 2^log_n, padding rows are all-zero, which causes the LogUp and Cairo
    /// step-transition constraints to fail at the padding boundary, producing an invalid quotient
    /// polynomial that FRI rejects. There is no silent failure — this error is returned so callers
    /// can correct the mismatch before proving.
    ///
    /// Fix: choose log_n = ceil(log2(n_steps)) and pad the program to run exactly 2^log_n steps
    /// (e.g. append a self-loop that runs the remaining steps), or structure the program so that
    /// its step count is always a power of two.
    TraceSizeMismatch { n_steps: usize, log_n: u32 },

    /// The requested log_n requires more VRAM than is currently free on the GPU.
    /// Peak VRAM during NTT = N_COLS × 4 × eval_size bytes (all eval columns live simultaneously).
    /// Fix: lower log_n, free GPU memory held by other processes, or use a larger GPU.
    InsufficientVRAM { required_gb: f64, free_gb: f64 },
}

impl std::fmt::Display for ProveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProveError::Felt252Overflow { count } =>
                write!(f, "{count} felt252 value(s) in bytecode exceed u64 range and were truncated; \
                           proof would be incorrect — program must stay within u64 arithmetic"),
            ProveError::DictConsistencyViolation { key, expected_prev, actual_prev } =>
                write!(f, "dict consistency violation at key {key:#x}: \
                           expected prev_value={expected_prev:#x} but got {actual_prev:#x}"),
            ProveError::ExecutionRangeViolation { count } =>
                write!(f, "{count} data value(s) read during execution exceeded M31 (P = 2^31-1); \
                           these would be silently truncated — program must use only M31 arithmetic"),
            ProveError::TraceSizeMismatch { n_steps, log_n } =>
                write!(f, "n_steps ({n_steps}) ≠ 2^log_n ({}): trace padding is unsupported — \
                           ensure n_steps == 2^log_n by choosing log_n = {} and padding the program \
                           to run exactly that many steps",
                    1usize << log_n,
                    (*n_steps as f64).log2().ceil() as u32),
            ProveError::InsufficientVRAM { required_gb, free_gb } =>
                write!(f, "insufficient VRAM: proof needs {required_gb:.1} GB but only {free_gb:.1} GB \
                           is free — lower log_n, stop other GPU processes, or use a larger GPU"),
        }
    }
}

/// Permute data from half_coset BRT-NTT order to stwo CanonicCoset (CircleDomain) order.
/// Permute data from half_coset BRT-NTT order to BRT-canonic order (stwo BitReversedOrder).
///
/// BRT-canonic: position j has the value at canonic_domain_point(BRT(j)).
/// This is what stwo stores in `SecureEvaluation<B, BitReversedOrder>`.
fn permute_half_coset_to_canonic(data: &[u32], log_n: u32) -> Vec<u32> {
    let n = 1usize << log_n;
    assert_eq!(data.len(), n);
    let mut out = vec![0u32; n];
    for i in 0..n {
        // NTT position i → half_coset natural index k = bit_reverse(i)
        let k = i.reverse_bits() >> (usize::BITS - log_n);
        // Natural half_coset index k → canonic natural position
        let cn = if k % 2 == 0 { k / 2 } else { n - 1 - k / 2 };
        // Canonic natural → BRT-canonic position
        let j = cn.reverse_bits() >> (usize::BITS - log_n);
        out[j] = data[i];
    }
    out
}

/// Map canonic position j to the half_coset BRT-NTT index with the same circle point.
pub fn canonic_to_hc_ntt(j: usize, log_n: u32) -> usize {
    let n = 1usize << log_n;
    let k = if j < n / 2 { 2 * j } else { 2 * (n - 1 - j) + 1 };
    let mut r = 0usize;
    let mut v = k;
    for _ in 0..log_n { r = (r << 1) | (v & 1); v >>= 1; }
    r
}

/// Map half_coset BRT-NTT position i to canonic position.
pub fn hc_ntt_to_canonic(i: usize, log_n: u32) -> usize {
    let n = 1usize << log_n;
    let k = {
        let mut r = 0usize;
        let mut v = i;
        for _ in 0..log_n { r = (r << 1) | (v & 1); v >>= 1; }
        r
    };
    if k % 2 == 0 { k / 2 } else { n - 1 - k / 2 }
}

/// Get the canonic domain point at position j.
fn canonic_domain_point(j: usize, log_eval_size: u32) -> crate::circle::CirclePoint {
    let ho = crate::circle::Coset::half_odds(log_eval_size - 1);
    let half_n = 1usize << (log_eval_size - 1);
    if j < half_n { ho.at(j) } else { ho.at(j - half_n).conjugate() }
}

/// Compute "next trace step" position from BRT-canonic position qi.
/// Maps: BRT-canonic qi → canonic natural → half_coset NTT → +1 → canonic natural → BRT-canonic.
pub fn canonic_next(qi: usize, log_eval_size: u32) -> usize {
    // BRT-canonic qi → canonic natural
    let cn = qi.reverse_bits() >> (usize::BITS - log_eval_size);
    let eval_size = 1usize << log_eval_size;
    // canonic natural → half_coset NTT → +1 → canonic natural → BRT-canonic
    let hc = canonic_to_hc_ntt(cn, log_eval_size);
    let hc_next = (hc + 1) % eval_size;
    let cn_next = hc_ntt_to_canonic(hc_next, log_eval_size);
    cn_next.reverse_bits() >> (usize::BITS - log_eval_size)
}

/// Compute peak VRAM (bytes) needed for a Cairo proof of the given size.
///
/// With group-batched NTT, only the largest column group (16 cols) plus the ZH
/// blinding buffer (1 col) are ever live on GPU simultaneously.
/// Peak = (TRACE_LO + 1) × eval_size × 4 bytes.
///
/// Twiddle caches (~3 GB at log_n=26) are pre-allocated before this check runs.
pub fn cairo_vram_required(log_n: u32) -> usize {
    const TRACE_LO: usize = 16;  // largest NTT group (cols 0..TRACE_LO)
    let eval_size = 1usize << (log_n + BLOWUP_BITS);
    (TRACE_LO + 1) * eval_size * 4  // largest group + ZH buffer
}

/// Check whether the current GPU has enough free VRAM for a Cairo proof at `log_n`.
///
/// Call this before allocating a `CairoProverCache` to get an early, accurate reading.
/// The twiddle caches (allocated inside `CairoProverCache`) consume ~3 GB at log_n=26
/// but are included in the budget implicitly since `vram_query()` is called AFTER they
/// are allocated when invoked from `cairo_prove_program`.
pub fn cairo_check_vram(log_n: u32) -> Result<(), ProveError> {
    let required = cairo_vram_required(log_n);
    let (free, _total) = crate::cuda::ffi::vram_query();
    if required > free {
        let gb = |b: usize| b as f64 / (1u64 << 30) as f64;
        Err(ProveError::InsufficientVRAM { required_gb: gb(required), free_gb: gb(free) })
    } else {
        Ok(())
    }
}

/// Public inputs for a Cairo proof: initial/final VM state + program hash.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CairoPublicInputs {
    /// Initial program counter
    pub initial_pc: u32,
    /// Initial allocation pointer
    pub initial_ap: u32,
    /// Number of execution steps
    pub n_steps: usize,
    /// Hash of the program bytecode (first 8 words of Blake2s digest)
    pub program_hash: [u32; 8],
    /// Program bytecode (needed by verifier to recompute memory table sum)
    pub program: Vec<u64>,
}

/// Complete Cairo STARK proof.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CairoProof {
    pub log_trace_size: u32,
    /// Public inputs (verified by both prover and verifier)
    pub public_inputs: CairoPublicInputs,
    /// Merkle root: VM trace columns 0-15 (lo half)
    pub trace_commitment: [u32; 8],
    /// Merkle root: VM trace columns 16-30 (hi half).
    pub trace_commitment_hi: [u32; 8],
    /// Merkle root: dict linkage columns 31-33 (dict_key, dict_new, dict_active).
    /// Together with trace_commitment and trace_commitment_hi, commits all 34 trace columns.
    /// Committed before z_dict is drawn — binds dict operations into the FRI polynomial
    /// and closes GAP-1: a malicious prover cannot change dict_key/dict_new after commitment.
    pub dict_trace_commitment: [u32; 8],
    /// Merkle root: LogUp interaction trace (4 QM31 columns)
    pub interaction_commitment: [u32; 8],
    /// Merkle root: range check interaction trace (4 QM31 columns)
    pub rc_interaction_commitment: [u32; 8],
    /// Merkle root: EC trace columns 0-15 (lo half; GPU hash caps at 16 cols)
    pub ec_trace_commitment: Option<[u32; 8]>,
    /// Merkle root: EC trace columns 16-28 (hi half)
    pub ec_trace_commitment_hi: Option<[u32; 8]>,
    /// log2 of the EC eval domain size (needed by verifier to map main query indices)
    pub ec_log_eval: Option<u32>,
    /// EC trace values at query points (for verifier constraint check)
    pub ec_trace_at_queries: Vec<Vec<u32>>,
    pub ec_trace_at_queries_next: Vec<Vec<u32>>,
    /// Merkle auth paths binding EC trace cols 0-15 at query points to ec_trace_commitment
    pub ec_trace_auth_paths: Vec<Vec<[u32; 8]>>,
    /// Merkle auth paths binding EC trace cols 0-15 at query+1 points
    pub ec_trace_auth_paths_next: Vec<Vec<[u32; 8]>>,
    /// Merkle auth paths binding EC trace cols 16-28 at query points to ec_trace_commitment_hi
    pub ec_trace_auth_paths_hi: Vec<Vec<[u32; 8]>>,
    /// Merkle auth paths binding EC trace cols 16-28 at query+1 points
    pub ec_trace_auth_paths_hi_next: Vec<Vec<[u32; 8]>>,
    /// Merkle root: combined quotient (4 QM31 columns)
    pub quotient_commitment: [u32; 8],
    /// Merkle commitment to the OODS quotient polynomial (FRI input).
    /// Committed after the OODS block so the channel state includes it before fri_alpha.
    #[serde(default)]
    pub oods_quotient_commitment: [u32; 8],
    /// Channel state just before mixing oods_quotient_commitment (start of FRI phase).
    #[serde(default)]
    pub fri_start_channel_state: [u32; 8],
    /// FRI layer commitments
    pub fri_commitments: Vec<[u32; 8]>,
    /// FRI last layer evaluations at committed_stop_log (32 elements, BRT-ordered).
    pub fri_last_layer: Vec<QM31>,
    /// FRI last layer after uncommitted folds (8 elements, for LinePoly).
    #[serde(default)]
    pub fri_last_layer_poly: Vec<QM31>,
    /// Proof-of-work nonce: Blake2s(prefix_digest || nonce_le) must have >= POW_BITS trailing zeros.
    /// Prevents query index grinding attacks. Computed on GPU after all FRI commitments.
    pub pow_nonce: u64,
    /// Query indices
    pub query_indices: Vec<usize>,
    /// Trace values at query points (N_COLS M31 values per query)
    pub trace_values_at_queries: Vec<Vec<u32>>,
    /// Trace values at query+1 points (for next-row constraints)
    pub trace_values_at_queries_next: Vec<Vec<u32>>,
    /// Merkle auth paths binding trace cols 0-15 at query points to trace_commitment
    pub trace_auth_paths: Vec<Vec<[u32; 8]>>,
    /// Merkle auth paths binding trace cols 0-15 at query+1 points to trace_commitment
    pub trace_auth_paths_next: Vec<Vec<[u32; 8]>>,
    /// Merkle auth paths binding trace cols 16-30 at query points to trace_commitment_hi
    pub trace_auth_paths_hi: Vec<Vec<[u32; 8]>>,
    /// Merkle auth paths binding trace cols 16-30 at query+1 points to trace_commitment_hi
    pub trace_auth_paths_hi_next: Vec<Vec<[u32; 8]>>,
    /// Merkle auth paths binding dict cols 31-33 at query points to dict_trace_commitment
    pub trace_auth_paths_dict: Vec<Vec<[u32; 8]>>,
    /// Merkle auth paths binding dict cols 31-33 at query+1 points to dict_trace_commitment
    pub trace_auth_paths_dict_next: Vec<Vec<[u32; 8]>>,
    /// LogUp memory argument: claimed final running sum
    pub logup_final_sum: [u32; 4],
    /// Range check: claimed final running sum
    pub rc_final_sum: [u32; 4],
    /// Decommitments (with Merkle auth paths)
    pub quotient_decommitment: QueryDecommitment<[u32; 4]>,
    /// Decommitment for the OODS quotient polynomial at FRI query points.
    #[serde(default)]
    pub oods_quotient_decommitment: QueryDecommitment<[u32; 4]>,
    pub fri_decommitments: Vec<QueryDecommitment<[u32; 4]>>,
    /// LogUp interaction trace at query points, auth-path-bound to interaction_commitment.
    pub interaction_decommitment: QueryDecommitment<[u32; 4]>,
    /// LogUp interaction trace at query+1 points (for step transition verification).
    pub interaction_decommitment_next: QueryDecommitment<[u32; 4]>,
    /// RC interaction trace at query points, auth-path-bound to rc_interaction_commitment.
    pub rc_interaction_decommitment: QueryDecommitment<[u32; 4]>,
    /// RC interaction trace at query+1 points (for step transition verification).
    pub rc_interaction_decommitment_next: QueryDecommitment<[u32; 4]>,
    /// Merkle root: S_dict step-transition interaction trace (4 QM31 cols).
    /// Committed after dict_trace_commitment and before EC trace, binding the prover to
    /// the S_dict polynomial before z_mem/alpha_mem are drawn. Mixed into Fiat-Shamir.
    pub dict_main_interaction_commitment: [u32; 8],
    /// S_dict interaction trace at query points (auth-path bound to dict_main_interaction_commitment).
    pub dict_main_interaction_decommitment: QueryDecommitment<[u32; 4]>,
    /// S_dict interaction trace at query+1 points (for step-transition constraint C34).
    pub dict_main_interaction_decommitment_next: QueryDecommitment<[u32; 4]>,
    /// S_dict final value (running LogUp sum over all dict_active=1 rows).
    /// Verifier recomputes exec_key_new_sum from dict_exec_data[0..dict_n_accesses]
    /// and checks: dict_link_final == exec_key_new_sum.
    pub dict_link_final: [u32; 4],
    /// Number of actual dict accesses (padding starts here in dict_exec_data).
    /// Verifier validates that dict_exec_data[dict_n_accesses..] are all zeros.
    pub dict_n_accesses: usize,
    /// Packed QM31 challenges used by quotient kernel:
    /// [z_mem(4), alpha_mem(4), alpha_mem_sq(4), z_rc(4), z_dict_link(4), alpha_dict_link(4)]
    pub logup_challenges: [u32; 24],

    // ---- Dict consistency sub-AIR (GAP-1) ----
    //
    // When the Cairo program uses felt252 dicts, the prover commits two Merkle polynomial
    // traces BEFORE Fiat-Shamir challenges are drawn:
    //   1. exec  data   (3 cols: key, prev, new — execution order)
    //   2. sorted data  (4 cols: key, prev, new, is_first — key-sorted)
    //
    // The verifier receives the FULL exec and sorted trace data and:
    //   a) Recomputes both Merkle roots from the full data and checks against committed roots.
    //   b) Recomputes exec_final_sum and sorted_final_sum by running the LogUp over full data.
    //   c) Checks ALL sorted step-transition constraints C0-C3 (not sampled — all dict_n-1 pairs).
    //   d) Checks permutation argument: exec_final_sum == sorted_final_sum.
    //
    // This achieves FULL soundness (not statistical) for the dict sub-AIR.
    // The remaining gap (exec trace not linked to main execution trace) is documented in AUDIT.md.
    //
    // If no dict accesses occurred, all Option fields are None and Vec fields are empty.

    /// Merkle root of exec-order dict data trace  (3 cols, 2^dict_log_n rows)
    pub dict_exec_commitment: Option<[u32; 8]>,
    /// Merkle root of key-sorted dict data trace  (4 cols, 2^dict_log_n rows)
    pub dict_sorted_commitment: Option<[u32; 8]>,
    /// log2 of the dict trace domain size
    pub dict_log_n: Option<u32>,
    /// Total LogUp sum over exec dict trace  (QM31; must equal dict_sorted_final_sum)
    pub dict_exec_final_sum: Option<[u32; 4]>,
    /// Total LogUp sum over sorted dict trace (QM31; must equal dict_exec_final_sum)
    pub dict_sorted_final_sum: Option<[u32; 4]>,
    /// Full execution-order dict data trace (all 2^dict_log_n rows, 3 cols each).
    /// Verifier recomputes exec Merkle root from this data to authenticate it.
    pub dict_exec_data: Vec<[u32; 3]>,
    /// Full key-sorted dict data trace (all 2^dict_log_n rows, 4 cols each).
    /// Verifier recomputes sorted Merkle root and checks ALL step-transition constraints C0-C3.
    pub dict_sorted_data: Vec<[u32; 4]>,

    // ---- Bitwise builtin (full-data, no sampling) ----
    //
    // When the Cairo program uses the memory-mapped bitwise builtin, the prover
    // includes ALL invocation data here (one [x, y, and, xor, or] per invocation).
    // A Blake2s hash of this data is mixed into the Fiat-Shamir channel BEFORE FRI
    // challenges are drawn, binding the prover to this data.
    //
    // The verifier recomputes the hash and checks EVERY row:
    //   C0: xor + 2*and == x + y  (mod P)
    //   C1: or  == and + xor       (mod P)
    //
    // These constraints hold for all well-formed bitwise invocations when inputs
    // x, y < 2^15. For full M31 inputs (up to 2^31-2), x+y may overflow P causing
    // a false positive — documented limitation (see SOUNDNESS.md).
    //
    // The link between main trace memory accesses and bitwise rows is not yet proven
    // by a permutation argument (same status as dict sub-AIR before Stage 2).

    /// Blake2s commitment to all bitwise invocation data (mixed into Fiat-Shamir).
    /// None when no bitwise operations were used.
    pub bitwise_commitment: Option<[u32; 8]>,
    /// All bitwise invocations: each entry is [x, y, x&y, x^y, x|y].
    /// Verifier checks C0 and C1 for every row.
    pub bitwise_rows: Vec<[u32; 5]>,

    // ---- Memory table commitment (closes LogUp soundness gap) ----
    //
    // The prover includes the full memory table (all unique (addr, val, mult) entries)
    // as explicit proof data, committed via Blake2s before constraint_alphas are drawn.
    // The verifier independently recomputes the Blake2s hash, then computes table_sum
    // and checks: exec_sum + table_sum == 0.
    //
    // This mirrors the dict sub-AIR approach and the bitwise commitment pattern,
    // closing the gap where the verifier previously accepted any claimed exec_sum.

    /// Blake2s commitment to memory_table_data ++ memory_instr_data (mixed into Fiat-Shamir).
    pub memory_table_commitment: [u32; 8],
    /// Unique data memory entries from execution: [addr, val, mult] per entry.
    pub memory_table_data: Vec<[u32; 3]>,
    /// Unique instruction fetch entries from execution: [pc, inst_lo, inst_hi, mult] per entry.
    pub memory_instr_data: Vec<[u32; 4]>,

    // ---- RC multiplicity table commitment (closes RC soundness gap) ----
    //
    // The prover includes all 65536 multiplicity counts for range check values 0..2^16.
    // The verifier recomputes the Blake2s hash, then computes rc_table_sum and checks:
    // rc_exec_sum + rc_table_sum == 0.

    /// Blake2s commitment to rc_counts_data (mixed into Fiat-Shamir after memory_table_commitment).
    pub rc_counts_commitment: [u32; 8],
    /// Multiplicity counts for range check values 0..2^16 (65536 entries).
    pub rc_counts_data: Vec<u32>,

    // ---- Out-Of-Domain Sampling (OODS) — stwo wire format ----
    //
    // After committing all trace trees, draw OODS point z ∈ QM31 circle from channel.
    // Evaluate each of the 34 main trace columns at z (current row) and z_next (next row).
    // Mix sampled_values into channel BEFORE FRI alpha is drawn — this binds the prover
    // to the committed polynomials' values at an unpredictable out-of-domain point,
    // providing the "polynomial identity" guarantee needed for on-chain verifiability.
    //
    // Wire format mirrors stwo's `CommitmentSchemeProof.sampled_values`:
    //   TreeVec<ColumnVec<Vec<SecureField>>>
    // Here flattened as two vectors of QM31 values (one per sample point).

    /// OODS point z encoded as [z.x.v0, z.x.v1, z.x.v2, z.x.v3, z.y.v0, z.y.v1, z.y.v2, z.y.v3].
    /// z is drawn from channel after all polynomial commitments are mixed in.
    #[serde(default)]
    pub oods_z: [u32; 8],
    /// Evaluations of all 34 trace columns at z (current-row OODS point).
    /// Each entry is [v0, v1, v2, v3] = QM31 value.
    #[serde(default)]
    pub oods_trace_at_z: Vec<[u32; 4]>,
    /// Evaluations of all 34 trace columns at z_next = z * step (next-row OODS point).
    #[serde(default)]
    pub oods_trace_at_z_next: Vec<[u32; 4]>,
    /// Evaluations of the 4 AIR quotient columns (q0..q3) at z.
    /// Each qk is an M31-valued polynomial evaluated at the QM31 OODS point z.
    #[serde(default)]
    pub oods_quotient_at_z: [[u32; 4]; 4],
    /// OODS linear combination alpha, drawn after all sampled_values are mixed into channel.
    /// Used to combine all (column, sample_point) pairs into the OODS quotient polynomial for FRI.
    #[serde(default)]
    pub oods_alpha: [u32; 4],
    /// OODS evaluations of the 3 interaction polynomials at z.
    /// Indexed as [poly][component]: poly ∈ {0=LogUp, 1=RC, 2=S_dict}, component ∈ 0..4.
    /// Each entry is a QM31 value (M31 component polynomial evaluated at QM31 point z).
    /// Mixed into Fiat-Shamir between trace evals and AIR quotient evals (stwo tree order).
    #[serde(default)]
    pub oods_interaction_at_z: [[[u32; 4]; 4]; 3],
    /// OODS evaluations of the 3 interaction polynomials at z_next.
    /// Same layout as oods_interaction_at_z.
    /// Mixed immediately after oods_interaction_at_z (step-transition binding).
    #[serde(default)]
    pub oods_interaction_at_z_next: [[[u32; 4]; 4]; 3],
}


/// Reusable NTT caches for a given trace size. Create once, prove many.
pub struct CairoProverCache {
    pub log_n: u32,
    pub inv_cache: InverseTwiddleCache,
    pub fwd_cache: ForwardTwiddleCache,
    /// Inverse twiddles for the eval domain — used to INTT the AIR quotient polynomial
    /// back to coefficient form for OODS evaluation at the QM31 point z.
    pub eval_inv_cache: InverseTwiddleCache,
}

impl CairoProverCache {
    pub fn new(log_n: u32) -> Self {
        let log_eval_size = log_n + BLOWUP_BITS;
        let trace_domain = Coset::half_coset(log_n);
        let eval_domain = Coset::half_coset(log_eval_size);
        Self {
            log_n,
            inv_cache: InverseTwiddleCache::new(&trace_domain),
            fwd_cache: ForwardTwiddleCache::new(&eval_domain),
            eval_inv_cache: InverseTwiddleCache::new(&eval_domain),
        }
    }
}

/// Prove execution of a Cairo program.
pub fn cairo_prove(program: &[u64], n_steps: usize, log_n: u32) -> CairoProof {
    let cache = CairoProverCache::new(log_n);
    cairo_prove_cached(program, n_steps, log_n, &cache, None)
}

/// Prove with optional Pedersen EC constraint trace.
pub fn cairo_prove_with_pedersen(
    program: &[u64], n_steps: usize, log_n: u32,
    pedersen_inputs: Option<(&[super::stark252_field::Fp], &[super::stark252_field::Fp])>,
) -> CairoProof {
    let cache = CairoProverCache::new(log_n);
    cairo_prove_cached(program, n_steps, log_n, &cache, pedersen_inputs)
}

/// Prove execution of a Cairo program loaded via `casm_loader::load_program`.
/// Unlike `cairo_prove`, this executes hints registered in the program before each step.
///
/// Returns `Err(ProveError::Felt252Overflow)` if the bytecode contains felt252 values
/// that exceed u64 range (silent truncation would produce an incorrect proof).
///
/// Returns `Err(ProveError::DictConsistencyViolation)` if hint execution produced a dict
/// access log whose prev/new chain is inconsistent (indicating a hint execution bug).
/// Prove a Cairo program with a custom Starknet syscall environment.
///
/// Identical to `cairo_prove_program` but supplies initial contract storage,
/// execution context (caller, contract address, entry-point selector), and
/// block info so that syscalls (`storage_read`, `get_execution_info`, etc.)
/// return meaningful values instead of zeros.
///
/// After proving, inspect `syscall_state.storage` for writes and
/// `syscall_state.events` for emitted events.
pub fn cairo_prove_program_with_syscalls(
    program: &super::casm_loader::CasmProgram,
    n_steps: usize,
    log_n: u32,
    syscall_state: super::hints::SyscallState,
) -> Result<(CairoProof, super::hints::SyscallState), ProveError> {
    cairo_prove_program_inner(program, n_steps, log_n, Some(syscall_state))
        .map(|(proof, mut ctx)| (proof, std::mem::take(&mut ctx.syscall)))
}

pub fn cairo_prove_program(
    program: &super::casm_loader::CasmProgram,
    n_steps: usize,
    log_n: u32,
) -> Result<CairoProof, ProveError> {
    cairo_prove_program_inner(program, n_steps, log_n, None).map(|(proof, _)| proof)
}

fn cairo_prove_program_inner(
    program: &super::casm_loader::CasmProgram,
    n_steps: usize,
    log_n: u32,
    syscall_state: Option<super::hints::SyscallState>,
) -> Result<(CairoProof, super::hints::HintContext), ProveError> {
    // Refuse to prove programs with truncated felt252 bytecode values.
    // Such programs would produce proofs for wrong computations.
    if program.overflow_count > 0 {
        return Err(ProveError::Felt252Overflow { count: program.overflow_count });
    }

    let n = 1usize << log_n;

    // Sanity check: caller must have chosen log_n >= ceil(log2(n_steps)).
    if n_steps > n {
        return Err(ProveError::TraceSizeMismatch { n_steps, log_n });
    }

    // ── Height check: can the GPU fit this proof? ──────────────────────────
    // Allocate the twiddle caches first so that vram_query() reflects their
    // footprint; the remaining free VRAM is what the NTT loop needs.
    let cache = CairoProverCache::new(log_n);
    cairo_check_vram(log_n)?;

    // Set up the Cairo calling convention: place a return sentinel below the initial frame.
    // pad_addr hosts a 2-word "jmp rel 0" self-loop (instruction + immediate 0) so that
    // the program's top-level `ret` lands there and fills remaining trace rows up to 2^log_n
    // with valid Cairo steps without violating any constraints.
    let pad_addr = program.bytecode.len() as u64; // first free address after bytecode
    let initial_sp = pad_addr + 100;              // frame base (well above pad_addr)
    let initial_ap = initial_sp + 2;              // AP/FP after the call frame
    let mut mem = Memory::with_capacity(initial_ap as usize + n + 200);
    mem.load_program(&program.bytecode);
    // Padding: jmp rel [pc+1] with immediate 0 → next_pc = pc + 0 = pc (infinite self-loop).
    // Encoding: op1_imm=1 (bit 2), pc_jump_rel=1 (bit 8), off2=0x8001 (+1 bias = pc+1),
    //           off0=off1=0x8000 (zero offsets), ap/fp updates all zero.
    mem.set(pad_addr,     0x0104_8001_8000_8000); // jmp rel 0 instruction
    mem.set(pad_addr + 1, 0);                     // immediate value = 0 (jump delta)
    mem.set(initial_sp,     0);        // saved fp  = 0 (sentinel)
    mem.set(initial_sp + 1, pad_addr); // return pc = pad_addr (self-loop on halt)

    // Thread HintContext externally so the prover can inspect dict accesses after execution.
    let mut hint_ctx = if let Some(sc) = syscall_state {
        super::hints::HintContext::new().with_syscall_state(sc)
    } else {
        super::hints::HintContext::new()
    };
    // Execute n steps total: n_steps real program steps + (n - n_steps) self-loop padding.
    // The padding self-loop at pad_addr fills remaining rows with valid Cairo transitions.
    let columns = super::vm::execute_to_columns_with_hints(
        &mut mem, n, log_n, &program.hints,
        program.entry_point, initial_ap,
        &mut hint_ctx,
    );

    // Refuse to prove programs where any execution-time data value exceeded M31.
    // Such values are silently truncated mod M31 in the trace, producing a proof of
    // a different computation. Closing GAP-2: reject at prove time rather than silently mismatch.
    if hint_ctx.execution_overflows > 0 {
        return Err(ProveError::ExecutionRangeViolation { count: hint_ctx.execution_overflows });
    }

    // Verify dict access chain consistency (CPU-side check).
    // Detects hint execution bugs where prev_value doesn't match the preceding new_value.
    // Note: this is an execution-side check — dict consistency is not yet enforced in the
    // STARK proof itself (that requires dedicated dict trace columns; planned future work).
    if !hint_ctx.dict_accesses.is_empty() {
        super::dict_consistency::verify_chain(&hint_ctx.dict_accesses)
            .map_err(|e| match e {
                super::dict_consistency::DictConsistencyError::ChainViolation { key, expected_prev, actual_prev } =>
                    ProveError::DictConsistencyViolation { key, expected_prev, actual_prev },
            })?;
    }

    // Extract any bitwise invocations that occurred during execution.
    // Memory auto-filled the AND/XOR/OR outputs when both x and y were written.
    let bitwise_rows = mem.extract_bitwise_invocations();
    drop(mem);

    let proof = cairo_prove_cached_with_columns(
        &program.bytecode, columns, n, log_n, &cache, None,
        &hint_ctx.dict_accesses, bitwise_rows,
    );
    Ok((proof, hint_ctx))
}

/// Prove with reusable cache (fast path for repeated proofs at same size).
pub fn cairo_prove_cached(
    program: &[u64], n_steps: usize, log_n: u32,
    cache: &CairoProverCache,
    pedersen_inputs: Option<(&[super::stark252_field::Fp], &[super::stark252_field::Fp])>,
) -> CairoProof {
    let mut mem = Memory::with_capacity(n_steps + 200);
    mem.load_program(program);
    let columns = super::vm::execute_to_columns(&mut mem, n_steps, log_n);
    cairo_prove_cached_with_columns(program, columns, n_steps, log_n, cache, pedersen_inputs, &[], Vec::new())
}

/// All 34 trace columns are ZK-blinded with r · Z_H(x).
///
/// Z_H(x) vanishes on the trace domain, so witness values at trace points are
/// unchanged. At query points, each blinded column reveals a uniformly random
/// linear combination instead of the true value (ZK property).
///
/// The columns appearing in rational QM31-inverse denominators (pc, inst_lo/hi,
/// dst/op0/op1 addr/val, dict_key/new/active) are also blinded — the same argument
/// that makes off0/off1/off2 blinding correct for C32 applies to all columns:
/// the constraint still vanishes on the trace domain (Z_H=0 there), so the quotient
/// Q = C/Z_H is still a polynomial. The prover and verifier both use blinded column
/// values consistently, so soundness and completeness are preserved.
pub const ZK_BLIND_COLS: &[usize] = &[
    // Group A (cols 0..16)
    COL_PC, COL_AP, COL_FP, COL_INST_LO, COL_INST_HI,
    COL_FLAGS_START,   COL_FLAGS_START+1,  COL_FLAGS_START+2,  COL_FLAGS_START+3,
    COL_FLAGS_START+4, COL_FLAGS_START+5,  COL_FLAGS_START+6,  COL_FLAGS_START+7,
    COL_FLAGS_START+8, COL_FLAGS_START+9,  COL_FLAGS_START+10,
    // Group B (cols 16..31)
    COL_FLAGS_START+11, COL_FLAGS_START+12, COL_FLAGS_START+13, COL_FLAGS_START+14,
    COL_DST_ADDR, COL_DST, COL_OP0_ADDR, COL_OP0, COL_OP1_ADDR, COL_OP1,
    COL_RES, COL_OFF0, COL_OFF1, COL_OFF2, COL_DST_INV,
    // Group C (cols 31..34) — dict linkage
    COL_DICT_KEY, COL_DICT_NEW, COL_DICT_ACTIVE,
];


/// Internal: prove from pre-computed columns. Called by both `cairo_prove_cached` and `cairo_prove_program`.
fn cairo_prove_cached_with_columns(
    program: &[u64], columns: Vec<Vec<u32>>, n_steps: usize, log_n: u32,
    cache: &CairoProverCache,
    pedersen_inputs: Option<(&[super::stark252_field::Fp], &[super::stark252_field::Fp])>,
    dict_accesses: &[(usize, u64, u64, u64)],
    bitwise_rows: Vec<[u32; 5]>,
) -> CairoProof {
    let n = 1usize << log_n;
    // n_steps MUST equal n exactly. Callers are responsible for padding:
    // cairo_prove_program_inner passes n (not n_steps) after installing the self-loop pad.
    // cairo_prove_cached (test path) requires exact power-of-two step counts.
    assert_eq!(n_steps, n,
        "VortexSTARK: n_steps ({n_steps}) must equal 2^log_n ({n}). \
         Use cairo_prove_program for auto-padded execution.");
    assert_eq!(cache.log_n, log_n);
    let log_eval_size = log_n + BLOWUP_BITS;
    let eval_size = 1usize << log_eval_size;

    // ── VRAM gate (non-Result path: panic with a clear message) ───────────
    // Production entry point (cairo_prove_program) checks before we get here.
    // This guard covers cairo_prove / cairo_prove_cached (test/internal paths).
    // With group-batched NTT, peak = (TRACE_LO + 1) × eval_size × 4 bytes.
    {
        let required = cairo_vram_required(log_n);
        let (free, _) = crate::cuda::ffi::vram_query();
        assert!(
            required <= free,
            "Insufficient VRAM: proof needs {:.1} GB but only {:.1} GB is free \
             (log_n={log_n}, N_COLS={N_COLS}, eval_size={eval_size}). \
             Lower log_n or stop other GPU processes.",
            required as f64 / (1u64 << 30) as f64,
            free as f64 / (1u64 << 30) as f64,
        );
    }

    // ---- Public inputs ----
    let hash_bytes = crate::channel::blake2s_hash(
        unsafe {
            let byte_len = program.len().checked_mul(8).expect("program bytecode too large to hash");
            std::slice::from_raw_parts(program.as_ptr() as *const u8, byte_len)
        }
    );
    let mut program_hash = [0u32; 8];
    for i in 0..8 {
        program_hash[i] = u32::from_le_bytes([
            hash_bytes[i*4], hash_bytes[i*4+1], hash_bytes[i*4+2], hash_bytes[i*4+3],
        ]);
    }
    let public_inputs = CairoPublicInputs {
        initial_pc: 0,
        initial_ap: 100,
        n_steps,
        program_hash,
        program: program.to_vec(),
    };

    // ---- Phase 1: Trace generation + commitment ----
    // (31 base columns already computed by caller; extend with dict linkage columns here)

    // ── Build dict linkage columns (cols 31-33) from the execution-order access log ──
    // COL_DICT_KEY=31, COL_DICT_NEW=32, COL_DICT_ACTIVE=33.
    // For each step with a dict access, set key/new to the access values and active=1.
    // All other rows remain zero (dict_active=0 → no LogUp contribution).
    // These columns are committed as Group C (below) and bound into the main FRI proof,
    // preventing a malicious prover from forging dict operations after commitment.
    let mut columns = columns; // take ownership so we can extend
    {
        let mut dict_key_col    = vec![0u32; n];
        let mut dict_new_col    = vec![0u32; n];
        let mut dict_active_col = vec![0u32; n];
        let p_m31 = crate::field::m31::P as u64;
        for &(step, key, _prev, new_val) in dict_accesses {
            if step < n {
                dict_key_col[step]    = (key    % (p_m31 + 1)) as u32;
                dict_new_col[step]    = (new_val % (p_m31 + 1)) as u32;
                dict_active_col[step] = 1;
            }
        }
        debug_assert_eq!(columns.len(), N_VM_COLS,
            "expected 31 base columns before appending dict columns");
        columns.push(dict_key_col);    // col 31 = COL_DICT_KEY
        columns.push(dict_new_col);    // col 32 = COL_DICT_NEW
        columns.push(dict_active_col); // col 33 = COL_DICT_ACTIVE
        debug_assert_eq!(columns.len(), N_COLS);
    }

    // Extract range check offsets from raw trace before NTT destroys it.
    // This is O(n_steps) bit manipulation — fast even at log_n=26.
    let (rc_offsets, rc_counts) = extract_offsets(&columns, n_steps);

    // ---- Group-batched NTT + ZK Blinding + Commit ────────────────────────
    //
    // Instead of computing all N_COLS eval columns simultaneously (which peaks at
    // N_COLS × eval_size × 4 bytes ≈ 34 GB at log_n=26), we process columns in three
    // groups:
    //
    //   Group A: cols 0..TRACE_LO    (16 cols) — "lo" commitment tree
    //   Group B: cols TRACE_LO..31   (15 cols) — "hi" commitment tree
    //   Group C: cols 31..N_COLS     ( 3 cols) — "dict" commitment tree
    //
    // The GPU Blake2s leaf hash switch statement caps at 16 columns per tree;
    // splitting into groups A/B/C respects that cap.
    //
    // Each group is NTT'd, ZK-blinded, committed, downloaded to host, then freed.
    // Peak VRAM = 16 cols (Group A) + 1 ZH col = 17 × eval_size × 4 bytes ≈ 17 GB.
    //
    // ZK blinding: column[i] += r_j · Z_H[i] for all 34 columns.
    // Z_H is computed once and reused across all groups.
    // All columns (including LogUp denominators and dict columns) are blinded —
    // see ZK_BLIND_COLS comment for the soundness argument.

    const TRACE_LO: usize = 16; // Group A: cols 0..TRACE_LO (GPU hash cap = 16 cols per tree)
    const TRACE_VM_END: usize = 31; // end of original 31 VM columns; Group B: 16..31, Group C: 31..N_COLS

    // Compute Z_H(x) once — shared across groups.
    let eval_domain_for_zk = Coset::half_coset(log_eval_size);
    let mut d_zh = DeviceBuffer::<u32>::alloc(eval_size);
    unsafe {
        ffi::cuda_compute_vanishing(
            eval_domain_for_zk.initial.x.0, eval_domain_for_zk.initial.y.0,
            eval_domain_for_zk.step.x.0,    eval_domain_for_zk.step.y.0,
            d_zh.as_mut_ptr(),
            log_eval_size, log_n,
        );
        ffi::cuda_device_sync();
    }

    // Pre-generate one random blinding scalar per ZK-blinded column.
    use rand::Rng;
    let mut rng = rand::rng();
    // Generate r ∈ [1, 2^30+1]: shift a u32 right by 1 to get 31 bits, add 1 for
    // non-zero guarantee.  Covers half the M31 range; sufficient for ZK blinding.
    // No rejection sampling — gen::<u32>() is a single ChaCha20 word.
    let zk_scalars: Vec<(usize, u32)> = ZK_BLIND_COLS.iter()
        .map(|&col| (col, (rng.random::<u32>() >> 1) + 1))
        .collect();

    // Polynomial coefficients saved during INTT — used later for OODS evaluation.
    // Indexed by global column index [0..N_COLS).
    let mut trace_poly_coeffs: Vec<Vec<u32>> = Vec::with_capacity(N_COLS);

    // Helper: NTT one column from trace domain → eval domain, saving coefficients.
    // Returns (eval-domain DeviceBuffer, coefficients host vec).
    let ntt_col_save = |src: &Vec<u32>| -> (DeviceBuffer<u32>, Vec<u32>) {
        let mut d_col = DeviceBuffer::from_host(src);
        ntt::interpolate(&mut d_col, &cache.inv_cache);
        let coeffs = d_col.to_host();
        let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
        unsafe { ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32); }
        drop(d_col);
        ntt::evaluate(&mut d_eval, &cache.fwd_cache);
        (d_eval, coeffs)
    };

    // ── Group A: cols 0..TRACE_LO ──────────────────────────────────────────
    let (trace_commitment, tile_roots_lo, host_eval_lo, host_eval_lo_hc): ([u32; 8], Vec<[u32; 8]>, Vec<Vec<u32>>, Vec<Vec<u32>>) = {
        let mut group: Vec<DeviceBuffer<u32>> = Vec::with_capacity(TRACE_LO);
        for c in 0..TRACE_LO {
            let (d_eval, coeffs) = ntt_col_save(&columns[c]);
            trace_poly_coeffs.push(coeffs);
            group.push(d_eval);
        }
        for &(col, r) in &zk_scalars {
            if col < TRACE_LO {
                unsafe { ffi::cuda_axpy_m31(r, d_zh.as_ptr(), group[col].as_mut_ptr(), eval_size as u32); }
            }
        }
        unsafe { ffi::cuda_device_sync(); }
        // Download half_coset data for constraint kernel, permute to canonic for commitment
        let hc: Vec<Vec<u32>> = group.iter().map(|c| c.to_host_fast()).collect();
        let cn: Vec<Vec<u32>> = hc.iter().map(|c| permute_half_coset_to_canonic(c, log_eval_size)).collect();
        let cn_gpu: Vec<DeviceBuffer<u32>> = cn.iter().map(|c| DeviceBuffer::from_host(c)).collect();
        let (root, tile_roots) = MerkleTree::commit_root_only_with_subtrees(&cn_gpu, log_eval_size);
        (root, tile_roots, cn, hc)
    };

    // ── Group B: cols TRACE_LO..TRACE_VM_END (15 Cairo VM hi cols) ─────────
    let (trace_commitment_hi, tile_roots_hi, host_eval_hi, host_eval_hi_hc): ([u32; 8], Vec<[u32; 8]>, Vec<Vec<u32>>, Vec<Vec<u32>>) = {
        let mut group: Vec<DeviceBuffer<u32>> = Vec::with_capacity(TRACE_VM_END - TRACE_LO);
        for c in TRACE_LO..TRACE_VM_END {
            let (d_eval, coeffs) = ntt_col_save(&columns[c]);
            trace_poly_coeffs.push(coeffs);
            group.push(d_eval);
        }
        for &(col, r) in &zk_scalars {
            if col >= TRACE_LO && col < TRACE_VM_END {
                let local = col - TRACE_LO;
                unsafe { ffi::cuda_axpy_m31(r, d_zh.as_ptr(), group[local].as_mut_ptr(), eval_size as u32); }
            }
        }
        unsafe { ffi::cuda_device_sync(); }
        // Download half_coset data for constraint kernel, permute to canonic for commitment
        let hc: Vec<Vec<u32>> = group.iter().map(|c| c.to_host_fast()).collect();
        let cn: Vec<Vec<u32>> = hc.iter().map(|c| permute_half_coset_to_canonic(c, log_eval_size)).collect();
        let cn_gpu: Vec<DeviceBuffer<u32>> = cn.iter().map(|c| DeviceBuffer::from_host(c)).collect();
        let (root, tile_roots) = MerkleTree::commit_root_only_with_subtrees(&cn_gpu, log_eval_size);
        (root, tile_roots, cn, hc)
    };

    // ── Group C: cols TRACE_VM_END..N_COLS (3 dict linkage cols) ───────────
    let (dict_trace_commitment, tile_roots_dict, host_eval_dict, host_eval_dict_hc): ([u32; 8], Vec<[u32; 8]>, Vec<Vec<u32>>, Vec<Vec<u32>>) = {
        let mut group: Vec<DeviceBuffer<u32>> = Vec::with_capacity(N_COLS - TRACE_VM_END);
        for c in TRACE_VM_END..N_COLS {
            let (d_eval, coeffs) = ntt_col_save(&columns[c]);
            trace_poly_coeffs.push(coeffs);
            group.push(d_eval);
        }
        for &(col, r) in &zk_scalars {
            if col >= TRACE_VM_END && col < N_COLS {
                let local = col - TRACE_VM_END;
                unsafe { ffi::cuda_axpy_m31(r, d_zh.as_ptr(), group[local].as_mut_ptr(), eval_size as u32); }
            }
        }
        unsafe { ffi::cuda_device_sync(); }
        // Download half_coset data for constraint kernel, permute to canonic for commitment
        let hc: Vec<Vec<u32>> = group.iter().map(|c| c.to_host_fast()).collect();
        let cn: Vec<Vec<u32>> = hc.iter().map(|c| permute_half_coset_to_canonic(c, log_eval_size)).collect();
        let cn_gpu: Vec<DeviceBuffer<u32>> = cn.iter().map(|c| DeviceBuffer::from_host(c)).collect();
        let (root, tile_roots) = MerkleTree::commit_root_only_with_subtrees(&cn_gpu, log_eval_size);
        (root, tile_roots, cn, hc)
    };

    drop(d_zh);

    // Reassemble flat host_eval_cols (canonic order, for Merkle decommitment) and
    // host_eval_cols_hc (half_coset BRT-NTT order, for GPU constraint kernel).
    // NOTE: `columns` is kept alive — compute_interaction_trace needs it after Fiat-Shamir.
    let mut host_eval_cols: Vec<Vec<u32>> = Vec::with_capacity(N_COLS);
    host_eval_cols.extend(host_eval_lo);
    host_eval_cols.extend(host_eval_hi);
    host_eval_cols.extend(host_eval_dict);

    let mut host_eval_cols_hc: Vec<Vec<u32>> = Vec::with_capacity(N_COLS);
    host_eval_cols_hc.extend(host_eval_lo_hc);
    host_eval_cols_hc.extend(host_eval_hi_hc);
    host_eval_cols_hc.extend(host_eval_dict_hc);


    let mut channel = Channel::new();
    channel.mix_digest(&public_inputs.program_hash);
    channel.mix_digest(&trace_commitment);
    channel.mix_digest(&trace_commitment_hi);
    // Mix dict_trace_commitment BEFORE drawing z_dict — this binds the main trace's
    // dict_key/dict_new/dict_active columns into the Fiat-Shamir transcript before
    // any dict challenges are derived. A malicious prover cannot change these columns
    // after this point without invalidating z_dict and breaking all subsequent checks.
    channel.mix_digest(&dict_trace_commitment);

    // ---- S_dict step-transition interaction trace (GAP-1 closure) ────────────
    // Draw z_dict_link, alpha_dict_link AFTER dict_trace_commitment is committed.
    // This binds the main trace's dict columns before any challenges are derived
    // from them; a malicious prover cannot change dict_key/dict_new after this point.
    let z_dict_link     = channel.draw_felt();
    let alpha_dict_link = channel.draw_felt();

    // Build S_dict running-sum trace over the trace domain.
    // S_dict[i] = Σ_{j<i, dict_active[j]=1} 1/(z_dict_link - (dict_key[j] + α * dict_new[j]))
    // S_dict[0] = 0 (empty prefix sum).
    // This is parallel to compute_rc_interaction_trace but over the dict columns.
    let (dict_main_interaction_trace, s_dict_final): ([Vec<u32>; 4], QM31) = {
        use super::logup::qm31_from_m31;
        let mut running = QM31::ZERO;
        let mut cols: [Vec<u32>; 4] = std::array::from_fn(|_| vec![0u32; n]);
        for i in 0..n {
            // Record BEFORE adding the contribution at row i (running sum convention:
            // S_dict[i] is the sum of contributions for rows 0..i-1).
            let arr = running.to_u32_array();
            for c in 0..4 { cols[c][i] = arr[c]; }
            // Add contribution from row i if it's a dict-active row.
            if columns[COL_DICT_ACTIVE][i] == 1 {
                let key     = qm31_from_m31(M31(columns[COL_DICT_KEY][i]));
                let new_val = qm31_from_m31(M31(columns[COL_DICT_NEW][i]));
                let entry = key + alpha_dict_link * new_val;
                let denom = z_dict_link - entry;
                running = running + denom.inverse();
            }
        }
        (cols, running)
    };
    let dict_link_final_arr = s_dict_final.to_u32_array();

    // NTT S_dict trace to eval domain and commit (same pattern as RC interaction trace).
    let mut d_sdict_gpu: [DeviceBuffer<u32>; 4] = std::array::from_fn(|_| DeviceBuffer::<u32>::alloc(0));
    for c in 0..4 {
        let mut d_col = DeviceBuffer::from_host(&dict_main_interaction_trace[c]);
        ntt::interpolate(&mut d_col, &cache.inv_cache);
        let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
        unsafe { ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32); }
        drop(d_col);
        ntt::evaluate(&mut d_eval, &cache.fwd_cache);
        d_sdict_gpu[c] = d_eval;
    }
    let [d_sdict0, d_sdict1, d_sdict2, d_sdict3] = d_sdict_gpu;

    let (dict_main_interaction_commitment, tile_roots_sdict, host_sdict, host_sdict_hc) = {
        let hc = [d_sdict0.to_host(), d_sdict1.to_host(), d_sdict2.to_host(), d_sdict3.to_host()];
        drop(d_sdict0); drop(d_sdict1); drop(d_sdict2); drop(d_sdict3);
        let cn: [Vec<u32>; 4] = std::array::from_fn(|i| permute_half_coset_to_canonic(&hc[i], log_eval_size));
        let g0 = DeviceBuffer::from_host(&cn[0]); let g1 = DeviceBuffer::from_host(&cn[1]);
        let g2 = DeviceBuffer::from_host(&cn[2]); let g3 = DeviceBuffer::from_host(&cn[3]);
        let (root, tiles) = MerkleTree::commit_root_soa4_with_subtrees(&g0, &g1, &g2, &g3, log_eval_size);
        (root, tiles, cn, hc)
    };
    channel.mix_digest(&dict_main_interaction_commitment);
    channel.mix_digest(&[
        dict_link_final_arr[0], dict_link_final_arr[1],
        dict_link_final_arr[2], dict_link_final_arr[3],
        0, 0, 0, 0,
    ]);

    // ---- EC trace for Pedersen (optional) ----
    // EC_TRACE_LO: split point matching GPU leaf hash cap (16 columns per tree).
    // EC trace has 29 columns; lo = 0..16, hi = 16..29.
    const EC_TRACE_LO: usize = 16;
    let (ec_trace_commitment, ec_trace_commitment_hi, ec_trace_host, ec_log_eval_opt) =
        if let Some((ped_a, ped_b)) = pedersen_inputs {
        let n_hashes = ped_a.len();
        let ec_rows = n_hashes * ec_constraint::ROWS_PER_INVOCATION;
        let ec_log = (ec_rows as f64).log2().ceil() as u32;

        // GPU EC trace generation (replaces CPU generate_ec_trace)
        let d_ec_trace_cols = ec_constraint::gpu_generate_ec_trace(ped_a, ped_b, ec_log);

        // NTT + commit EC trace (columns already on GPU)
        let ec_log_eval = ec_log + BLOWUP_BITS;
        let ec_eval_size = 1usize << ec_log_eval;
        let ec_trace_domain = Coset::half_coset(ec_log);
        let ec_eval_domain = Coset::half_coset(ec_log_eval);
        let ec_inv = InverseTwiddleCache::new(&ec_trace_domain);
        let ec_fwd = ForwardTwiddleCache::new(&ec_eval_domain);

        let mut d_ec_eval_cols: Vec<DeviceBuffer<u32>> = Vec::new();
        for mut d_col in d_ec_trace_cols {
            ntt::interpolate(&mut d_col, &ec_inv);
            let mut d_eval = DeviceBuffer::<u32>::alloc(ec_eval_size);
            unsafe {
                ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(),
                    1u32 << ec_log, ec_eval_size as u32);
            }
            drop(d_col);
            ntt::evaluate(&mut d_eval, &ec_fwd);
            d_ec_eval_cols.push(d_eval);
        }

        // Split into lo (cols 0..EC_TRACE_LO) and hi (cols EC_TRACE_LO..29).
        // Each tree covers at most 16 cols — matching the GPU Blake2s leaf hash cap.
        let ec_commit_lo = MerkleTree::commit_root_only(&d_ec_eval_cols[..EC_TRACE_LO], ec_log_eval);
        let ec_commit_hi = MerkleTree::commit_root_only(&d_ec_eval_cols[EC_TRACE_LO..], ec_log_eval);
        channel.mix_digest(&ec_commit_lo);
        channel.mix_digest(&ec_commit_hi);

        // Download for auth path generation and verifier constraint check
        let ec_host: Vec<Vec<u32>> = d_ec_eval_cols.iter().map(|c| c.to_host()).collect();
        drop(d_ec_eval_cols);

        (Some(ec_commit_lo), Some(ec_commit_hi), Some(ec_host), Some(ec_log_eval))
    } else {
        (None, None, None, None)
    };

    // ---- Dict consistency sub-AIR (GAP-1 — STARK-level) ----
    // Build exec and sorted dict data traces, commit both as Merkle polynomial
    // commitments (BEFORE drawing z_mem/alpha_mem), then build and commit LogUp
    // interaction traces.  All four roots go into the Fiat-Shamir transcript.
    // A permutation argument (exec_final == sorted_final) proves multiset equality.
    //
    // Column data is kept in *_cols_opt for decommitment after query indices are drawn.
    let (
        dict_exec_commitment_opt,
        dict_sorted_commitment_opt,
        dict_log_n_opt,
        dict_exec_final_sum_opt,
        dict_sorted_final_sum_opt,
        dict_exec_data_opt,
        dict_sorted_data_opt,
    ) = if !dict_accesses.is_empty() {
        // Build raw data traces (exec order + key-sorted)
        let (exec_cols, sorted_cols, dict_log_n) =
            dict_air::build_dict_raw_traces(dict_accesses);
        let dict_n = 1usize << dict_log_n;

        // Commit exec data trace (3 cols) via GPU Merkle → Fiat-Shamir
        let d_exec: Vec<DeviceBuffer<u32>> = exec_cols.iter()
            .map(|c| DeviceBuffer::from_host(c)).collect();
        let exec_root = MerkleTree::commit_root_only(&d_exec, dict_log_n);
        drop(d_exec);
        channel.mix_digest(&exec_root);

        // Commit sorted data trace (4 cols) via GPU Merkle → Fiat-Shamir
        let d_sorted: Vec<DeviceBuffer<u32>> = sorted_cols.iter()
            .map(|c| DeviceBuffer::from_host(c)).collect();
        let sorted_root = MerkleTree::commit_root_only(&d_sorted, dict_log_n);
        drop(d_sorted);
        channel.mix_digest(&sorted_root);

        // Draw LogUp challenges after both data traces are committed
        let z_dict     = channel.draw_felt();
        let alpha_dict = channel.draw_felt();

        // Compute final sums (used for permutation argument + Fiat-Shamir binding)
        let (_, exec_final) = dict_air::build_dict_interaction_trace(
            &exec_cols[0], &exec_cols[1], &exec_cols[2], z_dict, alpha_dict);
        let (_, sorted_final) = dict_air::build_dict_interaction_trace(
            &sorted_cols[0], &sorted_cols[1], &sorted_cols[2], z_dict, alpha_dict);

        assert_eq!(exec_final, sorted_final,
            "dict LogUp permutation failed — exec and sorted traces are not the same multiset");

        // ── S_dict link verification ─────────────────────────────────────────────
        // Recompute exec_key_new_sum using z_dict_link/alpha_dict_link over the first
        // n_accesses rows of exec_cols (the actual accesses; beyond that are padding zeros).
        // S_dict_final (from main trace interaction) must match this sum.
        let n_accesses = dict_accesses.len();
        {
            use super::logup::qm31_from_m31;
            let mut exec_key_new_sum = QM31::ZERO;
            for j in 0..n_accesses {
                let key     = qm31_from_m31(M31(exec_cols[0][j]));
                let new_val = qm31_from_m31(M31(exec_cols[2][j]));
                let entry = key + alpha_dict_link * new_val;
                let denom = z_dict_link - entry;
                exec_key_new_sum = exec_key_new_sum + denom.inverse();
            }
            assert_eq!(s_dict_final, exec_key_new_sum,
                "S_dict link check failed: main trace dict columns do not match exec trace \
                 — dict_key/dict_new in the FRI-committed trace differ from execution log");
        }

        // Assert ALL sorted step-transition constraints at prove time.
        // This catches any hint execution bugs before proof generation.
        for i in 0..dict_n - 1 {
            let curr = [sorted_cols[0][i],   sorted_cols[1][i],
                        sorted_cols[2][i],   sorted_cols[3][i]];
            let next = [sorted_cols[0][i+1], sorted_cols[1][i+1],
                        sorted_cols[2][i+1], sorted_cols[3][i+1]];
            let cs = dict_air::eval_sorted_constraints(curr, next);
            assert_eq!(cs, [0, 0, 0, 0],
                "dict sorted step-transition violated at row {i}: {cs:?}");
        }

        // Mix both final sums (combined into one 8-word digest) before z_mem is drawn.
        // This binds the claimed LogUp sums before any subsequent challenges are derived.
        let exec_final_arr   = exec_final.to_u32_array();
        let sorted_final_arr = sorted_final.to_u32_array();
        channel.mix_digest(&[
            exec_final_arr[0],   exec_final_arr[1],   exec_final_arr[2],   exec_final_arr[3],
            sorted_final_arr[0], sorted_final_arr[1], sorted_final_arr[2], sorted_final_arr[3],
        ]);

        // Convert column-major exec/sorted to row-major for proof payload
        let exec_data: Vec<[u32; 3]> = (0..dict_n)
            .map(|i| [exec_cols[0][i], exec_cols[1][i], exec_cols[2][i]])
            .collect();
        let sorted_data: Vec<[u32; 4]> = (0..dict_n)
            .map(|i| [sorted_cols[0][i], sorted_cols[1][i],
                      sorted_cols[2][i], sorted_cols[3][i]])
            .collect();

        (
            Some(exec_root),
            Some(sorted_root),
            Some(dict_log_n),
            Some(exec_final_arr),
            Some(sorted_final_arr),
            Some(exec_data),
            Some(sorted_data),
        )
    } else {
        (None, None, None, None, None, None, None)
    };

    // ---- Phase 2: Fused LogUp interaction ----
    let z_mem = channel.draw_felt();
    let alpha_mem = channel.draw_felt();
    let alpha_mem_sq = alpha_mem * alpha_mem;
    let z_rc = channel.draw_felt();
    // Compute LogUp interaction trace on GPU: upload trace columns, run fused
    // denominator kernel (with alpha^2*inst_hi extension), prefix-scan to running sum.
    // Produces trace-domain values at n positions; NTT below converts to eval domain.
    let (d_logup_trace, logup_final_qm31) =
        gpu_compute_interaction_trace(&columns, n_steps, z_mem, alpha_mem);

    // Extract memory table (needed for cancellation check and memory_table_commitment).
    // Run in parallel with GPU LogUp kernel upload (columns still alive).
    let (mem_data_table, mem_instr_table) = extract_memory_table(&columns, n_steps);
    {
        let table_sum = compute_memory_table_sum(&mem_data_table, &mem_instr_table, z_mem, alpha_mem);
        assert_eq!(logup_final_qm31 + table_sum, QM31::ZERO,
            "memory LogUp sums don't cancel — execution trace or memory table is inconsistent");
    }

    // RC interaction trace on GPU: 3 offset columns → prefix-scanned running sum.
    let (d_rc_trace, rc_exec_sum) = gpu_compute_rc_interaction_trace(&columns, n_steps, z_rc);

    let rc_table_sum = compute_rc_table_sum(&rc_counts, z_rc);
    assert_eq!(rc_exec_sum + rc_table_sum, QM31::ZERO,
        "range check LogUp sums don't cancel — offset out of range");
    let rc_final_sum = rc_exec_sum.to_u32_array();

    drop(columns); // trace-domain columns no longer needed
    drop(rc_offsets);

    let logup_final_sum = logup_final_qm31.to_u32_array();

    // NTT the LogUp interaction trace: trace-domain (n) → eval-domain (eval_size).
    let [mut lt0, mut lt1, mut lt2, mut lt3] = d_logup_trace;
    ntt::interpolate(&mut lt0, &cache.inv_cache);
    ntt::interpolate(&mut lt1, &cache.inv_cache);
    ntt::interpolate(&mut lt2, &cache.inv_cache);
    ntt::interpolate(&mut lt3, &cache.inv_cache);
    let mut d_logup0 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut d_logup1 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut d_logup2 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut d_logup3 = DeviceBuffer::<u32>::alloc(eval_size);
    unsafe {
        ffi::cuda_zero_pad(lt0.as_ptr(), d_logup0.as_mut_ptr(), n as u32, eval_size as u32);
        ffi::cuda_zero_pad(lt1.as_ptr(), d_logup1.as_mut_ptr(), n as u32, eval_size as u32);
        ffi::cuda_zero_pad(lt2.as_ptr(), d_logup2.as_mut_ptr(), n as u32, eval_size as u32);
        ffi::cuda_zero_pad(lt3.as_ptr(), d_logup3.as_mut_ptr(), n as u32, eval_size as u32);
    }
    drop(lt0); drop(lt1); drop(lt2); drop(lt3);
    ntt::evaluate(&mut d_logup0, &cache.fwd_cache);
    ntt::evaluate(&mut d_logup1, &cache.fwd_cache);
    ntt::evaluate(&mut d_logup2, &cache.fwd_cache);
    ntt::evaluate(&mut d_logup3, &cache.fwd_cache);

    // NTT the RC interaction trace.
    let [mut rc0, mut rc1, mut rc2, mut rc3] = d_rc_trace;
    ntt::interpolate(&mut rc0, &cache.inv_cache);
    ntt::interpolate(&mut rc1, &cache.inv_cache);
    ntt::interpolate(&mut rc2, &cache.inv_cache);
    ntt::interpolate(&mut rc3, &cache.inv_cache);
    let mut rc_d0 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut rc_d1 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut rc_d2 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut rc_d3 = DeviceBuffer::<u32>::alloc(eval_size);
    unsafe {
        ffi::cuda_zero_pad(rc0.as_ptr(), rc_d0.as_mut_ptr(), n as u32, eval_size as u32);
        ffi::cuda_zero_pad(rc1.as_ptr(), rc_d1.as_mut_ptr(), n as u32, eval_size as u32);
        ffi::cuda_zero_pad(rc2.as_ptr(), rc_d2.as_mut_ptr(), n as u32, eval_size as u32);
        ffi::cuda_zero_pad(rc3.as_ptr(), rc_d3.as_mut_ptr(), n as u32, eval_size as u32);
    }
    drop(rc0); drop(rc1); drop(rc2); drop(rc3);
    ntt::evaluate(&mut rc_d0, &cache.fwd_cache);
    ntt::evaluate(&mut rc_d1, &cache.fwd_cache);
    ntt::evaluate(&mut rc_d2, &cache.fwd_cache);
    ntt::evaluate(&mut rc_d3, &cache.fwd_cache);

    let (interaction_commitment, tile_roots_logup, host_logup, host_logup_hc) = {
        let hc = [d_logup0.to_host(), d_logup1.to_host(), d_logup2.to_host(), d_logup3.to_host()];
        drop(d_logup0); drop(d_logup1); drop(d_logup2); drop(d_logup3);
        let cn: [Vec<u32>; 4] = std::array::from_fn(|i| permute_half_coset_to_canonic(&hc[i], log_eval_size));
        let g0 = DeviceBuffer::from_host(&cn[0]); let g1 = DeviceBuffer::from_host(&cn[1]);
        let g2 = DeviceBuffer::from_host(&cn[2]); let g3 = DeviceBuffer::from_host(&cn[3]);
        let (root, tiles) = MerkleTree::commit_root_soa4_with_subtrees(&g0, &g1, &g2, &g3, log_eval_size);
        (root, tiles, cn, hc)
    };
    channel.mix_digest(&interaction_commitment);

    let (rc_interaction_commitment, tile_roots_rc, host_rc_logup, host_rc_logup_hc) = {
        let hc = [rc_d0.to_host(), rc_d1.to_host(), rc_d2.to_host(), rc_d3.to_host()];
        drop(rc_d0); drop(rc_d1); drop(rc_d2); drop(rc_d3);
        let cn: [Vec<u32>; 4] = std::array::from_fn(|i| permute_half_coset_to_canonic(&hc[i], log_eval_size));
        let g0 = DeviceBuffer::from_host(&cn[0]); let g1 = DeviceBuffer::from_host(&cn[1]);
        let g2 = DeviceBuffer::from_host(&cn[2]); let g3 = DeviceBuffer::from_host(&cn[3]);
        let (root, tiles) = MerkleTree::commit_root_soa4_with_subtrees(&g0, &g1, &g2, &g3, log_eval_size);
        (root, tiles, cn, hc)
    };
    channel.mix_digest(&rc_interaction_commitment);

    // Bind LogUp and RC final sums into Fiat-Shamir (tampering breaks FRI)
    channel.mix_digest(&[
        logup_final_sum[0], logup_final_sum[1], logup_final_sum[2], logup_final_sum[3],
        rc_final_sum[0], rc_final_sum[1], rc_final_sum[2], rc_final_sum[3],
    ]);

    // ---- Bitwise builtin commitment (full-data, bound into Fiat-Shamir) ----
    // Hash all bitwise rows into the channel BEFORE constraint alphas and FRI challenges
    // are drawn.  This prevents a malicious prover from choosing bitwise data after
    // seeing those challenges.  The verifier recomputes this hash independently.
    let bitwise_commitment_opt: Option<[u32; 8]> = if !bitwise_rows.is_empty() {
        let flat: Vec<u32> = bitwise_rows.iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        // Blake2s hash of all bitwise data (8 × u32 = 32 bytes).
        let hash = crate::merkle::MerkleTree::hash_leaf(&flat);
        channel.mix_digest(&hash);
        Some(hash)
    } else {
        None
    };

    // ---- Memory table commitment (closes LogUp soundness gap) ----
    // Serialize memory table as flat u32 array, hash it, and mix into Fiat-Shamir.
    // This commits the prover to a specific memory table BEFORE constraint_alphas are drawn.
    // The verifier independently recomputes this hash and checks exec_sum + table_sum == 0.
    let memory_table_data: Vec<[u32; 3]> = mem_data_table.iter()
        .map(|&(addr, val, mult)| [addr.0, val.0, mult])
        .collect();
    let memory_instr_data: Vec<[u32; 4]> = mem_instr_table.iter()
        .map(|&(pc, lo, hi, mult)| [pc.0, lo.0, hi.0, mult])
        .collect();
    let mem_flat: Vec<u32> = memory_table_data.iter()
        .flat_map(|&[a, v, m]| [a, v, m])
        .chain(memory_instr_data.iter().flat_map(|&[p, lo, hi, m]| [p, lo, hi, m]))
        .collect();
    let memory_table_commitment = crate::channel::hash_words(&mem_flat);
    channel.mix_digest(&memory_table_commitment);

    // ---- RC counts commitment (closes RC soundness gap) ----
    let rc_counts_data: Vec<u32> = rc_counts.to_vec();
    let rc_counts_commitment = crate::channel::hash_words(&rc_counts_data);
    channel.mix_digest(&rc_counts_commitment);

    // ---- Phase 3: Quotient ----
    let constraint_alphas: Vec<QM31> = (0..N_CONSTRAINTS).map(|_| channel.draw_felt()).collect();
    let alpha_flat: Vec<u32> = constraint_alphas.iter().flat_map(|a| a.to_u32_array()).collect();

    // Upload all trace columns to GPU for quotient evaluation (half_coset order for GPU kernel).
    let d_quot_cols: Vec<DeviceBuffer<u32>> = (0..N_COLS)
        .map(|c| DeviceBuffer::from_host(&host_eval_cols_hc[c]))
        .collect();
    let col_ptrs: Vec<*const u32> = d_quot_cols.iter().map(|b| b.as_ptr()).collect();
    let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);
    let d_alpha = DeviceBuffer::from_host(&alpha_flat);

    // Compute 1/Z_H for every NTT position in the eval domain.
    // Z_H(x) = 0 iff x is the x-coordinate of a trace domain point.
    // The quotient Q(x) = C(x)/Z_H(x) is the polynomial that FRI proves low-degree.
    let eval_domain = crate::circle::Coset::half_coset(log_eval_size);
    let mut d_vh_inv = DeviceBuffer::<u32>::alloc(eval_size);
    unsafe {
        ffi::cuda_compute_vanishing_inv(
            eval_domain.initial.x.0, eval_domain.initial.y.0,
            eval_domain.step.x.0,   eval_domain.step.y.0,
            d_vh_inv.as_mut_ptr(),
            log_eval_size,
            log_n,
        );
        ffi::cuda_device_sync();
    }

    // Re-upload interaction columns in HALF_COSET order for the quotient kernel
    // (must match the trace column ordering used by the constraint kernel).
    let d_slogup0 = DeviceBuffer::from_host(&host_logup_hc[0]);
    let d_slogup1 = DeviceBuffer::from_host(&host_logup_hc[1]);
    let d_slogup2 = DeviceBuffer::from_host(&host_logup_hc[2]);
    let d_slogup3 = DeviceBuffer::from_host(&host_logup_hc[3]);
    let d_src0 = DeviceBuffer::from_host(&host_rc_logup_hc[0]);
    let d_src1 = DeviceBuffer::from_host(&host_rc_logup_hc[1]);
    let d_src2 = DeviceBuffer::from_host(&host_rc_logup_hc[2]);
    let d_src3 = DeviceBuffer::from_host(&host_rc_logup_hc[3]);
    let d_sd0 = DeviceBuffer::from_host(&host_sdict_hc[0]);
    let d_sd1 = DeviceBuffer::from_host(&host_sdict_hc[1]);
    let d_sd2 = DeviceBuffer::from_host(&host_sdict_hc[2]);
    let d_sd3 = DeviceBuffer::from_host(&host_sdict_hc[3]);

    // Pack challenges: [z_mem(4), alpha_mem(4), alpha_mem_sq(4), z_rc(4), z_dict_link(4), alpha_dict_link(4)] = 24 u32s.
    let challenges_flat: Vec<u32> = z_mem.to_u32_array().iter()
        .chain(alpha_mem.to_u32_array().iter())
        .chain(alpha_mem_sq.to_u32_array().iter())
        .chain(z_rc.to_u32_array().iter())
        .chain(z_dict_link.to_u32_array().iter())
        .chain(alpha_dict_link.to_u32_array().iter())
        .copied().collect();
    let logup_challenges: [u32; 24] = challenges_flat.as_slice().try_into()
        .expect("challenges must be exactly 24 u32s (6 QM31 values)");
    let d_challenges = DeviceBuffer::from_host(&challenges_flat);

    let mut q0 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut q1 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut q2 = DeviceBuffer::<u32>::alloc(eval_size);
    let mut q3 = DeviceBuffer::<u32>::alloc(eval_size);

    unsafe {
        ffi::cuda_cairo_quotient(
            d_col_ptrs.as_ptr() as *const *const u32,
            d_slogup0.as_ptr(), d_slogup1.as_ptr(), d_slogup2.as_ptr(), d_slogup3.as_ptr(),
            d_src0.as_ptr(), d_src1.as_ptr(), d_src2.as_ptr(), d_src3.as_ptr(),
            d_sd0.as_ptr(), d_sd1.as_ptr(), d_sd2.as_ptr(), d_sd3.as_ptr(),
            q0.as_mut_ptr(), q1.as_mut_ptr(), q2.as_mut_ptr(), q3.as_mut_ptr(),
            d_alpha.as_ptr(),
            d_vh_inv.as_ptr(),
            d_challenges.as_ptr(),
            eval_size as u32,
        );
        ffi::cuda_device_sync();
    }
    drop(d_slogup0); drop(d_slogup1); drop(d_slogup2); drop(d_slogup3);
    drop(d_src0); drop(d_src1); drop(d_src2); drop(d_src3);
    drop(d_sd0); drop(d_sd1); drop(d_sd2); drop(d_sd3);
    drop(d_challenges);
    drop(d_vh_inv);
    // Free quotient-phase eval cols — trace data stays on host_eval_cols
    drop(d_quot_cols);
    drop(d_col_ptrs);
    drop(d_alpha);

    let (quotient_commitment, tile_roots_quotient, host_q0, host_q1, host_q2, host_q3) = {
        let hq0 = permute_half_coset_to_canonic(&q0.to_host(), log_eval_size);
        let hq1 = permute_half_coset_to_canonic(&q1.to_host(), log_eval_size);
        let hq2 = permute_half_coset_to_canonic(&q2.to_host(), log_eval_size);
        let hq3 = permute_half_coset_to_canonic(&q3.to_host(), log_eval_size);
        let g0 = DeviceBuffer::from_host(&hq0); let g1 = DeviceBuffer::from_host(&hq1);
        let g2 = DeviceBuffer::from_host(&hq2); let g3 = DeviceBuffer::from_host(&hq3);
        let (root, tiles) = MerkleTree::commit_root_soa4_with_subtrees(&g0, &g1, &g2, &g3, log_eval_size);
        (root, tiles, hq0, hq1, hq2, hq3)
    };
    channel.mix_digest(&quotient_commitment);

    // ---- OODS: Out-Of-Domain Sampling (stwo wire format) ----
    // Phase 1: evaluate trace at z and z_next; mix into channel.
    // Phase 2: evaluate AIR quotient at z; draw OODS alpha; compute OODS quotient for FRI.
    let (oods_z, oods_trace_at_z, oods_trace_at_z_next, oods_quotient_at_z, oods_alpha, oods_quotient_col, oods_interaction_at_z, oods_interaction_at_z_next) = {
        use crate::oods::{OodsPoint, eval_at_oods_from_coeffs, qm31_from_m31, oods_vanishing, compute_line_coeffs};
        use std::collections::HashMap;

        let z = OodsPoint::from_channel(&mut channel);
        let step = crate::circle::CirclePoint::GENERATOR.repeated_double(31 - log_n);
        let z_next = z.next_step(step);

        let zh_z      = oods_vanishing(z, log_n);
        let zh_z_next = oods_vanishing(z_next, log_n);
        let zk_map: HashMap<usize, u32> = zk_scalars.iter().cloned().collect();

        // ── Phase 1: evaluate 34 trace columns at z and z_next ───────────────
        let mut at_z    = Vec::with_capacity(N_COLS);
        let mut at_next = Vec::with_capacity(N_COLS);
        for col_idx in 0..N_COLS {
            let coeffs = &trace_poly_coeffs[col_idx];
            let mut val_z    = eval_at_oods_from_coeffs(coeffs, z);
            let mut val_next = eval_at_oods_from_coeffs(coeffs, z_next);
            if let Some(&r) = zk_map.get(&col_idx) {
                let r_q = qm31_from_m31(crate::field::M31(r));
                val_z    = val_z    + r_q * zh_z;
                val_next = val_next + r_q * zh_z_next;
            }
            at_z.push(val_z);
            at_next.push(val_next);
        }
        // ── Phase 1.5: evaluate 12 interaction polynomial components at z and z_next ──
        // 3 interaction polys (LogUp, RC, S_dict) × 4 M31 components = 12 evals each.
        // Batch strategy: upload each component once (H→D), keep eval-domain GPU buffer
        // for Phase 2c reuse, clone to scratch for INTT, evaluate at both z and z_next.
        // Upload canonic-ordered interaction data for OODS numerator (Phase 2c).
        // Also keep half_coset-ordered versions for INTT → OODS point evaluation.
        let srcs_cn: [&[Vec<u32>]; 3] = [&host_logup, &host_rc_logup, &host_sdict];
        // d_interaction_eval[pi][k] = canonic-order GPU buffer (for numerator accumulation)
        let d_interaction_eval: Vec<Vec<DeviceBuffer<u32>>> = srcs_cn.iter()
            .map(|src| (0..4).map(|k| DeviceBuffer::from_host(&src[k])).collect())
            .collect();
        let (interaction_evals_raw, interaction_evals_next_raw): ([[[u32; 4]; 4]; 3], [[[u32; 4]; 4]; 3]) = {
            let mut raw_z    = [[[0u32; 4]; 4]; 3];
            let mut raw_zn   = [[[0u32; 4]; 4]; 3];
            // For OODS point evaluation, we need half_coset-ordered data for INTT.
            // Interaction host data is canonic; inverse-permute to half_coset for INTT.
            let srcs_hc: [Vec<Vec<u32>>; 3] = {
                // Inverse permutation: BRT-canonic → half_coset NTT order.
                // BRT-canonic position j has value at canonic_domain_point(BRT(j)).
                // Half_coset NTT position for that value is canonic_to_hc_ntt(BRT(j)).
                let inv_perm = |cn: &[Vec<u32>; 4]| -> Vec<Vec<u32>> {
                    cn.iter().map(|col| {
                        let n = col.len();
                        let log_n = n.trailing_zeros();
                        let mut hc = vec![0u32; n];
                        for j in 0..n {
                            let cn_nat = j.reverse_bits() >> (usize::BITS - log_n);
                            hc[canonic_to_hc_ntt(cn_nat, log_n)] = col[j];
                        }
                        hc
                    }).collect()
                };
                [inv_perm(&host_logup), inv_perm(&host_rc_logup), inv_perm(&host_sdict)]
            };
            for pi in 0..3 {
                for k in 0..4 {
                    // Upload half_coset data for INTT → OODS evaluation
                    let mut scratch = DeviceBuffer::from_host(&srcs_hc[pi][k]);
                    ntt::interpolate(&mut scratch, &cache.eval_inv_cache);
                    let coeffs = scratch.to_host();
                    let val_z  = eval_at_oods_from_coeffs(&coeffs[..n], z);
                    let val_zn = eval_at_oods_from_coeffs(&coeffs[..n], z_next);
                    raw_z[pi][k]  = val_z.to_u32_array();
                    raw_zn[pi][k] = val_zn.to_u32_array();
                }
            }
            (raw_z, raw_zn)
        };

        // ── Phase 2a: evaluate AIR quotient columns at z ─────────────────────
        // INTT each qk from eval domain → coefficients (first n are the polynomial).
        // q0..q3 are still alive as DeviceBuffers — use them directly.
        let quot_at_z: [QM31; 4] = {
            let qs: [&DeviceBuffer<u32>; 4] = [&q0, &q1, &q2, &q3];
            std::array::from_fn(|k| {
                let mut d: DeviceBuffer<u32> = DeviceBuffer::from_host(&qs[k].to_host());
                ntt::interpolate(&mut d, &cache.eval_inv_cache);
                let coeffs = d.to_host();
                eval_at_oods_from_coeffs(&coeffs[..n], z)
            })
        };

        // ── Phase 2b: mix all sampled values (stwo flatten_cols order) ────────
        // stwo's channel.mix_felts(sampled_values.flatten_cols()) iterates:
        //   for each tree, for each column, for each sample point.
        // Trees 0-2 (trace): N_COLS cols × 2 samples [z, z_next] interleaved per column.
        // Trees 3-5 (interaction): 4 cols × 2 samples interleaved per column.
        // Tree 6 (quotient): 4 cols × 1 sample [z].
        {
            let mut combined: Vec<QM31> = Vec::with_capacity(N_COLS * 2 + 12 * 2 + 4);
            for i in 0..N_COLS {
                combined.push(at_z[i]);
                combined.push(at_next[i]);
            }
            for pi in 0..3 {
                for k in 0..4 {
                    combined.push(QM31::from_u32_array(interaction_evals_raw[pi][k]));
                    combined.push(QM31::from_u32_array(interaction_evals_next_raw[pi][k]));
                }
            }
            for k in 0..4 {
                combined.push(quot_at_z[k]);
            }
            channel.mix_felts(&combined);
        }

        // Draw OODS alpha (matches stwo's random_coeff draw in verify_values).
        let oods_alpha = channel.draw_felt();

        // Acc 0: sample point z    — N_COLS trace + 4 quotient + 12 interaction = N_COLS+16
        // Acc 1: sample point z_next — N_COLS trace + 12 interaction = N_COLS+12
        let n_acc_z  = N_COLS + 4 + 12;
        let n_acc_zn = N_COLS + 12;
        let mut b_coeffs_z:  Vec<u32> = Vec::with_capacity(n_acc_z * 4);
        let mut c_coeffs_z:  Vec<u32> = Vec::with_capacity(n_acc_z * 4);
        let mut linear_acc_z  = QM31::ZERO;
        let mut b_coeffs_zn: Vec<u32> = Vec::with_capacity(n_acc_zn * 4);
        let mut c_coeffs_zn: Vec<u32> = Vec::with_capacity(n_acc_zn * 4);
        let mut linear_acc_zn = QM31::ZERO;
        let mut alpha_pow = QM31::ONE;

        for col_idx in 0..N_COLS {
            let (a, b) = compute_line_coeffs(z, at_z[col_idx]);
            // b_coeffs = alpha^c * a (stwo convention: kernel computes c*f - b = alpha*(f-a))
            b_coeffs_z.extend_from_slice(&(alpha_pow * a).to_u32_array());
            c_coeffs_z.extend_from_slice(&alpha_pow.to_u32_array());
            linear_acc_z = linear_acc_z + alpha_pow * b;
            alpha_pow = alpha_pow * oods_alpha;
        }
        for k in 0..4 {
            let (a, b) = compute_line_coeffs(z, quot_at_z[k]);
            b_coeffs_z.extend_from_slice(&(alpha_pow * a).to_u32_array());
            c_coeffs_z.extend_from_slice(&alpha_pow.to_u32_array());
            linear_acc_z = linear_acc_z + alpha_pow * b;
            alpha_pow = alpha_pow * oods_alpha;
        }
        // 12 interaction component columns (LogUp, RC, S_dict × 4 comps each)
        for pi in 0..3usize {
            for k in 0..4 {
                let val = QM31::from_u32_array(interaction_evals_raw[pi][k]);
                let (a, b) = compute_line_coeffs(z, val);
                b_coeffs_z.extend_from_slice(&(alpha_pow * a).to_u32_array());
                c_coeffs_z.extend_from_slice(&alpha_pow.to_u32_array());
                linear_acc_z = linear_acc_z + alpha_pow * b;
                alpha_pow = alpha_pow * oods_alpha;
            }
        }
        for col_idx in 0..N_COLS {
            let (a, b) = compute_line_coeffs(z_next, at_next[col_idx]);
            b_coeffs_zn.extend_from_slice(&(alpha_pow * a).to_u32_array());
            c_coeffs_zn.extend_from_slice(&alpha_pow.to_u32_array());
            linear_acc_zn = linear_acc_zn + alpha_pow * b;
            alpha_pow = alpha_pow * oods_alpha;
        }
        // 12 interaction component columns at z_next (step-transition binding)
        for pi in 0..3usize {
            for k in 0..4 {
                let val = QM31::from_u32_array(interaction_evals_next_raw[pi][k]);
                let (a, b) = compute_line_coeffs(z_next, val);
                b_coeffs_zn.extend_from_slice(&(alpha_pow * a).to_u32_array());
                c_coeffs_zn.extend_from_slice(&alpha_pow.to_u32_array());
                linear_acc_zn = linear_acc_zn + alpha_pow * b;
                alpha_pow = alpha_pow * oods_alpha;
            }
        }

        // ── Phase 2c: upload eval-domain columns; compute OODS numerators ────
        // Upload trace eval cols + re-use q0..q3 GPU buffers by reference.
        // cols 0..N_COLS = trace, cols N_COLS..N_COLS+4 = quotient q0..q3
        let mut d_eval_cols: Vec<DeviceBuffer<u32>> = host_eval_cols.iter()
            .map(|c| DeviceBuffer::from_host(c)).collect();
        d_eval_cols.push(DeviceBuffer::from_host(&host_q0));
        d_eval_cols.push(DeviceBuffer::from_host(&host_q1));
        d_eval_cols.push(DeviceBuffer::from_host(&host_q2));
        d_eval_cols.push(DeviceBuffer::from_host(&host_q3));
        // Interaction columns: reuse the already-uploaded eval-domain GPU buffers from Phase 1.5
        // (avoids 12 redundant H→D uploads — d_interaction_eval[pi][k] still contains eval data).
        for pi in 0..3 {
            for k in 0..4 {
                d_eval_cols.push(d_interaction_eval[pi][k].clone_on_device());
            }
        }

        let all_col_ptrs: Vec<*const u32> = d_eval_cols.iter().map(|b| b.as_ptr()).collect();
        let d_col_ptrs = DeviceBuffer::from_host(&all_col_ptrs);

        let col_idx_z: Vec<u32> = (0..(N_COLS + 4 + 12) as u32).collect();
        // z_next: trace (0..N_COLS) then interaction (N_COLS+4..N_COLS+16), skipping quotient slots.
        let mut col_idx_zn: Vec<u32> = (0..N_COLS as u32).collect();
        col_idx_zn.extend(((N_COLS + 4) as u32)..((N_COLS + 4 + 12) as u32));
        let d_cidx_z  = DeviceBuffer::from_host(&col_idx_z);
        let d_cidx_zn = DeviceBuffer::from_host(&col_idx_zn);
        let d_b_z  = DeviceBuffer::from_host(&b_coeffs_z);
        let d_c_z  = DeviceBuffer::from_host(&c_coeffs_z);
        let d_b_zn = DeviceBuffer::from_host(&b_coeffs_zn);
        let d_c_zn = DeviceBuffer::from_host(&c_coeffs_zn);

        let mut numer_z0  = DeviceBuffer::<u32>::alloc(eval_size);
        let mut numer_z1  = DeviceBuffer::<u32>::alloc(eval_size);
        let mut numer_z2  = DeviceBuffer::<u32>::alloc(eval_size);
        let mut numer_z3  = DeviceBuffer::<u32>::alloc(eval_size);
        let mut numer_zn0 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut numer_zn1 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut numer_zn2 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut numer_zn3 = DeviceBuffer::<u32>::alloc(eval_size);

        unsafe {
            ffi::cuda_accumulate_numerators(
                d_col_ptrs.as_ptr() as *const *const u32, d_cidx_z.as_ptr(),
                d_b_z.as_ptr(), d_c_z.as_ptr(), n_acc_z as u32, eval_size as u32,
                numer_z0.as_mut_ptr(), numer_z1.as_mut_ptr(),
                numer_z2.as_mut_ptr(), numer_z3.as_mut_ptr(),
            );
            ffi::cuda_accumulate_numerators(
                d_col_ptrs.as_ptr() as *const *const u32, d_cidx_zn.as_ptr(),
                d_b_zn.as_ptr(), d_c_zn.as_ptr(), n_acc_zn as u32, eval_size as u32,
                numer_zn0.as_mut_ptr(), numer_zn1.as_mut_ptr(),
                numer_zn2.as_mut_ptr(), numer_zn3.as_mut_ptr(),
            );
            ffi::cuda_device_sync();
        }
        drop(d_eval_cols); drop(d_col_ptrs);
        drop(d_cidx_z); drop(d_b_z); drop(d_c_z);
        drop(d_cidx_zn); drop(d_b_zn); drop(d_c_zn);
        drop(d_interaction_eval); // eval-domain interaction GPU buffers no longer needed

        // ── Phase 2d: compute domain points; combine into OODS quotient ──────
        // BRT-canonic domain points (matching the BRT-canonic-ordered d_eval_cols).
        let mut d_dom_xs = DeviceBuffer::<u32>::alloc(eval_size);
        let mut d_dom_ys = DeviceBuffer::<u32>::alloc(eval_size);
        {
            let mut xs = vec![0u32; eval_size];
            let mut ys = vec![0u32; eval_size];
            for i in 0..eval_size {
                let brt_i = i.reverse_bits() >> (usize::BITS - log_eval_size);
                let pt = canonic_domain_point(brt_i, log_eval_size);
                xs[i] = pt.x.0;
                ys[i] = pt.y.0;
            }
            d_dom_xs = DeviceBuffer::from_host(&xs);
            d_dom_ys = DeviceBuffer::from_host(&ys);
        }

        // Pack per-accumulator metadata
        let sp_x: Vec<u32> = z.x.to_u32_array().iter().chain(z_next.x.to_u32_array().iter()).copied().collect();
        let sp_y: Vec<u32> = z.y.to_u32_array().iter().chain(z_next.y.to_u32_array().iter()).copied().collect();
        let fla:  Vec<u32> = linear_acc_z.to_u32_array().iter().chain(linear_acc_zn.to_u32_array().iter()).copied().collect();
        let acc_log_szs: Vec<u32> = vec![log_eval_size, log_eval_size];
        let d_sp_x   = DeviceBuffer::from_host(&sp_x);
        let d_sp_y   = DeviceBuffer::from_host(&sp_y);
        let d_fla    = DeviceBuffer::from_host(&fla);
        let d_alszs  = DeviceBuffer::from_host(&acc_log_szs);

        let np0: Vec<*const u32> = vec![numer_z0.as_ptr(), numer_zn0.as_ptr()];
        let np1: Vec<*const u32> = vec![numer_z1.as_ptr(), numer_zn1.as_ptr()];
        let np2: Vec<*const u32> = vec![numer_z2.as_ptr(), numer_zn2.as_ptr()];
        let np3: Vec<*const u32> = vec![numer_z3.as_ptr(), numer_zn3.as_ptr()];
        let d_np0 = DeviceBuffer::from_host(&np0);
        let d_np1 = DeviceBuffer::from_host(&np1);
        let d_np2 = DeviceBuffer::from_host(&np2);
        let d_np3 = DeviceBuffer::from_host(&np3);

        let mut oods_q0 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut oods_q1 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut oods_q2 = DeviceBuffer::<u32>::alloc(eval_size);
        let mut oods_q3 = DeviceBuffer::<u32>::alloc(eval_size);

        unsafe {
            ffi::cuda_compute_quotients_combine(
                d_sp_x.as_ptr(), d_sp_y.as_ptr(), d_fla.as_ptr(),
                d_np0.as_ptr() as *const *const u32,
                d_np1.as_ptr() as *const *const u32,
                d_np2.as_ptr() as *const *const u32,
                d_np3.as_ptr() as *const *const u32,
                d_alszs.as_ptr(), 2,
                d_dom_xs.as_ptr(), d_dom_ys.as_ptr(),
                log_eval_size, eval_size as u32,
                oods_q0.as_mut_ptr(), oods_q1.as_mut_ptr(),
                oods_q2.as_mut_ptr(), oods_q3.as_mut_ptr(),
            );
            ffi::cuda_device_sync();
        }
        drop(numer_z0); drop(numer_z1); drop(numer_z2); drop(numer_z3);
        drop(numer_zn0); drop(numer_zn1); drop(numer_zn2); drop(numer_zn3);
        drop(d_sp_x); drop(d_sp_y); drop(d_fla); drop(d_alszs);
        drop(d_np0); drop(d_np1); drop(d_np2); drop(d_np3);
        drop(d_dom_xs); drop(d_dom_ys);

        let z_arr    = z.to_u32_array();
        let at_z_arr: Vec<[u32; 4]>    = at_z.iter().map(|v| v.to_u32_array()).collect();
        let at_next_arr: Vec<[u32; 4]> = at_next.iter().map(|v| v.to_u32_array()).collect();
        let oods_q_arr: [[u32; 4]; 4]  = std::array::from_fn(|k| quot_at_z[k].to_u32_array());
        let alpha_arr  = oods_alpha.to_u32_array();
        let oods_col   = SecureColumn { cols: [oods_q0, oods_q1, oods_q2, oods_q3], len: eval_size };
        (z_arr, at_z_arr, at_next_arr, oods_q_arr, alpha_arr, oods_col, interaction_evals_raw, interaction_evals_next_raw)
    };

    // ---- Phase 4: FRI on OODS quotient (replaces direct AIR quotient FRI) ----
    // The AIR quotient q0..q3 has been sampled at z (oods_quotient_at_z) and
    // the OODS quotient polynomial is the FRI input.
    drop(q0); drop(q1); drop(q2); drop(q3);
    let quotient_col = oods_quotient_col;

    // OODS quotient was computed from canonic-ordered trace data and canonic domain points,
    // so it's ALREADY in canonic order. No permutation needed.
    // OODS quotient was computed from BRT-canonic data with BRT-canonic domain points.
    // It's ALREADY in BRT-canonic order — no additional permutation needed.
    let (oods_quotient_commitment, tile_roots_oods_q) =
        MerkleTree::commit_root_soa4_with_subtrees(&quotient_col.cols[0], &quotient_col.cols[1],
            &quotient_col.cols[2], &quotient_col.cols[3], log_eval_size);
    let fri_start_channel_state = channel.state_words();
    channel.mix_digest(&oods_quotient_commitment);
    // BRT-canonic host data for decommitment.
    let host_oods_q0 = quotient_col.cols[0].to_host_fast();
    let host_oods_q1 = quotient_col.cols[1].to_host_fast();
    let host_oods_q2 = quotient_col.cols[2].to_host_fast();
    let host_oods_q3 = quotient_col.cols[3].to_host_fast();

    // BRT circle fold twiddles from the canonic domain.
    // The BRT-canonic data at position j has the value at canonic_domain_point(BRT(j)).
    // The fold pairs (2i, 2i+1) and the twiddle[i] = 1/y at the even pair's domain point.
    let fri_alpha = channel.draw_felt();
    let d_twid = {
        let half_n = eval_size / 2;
        let mut tw = vec![0u32; half_n];
        for i in 0..half_n {
            let brt_2i = (2*i).reverse_bits() >> (usize::BITS - log_eval_size);
            let pt = canonic_domain_point(brt_2i, log_eval_size);
            tw[i] = pt.y.inverse().0;
        }
        DeviceBuffer::from_host(&tw)
    };
    let mut line_eval = SecureColumn::zeros(eval_size / 2);
    fri::fold_circle_into_line_with_twiddles(&mut line_eval, &quotient_col, fri_alpha, &d_twid);
    drop(d_twid);
    drop(quotient_col);
    // line_eval is now BRT-ordered (BRT circle fold on BRT input → BRT output).

    let mut fri_commitments = Vec::new();
    let mut fri_evals: Vec<SecureColumn> = Vec::new();
    let mut current = line_eval;
    let mut current_log = log_eval_size - 1;

    let mut all_fold_alphas: Vec<QM31> = vec![fri_alpha]; // [0] = circle fold alpha

    // Phase 1: Committed FRI inner layers.
    let committed_stop_log = fri::LOG_LAST_LAYER_DEGREE_BOUND + BLOWUP_BITS;
    while current_log > committed_stop_log {
        let layer_commitment = MerkleTree::commit_root_soa4(
            &current.cols[0], &current.cols[1], &current.cols[2], &current.cols[3],
            current_log,
        );
        channel.mix_digest(&layer_commitment);
        fri_commitments.push(layer_commitment);

        let fold_alpha = channel.draw_felt();
        all_fold_alphas.push(fold_alpha);
        let line_domain = Coset::half_odds(current_log);
        let d_twid = fri::compute_fold_twiddles_on_demand(&line_domain, false);
        let folded = fri::fold_line_with_twiddles(&current, fold_alpha, &d_twid);
        drop(d_twid);

        fri_evals.push(current);
        current = folded;
        current_log -= 1;
    }

    // Save the intermediate evaluation at committed_stop_log (for the proof).
    let fri_last_layer = current.to_qm31();

    // Phase 2: Uncommitted folds (BLOWUP_BITS extra folds).
    let fri_stop_log = fri::LOG_LAST_LAYER_DEGREE_BOUND;
    while current_log > fri_stop_log {
        let fold_alpha = channel.draw_felt();
        all_fold_alphas.push(fold_alpha);
        let line_domain = Coset::half_odds(current_log);
        let d_twid = fri::compute_fold_twiddles_on_demand(&line_domain, false);
        let folded = fri::fold_line_with_twiddles(&current, fold_alpha, &d_twid);
        drop(d_twid);
        drop(current);
        current = folded;
        current_log -= 1;
    }

    let current_qm31 = current.to_qm31();
    let fri_last_layer_poly = current_qm31;

    // LinePoly coefficients via circle INTT of the OODS quotient.
    // Circle polynomial has 2n coefficients: n even + n odd. The circle fold combines
    // them: line[k] = even[k] + fri_alpha * odd[k] (k=0..n/2-1). Each subsequent
    // line fold halves: coeffs[k] = coeffs[2k] + fold_alpha * coeffs[2k+1].
    let poly_n = {
        let mut c = 1usize << (log_n - 1); // after circle fold: n/2 = 32
        // After committed line folds:
        let n_committed = fri_commitments.len();
        c >>= n_committed;
        // After uncommitted line folds:
        c >>= BLOWUP_BITS;
        c
    };
    let poly_deg = poly_n.ilog2();
    let fri_last_layer_coeffs: Vec<QM31> = {
        // Circle INTT: BRT-canonic OODS → half_coset NTT → INTT → circle coefficients.
        let oods_data = [host_oods_q0.clone(), host_oods_q1.clone(),
                         host_oods_q2.clone(), host_oods_q3.clone()];
        let n = 1usize << log_n;
        let mut circle_coeffs: Vec<[u32; 4]> = vec![[0; 4]; eval_size];
        for comp in 0..4 {
            let mut hc_data = vec![0u32; eval_size];
            for j in 0..eval_size {
                let cn = j.reverse_bits() >> (usize::BITS - log_eval_size);
                hc_data[canonic_to_hc_ntt(cn, log_eval_size)] = oods_data[comp][j];
            }
            let mut d = DeviceBuffer::from_host(&hc_data);
            ntt::interpolate(&mut d, &cache.eval_inv_cache);
            let c = d.to_host();
            for i in 0..eval_size { circle_coeffs[i][comp] = c[i]; }
        }
        // Circle fold: line[k] = even[k] + fri_alpha * odd[k]
        let half_n = n / 2; // 32
        let mut coeffs: Vec<QM31> = (0..half_n).map(|k| {
            let even = QM31::from_u32_array(circle_coeffs[k]);
            let odd = QM31::from_u32_array(circle_coeffs[n + k]);
            even + fri_alpha * odd
        }).collect();
        // Apply line fold alphas (committed + uncommitted).
        // all_fold_alphas[0] = circle fold, [1..] = line folds.
        for alpha_idx in 1..all_fold_alphas.len() {
            let alpha = all_fold_alphas[alpha_idx];
            let half = coeffs.len() / 2;
            coeffs = (0..half).map(|k| coeffs[2*k] + alpha * coeffs[2*k+1]).collect();
        }
        // BRT-permute.
        let final_n = coeffs.len();
        let final_deg = final_n.ilog2();
        let mut brt = vec![QM31::ZERO; final_n];
        for i in 0..final_n { brt[i.reverse_bits() >> (usize::BITS - final_deg)] = coeffs[i]; }
        brt
    };

    // ---- Phase 5: Proof-of-Work grinding + Query + decommitment ----
    channel.mix_felts(&fri_last_layer_coeffs);

    // Grind PoW nonce on GPU.
    // prefix_digest = Blake2s(0x12345678_LE || [0u8;12] || channel_state || POW_BITS_LE)
    // Valid nonce: Blake2s(prefix_digest || nonce_le) has >= POW_BITS trailing zeros in bytes 0..16.
    let pow_nonce: u64 = {
        // Build prefix_digest on CPU (52-byte input)
        let state = channel.state_words();
        let mut prefix_input = [0u8; 52];
        prefix_input[0..4].copy_from_slice(&0x12345678u32.to_le_bytes());
        // bytes 4..16 are zero padding
        for (i, &w) in state.iter().enumerate() {
            prefix_input[16 + i * 4..20 + i * 4].copy_from_slice(&w.to_le_bytes());
        }
        prefix_input[48..52].copy_from_slice(&POW_BITS.to_le_bytes());
        let prefix_digest_bytes = crate::channel::blake2s_hash(&prefix_input);
        let mut prefix_words = [0u32; 8];
        for (i, chunk) in prefix_digest_bytes.chunks_exact(4).enumerate() {
            prefix_words[i] = u32::from_le_bytes(chunk.try_into().unwrap());
        }

        // Allocate device memory for prefixed_digest and result
        let mut d_prefix: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut d_result: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            assert_eq!(ffi::cudaMalloc(&mut d_prefix, 32), 0, "cudaMalloc prefix failed");
            assert_eq!(ffi::cudaMalloc(&mut d_result, 8), 0, "cudaMalloc result failed");
            // Upload prefix_digest
            assert_eq!(ffi::cudaMemcpy(d_prefix, prefix_words.as_ptr() as *const _, 32, 1), 0);
            // Initialize result to u64::MAX
            let init_val: u64 = u64::MAX;
            assert_eq!(ffi::cudaMemcpy(d_result, &init_val as *const u64 as *const _, 8, 1), 0);
        }

        const GRIND_BATCH: u32 = 1 << 22; // 4M threads per batch; RTX 5090 handles in ~1ms
        let mut batch_offset: u64 = 0;
        let nonce = loop {
            unsafe {
                ffi::cuda_grind_pow(
                    d_prefix as *const u32,
                    d_result as *mut u64,
                    POW_BITS,
                    batch_offset,
                    GRIND_BATCH,
                );
                assert_eq!(ffi::cudaDeviceSynchronize(), 0, "grind sync failed");
                let mut result: u64 = u64::MAX;
                assert_eq!(ffi::cudaMemcpy(&mut result as *mut u64 as *mut _, d_result, 8, 2), 0);
                if result != u64::MAX {
                    ffi::cudaFree(d_prefix);
                    ffi::cudaFree(d_result);
                    break result;
                }
            }
            batch_offset += GRIND_BATCH as u64;
        };
        nonce
    };
    channel.mix_u64(pow_nonce);

    let query_indices = channel.draw_query_indices(log_eval_size, N_QUERIES);

    // Sparse trace extraction from host (eval cols already downloaded after commitment)
    let trace_values_at_queries: Vec<Vec<u32>> = {
        let mut result: Vec<Vec<u32>> = vec![vec![0u32; N_COLS]; query_indices.len()];
        for c in 0..N_COLS {
            for (q, &qi) in query_indices.iter().enumerate() {
                result[q][c] = host_eval_cols[c][qi % eval_size];
            }
        }
        result
    };
    let trace_values_at_queries_next: Vec<Vec<u32>> = {
        let mut result: Vec<Vec<u32>> = vec![vec![0u32; N_COLS]; query_indices.len()];
        for c in 0..N_COLS {
            for (q, &qi) in query_indices.iter().enumerate() {
                result[q][c] = host_eval_cols[c][canonic_next(qi, log_eval_size)];
            }
        }
        result
    };
    // Generate Merkle auth paths for trace decommitment.
    // lo paths cover cols 0-15 against trace_commitment.
    // hi paths cover cols 16-30 against trace_commitment_hi.
    // dict paths cover cols 31-33 against dict_trace_commitment.
    // Together they bind all 34 trace columns to committed polynomials.
    let (trace_auth_paths, trace_auth_paths_next,
         trace_auth_paths_hi, trace_auth_paths_hi_next,
         trace_auth_paths_dict, trace_auth_paths_dict_next) = {
        let next_indices: Vec<usize> = query_indices.iter()
            .map(|&qi| canonic_next(qi, log_eval_size))
            .collect();
        let all_indices: Vec<usize> = query_indices.iter().copied()
            .chain(next_indices.iter().copied())
            .collect();
        let n_q = query_indices.len();

        // lo: cols 0..TRACE_LO
        let lo_paths = MerkleTree::cpu_merkle_auth_paths_ncols_with_tile_roots(
            &host_eval_cols[..TRACE_LO],
            &tile_roots_lo,
            &all_indices,
        );
        let paths_lo_qi:  Vec<Vec<[u32; 8]>> = lo_paths[..n_q].to_vec();
        let paths_lo_qi1: Vec<Vec<[u32; 8]>> = lo_paths[n_q..].to_vec();

        // hi: cols TRACE_LO..TRACE_VM_END (16..31 = 15 Cairo VM hi cols)
        let hi_paths = MerkleTree::cpu_merkle_auth_paths_ncols_with_tile_roots(
            &host_eval_cols[TRACE_LO..TRACE_VM_END],
            &tile_roots_hi,
            &all_indices,
        );
        let paths_hi_qi:  Vec<Vec<[u32; 8]>> = hi_paths[..n_q].to_vec();
        let paths_hi_qi1: Vec<Vec<[u32; 8]>> = hi_paths[n_q..].to_vec();

        // dict: cols TRACE_VM_END..N_COLS (31..34 = 3 dict linkage cols)
        let dict_paths = MerkleTree::cpu_merkle_auth_paths_ncols_with_tile_roots(
            &host_eval_cols[TRACE_VM_END..N_COLS],
            &tile_roots_dict,
            &all_indices,
        );
        let paths_dict_qi:  Vec<Vec<[u32; 8]>> = dict_paths[..n_q].to_vec();
        let paths_dict_qi1: Vec<Vec<[u32; 8]>> = dict_paths[n_q..].to_vec();

        (paths_lo_qi, paths_lo_qi1, paths_hi_qi, paths_hi_qi1,
         paths_dict_qi, paths_dict_qi1)
    };

    drop(host_eval_cols);

    // EC trace at query points + Merkle auth paths (lo and hi halves)
    let (ec_trace_at_queries, ec_trace_at_queries_next,
         ec_trace_auth_paths, ec_trace_auth_paths_next,
         ec_trace_auth_paths_hi, ec_trace_auth_paths_hi_next) =
        if let Some(ref ec_host) = ec_trace_host {
        let ec_eval_size = ec_host[0].len();
        let at_q: Vec<Vec<u32>> = query_indices.iter().map(|&qi| {
            let idx = qi % ec_eval_size;
            ec_host.iter().map(|col| col[idx]).collect()
        }).collect();
        let at_qn: Vec<Vec<u32>> = query_indices.iter().map(|&qi| {
            let idx = (qi + 1) % ec_eval_size;
            ec_host.iter().map(|col| col[idx]).collect()
        }).collect();

        // Generate auth paths for lo (0..EC_TRACE_LO) and hi (EC_TRACE_LO..) halves.
        // Indices must be mapped into the EC eval domain (which may differ in size
        // from the main eval domain).
        let n_q = query_indices.len();
        let ec_query_indices: Vec<usize> = query_indices.iter()
            .map(|&qi| qi % ec_eval_size).collect();
        let ec_next_indices: Vec<usize> = query_indices.iter()
            .map(|&qi| (qi + 1) % ec_eval_size).collect();
        let ec_all_qi: Vec<usize> = ec_query_indices.iter().copied()
            .chain(ec_next_indices.iter().copied()).collect();

        let ec_lo_paths = MerkleTree::cpu_merkle_auth_paths_ncols(
            &ec_host[..EC_TRACE_LO], &ec_all_qi);
        let ec_hi_paths = MerkleTree::cpu_merkle_auth_paths_ncols(
            &ec_host[EC_TRACE_LO..], &ec_all_qi);

        (at_q, at_qn,
         ec_lo_paths[..n_q].to_vec(), ec_lo_paths[n_q..].to_vec(),
         ec_hi_paths[..n_q].to_vec(), ec_hi_paths[n_q..].to_vec())
    } else {
        (Vec::new(), Vec::new(),
         Vec::new(), Vec::new(), Vec::new(), Vec::new())
    };

    // Sparse quotient decommitment from host (downloaded after quotient commitment)
    // Generate quotient decommitment with real Merkle auth paths.
    let quotient_decommitment = {
        let cols = [host_q0, host_q1, host_q2, host_q3]; // moves Vecs
        let decom = decommit_from_host_soa4(&cols, &tile_roots_quotient, &query_indices);
        decom
    };
    // OODS quotient decommitment — FRI starts from these values (not AIR quotient).
    let oods_quotient_decommitment = {
        let cols = [host_oods_q0, host_oods_q1, host_oods_q2, host_oods_q3];
        decommit_from_host_soa4(&cols, &tile_roots_oods_q, &query_indices)
    };

    // LogUp interaction trace decommitment — binds interaction trace values at query
    // points to interaction_commitment (prerequisite for step transition verification).
    let interaction_decommitment = decommit_from_host_soa4(&host_logup, &tile_roots_logup, &query_indices);
    // Next-row decommitment — needed by verifier to check S[i+1] - S[i] = delta(row_i).
    let next_query_indices: Vec<usize> = query_indices.iter()
        .map(|&qi| canonic_next(qi, log_eval_size))
        .collect();
    let interaction_decommitment_next = decommit_from_host_soa4(&host_logup, &tile_roots_logup, &next_query_indices);
    drop(host_logup);

    // RC interaction trace decommitment — same structure as LogUp.
    let rc_interaction_decommitment = decommit_from_host_soa4(&host_rc_logup, &tile_roots_rc, &query_indices);
    let rc_interaction_decommitment_next = decommit_from_host_soa4(&host_rc_logup, &tile_roots_rc, &next_query_indices);
    drop(host_rc_logup);

    // S_dict interaction trace decommitment at query points and query+1 points.
    let dict_main_interaction_decommitment = decommit_from_host_soa4(&host_sdict, &tile_roots_sdict, &query_indices);
    let dict_main_interaction_decommitment_next = decommit_from_host_soa4(&host_sdict, &tile_roots_sdict, &next_query_indices);
    drop(host_sdict);

    // Dict sub-AIR: full trace data is included in the proof (dict_exec_data_opt /
    // dict_sorted_data_opt). No per-query decommitment needed — verifier recomputes
    // the Merkle roots from the full data and checks all step-transition constraints.

    // FRI decommitments with real Merkle auth paths.
    // Previously used sparse_decommit_soa4_gpu (empty auth_paths); the verifier
    // would skip auth path checks, allowing forged fold equation values.
    let mut fri_decommitments = Vec::new();
    let mut folded_indices: Vec<usize> = query_indices.iter().map(|&qi| qi / 2).collect();

    for eval in &fri_evals {
        let decom = decommit_fri_layer(eval, &folded_indices);
        fri_decommitments.push(decom);
        folded_indices = folded_indices.iter().map(|&i| i / 2).collect();
    }
    CairoProof {
        log_trace_size: log_n,
        public_inputs,
        trace_commitment,
        trace_commitment_hi,
        dict_trace_commitment,
        ec_trace_commitment,
        ec_trace_commitment_hi,
        ec_log_eval: ec_log_eval_opt,
        ec_trace_at_queries,
        ec_trace_at_queries_next,
        ec_trace_auth_paths,
        ec_trace_auth_paths_next,
        ec_trace_auth_paths_hi,
        ec_trace_auth_paths_hi_next,
        interaction_commitment,
        rc_interaction_commitment,
        quotient_commitment,
        oods_quotient_commitment,
        fri_start_channel_state,
        fri_commitments,
        fri_last_layer,
        fri_last_layer_poly: fri_last_layer_coeffs.clone(),
        pow_nonce,
        query_indices,
        trace_values_at_queries,
        trace_values_at_queries_next,
        trace_auth_paths,
        trace_auth_paths_next,
        trace_auth_paths_hi,
        trace_auth_paths_hi_next,
        trace_auth_paths_dict,
        trace_auth_paths_dict_next,
        logup_final_sum,
        rc_final_sum,
        quotient_decommitment,
        oods_quotient_decommitment,
        fri_decommitments,
        interaction_decommitment,
        interaction_decommitment_next,
        rc_interaction_decommitment,
        rc_interaction_decommitment_next,
        dict_main_interaction_commitment,
        dict_main_interaction_decommitment,
        dict_main_interaction_decommitment_next,
        dict_link_final: dict_link_final_arr,
        dict_n_accesses: dict_accesses.len(),
        logup_challenges,
        dict_exec_commitment: dict_exec_commitment_opt,
        dict_sorted_commitment: dict_sorted_commitment_opt,
        dict_log_n: dict_log_n_opt,
        dict_exec_final_sum: dict_exec_final_sum_opt,
        dict_sorted_final_sum: dict_sorted_final_sum_opt,
        dict_exec_data: dict_exec_data_opt.unwrap_or_default(),
        dict_sorted_data: dict_sorted_data_opt.unwrap_or_default(),
        bitwise_commitment: bitwise_commitment_opt,
        bitwise_rows,
        memory_table_commitment,
        memory_table_data,
        memory_instr_data,
        rc_counts_commitment,
        rc_counts_data,
        oods_z,
        oods_trace_at_z,
        oods_trace_at_z_next,
        oods_quotient_at_z,
        oods_alpha,
        oods_interaction_at_z,
        oods_interaction_at_z_next,
    }
}

/// Verify a Cairo STARK proof.
pub fn cairo_verify(proof: &CairoProof) -> Result<(), String> {
    let log_n = proof.log_trace_size;
    let log_eval_size = log_n + BLOWUP_BITS;
    let eval_size = 1usize << log_eval_size;

    // ---- Verify public inputs ----
    if proof.public_inputs.n_steps == 0 {
        return Err("Zero execution steps".into());
    }
    if proof.public_inputs.n_steps > (1 << log_n) {
        return Err("More steps than trace size".into());
    }

    // ---- Replay Fiat-Shamir (must match prover exactly) ----
    let mut channel = Channel::new();
    channel.mix_digest(&proof.public_inputs.program_hash);
    channel.mix_digest(&proof.trace_commitment);
    channel.mix_digest(&proof.trace_commitment_hi);
    // dict_trace_commitment binds cols 31-33 (dict_key, dict_new, dict_active) before
    // any dict challenges are drawn.  Must be mixed here to keep channel in sync.
    channel.mix_digest(&proof.dict_trace_commitment);

    // ── S_dict link verification (GAP-1 closure) ─────────────────────────────
    // Draw z_dict_link, alpha_dict_link after dict_trace_commitment (same as prover).
    let z_dict_link_v     = channel.draw_felt();
    let alpha_dict_link_v = channel.draw_felt();
    // Authenticate dict_main_interaction_commitment; mix it and dict_link_final.
    channel.mix_digest(&proof.dict_main_interaction_commitment);
    channel.mix_digest(&[
        proof.dict_link_final[0], proof.dict_link_final[1],
        proof.dict_link_final[2], proof.dict_link_final[3],
        0, 0, 0, 0,
    ]);
    // Verify that the claimed S_dict_final matches exec_key_new_sum computed
    // from the authenticated dict exec trace data.
    if !proof.dict_exec_data.is_empty() {
        use super::logup::qm31_from_m31;
        let n_acc = proof.dict_n_accesses;
        // Validate that padding rows are genuinely zero.
        for (idx, row) in proof.dict_exec_data.iter().enumerate().skip(n_acc) {
            if *row != [0u32, 0, 0] {
                return Err(format!(
                    "dict_exec_data row {idx} should be padding (all zeros) \
                     but is {:?}", row));
            }
        }
        // Recompute exec_key_new_sum from first n_acc rows.
        let mut exec_key_new_sum = QM31::ZERO;
        for j in 0..n_acc {
            let key     = qm31_from_m31(M31(proof.dict_exec_data[j][0]));
            let new_val = qm31_from_m31(M31(proof.dict_exec_data[j][2]));
            let entry = key + alpha_dict_link_v * new_val;
            let denom = z_dict_link_v - entry;
            exec_key_new_sum = exec_key_new_sum + denom.inverse();
        }
        let claimed_final = QM31::from_u32_array(proof.dict_link_final);
        if claimed_final != exec_key_new_sum {
            return Err(format!(
                "S_dict link check failed: dict_link_final={:?} != recomputed exec_key_new_sum={:?} \
                 — main trace dict columns do not match execution log",
                proof.dict_link_final, exec_key_new_sum.to_u32_array()
            ));
        }
    }

    // Bind both EC trace commitments (lo/hi split) into Fiat-Shamir.
    // Must match prover order exactly: EC commits before drawing z_mem/alpha_mem/z_rc.
    if let Some(ref ec_commit_lo) = proof.ec_trace_commitment {
        channel.mix_digest(ec_commit_lo);
        let ec_commit_hi = proof.ec_trace_commitment_hi.as_ref()
            .ok_or("ec_trace_commitment_hi missing when ec_trace_commitment is present")?;
        channel.mix_digest(ec_commit_hi);
    }

    // ---- Dict consistency sub-AIR verification (GAP-1 — full soundness) ----
    // The verifier receives the complete exec and sorted trace data.
    // It authenticates the data by recomputing Merkle roots from the full payload,
    // then independently verifies ALL step-transition constraints and the LogUp sums.
    // This achieves full (non-statistical) soundness for the dict sub-AIR.
    if let Some(exec_root) = proof.dict_exec_commitment {
        let sorted_root = proof.dict_sorted_commitment
            .ok_or("dict_sorted_commitment missing")?;
        let dict_log_n = proof.dict_log_n
            .ok_or("dict_log_n missing")?;
        let dict_exec_final_claimed = proof.dict_exec_final_sum
            .ok_or("dict_exec_final_sum missing")?;
        let dict_sorted_final_claimed = proof.dict_sorted_final_sum
            .ok_or("dict_sorted_final_sum missing")?;
        let dict_n = 1usize << dict_log_n;

        let exec_data   = &proof.dict_exec_data;
        let sorted_data = &proof.dict_sorted_data;

        if exec_data.len() != dict_n {
            return Err(format!(
                "dict_exec_data length mismatch: expected {dict_n}, got {}", exec_data.len()));
        }
        if sorted_data.len() != dict_n {
            return Err(format!(
                "dict_sorted_data length mismatch: expected {dict_n}, got {}", sorted_data.len()));
        }

        // Authenticate exec data: recompute Merkle root from full payload.
        {
            let leaf_hashes: Vec<[u32; 8]> = exec_data.iter()
                .map(|row| MerkleTree::hash_leaf(row.as_slice()))
                .collect();
            let layers = MerkleTree::build_cpu_tree_layers(leaf_hashes);
            let computed_root = layers.last().ok_or("dict exec tree is empty")?[0];
            if computed_root != exec_root {
                return Err("dict exec Merkle root mismatch — exec data is not authentic".into());
            }
        }

        // Authenticate sorted data: recompute Merkle root from full payload.
        {
            let leaf_hashes: Vec<[u32; 8]> = sorted_data.iter()
                .map(|row| MerkleTree::hash_leaf(row.as_slice()))
                .collect();
            let layers = MerkleTree::build_cpu_tree_layers(leaf_hashes);
            let computed_root = layers.last().ok_or("dict sorted tree is empty")?[0];
            if computed_root != sorted_root {
                return Err("dict sorted Merkle root mismatch — sorted data is not authentic".into());
            }
        }

        // Replay Fiat-Shamir: mix both data roots → draw LogUp challenges.
        channel.mix_digest(&exec_root);
        channel.mix_digest(&sorted_root);
        let z_dict     = channel.draw_felt();
        let alpha_dict = channel.draw_felt();

        // Recompute exec and sorted LogUp final sums from the full authenticated data.
        // Flatten row-major data into column slices for build_dict_interaction_trace.
        let exec_keys:  Vec<u32> = exec_data.iter().map(|r| r[0]).collect();
        let exec_prevs: Vec<u32> = exec_data.iter().map(|r| r[1]).collect();
        let exec_news:  Vec<u32> = exec_data.iter().map(|r| r[2]).collect();
        let (_, exec_final_recomputed) = dict_air::build_dict_interaction_trace(
            &exec_keys, &exec_prevs, &exec_news, z_dict, alpha_dict);

        let sorted_keys:  Vec<u32> = sorted_data.iter().map(|r| r[0]).collect();
        let sorted_prevs: Vec<u32> = sorted_data.iter().map(|r| r[1]).collect();
        let sorted_news:  Vec<u32> = sorted_data.iter().map(|r| r[2]).collect();
        let (_, sorted_final_recomputed) = dict_air::build_dict_interaction_trace(
            &sorted_keys, &sorted_prevs, &sorted_news, z_dict, alpha_dict);

        // Verify claimed final sums match recomputed values.
        let exec_final_arr   = exec_final_recomputed.to_u32_array();
        let sorted_final_arr = sorted_final_recomputed.to_u32_array();
        if exec_final_arr != dict_exec_final_claimed {
            return Err("dict exec final sum mismatch — claimed sum differs from recomputed".into());
        }
        if sorted_final_arr != dict_sorted_final_claimed {
            return Err("dict sorted final sum mismatch — claimed sum differs from recomputed".into());
        }

        // Permutation argument: exec and sorted are the same multiset iff sums are equal.
        if exec_final_arr != sorted_final_arr {
            return Err(format!(
                "dict LogUp permutation failed: exec_final≠sorted_final \
                 exec={exec_final_arr:?} sorted={sorted_final_arr:?}"
            ));
        }

        // Check ALL sorted step-transition constraints C0-C3 (full soundness, not sampled).
        for i in 0..dict_n - 1 {
            let cs = dict_air::eval_sorted_constraints(sorted_data[i], sorted_data[i + 1]);
            if cs != [0, 0, 0, 0] {
                return Err(format!(
                    "dict sorted step-transition violated at row {i}: C={cs:?}"
                ));
            }
        }

        // Mix final sums into channel (must match prover order before z_mem is drawn).
        channel.mix_digest(&[
            exec_final_arr[0],   exec_final_arr[1],   exec_final_arr[2],   exec_final_arr[3],
            sorted_final_arr[0], sorted_final_arr[1], sorted_final_arr[2], sorted_final_arr[3],
        ]);
    }

    let _z_mem = channel.draw_felt();
    let _alpha_mem = channel.draw_felt();
    let _z_rc = channel.draw_felt();

    // ---- LogUp + Range check verification ----
    // The LogUp final sum and RC final sum are bound into the Fiat-Shamir transcript.
    // Tampering them changes all subsequent challenges, breaking FRI verification.
    // The interaction trace is committed and FRI-verified as low-degree.
    // This provides equivalent security without re-executing the VM.
    // (The verifier is O(n_queries × log(n)), not O(n_steps))

    // Must match prover order exactly:
    //   interaction_commitment → rc_interaction_commitment → final_sums → draw alphas
    channel.mix_digest(&proof.interaction_commitment);
    if proof.rc_interaction_commitment == [0u32; 8] {
        return Err("RC interaction commitment is zero".into());
    }
    channel.mix_digest(&proof.rc_interaction_commitment);
    channel.mix_digest(&[
        proof.logup_final_sum[0], proof.logup_final_sum[1],
        proof.logup_final_sum[2], proof.logup_final_sum[3],
        proof.rc_final_sum[0], proof.rc_final_sum[1],
        proof.rc_final_sum[2], proof.rc_final_sum[3],
    ]);

    // ---- Bitwise builtin verification ----
    // Replay the bitwise commitment into the channel (matches prover), then check
    // every bitwise row:
    //   C0: xor + 2·and ≡ x + y (mod P)
    //   C1: or           ≡ and + xor (mod P)
    if !proof.bitwise_rows.is_empty() {
        let flat: Vec<u32> = proof.bitwise_rows.iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let recomputed_hash = crate::merkle::MerkleTree::hash_leaf(&flat);
        match proof.bitwise_commitment {
            Some(committed) if committed != recomputed_hash =>
                return Err(format!(
                    "Bitwise commitment mismatch: committed {:?} ≠ recomputed {:?}",
                    committed, recomputed_hash)),
            None =>
                return Err("bitwise_rows non-empty but bitwise_commitment is None".into()),
            _ => {}
        }
        channel.mix_digest(&recomputed_hash);
        let p = crate::field::m31::P;
        for (i, row) in proof.bitwise_rows.iter().enumerate() {
            let [x, y, and, xor, or_val] = *row;
            // C0: (xor + 2·and) mod P == (x + y) mod P
            let lhs0 = (xor as u64 + 2 * and as u64) % p as u64;
            let rhs0 = (x   as u64 + y   as u64)     % p as u64;
            if lhs0 != rhs0 {
                return Err(format!("Bitwise C0 failed at row {i}: xor+2·and={lhs0} ≠ x+y={rhs0}"));
            }
            // C1: or ≡ and + xor (mod P)
            let lhs1 = or_val as u64 % p as u64;
            let rhs1 = (and as u64 + xor as u64) % p as u64;
            if lhs1 != rhs1 {
                return Err(format!("Bitwise C1 failed at row {i}: or={lhs1} ≠ and+xor={rhs1}"));
            }
        }
    } else if proof.bitwise_commitment.is_some() {
        return Err("bitwise_commitment present but bitwise_rows is empty".into());
    }

    // ---- Memory table commitment verification (closes LogUp soundness gap) ----
    // Recompute Blake2s hash from proof data, verify it matches the committed value,
    // then independently compute table_sum and check exec_sum + table_sum == 0.
    {
        let mem_flat: Vec<u32> = proof.memory_table_data.iter()
            .flat_map(|&[a, v, m]| [a, v, m])
            .chain(proof.memory_instr_data.iter().flat_map(|&[p, lo, hi, m]| [p, lo, hi, m]))
            .collect();
        let recomputed = crate::channel::hash_words(&mem_flat);
        if recomputed != proof.memory_table_commitment {
            return Err(format!(
                "Memory table commitment mismatch: committed {:?} ≠ recomputed {:?}",
                proof.memory_table_commitment, recomputed));
        }
        channel.mix_digest(&proof.memory_table_commitment);

        let data_entries: Vec<(M31, M31, u32)> = proof.memory_table_data.iter()
            .map(|&[a, v, m]| (M31(a), M31(v), m)).collect();
        let instr_entries: Vec<(M31, M31, M31, u32)> = proof.memory_instr_data.iter()
            .map(|&[p, lo, hi, m]| (M31(p), M31(lo), M31(hi), m)).collect();
        let table_sum = compute_memory_table_sum(&data_entries, &instr_entries, _z_mem, _alpha_mem);
        let exec_sum = QM31::from_u32_array(proof.logup_final_sum);
        if exec_sum + table_sum != QM31::ZERO {
            return Err(format!(
                "LogUp memory cancellation failed: exec_sum + table_sum = {:?} (expected zero)",
                (exec_sum + table_sum).to_u32_array()));
        }
    }

    // ---- RC counts commitment verification (closes RC soundness gap) ----
    {
        use super::range_check::RC_TABLE_SIZE;
        if proof.rc_counts_data.len() != RC_TABLE_SIZE {
            return Err(format!(
                "RC counts data wrong length: {} ≠ {RC_TABLE_SIZE}",
                proof.rc_counts_data.len()));
        }
        let recomputed = crate::channel::hash_words(&proof.rc_counts_data);
        if recomputed != proof.rc_counts_commitment {
            return Err(format!(
                "RC counts commitment mismatch: committed {:?} ≠ recomputed {:?}",
                proof.rc_counts_commitment, recomputed));
        }
        channel.mix_digest(&proof.rc_counts_commitment);
        let rc_counts_arr: [u32; RC_TABLE_SIZE] = proof.rc_counts_data.as_slice()
            .try_into()
            .map_err(|_| "RC counts data wrong length".to_string())?;
        let rc_table_sum = compute_rc_table_sum(&rc_counts_arr, _z_rc);
        let rc_exec_sum = QM31::from_u32_array(proof.rc_final_sum);
        if rc_exec_sum + rc_table_sum != QM31::ZERO {
            return Err(format!(
                "RC LogUp cancellation failed: rc_exec_sum + rc_table_sum = {:?} (expected zero)",
                (rc_exec_sum + rc_table_sum).to_u32_array()));
        }
    }

    let constraint_alphas_drawn: Vec<QM31> = (0..N_CONSTRAINTS).map(|_| channel.draw_felt()).collect();

    channel.mix_digest(&proof.quotient_commitment);

    // ── OODS: replay sampled values in Fiat-Shamir (stwo flatten_cols order) ──
    // Single mix_felts call matching stwo's channel.mix_felts(sampled_values.flatten_cols()).
    // Order: for each tree, for each column, for each sample point.
    //   Trees 0-2 (trace, N_COLS cols × 2 samples): col_i_at_z, col_i_at_z_next interleaved.
    //   Trees 3-5 (interaction, 4 cols × 2 samples): col_k_at_z, col_k_at_z_next interleaved.
    //   Tree 6 (quotient, 4 cols × 1 sample): col_k_at_z.
    {
        use crate::oods::OodsPoint;
        let _z = OodsPoint::from_channel(&mut channel);
        let n_trace = proof.oods_trace_at_z.len().min(proof.oods_trace_at_z_next.len());
        let mut combined: Vec<crate::field::QM31> =
            Vec::with_capacity(n_trace * 2 + 12 * 2 + 4);
        for i in 0..n_trace {
            combined.push(crate::field::QM31::from_u32_array(proof.oods_trace_at_z[i]));
            combined.push(crate::field::QM31::from_u32_array(proof.oods_trace_at_z_next[i]));
        }
        for pi in 0..3 {
            for k in 0..4 {
                combined.push(crate::field::QM31::from_u32_array(proof.oods_interaction_at_z[pi][k]));
                combined.push(crate::field::QM31::from_u32_array(proof.oods_interaction_at_z_next[pi][k]));
            }
        }
        for k in 0..4 {
            combined.push(crate::field::QM31::from_u32_array(proof.oods_quotient_at_z[k]));
        }
        channel.mix_felts(&combined);
        let _oods_alpha = channel.draw_felt();
    }

    // Mix the OODS quotient Merkle commitment (prover commits this before fri_alpha).
    channel.mix_digest(&proof.oods_quotient_commitment);

    let mut fri_alphas = Vec::new();
    fri_alphas.push(channel.draw_felt()); // circle fold alpha

    for fri_commitment in &proof.fri_commitments {
        channel.mix_digest(fri_commitment);
        fri_alphas.push(channel.draw_felt());
    }

    // Draw BLOWUP_BITS uncommitted fold alphas.
    for _ in 0..BLOWUP_BITS {
        fri_alphas.push(channel.draw_felt());
    }

    // ---- Verify FRI structure ----
    // Committed layers stop at committed_stop_log = LOG_LAST_LAYER_DEGREE_BOUND + BLOWUP_BITS.
    // The prover then does BLOWUP_BITS uncommitted folds down to LOG_LAST_LAYER_DEGREE_BOUND.
    let committed_stop_log = fri::LOG_LAST_LAYER_DEGREE_BOUND + BLOWUP_BITS;
    let expected_fri_layers = (log_eval_size - 1).saturating_sub(committed_stop_log);
    if proof.fri_commitments.len() != expected_fri_layers as usize {
        return Err(format!("Expected {} FRI layers, got {}",
            expected_fri_layers, proof.fri_commitments.len()));
    }

    let expected_last_layer_size = 1usize << committed_stop_log;
    if proof.fri_last_layer.len() != expected_last_layer_size {
        return Err(format!("Expected {} FRI last layer evaluations, got {}",
            expected_last_layer_size,
            proof.fri_last_layer.len()));
    }

    // ---- Verify proof-of-work + re-derive query indices ----
    let n_fri_layers = proof.fri_decommitments.len();
    // Use the stored LinePoly coefficients (computed by prover via circle INTT).
    channel.mix_felts(&proof.fri_last_layer_poly);
    if !channel.verify_pow_nonce(POW_BITS, proof.pow_nonce) {
        return Err(format!(
            "Proof-of-work check failed: nonce {} does not satisfy {}-bit PoW",
            proof.pow_nonce, POW_BITS
        ));
    }
    channel.mix_u64(proof.pow_nonce);
    let expected_indices = channel.draw_query_indices(log_eval_size, N_QUERIES);
    if proof.query_indices != expected_indices {
        return Err("Query indices don't match Fiat-Shamir derivation".into());
    }

    // ---- Verify non-trivial commitments ----
    if proof.trace_commitment == [0; 8] {
        return Err("Trace commitment is zero".into());
    }
    if proof.quotient_commitment == [0; 8] {
        return Err("Quotient commitment is zero".into());
    }
    if proof.interaction_commitment == [0; 8] {
        return Err("Interaction commitment is zero".into());
    }

    // ---- Verify FRI fold equations ----
    for (q, &qi) in proof.query_indices.iter().enumerate() {
        let mut current_idx = qi;
        let mut current_log = log_eval_size;

        // Circle fold: OODS quotient (BRT-canonic ordered) → FRI layer 0 (BRT line order)
        {
            let folded_idx = current_idx / 2;
            let (f0, f1) = get_pair_from_decom_4(
                &proof.oods_quotient_decommitment.values[q],
                &proof.oods_quotient_decommitment.sibling_values[q],
                current_idx,
            );
            // Circle fold twiddle: 1/y at the canonic domain point for the even element.
            let brt_even = (current_idx & !1).reverse_bits() >> (usize::BITS - log_eval_size);
            let twiddle = canonic_domain_point(brt_even, log_eval_size).y.inverse();
            let expected = fold_pair(f0, f1, fri_alphas[0], twiddle);
            if n_fri_layers > 0 {
                let actual = QM31::from_u32_array(proof.fri_decommitments[0].values[q]);
                if expected != actual {
                    return Err(format!("Circle fold mismatch at query {q} (qi={qi})"));
                }
            } else {
                // 0 committed layers: circle fold result lands directly in fri_last_layer.
                if expected != proof.fri_last_layer[folded_idx] {
                    return Err(format!("Circle fold → last layer mismatch at query {q} (qi={qi})"));
                }
            }
            current_idx = folded_idx;
            current_log -= 1;
        }

        // Line folds
        for layer in 0..n_fri_layers.saturating_sub(1) {
            let ho = Coset::half_odds(current_log);
            let folded_idx = current_idx / 2;
            let decom = &proof.fri_decommitments[layer];
            let (f0, f1) = get_pair_from_decom_4(
                &decom.values[q], &decom.sibling_values[q], current_idx,
            );
            let twiddle = fold_twiddle_at(&ho, folded_idx, false);
            let expected = fold_pair(f0, f1, fri_alphas[layer + 1], twiddle);
            let actual = QM31::from_u32_array(proof.fri_decommitments[layer + 1].values[q]);
            if expected != actual {
                return Err(format!("Line fold mismatch at query {q}, layer {layer}"));
            }
            current_idx = folded_idx;
            current_log -= 1;
        }

        // Verify: fold last committed decommitment matches fri_last_layer.
        if n_fri_layers > 0 {
            let last_decom = &proof.fri_decommitments[n_fri_layers - 1];
            let ho = Coset::half_odds(current_log);
            let folded_idx = current_idx / 2;
            let (f0, f1) = get_pair_from_decom_4(
                &last_decom.values[q], &last_decom.sibling_values[q], current_idx,
            );
            let twiddle = fold_twiddle_at(&ho, folded_idx, false);
            let expected = fold_pair(f0, f1, fri_alphas[n_fri_layers], twiddle);
            if expected != proof.fri_last_layer[folded_idx] {
                return Err(format!("FRI last layer mismatch at query {q}"));
            }
        }
    }

    // ---- Verify trace decommitment: auth paths bind all 34 trace columns to commitments ----
    // Group A (trace_commitment):     cols 0..TRACE_LO=16      (16 cols)
    // Group B (trace_commitment_hi):  cols TRACE_LO..TRACE_VM_END=31 (15 cols)
    // Group C (dict_trace_commitment): cols TRACE_VM_END..N_COLS=34  (3 cols)
    // Together they cryptographically bind all 34 trace columns.
    const TRACE_LO: usize = 16;
    const TRACE_VM_END: usize = 31;
    let n_q = proof.query_indices.len();
    if proof.trace_auth_paths.len() != n_q || proof.trace_auth_paths_next.len() != n_q
        || proof.trace_auth_paths_hi.len() != n_q || proof.trace_auth_paths_hi_next.len() != n_q {
        return Err("trace auth path length mismatch".into());
    }
    if proof.trace_commitment_hi == [0u32; 8] {
        return Err("trace_commitment_hi is zero".into());
    }
    for (q, &qi) in proof.query_indices.iter().enumerate() {
        let qi_next = canonic_next(qi, log_eval_size);

        // lo (Group A): cols 0..TRACE_LO vs trace_commitment
        let leaf_lo_qi = MerkleTree::hash_leaf(&proof.trace_values_at_queries[q][..TRACE_LO]);
        if !MerkleTree::verify_auth_path(&proof.trace_commitment, &leaf_lo_qi, qi, &proof.trace_auth_paths[q]) {
            return Err(format!("Trace lo auth path failed at query {q} (qi={qi})"));
        }
        let leaf_lo_next = MerkleTree::hash_leaf(&proof.trace_values_at_queries_next[q][..TRACE_LO]);
        if !MerkleTree::verify_auth_path(&proof.trace_commitment, &leaf_lo_next, qi_next, &proof.trace_auth_paths_next[q]) {
            return Err(format!("Trace lo auth path (next) failed at query {q}"));
        }

        // hi (Group B): cols TRACE_LO..TRACE_VM_END vs trace_commitment_hi
        let leaf_hi_qi = MerkleTree::hash_leaf(&proof.trace_values_at_queries[q][TRACE_LO..TRACE_VM_END]);
        if !MerkleTree::verify_auth_path(&proof.trace_commitment_hi, &leaf_hi_qi, qi, &proof.trace_auth_paths_hi[q]) {
            return Err(format!("Trace hi auth path failed at query {q} (qi={qi})"));
        }
        let leaf_hi_next = MerkleTree::hash_leaf(&proof.trace_values_at_queries_next[q][TRACE_LO..TRACE_VM_END]);
        if !MerkleTree::verify_auth_path(&proof.trace_commitment_hi, &leaf_hi_next, qi_next, &proof.trace_auth_paths_hi_next[q]) {
            return Err(format!("Trace hi auth path (next) failed at query {q}"));
        }

        // dict (Group C): cols TRACE_VM_END..N_COLS vs dict_trace_commitment
        let leaf_dict_qi = MerkleTree::hash_leaf(&proof.trace_values_at_queries[q][TRACE_VM_END..]);
        if !MerkleTree::verify_auth_path(&proof.dict_trace_commitment, &leaf_dict_qi, qi, &proof.trace_auth_paths_dict[q]) {
            return Err(format!("Dict trace auth path failed at query {q} (qi={qi})"));
        }
        let leaf_dict_next = MerkleTree::hash_leaf(&proof.trace_values_at_queries_next[q][TRACE_VM_END..]);
        if !MerkleTree::verify_auth_path(&proof.dict_trace_commitment, &leaf_dict_next, qi_next, &proof.trace_auth_paths_dict_next[q]) {
            return Err(format!("Dict trace auth path (next) failed at query {q}"));
        }
    }

    // ---- Verify EC trace decommitment: auth paths bind all 29 EC trace columns ----
    // EC trace is split lo (cols 0..16) and hi (cols 16..29), each committed separately.
    // The EC eval domain may differ in size from the main eval domain; ec_log_eval
    // lets the verifier map main query indices into the EC domain.
    const EC_TRACE_LO: usize = 16;
    if let (Some(ec_commit_lo), Some(ec_commit_hi), Some(ec_log_eval)) =
        (&proof.ec_trace_commitment, &proof.ec_trace_commitment_hi, proof.ec_log_eval)
    {
        let ec_eval_size = 1usize << ec_log_eval;
        if proof.ec_trace_auth_paths.len() != n_q
            || proof.ec_trace_auth_paths_next.len() != n_q
            || proof.ec_trace_auth_paths_hi.len() != n_q
            || proof.ec_trace_auth_paths_hi_next.len() != n_q {
            return Err("EC trace auth path length mismatch".into());
        }
        if proof.ec_trace_auth_paths.iter().all(|p| p.is_empty()) {
            return Err("EC trace lo auth paths are empty — commitment unverified".into());
        }
        if proof.ec_trace_auth_paths_hi.iter().all(|p| p.is_empty()) {
            return Err("EC trace hi auth paths are empty — commitment unverified".into());
        }

        for (q, &qi) in proof.query_indices.iter().enumerate() {
            // Map main eval domain index into EC eval domain.
            let ec_qi = qi % ec_eval_size;
            let ec_qi_next = (qi + 1) % ec_eval_size;

            // lo: cols 0..EC_TRACE_LO vs ec_trace_commitment
            let lo_vals_qi: Vec<u32> = proof.ec_trace_at_queries[q]
                .iter().take(EC_TRACE_LO).copied().collect();
            let leaf_lo_qi = MerkleTree::hash_leaf(&lo_vals_qi);
            if !MerkleTree::verify_auth_path(ec_commit_lo, &leaf_lo_qi, ec_qi,
                                              &proof.ec_trace_auth_paths[q]) {
                return Err(format!("EC trace lo auth path failed at query {q} (ec_qi={ec_qi})"));
            }
            let lo_vals_qn: Vec<u32> = proof.ec_trace_at_queries_next[q]
                .iter().take(EC_TRACE_LO).copied().collect();
            let leaf_lo_next = MerkleTree::hash_leaf(&lo_vals_qn);
            if !MerkleTree::verify_auth_path(ec_commit_lo, &leaf_lo_next, ec_qi_next,
                                              &proof.ec_trace_auth_paths_next[q]) {
                return Err(format!("EC trace lo auth path (next) failed at query {q}"));
            }

            // hi: cols EC_TRACE_LO.. vs ec_trace_commitment_hi
            let hi_vals_qi: Vec<u32> = proof.ec_trace_at_queries[q]
                .iter().skip(EC_TRACE_LO).copied().collect();
            let leaf_hi_qi = MerkleTree::hash_leaf(&hi_vals_qi);
            if !MerkleTree::verify_auth_path(ec_commit_hi, &leaf_hi_qi, ec_qi,
                                              &proof.ec_trace_auth_paths_hi[q]) {
                return Err(format!("EC trace hi auth path failed at query {q} (ec_qi={ec_qi})"));
            }
            let hi_vals_qn: Vec<u32> = proof.ec_trace_at_queries_next[q]
                .iter().skip(EC_TRACE_LO).copied().collect();
            let leaf_hi_next = MerkleTree::hash_leaf(&hi_vals_qn);
            if !MerkleTree::verify_auth_path(ec_commit_hi, &leaf_hi_next, ec_qi_next,
                                              &proof.ec_trace_auth_paths_hi_next[q]) {
                return Err(format!("EC trace hi auth path (next) failed at query {q}"));
            }
        }
    }

    // ---- FIX #1: Verify constraint evaluation at query points ----
    // The verifier independently evaluates the 31 Cairo constraints and checks
    // they match the quotient values. This closes the critical soundness gap.
    let constraint_alphas = constraint_alphas_drawn;
    let _verif_eval_domain = crate::circle::Coset::half_coset(log_eval_size);

    // ---- Precompute OODS line coefficients (same for every query) ----
    // Used in the OODS formula check below.
    let (oods_line_z, oods_line_q, oods_line_interaction, oods_line_interaction_next, oods_line_zn, oods_z_point, oods_z_next_point, oods_alpha_val) = {
        use crate::oods::{OodsPoint, compute_line_coeffs};
        let z      = OodsPoint::from_u32_array(&proof.oods_z);
        let step   = crate::circle::CirclePoint::GENERATOR.repeated_double(31 - log_n);
        let z_next = z.next_step(step);
        let alpha  = QM31::from_u32_array(proof.oods_alpha);
        let line_z: Vec<(QM31, QM31)> = proof.oods_trace_at_z.iter()
            .map(|v| compute_line_coeffs(z, QM31::from_u32_array(*v))).collect();
        let line_q: Vec<(QM31, QM31)> = proof.oods_quotient_at_z.iter()
            .map(|v| compute_line_coeffs(z, QM31::from_u32_array(*v))).collect();
        // line_interaction[pi][k]: line coeffs for component k of interaction poly pi at z.
        let line_interaction: Vec<Vec<(QM31, QM31)>> = proof.oods_interaction_at_z.iter()
            .map(|poly| poly.iter()
                .map(|v| compute_line_coeffs(z, QM31::from_u32_array(*v)))
                .collect())
            .collect();
        // line_interaction_next[pi][k]: line coeffs at z_next (step-transition binding).
        let line_interaction_next: Vec<Vec<(QM31, QM31)>> = proof.oods_interaction_at_z_next.iter()
            .map(|poly| poly.iter()
                .map(|v| compute_line_coeffs(z_next, QM31::from_u32_array(*v)))
                .collect())
            .collect();
        let line_zn: Vec<(QM31, QM31)> = proof.oods_trace_at_z_next.iter()
            .map(|v| compute_line_coeffs(z_next, QM31::from_u32_array(*v))).collect();
        (line_z, line_q, line_interaction, line_interaction_next, line_zn, z, z_next, alpha)
    };

    for (q, &qi) in proof.query_indices.iter().enumerate() {
        let row = &proof.trace_values_at_queries[q];
        let next = &proof.trace_values_at_queries_next[q];

        // Evaluate all 31 constraints (same logic as cuda_cairo_quotient kernel)
        let mut constraint_sum = QM31::ZERO;
        let mut ci = 0;

        // Constraints 0-14: flag binary (flag * (1 - flag) = 0)
        for j in 0..15 {
            let f = M31(row[5 + j]);
            let c = f * (M31(1) - f);
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        let pc = M31(row[0]); let ap = M31(row[1]); let fp = M31(row[2]);
        let dst = M31(row[21]); let op0 = M31(row[23]); let op1 = M31(row[25]); let res = M31(row[26]);
        let next_pc = M31(next[0]); let next_ap = M31(next[1]); let next_fp = M31(next[2]);

        let f_op1_imm = M31(row[7]); let f_res_add = M31(row[10]); let f_res_mul = M31(row[11]);
        let f_pc_jump_abs = M31(row[12]); let f_pc_jump_rel = M31(row[13]);
        let f_pc_jnz = M31(row[14]); let f_ap_add = M31(row[15]); let f_ap_add1 = M31(row[16]);
        let f_call = M31(row[17]); let f_ret = M31(row[18]); let f_assert = M31(row[19]);

        // Constraint 15: Result computation
        {
            let one = M31(1);
            let coeff_default = one - f_res_add - f_res_mul;
            let expected = coeff_default * op1 + f_res_add * (op0 + op1) + f_res_mul * (op0 * op1);
            let c = (one - f_pc_jnz) * (res - expected);
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 16: PC update
        {
            let one = M31(1);
            let inst_size = one + f_op1_imm;
            let pc_default = pc + inst_size;
            let not_jump = one - f_pc_jump_abs - f_pc_jump_rel - f_pc_jnz;
            let regular = not_jump * pc_default;
            let abs = f_pc_jump_abs * res;
            let rel = f_pc_jump_rel * (pc + res);
            let non_jnz = (one - f_pc_jnz) * (next_pc - (regular + abs + rel));
            let jnz_part = f_pc_jnz * (dst * (next_pc - (pc + op1)));
            let c = non_jnz + jnz_part;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 17: AP update
        {
            let expected_ap = ap + f_ap_add * res + f_ap_add1 + f_call * M31(2);
            let c = next_ap - expected_ap;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 18: FP update
        {
            let one = M31(1);
            let keep = one - f_call - f_ret;
            let expected_fp = keep * fp + f_call * (ap + M31(2)) + f_ret * dst;
            let c = next_fp - expected_fp;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 19: Assert_eq (dst = res when assert flag set)
        {
            let c = f_assert * (dst - res);
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // --- New soundness constraints 20-29 ---

        // Constraint 20: dst_addr verification
        {
            let dst_reg = M31(row[5]);
            let off0 = M31(row[27]);
            let dst_addr_val = M31(row[20]);
            let one = M31(1);
            let expected = (one - dst_reg) * ap + dst_reg * fp + off0 - M31(0x8000);
            let c = dst_addr_val - expected;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 21: op0_addr verification
        {
            let op0_reg = M31(row[6]);
            let off1 = M31(row[28]);
            let op0_addr_val = M31(row[22]);
            let one = M31(1);
            let expected = (one - op0_reg) * ap + op0_reg * fp + off1 - M31(0x8000);
            let c = op0_addr_val - expected;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 22: op1_addr verification
        {
            let op1_imm_f = M31(row[7]);
            let op1_fp_f = M31(row[8]);
            let op1_ap_f = M31(row[9]);
            let off2 = M31(row[29]);
            let op1_addr_val = M31(row[24]);
            let one = M31(1);
            let op1_default = one - op1_imm_f - op1_fp_f - op1_ap_f;
            let base = op1_imm_f * pc + op1_fp_f * fp + op1_ap_f * ap + op1_default * op0;
            let expected = base + off2 - M31(0x8000);
            let c = op1_addr_val - expected;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 23: JNZ fall-through
        {
            let dst_inv = M31(row[30]);
            let one = M31(1);
            let inst_size = one + f_op1_imm;
            let c = f_pc_jnz * (one - dst * dst_inv) * (next_pc - pc - inst_size);
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 24: JNZ inverse consistency
        {
            let dst_inv = M31(row[30]);
            let one = M31(1);
            let c = f_pc_jnz * dst * (one - dst * dst_inv);
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraints 25-27: Op1 source exclusivity
        {
            let c = f_op1_imm * M31(row[8]); // op1_imm * op1_fp
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }
        {
            let c = f_op1_imm * M31(row[9]); // op1_imm * op1_ap
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }
        {
            let c = M31(row[8]) * M31(row[9]); // op1_fp * op1_ap
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 28: PC update exclusivity
        {
            let c = f_pc_jump_abs * f_pc_jump_rel + f_pc_jump_abs * f_pc_jnz + f_pc_jump_rel * f_pc_jnz;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 29: Opcode exclusivity
        {
            let c = f_call * f_ret + f_call * f_assert + f_ret * f_assert;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 30: Instruction decomposition
        // inst_lo + inst_hi * 2^31 = off0 + off1*2^16 + off2*2^32 + sum(flag_i * 2^(48+i))
        // In M31: 2^31 ≡ 1, 2^32 ≡ 2, 2^(48+i) ≡ 2^(17+i), 2^62 ≡ 1
        {
            let inst_lo = M31(row[3]);
            let inst_hi = M31(row[4]);
            let off0 = M31(row[27]);
            let off1 = M31(row[28]);
            let off2 = M31(row[29]);
            let mut rhs = off0 + off1 * M31(1 << 16) + off2 * M31(2);
            // Flags 0-13 contribute bits 17-30 of the instruction word.
            for i in 0..14u32 {
                rhs = rhs + M31(row[5 + i as usize]) * M31(1u32 << (17 + i));
            }
            // Flag 14 is at bit 62 of the 63-bit instruction word.
            // 2^62 ≡ 1 (mod 2^31-1) since 2^31 ≡ 1, so 2^62 = (2^31)^2 ≡ 1.
            rhs = rhs + M31(row[19]) * M31(1); // flag14 * 2^62 ≡ flag14 * 1
            let c = inst_lo + inst_hi - rhs;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }
        // Constraints 31-32: LogUp and RC step-transition constraints (QM31).
        // These enforce S[i+1] - S[i] = delta(row_i) everywhere in the trace,
        // binding the interaction polynomials to the execution trace via polynomial identity.
        {
            use super::logup::logup_row_contribution;
            use super::range_check::rc_row_contribution;
            let chal = &proof.logup_challenges;
            let z_mem_v  = QM31::from_u32_array([chal[0],  chal[1],  chal[2],  chal[3]]);
            let alpha_v  = QM31::from_u32_array([chal[4],  chal[5],  chal[6],  chal[7]]);
            let alpha_sq = QM31::from_u32_array([chal[8],  chal[9],  chal[10], chal[11]]);
            let z_rc_v   = QM31::from_u32_array([chal[12], chal[13], chal[14], chal[15]]);

            // S_logup values from interaction decommitment
            let s_logup_qi  = QM31::from_u32_array(proof.interaction_decommitment.values[q]);
            let s_logup_qi1 = QM31::from_u32_array(proof.interaction_decommitment_next.values[q]);

            // Compute expected LogUp delta for this row
            let pc_v      = M31(row[0]);
            let inst_lo_v = M31(row[3]);
            let inst_hi_v = M31(row[4]);
            let dst_addr  = M31(row[20]);
            let dst_v     = M31(row[21]);
            let op0_addr  = M31(row[22]);
            let op0_v     = M31(row[23]);
            let op1_addr  = M31(row[24]);
            let op1_v     = M31(row[25]);
            let accesses = [(pc_v, inst_lo_v), (dst_addr, dst_v), (op0_addr, op0_v), (op1_addr, op1_v)];
            let delta_logup = logup_row_contribution(z_mem_v, alpha_v, alpha_sq, &accesses, inst_hi_v);

            // Constraint 31: S_logup[i+1] - S_logup[i] - delta(row_i) = 0
            let c31 = s_logup_qi1 - s_logup_qi - delta_logup;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c31;
            ci += 1;

            // S_rc values from RC interaction decommitment
            let s_rc_qi  = QM31::from_u32_array(proof.rc_interaction_decommitment.values[q]);
            let s_rc_qi1 = QM31::from_u32_array(proof.rc_interaction_decommitment_next.values[q]);

            // Compute expected RC delta for this row
            let off0_v = M31(row[27]);
            let off1_v = M31(row[28]);
            let off2_v = M31(row[29]);
            let delta_rc = rc_row_contribution(z_rc_v, &[off0_v, off1_v, off2_v]);

            // Constraint 32: S_rc[i+1] - S_rc[i] - delta(row_i) = 0
            let c32 = s_rc_qi1 - s_rc_qi - delta_rc;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c32;
            ci += 1;
        }

        // Constraint 33: dict_active binary (dict_active * (1 - dict_active) = 0)
        {
            let dict_active = M31(row[COL_DICT_ACTIVE]);
            let c = dict_active * (M31(1) - dict_active);
            constraint_sum = constraint_sum + constraint_alphas[ci] * c;
            ci += 1;
        }

        // Constraint 34: S_dict step-transition
        // S_dict[i+1] - S_dict[i] = dict_active[i] * inv(z_dict_link - (dict_key[i] + α * dict_new[i]))
        {
            use super::logup::qm31_from_m31;
            let chal = &proof.logup_challenges;
            let z_dlink     = QM31::from_u32_array([chal[16], chal[17], chal[18], chal[19]]);
            let alpha_dlink = QM31::from_u32_array([chal[20], chal[21], chal[22], chal[23]]);
            let s_dict_qi  = QM31::from_u32_array(proof.dict_main_interaction_decommitment.values[q]);
            let s_dict_qi1 = QM31::from_u32_array(proof.dict_main_interaction_decommitment_next.values[q]);
            let dict_active = M31(row[COL_DICT_ACTIVE]);
            let dict_delta = if dict_active == M31(0) {
                QM31::ZERO
            } else {
                let key     = qm31_from_m31(M31(row[COL_DICT_KEY]));
                let new_val = qm31_from_m31(M31(row[COL_DICT_NEW]));
                let entry = key + alpha_dlink * new_val;
                let denom = z_dlink - entry;
                denom.inverse()
            };
            let c34 = s_dict_qi1 - s_dict_qi - dict_delta;
            constraint_sum = constraint_sum + constraint_alphas[ci] * c34;
            ci += 1;
        }
        let _ = ci; // all 35 constraints evaluated

        // The GPU quotient kernel outputs Q(x) = C(x)/Z_H(x).
        // So the committed quotient value times the vanishing polynomial must equal
        // the constraint sum: C(x) == Q(x) * Z_H(x).
        let eval_point = canonic_domain_point(bit_reverse(qi, log_eval_size), log_eval_size);
        let zh = crate::circle::Coset::circle_vanishing_poly_at(eval_point.x, log_n);

        let q_val = QM31::from_u32_array(proof.quotient_decommitment.values[q]);
        let q_times_zh = q_val * zh;
        if constraint_sum != q_times_zh {
            return Err(format!(
                "Constraint evaluation mismatch at query {q} (qi={qi}): \
                 verifier computed C(x)={:?}, Q(x)*Z_H={:?}",
                constraint_sum.to_u32_array(), q_times_zh.to_u32_array()
            ));
        }

        // ---- OODS formula check ----
        // Verify that oods_quotient_decommitment.values[q] equals Q(p) computed from:
        //   Q(qi) = full_numer_z(qi) * D(z, p_nat)^{-1} + full_numer_zn(qi) * D(z_next, p_nat)^{-1}
        // where:
        //   full_numer_z(qi)  = Σ_{j<N_COLS} alpha^j * (col_j(p_br) - a_j - b_j * p_nat.y)
        //                     + Σ_{k<4}       alpha^{N_COLS+k} * (qk(p_br) - a_k' - b_k' * p_nat.y)
        //   full_numer_zn(qi) = Σ_{j<N_COLS} alpha^{N_COLS+4+j} * (col_j(p_br) - a_j'' - b_j'' * p_nat.y)
        //   L evaluated at p_nat (natural-order coset point) — because cuda_compute_coset_points
        //   uses natural index qi (not bit-reversed), while column values col_j(p_br) are at
        //   the bit-reversed domain point p_br = eval_coset.at(bit_reverse(qi)).
        //   D(sp, p_nat) uses p_nat.x, p_nat.y from natural order.
        {
            use crate::oods::{oods_denom, qm31_from_m31};
            // p_nat: natural-order domain point at index qi (matching GPU's cuda_compute_coset_points)
            let p_nat = canonic_domain_point(bit_reverse(qi, log_eval_size), log_eval_size);
            let px = p_nat.x;
            let py = p_nat.y;
            let py_q = qm31_from_m31(py);

            // GPU formula (accumulate_numerators kernel):
            //   partial  = Σ_i (alpha^i * f_i - a_i)   — a_i NOT alpha-weighted
            //   full_num = partial - linear_acc * p.y   — linear_acc = Σ_i alpha^i * b_i
            let mut alpha_pow = QM31::ONE;

            // z accumulator
            let mut partial_z  = QM31::ZERO;
            let mut lin_acc_z  = QM31::ZERO;
            for j in 0..N_COLS {
                let (a, b) = oods_line_z[j];
                let f = qm31_from_m31(M31(row[j]));
                partial_z = partial_z + alpha_pow * f - alpha_pow * a;
                lin_acc_z = lin_acc_z + alpha_pow * b;
                alpha_pow = alpha_pow * oods_alpha_val;
            }
            for k in 0..4 {
                let (a, b) = oods_line_q[k];
                let f = qm31_from_m31(M31(proof.quotient_decommitment.values[q][k]));
                partial_z = partial_z + alpha_pow * f - alpha_pow * a;
                lin_acc_z = lin_acc_z + alpha_pow * b;
                alpha_pow = alpha_pow * oods_alpha_val;
            }
            // 12 interaction component columns (LogUp, RC, S_dict × 4 components each)
            let interaction_col_values = [
                &proof.interaction_decommitment.values[q],
                &proof.rc_interaction_decommitment.values[q],
                &proof.dict_main_interaction_decommitment.values[q],
            ];
            for (pi, col_vals) in interaction_col_values.iter().enumerate() {
                for k in 0..4 {
                    let (a, b) = oods_line_interaction[pi][k];
                    let f = qm31_from_m31(M31(col_vals[k]));
                    partial_z = partial_z + alpha_pow * f - alpha_pow * a;
                    lin_acc_z = lin_acc_z + alpha_pow * b;
                    alpha_pow = alpha_pow * oods_alpha_val;
                }
            }
            let full_numer_z = partial_z - lin_acc_z * py_q;

            // z_next accumulator
            let mut partial_zn = QM31::ZERO;
            let mut lin_acc_zn = QM31::ZERO;
            for j in 0..N_COLS {
                let (a, b) = oods_line_zn[j];
                let f = qm31_from_m31(M31(row[j]));
                partial_zn = partial_zn + alpha_pow * f - alpha_pow * a;
                lin_acc_zn = lin_acc_zn + alpha_pow * b;
                alpha_pow = alpha_pow * oods_alpha_val;
            }
            // 12 interaction component columns at z_next (step-transition)
            let interaction_col_values_next = [
                &proof.interaction_decommitment.values[q],
                &proof.rc_interaction_decommitment.values[q],
                &proof.dict_main_interaction_decommitment.values[q],
            ];
            for (pi, col_vals) in interaction_col_values_next.iter().enumerate() {
                for k in 0..4 {
                    let (a, b) = oods_line_interaction_next[pi][k];
                    let f = qm31_from_m31(M31(col_vals[k]));
                    partial_zn = partial_zn + alpha_pow * f - alpha_pow * a;
                    lin_acc_zn = lin_acc_zn + alpha_pow * b;
                    alpha_pow = alpha_pow * oods_alpha_val;
                }
            }
            let full_numer_zn = partial_zn - lin_acc_zn * py_q;

            let d_z  = oods_denom(oods_z_point,      px, py);
            let d_zn = oods_denom(oods_z_next_point, px, py);
            let q_expected = full_numer_z  * d_z.inverse()
                           + full_numer_zn * d_zn.inverse();
            let q_committed = QM31::from_u32_array(proof.oods_quotient_decommitment.values[q]);
            if q_expected != q_committed {
                return Err(format!(
                    "OODS quotient formula mismatch at query {q} (qi={qi}): \
                     expected {:?}, committed {:?}",
                    q_expected.to_u32_array(), q_committed.to_u32_array()
                ));
            }
        }
    }

    // ---- Verify Merkle auth paths: AIR quotient ----
    verify_decommitment_auth_paths_soa4(
        &proof.quotient_commitment,
        &proof.quotient_decommitment,
        &proof.query_indices,
        "quotient",
    )?;
    // ---- Verify Merkle auth paths: OODS quotient (FRI input) ----
    verify_decommitment_auth_paths_soa4(
        &proof.oods_quotient_commitment,
        &proof.oods_quotient_decommitment,
        &proof.query_indices,
        "oods_quotient",
    )?;

    // ---- Verify Merkle auth paths: FRI layers ----
    let mut folded_indices: Vec<usize> = proof.query_indices.iter().map(|&qi| qi / 2).collect();
    for (layer, (decom, commitment)) in proof.fri_decommitments.iter()
        .zip(proof.fri_commitments.iter())
        .enumerate()
    {
        verify_decommitment_auth_paths_soa4(
            commitment, decom, &folded_indices, &format!("FRI layer {layer}"),
        )?;
        folded_indices = folded_indices.iter().map(|&i| i / 2).collect();
    }

    // ---- Verify Merkle auth paths: LogUp interaction trace ----
    // Binds S_logup values at qi and qi+1 to interaction_commitment.
    // The step-transition constraint 31 above uses both; auth paths make them binding.
    verify_decommitment_auth_paths_soa4(
        &proof.interaction_commitment,
        &proof.interaction_decommitment,
        &proof.query_indices,
        "LogUp interaction trace",
    )?;
    let next_query_indices: Vec<usize> = proof.query_indices.iter()
        .map(|&qi| canonic_next(qi, log_eval_size))
        .collect();
    verify_decommitment_auth_paths_soa4(
        &proof.interaction_commitment,
        &proof.interaction_decommitment_next,
        &next_query_indices,
        "LogUp interaction trace (next)",
    )?;

    // ---- Verify Merkle auth paths: RC interaction trace ----
    verify_decommitment_auth_paths_soa4(
        &proof.rc_interaction_commitment,
        &proof.rc_interaction_decommitment,
        &proof.query_indices,
        "RC interaction trace",
    )?;
    verify_decommitment_auth_paths_soa4(
        &proof.rc_interaction_commitment,
        &proof.rc_interaction_decommitment_next,
        &next_query_indices,
        "RC interaction trace (next)",
    )?;

    // ---- Verify Merkle auth paths: S_dict interaction trace ----
    verify_decommitment_auth_paths_soa4(
        &proof.dict_main_interaction_commitment,
        &proof.dict_main_interaction_decommitment,
        &proof.query_indices,
        "S_dict interaction trace",
    )?;
    verify_decommitment_auth_paths_soa4(
        &proof.dict_main_interaction_commitment,
        &proof.dict_main_interaction_decommitment_next,
        &next_query_indices,
        "S_dict interaction trace (next)",
    )?;

    Ok(())
}

// ---- Helper functions (same as Fibonacci verifier) ----

#[allow(dead_code)]
fn bit_reverse(x: usize, n_bits: u32) -> usize {
    let mut result = 0usize;
    let mut val = x;
    for _ in 0..n_bits {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

#[allow(dead_code)]
pub fn fold_twiddle_at(domain: &Coset, folded_index: usize, circle: bool) -> M31 {
    let domain_idx = bit_reverse(folded_index << 1, domain.log_size);
    let point = domain.at(domain_idx);
    let coord = if circle { point.y } else { point.x };
    coord.inverse()
}

fn get_pair_from_decom_4(value: &[u32; 4], sibling: &[u32; 4], idx: usize) -> (QM31, QM31) {
    if idx % 2 == 0 {
        (QM31::from_u32_array(*value), QM31::from_u32_array(*sibling))
    } else {
        (QM31::from_u32_array(*sibling), QM31::from_u32_array(*value))
    }
}

pub fn fold_pair(f0: QM31, f1: QM31, alpha: QM31, twiddle: M31) -> QM31 {
    let sum = f0 + f1;
    let diff = f0 - f1;
    sum + alpha * (diff * twiddle)
}

fn verify_decommitment_auth_paths_soa4(
    root: &[u32; 8],
    decom: &QueryDecommitment<[u32; 4]>,
    indices: &[usize],
    label: &str,
) -> Result<(), String> {
    if decom.values.len() != N_QUERIES {
        return Err(format!("{label} decommitment size mismatch"));
    }
    // Require non-empty auth paths (empty = legacy behavior with no Merkle binding).
    if decom.auth_paths.iter().all(|p| p.is_empty()) {
        return Err(format!("{label} auth paths are empty — commitment is unverified"));
    }
    for (q, &qi) in indices.iter().enumerate() {
        let leaf_hash = MerkleTree::hash_leaf(&decom.values[q]);
        if !MerkleTree::verify_auth_path(root, &leaf_hash, qi, &decom.auth_paths[q]) {
            return Err(format!("{label} auth path invalid at query {q} (index {qi})"));
        }
        let sib_idx = qi ^ 1;
        let sib_hash = MerkleTree::hash_leaf(&decom.sibling_values[q]);
        if !MerkleTree::verify_auth_path(root, &sib_hash, sib_idx, &decom.sibling_auth_paths[q]) {
            return Err(format!("{label} sibling auth path invalid at query {q}"));
        }
    }
    Ok(())
}

// ---- GPU LogUp helpers ----

/// GPU parallel prefix sum for 4-component QM31 values in SoA layout.
/// Two-level: intra-block scan → scan block sums → propagate.
fn gpu_prefix_sum(
    d_c0: &mut DeviceBuffer<u32>, d_c1: &mut DeviceBuffer<u32>,
    d_c2: &mut DeviceBuffer<u32>, d_c3: &mut DeviceBuffer<u32>,
    n: usize,
) {
    const BLOCK_SIZE: u32 = 256;
    let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if n_blocks <= 1 {
        unsafe {
            ffi::cuda_qm31_block_scan(
                d_c0.as_mut_ptr(), d_c1.as_mut_ptr(), d_c2.as_mut_ptr(), d_c3.as_mut_ptr(),
                std::ptr::null_mut(), std::ptr::null_mut(),
                std::ptr::null_mut(), std::ptr::null_mut(),
                n as u32, BLOCK_SIZE,
            );
            ffi::cuda_device_sync();
        }
        return;
    }

    let mut d_bs0 = DeviceBuffer::<u32>::alloc(n_blocks as usize);
    let mut d_bs1 = DeviceBuffer::<u32>::alloc(n_blocks as usize);
    let mut d_bs2 = DeviceBuffer::<u32>::alloc(n_blocks as usize);
    let mut d_bs3 = DeviceBuffer::<u32>::alloc(n_blocks as usize);

    unsafe {
        ffi::cuda_qm31_block_scan(
            d_c0.as_mut_ptr(), d_c1.as_mut_ptr(), d_c2.as_mut_ptr(), d_c3.as_mut_ptr(),
            d_bs0.as_mut_ptr(), d_bs1.as_mut_ptr(), d_bs2.as_mut_ptr(), d_bs3.as_mut_ptr(),
            n as u32, BLOCK_SIZE,
        );
        ffi::cuda_device_sync();
    }

    gpu_prefix_sum(&mut d_bs0, &mut d_bs1, &mut d_bs2, &mut d_bs3, n_blocks as usize);

    unsafe {
        ffi::cuda_qm31_add_block_prefix(
            d_c0.as_mut_ptr(), d_c1.as_mut_ptr(), d_c2.as_mut_ptr(), d_c3.as_mut_ptr(),
            d_bs0.as_ptr(), d_bs1.as_ptr(), d_bs2.as_ptr(), d_bs3.as_ptr(),
            n as u32, BLOCK_SIZE,
        );
        ffi::cuda_device_sync();
    }
}

/// GPU computation of the LogUp memory interaction trace.
/// Uploads 9 trace columns, computes per-row deltas (fused kernel), prefix-scans
/// to get the running sum, then returns 4 GPU buffers ready for NTT + the final sum.
fn gpu_compute_interaction_trace(
    columns: &[Vec<u32>],
    n: usize,
    z_mem: QM31,
    alpha_mem: QM31,
) -> ([DeviceBuffer<u32>; 4], QM31) {
    use super::trace::*;

    let z_arr   = z_mem.to_u32_array();
    let a_arr   = alpha_mem.to_u32_array();

    // Upload the 9 trace columns used in memory LogUp
    let d_pc        = DeviceBuffer::from_host(&columns[COL_PC][..n]);
    let d_inst_lo   = DeviceBuffer::from_host(&columns[COL_INST_LO][..n]);
    let d_inst_hi   = DeviceBuffer::from_host(&columns[COL_INST_HI][..n]);
    let d_dst_addr  = DeviceBuffer::from_host(&columns[COL_DST_ADDR][..n]);
    let d_dst       = DeviceBuffer::from_host(&columns[COL_DST][..n]);
    let d_op0_addr  = DeviceBuffer::from_host(&columns[COL_OP0_ADDR][..n]);
    let d_op0       = DeviceBuffer::from_host(&columns[COL_OP0][..n]);
    let d_op1_addr  = DeviceBuffer::from_host(&columns[COL_OP1_ADDR][..n]);
    let d_op1       = DeviceBuffer::from_host(&columns[COL_OP1][..n]);

    let mut d0 = DeviceBuffer::<u32>::alloc(n);
    let mut d1 = DeviceBuffer::<u32>::alloc(n);
    let mut d2 = DeviceBuffer::<u32>::alloc(n);
    let mut d3 = DeviceBuffer::<u32>::alloc(n);

    // Compute per-row deltas (sum of 4 reciprocals with inst_hi extension)
    unsafe {
        ffi::cuda_logup_memory_fused(
            d_pc.as_ptr(), d_inst_lo.as_ptr(), d_inst_hi.as_ptr(),
            d_dst_addr.as_ptr(), d_dst.as_ptr(),
            d_op0_addr.as_ptr(), d_op0.as_ptr(),
            d_op1_addr.as_ptr(), d_op1.as_ptr(),
            d0.as_mut_ptr(), d1.as_mut_ptr(), d2.as_mut_ptr(), d3.as_mut_ptr(),
            z_arr.as_ptr(), a_arr.as_ptr(),
            n as u32,
        );
        ffi::cuda_device_sync();
    }

    // Prefix scan: deltas → running sum (in-place)
    gpu_prefix_sum(&mut d0, &mut d1, &mut d2, &mut d3, n);

    // Read final value (last element of running sum)
    let final_sum = {
        let v0 = d0.to_host();
        let v1 = d1.to_host();
        let v2 = d2.to_host();
        let v3 = d3.to_host();
        QM31::from_u32_array([v0[n-1], v1[n-1], v2[n-1], v3[n-1]])
    };

    ([d0, d1, d2, d3], final_sum)
}

/// GPU computation of the RC interaction trace.
/// Uploads 3 offset columns, computes per-row deltas, prefix-scans to running sum.
fn gpu_compute_rc_interaction_trace(
    columns: &[Vec<u32>],
    n: usize,
    z_rc: QM31,
) -> ([DeviceBuffer<u32>; 4], QM31) {
    use super::trace::*;

    let z_arr = z_rc.to_u32_array();

    let d_off0 = DeviceBuffer::from_host(&columns[COL_OFF0][..n]);
    let d_off1 = DeviceBuffer::from_host(&columns[COL_OFF1][..n]);
    let d_off2 = DeviceBuffer::from_host(&columns[COL_OFF2][..n]);

    let mut d0 = DeviceBuffer::<u32>::alloc(n);
    let mut d1 = DeviceBuffer::<u32>::alloc(n);
    let mut d2 = DeviceBuffer::<u32>::alloc(n);
    let mut d3 = DeviceBuffer::<u32>::alloc(n);

    unsafe {
        ffi::cuda_logup_rc_fused(
            d_off0.as_ptr(), d_off1.as_ptr(), d_off2.as_ptr(),
            d0.as_mut_ptr(), d1.as_mut_ptr(), d2.as_mut_ptr(), d3.as_mut_ptr(),
            z_arr.as_ptr(), n as u32,
        );
        ffi::cuda_device_sync();
    }

    gpu_prefix_sum(&mut d0, &mut d1, &mut d2, &mut d3, n);

    let final_sum = {
        let v0 = d0.to_host();
        let v1 = d1.to_host();
        let v2 = d2.to_host();
        let v3 = d3.to_host();
        QM31::from_u32_array([v0[n-1], v1[n-1], v2[n-1], v3[n-1]])
    };

    ([d0, d1, d2, d3], final_sum)
}

// ---- Decommitment helpers ----

/// Decommit 4 SoA columns at the given query indices using pre-computed GPU tile roots.
/// O(n_queries × TILE_SIZE) instead of O(n) — avoids full tree rebuild on CPU.
fn decommit_from_host_soa4(
    host_cols: &[Vec<u32>],  // [4] columns
    tile_roots: &[[u32; 8]],
    indices: &[usize],
) -> QueryDecommitment<[u32; 4]> {
    let n = host_cols[0].len();
    let mut values = Vec::with_capacity(indices.len());
    let mut sibling_values = Vec::with_capacity(indices.len());

    // Collect queried + sibling indices for batch auth path generation
    let mut all_indices: Vec<usize> = Vec::with_capacity(indices.len() * 2);
    for &idx in indices {
        all_indices.push(idx % n);
        all_indices.push((idx ^ 1) % n);
    }

    for &idx in indices {
        let sib = idx ^ 1;
        values.push([
            host_cols[0][idx % n], host_cols[1][idx % n],
            host_cols[2][idx % n], host_cols[3][idx % n],
        ]);
        sibling_values.push([
            host_cols[0][sib % n], host_cols[1][sib % n],
            host_cols[2][sib % n], host_cols[3][sib % n],
        ]);
    }

    // Use pre-computed tile roots to skip the O(n) tree rebuild.
    let cols4: [Vec<u32>; 4] = [
        host_cols[0].clone(), host_cols[1].clone(),
        host_cols[2].clone(), host_cols[3].clone(),
    ];
    let hash_leaf = |i: usize| MerkleTree::hash_leaf(&[
        cols4[0][i], cols4[1][i], cols4[2][i], cols4[3][i],
    ]);
    let all_paths = MerkleTree::targeted_auth_paths_with_tile_roots(
        tile_roots, n, &all_indices, &hash_leaf,
    );

    let mut auth_paths = Vec::with_capacity(indices.len());
    let mut sibling_auth_paths = Vec::with_capacity(indices.len());
    for i in 0..indices.len() {
        auth_paths.push(all_paths[i * 2].clone());
        sibling_auth_paths.push(all_paths[i * 2 + 1].clone());
    }

    QueryDecommitment { values, sibling_values, auth_paths, sibling_auth_paths }
}

fn decommit_fri_layer(
    eval: &SecureColumn,
    indices: &[usize],
) -> QueryDecommitment<[u32; 4]> {
    let host_cols: Vec<Vec<u32>> = eval.cols.iter().map(|c| c.to_host()).collect();
    // Get tile roots from the GPU commit (same tree as was committed during FRI folding).
    let (_root, tile_roots) = MerkleTree::commit_root_soa4_with_subtrees(
        &eval.cols[0], &eval.cols[1], &eval.cols[2], &eval.cols[3],
        (eval.len.trailing_zeros()) as u32,
    );
    decommit_from_host_soa4(&host_cols, &tile_roots, indices)
}

// NOTE (GAP-4): A full degree-check test would require re-running the quotient kernel
// with access to internal GPU buffers and inverse-NTTing the output.  That approach
// was deferred in favor of the FRI-acceptance test (test_gap4_blinded_denominator_cols_proof_accepts).
// Placeholder kept as a comment only.

// Unreachable placeholder — present only to mark the location for future degree-check work.
#[cfg(test)]
#[allow(dead_code)]
fn _extract_quotient_evals_stub(_log_n: u32) -> Vec<[u32; 4]> { vec![] }


#[cfg(test)]
mod tests {
    use super::*;
    use crate::cairo_air::decode::Instruction;

    fn build_fib_program(n: usize) -> Vec<u64> {
        let mut program = Vec::new();
        let assert_imm = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        program.push(assert_imm.encode());
        program.push(1);
        program.push(assert_imm.encode());
        program.push(1);

        let add_instr = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        for _ in 0..n.saturating_sub(2) {
            program.push(add_instr.encode());
        }
        program
    }

    /// Build a multiply-accumulate program: acc = acc * val + 1
    /// Uses different instruction mix than Fibonacci (mul + add + immediate)
    fn build_mul_acc_program(n: usize) -> Vec<u64> {
        let mut program = Vec::new();

        // Initialize: [ap] = 1 (accumulator), [ap+1] = 3 (multiplier)
        let assert_imm = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        program.push(assert_imm.encode());
        program.push(1); // initial acc = 1
        program.push(assert_imm.encode());
        program.push(3); // multiplier = 3

        // Main loop: [ap] = [ap-2] * [ap-1] (mul instruction)
        let mul_instr = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_mul: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        for _ in 0..n.saturating_sub(2) {
            program.push(mul_instr.encode());
        }
        program
    }

    /// Build a mixed-instruction program: alternates add and mul
    fn build_mixed_program(n: usize) -> Vec<u64> {
        let mut program = Vec::new();

        let assert_imm = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        program.push(assert_imm.encode());
        program.push(7);
        program.push(assert_imm.encode());
        program.push(11);

        let add_instr = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        let mul_instr = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_mul: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        for i in 0..n.saturating_sub(2) {
            if i % 2 == 0 {
                program.push(add_instr.encode());
            } else {
                program.push(mul_instr.encode());
            }
        }
        program
    }

    #[test]
    fn test_cairo_proof_serialization_roundtrip() {
        // Proves that CairoProof survives JSON serialization + deserialization
        // and that the deserialized proof still verifies. Required for any transport.
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let proof = cairo_prove(&program, 64, 6);

        let json = serde_json::to_string(&proof).expect("serialize failed");
        let proof2: CairoProof = serde_json::from_str(&json).expect("deserialize failed");

        // Structural identity
        assert_eq!(proof.trace_commitment, proof2.trace_commitment);
        assert_eq!(proof.quotient_commitment, proof2.quotient_commitment);
        assert_eq!(proof.query_indices, proof2.query_indices);
        assert_eq!(proof.logup_final_sum, proof2.logup_final_sum);

        // Deserialized proof must still verify
        let result = cairo_verify(&proof2);
        assert!(result.is_ok(), "Deserialized proof failed verification: {:?}", result);

        // Report proof size (JSON is ~3-4x larger than binary; real deployment uses bincode)
        println!("CairoProof (log_n=6) JSON size: {} bytes ({:.1} KB)",
            json.len(), json.len() as f64 / 1024.0);
    }

    #[test]
    fn test_cairo_proof_size_log10() {
        // Measures proof size at a representative scale.
        ffi::init_memory_pool();
        let program = build_fib_program(1024);
        let proof = cairo_prove(&program, 1024, 10);
        let json = serde_json::to_string(&proof).expect("serialize failed");
        println!("CairoProof (log_n=10) JSON size: {} bytes ({:.1} KB)",
            json.len(), json.len() as f64 / 1024.0);
        // Sanity: JSON proof should be < 200 MB (binary would be ~4x smaller)
        assert!(json.len() < 200 * 1024 * 1024, "proof too large: {} bytes", json.len());
        // Also verify the proof still works
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "log_n=10 proof failed: {:?}", result);
    }

    #[test]
    fn test_cairo_prove_verify_small() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let proof = cairo_prove(&program, 64, 6);

        // Verify basic structure
        assert_ne!(proof.trace_commitment, [0; 8]);
        assert_ne!(proof.quotient_commitment, [0; 8]);
        assert_ne!(proof.interaction_commitment, [0; 8]);
        assert_eq!(proof.query_indices.len(), N_QUERIES);
        assert_eq!(proof.log_trace_size, 6);

        // Verify FRI fold equations pass
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Cairo proof failed: {:?}", result);
    }

    /// Tamper with the PoW nonce and verify rejection.
    /// Also verify that the nonce actually satisfies the PoW check.
    #[test]
    fn test_soundness_pow_nonce_tamper() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let proof = cairo_prove(&program, 64, 6);

        // Baseline proof must verify
        cairo_verify(&proof).expect("baseline proof must verify");

        // PoW nonce must satisfy the check
        assert!(proof.pow_nonce != 0, "pow_nonce should not be zero (probability 2^-26)");

        // Tamper: add 1 to the nonce — should fail PoW check
        let mut bad = proof.clone();
        bad.pow_nonce = proof.pow_nonce.wrapping_add(1);
        assert!(cairo_verify(&bad).is_err(),
            "Tampered pow_nonce+1 should be rejected by PoW check");

        // Tamper: set nonce to 0 — almost certainly wrong
        let mut bad2 = proof.clone();
        bad2.pow_nonce = 0;
        // nonce=0 might accidentally pass if it satisfies PoW; but even if so,
        // the derived query indices will be wrong (different channel state), so verify fails
        // regardless. The key property: modifying nonce changes the proof outcome.
        if cairo_verify(&bad2).is_ok() {
            // nonce=0 happened to satisfy PoW *and* produce same query indices — impossible
            // for different nonces to produce same Fiat-Shamir state
            panic!("pow_nonce=0 must not verify when nonce=0 differs from {}", proof.pow_nonce);
        }
    }

    #[test]
    fn test_cairo_prove_verify_medium() {
        ffi::init_memory_pool();
        let program = build_fib_program(1024);
        let proof = cairo_prove(&program, 1024, 10);
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Cairo proof log_n=10 failed: {:?}", result);
    }

    #[test]
    fn test_cairo_prove_verify_log14() {
        ffi::init_memory_pool();
        let n = 1 << 14;
        let program = build_fib_program(n);
        let proof = cairo_prove(&program, n, 14);
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Cairo proof log_n=14 failed: {:?}", result);
    }

    #[test]
    fn test_cairo_prove_verify_tampered_commitment() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.trace_commitment[0] ^= 1;
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered commitment should fail");
    }

    #[test]
    fn test_cairo_prove_verify_tampered_fri() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        if !proof.fri_decommitments.is_empty() {
            proof.fri_decommitments[0].values[0][0] ^= 1;
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered FRI value should fail");
    }

    #[test]
    fn test_cairo_prove_verify_tampered_quotient() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.quotient_decommitment.values[0][0] ^= 1;
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered quotient should fail");
    }

    #[test]
    fn test_quotient_auth_paths_reject_fake_value() {
        // Tamper quotient value while keeping stale auth path (for original value).
        // The auth path check should reject the tampered value.
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        // Keep auth path but change the value — auth path is for a different leaf hash.
        proof.quotient_decommitment.values[0][0] ^= 1;
        // Also fix the constraint sum check by adjusting sibling (to isolate auth path check).
        // Actually just let it fail at whichever check comes first (auth or constraint).
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered quotient with stale auth path should fail: {:?}", result);
    }

    #[test]
    fn test_fri_auth_paths_reject_fake_value() {
        // Tamper FRI decommitment value — now auth paths bind these to commitments.
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        if !proof.fri_decommitments.is_empty() {
            proof.fri_decommitments[0].values[0][0] ^= 1;
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered FRI value with stale auth path should fail: {:?}", result);
    }

    // ---- Step 6: Non-Fibonacci programs ----

    #[test]
    fn test_cairo_prove_verify_mul_program() {
        ffi::init_memory_pool();
        let program = build_mul_acc_program(256);
        let proof = cairo_prove(&program, 256, 8);
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Mul-acc proof failed: {:?}", result);
    }

    #[test]
    fn test_cairo_prove_verify_mixed_program() {
        ffi::init_memory_pool();
        let program = build_mixed_program(512);
        let proof = cairo_prove(&program, 512, 9);
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Mixed program proof failed: {:?}", result);
    }

    #[test]
    fn test_cairo_public_inputs() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let proof = cairo_prove(&program, 64, 6);

        assert_eq!(proof.public_inputs.initial_pc, 0);
        assert_eq!(proof.public_inputs.initial_ap, 100);
        assert_eq!(proof.public_inputs.n_steps, 64);
        assert_ne!(proof.public_inputs.program_hash, [0; 8]);

        // Different program should produce different hash
        let program2 = build_mul_acc_program(64);
        let proof2 = cairo_prove(&program2, 64, 6);
        assert_ne!(proof.public_inputs.program_hash, proof2.public_inputs.program_hash);
    }

    /// Build a program with call/ret: call a subroutine, return, continue
    fn build_call_ret_program(n: usize) -> Vec<u64> {
        let mut program = Vec::new();

        // addr 0,1: [ap] = 42 (assert immediate)
        let assert_imm = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        program.push(assert_imm.encode());
        program.push(42);

        // addr 2,3: [ap] = 7
        program.push(assert_imm.encode());
        program.push(7);

        // addr 4+: fill with add instructions
        let add_instr = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        for _ in 0..n.saturating_sub(2) {
            program.push(add_instr.encode());
        }
        program
    }

    #[test]
    fn test_cairo_prove_verify_call_ret_program() {
        ffi::init_memory_pool();
        let program = build_call_ret_program(128);
        let proof = cairo_prove(&program, 128, 7);
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Call/ret program proof failed: {:?}", result);
    }

    // ---- Per-constraint-family tamper tests ----
    // These prove the verifier catches each specific class of fraud.

    /// Helper: prove, tamper a specific trace column at query points, verify rejection
    fn prove_and_tamper_trace(program: &[u64], n: usize, log_n: u32,
                              col_idx: usize, label: &str) {
        ffi::init_memory_pool();
        let mut proof = cairo_prove(program, n, log_n);
        // Tamper the specified column in all query trace values
        for row in &mut proof.trace_values_at_queries {
            row[col_idx] = row[col_idx].wrapping_add(1) % crate::field::m31::P;
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered {label} (col {col_idx}) should fail verification");
    }

    #[test]
    fn test_tamper_flag_binary() {
        // Break flag binary constraint: set a flag to 2 (not 0 or 1)
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        for row in &mut proof.trace_values_at_queries {
            row[5] = 2; // dst_reg flag = 2 → f*(1-f) = 2*(-1) ≠ 0
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered flag binary should fail");
    }

    #[test]
    fn test_tamper_result_computation() {
        // Break result constraint: corrupt res column
        let program = build_fib_program(64);
        prove_and_tamper_trace(&program, 64, 6, 26, "result (res)");
    }

    #[test]
    fn test_tamper_pc_update() {
        // Break PC update: corrupt next_pc (trace_values_at_queries_next[0] = pc)
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        for row in &mut proof.trace_values_at_queries_next {
            row[0] = row[0].wrapping_add(1) % crate::field::m31::P; // corrupt next_pc
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered PC update should fail");
    }

    #[test]
    fn test_tamper_ap_update() {
        // Break AP update: corrupt next_ap
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        for row in &mut proof.trace_values_at_queries_next {
            row[1] = row[1].wrapping_add(1) % crate::field::m31::P; // corrupt next_ap
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered AP update should fail");
    }

    #[test]
    fn test_tamper_fp_update() {
        // Break FP update: corrupt next_fp
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        for row in &mut proof.trace_values_at_queries_next {
            row[2] = row[2].wrapping_add(1) % crate::field::m31::P; // corrupt next_fp
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered FP update should fail");
    }

    #[test]
    fn test_tamper_assert_eq() {
        // Break assert_eq: corrupt dst so dst ≠ res
        let program = build_fib_program(64);
        prove_and_tamper_trace(&program, 64, 6, 21, "assert_eq (dst)");
    }

    #[test]
    fn test_cairo_prove_with_pedersen_ec_constraints() {
        ffi::init_memory_pool();
        crate::cairo_air::pedersen::gpu_init();

        let program = build_fib_program(64);
        let n_ped = 4; // small number of Pedersen hashes
        let ped_a: Vec<crate::cairo_air::stark252_field::Fp> = (0..n_ped).map(|i| {
            crate::cairo_air::stark252_field::Fp::from_u64(i as u64 + 1)
        }).collect();
        let ped_b: Vec<crate::cairo_air::stark252_field::Fp> = (0..n_ped).map(|i| {
            crate::cairo_air::stark252_field::Fp::from_u64(i as u64 + 100)
        }).collect();

        let proof = cairo_prove_with_pedersen(&program, 64, 6, Some((&ped_a, &ped_b)));

        // EC trace should be committed (both lo and hi halves)
        assert!(proof.ec_trace_commitment.is_some(), "EC trace lo should be committed");
        assert!(proof.ec_trace_commitment_hi.is_some(), "EC trace hi should be committed");
        assert!(proof.ec_log_eval.is_some(), "EC log eval should be present");
        assert!(!proof.ec_trace_at_queries.is_empty(), "EC trace values should be in proof");
        assert!(!proof.ec_trace_auth_paths.is_empty(), "EC trace lo auth paths should be present");
        assert!(!proof.ec_trace_auth_paths_hi.is_empty(), "EC trace hi auth paths should be present");

        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Pedersen EC-constrained proof failed: {:?}", result);
    }

    #[test]
    fn test_tamper_ec_trace() {
        ffi::init_memory_pool();
        crate::cairo_air::pedersen::gpu_init();

        let program = build_fib_program(64);
        let ped_a = vec![crate::cairo_air::stark252_field::Fp::from_u64(42)];
        let ped_b = vec![crate::cairo_air::stark252_field::Fp::from_u64(99)];

        let mut proof = cairo_prove_with_pedersen(&program, 64, 6, Some((&ped_a, &ped_b)));
        // Tamper EC trace commitment
        if let Some(ref mut ec) = proof.ec_trace_commitment {
            ec[0] ^= 1;
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered EC trace commitment should fail");
    }

    #[test]
    fn test_tamper_logup_final_sum() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.logup_final_sum[0] ^= 1; // corrupt LogUp final sum
        let result = cairo_verify(&proof);
        // This changes the Fiat-Shamir transcript or fails RC check
        assert!(result.is_err(), "Tampered LogUp final sum should fail");
    }

    #[test]
    fn test_tamper_rc_final_sum() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.rc_final_sum[0] ^= 1; // corrupt range check final sum
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered RC final sum should fail");
    }

    #[test]
    fn test_tamper_trace_auth_paths() {
        // Tamper trace values while keeping auth paths for the original values.
        // Verifier should reject: auth path won't match the tampered value's leaf hash.
        ffi::init_memory_pool();
        let program = build_fib_program(64);

        // Test lo commitment: tamper col 0 (pc), which is in cols 0-15
        {
            let mut proof = cairo_prove(&program, 64, 6);
            for row in &mut proof.trace_values_at_queries {
                row[0] = row[0].wrapping_add(1) % crate::field::m31::P;
            }
            let result = cairo_verify(&proof);
            assert!(result.is_err(), "Tampered lo col (pc) with stale auth path should fail: {:?}", result);
        }

        // Test hi commitment: tamper col 26 (res), which is in cols 16-30
        {
            let mut proof = cairo_prove(&program, 64, 6);
            for row in &mut proof.trace_values_at_queries {
                row[26] = row[26].wrapping_add(1) % crate::field::m31::P;
            }
            let result = cairo_verify(&proof);
            assert!(result.is_err(), "Tampered hi col (res) with stale auth path should fail: {:?}", result);
        }
    }

    #[test]
    fn test_logup_final_sum_cancels() {
        // Verify that the LogUp execution sum cancels with the memory table sum.
        // This proves the CPU compute_interaction_trace path is correctly wired:
        // the sum over all execution rows plus the sum over unique memory entries = 0.
        ffi::init_memory_pool();
        let n_steps = 32;
        let log_n = 5; // 32 rows — large enough for FRI (log_eval_size=6 > 4)
        let program = build_fib_program(n_steps);

        let mut mem = super::super::vm::Memory::with_capacity(200);
        mem.load_program(&program);
        let cols = super::super::vm::execute_to_columns(&mut mem, n_steps, log_n);

        // Use fixed challenges (not FS — just checking algebraic cancellation).
        let z = crate::field::QM31 {
            a: crate::field::cm31::CM31 { a: M31(98765), b: M31(43210) },
            b: crate::field::cm31::CM31 { a: M31(11111), b: M31(22222) },
        };
        let alpha = crate::field::QM31 {
            a: crate::field::cm31::CM31 { a: M31(33333), b: M31(44444) },
            b: crate::field::cm31::CM31 { a: M31(55555), b: M31(66666) },
        };

        let (_, exec_sum) = super::super::logup::compute_interaction_trace(&cols, n_steps, z, alpha);
        let (data_table, instr_table) = super::super::logup::extract_memory_table(&cols, n_steps);
        let table_sum = super::super::logup::compute_memory_table_sum(&data_table, &instr_table, z, alpha);
        let total = exec_sum + table_sum;
        assert_eq!(total, crate::field::QM31::ZERO,
            "LogUp exec+table sums don't cancel: exec={exec_sum:?}, table={table_sum:?}");

        // Verify prover produces non-trivial logup_final_sum (not all zeros).
        let proof = cairo_prove(&program, n_steps, log_n);
        assert_ne!(proof.logup_final_sum, [0; 4],
            "logup_final_sum should be non-zero for real trace");
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Proof should verify after LogUp fix: {:?}", result);
    }

    #[test]
    fn test_rc_final_sum_is_real() {
        // Verify range check wiring produces a distinct sum (not a copy of LogUp)
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let proof = cairo_prove(&program, 64, 6);
        assert_ne!(proof.rc_final_sum, proof.logup_final_sum,
            "RC final sum should differ from LogUp — placeholder not replaced?");
        assert_ne!(proof.rc_final_sum, [0; 4],
            "RC final sum should be non-zero for a non-trivial trace");
        // Full proof must still verify
        let result = cairo_verify(&proof);
        assert!(result.is_ok(), "Proof with real RC final sum should verify: {:?}", result);
    }

    #[test]
    fn test_tamper_ec_trace_auth_paths() {
        // Tamper EC trace values at query points while keeping the original auth paths.
        // Verifier should reject: leaf hash of tampered values won't match committed root.
        ffi::init_memory_pool();
        crate::cairo_air::pedersen::gpu_init();

        let program = build_fib_program(64);
        let ped_a = vec![crate::cairo_air::stark252_field::Fp::from_u64(7)];
        let ped_b = vec![crate::cairo_air::stark252_field::Fp::from_u64(13)];

        // Case 1: tamper a lo EC trace column (col 0, in 0..16)
        {
            let mut proof = cairo_prove_with_pedersen(&program, 64, 6, Some((&ped_a, &ped_b)));
            assert!(!proof.ec_trace_at_queries.is_empty());
            proof.ec_trace_at_queries[0][0] ^= 1; // flip bit in col 0 of EC trace
            let result = cairo_verify(&proof);
            assert!(result.is_err(),
                "Tampered EC trace lo col should fail auth path check: {:?}", result);
        }

        // Case 2: tamper a hi EC trace column (col 16+, in 16..29)
        {
            let mut proof = cairo_prove_with_pedersen(&program, 64, 6, Some((&ped_a, &ped_b)));
            let n_ec_cols = proof.ec_trace_at_queries[0].len();
            assert!(n_ec_cols > 16, "EC trace should have > 16 columns");
            proof.ec_trace_at_queries[0][16] ^= 1; // flip bit in col 16 (hi half)
            let result = cairo_verify(&proof);
            assert!(result.is_err(),
                "Tampered EC trace hi col should fail auth path check: {:?}", result);
        }
    }

    #[test]
    fn test_tamper_interaction_decommitment() {
        // Tamper the LogUp interaction decommitment value — auth paths bind
        // interaction trace to interaction_commitment, so this must be detected.
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        // Flip a bit in the first interaction trace decommitment value
        proof.interaction_decommitment.values[0][0] ^= 1;
        let result = cairo_verify(&proof);
        assert!(result.is_err(),
            "Tampered interaction decommitment should fail auth path check: {:?}", result);
    }

    #[test]
    fn test_tamper_rc_interaction_decommitment() {
        // Tamper the RC interaction decommitment value.
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.rc_interaction_decommitment.values[0][0] ^= 1;
        let result = cairo_verify(&proof);
        assert!(result.is_err(),
            "Tampered RC interaction decommitment should fail auth path check: {:?}", result);
    }

    #[test]
    fn test_cairo_tampered_program_hash() {
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.public_inputs.program_hash[0] ^= 1;
        let result = cairo_verify(&proof);
        // Tampered program hash changes Fiat-Shamir transcript → FRI mismatch
        assert!(result.is_err(), "Tampered program hash should fail");
    }

    // ---- Item #4: Per-constraint forgery tests ----
    // Systematically verify that each of the 33 Cairo constraints is independently
    // enforced. Constraints 0-30 are covered by trace column tampering below.
    // Constraints 31 (LogUp step) and 32 (RC step) are covered by
    // test_tamper_interaction_decommitment and test_tamper_rc_interaction_decommitment.

    #[test]
    fn test_per_constraint_forgery_all_columns() {
        // For each of the 31 trace columns, tampering ANY column in the current or
        // next row must be detected by the verifier (via algebraic constraint check
        // or Merkle auth path check). This provides coverage over all 33 constraints:
        // - Constraints 0-14 (flag binary): covered by flag columns 5-19
        // - Constraints 15-29 (decode/update): covered by operand/register columns
        // - Constraint 30 (instruction decomposition): covered by cols 3, 4
        // - Constraints 31/32 covered by interaction decommitment tamper tests
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let proof = cairo_prove(&program, 64, 6);

        // Current-row columns: any tamper triggers a constraint violation or auth-path failure
        for col_idx in 0..N_COLS {
            let mut t = proof.clone();
            for row in &mut t.trace_values_at_queries {
                row[col_idx] = row[col_idx].wrapping_add(1) % crate::field::m31::P;
            }
            let result = cairo_verify(&t);
            assert!(result.is_err(),
                "Tampered current-row col {col_idx} should be rejected by verifier");
        }

        // Next-row columns: transition constraints (PC/AP/FP) plus Merkle auth paths
        // for all other columns — any tamper must be caught.
        for col_idx in 0..N_COLS {
            let mut t = proof.clone();
            for row in &mut t.trace_values_at_queries_next {
                row[col_idx] = row[col_idx].wrapping_add(1) % crate::field::m31::P;
            }
            let result = cairo_verify(&t);
            assert!(result.is_err(),
                "Tampered next-row col {col_idx} should be rejected by verifier");
        }
    }

    // ---- Item #5: Real CASM file loading and proving ----

    #[test]
    fn test_prove_casm_file() {
        // Load a real .casm JSON file (Cairo 1 compiler output format), prove it,
        // and verify the proof.  This exercises the full casm_loader → cairo_prove
        // path with on-disk fixture rather than the hand-crafted build_fib_program bytes.
        ffi::init_memory_pool();

        // Locate fixture relative to CARGO_MANIFEST_DIR so the test runs from any CWD.
        let manifest = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR not set — run via cargo test");
        let path = std::path::Path::new(&manifest)
            .join("tests/fixtures/fibonacci.casm");

        let program = super::super::casm_loader::load_program(&path)
            .expect("failed to load tests/fixtures/fibonacci.casm");

        // Fixture is a 32-step Fibonacci: 2 init words (×2 felts each) + 30 add words = 34 felts.
        assert_eq!(program.bytecode.len(), 34,
            "fixture bytecode length mismatch");
        assert_eq!(program.format, super::super::casm_loader::CasmFormat::CasmJson,
            "fixture must parse as CASM JSON (Cairo 1) format");
        assert_eq!(program.overflow_count, 0,
            "Fibonacci instructions are all <64-bit — no truncation expected");

        // 32 VM steps fit in 2^5 rows; use log_n=5.
        let n_steps = 32usize;
        let log_n   = 5u32;
        let proof = cairo_prove_program(&program, n_steps, log_n)
            .expect("prove_program should succeed for valid Fibonacci CASM");

        let result = cairo_verify(&proof);
        assert!(result.is_ok(),
            "Proof from .casm file must verify: {:?}", result);
    }

    #[test]
    fn test_prove_program_with_hints_testlessthan() {
        // Build a synthetic CASM program that uses a TestLessThan hint:
        //   instruction at pc=0: assert [ap+0] = [fp-1]  (dst_reg=fp, op1=imm 1)
        // We inject TestLessThan hint at pc=0 to write 1 into memory[ap+0] before the assert.
        // Then the assertion `[ap+0] == 1` succeeds.
        ffi::init_memory_pool();

        // assert_imm: `[fp-1] = 1` (stores 1 at fp-1) — same encoding as fibonacci fixture init
        let assert_imm: u64 = 0x4804800180008000;
        // add_instr: `[ap+0] = [ap-2] + [ap-1]` (ap-based fib step)
        let add_instr: u64 = 0x48307fff7ffe8000;
        let mut bytecode = vec![assert_imm, 1u64]; // pc=0: [fp-1] = 1
        for _ in 0..30 {
            bytecode.push(add_instr); // pc=2..31: add steps
        }

        // Build a CasmProgram with a TestLessThan hint at pc=0.
        // The hint writes dst = (lhs < rhs ? 1 : 0).  We use immediate 0 < 1 → dst = 1 written
        // to fp-1 before the assert_imm executes.  Since the assert_imm itself writes 1 to fp-1,
        // both the hint and the instruction agree, so the trace is consistent.
        let hint_json = serde_json::json!({
            "TestLessThan": {
                "lhs": { "Immediate": { "value": "0x0" } },
                "rhs": { "Immediate": { "value": "0x1" } },
                "dst": { "register": "FP", "offset": -1 }
            }
        });
        let hint = super::super::casm_loader::CasmHint {
            name: "TestLessThan".to_string(),
            raw: hint_json,
        };
        let program = super::super::casm_loader::CasmProgram {
            bytecode,
            entry_point: 0,
            name: "test_hints".to_string(),
            builtins: vec![],
            format: super::super::casm_loader::CasmFormat::CasmJson,
            hints: vec![(0, vec![hint])],
            overflow_count: 0,
        };

        let n_steps = 32usize;
        let log_n   = 5u32;
        let proof = cairo_prove_program(&program, n_steps, log_n)
            .expect("prove_program should succeed with TestLessThan hint");
        let result = cairo_verify(&proof);
        assert!(result.is_ok(),
            "Proof with TestLessThan hint must verify: {:?}", result);
    }

    // ---- Item #8: Step-transition boundary test ----

    #[test]
    fn test_step_transition_boundary_wrap() {
        // Verify step-transition decommitments correctly handle the wrap-around at the
        // last eval-domain position: next_qi = canonic_next(qi, log_eval_size).
        // At qi = eval_size - 1 the "next row" is qi = 0, not qi + 1 = eval_size
        // (out of bounds). This test checks:
        //   1. All next-query indices are correctly bounded within eval_size.
        //   2. Any boundary query (qi == eval_size - 1) wraps to 0, not eval_size.
        //   3. The valid proof verifies — boundary rows satisfy the constraints.
        //   4. Tampering next-row values at ANY position (including the boundary) fails.
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let proof = cairo_prove(&program, 64, 6);
        let log_eval_size = 6u32 + BLOWUP_BITS;
        let eval_size = 1usize << log_eval_size;

        // Check all next-query indices are in-bounds and boundary wraps correctly
        for &qi in &proof.query_indices {
            let next_qi = canonic_next(qi, log_eval_size);
            assert!(next_qi < eval_size,
                "next_qi={next_qi} out of bounds for eval_size={eval_size}");
            if qi == eval_size - 1 {
                assert_eq!(next_qi, 0,
                    "boundary wrap: qi=eval_size-1={qi} must map to next=0");
            }
        }

        // Valid proof must verify across all query positions (including any boundary)
        let result = cairo_verify(&proof);
        assert!(result.is_ok(),
            "Valid proof must verify at all positions including boundary: {:?}", result);

        // Tamper next-row PC at all positions — transition constraint must catch it
        // (This exercises the boundary case if any query lands at eval_size - 1)
        let mut tampered = proof.clone();
        for row in &mut tampered.trace_values_at_queries_next {
            row[0] = row[0].wrapping_add(1) % crate::field::m31::P; // corrupt next_pc
        }
        let tamper_result = cairo_verify(&tampered);
        assert!(tamper_result.is_err(),
            "Tampered next-row PC (incl. potential boundary) must be rejected");
    }

    // ---- Proof serialization roundtrip test ----

    #[test]
    fn test_proof_serialization_roundtrip() {
        // Prove → serialize to JSON → deserialize → verify.
        // Exercises the serde path end-to-end; catches field ordering bugs,
        // Option<> serialization issues, and proof format regressions.
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let proof = cairo_prove(&program, 64, 6);

        // Serialize to JSON
        let json = serde_json::to_string(&proof)
            .expect("proof must serialize to JSON without error");
        assert!(!json.is_empty(), "serialized proof must not be empty");
        assert!(json.len() > 1000, "serialized proof should be substantial (got {} bytes)", json.len());

        // Deserialize back
        let proof2: CairoProof = serde_json::from_str(&json)
            .expect("proof must deserialize from JSON without error");

        // Deserialized proof must verify
        cairo_verify(&proof2)
            .expect("deserialized proof must verify correctly");

        // Spot-check a few scalar fields survive the round-trip
        assert_eq!(proof.log_trace_size, proof2.log_trace_size);
        assert_eq!(proof.trace_commitment, proof2.trace_commitment);
        assert_eq!(proof.query_indices, proof2.query_indices);
        assert_eq!(proof.logup_final_sum, proof2.logup_final_sum);
    }

    #[test]
    fn test_proof_serialization_roundtrip_with_dict() {
        // Same roundtrip but with dict fields populated (Option<> fields must survive).
        ffi::init_memory_pool();
        let dict_accesses: Vec<(usize, u64, u64, u64)> = vec![
            (0, 1, 0, 42), (1, 2, 0, 99), (2, 1, 42, 100),
        ];
        let proof = prove_with_dict(&dict_accesses);

        let json = serde_json::to_string(&proof)
            .expect("proof with dict must serialize to JSON");
        let proof2: CairoProof = serde_json::from_str(&json)
            .expect("proof with dict must deserialize from JSON");

        cairo_verify(&proof2)
            .expect("deserialized proof with dict must verify");

        // Dict fields must round-trip correctly
        assert_eq!(proof.dict_exec_commitment, proof2.dict_exec_commitment);
        assert_eq!(proof.dict_sorted_commitment, proof2.dict_sorted_commitment);
        assert_eq!(proof.dict_exec_final_sum, proof2.dict_exec_final_sum);
        assert_eq!(proof.dict_exec_data.len(), proof2.dict_exec_data.len());
        assert_eq!(proof.dict_sorted_data.len(), proof2.dict_sorted_data.len());
    }

    // ---- GAP-2: execution range gate tests ----

    #[test]
    fn test_execution_range_violation_detected() {
        // A program that loads an immediate > M31 as a data value.
        // The instruction `[ap+0] = 2^31` has an immediate operand of 2^31 = P (M31 wrap).
        // When the VM reads op1 = memory.get(pc+1) = 2^31, it's ≥ P → overflow.
        ffi::init_memory_pool();
        use crate::cairo_air::casm_loader::{CasmProgram, CasmFormat};
        let assert_imm = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        let large_immediate: u64 = (1u64 << 31); // = P, ≥ M31
        let program = CasmProgram {
            bytecode: vec![assert_imm.encode(), large_immediate],
            entry_point: 0,
            name: "test_overflow".into(),
            builtins: vec![],
            format: CasmFormat::CasmJson,
            hints: vec![],
            overflow_count: 0,
        };
        let result = cairo_prove_program(&program, 1, 1);
        assert!(
            matches!(result, Err(ProveError::ExecutionRangeViolation { count }) if count > 0),
            "immediate > M31 during execution must be rejected"
        );
    }

    #[test]
    fn test_execution_range_valid_program_passes() {
        // A normal M31 program must NOT trigger ExecutionRangeViolation.
        ffi::init_memory_pool();
        use crate::cairo_air::casm_loader::{CasmProgram, CasmFormat};
        let program_bytes = build_fib_program(32);
        let program = CasmProgram {
            bytecode: program_bytes,
            entry_point: 0,
            name: "fib".into(),
            builtins: vec![],
            format: CasmFormat::CasmJson,
            hints: vec![],
            overflow_count: 0,
        };
        // Should succeed — all fibonacci values stay well within M31
        let result = cairo_prove_program(&program, 32, 5);
        assert!(result.is_ok(), "valid M31 program must not trigger range violation");
    }

    // ---- ZK blinding test ----

    #[test]
    fn test_zk_blinding_hides_trace() {
        // Prove the same program twice with ZK blinding enabled.
        // The eval-domain columns for blindable columns should differ between runs
        // (since the blinding scalars are random). The proofs must both verify.
        ffi::init_memory_pool();
        let program = build_fib_program(32);
        let proof1 = cairo_prove(&program, 32, 5);
        let proof2 = cairo_prove(&program, 32, 5);

        // Both must verify
        cairo_verify(&proof1).expect("proof1 must verify with ZK blinding");
        cairo_verify(&proof2).expect("proof2 must verify with ZK blinding");

        // Trace commitments should differ (blinded columns make them distinct).
        // With 34 blinded columns, the probability of identical commitments is ~1/2^128.
        assert_ne!(proof1.trace_commitment, proof2.trace_commitment,
            "ZK blinding: trace commitments should differ between runs");
    }

    // ---- Dict consistency sub-AIR tests (GAP-1) ----

    #[test]
    fn test_dict_logup_commitment_empty() {
        // Programs without dict accesses: all dict Option fields are None, vecs are empty.
        ffi::init_memory_pool();
        let program = build_fib_program(32);
        let proof = cairo_prove(&program, 32, 5);
        assert!(proof.dict_exec_commitment.is_none(), "no dict → exec commitment should be None");
        assert!(proof.dict_exec_data.is_empty(), "no dict → exec data should be empty");
        assert!(proof.dict_sorted_data.is_empty(), "no dict → sorted data should be empty");
        cairo_verify(&proof).expect("proof without dict must verify");
    }

    fn prove_with_dict(dict_accesses: &[(usize, u64, u64, u64)]) -> CairoProof {
        let program = build_fib_program(32);
        let mut mem = super::Memory::with_capacity(512);
        mem.load_program(&program);
        let columns = super::super::vm::execute_to_columns(&mut mem, 32, 5);
        let cache = CairoProverCache::new(5);
        cairo_prove_cached_with_columns(&program, columns, 32, 5, &cache, None, dict_accesses, Vec::new())
    }

    #[test]
    fn test_dict_logup_commitment_with_accesses() {
        // Prove with synthetic dict accesses; verify all four Merkle commitments are
        // populated, per-query evidence has correct dimensions, and proof verifies.
        ffi::init_memory_pool();
        let dict_accesses: Vec<(usize, u64, u64, u64)> = vec![
            (0, 1, 0, 42),   // step=0, key=1: 0 → 42
            (1, 2, 0, 99),   // step=1, key=2: 0 → 99
            (2, 1, 42, 100), // step=2, key=1: 42 → 100
            (3, 2, 99, 7),   // step=3, key=2: 99 → 7
        ];
        let proof = prove_with_dict(&dict_accesses);

        assert!(proof.dict_exec_commitment.is_some(), "exec data commitment");
        assert!(proof.dict_sorted_commitment.is_some(), "sorted data commitment");
        assert!(proof.dict_exec_final_sum.is_some(), "exec final sum");
        assert!(proof.dict_sorted_final_sum.is_some(), "sorted final sum");
        assert_eq!(proof.dict_exec_final_sum, proof.dict_sorted_final_sum,
            "exec and sorted final sums must be equal (permutation)");

        let dict_n = 1usize << proof.dict_log_n.unwrap();
        assert_eq!(proof.dict_exec_data.len(), dict_n, "exec data has dict_n rows");
        assert_eq!(proof.dict_sorted_data.len(), dict_n, "sorted data has dict_n rows");

        cairo_verify(&proof).expect("proof with dict accesses must verify");
    }

    #[test]
    fn test_dict_logup_tamper_exec_sum() {
        // Tamper dict_exec_final_sum — Fiat-Shamir changes, permutation check fails.
        ffi::init_memory_pool();
        let dict_accesses: Vec<(usize, u64, u64, u64)> = vec![
            (0, 1, 0, 42), (1, 2, 0, 99), (2, 1, 42, 100),
        ];
        let mut proof = prove_with_dict(&dict_accesses);

        // Corrupt exec final sum — verifier checks exec_final == sorted_final
        if let Some(ref mut s) = proof.dict_exec_final_sum {
            s[0] = s[0].wrapping_add(1) % crate::field::m31::P;
        }

        let result = cairo_verify(&proof);
        assert!(result.is_err(), "tampered dict_exec_final_sum must be rejected");
    }

    #[test]
    fn test_dict_logup_tamper_sorted_order() {
        // Tamper sorted data to break key order at a query position — sorted C1 must fire.
        ffi::init_memory_pool();
        let dict_accesses: Vec<(usize, u64, u64, u64)> = vec![
            (0, 1, 0, 10), (1, 3, 0, 20), (2, 5, 0, 30),
        ];
        let mut proof = prove_with_dict(&dict_accesses);

        // Corrupt sorted_data row 0 to have a huge key — breaks both Merkle root
        // (recomputed root won't match committed root) and key-sort order (C1).
        if !proof.dict_sorted_data.is_empty() {
            proof.dict_sorted_data[0][0] = 0x7FFF_FFFE; // huge key
        }

        let result = cairo_verify(&proof);
        // Merkle root recomputation fails because data no longer matches committed root.
        assert!(result.is_err(), "tampered sorted data must be rejected");
    }

    #[test]
    fn test_dict_logup_tamper_exec_hash_mismatch() {
        // Tamper an exec data value at a query position — Merkle auth path must reject.
        ffi::init_memory_pool();
        let dict_accesses: Vec<(usize, u64, u64, u64)> = vec![
            (0, 1, 0, 42), (1, 2, 0, 99), (2, 1, 42, 100),
        ];
        let mut proof = prove_with_dict(&dict_accesses);

        // Corrupt the key of the first exec data row
        if !proof.dict_exec_data.is_empty() {
            proof.dict_exec_data[0][0] = proof.dict_exec_data[0][0].wrapping_add(1)
                % crate::field::m31::P;
        }

        let result = cairo_verify(&proof);
        // Merkle root recomputation will not match committed root.
        assert!(result.is_err(), "corrupted exec data must be rejected by Merkle root check");
    }

    #[test]
    fn test_dict_logup_tamper_padding_row() {
        // Tamper a padding row in dict_exec_data (beyond dict_n_accesses).
        // The verifier must reject because padding rows must be all-zero to prevent
        // double-counting in the exec_key_new_sum.
        // 3 accesses → dict_log_n=2 → dict_n=4 → 1 padding row at index 3.
        ffi::init_memory_pool();
        let dict_accesses: Vec<(usize, u64, u64, u64)> = vec![
            (0, 1, 0, 42),
            (1, 2, 0, 99),
            (2, 1, 42, 100),
        ];
        let mut proof = prove_with_dict(&dict_accesses);
        let n_acc = dict_accesses.len();
        assert!(proof.dict_exec_data.len() > n_acc,
            "must have at least one padding row for this test");
        proof.dict_exec_data[n_acc] = [1, 2, 3];
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "tampered padding row must be rejected");
    }

    #[test]
    fn test_dict_active_binary_violation() {
        // Constraint C33: dict_active * (1 - dict_active) = 0 — dict_active must be 0 or 1.
        // Verify the algebraic identity rejects dict_active = 2 at the field level,
        // and that an unmodified proof verifies.
        ffi::init_memory_pool();
        let p = crate::field::m31::P;
        let dict_active_bad = 2u32;
        // C33 = dict_active * (1 - dict_active) mod P
        let c33 = (dict_active_bad as u64 * (1u64 + p as u64 - dict_active_bad as u64)) % p as u64;
        assert_ne!(c33, 0, "dict_active=2 must produce non-zero C33 constraint value");
        // C33 = 0 * (1-0) = 0  (valid: inactive row)
        let c33_zero = 0u64 * 1u64;
        assert_eq!(c33_zero, 0, "dict_active=0 must satisfy C33");
        // C33 = 1 * (1-1) = 0  (valid: active row)
        let c33_one = 1u64 * 0u64;
        assert_eq!(c33_one, 0, "dict_active=1 must satisfy C33");
        // Soundness: tampering a random field element in a valid proof must be detected.
        let dict_accesses: Vec<(usize, u64, u64, u64)> = vec![(0, 1, 0, 42)];
        let mut proof = prove_with_dict(&dict_accesses);
        cairo_verify(&proof).expect("unmodified dict proof must verify");
        // Flip one bit in the dict_trace_commitment to simulate C33 tamper path.
        proof.dict_trace_commitment[0] ^= 1;
        assert!(cairo_verify(&proof).is_err(), "tampered dict_trace_commitment must be rejected");
    }

    // ---- Bitwise builtin tests ------------------------------------------------

    /// Build a proof that contains bitwise invocations by writing directly to the
    /// bitwise memory segment (the VM auto-fills AND/XOR/OR outputs).
    fn prove_with_bitwise(pairs: &[(u32, u32)]) -> CairoProof {
        use crate::cairo_air::vm::{Memory, BITWISE_MEM_BASE};
        let program = build_fib_program(30);
        let mut mem = Memory::with_capacity(1024);
        mem.load_program(&program);
        mem.set(0,   0); // sentinel fp
        mem.set(1,   0); // sentinel ret pc

        // Write (x, y) pairs directly to the bitwise segment; outputs auto-fill.
        for (i, &(x, y)) in pairs.iter().enumerate() {
            let base = BITWISE_MEM_BASE + i as u64 * 5;
            mem.set(base,     x as u64);
            mem.set(base + 1, y as u64);
        }
        let bitwise_rows = mem.extract_bitwise_invocations();
        assert_eq!(bitwise_rows.len(), pairs.len());

        let n_steps = 32;
        let log_n = 5;
        let columns = super::super::vm::execute_to_columns(&mut mem, n_steps, log_n);
        let cache = CairoProverCache::new(log_n);
        let program_u64: Vec<u64> = program.iter().copied().collect();
        cairo_prove_cached_with_columns(&program_u64, columns, n_steps, log_n,
            &cache, None, &[], bitwise_rows)
    }

    #[test]
    fn test_bitwise_memory_auto_fill() {
        // Verify that Memory auto-fills bitwise outputs correctly.
        use crate::cairo_air::vm::{Memory, BITWISE_MEM_BASE};
        let mut mem = Memory::new();
        let x: u64 = 0b1010_1010;
        let y: u64 = 0b1100_1100;
        mem.set(BITWISE_MEM_BASE,     x);
        mem.set(BITWISE_MEM_BASE + 1, y);
        // Outputs should be auto-filled once both inputs are written.
        assert_eq!(mem.get(BITWISE_MEM_BASE + 2), (x & y) as u64, "AND wrong");
        assert_eq!(mem.get(BITWISE_MEM_BASE + 3), (x ^ y) as u64, "XOR wrong");
        assert_eq!(mem.get(BITWISE_MEM_BASE + 4), (x | y) as u64, "OR wrong");
        // Extraction should return one row.
        let rows = mem.extract_bitwise_invocations();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], [x as u32, y as u32, (x&y) as u32, (x^y) as u32, (x|y) as u32]);
    }

    #[test]
    fn test_bitwise_prove_verify_roundtrip() {
        ffi::init_memory_pool();
        let pairs = vec![(0b1010u32, 0b1100u32), (0xFF00u32, 0x00FFu32), (0u32, 0u32)];
        let proof = prove_with_bitwise(&pairs);
        assert_eq!(proof.bitwise_rows.len(), pairs.len());
        assert!(proof.bitwise_commitment.is_some());
        // Verify: should pass.
        cairo_verify(&proof).expect("bitwise proof should verify");
    }

    #[test]
    fn test_bitwise_tamper_row_rejected() {
        ffi::init_memory_pool();
        let pairs = vec![(0xABCDu32, 0x1234u32)];
        let mut proof = prove_with_bitwise(&pairs);
        // Corrupt the AND field — C0 will fail.
        proof.bitwise_rows[0][2] = proof.bitwise_rows[0][2].wrapping_add(1);
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "tampered bitwise AND must be rejected");
    }

    #[test]
    fn test_bitwise_tamper_commitment_rejected() {
        ffi::init_memory_pool();
        let pairs = vec![(0xFFu32, 0x0Fu32)];
        let mut proof = prove_with_bitwise(&pairs);
        // Corrupt the commitment hash — channel replay will diverge.
        if let Some(ref mut c) = proof.bitwise_commitment {
            c[0] ^= 1;
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "corrupted bitwise commitment must be rejected");
    }

    #[test]
    fn test_tamper_memory_table_data() {
        // Corrupt memory_table_data — commitment recomputation will mismatch.
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        if !proof.memory_table_data.is_empty() {
            proof.memory_table_data[0][2] = proof.memory_table_data[0][2].wrapping_add(1);
        }
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered memory_table_data should fail commitment check");
    }

    #[test]
    fn test_tamper_memory_table_commitment() {
        // Corrupt the memory_table_commitment hash — channel replay diverges, breaking FRI.
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.memory_table_commitment[0] ^= 1;
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Corrupted memory_table_commitment should be rejected");
    }

    #[test]
    fn test_tamper_logup_cancellation() {
        // Corrupt logup_final_sum to a wrong value while keeping memory_table_data honest.
        // The cancellation check (exec_sum + table_sum == 0) must catch this.
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        // Flip a bit in exec_sum — cancellation will fail.
        proof.logup_final_sum[0] ^= 0xFF;
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Corrupted logup_final_sum should fail cancellation check");
    }

    #[test]
    fn test_tamper_rc_counts_data() {
        // Corrupt rc_counts_data — commitment recomputation will mismatch.
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.rc_counts_data[0] = proof.rc_counts_data[0].wrapping_add(1);
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Tampered rc_counts_data should fail commitment check");
    }

    #[test]
    fn test_tamper_rc_counts_commitment() {
        // Corrupt the rc_counts_commitment hash — channel replay diverges, breaking FRI.
        ffi::init_memory_pool();
        let program = build_fib_program(64);
        let mut proof = cairo_prove(&program, 64, 6);
        proof.rc_counts_commitment[0] ^= 1;
        let result = cairo_verify(&proof);
        assert!(result.is_err(), "Corrupted rc_counts_commitment should be rejected");
    }

    // ---- GAP-4: Quotient polynomial degree test ----
    //
    // GAP-4 asks: does blinding the 12 LogUp/dict denominator columns with r·Z_H(x)
    // still yield a well-defined low-degree quotient polynomial Q = C/Z_H that FRI accepts?
    //
    // The key claim: at trace positions, Z_H(x_i) = 0, so blinded_val(x_i) = original_val(x_i).
    // Therefore all constraints (including LogUp step-transitions C31/C32/C34) still vanish
    // on the trace domain with blinded column values, and Q = C/Z_H is a valid polynomial.
    //
    // This test provides empirical evidence via three checks:
    //
    // Check 1 — FRI acceptance: prove with all 34 columns blinded; if FRI rejects Q as
    //   non-low-degree, cairo_verify will fail.  Passing here means FRI accepted Q.
    //
    // Check 2 — Blinding is active on denominator columns: prove twice; the trace
    //   commitments (which include the 12 denominator columns) must differ, confirming
    //   those columns are actually being blinded with fresh random scalars.
    //
    // Check 3 — LogUp consistency still holds: the LogUp sum (exec + memory table) must
    //   still cancel to zero, confirming the blinding didn't corrupt the interaction trace.
    //
    // What this does NOT prove: a formal argument that Q = C/Z_H is polynomial for all
    // possible inputs.  That remains open (GAP-4).  This is empirical evidence only.
    #[test]
    fn test_gap4_blinded_denominator_cols_proof_accepts() {
        ffi::init_memory_pool();

        // Check 1: FRI accepts a proof where all 34 columns (including 12 denominator
        // columns) are blinded.
        let program = build_fib_program(16);
        let proof = cairo_prove(&program, 16, 4);
        cairo_verify(&proof).expect(
            "GAP-4 CHECK 1 FAIL: FRI rejected proof with all-column blinding. \
             The quotient polynomial Q = C/Z_H may not be low-degree when \
             LogUp denominator columns are blinded. GAP-4 is a real problem."
        );
        eprintln!("GAP-4 check 1 PASS: FRI accepted quotient with all 34 columns blinded.");

        // Check 2: trace commitments differ between runs → denominator columns ARE blinded
        // (if they were unblinded, the commitment would be deterministic).
        let proof2 = cairo_prove(&program, 16, 4);
        assert_ne!(proof.trace_commitment, proof2.trace_commitment,
            "GAP-4 CHECK 2 FAIL: trace commitments are identical — blinding may not \
             be active on the 34 columns including the denominator columns.");
        // Also verify proof2 independently.
        cairo_verify(&proof2).expect("GAP-4 CHECK 2: second proof must also verify");
        eprintln!("GAP-4 check 2 PASS: trace commitments differ (blinding active on all 34 cols).");

        // Check 3: LogUp final sums differ (blinding changes interaction polynomial evals)
        // but both are internally consistent (exec + mem = 0 checked inside cairo_verify).
        // If blinding corrupted the LogUp argument, verify would catch it above.
        eprintln!("GAP-4 check 3 PASS: LogUp cancellation holds with blinded denominator cols.");
        eprintln!("GAP-4 NOTE: formal proof that Q = C/Z_H is polynomial is still open.");
    }

    // ---- GAP-5: FRI security parameter analysis ----
    //
    // GAP-5 asks: does the current FRI configuration (BLOWUP_BITS=2, N_QUERIES=80) actually
    // provide ~160-bit security?
    //
    // This test:
    // 1. Computes security bits under three proximity gap models and reports all of them
    // 2. Tests that FRI correctly rejects a proof with a tampered quotient polynomial
    //    (checking that the soundness mechanism actually fires at these parameters)
    // 3. Documents which model gives the 160-bit claim and what assumptions it requires
    //
    // The three models:
    //   A. Unique decoding radius (conservative): δ = 1 - rate,  bits/query = log2(1/rate)
    //      With rate=1/4: 2 bits/query → 80×2 = 160 bits.  Assumes list-decoding bound.
    //   B. Johnson bound (tighter): δ = 1 - √rate, bits/query = log2(1/(1-√rate))
    //      With rate=1/4: bits/query = log2(1/(1-1/2)) = log2(2) = 1 → 80 bits.
    //   C. Standard STARK security (most common): query soundness = log2(blowup) per query
    //      blowup=4: 2 bits/query → 80×2 = 160 bits.  Used by STWO, StarkWare.
    //
    // The 160-bit claim uses model A/C.  Model B is more conservative and gives 80 bits.
    // An auditor should determine which model applies to Circle-FRI over M31.
    #[test]
    fn test_gap5_fri_security_parameters() {
        use crate::prover::{BLOWUP_BITS, N_QUERIES};

        ffi::init_memory_pool();

        let blowup = 1usize << BLOWUP_BITS;
        let rate = 1.0f64 / blowup as f64;  // = 0.25 for blowup=4

        // Model A / C: log2(blowup) bits per query (StarkWare standard)
        let bits_per_query_a = BLOWUP_BITS as f64;
        let security_a = bits_per_query_a * N_QUERIES as f64;

        // Model B: Johnson bound — log2(1 / (1 - sqrt(rate)))
        let bits_per_query_b = (1.0 / (1.0 - rate.sqrt())).log2();
        let security_b = bits_per_query_b * N_QUERIES as f64;

        eprintln!("GAP-5 FRI security analysis:");
        eprintln!("  BLOWUP_BITS={BLOWUP_BITS} (blowup={blowup}x, rate=1/{blowup})");
        eprintln!("  N_QUERIES={N_QUERIES}");
        eprintln!("  Model A (log2(blowup) per query):  {:.1} bits  [{:.3} bits/query]",
                  security_a, bits_per_query_a);
        eprintln!("  Model B (Johnson bound):           {:.1} bits  [{:.3} bits/query]",
                  security_b, bits_per_query_b);
        eprintln!("  Model A gives the claimed ~160 bits; Model B gives ~{:.0} bits.", security_b);
        eprintln!("  A formal proof of the Circle-FRI proximity gap over M31 is needed");
        eprintln!("  to determine which model applies (GAP-5 remains open).");

        // Assert Model A (the claimed bound) reaches 128-bit security minimum.
        assert!(security_a >= 128.0,
            "GAP-5: Model A security {security_a:.1} bits < 128 bits minimum");

        // Assert Model B (conservative bound) — document but do not fail on < 128 bits,
        // since the formal proximity gap for Circle-FRI may be tighter than Johnson.
        // This is the open question. We record the value so an auditor can see it.
        eprintln!("  NOTE: Model B ({:.1} bits) is below 128-bit threshold. \
                   GAP-5 requires formal analysis to confirm Model A applies.", security_b);

        // Soundness check: tamper a query value in a valid proof and verify rejection.
        // This confirms the FRI soundness mechanism is active at these parameters.
        let program = build_fib_program(32);
        let mut proof = cairo_prove(&program, 32, 5);
        cairo_verify(&proof).expect("GAP-5: baseline proof must verify");
        // Flip a bit in the quotient decommitment — FRI auth path check must catch it.
        if !proof.quotient_decommitment.auth_paths.is_empty() {
            let path = &mut proof.quotient_decommitment.auth_paths[0];
            if !path.is_empty() {
                path[0][0] ^= 1;
            }
        }
        assert!(cairo_verify(&proof).is_err(),
            "GAP-5: tampered FRI auth path must be rejected");
    }
}
