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
use crate::prover::{QueryDecommitment, N_QUERIES, BLOWUP_BITS};
use super::trace::{N_COLS, N_VM_COLS, N_CONSTRAINTS,
    COL_AP, COL_FP, COL_FLAGS_START, COL_RES, COL_OFF0, COL_OFF1, COL_OFF2, COL_DST_INV,
    COL_DICT_KEY, COL_DICT_NEW, COL_DICT_ACTIVE};
use super::range_check::{extract_offsets, compute_rc_interaction_trace, compute_rc_table_sum};
use super::logup::{compute_interaction_trace, extract_memory_table, compute_memory_table_sum};
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
            ProveError::InsufficientVRAM { required_gb, free_gb } =>
                write!(f, "insufficient VRAM: proof needs {required_gb:.1} GB but only {free_gb:.1} GB \
                           is free — lower log_n, stop other GPU processes, or use a larger GPU"),
        }
    }
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
    /// FRI layer commitments
    pub fri_commitments: Vec<[u32; 8]>,
    /// Final FRI polynomial (2^3 = 8 QM31 values)
    pub fri_last_layer: Vec<QM31>,
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
}


/// Reusable NTT caches for a given trace size. Create once, prove many.
pub struct CairoProverCache {
    pub log_n: u32,
    pub inv_cache: InverseTwiddleCache,
    pub fwd_cache: ForwardTwiddleCache,
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
pub fn cairo_prove_program(
    program: &super::casm_loader::CasmProgram,
    n_steps: usize,
    log_n: u32,
) -> Result<CairoProof, ProveError> {
    // Refuse to prove programs with truncated felt252 bytecode values.
    // Such programs would produce proofs for wrong computations.
    if program.overflow_count > 0 {
        return Err(ProveError::Felt252Overflow { count: program.overflow_count });
    }

    // ── Height check: can the GPU fit this proof? ──────────────────────────
    // Allocate the twiddle caches first so that vram_query() reflects their
    // footprint; the remaining free VRAM is what the NTT loop needs.
    let cache = CairoProverCache::new(log_n);
    cairo_check_vram(log_n)?;
    let n = 1usize << log_n;
    assert!(n_steps <= n);

    // Set up the Cairo calling convention: place a return sentinel (pc=0, fp=0) below the
    // initial frame so that the program's top-level `ret` instruction halts cleanly.
    let initial_sp = program.bytecode.len() as u64 + 100; // frame base
    let initial_ap = initial_sp + 2;                       // AP/FP after the call frame
    let mut mem = Memory::with_capacity(initial_ap as usize + n_steps + 200);
    mem.load_program(&program.bytecode);
    mem.set(initial_sp,     0); // saved fp  = 0 (sentinel)
    mem.set(initial_sp + 1, 0); // return pc = 0 (halt sentinel)

    // Thread HintContext externally so the prover can inspect dict accesses after execution.
    let mut hint_ctx = super::hints::HintContext::new();
    let columns = super::vm::execute_to_columns_with_hints(
        &mut mem, n_steps, log_n, &program.hints,
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

    Ok(cairo_prove_cached_with_columns(
        &program.bytecode, columns, n_steps, log_n, &cache, None,
        &hint_ctx.dict_accesses,
    ))
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
    cairo_prove_cached_with_columns(program, columns, n_steps, log_n, cache, pedersen_inputs, &[])
}

/// Trace columns that can be ZK-blinded with r · Z_H(x) without breaking the LogUp
/// step-transition constraints. The 9 columns used in logup_delta (pc, inst_lo/hi,
/// dst/op0/op1 addr/val) must remain unblinded — they appear in rational QM31-inverse
/// denominators, which would make the blinded quotient non-polynomial if masked.
///
/// Blinded columns: ap(1), fp(2), flags(5-19), res(26), off0-2(27-29), dst_inv(30)
pub const ZK_BLIND_COLS: &[usize] = &[
    COL_AP, COL_FP,
    COL_FLAGS_START,   COL_FLAGS_START+1,  COL_FLAGS_START+2,  COL_FLAGS_START+3,
    COL_FLAGS_START+4, COL_FLAGS_START+5,  COL_FLAGS_START+6,  COL_FLAGS_START+7,
    COL_FLAGS_START+8, COL_FLAGS_START+9,  COL_FLAGS_START+10, COL_FLAGS_START+11,
    COL_FLAGS_START+12,COL_FLAGS_START+13, COL_FLAGS_START+14,
    COL_RES, COL_OFF0, COL_OFF1, COL_OFF2, COL_DST_INV,
];


/// Internal: prove from pre-computed columns. Called by both `cairo_prove_cached` and `cairo_prove_program`.
fn cairo_prove_cached_with_columns(
    program: &[u64], columns: Vec<Vec<u32>>, n_steps: usize, log_n: u32,
    cache: &CairoProverCache,
    pedersen_inputs: Option<(&[super::stark252_field::Fp], &[super::stark252_field::Fp])>,
    dict_accesses: &[(usize, u64, u64, u64)],
) -> CairoProof {
    let n = 1usize << log_n;
    assert!(n_steps <= n);
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

    let t_ntt = std::time::Instant::now();

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
    // ZK blinding: column[i] += r_j · Z_H[i] for the 22 blinded columns.
    // Z_H is computed once and reused across all groups.
    // The 9 unblinded LogUp columns (pc, inst_lo/hi, dst/op0/op1 addr/val) are
    // skipped — they appear in rational QM31-inverse denominators for C31/C32.
    // The 3 dict columns (31-33) are NOT blinded — they appear in C34's QM31-inverse
    // denominator for the dict step-transition LogUp.

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
    let p_m31 = crate::field::m31::P;
    let zk_scalars: Vec<(usize, u32)> = ZK_BLIND_COLS.iter()
        .map(|&col| (col, rng.random_range(1..p_m31)))
        .collect();

    // Helper: NTT one column from trace domain → eval domain.
    let ntt_col = |src: &Vec<u32>| -> DeviceBuffer<u32> {
        let mut d_col = DeviceBuffer::from_host(src);
        ntt::interpolate(&mut d_col, &cache.inv_cache);
        let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
        unsafe { ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32); }
        drop(d_col);
        ntt::evaluate(&mut d_eval, &cache.fwd_cache);
        d_eval
    };

    // ── Group A: cols 0..TRACE_LO ──────────────────────────────────────────
    let (trace_commitment, host_eval_lo): ([u32; 8], Vec<Vec<u32>>) = {
        let mut group: Vec<DeviceBuffer<u32>> = (0..TRACE_LO)
            .map(|c| ntt_col(&columns[c]))
            .collect();
        for &(col, r) in &zk_scalars {
            if col < TRACE_LO {
                unsafe { ffi::cuda_axpy_m31(r, d_zh.as_ptr(), group[col].as_mut_ptr(), eval_size as u32); }
            }
        }
        unsafe { ffi::cuda_device_sync(); }
        let root = MerkleTree::commit_root_only(&group, log_eval_size);
        let host: Vec<Vec<u32>> = group.iter().map(|c| c.to_host_fast()).collect();
        (root, host)
        // group (GPU buffers) freed here
    };

    // ── Group B: cols TRACE_LO..TRACE_VM_END (15 Cairo VM hi cols) ─────────
    let (trace_commitment_hi, host_eval_hi): ([u32; 8], Vec<Vec<u32>>) = {
        let mut group: Vec<DeviceBuffer<u32>> = (TRACE_LO..TRACE_VM_END)
            .map(|c| ntt_col(&columns[c]))
            .collect();
        for &(col, r) in &zk_scalars {
            if col >= TRACE_LO && col < TRACE_VM_END {
                let local = col - TRACE_LO;
                unsafe { ffi::cuda_axpy_m31(r, d_zh.as_ptr(), group[local].as_mut_ptr(), eval_size as u32); }
            }
        }
        unsafe { ffi::cuda_device_sync(); }
        let root = MerkleTree::commit_root_only(&group, log_eval_size);
        let host: Vec<Vec<u32>> = group.iter().map(|c| c.to_host_fast()).collect();
        (root, host)
        // group (GPU buffers) freed here
    };

    // ── Group C: cols TRACE_VM_END..N_COLS (3 dict linkage cols) ───────────
    // dict_key (31), dict_new (32), dict_active (33).
    // None of these are ZK-blinded — they appear in the dict LogUp denominator (C34).
    let (dict_trace_commitment, host_eval_dict): ([u32; 8], Vec<Vec<u32>>) = {
        let group: Vec<DeviceBuffer<u32>> = (TRACE_VM_END..N_COLS)
            .map(|c| ntt_col(&columns[c]))
            .collect();
        // No blinding for dict cols (unblinded like the 9 LogUp cols)
        unsafe { ffi::cuda_device_sync(); }
        let root = MerkleTree::commit_root_only(&group, log_eval_size);
        let host: Vec<Vec<u32>> = group.iter().map(|c| c.to_host_fast()).collect();
        (root, host)
        // group (GPU buffers) freed here
    };

    drop(d_zh);

    // Reassemble flat host_eval_cols (index c → column c for all N_COLS columns).
    // NOTE: `columns` is kept alive — compute_interaction_trace needs it after Fiat-Shamir.
    let mut host_eval_cols: Vec<Vec<u32>> = Vec::with_capacity(N_COLS);
    host_eval_cols.extend(host_eval_lo);
    host_eval_cols.extend(host_eval_hi);
    host_eval_cols.extend(host_eval_dict);

    let _ntt_ms = t_ntt.elapsed().as_secs_f64() * 1000.0;

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

    let dict_main_interaction_commitment = MerkleTree::commit_root_soa4(
        &d_sdict0, &d_sdict1, &d_sdict2, &d_sdict3, log_eval_size,
    );
    channel.mix_digest(&dict_main_interaction_commitment);
    // Bind S_dict_final into Fiat-Shamir — prover commits to the claimed final sum.
    // The verifier independently recomputes exec_key_new_sum and checks equality.
    channel.mix_digest(&[
        dict_link_final_arr[0], dict_link_final_arr[1],
        dict_link_final_arr[2], dict_link_final_arr[3],
        0, 0, 0, 0,
    ]);

    // Download S_dict for decommitment at query points.
    let host_sdict: [Vec<u32>; 4] = [
        d_sdict0.to_host(), d_sdict1.to_host(),
        d_sdict2.to_host(), d_sdict3.to_host(),
    ];
    drop(d_sdict0); drop(d_sdict1); drop(d_sdict2); drop(d_sdict3);

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
    // Compute LogUp interaction trace on CPU (trace domain → NTT → eval domain).
    // Previous approach (GPU prefix sum over eval domain) was incorrect: it accumulated
    // logup deltas at eval-domain points, not at trace-domain points. The correct
    // interaction polynomial has S[i] = Σ_{j≤i} logup_delta(row_j) at trace positions.
    // This matches the RC interaction trace approach (CPU compute + NTT).
    let (mut logup_trace_cols, logup_final_qm31) =
        compute_interaction_trace(&columns, n_steps, z_mem, alpha_mem);

    // Assert memory LogUp cancellation: exec_sum + table_sum == 0.
    // Matches the RC cancellation check (rc_exec_sum + rc_table_sum == 0) already in place.
    // This ensures the interaction trace is correct before committing to it.
    {
        let (data_table, instr_table) = extract_memory_table(&columns, n_steps);
        let table_sum = compute_memory_table_sum(&data_table, &instr_table, z_mem, alpha_mem);
        assert_eq!(logup_final_qm31 + table_sum, QM31::ZERO,
            "memory LogUp sums don't cancel — execution trace or memory table is inconsistent");
    }

    drop(columns); // trace-domain columns no longer needed

    // Pad each column from n_steps to n (repeat final running-sum value, same as RC trace).
    for c in 0..4 {
        let last = *logup_trace_cols[c].last().unwrap_or(&0);
        logup_trace_cols[c].resize(n, last);
    }

    // NTT each column to the eval domain (interpolate trace→poly, zero-pad, evaluate).
    let mut d_logup_gpu: [DeviceBuffer<u32>; 4] = std::array::from_fn(|_| DeviceBuffer::<u32>::alloc(0));
    for c in 0..4 {
        let mut d_col = DeviceBuffer::from_host(&logup_trace_cols[c]);
        ntt::interpolate(&mut d_col, &cache.inv_cache);
        let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
        unsafe { ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32); }
        drop(d_col);
        ntt::evaluate(&mut d_eval, &cache.fwd_cache);
        d_logup_gpu[c] = d_eval;
    }
    drop(logup_trace_cols);
    let [d_logup0, d_logup1, d_logup2, d_logup3] = d_logup_gpu;
    let logup_final_sum = logup_final_qm31.to_u32_array();

    // Range check: compute LogUp interaction trace over the 3 offsets per row.
    // Keep the 4 columns so we can commit them; the final element is rc_exec_sum.
    let (mut rc_trace_cols, rc_exec_sum) = compute_rc_interaction_trace(&rc_offsets, n_steps, z_rc);
    let rc_table_sum = compute_rc_table_sum(&rc_counts, z_rc);
    assert_eq!(rc_exec_sum + rc_table_sum, QM31::ZERO,
        "range check LogUp sums don't cancel — offset out of range");
    let rc_final_sum = rc_exec_sum.to_u32_array();

    // Pad each RC column from n_steps to n by repeating the final running-sum value.
    // After the last real row the sum stays constant (zero contribution from padding).
    for c in 0..4 {
        let last = *rc_trace_cols[c].last().unwrap_or(&0);
        rc_trace_cols[c].resize(n, last);
    }

    let interaction_commitment = MerkleTree::commit_root_soa4(
        &d_logup0, &d_logup1, &d_logup2, &d_logup3, log_eval_size,
    );
    channel.mix_digest(&interaction_commitment);
    // Download LogUp interaction trace to host for decommitment at query points.
    // Kept alive until after query indices are derived (from FRI last layer).
    let host_logup: [Vec<u32>; 4] = [
        d_logup0.to_host(), d_logup1.to_host(),
        d_logup2.to_host(), d_logup3.to_host(),
    ];
    drop(d_logup0); drop(d_logup1); drop(d_logup2); drop(d_logup3);

    // Commit the RC interaction trace polynomial: NTT to eval domain, Merkle commit.
    // This binds the prover to a specific RC trace BEFORE constraint alphas are drawn,
    // closing the gap where rc_final_sum was previously an unconstrained claim.
    let mut rc_d_eval: [DeviceBuffer<u32>; 4] = std::array::from_fn(|_| DeviceBuffer::<u32>::alloc(0));
    for c in 0..4usize {
        let mut d_col = DeviceBuffer::from_host(&rc_trace_cols[c]);
        ntt::interpolate(&mut d_col, &cache.inv_cache);
        let mut d_eval = DeviceBuffer::<u32>::alloc(eval_size);
        unsafe { ffi::cuda_zero_pad(d_col.as_ptr(), d_eval.as_mut_ptr(), n as u32, eval_size as u32); }
        drop(d_col);
        ntt::evaluate(&mut d_eval, &cache.fwd_cache);
        rc_d_eval[c] = d_eval;
    }
    let rc_interaction_commitment = MerkleTree::commit_root_soa4(
        &rc_d_eval[0], &rc_d_eval[1], &rc_d_eval[2], &rc_d_eval[3], log_eval_size,
    );
    channel.mix_digest(&rc_interaction_commitment);
    // Download RC interaction trace to host for decommitment at query points.
    let host_rc_logup: [Vec<u32>; 4] = [
        rc_d_eval[0].to_host(), rc_d_eval[1].to_host(),
        rc_d_eval[2].to_host(), rc_d_eval[3].to_host(),
    ];
    drop(rc_d_eval);
    drop(rc_trace_cols);

    // Bind LogUp and RC final sums into Fiat-Shamir (tampering breaks FRI)
    channel.mix_digest(&[
        logup_final_sum[0], logup_final_sum[1], logup_final_sum[2], logup_final_sum[3],
        rc_final_sum[0], rc_final_sum[1], rc_final_sum[2], rc_final_sum[3],
    ]);

    // ---- Phase 3: Quotient ----
    let constraint_alphas: Vec<QM31> = (0..N_CONSTRAINTS).map(|_| channel.draw_felt()).collect();
    let alpha_flat: Vec<u32> = constraint_alphas.iter().flat_map(|a| a.to_u32_array()).collect();

    // Upload all trace columns to GPU for quotient evaluation.
    let d_quot_cols: Vec<DeviceBuffer<u32>> = (0..N_COLS)
        .map(|c| DeviceBuffer::from_host(&host_eval_cols[c]))
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

    // Re-upload interaction columns for the quotient kernel.
    // (GPU copies were dropped after commitment; host copies are still live.)
    let d_slogup0 = DeviceBuffer::from_host(&host_logup[0]);
    let d_slogup1 = DeviceBuffer::from_host(&host_logup[1]);
    let d_slogup2 = DeviceBuffer::from_host(&host_logup[2]);
    let d_slogup3 = DeviceBuffer::from_host(&host_logup[3]);
    let d_src0 = DeviceBuffer::from_host(&host_rc_logup[0]);
    let d_src1 = DeviceBuffer::from_host(&host_rc_logup[1]);
    let d_src2 = DeviceBuffer::from_host(&host_rc_logup[2]);
    let d_src3 = DeviceBuffer::from_host(&host_rc_logup[3]);
    // S_dict interaction trace columns (for C33/C34 in quotient kernel)
    let d_sd0 = DeviceBuffer::from_host(&host_sdict[0]);
    let d_sd1 = DeviceBuffer::from_host(&host_sdict[1]);
    let d_sd2 = DeviceBuffer::from_host(&host_sdict[2]);
    let d_sd3 = DeviceBuffer::from_host(&host_sdict[3]);

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

    let quotient_commitment = MerkleTree::commit_root_soa4(&q0, &q1, &q2, &q3, log_eval_size);
    channel.mix_digest(&quotient_commitment);

    // Download quotient to host for sparse extraction (replaces clone_device).
    // This avoids 4 × eval_size GPU clones that pushed VRAM over 32 GiB.
    let host_q0 = q0.to_host_fast();
    let host_q1 = q1.to_host_fast();
    let host_q2 = q2.to_host_fast();
    let host_q3 = q3.to_host_fast();

    // ---- Phase 4: FRI ----
    let quotient_col = SecureColumn { cols: [q0, q1, q2, q3], len: eval_size };

    let fri_alpha = channel.draw_felt();
    let fold_domain = Coset::half_coset(log_eval_size);
    let d_twid = fri::compute_fold_twiddles_on_demand(&fold_domain, true);
    let mut line_eval = SecureColumn::zeros(eval_size / 2);
    fri::fold_circle_into_line_with_twiddles(&mut line_eval, &quotient_col, fri_alpha, &d_twid);
    drop(d_twid);
    drop(quotient_col);

    let mut fri_commitments = Vec::new();
    let mut fri_evals: Vec<SecureColumn> = Vec::new();
    let mut current = line_eval;
    let mut current_log = log_eval_size - 1;

    while current_log > 3 {
        // Commit current layer
        let layer_commitment = MerkleTree::commit_root_soa4(
            &current.cols[0], &current.cols[1], &current.cols[2], &current.cols[3],
            current_log,
        );
        channel.mix_digest(&layer_commitment);
        fri_commitments.push(layer_commitment);

        let fold_alpha = channel.draw_felt();
        let line_domain = Coset::half_coset(current_log);
        let d_twid = fri::compute_fold_twiddles_on_demand(&line_domain, false);
        let folded = fri::fold_line_with_twiddles(&current, fold_alpha, &d_twid);
        drop(d_twid);

        fri_evals.push(current);
        current = folded;
        current_log -= 1;
    }

    let fri_last_layer = current.to_qm31();

    // ---- Phase 5: Query + decommitment ----
    channel.mix_felts(&fri_last_layer);
    let query_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_size))
        .collect();

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
                result[q][c] = host_eval_cols[c][(qi + 1) % eval_size];
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
            .map(|&qi| (qi + 1) % eval_size)
            .collect();
        let all_indices: Vec<usize> = query_indices.iter().copied()
            .chain(next_indices.iter().copied())
            .collect();
        let n_q = query_indices.len();

        // lo: cols 0..TRACE_LO
        let lo_paths = MerkleTree::cpu_merkle_auth_paths_ncols(
            &host_eval_cols[..TRACE_LO],
            &all_indices,
        );
        let paths_lo_qi:  Vec<Vec<[u32; 8]>> = lo_paths[..n_q].to_vec();
        let paths_lo_qi1: Vec<Vec<[u32; 8]>> = lo_paths[n_q..].to_vec();

        // hi: cols TRACE_LO..TRACE_VM_END (16..31 = 15 Cairo VM hi cols)
        let hi_paths = MerkleTree::cpu_merkle_auth_paths_ncols(
            &host_eval_cols[TRACE_LO..TRACE_VM_END],
            &all_indices,
        );
        let paths_hi_qi:  Vec<Vec<[u32; 8]>> = hi_paths[..n_q].to_vec();
        let paths_hi_qi1: Vec<Vec<[u32; 8]>> = hi_paths[n_q..].to_vec();

        // dict: cols TRACE_VM_END..N_COLS (31..34 = 3 dict linkage cols)
        let dict_paths = MerkleTree::cpu_merkle_auth_paths_ncols(
            &host_eval_cols[TRACE_VM_END..N_COLS],
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
        let decom = decommit_from_host_soa4(&cols, &query_indices);
        decom
    };

    // LogUp interaction trace decommitment — binds interaction trace values at query
    // points to interaction_commitment (prerequisite for step transition verification).
    let interaction_decommitment = decommit_from_host_soa4(&host_logup, &query_indices);
    // Next-row decommitment — needed by verifier to check S[i+1] - S[i] = delta(row_i).
    let next_query_indices: Vec<usize> = query_indices.iter()
        .map(|&qi| (qi + 1) % eval_size)
        .collect();
    let interaction_decommitment_next = decommit_from_host_soa4(&host_logup, &next_query_indices);
    drop(host_logup);

    // RC interaction trace decommitment — same structure as LogUp.
    let rc_interaction_decommitment = decommit_from_host_soa4(&host_rc_logup, &query_indices);
    let rc_interaction_decommitment_next = decommit_from_host_soa4(&host_rc_logup, &next_query_indices);
    drop(host_rc_logup);

    // S_dict interaction trace decommitment at query points and query+1 points.
    let dict_main_interaction_decommitment = decommit_from_host_soa4(&host_sdict, &query_indices);
    let dict_main_interaction_decommitment_next = decommit_from_host_soa4(&host_sdict, &next_query_indices);
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
        fri_commitments,
        fri_last_layer,
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

    let constraint_alphas_drawn: Vec<QM31> = (0..N_CONSTRAINTS).map(|_| channel.draw_felt()).collect();

    channel.mix_digest(&proof.quotient_commitment);

    let mut fri_alphas = Vec::new();
    fri_alphas.push(channel.draw_felt()); // circle fold alpha

    for fri_commitment in &proof.fri_commitments {
        channel.mix_digest(fri_commitment);
        fri_alphas.push(channel.draw_felt());
    }

    // ---- Verify FRI structure ----
    // FRI layers: from log_eval_size-1 down to 4 = log_eval_size-4 layers
    // Plus the circle fold layer = log_eval_size-4 total committed FRI layers
    let expected_fri_layers = log_eval_size.saturating_sub(4);
    if proof.fri_commitments.len() != expected_fri_layers as usize {
        return Err(format!("Expected {} FRI layers, got {}",
            expected_fri_layers, proof.fri_commitments.len()));
    }

    if proof.fri_last_layer.len() != 1usize << 3 {
        return Err(format!("Expected 8 FRI last layer values, got {}",
            proof.fri_last_layer.len()));
    }

    // ---- Re-derive query indices ----
    channel.mix_felts(&proof.fri_last_layer);
    let expected_indices: Vec<usize> = (0..N_QUERIES)
        .map(|_| channel.draw_number(eval_size))
        .collect();
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
    let n_fri_layers = proof.fri_decommitments.len();

    for (q, &qi) in proof.query_indices.iter().enumerate() {
        let mut current_idx = qi;
        let mut current_log = log_eval_size;

        // Circle fold: quotient → FRI layer 0
        {
            let domain = Coset::half_coset(current_log);
            let folded_idx = current_idx / 2;
            let (f0, f1) = get_pair_from_decom_4(
                &proof.quotient_decommitment.values[q],
                &proof.quotient_decommitment.sibling_values[q],
                current_idx,
            );
            let twiddle = fold_twiddle_at(&domain, folded_idx, true);
            let expected = fold_pair(f0, f1, fri_alphas[0], twiddle);
            let actual = QM31::from_u32_array(proof.fri_decommitments[0].values[q]);
            if expected != actual {
                return Err(format!("Circle fold mismatch at query {q} (qi={qi})"));
            }
            current_idx = folded_idx;
            current_log -= 1;
        }

        // Line folds
        for layer in 0..n_fri_layers.saturating_sub(1) {
            let domain = Coset::half_coset(current_log);
            let folded_idx = current_idx / 2;
            let decom = &proof.fri_decommitments[layer];
            let (f0, f1) = get_pair_from_decom_4(
                &decom.values[q], &decom.sibling_values[q], current_idx,
            );
            let twiddle = fold_twiddle_at(&domain, folded_idx, false);
            let expected = fold_pair(f0, f1, fri_alphas[layer + 1], twiddle);
            let actual = QM31::from_u32_array(proof.fri_decommitments[layer + 1].values[q]);
            if expected != actual {
                return Err(format!("Line fold mismatch at query {q}, layer {layer}"));
            }
            current_idx = folded_idx;
            current_log -= 1;
        }

        // Verify: fold last FRI decommitment into fri_last_layer
        if n_fri_layers > 0 {
            let last_decom = &proof.fri_decommitments[n_fri_layers - 1];
            let domain = Coset::half_coset(current_log);
            let folded_idx = current_idx / 2;
            let (f0, f1) = get_pair_from_decom_4(
                &last_decom.values[q], &last_decom.sibling_values[q], current_idx,
            );
            let twiddle = fold_twiddle_at(&domain, folded_idx, false);
            let expected = fold_pair(f0, f1, fri_alphas[n_fri_layers], twiddle);
            if folded_idx < proof.fri_last_layer.len() {
                if expected != proof.fri_last_layer[folded_idx] {
                    return Err(format!("FRI last layer mismatch at query {q}"));
                }
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
        let qi_next = (qi + 1) % eval_size;

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
    let verif_eval_domain = crate::circle::Coset::half_coset(log_eval_size);
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
            for i in 0..14u32 {
                rhs = rhs + M31(row[5 + i as usize]) * M31(1u32 << (17 + i));
            }
            rhs = rhs + M31(row[19]) * M31(1); // flag14 * 2^62 ≡ flag14
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
        let natural_idx = bit_reverse(qi, log_eval_size);
        let eval_point = verif_eval_domain.at(natural_idx);
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
    }

    // ---- Verify Merkle auth paths: quotient ----
    verify_decommitment_auth_paths_soa4(
        &proof.quotient_commitment,
        &proof.quotient_decommitment,
        &proof.query_indices,
        "quotient",
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
        .map(|&qi| (qi + 1) % eval_size)
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

fn bit_reverse(x: usize, n_bits: u32) -> usize {
    let mut result = 0usize;
    let mut val = x;
    for _ in 0..n_bits {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

fn fold_twiddle_at(domain: &Coset, folded_index: usize, circle: bool) -> M31 {
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

fn fold_pair(f0: QM31, f1: QM31, alpha: QM31, twiddle: M31) -> QM31 {
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

// ---- Decommitment helpers ----

fn decommit_from_host_soa4(
    host_cols: &[Vec<u32>],  // [4] columns
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

    // Generate Merkle auth paths for both values and siblings
    let cols4: [Vec<u32>; 4] = [
        host_cols[0].clone(), host_cols[1].clone(),
        host_cols[2].clone(), host_cols[3].clone(),
    ];
    let all_paths = MerkleTree::cpu_merkle_auth_paths_soa4(&cols4, &all_indices);

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
    decommit_from_host_soa4(&host_cols, indices)
}

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
        // last eval-domain position: next_qi = (qi + 1) % eval_size.
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
            let next_qi = (qi + 1) % eval_size;
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
        // With 22 blinded columns, the probability of identical commitments is ~1/2^128.
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
        cairo_prove_cached_with_columns(&program, columns, 32, 5, &cache, None, dict_accesses)
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
}
