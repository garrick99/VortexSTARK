# VortexSTARK Internal Audit Document

**Status:** Internal pre-audit. No external audit has been conducted.
**Last updated:** 2026-03-27
**Auditor:** Self (automated + manual)

---

## Purpose

This document tracks VortexSTARK's internal audit posture: constraint coverage,
known gaps, test vectors, and what a third-party auditor should prioritize. It is
updated when new gaps are found or closed. It is meant to be the single source of
truth before external audit engagement.

---

## 1. Constraint Coverage Map

The Cairo VM AIR has **35 constraints (0-34)**: 31 VM execution constraints, 2 LogUp/RC step-transition constraints, and 2 dict linkage constraints. Each must be independently rejected by the verifier when violated. The table below maps each constraint to its dedicated test.

| # | Constraint | Test | Status |
|---|-----------|------|--------|
| 0-14 | Flag binary: `flag * (1-flag) = 0` for all 15 flags | `test_tamper_flag_binary`, `test_per_constraint_forgery_all_columns` | COVERED |
| 15 | Result computation: `(1-jnz) * (res - expected_res) = 0` | `test_tamper_result_computation`, `test_per_constraint_forgery_all_columns` | COVERED |
| 16 | PC update | `test_tamper_pc_update`, `test_per_constraint_forgery_all_columns` | COVERED |
| 17 | AP update | `test_tamper_ap_update`, `test_per_constraint_forgery_all_columns` | COVERED |
| 18 | FP update | `test_tamper_fp_update`, `test_per_constraint_forgery_all_columns` | COVERED |
| 19 | Assert_eq: `opcode_assert * (dst - res) = 0` | `test_tamper_assert_eq`, `test_per_constraint_forgery_all_columns` | COVERED |
| 20 | dst_addr correctness | `test_per_constraint_forgery_all_columns` (COL_DST_ADDR) | COVERED |
| 21 | op0_addr correctness | `test_per_constraint_forgery_all_columns` (COL_OP0_ADDR) | COVERED |
| 22 | op1_addr correctness | `test_per_constraint_forgery_all_columns` (COL_OP1_ADDR) | COVERED |
| 23 | JNZ fall-through: `jnz*(1-dst*dst_inv)*(next_pc - pc - inst_size) = 0` | `test_per_constraint_forgery_all_columns` | COVERED |
| 24 | JNZ inverse: `jnz*dst*(1-dst*dst_inv) = 0` | `test_per_constraint_forgery_all_columns` (COL_DST_INV) | COVERED |
| 25 | op1 exclusivity: `op1_imm*op1_fp = 0` | `test_constraint_op1_exclusivity` | COVERED |
| 26 | op1 exclusivity: `op1_imm*op1_ap = 0` | `test_constraint_op1_exclusivity` | COVERED |
| 27 | op1 exclusivity: `op1_fp*op1_ap = 0` | `test_constraint_op1_exclusivity` | COVERED |
| 28 | PC update exclusivity: `(jmp_abs*jmp_rel + jmp_abs*jnz + jmp_rel*jnz) = 0` | `test_constraint_pc_exclusivity` | COVERED |
| 29 | Opcode exclusivity: `(call*ret + call*assert + ret*assert) = 0` | `test_constraint_opcode_exclusivity` | COVERED |
| 30 | Instruction decomposition: all 63 bits verified | `test_tamper_instruction_decomposition` | COVERED |
| 31 | LogUp step transition: `S_logup[i+1] - S_logup[i] - logup_delta(row_i) = 0` | `test_tamper_logup_final_sum`, `test_tamper_interaction_decommitment` | COVERED |
| 32 | RC step transition: `S_rc[i+1] - S_rc[i] - rc_delta(row_i) = 0` | `test_tamper_rc_final_sum`, `test_tamper_rc_interaction_decommitment` | COVERED |
| 33 | Dict active binary: `dict_active * (1 - dict_active) = 0` (col 33) | `test_dict_logup_commitment_with_accesses`, `test_per_constraint_forgery_all_columns` | COVERED |
| 34 | S_dict step transition: `S_dict[i+1] - S_dict[i] - dict_active[i]*inv(z_dict_link - (dict_key[i] + α*dict_new[i])) = 0` | `test_dict_logup_commitment_with_accesses`, `test_dict_logup_tamper_exec_sum` | COVERED |

---

## 2. LogUp Arguments

| Argument | Purpose | Cancellation Verified | Bound to Fiat-Shamir |
|---------|---------|----------------------|---------------------|
| Memory LogUp | Every (addr, value) read matches a written entry | YES — `test_logup_memory_consistency` | YES — `logup_final_sum` mixed in |
| Range Check LogUp | All 16-bit offsets lie in [0, 2^16) | YES — `compute_rc_table_sum` assert | YES — `rc_final_sum` mixed in |
| Dict Permutation LogUp | Exec-order log and key-sorted log are the same multiset | YES — `test_dict_logup_commitment_with_accesses` | YES — both logs committed before challenges drawn |

| **S_dict link LogUp** | Dict columns in main FRI trace = dict exec trace (same (key,new) multiset) | YES — `test_dict_logup_commitment_with_accesses` | YES — `dict_link_final` mixed in post-commitment |

**Dict LogUp is now fully closed (GAP-1 resolved).** The S_dict argument (C34) links the main trace's dict columns to the authenticated dict exec trace. See §5 GAP-1 for the complete security argument.

---

## 3. Merkle / Auth-Path Coverage

| Commitment | Auth paths verified? | Test |
|-----------|---------------------|------|
| trace_commitment (cols 0-15, Group A) | YES | `test_tamper_trace_auth_paths` |
| trace_commitment_hi (cols 16-30, Group B) | YES | `test_tamper_trace_auth_paths` |
| dict_trace_commitment (cols 31-33, Group C) | YES | `test_dict_logup_commitment_with_accesses` |
| interaction_commitment (LogUp) | YES | `test_tamper_interaction_decommitment` |
| rc_interaction_commitment (RC LogUp) | YES | `test_tamper_rc_interaction_decommitment` |
| quotient_commitment (AIR quotient) | YES | `test_cairo_prove_verify_tampered_quotient` |
| oods_quotient_commitment (FRI input) | YES — auth paths + formula check | `test_soundness_oods_quotient_tamper` |
| FRI layer commitments | YES | `test_cairo_prove_verify_tampered_fri` |
| ec_trace_commitment (Pedersen lo) | YES | `test_tamper_ec_trace_auth_paths` |
| ec_trace_commitment_hi (Pedersen hi) | YES | `test_tamper_ec_trace_auth_paths` |
| dict_exec_commitment (exec data trace Merkle root) | YES — `test_dict_logup_tamper_exec_hash_mismatch` | |
| dict_sorted_commitment (sorted data trace Merkle root) | YES — `test_dict_logup_tamper_sorted_order` | |
| dict_exec_interact_commitment (exec interact trace Merkle root) | YES — `test_dict_logup_commitment_with_accesses` | |
| dict_sorted_interact_commitment (sorted interact trace Merkle root) | YES — `test_dict_logup_commitment_with_accesses` | |

---

## 4. ZK (Zero-Knowledge) Status

VortexSTARK provides **full ZK for all 34 main trace columns** as of 2026-03-26 (GAP-4 CLOSED).

### Blinded columns (34/34)
All 34 main trace columns — including all 9 LogUp-denominator columns (`pc`, `inst_lo`, `inst_hi`,
`dst_addr`, `dst`, `op0_addr`, `op0`, `op1_addr`, `op1`) and 3 dict linkage columns
(`dict_key`, `dict_new`, `dict_active`) — are blinded via `r_j · Z_H(x)` at the eval domain.

`Z_H(x)` vanishes on the trace domain (witnesses at trace points unchanged), is nonzero at query
points (evaluations are randomly masked). FRI proves each blinded polynomial is low-degree.

### Soundness of blinding LogUp/dict denominator columns (GAP-4 argument)

The concern was that blinding LogUp denominator columns introduces `r · Z_H(x)` terms into
QM31-inverse expressions, potentially making Q = C/Z_H a rational function. **This does not occur**
for the following reason:

Constraints 31, 32, and 34 are step-transition constraints evaluated at **query points** (eval domain),
where `Z_H ≠ 0`. At these points, both prover and verifier use the *same blinded values* consistently.
The constraint polynomial `C(x) = Q(x) · Z_H(x)` remains a polynomial because at **trace-domain**
points `Z_H = 0`, making the blinding term `r · Z_H(x) = 0` there. The quotient Q = C/Z_H has no
rational singularity. See SOUNDNESS.md §GAP-4 for the full argument.

### Interaction trace columns (NOT blinded — by design)
`S_logup`, `S_rc`, `S_dict` (interaction trace columns) are not blinded. Their correctness is
enforced via memory table commitment and RC counts commitment checks bound to Fiat-Shamir.

**Auditor note:** Verify that the GAP-4 argument in SOUNDNESS.md is sound, particularly
the claim that the `r · Z_H(x)` blinding term vanishes at trace-domain points for all
three constraint families (C31, C32, C34) with their respective denominator structures.

---

## 5. Known Gaps (Prioritized for Third-Party Audit)

### GAP-1: Dict consistency not fully in STARK proof — CLOSED
**Severity:** High → CLOSED
**Status (2026-03-26):** FULLY CLOSED via S_dict step-transition LogUp argument.

**What was added (session 3, 2026-03-26):** Full dict sub-AIR with full-soundness verification:
1. **Exec data trace (3 cols: key, prev, new)** committed as Merkle polynomial before Fiat-Shamir challenges.
2. **Sorted data trace (4 cols: key, prev, new, is_first)** committed as Merkle polynomial before challenges.
3. `z_dict` and `alpha_dict` drawn from Fiat-Shamir after both data commitments.
4. Both LogUp final sums mixed into Fiat-Shamir transcript.
5. **Permutation argument:** prover asserts `exec_final == sorted_final` (same multiset).
6. **Full trace data in proof:** verifier receives ALL dict_n rows of exec and sorted data.
7. **Verifier authenticates full data** by recomputing both Merkle roots from the full payload and checking against committed roots.
8. **Verifier checks ALL sorted step-transition constraints C0-C3** (not sampled — every row pair verified).
9. **Verifier recomputes both LogUp final sums** independently and checks they match claimed values and are equal (permutation argument).

**What was added (session 6, 2026-03-26) — S_dict link (GAP-1 closure):**
The remaining gap was that the dict exec trace was not linked to the main execution trace.
A malicious prover could fabricate exec trace (key/prev/new) as long as it was internally consistent.
This is now closed by the **S_dict step-transition LogUp argument**:

10. **Columns 31-33 added to main trace:** `dict_key` (col 31), `dict_new` (col 32), `dict_active` (col 33).
    These are committed as Group C in the main FRI proof before any LogUp challenges are drawn.
11. **Constraint C33 (binary):** `dict_active * (1 - dict_active) = 0` — ensures the selector is boolean.
12. **Constraint C34 (S_dict step-transition):**
    `S_dict[i+1] - S_dict[i] - dict_active[i] * inv(z_dict_link - (dict_key[i] + α_dict_link * dict_new[i])) = 0`
    This is a new 4-column QM31 interaction trace committed after `dict_trace_commitment` and before the EC section.
13. **dict_link_final** (the final S_dict value) is mixed into the Fiat-Shamir transcript.
14. **exec_key_new_sum** is independently computed by the verifier from the dict exec trace data (`key + α_dict_link * new_val` formula, over `dict_n_accesses` real accesses only).
15. **Verifier checks** `dict_link_final == exec_key_new_sum`. This forces the main trace's dict columns to contain exactly the same (key, new_val) multiset as the authenticated exec trace, closing the fabrication attack surface.

**Security argument:** If `S_dict_final = exec_key_new_sum` (same (key,new) multiset via C34 + dict sub-AIR permutation argument), AND the dict sub-AIR proves chain validity (sorted order → each prev matches predecessor's new), THEN prev values are uniquely determined by the chain. Full dict consistency follows.

**N_COLS:** 31 → **34** (cols 31-33 are dict linkage).
**N_CONSTRAINTS:** 33 → **35** (added C33 binary, C34 S_dict LogUp).

**Files:** `src/cairo_air/trace.rs` (N_VM_COLS, N_COLS, N_CONSTRAINTS), `src/cairo_air/prover.rs` (S_dict building, challenges, verifier), `cuda/cairo_constraint.cu` (C33, C34 in GPU kernel), `src/cuda/ffi.rs` (updated signatures).

### GAP-2: Execution-time values outside M31 — CLOSED
**Severity:** Medium → CLOSED
**Status (2026-03-25):** `cairo_prove_program` returns `Err(ProveError::ExecutionRangeViolation {count})`
if any data value read from memory during hint-aware execution exceeds M31 (P = 2^31-1).
At every instruction, `op0`, `op1`, and direct memory reads are checked against P.
The non-hint path (`cairo_prove`, `cairo_prove_cached`) is intended for hand-crafted M31
programs and does not have this gate; it is not accessible from the production entry point.
**Remaining limitation:** VortexSTARK fundamentally operates over M31. Programs that require
Stark252 arithmetic cannot be proven (they will be rejected). This is a design constraint,
not a soundness bug.
**Files to audit:** `src/cairo_air/vm.rs` (`execute_to_columns_with_hints`),
`src/cairo_air/prover.rs` (`cairo_prove_program`)

### GAP-3: Starknet syscalls not emulated
**Severity:** Low for current use cases (pure computation programs work correctly)
**Status:** Documented; any unrecognized syscall hint is silently skipped with a stderr warning.
**What's needed:** Syscall emulation table.

### GAP-4: Full ZK — CLOSED 2026-03-26
**Severity:** Privacy gap → CLOSED
**Status:** All 34 main trace columns blinded via `r · Z_H(x)`. Formal argument documented in
SOUNDNESS.md §GAP-4: the blinding term vanishes on the trace domain (Z_H = 0 there), so the
quotient Q = C/Z_H remains a polynomial despite blinding appearing in LogUp/dict denominator
columns. Tests `test_gap4_blinded_denominator_cols_proof_accepts` and `test_zk_blinding_hides_trace`
both pass.
**Remaining auditor action:** Review the formal argument in SOUNDNESS.md §GAP-4.

### GAP-5: FRI security parameters — BLOWUP_BITS=2 enforced, stwo FriVerifier confirmed
**Severity:** Medium → Substantially mitigated
**Status (2026-04-03):** BLOWUP_BITS=2 (4× blowup) enforced in `src/prover.rs`. Provides
conjectured 160-bit security (80 queries × 2 bits/query, Model A). `test_gap5_fri_security_parameters`
now passes (was failing at BLOWUP_BITS=1). `test_stwo_fri_verifier_e2e` fully passes (commit +
decommit) with the BLOWUP_BITS=2 parameter set. Formal Circle-FRI proximity gap proof over M31
is still pending in the literature; the claim is under the standard FRI proximity conjecture.
**Remaining gap:** No formal proof of the Circle-FRI proximity gap specific to M31; standard
industry assumption (Starkware/stwo use identical parameters). See SOUNDNESS.md §GAP-5.

### GAP-6: Missing proof-of-work nonce — CLOSED
**Severity:** Medium → CLOSED
**Status (2026-03-26):** `CairoProof` now includes `pow_nonce: u64`. The prover grinds a
valid nonce on GPU (RTX 5090, ~5ms for 26 bits) after all FRI commitments are mixed into the
Fiat-Shamir transcript. The verifier checks the nonce before sampling query indices.

**What the gap was:** Without PoW, an adversary could observe all Merkle commitments, predict
the deterministic query indices, and adaptively craft commitments that pass the queried
positions. The prover needed to commit to a nonce that requires 2^POW_BITS work to find
*after* seeing the commitments — preventing this adaptive strategy.

**Protocol (matches stwo's Blake2s convention):**
- `prefix_digest = Blake2s(0x12345678_LE || [0u8;12] || channel_state || POW_BITS_LE)`
- Valid nonce: `Blake2s(prefix_digest || nonce_LE)` has ≥ `POW_BITS` trailing zeros in bytes 0–15 (u128 LE)
- After verification: `channel.mix_u64(nonce)` then query indices sampled as normal
- `POW_BITS = 26` (64M expected hashes; ~5ms GPU, ~several seconds CPU)

**Files:** `src/prover.rs` (POW_BITS const), `src/channel.rs` (`mix_u64`, `verify_pow_nonce`, `state_words`), `src/cairo_air/prover.rs` (`pow_nonce` field, GPU grind, verifier check), `cuda/grind.cu` (existing kernel, now wired up).

---

## 6. Test Coverage Statistics

| Category | Count | Notes |
|---------|-------|-------|
| Field arithmetic (M31, CM31, QM31) | ~20 | All arithmetic properties |
| NTT / Circle NTT | ~12 | Forward, inverse, correctness |
| Merkle tree | ~10 | Commit, auth paths, domain separation |
| FRI fold equations | ~8 | Circle fold, line fold, deterministic |
| STARK prove + verify | ~35 | Multiple sizes, tamper detection (18+ attacks) |
| Cairo VM | ~20 | Decoder, executor, Fibonacci, constraints |
| LogUp memory | 4 | Cancellation for 2-step and 10-step programs |
| Range check | 3 | Offset validation |
| Cairo hints | 13 | AllocSegment, dict lifecycle, squash, U256InvModN |
| Dict consistency + LogUp | 18 | Chain verify, LogUp cancellation, Fiat-Shamir binding, tamper tests, dict sub-AIR (11 unit tests), permutation argument, padding row tamper, dict_active binary |
| ZK blinding | 1 | Different commitments per run, both verify (34/34 columns) |
| Execution range gate | 2 | Overflow detected, valid program passes |
| Proof serialization | 1 | Prove → JSON → deserialize → verify roundtrip |
| OODS quotient formula | 1 | test_soundness_oods_quotient_tamper — fake OODS quotient rejected |
| Property / cross-validation | ~30 | Random programs, reference VM comparison, soundness mutations |
| **Total** | **239** (lib) **+ 30** (integration) | |

---

## 7. Cross-Validation Strategy

### Reference VM
`tests/cross_validation.rs` contains a minimal independent Cairo VM implementation (~120 lines).
It executes the same programs as VortexSTARK's `vm.rs` and compares register state after every step.
The reference VM models memory writes (call saves fp/return-address; assert_eq writes result),
matching the semantics of the production VM. Any divergence is a VM execution bug.

### Test vectors
- `tests/fixtures/fibonacci.casm` — Fibonacci program (real CASM JSON from Sierra compiler)
- Hand-crafted bytecode vectors for: add, mul, call/ret, jnz, assert_eq

### Property tests
`tests/property_tests.rs` generates random programs from a small instruction template set,
proves them with VortexSTARK, and verifies the proof. Any prove+verify failure or false rejection
on a valid proof is a soundness/completeness bug.

---

## 8. What a Third-Party Auditor Should Prioritize

1. **Dict S_dict link** (GAP-1 now closed): verify the S_dict step-transition argument (C34) correctly
   links main trace cols 31-33 to the authenticated exec trace; confirm `dict_link_final == exec_key_new_sum`
   check closes the fabrication attack; verify padding rows are correctly excluded from `exec_key_new_sum`
2. **FRI soundness** (GAP-5): validate the Circle-FRI security argument over M31
3. **LogUp step-transition constraints** (constraints 31-32): check the QM31 inverse in the CUDA
   kernel matches the CPU verifier path
4. **Instruction decomposition constraint** (constraint 30): verify the algebraic identity over M31
   and the 2^31≡1 reduction
5. **ZK blinding** (GAP-4 closed): review the formal argument in SOUNDNESS.md §GAP-4 that blinding
   all 34 trace columns (including LogUp/dict denominator columns) preserves a well-defined low-degree
   quotient polynomial for FRI
6. **Merkle domain separation**: verify the `h[6]^=1` personalization prevents leaf/node confusion
7. **Fiat-Shamir ordering**: verify prover and verifier mix commitments and draw challenges in the
   same order — the dict commitment block must precede `z_mem`/`alpha_mem`/`z_rc` in both prover
   and verifier
8. **Execution range gate** (GAP-2 closed): confirm `execute_to_columns_with_hints` catches all
   data-path memory reads (op0, op1, direct reads) and that the non-hint path is unreachable from
   production entry points with Stark252 programs

---

## 9. Audit Readiness Checklist

- [x] All 35 constraints have at least one dedicated rejection test
- [x] LogUp memory cancellation verified for all test programs
- [x] Range check cancellation asserted at prove time
- [x] All Merkle auth paths verified by the verifier (3-group split: A/B/C)
- [x] Fiat-Shamir transcript binds all commitments and claims
- [x] EC trace split lo/hi, both halves verified
- [x] Dict access chain validated (CPU-side) before proof generation
- [x] Dict exec/sorted logs committed to Fiat-Shamir; LogUp permutation argument verified
- [x] Dict tamper tests: exec_sum tamper, sort-order tamper, hash mismatch — all detected
- [x] `cairo_prove_program` returns error for truncated felt252 bytecode (>u64)
- [x] `cairo_prove_program` returns error for execution-time values exceeding M31
- [x] 34/34 trace columns blinded via `r · Z_H(x)`; formal soundness argument for LogUp/dict denominator columns documented in SOUNDNESS.md §GAP-4 (GAP-4 CLOSED)
- [x] Proof serialize → deserialize → verify roundtrip passes
- [x] Dict sub-AIR: 2 Merkle polynomial commitments (exec data, sorted data)
- [x] Full exec and sorted trace data in proof; verifier recomputes Merkle roots for authentication
- [x] Dict sorted step-transition constraints C0-C3 verified for ALL dict_n-1 row pairs (full soundness)
- [x] Dict LogUp final sums recomputed by verifier from full data (not claimed)
- [x] Dict permutation argument: exec_final == sorted_final via Fiat-Shamir
- [x] Main memory LogUp cancellation asserted in production prover (exec_sum + table_sum == 0)
- [x] Dict exec trace wired to main execution trace via S_dict LogUp (cols 31-33, C33-C34) — GAP-1 CLOSED
- [x] dict_trace_commitment (Group C) verified with auth paths at every query
- [x] dict_main_interaction_commitment (S_dict) bound to Fiat-Shamir via dict_link_final
- [x] Verifier independently recomputes exec_key_new_sum and checks dict_link_final == exec_key_new_sum
- [x] Full ZK for 34/34 columns — formal soundness argument documented in SOUNDNESS.md §GAP-4 (GAP-4 CLOSED 2026-03-26)
- [x] OODS quotient formula check — verifier recomputes Q(p) from decommitted trace + AIR quotient values; fake polynomial rejected (`test_soundness_oods_quotient_tamper`)
- [ ] Formal FRI security analysis for Circle group — see GAP-5; current argument documented in SOUNDNESS.md
- [x] Proof-of-work nonce (GAP-6 CLOSED): `pow_nonce` in `CairoProof`, GPU grind + verifier check
- [ ] External third-party audit
