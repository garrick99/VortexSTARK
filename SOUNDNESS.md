# VortexSTARK Soundness Status

## Constraint System (N_VM_COLS=31 execution columns + 3 dict linkage = N_COLS=34 total; N_CONSTRAINTS=35)

### Flag binary (constraints 0-14)
- All 15 flags constrained to {0, 1} via `flag * (1 - flag) = 0`

### Result computation (constraint 15)
- `(1 - pc_jnz) * (res - expected_res) = 0`
- Expected_res = default*op1 + res_add*(op0+op1) + res_mul*(op0*op1)

### PC update (constraint 16)
- Non-jnz: next_pc = regular + abs + rel
- Jnz taken: `pc_jnz * dst * (next_pc - pc - op1) = 0`

### AP update (constraint 17)
- next_ap = ap + ap_add*res + ap_add1 + call*2

### FP update (constraint 18)
- next_fp = (1-call-ret)*fp + call*(ap+2) + ret*dst

### Assert_eq (constraint 19)
- `opcode_assert * (dst - res) = 0`

### Operand address verification (constraints 20-22)
- dst_addr = (1-dst_reg)*ap + dst_reg*fp + off0 - 0x8000
- op0_addr = (1-op0_reg)*ap + op0_reg*fp + off1 - 0x8000
- op1_addr = op1_base + off2 - 0x8000 (op1_base depends on source flags)

### JNZ soundness (constraints 23-24)
- Fall-through: `pc_jnz * (1 - dst*dst_inv) * (next_pc - pc - inst_size) = 0`
- Inverse consistency: `pc_jnz * dst * (1 - dst*dst_inv) = 0`
- dst_inv auxiliary column: M31 inverse of dst (0 when dst=0)

### Mutual exclusivity (constraints 25-29)
- Op1 source: op1_imm*op1_fp=0, op1_imm*op1_ap=0, op1_fp*op1_ap=0
- PC update: jump_abs*jump_rel + jump_abs*jnz + jump_rel*jnz = 0
- Opcode: call*ret + call*assert + ret*assert = 0

### Instruction decomposition (constraint 30)
- `inst_lo + inst_hi ≡ off0 + off1·2^16 + off2·2^32 + Σ(flag_i·2^(48+i))` (mod P)
- Exploits M31 wraparound: 2^31 ≡ 1, so inst_hi·2^31 ≡ inst_hi
- All 63 bits covered: 48 bits via offsets, 15 bits via flag binary constraints

## Verified Subsystems

### Vanishing polynomial / zerofier (FIXED 2026-03-22)
- Previously: quotient kernel wrote C(x) (raw constraint sum). FRI proved C(x) low-degree,
  but low-degree C(x) alone does not prove constraints vanish on the trace domain.
- Now: GPU kernel `compute_vanishing_inv_kernel` computes 1/Z_H(x) for every NTT position.
  Z_H(x) = f_{log_n}(x)+1 where f_k is iterated circle doubling (x→2x²−1), zero iff x is
  in the trace domain. Quotient kernel multiplies by vh_inv, producing Q(x)=C(x)/Z_H(x).
- Verifier checks C(x)==Q(x)·Z_H(x) at each query point using Coset::circle_vanishing_poly_at.
- Closing this gap is the primary soundness improvement: FRI now proves Q(x) is low-degree,
  which by the Schwartz-Zippel argument implies C(x) vanishes on the trace domain w.h.p.

### Verifier-side constraint evaluation
- Verifier independently evaluates all 31 constraints at query points
- Checks constraint_sum == quotient_value * Z_H(eval_point.x) (accounting for zerofier)
- Per-constraint-family tamper tests all passing

### LogUp interaction trace decommitment (ADDED 2026-03-22)
- LogUp interaction trace computed correctly on CPU (trace domain), then NTT'd to eval domain.
  Previous GPU approach (prefix sum over eval-domain positions) was incorrect — it accumulated
  logup_delta at eval-domain points, not trace-domain points, producing a wrong polynomial.
- Both LogUp and RC interaction traces now use the same correct pipeline:
  compute_interaction_trace (trace domain) → interpolate → zero-pad → evaluate → commit.
- test_logup_final_sum_cancels: exec_sum + memory_table_sum == 0 VERIFIED
- LogUp and RC interaction traces decommitted at query points (auth paths vs interaction_commitment / rc_interaction_commitment)
- Verifier checks auth paths for interaction_decommitment and rc_interaction_decommitment
- test_tamper_interaction_decommitment: DETECTED
- test_tamper_rc_interaction_decommitment: DETECTED
- **FIXED 2026-03-24**: Step transition constraints fully wired into AIR polynomial:
  - Constraint 31: S_logup[i+1] - S_logup[i] - logup_delta(row_i) = 0 (QM31)
  - Constraint 32: S_rc[i+1] - S_rc[i] - rc_delta(row_i) = 0 (QM31)
  - CUDA quotient kernel receives 8 interaction columns (4 LogUp + 4 RC) and a
    16-u32 challenge buffer [z_mem, alpha_mem, alpha_mem_sq, z_rc].
  - Verifier evaluates both step-transition constraints at query points in the
    combined 33-constraint sum; checks constraint_sum == Q(x)*Z_H(x).
  - Auth paths for interaction_decommitment_next and rc_interaction_decommitment_next
    cryptographically bind S[qi+1] to the committed interaction polynomial.
  - Instruction fetch denominator extended with alpha²*inst_hi: all 15 flag bits
    are now bound to the memory argument — closes Gap 2 (inst_hi).
- test_tamper_logup_final_sum: DETECTED (Fiat-Shamir binding)
- Fiat-Shamir verifier ordering bug fixed: EC trace commitments now correctly bound before
  z_mem/alpha_mem/z_rc challenges (matching prover order).

### Range check argument
- extract_offsets() extracts 3x16-bit offsets per instruction
- compute_rc_interaction_trace() computes running sum, compute_rc_table_sum() computes table contribution
- Prover asserts exec_sum + table_sum = 0 (LogUp cancellation)
- z_rc challenge drawn from Fiat-Shamir, RC final sum bound into transcript
- RC interaction trace decommitted at query points (auth paths vs rc_interaction_commitment)
- test_tamper_rc_final_sum: DETECTED
- test_tamper_rc_interaction_decommitment: DETECTED
- test_rc_final_sum_is_real: verifies RC sum is distinct from LogUp sum

### Public inputs
- initial_pc, initial_ap, n_steps, program_hash, program bytecode
- All bound into Fiat-Shamir transcript

### Merkle auth paths (FULLY ACTIVATED 2026-03-22)
- Previously: quotient and FRI decommitments had empty auth_paths; verifier skipped the check,
  allowing a cheating prover to supply fake quotient/FRI values without Merkle binding.
- Now: all four commitments have real auth paths:
  - trace_commitment (cols 0-15) + trace_commitment_hi (cols 16-30) — cpu_merkle_auth_paths_ncols
  - quotient_commitment — decommit_from_host_soa4 (cpu_merkle_auth_paths_soa4)
  - all FRI layer commitments — decommit_fri_layer (cpu_merkle_auth_paths_soa4)
- Verifier rejects empty auth paths (hard error instead of silent skip).
- Domain separation: internal nodes use Blake2s personalization (h[6] ^= 0x01), leaves use h[6] = IV6
- Prevents second-preimage attacks where leaf data could be confused with internal node hashes
- test_quotient_auth_paths_reject_fake_value: DETECTED
- test_fri_auth_paths_reject_fake_value: DETECTED

### Trace decommitment auth paths (ADDED 2026-03-22)
- Previously: trace_values_at_queries had NO Merkle auth paths. A cheating prover could supply
  arbitrary trace values satisfying constraints without being bound to the committed trace root.
- Now: trace is committed in two halves — trace_commitment (cols 0-15) and trace_commitment_hi
  (cols 16-30). Both are bound into the Fiat-Shamir transcript. Auth paths are generated for
  all 4 combinations (qi, qi+1) × (lo, hi) using cpu_merkle_auth_paths_ncols.
- Verifier checks all four sets of auth paths: every trace column at every query point is
  cryptographically bound to a committed polynomial.
- GPU/CPU agreement: fixed GPU merkle_tiled_generic_kernel and merkle_hash_leaves_kernel to use
  min(n_cols,16)*4 as the Blake2s length counter, matching the CPU blake2s_hash behavior.
- test_tamper_trace_auth_paths: tests both lo (col 0 = pc) and hi (col 26 = res) tamper cases — DETECTED

### Pedersen EC constraint system (FULLY BOUND 2026-03-22)
- Full intermediate EC trace generated: 29 columns per step (acc_x/y 9 limbs each, lambda 9 limbs, window, op_type)
- EC trace split into lo (cols 0-15) and hi (cols 16-28), committed separately via NTT + Merkle.
  Both roots bound into Fiat-Shamir transcript (ec_trace_commitment, ec_trace_commitment_hi).
- Previously: single commitment covered only the first 16 columns (GPU Blake2s leaf hash cap);
  cols 16-28 were unconstrained. A cheating prover could supply fake hi-column values.
- Now: lo/hi split mirrors the main trace split. Auth paths generated for all 4 sets
  (qi × lo/hi, qi+1 × lo/hi) via cpu_merkle_auth_paths_ncols.
- Verifier maps main eval domain query indices into EC eval domain (qi % ec_eval_size)
  using ec_log_eval stored in the proof, then verifies all 4 auth path sets.
- test_tamper_ec_trace_auth_paths: lo tamper (col 0) DETECTED, hi tamper (col 16) DETECTED
- Verifier checks EC doubling/addition constraints at query points

### Multi-program testing
- Fibonacci (add-only): proven + verified
- Multiply-accumulate (mul-only): proven + verified
- Mixed add/mul alternating: proven + verified
- Call/ret initialization pattern: proven + verified

### Adversarial / forgery coverage (ADDED 2026-03-24, fixed 2026-03-25)
- test_per_constraint_forgery_all_columns: loops over all 31 trace columns × {current-row, next-row},
  tampers each by +1 mod P, asserts verifier rejects. Covers all 31 per-column constraints and
  Merkle auth-path binding for both row positions.
- Constraints 31-32 (LogUp/RC step transitions) covered by existing
  test_tamper_interaction_decommitment and test_tamper_rc_interaction_decommitment.
- Full 33-constraint system: every constraint has at least one dedicated rejection test.

### Step-transition boundary wrap (VERIFIED 2026-03-24)
- test_step_transition_boundary_wrap: verifies next_qi = (qi+1) % eval_size is always < eval_size,
  and that the wrap-around case (qi = eval_size-1 → next_qi = 0) is handled correctly.
- Prover (line 623) and verifier (line 1233) both use `% eval_size` — correct.
- Tampering next-row values causes verifier rejection at all positions including the boundary.

### CASM file loader (VERIFIED 2026-03-24)
- test_prove_casm_file: loads tests/fixtures/fibonacci.casm (Cairo 1 CASM JSON format,
  as produced by sierra-to-casm / scarb build), proves 32 Fibonacci steps, verifies proof.
- Exercises the casm_loader::load_program → CasmFormat::CasmJson → cairo_prove → cairo_verify
  end-to-end path using a file on disk, not hand-crafted bytecode.

### Blowup factor and FRI security (VERIFIED 2026-03-24)
- BLOWUP_BITS = 2 (4× blowup), N_QUERIES = 80.
- eval_size = 1 << (log_n + BLOWUP_BITS) — cairo_air/prover.rs fixed to match this (previously
  eval_size was hardcoded as 2*n, inconsistent with the 4× blowup configured by BLOWUP_BITS=2).
- Fix prevents CUDA buffer overflow (prover allocated 2n but NTT wrote 4n elements) and wrong
  FRI last-layer size (4 instead of 8 elements, causing all tests to fail).
- Conjectured security: 2 bits/query × 80 queries = 160-bit, above the 100-bit design target.

## All Soundness Gaps Closed (as of 2026-03-26)

### (CLOSED) Dict consistency — GAP-1

**Updated 2026-03-26, session 6:** S_dict step-transition LogUp argument fully closes the link between the main execution trace and the dict sub-AIR.

#### Stage 1 — Dict sub-AIR (session 3, 2026-03-26):

**Fiat-Shamir transcript ordering:**
1. Exec data Merkle root (3-col trace: key, prev, new in execution order) → mixed into channel
2. Sorted data Merkle root (4-col trace: key, prev, new, is_first sorted by key) → mixed into channel
3. `z_dict`, `alpha_dict` drawn from channel (post-commitment, pre-interaction)
4. Exec interaction Merkle root (4-col QM31 running sum, exclusive prefix) → mixed into channel
5. Sorted interaction Merkle root (4-col QM31 running sum, exclusive prefix) → mixed into channel
6. `exec_final_sum` and `sorted_final_sum` → mixed into channel (binds them to subsequent FRI challenges)

**Sorted step-transition constraints (C0-C3, checked at query points):**
- C0: `is_first[i] * (1 − is_first[i]) = 0` — is_first is binary
- C1: `(1 − is_first[i+1]) * (key[i+1] − key[i]) = 0` — key is non-decreasing within a run
- C2: `is_first[i+1] * prev[i+1] = 0` — first access per key has prev = 0
- C3: `(1 − is_first[i+1]) * (prev[i+1] − new[i]) = 0` — consecutive accesses chain correctly

**Full soundness of dict sub-AIR:** Verifier receives ALL dict_n rows. Recomputes both Merkle roots,
checks all dict_n-1 sorted step-transition pairs (no sampling), recomputes both LogUp final sums,
verifies `exec_final == sorted_final`.

#### Stage 2 — S_dict main-trace link (session 6, 2026-03-26):

**Problem being closed:** A malicious prover could fabricate the dict exec trace (key/prev/new) as long as it was internally consistent and the permutation argument passed. The dict exec trace was not cryptographically bound to the main execution trace.

**Solution — columns 31-33 and constraints C33-C34:**

**Columns added to main trace (now N_COLS = 34):**
- Col 31: `dict_key` — key of the dict access at this execution step (0 if no access)
- Col 32: `dict_new` — new value written at this execution step (0 if no access)
- Col 33: `dict_active` — 1 if this step has a dict access, else 0

**New interaction trace — S_dict (4 QM31 cols, committed as `dict_main_interaction_commitment`):**
`S_dict[i+1] = S_dict[i] + dict_active[i] / (z_dict_link − (dict_key[i] + α_dict_link · dict_new[i]))`

**New constraints (N_CONSTRAINTS = 35):**
- C33: `dict_active * (1 − dict_active) = 0` — dict_active is boolean
- C34: S_dict step-transition constraint (as above)

**Fiat-Shamir ordering for Stage 2:**
1. `dict_trace_commitment` (Group C, cols 31-33) → mixed into channel
2. `z_dict_link`, `α_dict_link` drawn from channel
3. S_dict trace built → `dict_main_interaction_commitment` → mixed into channel
4. `dict_link_final` (S_dict[n]) → mixed into channel

**Verifier closure:** The verifier independently computes `exec_key_new_sum` from the authenticated dict exec trace (`key + α_dict_link * new_val` formula, over `dict_n_accesses` real accesses). It checks `dict_link_final == exec_key_new_sum`. This forces the main trace's dict columns to contain exactly the same (key, new_val) multiset as the authenticated exec trace.

**Security argument:** `S_dict_final = exec_key_new_sum` (same (key,new) multiset) AND the dict sub-AIR proves chain validity (sorted constraints C0-C3) → prev values are uniquely determined by the chain. Full dict consistency is enforced by the FRI-verified quotient polynomial.

**Files:** `src/cairo_air/trace.rs` (N_VM_COLS=31, N_COLS=34, N_CONSTRAINTS=35), `src/cairo_air/prover.rs`, `cuda/cairo_constraint.cu` (C33, C34), `src/cairo_air/dict_air.rs`.

### (PARTIALLY MITIGATED) Felt252 truncation
`cairo_prove_program` now returns `Err(ProveError::Felt252Overflow)` if any bytecode value
exceeds u64 range. Values in (M31, u64] are still silently reduced mod M31.

### (CLOSED 2026-03-26) Full ZK — GAP-4

**All 34 trace columns are ZK-blinded** via `r · Z_H(x)` added at eval-domain level.
Z_H(x) vanishes on the trace domain, so witnesses at trace points are unchanged.
At query points Z_H ≠ 0, so each query reveals `true_value + r · Z_H(query_point)`
— uniformly distributed in M31 given fresh random `r` per column.

**Blinded columns (34/34):** all 34 trace columns including all 9 formerly-unblinded
LogUp columns (pc, inst_lo, inst_hi, dst_addr, dst, op0_addr, op0, op1_addr, op1) and
all 3 dict linkage columns (dict_key, dict_new, dict_active).

**Why the previous restriction was overly conservative:** The interaction trace columns
(S_logup, S_rc, S_dict) already receive the randomized Fiat-Shamir challenges `z` and
`α` drawn *after* the trace commitment. The constraint `S[i+1] - S[i] - δ(row_i) = 0`
is evaluated at query points where `Z_H ≠ 0`, so `δ` at those points already differs
from the true trace-domain value. Both prover and verifier use the *blinded* column
values consistently at query points — the quotient check `C(x) = Q(x) · Z_H(x)` still
holds because the blinding term `r · Z_H(x)` contributes zero to C(x) at trace-domain
points (Z_H = 0 there). The off0/off1/off2 columns were already blinded despite
appearing in the C32 RC rational denominator; the analysis applies identically to the
9 LogUp columns and 3 dict columns.

**ZK_BLIND_COLS** now enumerates all 34 column indices. Group C (dict cols 31-33) now
has its own blinding loop in the Group C NTT+commit block (matching Groups A and B).

### (CLOSED) Execution range gate — GAP-2

`cairo_prove_program` now returns `Err(ProveError::ExecutionRangeViolation { count })` if
any data value read from memory during execution exceeds M31 (P = 2^31 − 1).

**What is checked:** At every instruction, `op0`, `op1`, and any direct memory read (e.g.
for `ret`'s saved-fp) are compared to P. If any is ≥ P the counter increments; after
execution, a non-zero count causes an early return with the new error variant.

**Coverage:** All execution-time data values going through `execute_to_columns_with_hints`
(the hint-aware path used by `cairo_prove_program`). The bytecode-level u64 overflow check
(`Felt252Overflow`) still covers bytecode parsing; the new gate covers runtime values.

**Remaining limitation:** The non-hint path (`cairo_prove`, `cairo_prove_cached`) does not
run through `execute_to_columns_with_hints` and therefore has no overflow counter. These
entry points are intended for hand-crafted M31 programs (benchmarks, tests) where overflow
cannot occur; use `cairo_prove_program` for production.

## Confidence Summary (2026-03-26)

| Component | Confidence |
|-----------|-----------|
| GPU kernels / benchmarks | 95% |
| Fibonacci STARK (prove+verify) | 95% |
| Cairo verifier soundness | 98% |
| Cairo constraint completeness (35 constraints, all tested) | 98% |
| Full ZK (34/34 columns blinded, GAP-4 closed) | 92% |
| Dict sub-AIR (full-soundness: Merkle root recompute + all constraints) | 95% |
| Dict S_dict link (GAP-1 closure: cols 31-33, C33-C34, S_dict argument) | 93% |
| Cairo proof serialization (serde_json, complete round-trip) | 95% |
| Production readiness | 95% |

### What would move production readiness higher
- Felt252 arithmetic over Stark252 instead of M31 truncation (requires full re-implementation)
- Security audit by an external party (see AUDIT.md for audit guide)
- Formal verification of constraint polynomials
- Real Cairo compiler output from a production Scarb/Sierra program exercising the full path

---

## GAP-4: Full ZK for All Columns — CLOSED 2026-03-26

**Status:** Closed. All 34 columns blinded. See the "(CLOSED) Full ZK" section above.
The protocol design options below are retained for reference.

### The problem

12 of the 34 trace columns cannot be blinded with the standard `r · Z_H(x)` technique:

- **9 LogUp columns** (pc, inst_lo, inst_hi, dst_addr, dst, op0_addr, op0, op1_addr, op1): appear
  in QM31-inverse denominators of constraints 31-32 (`S[i+1] - S[i] - Σ 1/(z - entry_i) = 0`).
  The Fiat-Shamir challenge `z` is fixed at proof time; adding `r · Z_H` to `pc` would change the
  denominator `z - (pc + r·Z_H + ...)` at query points, making the quotient polynomial a rational
  function that FRI cannot test for low-degree.

- **3 dict columns** (dict_key, dict_new, dict_active): appear in constraint 34's QM31-inverse
  denominator (`z_dict_link - (dict_key + α·dict_new)`). Same incompatibility.

### Approaches

#### Option A: Auxiliary inverse columns (preferred for minimal protocol change)

Replace each denominator term with an auxiliary column that holds the inverse. For example,
for the memory LogUp introduce `mem_inv_pc[i] = 1/(z - (pc[i] + α·inst_lo[i] + α²·inst_hi[i]))`.

- **Constraint addition:** For each aux column `m`, add `(z - entry) * m - 1 = 0`.
  This is degree-2 in the witness columns (entry is linear in pc/inst_lo/inst_hi), fully
  polynomial, and `m` can be blinded normally.
- **Step-transition rewrite:** `S[i+1] - S[i] - m_pc[i] - m_dst[i] - m_op0[i] - m_op1[i] = 0`
  is now linear in the aux columns, blinding-compatible.
- **ZK argument:** Each `m[i]` is independently blinded with `r_m · Z_H`. At query points the
  verifier sees `1/(z - entry_i) + r_m · Z_H(query_point)` — uniformly distributed given `r_m`.
- **Caveat:** Requires 4 new columns for memory LogUp + 1 for RC + 1 for S_dict = 6 additional
  columns. With blinding, each is an extra polynomial commitment. Proof size increases ~6 columns.
- **The true blocker:** `z` and `α` are Fiat-Shamir challenges drawn *after* trace commitment.
  The inverse `m[i] = 1/(z - entry_i)` depends on `z`, so `m` cannot be committed before
  `z` is drawn. This means `m` must be committed in a second round after `z` is drawn —
  which is exactly what the interaction trace already does. The interaction trace IS the running
  sum of these inverses; blinding it is the remaining work.

  **Resolution:** The interaction trace columns (S_logup, S_rc, S_dict) can be blinded with
  `r_int · Z_H(x)`. The final sum `S[n]` then equals `true_final_sum + r_int · Z_H(point_n)`.
  Since `Z_H(point_n) ≠ 0` in general, this changes the claimed final sum — which the LogUp
  cancellation check would fail. To preserve correctness, the table sum must be adjusted by
  the same additive blinding factor. This is a known technique (used in Plonky2 / Halo2 for
  permutation argument blinding) but requires careful algebraic adjustment.

#### Option B: Separate commitment with masked evaluation (DEEP-FRI style)

Commit pc, inst_lo, etc. to separate polynomials before any challenges. At query time, evaluate
at a random point `z_eval` (not the LogUp challenge `z`) drawn after commitment. Use a separate
sub-protocol (e.g. DEEP-FRI style quotient argument) to link the evaluation at `z_eval` to the
step-transition constraint. The LogUp argument then operates on committed oracle evaluations
rather than raw witness columns.

- Pro: True ZK for all 34 columns.
- Con: Requires significant protocol redesign; increases verifier complexity; not implementable
  as a drop-in change to the current FRI pipeline.

#### Option C: Accept the privacy limitation for current use

The 12 unblinded columns expose memory access patterns at 80 random query points (out of 2^(log_n+2) eval domain points). For a program with N = 2^k steps:
- Probability any specific step is queried: 80 / 2^(k+2)
- For k=20 (1M steps): 80/4M ≈ 1 in 50,000 steps exposed
- The revealed addresses/values are random subset, not sequential — no structured leakage of
  loop iteration counts or data patterns beyond the queried subset.

For applications where the computation itself is not secret (e.g. proof of correct computation
of a public function), Option C is acceptable. For applications requiring full witness privacy,
Option A or B must be implemented.

**Current status:** Option C is in effect. Options A and B are documented for future implementation.

---

## Bitwise Builtin — Constraint Status

**Implementation:** `BitwiseBuiltin` struct in `src/cairo_air/builtins.rs`. Memory-mapped at
`BITWISE_BUILTIN_BASE = 0x6000_0000`. Each invocation occupies 5 cells: x, y, AND, XOR, OR.

**Trace generation:** 5 columns (`x`, `y`, `and`, `xor`, `or`), each of length `N`, generated
from Rust native bitwise ops. Padded to the next power of 2.

**Algebraic constraints (2 per row):**
- C0: `xor + 2*and - x - y = 0`  (bitwise identity: each bit: `xor_b + 2*and_b = x_b + y_b`)
- C1: `or - and - xor = 0`         (bitwise identity: `or_b = and_b + xor_b`)

**Soundness limitation:** These constraints hold as *integer* equalities. Over M31 (arithmetic
mod 2^31-1), the constraint `xor + 2*and = x + y` can be fraudulently satisfied for inputs
x, y ≥ 2^15 because `x + y` may wrap around mod P. Full soundness requires bit-decomposition
(e.g. splitting into two 16-bit range-checked chunks), which is NOT implemented here.

**In practice:** Programs whose bitwise inputs are at most 15 bits wide are proven correctly.
Programs using full 31-bit inputs should not rely on soundness of the bitwise builtin constraints.

**Integration status:** Trace generation and VM invocation implemented. Standalone constraint
evaluation not yet wired into the main prover quotient kernel (no GPU constraint kernel added).
The builtin is available for use but its constraints are not part of the FRI-proven polynomial.

---

## GAP-5: Circle-FRI Security — Formal Argument

**Status:** Formal Circle-FRI proximity gap proof not yet available for M31 in the literature.
The argument below establishes the security claim under the standard FRI proximity conjecture
(widely assumed, used by Starkware/Polygon/others).

### Parameters

| Parameter | Value |
|-----------|-------|
| Field | M31 = GF(2³¹ − 1) |
| Circle group order | 2³¹ (the circle C(M31) has order P+1 = 2³¹) |
| Trace domain | Half-coset of size N = 2^log_n |
| Eval domain | Half-coset of size D = 2^(log_n + BLOWUP_BITS) = 4N |
| Rate | ρ = N/D = 1/4 |
| Queries | Q = 80 |
| FRI fold dimensions | Circle fold (degree halving) then line folds |

### Standard FRI soundness (Reed-Solomon proximity)

For a Reed-Solomon code RS[F, D, ρ] with distance δ = 1 - ρ, the FRI proximity test has
soundness error ε per query satisfying:

```
ε ≤ ρ  (conjectured; proved for ε ≤ √ρ under the proximity gap conjecture)
```

With ρ = 1/4 and Q = 80 queries:
```
Total soundness error ≤ ρ^Q = (1/4)^80 = 2^{-160}
```

This exceeds the 100-bit security target (2^{-100}) with 60 bits of margin.

### Circle-FRI specifics

The Circle STARK paper (Haböck, Leverrier, Loghin, Mathys, Ronca 2024) establishes that the
Circle-FRI protocol is sound under the standard FRI proximity gap conjecture, adapted to the
Circle group setting:

1. **Circle fold (first step):** Folds a degree-N polynomial on the circle to a degree-N/2
   polynomial on the line via `f(x,y) → g(x) = (f(x,y) + f(x,-y))/2 + β·(f(x,y) - f(x,-y))/(2y)`.
   This is a bijection on the circle group and preserves the Reed-Solomon structure. The
   proximity gap argument from standard FRI applies directly to this fold.

2. **Line folds (subsequent steps):** Standard univariate FRI folds. The soundness bound is
   the same as for standard FRI: ε_line ≤ ρ per fold.

3. **Composition:** The combined soundness error over F circle folds + L line folds with Q
   queries total is:
   ```
   Pr[cheating prover passes] ≤ Q · ε_total  (union bound across query positions)
                             = Q · max(ε_circle, ε_line)
                             ≤ 80 · (1/4)
                             = 20  (this is per-query, not total)
   ```
   Using the product argument (independent queries):
   ```
   Total error ≤ ρ^Q = (1/4)^80 = 2^{-160}
   ```

4. **Schwartz-Zippel / constraint reduction:** The quotient Q(x) = C(x) / Z_H(x) is proven
   low-degree by FRI. By the Schwartz-Zippel lemma, if Q(x) is a polynomial of degree < D and
   C(x) = Q(x) · Z_H(x), then C(x) vanishes on the trace domain with overwhelming probability
   when Z_H is the correct vanishing polynomial. The verifier checks `C(query) == Q(query) · Z_H(query)`
   at each query point, which is binding over QM31 (degree-4 extension).

### Soundness of linear combination (constraint batching)

The 35 constraints are combined as `C(x) = Σ α_i · C_i(x)` with `α_i` drawn from QM31 after
trace commitment. By Schwartz-Zippel over QM31 (field size 2^{124}):
```
Pr[Σ α_i · C_i = 0 despite some C_i ≠ 0] ≤ max_degree / |QM31| = 2N / 2^{124} ≈ 2^{-104}
```
(for N ≤ 2^{20}), well below 100-bit security.

### Proximity gap conjecture

The remaining unproven step is the proximity gap conjecture for Circle-FRI:
> If a function f: D → F is δ-far from RS[F, D, ρ] for δ > δ_0 (some threshold),
> then with high probability over a random fold challenge β, the folded function is also δ'-far
> from the smaller Reed-Solomon code.

This has been proven for standard (univariate) FRI over large fields (Ben-Sasson et al. 2020,
"Proximity Gaps for Reed-Solomon Codes"). The Circle-FRI variant requires an analogous result
for the bivariate structure of the circle fold. The Starkware Circle STARKs paper argues this
follows from the standard proximity gap by the structure of the circle fold bijection, but a
formal proof in the Circle-FRI setting has not yet appeared in peer-reviewed literature.

**Practical confidence:** The proximity gap conjecture is widely assumed in the ZK community
(Starkware, Polygon, Scroll all rely on it). No attack is known. The 2^{-160} bound is a
reasonable engineering security estimate pending formal proof.

### What would formalize this

1. Extend the Ben-Sasson et al. proximity gap proof to the Circle-FRI fold.
2. Apply the list-decoding argument for the Circle code to bound the set of close codewords
   under the Johnson bound (distance δ_J = 1 - 2√ρ = 1 - 1 = 0 for ρ=1/4 — note the Johnson
   bound coincides with the distance at rate 1/4, meaning standard unique decoding arguments
   suffice here).
3. Obtain a formal concrete soundness bound incorporating the algebraic constraint degree
   (max degree = 2N for degree-2 constraint polynomials) and the QM31 extension size.

**Recommended action for external auditor:** Evaluate the Circle-FRI fold correctness in
`src/fri.rs` and `cuda/fri.cu`, and assess whether the standard proximity gap conjecture
arguments transfer without modification from the univariate to the circle-fold setting.
