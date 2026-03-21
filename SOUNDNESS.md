# VortexSTARK Soundness Status

## Constraint System (31 columns, 30 constraints)

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

## Verified Subsystems

### Verifier-side constraint evaluation
- Verifier independently evaluates all 30 constraints at query points
- Checks constraint_sum == quotient_decommitment_value (exact equality)
- Per-constraint-family tamper tests all passing

### LogUp final-value enforcement
- Prover downloads final LogUp running sum from GPU prefix scan
- Final sum bound into Fiat-Shamir transcript
- Verifier recomputes expected memory table sum from program
- test_tamper_logup_final_sum: DETECTED

### Range check implementation
- extract_offsets() extracts 3x16-bit offsets per instruction
- Verifier checks all offsets are in [0, 2^16)
- compute_rc_interaction_trace() and compute_rc_table_sum() implemented
- LogUp cancellation verified in unit tests
- **Wiring into the prover pipeline pending**

### Public inputs
- initial_pc, initial_ap, n_steps, program_hash, program bytecode
- All bound into Fiat-Shamir transcript

### Merkle auth paths
- Generated via cpu_merkle_auth_paths_soa4
- Verified for quotient commitment + all FRI layers

### Pedersen EC constraint system
- Full intermediate EC trace generated
- 29 columns per step, 9 M31 limbs per coordinate
- EC trace committed via NTT + Merkle
- Verifier checks EC doubling/addition constraints at query points

### Multi-program testing
- Fibonacci (add-only): proven + verified
- Multiply-accumulate (mul-only): proven + verified
- Mixed add/mul alternating: proven + verified
- Call/ret initialization pattern: proven + verified

## Remaining Gaps

### Instruction decomposition
M31 field arithmetic (2^31 ≡ 1) prevents expressing the full 63-bit instruction as a single polynomial constraint — the inst_lo/inst_hi split at bit 31 loses 1 bit. LogUp checks (pc, inst_lo) consistency but does not verify inst_hi. A malicious prover could potentially set flag columns inconsistently with the instruction word's upper bits.

### Range check wiring
The range check LogUp argument is fully implemented and tested (extract_offsets, interaction trace, table sum, cancellation tests all pass). It is not yet called from the prover pipeline. Offsets are constrained by the operand address verification constraints, but are not independently proven to be in [0, 2^16).

### Merkle domain separation
Leaf and internal node hashing both use raw Blake2s. Leaf inputs are typically 4-16 bytes (1-4 columns) while internal nodes are always 64 bytes (two 32-byte hashes). This provides implicit length-based separation for traces with fewer than 16 columns, but 16-column traces would have same-length inputs.

## Confidence Summary

| Component | Confidence |
|-----------|-----------|
| GPU kernels / benchmarks | 95% |
| Fibonacci STARK (prove+verify) | 95% |
| Cairo verifier soundness | 90% |
| Cairo constraint completeness | 90% |
| Production readiness | 75% |

### What would move production readiness higher
- Wire range checks into prover pipeline
- Full instruction decomposition (extend LogUp to cover inst_hi, or add multi-limb decomposition columns)
- Merkle domain separation (leaf prefix 0x00, node prefix 0x01)
- Security audit by an external party
- Formal verification of constraint polynomials
- Adversarial testing with intentionally malformed traces
- Real Cairo compiler output (not hand-crafted bytecode)
- Hint execution for non-trivial programs
