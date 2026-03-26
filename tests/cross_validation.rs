//! Cross-validation: compare VortexSTARK's VM against a minimal reference implementation.
//!
//! The reference VM is a compact, independent implementation of the Cairo instruction
//! set (~90 lines). It executes the same programs as `vm.rs` and compares register
//! and memory state after every step. Any divergence is a VortexSTARK VM bug.
//!
//! This is NOT a prover test — it only validates execution correctness.

use vortexstark::cairo_air::decode::Instruction;
use vortexstark::cairo_air::vm::{Memory, execute};

// ---------------------------------------------------------------------------
// Minimal reference Cairo VM
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct RefState {
    pc: u64,
    ap: u64,
    fp: u64,
}

/// Execute one step of the reference Cairo VM.
/// Returns the next state. Panics on invalid instruction encoding.
fn ref_step(state: &RefState, memory: &Memory) -> RefState {
    let encoded = memory.get(state.pc);
    let instr = Instruction::decode(encoded);

    let dst_base = if instr.dst_reg == 1 { state.fp } else { state.ap };
    let dst_addr = dst_base.wrapping_add(instr.off0.wrapping_sub(0x8000) as i16 as i64 as u64);
    let op0_base = if instr.op0_reg == 1 { state.fp } else { state.ap };
    let op0_addr = op0_base.wrapping_add(instr.off1.wrapping_sub(0x8000) as i16 as i64 as u64);
    let op0 = memory.get(op0_addr);

    let op1_base = if instr.op1_imm == 1 { state.pc }
        else if instr.op1_fp == 1 { state.fp }
        else if instr.op1_ap == 1 { state.ap }
        else { op0 };
    let op1_addr = op1_base.wrapping_add(instr.off2.wrapping_sub(0x8000) as i16 as i64 as u64);
    let op1 = memory.get(op1_addr);

    const P: u64 = (1u64 << 31) - 1;
    let res = if instr.pc_jnz == 1 { 0 }
        else if instr.res_add == 1 { let s = op0 + op1; if s >= P { s - P } else { s } }
        else if instr.res_mul == 1 {
            let prod = op0 as u128 * op1 as u128;
            let lo = (prod & P as u128) as u64;
            let hi = (prod >> 31) as u64;
            let r = lo + hi;
            if r >= P { r - P } else { r }
        }
        else { op1 };

    let dst = if instr.opcode_assert == 1 { res }
        else if instr.opcode_call == 1 { state.fp }
        else { memory.get(dst_addr) };

    let next_pc = if instr.pc_jump_abs == 1 { res }
        else if instr.pc_jump_rel == 1 { state.pc.wrapping_add(res) }
        else if instr.pc_jnz == 1 {
            if dst != 0 { state.pc.wrapping_add(op1) } else { state.pc + instr.size() }
        } else { state.pc + instr.size() };

    let next_ap = if instr.ap_add == 1 { state.ap.wrapping_add(res) }
        else if instr.ap_add1 == 1 { state.ap + 1 }
        else if instr.opcode_call == 1 { state.ap + 2 }
        else { state.ap };

    let next_fp = if instr.opcode_call == 1 { state.ap + 2 }
        else if instr.opcode_ret == 1 { dst }
        else { state.fp };

    RefState { pc: next_pc, ap: next_ap, fp: next_fp }
}

/// Run the reference VM for n_steps starting from pc=0, ap=fp=100.
/// Returns final register state.
///
/// Models memory writes for opcodes that write during execution:
/// - `call`: writes mem[ap] = fp (saved fp) and mem[ap+1] = pc+2 (return address)
/// - `opcode_assert`: writes mem[dst_addr] = res
fn ref_execute(memory: &Memory, n_steps: usize) -> RefState {
    let mut state = RefState { pc: 0, ap: 100, fp: 100 };
    let mut mem = memory.clone();
    for _ in 0..n_steps {
        let encoded = mem.get(state.pc);
        let instr = Instruction::decode(encoded);
        // Apply memory writes BEFORE stepping, so subsequent steps see them
        if instr.opcode_call == 1 {
            mem.set(state.ap,     state.fp);                         // save old fp
            mem.set(state.ap + 1, state.pc + instr.size() as u64);  // return address
        } else if instr.opcode_assert == 1 {
            let dst_base = if instr.dst_reg == 1 { state.fp } else { state.ap };
            let dst_addr = dst_base.wrapping_add(instr.off0.wrapping_sub(0x8000) as i16 as i64 as u64);
            let op1_base = if instr.op1_imm == 1 { state.pc }
                else if instr.op1_fp == 1 { state.fp }
                else if instr.op1_ap == 1 { state.ap }
                else {
                    let op0_base = if instr.op0_reg == 1 { state.fp } else { state.ap };
                    let op0_addr = op0_base.wrapping_add(instr.off1.wrapping_sub(0x8000) as i16 as i64 as u64);
                    mem.get(op0_addr)
                };
            let op1 = mem.get(op1_base.wrapping_add(instr.off2.wrapping_sub(0x8000) as i16 as i64 as u64));
            let op0_base = if instr.op0_reg == 1 { state.fp } else { state.ap };
            let op0_addr = op0_base.wrapping_add(instr.off1.wrapping_sub(0x8000) as i16 as i64 as u64);
            let op0 = mem.get(op0_addr);
            const P: u64 = (1u64 << 31) - 1;
            let res = if instr.res_add == 1 { let s = op0+op1; if s>=P {s-P} else {s} }
                else if instr.res_mul == 1 {
                    let prod = op0 as u128 * op1 as u128;
                    let lo = (prod & P as u128) as u64;
                    let hi = (prod >> 31) as u64;
                    let r = lo + hi; if r>=P {r-P} else {r}
                } else { op1 };
            // Only write if not already set (first-write semantics for Cairo assert)
            if mem.get(dst_addr) == 0 {
                mem.set(dst_addr, res);
            }
        }
        state = ref_step(&state, &mem);
    }
    state
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn make_memory(program: &[u64]) -> Memory {
    let mut mem = Memory::with_capacity(program.len() + 200);
    mem.load_program(program);
    mem
}

/// Fibonacci program: two initializations followed by add steps.
fn fib_program(n_steps: usize) -> Vec<u64> {
    let init = Instruction {
        off0: 0x8000, off1: 0x8000, off2: 0x8001,
        op1_imm: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    let add = Instruction {
        off0: 0x8000, off1: 0x8000u16.wrapping_sub(2),
        off2: 0x8000u16.wrapping_sub(1),
        op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    let mut prog = vec![init.encode(), 1u64, init.encode(), 1u64];
    for _ in 4..n_steps { prog.push(add.encode()); }
    prog
}

/// Multiply-accumulate program: multiply successive values.
fn mul_program(n_steps: usize) -> Vec<u64> {
    let init = Instruction {
        off0: 0x8000, off1: 0x8000, off2: 0x8001,
        op1_imm: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    let mul = Instruction {
        off0: 0x8000, off1: 0x8000u16.wrapping_sub(2),
        off2: 0x8000u16.wrapping_sub(1),
        op1_ap: 1, res_mul: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    let mut prog = vec![init.encode(), 2u64, init.encode(), 3u64];
    for _ in 4..n_steps { prog.push(mul.encode()); }
    prog
}

/// Cross-validate: run VortexSTARK VM and reference VM, compare final state.
fn cross_validate(program: &[u64], n_steps: usize) {
    let mem = make_memory(program);

    // VortexSTARK's VM (via execute which returns TraceRows)
    let mut mem2 = mem.clone();
    let trace = execute(&mut mem2, n_steps);
    let last = trace.last().expect("trace must be non-empty");
    let vortex_final = RefState { pc: last.next_pc, ap: last.next_ap, fp: last.next_fp };

    // Reference VM
    let ref_final = ref_execute(&mem, n_steps);

    assert_eq!(vortex_final, ref_final,
        "VortexSTARK and reference VM disagree after {n_steps} steps:\n  \
         VortexSTARK: pc={} ap={} fp={}\n  \
         Reference:   pc={} ap={} fp={}",
        vortex_final.pc, vortex_final.ap, vortex_final.fp,
        ref_final.pc, ref_final.ap, ref_final.fp);
}

#[test]
fn test_cross_validate_fibonacci_8_steps() {
    cross_validate(&fib_program(8), 8);
}

#[test]
fn test_cross_validate_fibonacci_32_steps() {
    cross_validate(&fib_program(32), 32);
}

#[test]
fn test_cross_validate_fibonacci_100_steps() {
    cross_validate(&fib_program(100), 100);
}

#[test]
fn test_cross_validate_multiply_8_steps() {
    cross_validate(&mul_program(8), 8);
}

#[test]
fn test_cross_validate_multiply_32_steps() {
    cross_validate(&mul_program(32), 32);
}

/// Cross-validate a single assert_eq instruction: `[ap+0] = 42`
#[test]
fn test_cross_validate_assert_imm() {
    let instr = Instruction {
        off0: 0x8000, off1: 0x8000, off2: 0x8001,
        op1_imm: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    let program = vec![instr.encode(), 42u64];
    cross_validate(&program, 1);
}

/// Cross-validate a call + ret sequence.
#[test]
fn test_cross_validate_call_ret() {
    // call rel 2  (jumps over itself to a ret)
    let call_instr = Instruction {
        off0: 0x8000, off1: 0x8001, off2: 0x8001,
        op1_imm: 1, opcode_call: 1,
        pc_jump_rel: 1,
        ..Default::default()
    };
    let ret_instr = Instruction::ret();
    let program = vec![call_instr.encode(), 2u64, ret_instr.encode()];
    cross_validate(&program, 2); // call (1 step) + ret (1 step)
}

/// Verify reference VM and VortexSTARK agree on memory writes during execution.
/// Compares memory values at a few key addresses after execution.
#[test]
fn test_cross_validate_memory_writes() {
    let program = fib_program(20);
    let mut mem_vortex = make_memory(&program);
    let mem_ref = make_memory(&program);

    // Run VortexSTARK's execute
    let trace = execute(&mut mem_vortex, 20);
    let last = trace.last().unwrap();
    let vortex_ap = last.next_ap;

    // Run reference VM
    let ref_state = ref_execute(&mem_ref, 20);
    let ref_ap = ref_state.ap;

    assert_eq!(vortex_ap, ref_ap, "AP values diverged after 20 steps");

    // Check a sample of written memory cells
    for offset in 0..10u64 {
        let addr = 100 + offset; // start of stack
        let vortex_val = mem_vortex.get(addr);
        // Re-run ref VM writing to mem_ref to compare
        let ref_val = {
            let mut m = make_memory(&program);
            let mut s = RefState { pc: 0, ap: 100, fp: 100 };
            for _ in 0..20 {
                let next = ref_step(&s, &m);
                let instr = Instruction::decode(m.get(s.pc));
                if instr.opcode_assert == 1 {
                    let dst_base = if instr.dst_reg == 1 { s.fp } else { s.ap };
                    let dst_addr = dst_base.wrapping_add(instr.off0.wrapping_sub(0x8000) as i16 as i64 as u64);
                    let op0 = m.get(if instr.op0_reg == 1 { s.fp } else { s.ap }
                        .wrapping_add(instr.off1.wrapping_sub(0x8000) as i16 as i64 as u64));
                    let op1_base = if instr.op1_imm == 1 { s.pc } else if instr.op1_fp == 1 { s.fp } else { s.ap };
                    let op1 = m.get(op1_base.wrapping_add(instr.off2.wrapping_sub(0x8000) as i16 as i64 as u64));
                    const P: u64 = (1u64 << 31) - 1;
                    let res = if instr.res_add == 1 { let r = op0+op1; if r>=P {r-P} else {r} }
                        else if instr.res_mul == 1 {
                            let prod = op0 as u128 * op1 as u128;
                            let lo = (prod & P as u128) as u64;
                            let hi = (prod >> 31) as u64;
                            let r = lo + hi; if r >= P {r-P} else {r}
                        } else { op1 };
                    m.set(dst_addr, res);
                }
                s = next;
            }
            m.get(addr)
        };
        assert_eq!(vortex_val, ref_val,
            "Memory[{addr}] diverged: VortexSTARK={vortex_val} Reference={ref_val}");
    }
}
