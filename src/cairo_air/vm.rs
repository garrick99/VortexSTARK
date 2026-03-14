//! Minimal Cairo VM executor for trace generation.
//!
//! Executes a Cairo program (list of encoded instructions + optional immediates)
//! and records the execution trace: registers, operands, and flags per step.

use crate::field::m31::P;
use super::decode::Instruction;

/// Cairo VM state.
#[derive(Clone, Debug)]
pub struct CairoState {
    pub pc: u64,
    pub ap: u64,
    pub fp: u64,
}

/// A single step of the execution trace.
#[derive(Clone, Debug, Default)]
pub struct TraceRow {
    // Registers
    pub pc: u64,
    pub ap: u64,
    pub fp: u64,
    // Instruction
    pub instruction: u64,
    // Decoded flags (15 bits)
    pub flags: [u32; 15],
    // Operands
    pub dst_addr: u64,
    pub dst: u64,
    pub op0_addr: u64,
    pub op0: u64,
    pub op1_addr: u64,
    pub op1: u64,
    pub res: u64,
    // Next state
    pub next_pc: u64,
    pub next_ap: u64,
    pub next_fp: u64,
}

/// Memory: simple flat address space.
#[derive(Clone)]
pub struct Memory {
    data: std::collections::HashMap<u64, u64>,
}

impl Memory {
    pub fn new() -> Self {
        Self { data: std::collections::HashMap::new() }
    }

    pub fn get(&self, addr: u64) -> u64 {
        *self.data.get(&addr).unwrap_or(&0)
    }

    pub fn set(&mut self, addr: u64, val: u64) {
        self.data.insert(addr, val);
    }

    /// Load a program (encoded instructions) starting at address 0.
    pub fn load_program(&mut self, program: &[u64]) {
        for (i, &word) in program.iter().enumerate() {
            self.set(i as u64, word);
        }
    }
}

/// Execute a Cairo program and return the execution trace.
pub fn execute(memory: &mut Memory, n_steps: usize) -> Vec<TraceRow> {
    let mut state = CairoState { pc: 0, ap: 100, fp: 100 }; // ap/fp start at 100 (above program)
    let mut trace = Vec::with_capacity(n_steps);

    for _step in 0..n_steps {
        let row = execute_step(&state, memory);
        state = CairoState {
            pc: row.next_pc,
            ap: row.next_ap,
            fp: row.next_fp,
        };
        trace.push(row);
    }

    trace
}

/// Execute a single Cairo step.
fn execute_step(state: &CairoState, memory: &mut Memory) -> TraceRow {
    let encoded = memory.get(state.pc);
    let instr = Instruction::decode(encoded);

    // Compute operand addresses
    let dst_base = if instr.dst_reg == 1 { state.fp } else { state.ap };
    let dst_addr = dst_base.wrapping_add(instr.off0.wrapping_sub(0x8000) as i16 as i64 as u64);

    let op0_base = if instr.op0_reg == 1 { state.fp } else { state.ap };
    let op0_addr = op0_base.wrapping_add(instr.off1.wrapping_sub(0x8000) as i16 as i64 as u64);

    let op0 = memory.get(op0_addr);

    // op1 source
    let op1_base = if instr.op1_imm == 1 {
        state.pc // immediate: [pc + off2]
    } else if instr.op1_fp == 1 {
        state.fp
    } else if instr.op1_ap == 1 {
        state.ap
    } else {
        op0 // op1 from [op0 + off2]
    };
    let op1_addr = op1_base.wrapping_add(instr.off2.wrapping_sub(0x8000) as i16 as i64 as u64);
    let op1 = memory.get(op1_addr);

    // Result computation
    let res = if instr.pc_jnz == 1 {
        // For jnz, res is undefined (not used for assert)
        0
    } else if instr.res_add == 1 {
        let sum = (op0 as u128 + op1 as u128) % P as u128;
        sum as u64
    } else if instr.res_mul == 1 {
        let prod = (op0 as u128 * op1 as u128) % P as u128;
        prod as u64
    } else {
        op1 // default: res = op1
    };

    // DST value: for assert_eq, dst = res. For call, dst = fp (pushed). Otherwise read from memory.
    let dst = if instr.opcode_assert == 1 {
        memory.set(dst_addr, res);
        res
    } else if instr.opcode_call == 1 {
        // call: [ap] = fp, [ap+1] = pc + instr_size
        memory.set(dst_addr, state.fp);
        memory.set(dst_addr + 1, state.pc + instr.size());
        state.fp
    } else {
        memory.get(dst_addr)
    };

    // PC update
    let next_pc = if instr.pc_jump_abs == 1 {
        res
    } else if instr.pc_jump_rel == 1 {
        state.pc.wrapping_add(res)
    } else if instr.pc_jnz == 1 {
        if dst != 0 {
            state.pc.wrapping_add(op1)
        } else {
            state.pc + instr.size()
        }
    } else {
        state.pc + instr.size()
    };

    // AP update
    let next_ap = if instr.ap_add == 1 {
        state.ap.wrapping_add(res)
    } else if instr.ap_add1 == 1 {
        state.ap + 1
    } else if instr.opcode_call == 1 {
        state.ap + 2
    } else {
        state.ap
    };

    // FP update
    let next_fp = if instr.opcode_call == 1 {
        state.ap + 2 // new frame
    } else if instr.opcode_ret == 1 {
        dst // restore from stack
    } else {
        state.fp
    };

    let flags = [
        instr.dst_reg, instr.op0_reg, instr.op1_imm, instr.op1_fp, instr.op1_ap,
        instr.res_add, instr.res_mul, instr.pc_jump_abs, instr.pc_jump_rel, instr.pc_jnz,
        instr.ap_add, instr.ap_add1, instr.opcode_call, instr.opcode_ret, instr.opcode_assert,
    ];

    TraceRow {
        pc: state.pc, ap: state.ap, fp: state.fp,
        instruction: encoded,
        flags,
        dst_addr, dst, op0_addr, op0, op1_addr, op1, res,
        next_pc, next_ap, next_fp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_program() {
        // Program: assert [ap] = 42; ap++; assert [ap] = 99; ap++
        let mut mem = Memory::new();

        // assert [ap+0] = 42: dst=ap+0, op1=imm(42), res=op1, assert_eq
        let i1 = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001, // off2=1 → pc+1 = immediate
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        mem.set(0, i1.encode());
        mem.set(1, 42); // immediate value

        let i2 = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        mem.set(2, i2.encode());
        mem.set(3, 99);

        let trace = execute(&mut mem, 2);
        assert_eq!(trace.len(), 2);
        assert_eq!(trace[0].dst, 42);
        assert_eq!(trace[0].next_ap, 101); // ap started at 100, incremented by 1
        assert_eq!(trace[1].dst, 99);
        assert_eq!(trace[1].next_ap, 102);
        assert_eq!(mem.get(100), 42);
        assert_eq!(mem.get(101), 99);
    }

    #[test]
    fn test_add_program() {
        // a = 10; b = 20; c = a + b
        let mut mem = Memory::new();

        // assert [ap+0] = 10
        let i1 = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        mem.set(0, i1.encode());
        mem.set(1, 10);

        // assert [ap+0] = 20
        let i2 = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        mem.set(2, i2.encode());
        mem.set(3, 20);

        // [ap+0] = [ap-2] + [ap-1] (add previous two values)
        let i3 = Instruction {
            off0: 0x8000,           // dst = [ap+0]
            off1: 0x8000u16 - 2,    // op0 = [ap-2]
            off2: 0x8000u16 - 1,    // op1 = [ap-1]
            op1_ap: 1,              // op1 from ap-relative
            res_add: 1,             // res = op0 + op1
            opcode_assert: 1,       // assert dst = res
            ap_add1: 1,
            ..Default::default()
        };
        mem.set(4, i3.encode());

        let trace = execute(&mut mem, 3);
        assert_eq!(trace[2].res, 30); // 10 + 20
        assert_eq!(mem.get(102), 30);
    }

    #[test]
    fn test_fibonacci_program() {
        // Compute fib(10) using Cairo VM
        let mut mem = Memory::new();
        let mut pc = 0u64;

        // [ap+0] = 1 (fib_0)
        let i = Instruction { off0: 0x8000, off1: 0x8000, off2: 0x8001, op1_imm: 1, opcode_assert: 1, ap_add1: 1, ..Default::default() };
        mem.set(pc, i.encode()); pc += 1;
        mem.set(pc, 1); pc += 1;

        // [ap+0] = 1 (fib_1)
        mem.set(pc, i.encode()); pc += 1;
        mem.set(pc, 1); pc += 1;

        // Loop body: [ap] = [ap-2] + [ap-1]; ap++
        let add_instr = Instruction {
            off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
            op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        let loop_start = pc;
        for _ in 0..8 {
            mem.set(pc, add_instr.encode()); pc += 1;
        }

        let trace = execute(&mut mem, 10);

        // Check Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55
        let fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
        for (i, &expected) in fib.iter().enumerate() {
            assert_eq!(mem.get(100 + i as u64), expected, "fib[{i}] wrong");
        }
    }
}
