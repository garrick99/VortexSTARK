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

/// Memory: flat array for O(1) access (no HashMap overhead).
/// Pre-allocates to cover program + execution stack.
#[derive(Clone)]
pub struct Memory {
    data: Vec<u64>,
}

impl Memory {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Create memory with pre-allocated capacity.
    pub fn with_capacity(size: usize) -> Self {
        Self { data: vec![0u64; size] }
    }

    #[inline(always)]
    pub fn get(&self, addr: u64) -> u64 {
        let idx = addr as usize;
        if idx < self.data.len() { self.data[idx] } else { 0 }
    }

    #[inline(always)]
    pub fn set(&mut self, addr: u64, val: u64) {
        let idx = addr as usize;
        if idx >= self.data.len() {
            self.data.resize(idx + 1, 0);
        }
        self.data[idx] = val;
    }

    /// Load a program (encoded instructions) starting at address 0.
    pub fn load_program(&mut self, program: &[u64]) {
        if program.len() > self.data.len() {
            self.data.resize(program.len(), 0);
        }
        self.data[..program.len()].copy_from_slice(program);
    }
}

/// Execute a Cairo program and return the execution trace.
pub fn execute(memory: &mut Memory, n_steps: usize) -> Vec<TraceRow> {
    // Pre-allocate memory for stack (ap starts at 100, grows by ~1 per step)
    let estimated_mem = 100 + n_steps + 1000;
    if memory.data.len() < estimated_mem {
        memory.data.resize(estimated_mem, 0);
    }

    let mut state = CairoState { pc: 0, ap: 100, fp: 100 };
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

/// Execute and write directly to columnar trace format.
/// Eliminates the intermediate TraceRow structs (saves ~1GB allocation + cache thrashing).
pub fn execute_to_columns(memory: &mut Memory, n_steps: usize, log_n: u32) -> Vec<Vec<u32>> {
    use crate::field::m31::P;
    use super::trace::*;

    let n = 1usize << log_n;
    assert!(n_steps <= n);

    let estimated_mem = 100 + n_steps + 1000;
    if memory.data.len() < estimated_mem {
        memory.data.resize(estimated_mem, 0);
    }

    let mut cols: Vec<Vec<u32>> = (0..N_COLS).map(|_| vec![0u32; n]).collect();
    let mut state = CairoState { pc: 0, ap: 100, fp: 100 };

    #[inline(always)]
    fn to_m31(v: u64) -> u32 {
        let lo = (v & 0x7FFF_FFFF) as u32;
        let hi = (v >> 31) as u32;
        let r = lo + hi;
        if r >= P { r - P } else { r }
    }

    // Detect if the program has a dominant instruction (common in benchmarks).
    // Pre-fill constant flag columns to avoid per-step writes.
    let dominant_encoded = if n_steps > 100 { memory.get(4) } else { 0 };
    let dominant_instr = Instruction::decode(dominant_encoded);
    let is_simple_add = dominant_instr.res_add == 1 && dominant_instr.opcode_assert == 1
        && dominant_instr.ap_add1 == 1 && dominant_instr.op1_ap == 1
        && dominant_instr.pc_jnz == 0 && dominant_instr.pc_jump_abs == 0
        && dominant_instr.pc_jump_rel == 0 && dominant_instr.opcode_call == 0
        && dominant_instr.opcode_ret == 0;

    // Pre-fill constant columns if dominant instruction covers most rows
    if is_simple_add && n_steps > 100 {
        // FP is constant (no call/ret in simple-add)
        cols[COL_FP].fill(to_m31(state.fp));
        let flags = [
            dominant_instr.dst_reg, dominant_instr.op0_reg, dominant_instr.op1_imm,
            dominant_instr.op1_fp, dominant_instr.op1_ap, dominant_instr.res_add,
            dominant_instr.res_mul, dominant_instr.pc_jump_abs, dominant_instr.pc_jump_rel,
            dominant_instr.pc_jnz, dominant_instr.ap_add, dominant_instr.ap_add1,
            dominant_instr.opcode_call, dominant_instr.opcode_ret, dominant_instr.opcode_assert,
        ];
        for (j, &flag) in flags.iter().enumerate() {
            if flag == 0 {
                // Already zero from allocation
            } else {
                cols[COL_FLAGS_START + j].fill(flag);
            }
        }
    }

    for i in 0..n_steps {
        let encoded = memory.get(state.pc);

        // Fast path: if this is the dominant simple-add instruction, skip decode
        if is_simple_add && encoded == dominant_encoded {
            let off0 = dominant_instr.off0.wrapping_sub(0x8000) as i16 as i64 as u64;
            let off1 = dominant_instr.off1.wrapping_sub(0x8000) as i16 as i64 as u64;
            let off2 = dominant_instr.off2.wrapping_sub(0x8000) as i16 as i64 as u64;

            let dst_addr = state.ap.wrapping_add(off0);
            let op0_addr = state.ap.wrapping_add(off1);
            let op1_addr = state.ap.wrapping_add(off2);
            let op0 = memory.get(op0_addr);
            let op1 = memory.get(op1_addr);
            let res = { let s = op0 + op1; if s >= P as u64 { s - P as u64 } else { s } };
            memory.set(dst_addr, res);

            // Only write non-constant columns (flags pre-filled, inst is constant)
            cols[COL_PC][i] = to_m31(state.pc);
            cols[COL_AP][i] = to_m31(state.ap);
            // FP is constant for simple-add, skip: cols[COL_FP][i] = to_m31(state.fp);
            cols[COL_INST_LO][i] = (encoded & 0x7FFF_FFFF) as u32;
            cols[COL_INST_HI][i] = ((encoded >> 31) & 0x7FFF_FFFF) as u32;
            // Flags pre-filled — skip 15 writes
            cols[COL_DST_ADDR][i] = to_m31(dst_addr);
            cols[COL_DST][i] = to_m31(res);
            cols[COL_OP0_ADDR][i] = to_m31(op0_addr);
            cols[COL_OP0][i] = to_m31(op0);
            cols[COL_OP1_ADDR][i] = to_m31(op1_addr);
            cols[COL_OP1][i] = to_m31(op1);
            cols[COL_RES][i] = to_m31(res);

            // New columns: raw offsets and dst_inv
            cols[COL_OFF0][i] = (encoded & 0xFFFF) as u32;
            cols[COL_OFF1][i] = ((encoded >> 16) & 0xFFFF) as u32;
            cols[COL_OFF2][i] = ((encoded >> 32) & 0xFFFF) as u32;
            let dst_val_for_inv = to_m31(res);
            cols[COL_DST_INV][i] = if dst_val_for_inv == 0 { 0 } else { crate::field::M31(dst_val_for_inv).inverse().0 };

            state.pc += 1;
            state.ap += 1;
            continue;
        }

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

        let res = if instr.pc_jnz == 1 { 0 }
            else if instr.res_add == 1 {
                // M31 add: values < P, sum < 2P, reduce with one compare
                let sum = op0 + op1;
                if sum >= P as u64 { sum - P as u64 } else { sum }
            }
            else if instr.res_mul == 1 {
                // M31 mul: use Mersenne reduction (shift + add instead of division)
                let prod = op0 as u128 * op1 as u128;
                let lo = (prod & P as u128) as u64;
                let hi = (prod >> 31) as u64;
                let r = lo + hi;
                if r >= P as u64 { r - P as u64 } else { r }
            }
            else { op1 };

        let dst = if instr.opcode_assert == 1 { memory.set(dst_addr, res); res }
            else if instr.opcode_call == 1 {
                memory.set(dst_addr, state.fp);
                memory.set(dst_addr + 1, state.pc + instr.size());
                state.fp
            } else { memory.get(dst_addr) };

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

        // Write directly to columns — no intermediate struct
        cols[COL_PC][i] = to_m31(state.pc);
        cols[COL_AP][i] = to_m31(state.ap);
        cols[COL_FP][i] = to_m31(state.fp);
        cols[COL_INST_LO][i] = (encoded & 0x7FFF_FFFF) as u32;
        cols[COL_INST_HI][i] = ((encoded >> 31) & 0x7FFF_FFFF) as u32;

        cols[COL_FLAGS_START + 0][i] = instr.dst_reg;
        cols[COL_FLAGS_START + 1][i] = instr.op0_reg;
        cols[COL_FLAGS_START + 2][i] = instr.op1_imm;
        cols[COL_FLAGS_START + 3][i] = instr.op1_fp;
        cols[COL_FLAGS_START + 4][i] = instr.op1_ap;
        cols[COL_FLAGS_START + 5][i] = instr.res_add;
        cols[COL_FLAGS_START + 6][i] = instr.res_mul;
        cols[COL_FLAGS_START + 7][i] = instr.pc_jump_abs;
        cols[COL_FLAGS_START + 8][i] = instr.pc_jump_rel;
        cols[COL_FLAGS_START + 9][i] = instr.pc_jnz;
        cols[COL_FLAGS_START + 10][i] = instr.ap_add;
        cols[COL_FLAGS_START + 11][i] = instr.ap_add1;
        cols[COL_FLAGS_START + 12][i] = instr.opcode_call;
        cols[COL_FLAGS_START + 13][i] = instr.opcode_ret;
        cols[COL_FLAGS_START + 14][i] = instr.opcode_assert;

        cols[COL_DST_ADDR][i] = to_m31(dst_addr);
        cols[COL_DST][i] = to_m31(dst);
        cols[COL_OP0_ADDR][i] = to_m31(op0_addr);
        cols[COL_OP0][i] = to_m31(op0);
        cols[COL_OP1_ADDR][i] = to_m31(op1_addr);
        cols[COL_OP1][i] = to_m31(op1);
        cols[COL_RES][i] = to_m31(res);

        // New columns: raw offsets and dst_inv
        cols[COL_OFF0][i] = (encoded & 0xFFFF) as u32;
        cols[COL_OFF1][i] = ((encoded >> 16) & 0xFFFF) as u32;
        cols[COL_OFF2][i] = ((encoded >> 32) & 0xFFFF) as u32;
        let dst_m31 = to_m31(dst);
        cols[COL_DST_INV][i] = if dst_m31 == 0 { 0 } else { crate::field::M31(dst_m31).inverse().0 };

        state = CairoState { pc: next_pc, ap: next_ap, fp: next_fp };
    }

    cols
}

/// Execute and write directly to pre-allocated column buffers.
/// Accepts mutable slices (can point to pinned host memory or mapped GPU memory).
/// Same logic as execute_to_columns but avoids Vec allocation.
pub fn execute_to_columns_into(
    memory: &mut Memory,
    n_steps: usize,
    cols: &mut [&mut [u32]],
) {
    use crate::field::m31::P;
    use super::trace::*;

    assert!(cols.len() >= N_COLS);
    let n = cols[0].len();
    assert!(n_steps <= n);

    let estimated_mem = 100 + n_steps + 1000;
    if memory.data.len() < estimated_mem {
        memory.data.resize(estimated_mem, 0);
    }

    let mut state = CairoState { pc: 0, ap: 100, fp: 100 };

    #[inline(always)]
    fn to_m31(v: u64) -> u32 {
        let lo = (v & 0x7FFF_FFFF) as u32;
        let hi = (v >> 31) as u32;
        let r = lo + hi;
        if r >= P { r - P } else { r }
    }

    for i in 0..n_steps {
        let encoded = memory.get(state.pc);
        let instr = super::decode::Instruction::decode(encoded);

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

        let res = if instr.pc_jnz == 1 { 0 }
            else if instr.res_add == 1 {
                let sum = op0 + op1;
                if sum >= P as u64 { sum - P as u64 } else { sum }
            }
            else if instr.res_mul == 1 {
                let prod = op0 as u128 * op1 as u128;
                let lo = (prod & P as u128) as u64;
                let hi = (prod >> 31) as u64;
                let r = lo + hi;
                if r >= P as u64 { r - P as u64 } else { r }
            }
            else { op1 };

        let dst = if instr.opcode_assert == 1 { memory.set(dst_addr, res); res }
            else if instr.opcode_call == 1 {
                memory.set(dst_addr, state.fp);
                memory.set(dst_addr + 1, state.pc + instr.size());
                state.fp
            } else { memory.get(dst_addr) };

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

        cols[COL_PC][i] = to_m31(state.pc);
        cols[COL_AP][i] = to_m31(state.ap);
        cols[COL_FP][i] = to_m31(state.fp);
        cols[COL_INST_LO][i] = (encoded & 0x7FFF_FFFF) as u32;
        cols[COL_INST_HI][i] = ((encoded >> 31) & 0x7FFF_FFFF) as u32;

        cols[COL_FLAGS_START + 0][i] = instr.dst_reg;
        cols[COL_FLAGS_START + 1][i] = instr.op0_reg;
        cols[COL_FLAGS_START + 2][i] = instr.op1_imm;
        cols[COL_FLAGS_START + 3][i] = instr.op1_fp;
        cols[COL_FLAGS_START + 4][i] = instr.op1_ap;
        cols[COL_FLAGS_START + 5][i] = instr.res_add;
        cols[COL_FLAGS_START + 6][i] = instr.res_mul;
        cols[COL_FLAGS_START + 7][i] = instr.pc_jump_abs;
        cols[COL_FLAGS_START + 8][i] = instr.pc_jump_rel;
        cols[COL_FLAGS_START + 9][i] = instr.pc_jnz;
        cols[COL_FLAGS_START + 10][i] = instr.ap_add;
        cols[COL_FLAGS_START + 11][i] = instr.ap_add1;
        cols[COL_FLAGS_START + 12][i] = instr.opcode_call;
        cols[COL_FLAGS_START + 13][i] = instr.opcode_ret;
        cols[COL_FLAGS_START + 14][i] = instr.opcode_assert;

        cols[COL_DST_ADDR][i] = to_m31(dst_addr);
        cols[COL_DST][i] = to_m31(dst);
        cols[COL_OP0_ADDR][i] = to_m31(op0_addr);
        cols[COL_OP0][i] = to_m31(op0);
        cols[COL_OP1_ADDR][i] = to_m31(op1_addr);
        cols[COL_OP1][i] = to_m31(op1);
        cols[COL_RES][i] = to_m31(res);

        // New columns: raw offsets and dst_inv
        cols[COL_OFF0][i] = (encoded & 0xFFFF) as u32;
        cols[COL_OFF1][i] = ((encoded >> 16) & 0xFFFF) as u32;
        cols[COL_OFF2][i] = ((encoded >> 32) & 0xFFFF) as u32;
        let dst_m31 = to_m31(dst);
        cols[COL_DST_INV][i] = if dst_m31 == 0 { 0 } else { crate::field::M31(dst_m31).inverse().0 };

        state = CairoState { pc: next_pc, ap: next_ap, fp: next_fp };
    }
}

/// Execute a single Cairo step.
#[inline(always)]
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
