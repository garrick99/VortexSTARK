//! Cairo instruction decoding.
//!
//! A Cairo instruction is 63 bits (stored as u64):
//!   [offset0: 16][offset1: 16][offset2: 16][flags: 15]
//!
//! Offsets are biased by 2^15 (stored as unsigned, interpreted as signed).
//! Flags encode the opcode behavior.

/// Decoded Cairo instruction.
#[derive(Clone, Debug, Default)]
pub struct Instruction {
    /// Offsets (biased by 2^15, stored as raw u16)
    pub off0: u16,
    pub off1: u16,
    pub off2: u16,

    // Flag bits (each is 0 or 1)
    pub dst_reg: u32,       // 0=ap, 1=fp
    pub op0_reg: u32,       // 0=ap, 1=fp
    pub op1_imm: u32,       // op1 source: immediate (pc-relative)
    pub op1_fp: u32,        // op1 source: fp-relative
    pub op1_ap: u32,        // op1 source: ap-relative
    pub res_add: u32,       // result = op0 + op1
    pub res_mul: u32,       // result = op0 * op1
    pub pc_jump_abs: u32,   // pc update: absolute jump
    pub pc_jump_rel: u32,   // pc update: relative jump
    pub pc_jnz: u32,        // pc update: conditional jump
    pub ap_add: u32,        // ap update: ap += res
    pub ap_add1: u32,       // ap update: ap += 1
    pub opcode_call: u32,   // opcode: call
    pub opcode_ret: u32,    // opcode: ret
    pub opcode_assert: u32, // opcode: assert_eq
}

impl Instruction {
    /// Decode a 63-bit Cairo instruction from a u64.
    pub fn decode(encoded: u64) -> Self {
        let off0 = (encoded & 0xFFFF) as u16;
        let off1 = ((encoded >> 16) & 0xFFFF) as u16;
        let off2 = ((encoded >> 32) & 0xFFFF) as u16;
        let flags = encoded >> 48;

        Self {
            off0,
            off1,
            off2,
            dst_reg: ((flags >> 0) & 1) as u32,
            op0_reg: ((flags >> 1) & 1) as u32,
            op1_imm: ((flags >> 2) & 1) as u32,
            op1_fp: ((flags >> 3) & 1) as u32,
            op1_ap: ((flags >> 4) & 1) as u32,
            res_add: ((flags >> 5) & 1) as u32,
            res_mul: ((flags >> 6) & 1) as u32,
            pc_jump_abs: ((flags >> 7) & 1) as u32,
            pc_jump_rel: ((flags >> 8) & 1) as u32,
            pc_jnz: ((flags >> 9) & 1) as u32,
            ap_add: ((flags >> 10) & 1) as u32,
            ap_add1: ((flags >> 11) & 1) as u32,
            opcode_call: ((flags >> 12) & 1) as u32,
            opcode_ret: ((flags >> 13) & 1) as u32,
            opcode_assert: ((flags >> 14) & 1) as u32,
        }
    }

    /// Encode back to a 63-bit u64.
    pub fn encode(&self) -> u64 {
        let mut flags: u64 = 0;
        flags |= (self.dst_reg as u64) << 0;
        flags |= (self.op0_reg as u64) << 1;
        flags |= (self.op1_imm as u64) << 2;
        flags |= (self.op1_fp as u64) << 3;
        flags |= (self.op1_ap as u64) << 4;
        flags |= (self.res_add as u64) << 5;
        flags |= (self.res_mul as u64) << 6;
        flags |= (self.pc_jump_abs as u64) << 7;
        flags |= (self.pc_jump_rel as u64) << 8;
        flags |= (self.pc_jnz as u64) << 9;
        flags |= (self.ap_add as u64) << 10;
        flags |= (self.ap_add1 as u64) << 11;
        flags |= (self.opcode_call as u64) << 12;
        flags |= (self.opcode_ret as u64) << 13;
        flags |= (self.opcode_assert as u64) << 14;

        (self.off0 as u64)
            | ((self.off1 as u64) << 16)
            | ((self.off2 as u64) << 32)
            | (flags << 48)
    }

    /// Instruction size: 1 for regular, 2 for immediate (op1 from [pc+1]).
    pub fn size(&self) -> u64 {
        1 + self.op1_imm as u64
    }

    /// Build common instructions.
    pub fn assert_eq_imm(dst_offset: i16, imm_offset: i16) -> Self {
        Self {
            off0: (dst_offset as u16).wrapping_add(0x8000),
            off1: 0x8000, // op0 offset = 0
            off2: (imm_offset as u16).wrapping_add(0x8000),
            op1_imm: 1,
            opcode_assert: 1,
            ..Default::default()
        }
    }

    pub fn add_ap_imm(op0_offset: i16, imm_offset: i16, dst_offset: i16) -> Self {
        Self {
            off0: (dst_offset as u16).wrapping_add(0x8000),
            off1: (op0_offset as u16).wrapping_add(0x8000),
            off2: (imm_offset as u16).wrapping_add(0x8000),
            op1_imm: 1,
            res_add: 1,
            opcode_assert: 1,
            ap_add1: 1,
            ..Default::default()
        }
    }

    pub fn ret() -> Self {
        Self {
            off0: 0x8000u16.wrapping_sub(2), // fp - 2
            off1: 0x8000u16.wrapping_sub(1), // fp - 1
            off2: 0x8000u16.wrapping_sub(1), // fp - 1
            dst_reg: 1,   // fp-based
            op0_reg: 1,   // fp-based
            op1_fp: 1,    // op1 from fp
            pc_jump_abs: 1,
            opcode_ret: 1,
            ..Default::default()
        }
    }

    pub fn call_abs(target_offset: i16) -> Self {
        Self {
            off0: 0x8000, // ap + 0
            off1: 0x8001, // ap + 1
            off2: (target_offset as u16).wrapping_add(0x8000),
            op1_imm: 1,
            pc_jump_abs: 1,
            ap_add: 0,
            opcode_call: 1,
            ..Default::default()
        }
    }

    pub fn jnz_rel(dst_offset: i16, jump_offset: i16) -> Self {
        Self {
            off0: (dst_offset as u16).wrapping_add(0x8000),
            off1: 0x8001, // pc + 1 (instruction size)
            off2: (jump_offset as u16).wrapping_add(0x8000),
            op1_imm: 1,
            pc_jnz: 1,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_encode_roundtrip() {
        let instr = Instruction {
            off0: 0x8000, off1: 0x8001, off2: 0x8002,
            dst_reg: 1, op0_reg: 0, op1_imm: 1, op1_fp: 0, op1_ap: 0,
            res_add: 1, res_mul: 0, pc_jump_abs: 0, pc_jump_rel: 0,
            pc_jnz: 0, ap_add: 0, ap_add1: 1, opcode_call: 0,
            opcode_ret: 0, opcode_assert: 1,
        };
        let encoded = instr.encode();
        let decoded = Instruction::decode(encoded);
        assert_eq!(decoded.encode(), encoded);
    }

    #[test]
    fn test_flag_isolation() {
        for bit in 0..15 {
            let encoded: u64 = 1u64 << (48 + bit);
            let instr = Instruction::decode(encoded);
            let flags = [
                instr.dst_reg, instr.op0_reg, instr.op1_imm, instr.op1_fp,
                instr.op1_ap, instr.res_add, instr.res_mul, instr.pc_jump_abs,
                instr.pc_jump_rel, instr.pc_jnz, instr.ap_add, instr.ap_add1,
                instr.opcode_call, instr.opcode_ret, instr.opcode_assert,
            ];
            for (i, &f) in flags.iter().enumerate() {
                if i == bit as usize {
                    assert_eq!(f, 1, "bit {bit} should be set at position {i}");
                } else {
                    assert_eq!(f, 0, "bit {bit} should not set position {i}");
                }
            }
        }
    }
}
