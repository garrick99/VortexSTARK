//! CASM (Compiled Cairo Assembly) loader.
//!
//! Parses `.casm` JSON files produced by the Cairo compiler (sierra-to-casm)
//! and Cairo 0 compiled JSON files, converting them to VM bytecode.
//!
//! Supported formats:
//! - CASM JSON (Cairo 1/Sierra compiled): `{ "bytecode": ["0x...", ...], "entry_points_by_type": {...} }`
//! - Cairo 0 compiled JSON: `{ "data": ["0x...", ...], "main": <offset> }`

use serde::Deserialize;
use std::path::Path;

/// A loaded CASM program ready for VM execution.
#[derive(Clone, Debug)]
pub struct CasmProgram {
    /// VM bytecode as u64 values (instructions + immediates).
    pub bytecode: Vec<u64>,
    /// Entry point offset (PC to start execution at).
    pub entry_point: u64,
    /// Program name (from filename or metadata).
    pub name: String,
    /// Required builtins declared by the program.
    pub builtins: Vec<String>,
    /// Source format that was loaded.
    pub format: CasmFormat,
    /// Hints indexed by PC offset.
    pub hints: Vec<(usize, Vec<CasmHint>)>,
    /// Number of felt252 values that overflowed u64 during parsing.
    pub overflow_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CasmFormat {
    /// Cairo 1 CASM (from sierra-to-casm / scarb build)
    CasmJson,
    /// Cairo 0 compiled JSON
    Cairo0Json,
}

/// Minimal hint representation (for future hint execution).
#[derive(Clone, Debug)]
pub struct CasmHint {
    pub name: String,
    pub raw: serde_json::Value,
}

// --- CASM JSON format (Cairo 1) ---

#[derive(Deserialize)]
#[allow(dead_code)]
struct CasmJsonFile {
    #[serde(default)]
    prime: Option<String>,
    #[serde(default)]
    compiler_version: Option<String>,
    bytecode: Vec<serde_json::Value>,
    #[serde(default)]
    hints: serde_json::Value,
    #[serde(default)]
    entry_points_by_type: Option<CasmEntryPoints>,
}

#[derive(Deserialize, Clone, Debug)]
struct CasmEntryPoints {
    #[serde(alias = "External", alias = "EXTERNAL", default)]
    external: Vec<CasmEntryPoint>,
    #[serde(alias = "L1Handler", alias = "L1_HANDLER", default)]
    l1_handler: Vec<CasmEntryPoint>,
    #[serde(alias = "Constructor", alias = "CONSTRUCTOR", default)]
    constructor: Vec<CasmEntryPoint>,
}

#[derive(Deserialize, Clone, Debug)]
#[allow(dead_code)]
struct CasmEntryPoint {
    #[serde(default)]
    selector: Option<String>,
    offset: usize,
    #[serde(default)]
    builtins: Vec<String>,
}

// --- Cairo 0 compiled JSON format ---

#[derive(Deserialize)]
#[allow(dead_code)]
struct Cairo0JsonFile {
    data: Vec<String>,
    #[serde(default)]
    main: Option<usize>,
    #[serde(default)]
    prime: Option<String>,
    #[serde(default)]
    builtins: Vec<String>,
}

/// Load a CASM or Cairo 0 program from a JSON file.
pub fn load_program(path: &Path) -> Result<CasmProgram, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;

    let name = path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Try CASM JSON first, then Cairo 0
    if let Ok(program) = parse_casm_json(&contents, &name) {
        return Ok(program);
    }
    if let Ok(program) = parse_cairo0_json(&contents, &name) {
        return Ok(program);
    }

    Err(format!("cannot parse {}: not a valid CASM or Cairo 0 JSON file", path.display()))
}

/// Parse CASM JSON (Cairo 1 compiled output).
fn parse_casm_json(json: &str, name: &str) -> Result<CasmProgram, String> {
    let file: CasmJsonFile = serde_json::from_str(json)
        .map_err(|e| format!("CASM JSON parse error: {e}"))?;

    // Must have a bytecode field
    if file.bytecode.is_empty() {
        return Err("empty bytecode".to_string());
    }

    let mut bytecode = Vec::with_capacity(file.bytecode.len());
    let mut overflow_count = 0;

    for (i, val) in file.bytecode.iter().enumerate() {
        let v = parse_felt_value(val)
            .map_err(|e| format!("bytecode[{i}]: {e}"))?;
        if v.1 {
            overflow_count += 1;
        }
        bytecode.push(v.0);
    }

    // Determine entry point and builtins
    let (entry_point, builtins) = if let Some(eps) = &file.entry_points_by_type {
        if let Some(ep) = eps.external.first() {
            (ep.offset as u64, ep.builtins.clone())
        } else if let Some(ep) = eps.constructor.first() {
            (ep.offset as u64, ep.builtins.clone())
        } else if let Some(ep) = eps.l1_handler.first() {
            (ep.offset as u64, ep.builtins.clone())
        } else {
            (0u64, vec![])
        }
    } else {
        (0u64, vec![])
    };

    // Parse hints
    let hints = parse_hints(&file.hints);

    if entry_point as usize > bytecode.len() {
        return Err(format!(
            "entry_point offset {} exceeds bytecode length {}",
            entry_point, bytecode.len()
        ));
    }

    Ok(CasmProgram {
        bytecode,
        entry_point,
        name: name.to_string(),
        builtins,
        format: CasmFormat::CasmJson,
        hints,
        overflow_count,
    })
}

/// Parse Cairo 0 compiled JSON.
fn parse_cairo0_json(json: &str, name: &str) -> Result<CasmProgram, String> {
    let file: Cairo0JsonFile = serde_json::from_str(json)
        .map_err(|e| format!("Cairo 0 JSON parse error: {e}"))?;

    if file.data.is_empty() {
        return Err("empty data".to_string());
    }

    let mut bytecode = Vec::with_capacity(file.data.len());
    let mut overflow_count = 0;

    for (i, hex_str) in file.data.iter().enumerate() {
        let v = parse_hex_felt(hex_str)
            .map_err(|e| format!("data[{i}]: {e}"))?;
        if v.1 {
            overflow_count += 1;
        }
        bytecode.push(v.0);
    }

    let entry_point = file.main.unwrap_or(0) as u64;

    if entry_point as usize > bytecode.len() {
        return Err(format!(
            "entry_point offset {} exceeds bytecode length {}",
            entry_point, bytecode.len()
        ));
    }

    Ok(CasmProgram {
        bytecode,
        entry_point,
        name: name.to_string(),
        builtins: file.builtins,
        format: CasmFormat::Cairo0Json,
        hints: vec![],
        overflow_count,
    })
}

/// Parse a felt252 value from JSON (can be hex string, decimal string, or number).
/// Returns (u64_value, overflowed).
fn parse_felt_value(val: &serde_json::Value) -> Result<(u64, bool), String> {
    match val {
        serde_json::Value::String(s) => parse_hex_felt(s),
        serde_json::Value::Number(n) => {
            if let Some(v) = n.as_u64() {
                Ok((v, false))
            } else if let Some(v) = n.as_i64() {
                Ok((v as u64, false))
            } else {
                Err(format!("invalid number: {n}"))
            }
        }
        _ => Err(format!("expected string or number, got: {val}")),
    }
}

/// Parse a hex string (with or without 0x prefix) as a felt252.
/// Returns (u64_value, overflowed) — truncates to u64 if the value exceeds 64 bits.
pub fn parse_hex_felt(s: &str) -> Result<(u64, bool), String> {
    let s = s.trim().trim_start_matches("0x").trim_start_matches("0X");
    if s.is_empty() {
        return Ok((0, false));
    }

    // If it fits in u64 (16 hex digits or fewer), parse directly
    if s.len() <= 16 {
        let v = u64::from_str_radix(s, 16)
            .map_err(|e| format!("invalid hex '{s}': {e}"))?;
        return Ok((v, false));
    }

    // Longer than 16 hex digits — this is a felt252 that overflows u64.
    // Truncate to lowest 64 bits for instructions (which are always ≤63 bits).
    // Large immediates (addresses, hashes) will lose high bits.
    let low_hex = &s[s.len() - 16..];
    let v = u64::from_str_radix(low_hex, 16)
        .map_err(|e| format!("invalid hex '{s}': {e}"))?;
    Ok((v, true))
}

/// Parse CASM hints from the JSON hints field.
pub fn parse_hints(hints_val: &serde_json::Value) -> Vec<(usize, Vec<CasmHint>)> {
    let mut result = Vec::new();

    match hints_val {
        // CASM format: array of [pc_offset, [hint_objects]]
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let serde_json::Value::Array(pair) = item {
                    if pair.len() >= 2 {
                        if let Some(pc) = pair[0].as_u64() {
                            let mut hints = Vec::new();
                            if let serde_json::Value::Array(hint_list) = &pair[1] {
                                for h in hint_list {
                                    let name = extract_hint_name(h);
                                    hints.push(CasmHint {
                                        name,
                                        raw: h.clone(),
                                    });
                                }
                            }
                            if !hints.is_empty() {
                                result.push((pc as usize, hints));
                            }
                        }
                    }
                }
            }
        }
        // Object format: { "pc_offset": [hint_objects] }
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                if let Ok(pc) = key.parse::<usize>() {
                    let mut hints = Vec::new();
                    if let serde_json::Value::Array(hint_list) = val {
                        for h in hint_list {
                            let name = extract_hint_name(h);
                            hints.push(CasmHint {
                                name,
                                raw: h.clone(),
                            });
                        }
                    }
                    if !hints.is_empty() {
                        result.push((pc, hints));
                    }
                }
            }
        }
        _ => {}
    }

    result.sort_by_key(|(pc, _)| *pc);
    result
}

/// Extract hint type name from a hint JSON object.
fn extract_hint_name(hint: &serde_json::Value) -> String {
    // CASM hints are usually { "HintName": { ... } }
    if let serde_json::Value::Object(map) = hint {
        if let Some(key) = map.keys().next() {
            return key.clone();
        }
    }
    "Unknown".to_string()
}

/// Auto-detect the number of steps needed to execute a program.
/// Runs the VM until it hits a ret that would jump to address 0 (halt convention),
/// or until max_steps is reached.
///
/// Uses the same initial state as `execute_to_columns_with_hints` / `cairo_prove_program`
/// and runs hints so that hint-dependent control flow is counted correctly.
pub fn detect_steps(program: &CasmProgram, max_steps: usize) -> usize {
    use super::vm::{Memory, CairoState};
    use super::decode::Instruction;
    use super::hints::{HintContext, run_hints};

    // Initial state must match cairo_prove_program exactly.
    let initial_sp = program.bytecode.len() as u64 + 100;
    let initial_ap = initial_sp + 2;

    let mut memory = Memory::with_capacity(initial_ap as usize + max_steps + 1000);
    memory.load_program(&program.bytecode);

    // Calling convention: sentinel frame below initial AP/FP.
    memory.set(initial_sp,     0); // saved fp  = 0
    memory.set(initial_sp + 1, 0); // return pc = 0 (halt)

    let mut state = CairoState {
        pc: program.entry_point,
        ap: initial_ap,
        fp: initial_ap,
    };

    let mut ctx = HintContext::new();

    for step in 0..max_steps {
        // Run hints before executing the instruction (same order as prove path).
        run_hints(&program.hints, state.pc, step, &state, &mut memory, &mut ctx);

        let encoded = memory.get(state.pc);
        if encoded == 0 && state.pc < program.entry_point {
            // Hit uninitialized memory before entry point — likely halted
            return step;
        }

        let instr = Instruction::decode(encoded);

        // Execute one step
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
            else if instr.res_add == 1 { op0.wrapping_add(op1) }
            else if instr.res_mul == 1 { op0.wrapping_mul(op1) }
            else { op1 };

        let dst = if instr.opcode_assert == 1 {
            memory.set(dst_addr, res);
            res
        } else if instr.opcode_call == 1 {
            memory.set(dst_addr, state.fp);
            memory.set(dst_addr + 1, state.pc + instr.size());
            state.fp
        } else {
            memory.get(dst_addr)
        };

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

        // Halt detection: ret jumped pc to 0 (sentinel)
        if next_pc == 0 && instr.opcode_ret == 1 {
            return step + 1;
        }

        // Infinite loop detection: pc didn't change
        if next_pc == state.pc && instr.op1_imm == 0 {
            return step + 1;
        }

        state = CairoState { pc: next_pc, ap: next_ap, fp: next_fp };
    }

    max_steps
}

/// Print a summary of a loaded CASM program.
pub fn print_summary(program: &CasmProgram) {
    eprintln!("Program: {}", program.name);
    eprintln!("  Format:      {:?}", program.format);
    eprintln!("  Bytecode:    {} felts", program.bytecode.len());
    eprintln!("  Entry point: {}", program.entry_point);
    if !program.builtins.is_empty() {
        eprintln!("  Builtins:    {}", program.builtins.join(", "));
    }
    if !program.hints.is_empty() {
        let total_hints: usize = program.hints.iter().map(|(_, h)| h.len()).sum();
        eprintln!("  Hints:       {} at {} PCs", total_hints, program.hints.len());
    }
    if program.overflow_count > 0 {
        eprintln!("  WARNING:     {} felt252 values truncated to u64", program.overflow_count);
    }
}

/// Disassemble and print the first N instructions of a program.
pub fn disassemble(program: &CasmProgram, max_instructions: usize) {
    use super::decode::Instruction;

    let mut pc = 0usize;
    let mut count = 0;

    while pc < program.bytecode.len() && count < max_instructions {
        let encoded = program.bytecode[pc];
        let instr = Instruction::decode(encoded);

        let opcode = if instr.opcode_call == 1 { "call" }
            else if instr.opcode_ret == 1 { "ret" }
            else if instr.opcode_assert == 1 { "assert_eq" }
            else { "nop" };

        let res_op = if instr.res_add == 1 { "add" }
            else if instr.res_mul == 1 { "mul" }
            else { "mov" };

        let pc_update = if instr.pc_jump_abs == 1 { " jmp_abs" }
            else if instr.pc_jump_rel == 1 { " jmp_rel" }
            else if instr.pc_jnz == 1 { " jnz" }
            else { "" };

        let ap_update = if instr.ap_add == 1 { " ap+=" }
            else if instr.ap_add1 == 1 { " ap++" }
            else { "" };

        let imm = if instr.op1_imm == 1 && pc + 1 < program.bytecode.len() {
            format!(" imm=0x{:x}", program.bytecode[pc + 1])
        } else {
            String::new()
        };

        eprintln!("  [{pc:4}] {opcode:10} {res_op}{pc_update}{ap_update}{imm}  (0x{encoded:016x})");

        pc += instr.size() as usize;
        count += 1;
    }

    if pc < program.bytecode.len() {
        eprintln!("  ... ({} more felts)", program.bytecode.len() - pc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hex_felt_small() {
        let (v, overflow) = parse_hex_felt("0x2a").unwrap();
        assert_eq!(v, 42);
        assert!(!overflow);
    }

    #[test]
    fn test_parse_hex_felt_instruction() {
        // A typical Cairo instruction (63 bits)
        let (v, overflow) = parse_hex_felt("0x480680017fff8000").unwrap();
        assert_eq!(v, 0x480680017fff8000);
        assert!(!overflow);
    }

    #[test]
    fn test_parse_hex_felt_overflow() {
        // STARK prime (252 bits) — should truncate to low 64 bits
        let (v, overflow) = parse_hex_felt(
            "0x800000000000011000000000000000000000000000000000000000000000001"
        ).unwrap();
        assert!(overflow);
        assert_eq!(v, 1); // low 64 bits of the STARK prime
    }

    #[test]
    fn test_parse_casm_json() {
        let json = r#"{
            "prime": "0x800000000000011000000000000000000000000000000000000000000000001",
            "compiler_version": "2.6.0",
            "bytecode": ["0x480680017fff8000", "0x1", "0x208b7fff7fff7ffe"],
            "hints": [],
            "entry_points_by_type": {
                "External": [{"selector": "0x1234", "offset": 0, "builtins": ["range_check"]}],
                "L1Handler": [],
                "Constructor": []
            }
        }"#;

        let program = parse_casm_json(json, "test").unwrap();
        assert_eq!(program.bytecode.len(), 3);
        assert_eq!(program.entry_point, 0);
        assert_eq!(program.builtins, vec!["range_check"]);
        assert_eq!(program.format, CasmFormat::CasmJson);
    }

    #[test]
    fn test_parse_cairo0_json() {
        let json = r#"{
            "data": ["0x480680017fff8000", "0x1", "0x208b7fff7fff7ffe"],
            "main": 0,
            "builtins": ["pedersen"]
        }"#;

        let program = parse_cairo0_json(json, "test").unwrap();
        assert_eq!(program.bytecode.len(), 3);
        assert_eq!(program.entry_point, 0);
        assert_eq!(program.builtins, vec!["pedersen"]);
        assert_eq!(program.format, CasmFormat::Cairo0Json);
    }

    #[test]
    fn test_detect_steps_simple() {
        // A simple program: assert [ap] = 42; ret
        use super::super::decode::Instruction;

        let assert_imm = Instruction {
            off0: 0x8000, off1: 0x8000, off2: 0x8001,
            op1_imm: 1, opcode_assert: 1, ap_add1: 1,
            ..Default::default()
        };
        let ret = Instruction::ret();

        let bytecode = vec![assert_imm.encode(), 42, ret.encode()];
        let program = CasmProgram {
            bytecode,
            entry_point: 0,
            name: "test".to_string(),
            builtins: vec![],
            format: CasmFormat::CasmJson,
            hints: vec![],
            overflow_count: 0,
        };

        let steps = detect_steps(&program, 1000);
        // Should detect halt after assert + ret = 2 steps
        assert!(steps <= 3, "expected ~2 steps, got {steps}");
    }
}
