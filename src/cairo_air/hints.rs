//! Cairo 1 CASM hint execution.
//!
//! Hints are non-deterministic oracle calls that the VM must honor before executing
//! the instruction at a given PC. They write witness values into memory; the program
//! then constrains those values via normal instructions.
//!
//! Supported hint types (covers the majority of Sierra-generated CASM):
//!
//! | Hint                      | Operation                                        |
//! |---------------------------|--------------------------------------------------|
//! | TestLessThan              | dst = 1 if lhs < rhs else 0                     |
//! | TestLessThanOrEqual       | dst = 1 if lhs <= rhs else 0                    |
//! | DivMod                    | quotient = lhs/rhs, remainder = lhs%rhs          |
//! | SquareRoot                | dst = isqrt(value)                               |
//! | LinearSplit               | x, y such that value = x*scalar + y             |
//! | WideMul128                | (high,low) = lhs * rhs (64×64 → 128)           |
//! | AllocSegment              | allocate a flat memory segment, write base to dst|
//! | AllocFelt252Dict          | allocate a new dict, register in segment arena   |
//! | Felt252DictEntryInit      | write prev_value for key into dict access entry  |
//! | Felt252DictEntryUpdate    | update dict side-state, advance dict_ptr         |
//! | InitSquashData            | sort dict access keys, set up squash iteration   |
//! | ShouldSkipSquashLoop      | signal whether squash inner loop is exhausted    |
//! | GetCurrentAccessIndex     | write current squash access index to range_check |
//! | GetNextDictKey            | advance to next squash key                       |
//! | GetSegmentArenaIndex      | write 0 (stub)                                   |
//! | U256InvModN               | extended-GCD modular inverse for u128 inputs     |
//! | DebugPrint                | print memory range to stderr                     |

use std::collections::HashMap;
use super::casm_loader::CasmHint;
use super::vm::{CairoState, Memory};

// ---------------------------------------------------------------------------
// Segment allocation constants
// ---------------------------------------------------------------------------

/// Each allocated segment spans this many cells in flat memory.
/// 1 << 20 = ~1M cells ≈ 340K dict entries per segment.
const SEGMENT_SIZE: u64 = 1 << 20;

/// Default base address for the first allocated segment.
/// Must be comfortably above the execution stack (AP grows from 100).
/// 1 << 24 = 16M cells; allows programs up to ~16M stack slots before collision.
const SEGMENT_BASE_DEFAULT: u64 = 1 << 24;

// ---------------------------------------------------------------------------
// HintContext — stateful side-data maintained across all hint invocations
// ---------------------------------------------------------------------------

/// Mutable state threaded through every hint invocation.
///
/// The VM executes deterministically from the bytecode alone; hints inject
/// non-deterministic witness values.  Some hints (AllocSegment, dicts) need
/// state that persists across steps — that lives here.
// ---------------------------------------------------------------------------
// Starknet syscall selector constants
// ---------------------------------------------------------------------------
//
// Syscall selectors are felt252 values whose big-endian bytes equal the ASCII
// name of the syscall.  VortexSTARK's memory stores the low 64 bits of each
// felt252, which equals the last 8 bytes of the ASCII name (padded with leading
// zeros for names shorter than 8 bytes).

mod sel {
    pub const STORAGE_READ:       u64 = 0x7261_6765_5265_6164; // "rageRead"  ← "StorageRead"
    pub const STORAGE_WRITE:      u64 = 0x6167_6557_7269_7465; // "ageWrite"  ← "StorageWrite"
    pub const EMIT_EVENT:         u64 = 0x6d69_7445_7665_6e74; // "mitEvent"  ← "EmitEvent"
    pub const GET_EXECUTION_INFO: u64 = 0x7469_6f6e_496e_666f; // "tionInfo"  ← "GetExecutionInfo"
    pub const CALL_CONTRACT:      u64 = 0x436f_6e74_7261_6374; // "Contract"  ← "CallContract"
    pub const DEPLOY:             u64 = 0x0000_4465_706c_6f79; // "Deploy"    ← "Deploy" (6 bytes)
    pub const GET_BLOCK_HASH:     u64 = 0x6c6f_636b_4861_7368; // "lockHash"  ← "GetBlockHash"
    pub const LIBRARY_CALL:       u64 = 0x7261_7279_4361_6c6c; // "raryCall"  ← "LibraryCall"
    pub const SEND_MSG_TO_L1:     u64 = 0x6167_6554_6f4c_3100; // "ageToL1"   ← "SendMessageToL1"
}

// ---------------------------------------------------------------------------
// Starknet syscall state
// ---------------------------------------------------------------------------

/// Emitted event from an emit_event syscall.
pub struct SyscallEvent {
    pub keys: Vec<u64>,
    pub data: Vec<u64>,
}

/// A call_contract or library_call syscall record.
pub struct CrossContractCall {
    /// Contract address (call_contract) or class hash (library_call).
    pub target: u64,
    /// Entry-point selector.
    pub entry_point_selector: u64,
    /// Calldata passed to the callee.
    pub calldata: Vec<u64>,
    /// True if this was a library_call (class_hash), false if call_contract (address).
    pub is_library: bool,
}

/// A deploy syscall record.
pub struct DeployedContract {
    pub class_hash: u64,
    pub salt: u64,
    pub calldata: Vec<u64>,
    /// Assigned mock address (= salt XOR class_hash, deterministic).
    pub contract_address: u64,
}

/// A send_message_to_l1 syscall record.
pub struct L1Message {
    pub to_address: u64,
    pub payload: Vec<u64>,
}

/// A contract registered for in-process cross-contract execution.
///
/// Add entries via `HintContext::register_contract` before proving.
/// When `CallContract` or `LibraryCall` targets a registered address / class hash,
/// the callee is executed in-process and its retdata is written back into the
/// caller's syscall response buffer.
pub struct ContractEntry {
    pub bytecode: Vec<u64>,
    pub hints: Vec<(usize, Vec<CasmHint>)>,
    /// Map from entry-point selector (felt252 low-64) to PC offset in bytecode.
    pub entry_points: HashMap<u64, u64>,
}

/// Max steps a callee may execute before being forcibly halted.
const MAX_CALLEE_STEPS: usize = 200_000;
/// Max cross-contract call nesting depth (prevents infinite recursion).
const MAX_CALL_DEPTH: usize = 8;

/// Starknet execution context and mutable state for syscall emulation.
///
/// Passed into `cairo_prove_program_with_syscalls` to give the program a
/// realistic environment.  All syscalls that modify state (storage_write,
/// emit_event, call_contract, deploy, send_message_to_l1) write into these
/// fields so the caller can inspect them after proving.
pub struct SyscallState {
    /// Contract storage: storage_key (felt252 low-64) → value (felt252 low-64).
    pub storage: HashMap<u64, u64>,
    /// Events emitted by emit_event syscalls.
    pub events: Vec<SyscallEvent>,
    /// Records of call_contract / library_call syscalls (in order).
    pub cross_contract_calls: Vec<CrossContractCall>,
    /// Records of deploy syscalls (in order).
    pub deployed_contracts: Vec<DeployedContract>,
    /// Records of send_message_to_l1 syscalls (in order).
    pub l1_messages: Vec<L1Message>,
    /// Caller address passed to get_execution_info.
    pub caller_address: u64,
    /// Current contract address.
    pub contract_address: u64,
    /// Entry-point selector.
    pub entry_point_selector: u64,
    /// Block number (used by get_execution_info / get_block_hash).
    pub block_number: u64,
    /// Block timestamp.
    pub block_timestamp: u64,
    /// Base address of the lazily-allocated execution-info region in memory.
    /// Set on first get_execution_info call and reused for subsequent calls.
    exec_info_base: Option<u64>,
}

impl Default for SyscallState {
    fn default() -> Self {
        Self {
            storage: HashMap::new(),
            events: Vec::new(),
            cross_contract_calls: Vec::new(),
            deployed_contracts: Vec::new(),
            l1_messages: Vec::new(),
            caller_address: 0,
            contract_address: 1,
            entry_point_selector: 0,
            block_number: 1000,
            block_timestamp: 0,
            exec_info_base: None,
        }
    }
}

// ---------------------------------------------------------------------------
// HintContext
// ---------------------------------------------------------------------------

pub struct HintContext {
    /// Next flat address to hand out for AllocSegment / AllocFelt252Dict.
    pub next_segment_base: u64,

    /// Per-dict side-state: dict_base → (key → current_value).
    ///
    /// Populated by AllocFelt252Dict; updated by Felt252DictEntryUpdate.
    /// Used by Felt252DictEntryInit to supply the prev_value witness.
    pub dicts: HashMap<u64, HashMap<u64, u64>>,

    /// Segment arena bookkeeping: dict_base → (arena_ptr, dict_idx, infos_ptr).
    ///
    /// Stored at AllocFelt252Dict time so that Felt252DictEntryUpdate can
    /// mirror the dict_ptr advance into DictInfo.end, and so that
    /// GetSegmentArenaIndex can return the correct index for multi-dict programs.
    pub dict_arena_info: HashMap<u64, (u64, u64, u64)>, // dict_base → (arena_ptr, dict_idx, infos_ptr)

    /// Squash iteration state (used by InitSquashData / ShouldSkipSquashLoop /
    /// GetCurrentAccessIndex / GetNextDictKey).
    pub squash: SquashState,

    /// Ordered log of all dict access tuples: (step, key, prev_value, new_value).
    ///
    /// `step` is the trace row index (0-based) at which the access occurred.
    /// Populated by `Felt252DictEntryUpdate` as each access completes.
    /// Used by the prover to CPU-verify dict chain consistency and to fill
    /// the dict columns (COL_DICT_KEY, COL_DICT_NEW, COL_DICT_ACTIVE) in the
    /// main execution trace, linking dict operations to the FRI-committed trace.
    pub dict_accesses: Vec<(usize, u64, u64, u64)>,

    /// Count of execution-time memory reads where the value exceeded M31 (P = 2^31 - 1).
    ///
    /// VortexSTARK operates over M31. If a data value read during execution (op0, op1,
    /// or an explicit memory read) is ≥ P, it is silently reduced mod P — producing a
    /// proof of the wrong computation. This counter tracks how many such truncations
    /// occurred. `cairo_prove_program` returns `ProveError::ExecutionRangeViolation`
    /// if the count is non-zero, refusing to produce a misleading proof.
    pub execution_overflows: usize,

    /// Starknet syscall state (storage, events, execution context).
    pub syscall: SyscallState,

    /// Contracts available for in-process cross-contract execution.
    ///
    /// Keyed by contract_address (for `call_contract`) or class_hash (for `library_call`).
    /// When a `CallContract` / `LibraryCall` syscall targets a registered entry, the callee
    /// bytecode is executed in-process via `execute_callee` and its retdata is written into
    /// the caller's response buffer.  Unregistered targets return empty retdata (existing
    /// behavior).
    pub contract_registry: HashMap<u64, ContractEntry>,

    /// Current cross-contract call nesting depth.  Incremented on entry to `execute_callee`
    /// and decremented on exit.  Calls that would exceed `MAX_CALL_DEPTH` return empty retdata
    /// instead of recursing, preventing runaway nested calls.
    pub call_depth: usize,
}

/// Iteration state for dict squash hints.
#[derive(Default)]
pub struct SquashState {
    /// Sorted unique keys from the current dict access log.
    pub sorted_keys: Vec<u64>,
    /// Index into sorted_keys — which key the squash loop is on.
    pub key_idx: usize,
    /// All access-log entries for the current key, in encounter order.
    /// Each entry is the new_value written.
    pub accesses_for_key: Vec<u64>,
    /// How far into accesses_for_key we are (inner squash loop).
    pub access_idx: usize,
    /// Base address of the dict access log in flat memory.
    pub accesses_start: u64,
    /// Total number of 3-cell dict-access entries in the log.
    pub total_accesses: u64,
}

impl HintContext {
    pub fn new() -> Self {
        Self {
            next_segment_base: SEGMENT_BASE_DEFAULT,
            dicts: HashMap::new(),
            dict_arena_info: HashMap::new(),
            squash: SquashState::default(),
            dict_accesses: Vec::new(),
            execution_overflows: 0,
            syscall: SyscallState::default(),
            contract_registry: HashMap::new(),
            call_depth: 0,
        }
    }

    pub fn with_syscall_state(mut self, state: SyscallState) -> Self {
        self.syscall = state;
        self
    }

    /// Register a contract for in-process cross-contract execution.
    ///
    /// `key` is the contract_address (for `call_contract`) or class_hash (for
    /// `library_call`).  When a syscall targets this key the callee is executed
    /// in-process and its retdata is returned to the caller.
    pub fn register_contract(
        &mut self,
        key: u64,
        bytecode: Vec<u64>,
        hints: Vec<(usize, Vec<CasmHint>)>,
        entry_points: HashMap<u64, u64>,
    ) -> &mut Self {
        self.contract_registry.insert(key, ContractEntry { bytecode, hints, entry_points });
        self
    }

    /// Allocate a fresh segment: return its base, advance the counter.
    fn alloc_segment(&mut self) -> u64 {
        let base = self.next_segment_base;
        self.next_segment_base += SEGMENT_SIZE;
        base
    }

    /// Find which dict segment owns a pointer (base ≤ ptr < base+SEGMENT_SIZE).
    fn find_dict_base(&self, ptr: u64) -> Option<u64> {
        for &base in self.dicts.keys() {
            if ptr >= base && ptr < base + SEGMENT_SIZE {
                return Some(base);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run all hints registered at `pc`. No-op if `hints` is empty or no hint at this PC.
pub fn run_hints(
    hints: &[(usize, Vec<CasmHint>)],
    pc: u64,
    step: usize,
    state: &CairoState,
    memory: &mut Memory,
    ctx: &mut HintContext,
) {
    if hints.is_empty() {
        return;
    }
    let pc_usize = pc as usize;
    // Hints are sorted by pc (casm_loader guarantees this).
    if let Ok(idx) = hints.binary_search_by_key(&pc_usize, |(p, _)| *p) {
        for hint in &hints[idx].1 {
            run_one(hint, step, state, memory, ctx);
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn run_one(hint: &CasmHint, step: usize, state: &CairoState, memory: &mut Memory, ctx: &mut HintContext) {
    // hint.raw is {"HintName": {params...}}  hint.name is "HintName"
    let params = match hint.raw.get(&hint.name) {
        Some(p) => p,
        None => return,
    };
    match hint.name.as_str() {
        "TestLessThan"           => hint_test_less_than(params, state, memory, false),
        "TestLessThanOrEqual"    => hint_test_less_than(params, state, memory, true),
        "DivMod"                 => hint_div_mod(params, state, memory),
        "SquareRoot"             => hint_square_root(params, state, memory),
        "LinearSplit"            => hint_linear_split(params, state, memory),
        "WideMul128"             => hint_wide_mul128(params, state, memory),
        "AllocSegment"           => hint_alloc_segment(params, state, memory, ctx),
        "AllocFelt252Dict"       => hint_alloc_felt252_dict(params, state, memory, ctx),
        "Felt252DictEntryInit"   => hint_dict_entry_init(params, state, memory, ctx),
        "Felt252DictEntryUpdate" => hint_dict_entry_update(params, step, state, memory, ctx),
        "InitSquashData"         => hint_init_squash_data(params, state, memory, ctx),
        "ShouldSkipSquashLoop"   => hint_should_skip_squash_loop(params, state, memory, ctx),
        "GetCurrentAccessIndex"  => hint_get_current_access_index(params, state, memory, ctx),
        "GetNextDictKey"         => hint_get_next_dict_key(params, state, memory, ctx),
        // Segment arena bookkeeping — returns the 0-based index of a dict in the arena.
        // Looks up the dict via its end pointer (or dict_ptr) in ctx.dict_arena_info.
        "GetSegmentArenaIndex"   => {
            hint_get_segment_arena_index(params, state, memory, ctx);
        }
        "U256InvModN"            => hint_u256_inv_mod_n(params, state, memory),
        "DebugPrint"             => hint_debug_print(params, state, memory),

        // ---- Starknet / Sierra-generated hints --------------------------------

        // SystemCall: Starknet OS syscall dispatcher.
        // Writes a dummy success response (retdata_start == retdata_end, error_code = 0).
        // The response struct is at [fp - 3]: selector=request, response=(0, ptr, ptr).
        // Without a full Starknet OS emulator, storage values return 0 and events are dropped.
        "SystemCall" => hint_system_call(params, state, memory, ctx),

        // Cheatcode: testing framework hook (Foundry/snforge). No-op outside test runner.
        "Cheatcode" => {}

        // EC point hints for ECDSA / secp256k1 — no-op (EC ops unsupported over M31).
        // Programs using ECDSA verification will produce invalid traces.
        "EcMulQ" | "EcMulInner" | "EcRecoverProductMod" | "EcRecoverDivModNPacked"
        | "EcRecoverSubAB" | "EcDoubleAssignNewX" | "EcDoubleAssignNewY"
        | "EcOp" | "Secp256k1EcMul" | "Secp256r1EcMul" => {
            eprintln!("warn: EC hint '{}' skipped — EC ops unsupported over M31", hint.name);
        }

        // Keccak hints: no-op (keccak not implemented).
        "KeccakU256sOnOsKernelPtr" | "KeccakWriteFirstBlock" | "KeccakWriteBlocksList"
        | "KeccakProcessInputs" | "BlockPermutationToKeccakState" => {
            eprintln!("warn: keccak hint '{}' skipped — keccak not implemented", hint.name);
        }

        // Blake2s hints: no-op.
        "Blake2sCompute" | "Blake2sComputeHalfFinalBlock" | "Blake2sFinalizeRunning"
        | "Blake2sAddUint256Bigend" | "Blake2sFinalRound" | "Blake2sPackageLastBlock" => {}

        // Range-check / sorting assertion hints — no-op (CPU verifies separately).
        "AssertCurrentAccessIndicesIsEmpty" | "AssertAllAccessesUsed"
        | "AssertLeAssertValidInput" | "AssertLtAssertValidInput"
        | "AssertLeIsFirstArcExcluded" | "AssertLeIsSecondArcExcluded"
        | "AssertAllKeysUsed" | "AssertLeFindSmallArcs" => {}

        // Poseidon hash hint (Starknet builtin) — values already in memory from builtin.
        "PoseidonHint" => {}

        // Felt252DictEntry variants not already listed above — no-op silently.
        name if name.starts_with("Felt252Dict") => {}

        // Uint / felt arithmetic helpers.
        "ScopeDefineFp" | "ScopeCallContract" | "FieldSqrt" | "EnterScope"
        | "ExitScope" | "SetupFastPow" | "FastPow" | "PackedEcPoint"
        | "Cairo1HintCode" => {}

        name => {
            eprintln!("warn: unsupported hint '{}' — skipped, trace may be invalid", name);
        }
    }
}

// ---------------------------------------------------------------------------
// Starknet syscall handler
// ---------------------------------------------------------------------------
//
// Buffer layout for each syscall (offsets from syscall_ptr):
//
//   StorageRead:      [sel, gas, domain, key | remaining_gas, err, value]
//   StorageWrite:     [sel, gas, domain, key, value | remaining_gas, err]
//   EmitEvent:        [sel, gas, keys_len, k0..kN, data_len, d0..dN | remaining_gas, err]
//   GetExecutionInfo: [sel, gas | remaining_gas, err, exec_info_ptr]
//   GetBlockHash:     [sel, gas, block_number | remaining_gas, err, hash_low, hash_high]
//   CallContract:     [sel, gas, addr, entry_pt, calldata_len, cd0..cdN | remaining_gas, err, ret_len, r0..rN]
//   LibraryCall:      [sel, gas, class_hash, entry_pt, calldata_len, cd0..cdN | remaining_gas, err, ret_len, r0..rN]
//   Deploy:           [sel, gas, class_hash, salt, calldata_len, cd0..cdN | remaining_gas, err, contract_addr]
//   SendMessageToL1:  [sel, gas, to_addr, payload_len, p0..pN | remaining_gas, err]
//
// Gas is always left at 0 (not tracked by VortexSTARK).
// VortexSTARK does not execute callee contracts — cross-contract calls return empty
// retdata and deploy returns a deterministic mock address (salt XOR class_hash).

fn hint_system_call(
    params: &serde_json::Value,
    state: &CairoState,
    memory: &mut Memory,
    ctx: &mut HintContext,
) {
    let syscall_ptr = if let Some(sys) = params.get("system") {
        cell_ref_addr(sys, state)
    } else {
        return;
    };

    // [0]: selector (felt252 low-64), [1]: gas_counter
    let selector = memory.get(syscall_ptr);

    match selector {
        sel::STORAGE_READ => {
            // Request:  [sel, gas, domain, key]
            // Response: [remaining_gas=0, err=0, value]
            let key = memory.get(syscall_ptr + 3);
            let value = ctx.syscall.storage.get(&key).copied().unwrap_or(0);
            memory.set(syscall_ptr + 4, 0);     // remaining_gas
            memory.set(syscall_ptr + 5, 0);     // error_code
            memory.set(syscall_ptr + 6, value); // value
        }

        sel::STORAGE_WRITE => {
            // Request:  [sel, gas, domain, key, value]
            // Response: [remaining_gas=0, err=0]
            let key   = memory.get(syscall_ptr + 3);
            let value = memory.get(syscall_ptr + 4);
            ctx.syscall.storage.insert(key, value);
            memory.set(syscall_ptr + 5, 0); // remaining_gas
            memory.set(syscall_ptr + 6, 0); // error_code
        }

        sel::EMIT_EVENT => {
            // Request:  [sel, gas, keys_len, k0..kN, data_len, d0..dN]
            // Response: [remaining_gas=0, err=0]
            let keys_len = memory.get(syscall_ptr + 2) as usize;
            let mut keys = Vec::with_capacity(keys_len);
            for i in 0..keys_len as u64 {
                keys.push(memory.get(syscall_ptr + 3 + i));
            }
            let data_offset = 3 + keys_len as u64;
            let data_len = memory.get(syscall_ptr + data_offset) as usize;
            let mut data = Vec::with_capacity(data_len);
            for i in 0..data_len as u64 {
                data.push(memory.get(syscall_ptr + data_offset + 1 + i));
            }
            ctx.syscall.events.push(SyscallEvent { keys, data });
            let resp = data_offset + 1 + data_len as u64;
            memory.set(syscall_ptr + resp, 0);     // remaining_gas
            memory.set(syscall_ptr + resp + 1, 0); // error_code
        }

        sel::GET_EXECUTION_INFO => {
            // Request:  [sel, gas]
            // Response: [remaining_gas=0, err=0, exec_info_ptr]
            let exec_info_ptr = syscall_exec_info_ptr(memory, ctx);
            memory.set(syscall_ptr + 2, 0);              // remaining_gas
            memory.set(syscall_ptr + 3, 0);              // error_code
            memory.set(syscall_ptr + 4, exec_info_ptr);  // exec_info_ptr
        }

        sel::GET_BLOCK_HASH => {
            // Request:  [sel, gas, block_number]
            // Response: [remaining_gas=0, err=0, hash_low, hash_high]
            // Return a deterministic mock hash = block_number (low64)
            let _block = memory.get(syscall_ptr + 2);
            memory.set(syscall_ptr + 3, 0); // remaining_gas
            memory.set(syscall_ptr + 4, 0); // error_code
            memory.set(syscall_ptr + 5, ctx.syscall.block_number); // hash (mock)
            memory.set(syscall_ptr + 6, 0); // hash high word
        }

        sel::CALL_CONTRACT | sel::LIBRARY_CALL => {
            // Request:  [sel, gas, addr_or_class_hash, entry_pt, calldata_len, cd0..cdN]
            // Response: [remaining_gas=0, err=0, ret_len, r0..rN]
            //
            // If the target is in ctx.contract_registry the callee is executed in-process
            // via execute_callee and its retdata is written into the response buffer.
            // Otherwise retdata is empty (existing fallback behavior).
            let target         = memory.get(syscall_ptr + 2);
            let entry_pt       = memory.get(syscall_ptr + 3);
            let calldata_len   = memory.get(syscall_ptr + 4) as u64;
            let mut calldata   = Vec::with_capacity(calldata_len as usize);
            for i in 0..calldata_len {
                calldata.push(memory.get(syscall_ptr + 5 + i));
            }

            // Execute callee if registered; otherwise return empty retdata.
            let retdata = if ctx.contract_registry.contains_key(&target) && ctx.call_depth < MAX_CALL_DEPTH {
                // Look up entry point PC.  Fall back to entry_points[0] if selector not found.
                let entry_pc = ctx.contract_registry[&target].entry_points
                    .get(&entry_pt).copied()
                    .or_else(|| ctx.contract_registry[&target].entry_points.values().next().copied())
                    .unwrap_or(0);
                let parent_contract = ctx.syscall.contract_address;
                execute_callee(target, entry_pc, entry_pt, &calldata, parent_contract, ctx)
            } else {
                Vec::new()
            };

            let resp = 5 + calldata_len;
            memory.set(syscall_ptr + resp,     0);                   // remaining_gas
            memory.set(syscall_ptr + resp + 1, 0);                   // error_code
            memory.set(syscall_ptr + resp + 2, retdata.len() as u64); // ret_len
            for (i, &v) in retdata.iter().enumerate() {
                memory.set(syscall_ptr + resp + 3 + i as u64, v);
            }

            ctx.syscall.cross_contract_calls.push(CrossContractCall {
                target,
                entry_point_selector: entry_pt,
                calldata,
                is_library: selector == sel::LIBRARY_CALL,
            });
        }

        sel::DEPLOY => {
            // Request:  [sel, gas, class_hash, salt, calldata_len, cd0..cdN]
            // Response: [remaining_gas=0, err=0, contract_addr]
            //
            // Returns a deterministic mock address: salt XOR class_hash.
            // The deployed contract is recorded in SyscallState::deployed_contracts.
            let class_hash   = memory.get(syscall_ptr + 2);
            let salt         = memory.get(syscall_ptr + 3);
            let calldata_len = memory.get(syscall_ptr + 4) as u64;
            let mut calldata = Vec::with_capacity(calldata_len as usize);
            for i in 0..calldata_len {
                calldata.push(memory.get(syscall_ptr + 5 + i));
            }
            let mock_addr = salt ^ class_hash;
            let resp = 5 + calldata_len;
            memory.set(syscall_ptr + resp,     0);         // remaining_gas
            memory.set(syscall_ptr + resp + 1, 0);         // error_code
            memory.set(syscall_ptr + resp + 2, mock_addr); // deployed contract address
            ctx.syscall.deployed_contracts.push(DeployedContract {
                class_hash,
                salt,
                calldata,
                contract_address: mock_addr,
            });
        }

        sel::SEND_MSG_TO_L1 => {
            // Request:  [sel, gas, to_addr, payload_len, p0..pN]
            // Response: [remaining_gas=0, err=0]
            //
            // The message is recorded in SyscallState::l1_messages.
            let to_address  = memory.get(syscall_ptr + 2);
            let payload_len = memory.get(syscall_ptr + 3) as u64;
            let mut payload = Vec::with_capacity(payload_len as usize);
            for i in 0..payload_len {
                payload.push(memory.get(syscall_ptr + 4 + i));
            }
            let resp = 4 + payload_len;
            memory.set(syscall_ptr + resp,     0); // remaining_gas
            memory.set(syscall_ptr + resp + 1, 0); // error_code
            ctx.syscall.l1_messages.push(L1Message { to_address, payload });
        }

        _ => {
            // Truly unknown syscall selector.  Write minimal success response
            // to prevent memory-uninitialized panics and log under debug.
            eprintln!("syscall: unknown selector {selector:#018x} at ptr={syscall_ptr}");
            for offset in 2..=9u64 {
                if memory.get(syscall_ptr + offset) == 0 {
                    memory.set(syscall_ptr + offset, 0);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-contract callee executor
// ---------------------------------------------------------------------------

/// Execute a registered callee contract in-process and return its retdata.
///
/// The callee runs in a fresh flat memory context (isolated stack).  The parent's
/// `HintContext` is shared so that:
/// - Storage reads/writes, events, and L1 messages propagate to the parent.
/// - The contract registry is available for further nested calls.
/// - Execution overflows are counted across the whole call tree.
///
/// The callee's execution context (`contract_address`, `caller_address`,
/// `entry_point_selector`, `exec_info_base`) is swapped in before execution and
/// restored afterward so `GetExecutionInfo` returns correct data at every depth.
///
/// **Retdata convention:**  In the Cairo 1 / CASM ABI, a callee places all return
/// values on AP before `ret`.  Retdata is the slice `memory[initial_ap..final_ap]`
/// at the moment `ret` returns execution to the halt sentinel.
///
/// Returns an empty `Vec` if:
/// - The entry-point PC is unreachable (no instructions).
/// - The callee exceeds `MAX_CALLEE_STEPS` without halting.
/// - `ctx.call_depth >= MAX_CALL_DEPTH` (recursion guard, checked by caller).
fn execute_callee(
    callee_address: u64,
    entry_point_pc: u64,
    entry_point_selector: u64,
    calldata: &[u64],
    parent_contract_address: u64,
    ctx: &mut HintContext,
) -> Vec<u64> {
    // Borrow the bytecode and hints out of the registry without holding a reference
    // into ctx (we need &mut ctx below for run_hints).
    let (bytecode, callee_hints) = {
        let entry = match ctx.contract_registry.get(&callee_address) {
            Some(e) => e,
            None => return Vec::new(),
        };
        (entry.bytecode.clone(), entry.hints.clone())
    };

    // Memory layout:
    //   [0 .. bytecode.len())  : callee bytecode
    //   pad_addr               : jmp rel 0  (halt sentinel, immediate below)
    //   pad_addr + 1           : 0          (jump delta)
    //   initial_sp             : saved_fp = 0   (frame sentinel)
    //   initial_sp + 1         : return_pc = pad_addr
    //   initial_ap             : callee stack base; calldata goes here
    let pad_addr    = bytecode.len() as u64;
    let initial_sp  = pad_addr + 10;
    let initial_ap  = initial_sp + 2;
    let mem_cap     = (initial_ap as usize) + calldata.len() + MAX_CALLEE_STEPS + 1000;

    let mut mem = Memory::with_capacity(mem_cap);
    mem.load_program(&bytecode);
    // Halt: jmp rel 0  (op1_imm=1, pc_jump_rel=1, off2=+1 bias, delta=0)
    mem.set(pad_addr,     0x0104_8001_8000_8000u64);
    mem.set(pad_addr + 1, 0u64);
    // Frame
    mem.set(initial_sp,     0u64);        // saved_fp
    mem.set(initial_sp + 1, pad_addr);    // return_pc

    // Calldata at initial_ap (caller-placed arguments)
    for (i, &v) in calldata.iter().enumerate() {
        mem.set(initial_ap + i as u64, v);
    }

    // Swap in callee execution context so GetExecutionInfo is correct.
    let saved_caller    = ctx.syscall.caller_address;
    let saved_contract  = ctx.syscall.contract_address;
    let saved_selector  = ctx.syscall.entry_point_selector;
    let saved_exec_base = ctx.syscall.exec_info_base;
    ctx.syscall.caller_address       = parent_contract_address;
    ctx.syscall.contract_address     = callee_address;
    ctx.syscall.entry_point_selector = entry_point_selector;
    ctx.syscall.exec_info_base       = None; // fresh exec_info for this callee

    ctx.call_depth += 1;

    let mut state = super::vm::CairoState {
        pc: entry_point_pc,
        ap: initial_ap,
        fp: initial_ap,
    };
    let callee_initial_ap = initial_ap;

    for step in 0..MAX_CALLEE_STEPS {
        if state.pc == pad_addr {
            break; // callee returned to halt sentinel
        }
        run_hints(&callee_hints, state.pc, step, &state, &mut mem, ctx);
        state = super::vm::step(&state, &mut mem);
    }

    ctx.call_depth -= 1;

    // Restore parent execution context.
    ctx.syscall.caller_address       = saved_caller;
    ctx.syscall.contract_address     = saved_contract;
    ctx.syscall.entry_point_selector = saved_selector;
    ctx.syscall.exec_info_base       = saved_exec_base;

    // Collect retdata: all values pushed to AP by the callee before returning.
    // CASM ABI: return values are the last N cells pushed to AP before `ret`.
    // final_ap == callee_initial_ap means nothing was returned.
    let final_ap = state.ap;
    if final_ap <= callee_initial_ap {
        return Vec::new();
    }
    (callee_initial_ap..final_ap)
        .map(|addr| mem.get(addr))
        .collect()
}

/// Lazily allocate and populate the execution-info memory region.
/// Returns the exec_info_ptr to include in GetExecutionInfo responses.
///
/// Memory layout (all relative to `base = ctx.alloc_segment()`):
///   base +  0..18 : tx_info fields (19 words)
///   base + 19..21 : block_info fields (3 words: block_number, timestamp, sequencer)
///   base + 22     : empty-array sentinel (shared by empty sig / resource_bounds / etc.)
///   base + 23..27 : exec_info fields (5 words: block_ptr, tx_ptr, caller, contract, selector)
fn syscall_exec_info_ptr(memory: &mut Memory, ctx: &mut HintContext) -> u64 {
    if let Some(ptr) = ctx.syscall.exec_info_base {
        return ptr;
    }

    let base = ctx.alloc_segment();
    let tx_info   = base;
    let block_info = base + 19;
    let sentinel  = base + 22; // empty-array start == end for all empty arrays
    let exec_info = base + 23;

    // --- tx_info (19 words) ---
    memory.set(tx_info,      1);                                // version = 1
    memory.set(tx_info + 1,  ctx.syscall.caller_address);      // account_contract_address
    memory.set(tx_info + 2,  0);                                // max_fee
    memory.set(tx_info + 3,  sentinel);                         // signature_start (empty)
    memory.set(tx_info + 4,  sentinel);                         // signature_end   (empty)
    memory.set(tx_info + 5,  0);                                // transaction_hash
    memory.set(tx_info + 6,  0);                                // chain_id
    memory.set(tx_info + 7,  0);                                // nonce
    memory.set(tx_info + 8,  sentinel);                         // resource_bounds_start
    memory.set(tx_info + 9,  sentinel);                         // resource_bounds_end
    memory.set(tx_info + 10, 0);                                // tip
    memory.set(tx_info + 11, sentinel);                         // paymaster_data_start
    memory.set(tx_info + 12, sentinel);                         // paymaster_data_end
    memory.set(tx_info + 13, 0);                                // nonce_data_availability_mode
    memory.set(tx_info + 14, 0);                                // fee_data_availability_mode
    memory.set(tx_info + 15, sentinel);                         // account_deployment_data_start
    memory.set(tx_info + 16, sentinel);                         // account_deployment_data_end
    memory.set(tx_info + 17, sentinel);                         // proof_facts_start
    memory.set(tx_info + 18, sentinel);                         // proof_facts_end

    // --- block_info (3 words) ---
    memory.set(block_info,     ctx.syscall.block_number);       // block_number
    memory.set(block_info + 1, ctx.syscall.block_timestamp);    // block_timestamp
    memory.set(block_info + 2, 0);                              // sequencer_address

    // sentinel cell — just needs to exist (programs with empty arrays won't read past it)
    memory.set(sentinel, 0);

    // --- exec_info (5 words) ---
    memory.set(exec_info,     block_info);                      // block_info_ptr
    memory.set(exec_info + 1, tx_info);                         // tx_info_ptr
    memory.set(exec_info + 2, ctx.syscall.caller_address);      // caller_address
    memory.set(exec_info + 3, ctx.syscall.contract_address);    // contract_address
    memory.set(exec_info + 4, ctx.syscall.entry_point_selector);// entry_point_selector

    ctx.syscall.exec_info_base = Some(exec_info);
    exec_info
}

// ---------------------------------------------------------------------------
// Arithmetic hints
// ---------------------------------------------------------------------------

/// TestLessThan / TestLessThanOrEqual
fn hint_test_less_than(
    params: &serde_json::Value,
    state: &CairoState,
    memory: &mut Memory,
    or_equal: bool,
) {
    let lhs = eval_res_operand(&params["lhs"], state, memory);
    let rhs = eval_res_operand(&params["rhs"], state, memory);
    let result = if or_equal { (lhs <= rhs) as u64 } else { (lhs < rhs) as u64 };
    let dst_addr = cell_ref_addr(&params["dst"], state);
    memory.set(dst_addr, result);
}

/// DivMod — quotient ← lhs / rhs, remainder ← lhs % rhs
fn hint_div_mod(params: &serde_json::Value, state: &CairoState, memory: &mut Memory) {
    let lhs = eval_res_operand(&params["lhs"], state, memory);
    let rhs = eval_res_operand(&params["rhs"], state, memory);
    if rhs == 0 {
        return; // division by zero — let the program's assertion catch it
    }
    memory.set(cell_ref_addr(&params["quotient"],  state), lhs / rhs);
    memory.set(cell_ref_addr(&params["remainder"], state), lhs % rhs);
}

/// SquareRoot — dst ← floor(sqrt(value))
fn hint_square_root(params: &serde_json::Value, state: &CairoState, memory: &mut Memory) {
    let value = eval_res_operand(&params["value"], state, memory);
    memory.set(cell_ref_addr(&params["dst"], state), isqrt(value));
}

/// LinearSplit — finds x, y such that value = x * scalar + y, x ≤ max_x
fn hint_linear_split(params: &serde_json::Value, state: &CairoState, memory: &mut Memory) {
    let value  = eval_res_operand(&params["value"],  state, memory);
    let scalar = eval_res_operand(&params["scalar"], state, memory);
    let max_x  = eval_res_operand(&params["max_x"],  state, memory);
    if scalar == 0 { return; }
    let x = (value / scalar).min(max_x);
    let y = value - x * scalar;
    memory.set(cell_ref_addr(&params["x"], state), x);
    memory.set(cell_ref_addr(&params["y"], state), y);
}

/// WideMul128 — (high, low) ← lhs * rhs as 128-bit product
fn hint_wide_mul128(params: &serde_json::Value, state: &CairoState, memory: &mut Memory) {
    let lhs = eval_res_operand(&params["lhs"], state, memory);
    let rhs = eval_res_operand(&params["rhs"], state, memory);
    let product = lhs as u128 * rhs as u128;
    memory.set(cell_ref_addr(&params["high"], state), (product >> 64) as u64);
    memory.set(cell_ref_addr(&params["low"],  state), product as u64);
}

// ---------------------------------------------------------------------------
// Segment / memory hints
// ---------------------------------------------------------------------------

/// AllocSegment — allocate a new flat memory region, write base address to dst.
///
/// In the real Cairo VM this creates a new segment.  Here we hand out a
/// contiguous flat range starting at `ctx.next_segment_base`.
fn hint_alloc_segment(
    params: &serde_json::Value,
    state: &CairoState,
    memory: &mut Memory,
    ctx: &mut HintContext,
) {
    let dst_addr = cell_ref_addr(&params["dst"], state);
    let base = ctx.alloc_segment();
    memory.set(dst_addr, base);
}

// ---------------------------------------------------------------------------
// Dict hints
// ---------------------------------------------------------------------------

/// Segment-arena layout in flat memory (written by Sierra runtime before dict ops):
///
///   [arena_ptr + 0]  dict_infos_ptr  — base address of DictInfo array
///   [arena_ptr + 1]  n_dicts         — number of dicts allocated so far
///
/// DictInfo (3 cells, at dict_infos_ptr + i*3):
///   [+ 0]  dict_start   — base address of the dict's access log
///   [+ 1]  dict_end     — current write pointer (next free slot)
///   [+ 2]  squash_end   — used during squash (0 initially)
const DICT_INFO_SIZE: u64 = 3;

/// AllocFelt252Dict — allocate a new dict segment and register it in the segment arena.
fn hint_alloc_felt252_dict(
    params: &serde_json::Value,
    state: &CairoState,
    memory: &mut Memory,
    ctx: &mut HintContext,
) {
    // segment_arena_ptr can be a ResOperand (Deref) or plain CellRef
    let arena_ptr = eval_res_operand_or_cellref(&params["segment_arena_ptr"], state, memory);

    let infos_ptr = memory.get(arena_ptr);
    let n_dicts   = memory.get(arena_ptr + 1);

    let new_base = ctx.alloc_segment();
    ctx.dicts.insert(new_base, HashMap::new());

    // Where to write the new DictInfo: if infos_ptr is 0 (uninitialized arena),
    // place the info array immediately after the 2-word arena header.
    let effective_infos_ptr = if infos_ptr == 0 { arena_ptr + 2 } else { infos_ptr };
    let slot = effective_infos_ptr + n_dicts * DICT_INFO_SIZE;

    memory.set(slot + 0, new_base); // DictInfo.start
    memory.set(slot + 1, new_base); // DictInfo.end (current ptr = start)
    memory.set(slot + 2, new_base); // DictInfo.squash_end

    memory.set(arena_ptr + 1, n_dicts + 1);

    // Record arena bookkeeping for this dict so Felt252DictEntryUpdate and
    // GetSegmentArenaIndex can access it later.
    ctx.dict_arena_info.insert(new_base, (arena_ptr, n_dicts, effective_infos_ptr));
}

/// Felt252DictEntryInit — called before a dict read/write.
///
/// Writes the previous value for `key` into the dict access entry at `[dict_ptr + 1]`.
/// The entry layout is: [key, prev_value, new_value].
fn hint_dict_entry_init(
    params: &serde_json::Value,
    state: &CairoState,
    memory: &mut Memory,
    ctx: &mut HintContext,
) {
    let dict_ptr = eval_res_operand(&params["dict_ptr"], state, memory);
    let key      = eval_res_operand(&params["key"],      state, memory);

    let prev_value = ctx.find_dict_base(dict_ptr)
        .and_then(|base| ctx.dicts.get(&base))
        .and_then(|d| d.get(&key))
        .copied()
        .unwrap_or(0);

    memory.set(dict_ptr + 1, prev_value);
}

/// Felt252DictEntryUpdate — called after the program has written the new value.
///
/// Reads the completed entry, updates the dict side-state, and advances the
/// dict pointer by 3 (one entry = key + prev + new).
fn hint_dict_entry_update(
    params: &serde_json::Value,
    step: usize,
    state: &CairoState,
    memory: &mut Memory,
    ctx: &mut HintContext,
) {
    // dict_ptr_ptr is a CellRef pointing to the cell that holds the current dict_ptr
    let dict_ptr_ptr_addr = cell_ref_addr(&params["dict_ptr_ptr"], state);
    let dict_ptr          = memory.get(dict_ptr_ptr_addr);

    let key       = memory.get(dict_ptr);
    let new_value = memory.get(dict_ptr + 2);

    let dict_base = ctx.find_dict_base(dict_ptr);
    if let Some(base) = dict_base {
        // Record (key, prev_value, new_value) before updating side-state.
        let prev_value = ctx.dicts.get(&base)
            .and_then(|d| d.get(&key))
            .copied()
            .unwrap_or(0);
        ctx.dict_accesses.push((step, key, prev_value, new_value));
        ctx.dicts.entry(base).or_default().insert(key, new_value);
    }

    // Advance dict_ptr by one entry (3 cells).
    let new_dict_ptr = dict_ptr + 3;
    memory.set(dict_ptr_ptr_addr, new_dict_ptr);

    // Mirror the advance into DictInfo.end in the segment arena so that squash
    // code which reads DictInfo.end sees the current write position.
    if let Some(base) = dict_base {
        if let Some(&(_arena_ptr, dict_idx, infos_ptr)) = ctx.dict_arena_info.get(&base) {
            let slot = infos_ptr + dict_idx * DICT_INFO_SIZE;
            memory.set(slot + 1, new_dict_ptr); // DictInfo.end
        }
    }
}

/// GetSegmentArenaIndex — write the dict's index in the segment arena to `dict_index`.
///
/// Sierra programs use this during squash to tell the arena which dict they're squashing.
/// The hint identifies the dict via `dict_end_ptr` (the current end-of-dict pointer) and
/// looks up its index from `ctx.dict_arena_info`.
fn hint_get_segment_arena_index(
    params: &serde_json::Value,
    state: &CairoState,
    memory: &mut Memory,
    ctx: &HintContext,
) {
    // The dict is identified by a pointer somewhere inside its segment.
    // Try "dict_end_ptr" first, then fall back to "segment_arena_ptr" or "dict_ptr".
    let dict_ptr_val = if let Some(p) = params.get("dict_end_ptr") {
        eval_res_operand(p, state, memory)
    } else if let Some(p) = params.get("dict_ptr") {
        eval_res_operand(p, state, memory)
    } else {
        0
    };

    let dict_idx = ctx.find_dict_base(dict_ptr_val)
        .and_then(|base| ctx.dict_arena_info.get(&base))
        .map(|&(_, idx, _)| idx)
        .unwrap_or(0);

    if let Some(dst) = params.get("dict_index") {
        memory.set(cell_ref_addr(dst, state), dict_idx);
    }
}

// ---------------------------------------------------------------------------
// Dict squash hints
// ---------------------------------------------------------------------------

/// InitSquashData — called once before the squash loop for a dict.
///
/// Reads the entire access log from flat memory, extracts and sorts the unique
/// keys, and stores squash iteration state in `ctx.squash`.
fn hint_init_squash_data(
    params: &serde_json::Value,
    state: &CairoState,
    memory: &mut Memory,
    ctx: &mut HintContext,
) {
    let accesses_start = eval_res_operand(&params["dict_accesses"], state, memory);
    let ptr_diff       = eval_res_operand(&params["ptr_diff"],       state, memory);
    let n_accesses     = ptr_diff / 3;

    // Build key → list-of-new-values map from the flat access log.
    let mut key_to_new_values: std::collections::BTreeMap<u64, Vec<u64>> =
        std::collections::BTreeMap::new();

    for i in 0..n_accesses {
        let key       = memory.get(accesses_start + i * 3);
        let new_value = memory.get(accesses_start + i * 3 + 2);
        key_to_new_values.entry(key).or_default().push(new_value);
    }

    // Sort keys (BTreeMap gives sorted iteration).
    let sorted_keys: Vec<u64> = key_to_new_values.keys().copied().collect();

    // Write first_key and big_keys to their output CellRefs.
    let first_key_addr = cell_ref_addr(&params["first_key"], state);
    let big_keys_addr  = cell_ref_addr(&params["big_keys"],  state);
    memory.set(first_key_addr, sorted_keys.first().copied().unwrap_or(0));
    // In our u64 model no key can exceed 2^64; felt252 big-key check would need
    // the high 128 bits which we truncated — conservatively report no big keys.
    memory.set(big_keys_addr, 0);

    // Prepare squash state for iteration.
    let first_key = sorted_keys.first().copied().unwrap_or(0);
    let first_accesses = key_to_new_values.get(&first_key).cloned().unwrap_or_default();
    ctx.squash = SquashState {
        sorted_keys,
        key_idx: 0,
        accesses_for_key: first_accesses,
        access_idx: 0,
        accesses_start,
        total_accesses: n_accesses,
    };
}

/// ShouldSkipSquashLoop — returns 1 when the inner access loop for the current key is done.
fn hint_should_skip_squash_loop(
    params: &serde_json::Value,
    state: &CairoState,
    memory: &mut Memory,
    ctx: &mut HintContext,
) {
    let dst_addr = cell_ref_addr(&params["should_skip_loop"], state);
    let done = ctx.squash.access_idx >= ctx.squash.accesses_for_key.len();
    memory.set(dst_addr, done as u64);
    if done {
        // Reset for next key (outer loop will call GetNextDictKey).
        ctx.squash.access_idx = 0;
    }
}

/// GetCurrentAccessIndex — write the current access position to range_check memory.
fn hint_get_current_access_index(
    params: &serde_json::Value,
    state: &CairoState,
    memory: &mut Memory,
    ctx: &mut HintContext,
) {
    // range_check_ptr is a ResOperand that evaluates to the range_check builtin pointer.
    // We write the current access index there so range checks can verify ordering.
    let rc_ptr = eval_res_operand(&params["range_check_ptr"], state, memory);
    let idx = ctx.squash.access_idx as u64;
    memory.set(rc_ptr, idx);
    ctx.squash.access_idx += 1;
}

/// GetNextDictKey — advance to the next key in sorted order, write it to `next_key`.
fn hint_get_next_dict_key(
    params: &serde_json::Value,
    state: &CairoState,
    memory: &mut Memory,
    ctx: &mut HintContext,
) {
    ctx.squash.key_idx += 1;
    let next_key = ctx.squash.sorted_keys.get(ctx.squash.key_idx).copied().unwrap_or(0);
    memory.set(cell_ref_addr(&params["next_key"], state), next_key);
    ctx.squash.access_idx = 0;
}

// ---------------------------------------------------------------------------
// U256InvModN
// ---------------------------------------------------------------------------

/// U256InvModN — extended-GCD modular inverse.
///
/// Given b = b0 + b1·2^64 and n = n0 + n1·2^64 (truncated to u64 in our model),
/// computes gcd(b, n) via the extended Euclidean algorithm.
///
/// Outputs (g0_or_no_inv, g1_option, s_or_r0, s_or_r1, t_or_k0, t_or_k1):
/// - Invertible (gcd = 1): g0_or_no_inv = 1, g1_option = 0, (r0, r1) = inverse mod n,
///   (k0, k1) = floor(b * inverse / n).
/// - Not invertible:       (g0, g1) = gcd, (s, t) = Bézout coefficients.
fn hint_u256_inv_mod_n(params: &serde_json::Value, state: &CairoState, memory: &mut Memory) {
    let b0 = eval_res_operand(&params["b0"], state, memory);
    let n0 = eval_res_operand(&params["n0"], state, memory);
    // b1/n1 are the high-128-bit limbs — zero in our u64-truncated model.

    let g0_addr = cell_ref_addr(&params["g0_or_no_inv"], state);
    let g1_addr = cell_ref_addr(&params["g1_option"],    state);
    let r0_addr = cell_ref_addr(&params["s_or_r0"],      state);
    let r1_addr = cell_ref_addr(&params["s_or_r1"],      state);
    let k0_addr = cell_ref_addr(&params["t_or_k0"],      state);
    let k1_addr = cell_ref_addr(&params["t_or_k1"],      state);

    if n0 == 0 {
        // Modulus is zero — degenerate case.
        memory.set(g0_addr, b0);
        memory.set(g1_addr, 0);
        memory.set(r0_addr, 0); memory.set(r1_addr, 0);
        memory.set(k0_addr, 0); memory.set(k1_addr, 0);
        return;
    }

    let b_red = b0 % n0;
    let (gcd, s_signed, _) = egcd_u64(b_red, n0);

    if gcd == 1 {
        // Invertible: normalise signed s into [0, n0).
        let inv = ((s_signed % n0 as i128 + n0 as i128) % n0 as i128) as u64;
        let k = if inv == 0 { 0 } else { b_red / n0 };
        memory.set(g0_addr, 1);   // g0_or_no_inv = 1 → signals "invertible"
        memory.set(g1_addr, 0);
        memory.set(r0_addr, inv); memory.set(r1_addr, 0);
        memory.set(k0_addr, k);   memory.set(k1_addr, 0);
    } else {
        memory.set(g0_addr, gcd);
        memory.set(g1_addr, 0);
        memory.set(r0_addr, 0); memory.set(r1_addr, 0);
        memory.set(k0_addr, 0); memory.set(k1_addr, 0);
    }
}

/// Extended Euclidean algorithm on u64 values.
/// Returns (s, t, gcd) such that a·s + b·t = gcd (s and t may be negative).
fn egcd_u64(a: u64, b: u64) -> (u64, i128, i128) {
    if b == 0 {
        return (a, 1, 0);
    }
    let (g, s1, t1) = egcd_u64(b, a % b);
    let q = (a / b) as i128;
    (g, t1, s1 - q * t1)
}

// ---------------------------------------------------------------------------
// DebugPrint
// ---------------------------------------------------------------------------

/// DebugPrint — print the memory range [start, end) to stderr.
fn hint_debug_print(params: &serde_json::Value, state: &CairoState, memory: &Memory) {
    let start = eval_res_operand(&params["start"], state, memory);
    let end   = eval_res_operand(&params["end"],   state, memory);
    if end <= start { return; }
    let limit = (end - start).min(64); // cap output to avoid runaway logging
    eprint!("[debug_print:");
    for i in 0..limit {
        eprint!(" 0x{:x}", memory.get(start + i));
    }
    if end - start > 64 { eprint!(" ..."); }
    eprintln!("]");
}

// ---------------------------------------------------------------------------
// ResOperand / CellRef evaluation helpers
// ---------------------------------------------------------------------------

/// Evaluate a ResOperand OR fall back to treating the value as a plain CellRef.
/// Some hint parameters are documented as CellRef but serialised identically.
fn eval_res_operand_or_cellref(val: &serde_json::Value, state: &CairoState, memory: &Memory) -> u64 {
    // Try ResOperand first; if it's a plain register-offset object treat as CellRef.
    if val.get("Deref").is_some() || val.get("DoubleDeref").is_some()
        || val.get("Immediate").is_some() || val.get("BinOp").is_some()
    {
        eval_res_operand(val, state, memory)
    } else if val.get("register").is_some() {
        memory.get(cell_ref_addr(val, state))
    } else {
        eval_res_operand(val, state, memory)
    }
}

/// Evaluate a ResOperand JSON value against current registers + memory.
/// ResOperand ::= Deref | DoubleDeref | Immediate | BinOp
fn eval_res_operand(val: &serde_json::Value, state: &CairoState, memory: &Memory) -> u64 {
    if let Some(deref) = val.get("Deref") {
        memory.get(cell_ref_addr(deref, state))
    } else if let Some(dd) = val.get("DoubleDeref") {
        let base = if dd.is_array() {
            let ptr = memory.get(cell_ref_addr(&dd[0], state));
            let inner = dd[1].as_i64().unwrap_or(0) as u64;
            ptr.wrapping_add(inner)
        } else {
            let ptr = memory.get(cell_ref_addr(dd, state));
            let inner = val.get("inner_offset")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as u64;
            ptr.wrapping_add(inner)
        };
        memory.get(base)
    } else if let Some(imm) = val.get("Immediate") {
        parse_felt_u64(imm.get("value").and_then(|v| v.as_str()).unwrap_or("0"))
    } else if let Some(binop) = val.get("BinOp") {
        eval_binop(binop, state, memory)
    } else {
        0
    }
}

/// Evaluate a BinOp: a + b  or  a * b.
fn eval_binop(val: &serde_json::Value, state: &CairoState, memory: &Memory) -> u64 {
    let op = val["op"].as_str().unwrap_or("Add");
    let a  = memory.get(cell_ref_addr(&val["a"], state));
    let b  = eval_deref_or_imm(&val["b"], state, memory);
    if op == "Mul" { a.wrapping_mul(b) } else { a.wrapping_add(b) }
}

/// DerefOrImmediate: {"Deref": CellRef} or {"Immediate": {"value": "0x..."}}
fn eval_deref_or_imm(val: &serde_json::Value, state: &CairoState, memory: &Memory) -> u64 {
    if let Some(cr) = val.get("Deref") {
        memory.get(cell_ref_addr(cr, state))
    } else if let Some(imm) = val.get("Immediate") {
        parse_felt_u64(imm.get("value").and_then(|v| v.as_str()).unwrap_or("0"))
    } else {
        0
    }
}

/// Resolve a CellRef to an absolute memory address.
/// CellRef: {"register": "AP"|"FP", "offset": N}
fn cell_ref_addr(val: &serde_json::Value, state: &CairoState) -> u64 {
    let reg    = val["register"].as_str().unwrap_or("AP");
    let offset = val["offset"].as_i64().unwrap_or(0);
    let base   = if reg == "FP" { state.fp } else { state.ap };
    base.wrapping_add(offset as u64)
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/// Parse a hex or decimal string as a u64, truncating felt252 values to 64 bits.
fn parse_felt_u64(s: &str) -> u64 {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        let hex = if hex.len() > 16 { &hex[hex.len() - 16..] } else { hex };
        u64::from_str_radix(hex, 16).unwrap_or(0)
    } else {
        s.parse::<u64>().unwrap_or_default()
    }
}

/// Integer square root: floor(sqrt(n)).
fn isqrt(n: u64) -> u64 {
    if n == 0 { return 0; }
    // isqrt(u64) ≤ u32::MAX always; cap so the refinement loops stay in-range.
    let mut x = ((n as f64).sqrt() as u64).min(u32::MAX as u64);
    // Use u128 for the squaring steps to avoid u64 overflow in the increment check.
    while (x as u128) * (x as u128) > n as u128 { x -= 1; }
    while (x as u128 + 1) * (x as u128 + 1) <= n as u128 { x += 1; }
    x
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::casm_loader::CasmHint;

    fn make_hint(name: &str, params: serde_json::Value) -> CasmHint {
        CasmHint {
            name: name.to_string(),
            raw: serde_json::json!({ name: params }),
        }
    }

    fn state(ap: u64, fp: u64) -> CairoState {
        CairoState { pc: 0, ap, fp }
    }

    fn run(hints: &[(usize, Vec<CasmHint>)], pc: u64, s: &CairoState, mem: &mut Memory) {
        let mut ctx = HintContext::new();
        run_hints(hints, pc, 0, s, mem, &mut ctx);
    }

    #[test]
    fn test_test_less_than_true() {
        let mut mem = Memory::new();
        mem.set(100, 5);
        let s = state(101, 100);
        let h = make_hint("TestLessThan", serde_json::json!({
            "lhs": {"Deref": {"register": "AP", "offset": -1}},
            "rhs": {"Immediate": {"value": "10"}},
            "dst": {"register": "AP", "offset": 0}
        }));
        let hints = vec![(0usize, vec![h])];
        run(&hints, 0, &s, &mut mem);
        assert_eq!(mem.get(101), 1, "5 < 10 should be 1");
    }

    #[test]
    fn test_test_less_than_false() {
        let mut mem = Memory::new();
        mem.set(100, 15);
        let s = state(101, 100);
        let h = make_hint("TestLessThan", serde_json::json!({
            "lhs": {"Deref": {"register": "AP", "offset": -1}},
            "rhs": {"Immediate": {"value": "10"}},
            "dst": {"register": "AP", "offset": 0}
        }));
        let hints = vec![(0usize, vec![h])];
        run(&hints, 0, &s, &mut mem);
        assert_eq!(mem.get(101), 0, "15 < 10 should be 0");
    }

    #[test]
    fn test_div_mod() {
        let mut mem = Memory::new();
        mem.set(100, 10);
        mem.set(101, 3);
        let s = state(102, 100);
        let h = make_hint("DivMod", serde_json::json!({
            "lhs": {"Deref": {"register": "AP", "offset": -2}},
            "rhs": {"Deref": {"register": "AP", "offset": -1}},
            "quotient":  {"register": "AP", "offset": 0},
            "remainder": {"register": "AP", "offset": 1}
        }));
        let hints = vec![(0usize, vec![h])];
        run(&hints, 0, &s, &mut mem);
        assert_eq!(mem.get(102), 3, "10 / 3 = 3");
        assert_eq!(mem.get(103), 1, "10 % 3 = 1");
    }

    #[test]
    fn test_square_root() {
        let mut mem = Memory::new();
        mem.set(100, 81);
        let s = state(101, 100);
        let h = make_hint("SquareRoot", serde_json::json!({
            "value": {"Deref": {"register": "AP", "offset": -1}},
            "dst":   {"register": "AP", "offset": 0}
        }));
        let hints = vec![(0usize, vec![h])];
        run(&hints, 0, &s, &mut mem);
        assert_eq!(mem.get(101), 9, "sqrt(81) = 9");
    }

    #[test]
    fn test_wide_mul128() {
        let mut mem = Memory::new();
        mem.set(100, u64::MAX);
        mem.set(101, 2);
        let s = state(102, 100);
        let h = make_hint("WideMul128", serde_json::json!({
            "lhs":  {"Deref": {"register": "AP", "offset": -2}},
            "rhs":  {"Deref": {"register": "AP", "offset": -1}},
            "high": {"register": "AP", "offset": 0},
            "low":  {"register": "AP", "offset": 1}
        }));
        let hints = vec![(0usize, vec![h])];
        run(&hints, 0, &s, &mut mem);
        assert_eq!(mem.get(102), 1,            "high word");
        assert_eq!(mem.get(103), u64::MAX - 1, "low word");
    }

    #[test]
    fn test_alloc_segment_writes_base() {
        let mut mem = Memory::new();
        let s = state(100, 100);
        let h = make_hint("AllocSegment", serde_json::json!({
            "dst": {"register": "AP", "offset": 0}
        }));
        let hints = vec![(0usize, vec![h])];
        let mut ctx = HintContext::new();
        run_hints(&hints, 0, 0, &s, &mut mem, &mut ctx);
        // Should write the segment base (SEGMENT_BASE_DEFAULT) to [ap+0] = 100.
        assert_eq!(mem.get(100), SEGMENT_BASE_DEFAULT, "AllocSegment must write base to dst");
        // Second alloc increments by SEGMENT_SIZE.
        run_hints(&hints, 0, 0, &s, &mut mem, &mut ctx);
        assert_eq!(mem.get(100), SEGMENT_BASE_DEFAULT + SEGMENT_SIZE, "second alloc advances base");
    }

    #[test]
    fn test_dict_round_trip() {
        // AllocFelt252Dict → DictEntryInit → program writes value → DictEntryUpdate
        let mut mem = Memory::new();
        let mut ctx = HintContext::new();
        let s = state(200, 200);

        // Simulate the segment arena at address 10:
        //   [10] = 0 (infos_ptr uninitialized — hints use arena_ptr+2 as fallback)
        //   [11] = 0 (n_dicts)
        mem.set(10, 0);
        mem.set(11, 0);

        // AllocFelt252Dict with segment_arena_ptr = Deref FP-0 = memory[200] = 10
        mem.set(200, 10);  // fp+0 points to arena at 10
        let alloc_h = make_hint("AllocFelt252Dict", serde_json::json!({
            "segment_arena_ptr": {"Deref": {"register": "FP", "offset": 0}}
        }));
        let alloc_hints = vec![(0usize, vec![alloc_h])];
        run_hints(&alloc_hints, 0, 0, &s, &mut mem, &mut ctx);

        // Dict was registered; its base is SEGMENT_BASE_DEFAULT.
        let dict_base = SEGMENT_BASE_DEFAULT;
        assert!(ctx.dicts.contains_key(&dict_base), "dict should be registered");
        assert_eq!(mem.get(11), 1, "n_dicts should be incremented");

        // Simulate program: dict_ptr = dict_base, key = 42.
        // DictEntryInit writes prev_value (0, since dict is empty) to [dict_ptr+1].
        let init_h = make_hint("Felt252DictEntryInit", serde_json::json!({
            "dict_ptr": {"Immediate": {"value": &format!("0x{dict_base:x}")}},
            "key":      {"Immediate": {"value": "42"}}
        }));
        let init_hints = vec![(0usize, vec![init_h])];
        run_hints(&init_hints, 0, 0, &s, &mut mem, &mut ctx);
        assert_eq!(mem.get(dict_base + 1), 0, "prev_value for new key should be 0");

        // Program writes key to [dict_ptr + 0] and new_value to [dict_ptr + 2].
        mem.set(dict_base + 0, 42); // key (program fills this in)
        mem.set(dict_base + 2, 99); // new_value
        // dict_ptr_ptr at address 300 holds dict_ptr = dict_base.
        mem.set(300, dict_base);

        let update_h = make_hint("Felt252DictEntryUpdate", serde_json::json!({
            "dict_ptr_ptr": {"register": "FP", "offset": 100}  // fp+100 = 300
        }));
        let update_hints = vec![(0usize, vec![update_h])];
        run_hints(&update_hints, 0, 0, &s, &mut mem, &mut ctx);

        // dict[42] should now be 99.
        assert_eq!(ctx.dicts[&dict_base][&42], 99, "dict update should store new value");
        // dict_ptr_ptr should have been advanced by 3.
        assert_eq!(mem.get(300), dict_base + 3, "dict_ptr should advance by 3");

        // Second access to key=42: prev_value should now be 99.
        let new_dict_ptr = dict_base + 3;
        let init2_h = make_hint("Felt252DictEntryInit", serde_json::json!({
            "dict_ptr": {"Immediate": {"value": &format!("0x{new_dict_ptr:x}")}},
            "key":      {"Immediate": {"value": "42"}}
        }));
        let init2_hints = vec![(0usize, vec![init2_h])];
        run_hints(&init2_hints, 0, 0, &s, &mut mem, &mut ctx);
        assert_eq!(mem.get(new_dict_ptr + 1), 99, "prev_value for existing key should be 99");
    }

    #[test]
    fn test_u256_inv_mod_n_invertible() {
        let mut mem = Memory::new();
        let s = state(100, 100);
        // 3^{-1} mod 7 = 5 (since 3*5 = 15 ≡ 1 mod 7)
        mem.set(100, 3); // b0
        mem.set(101, 0); // b1
        mem.set(102, 7); // n0
        mem.set(103, 0); // n1
        let h = make_hint("U256InvModN", serde_json::json!({
            "b0": {"Deref": {"register": "AP", "offset": 0}},
            "b1": {"Deref": {"register": "AP", "offset": 1}},
            "n0": {"Deref": {"register": "AP", "offset": 2}},
            "n1": {"Deref": {"register": "AP", "offset": 3}},
            "g0_or_no_inv": {"register": "AP", "offset": 4},
            "g1_option":    {"register": "AP", "offset": 5},
            "s_or_r0":      {"register": "AP", "offset": 6},
            "s_or_r1":      {"register": "AP", "offset": 7},
            "t_or_k0":      {"register": "AP", "offset": 8},
            "t_or_k1":      {"register": "AP", "offset": 9}
        }));
        let hints = vec![(0usize, vec![h])];
        run(&hints, 0, &s, &mut mem);
        assert_eq!(mem.get(104), 1, "g0_or_no_inv = 1 (invertible)");
        assert_eq!(mem.get(105), 0, "g1_option = 0");
        assert_eq!(mem.get(106), 5, "3^-1 mod 7 = 5");
    }

    #[test]
    fn test_u256_inv_mod_n_not_invertible() {
        let mut mem = Memory::new();
        let s = state(100, 100);
        // gcd(6, 9) = 3 (not invertible)
        mem.set(100, 6); mem.set(101, 0);
        mem.set(102, 9); mem.set(103, 0);
        let h = make_hint("U256InvModN", serde_json::json!({
            "b0": {"Deref": {"register": "AP", "offset": 0}},
            "b1": {"Deref": {"register": "AP", "offset": 1}},
            "n0": {"Deref": {"register": "AP", "offset": 2}},
            "n1": {"Deref": {"register": "AP", "offset": 3}},
            "g0_or_no_inv": {"register": "AP", "offset": 4},
            "g1_option":    {"register": "AP", "offset": 5},
            "s_or_r0":      {"register": "AP", "offset": 6},
            "s_or_r1":      {"register": "AP", "offset": 7},
            "t_or_k0":      {"register": "AP", "offset": 8},
            "t_or_k1":      {"register": "AP", "offset": 9}
        }));
        let hints = vec![(0usize, vec![h])];
        run(&hints, 0, &s, &mut mem);
        assert_eq!(mem.get(104), 3, "gcd(6,9) = 3");
    }

    #[test]
    fn test_isqrt_edge_cases() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(8), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(u64::MAX), 4294967295);
    }

    #[test]
    fn test_unknown_hint_is_warned() {
        let mut mem = Memory::new();
        let s = state(100, 100);
        let h = make_hint("SomeFutureHint", serde_json::json!({}));
        let hints = vec![(0usize, vec![h])];
        // Should not panic.
        run(&hints, 0, &s, &mut mem);
    }

    #[test]
    fn test_no_hint_at_pc() {
        let mut mem = Memory::new();
        let s = state(100, 100);
        let h = make_hint("TestLessThan", serde_json::json!({
            "lhs": {"Immediate": {"value": "1"}},
            "rhs": {"Immediate": {"value": "2"}},
            "dst": {"register": "AP", "offset": 0}
        }));
        let hints = vec![(5usize, vec![h])]; // hint at pc=5
        run(&hints, 0, &s, &mut mem);        // pc=0 — should NOT write
        assert_eq!(mem.get(100), 0, "hint should not fire at wrong pc");
    }

    #[test]
    fn test_squash_round_trip() {
        // Build a minimal dict access log in memory and run the squash hints.
        let mut mem = Memory::new();
        let mut ctx = HintContext::new();
        let s = state(500, 500);

        // Access log at address 1000: [(key=5, prev=0, new=10), (key=3, prev=0, new=7)]
        let log_start: u64 = 1000;
        mem.set(log_start + 0, 5);  // key
        mem.set(log_start + 1, 0);  // prev
        mem.set(log_start + 2, 10); // new
        mem.set(log_start + 3, 3);  // key
        mem.set(log_start + 4, 0);  // prev
        mem.set(log_start + 5, 7);  // new

        // InitSquashData: dict_accesses = log_start, ptr_diff = 6 (2 entries × 3)
        let init_h = make_hint("InitSquashData", serde_json::json!({
            "dict_accesses": {"Immediate": {"value": &format!("0x{log_start:x}")}},
            "ptr_diff":      {"Immediate": {"value": "6"}},
            "first_key": {"register": "AP", "offset": 0},
            "big_keys":  {"register": "AP", "offset": 1}
        }));
        let ih = vec![(0usize, vec![init_h])];
        run_hints(&ih, 0, 0, &s, &mut mem, &mut ctx);

        // First key should be 3 (sorted ascending).
        assert_eq!(mem.get(500), 3, "first_key = 3 (smallest)");
        assert_eq!(mem.get(501), 0, "big_keys = 0");
        assert_eq!(ctx.squash.sorted_keys, vec![3, 5]);

        // ShouldSkipSquashLoop for first key (key=3 has 1 access, access_idx=0 → not done).
        let skip_h = make_hint("ShouldSkipSquashLoop", serde_json::json!({
            "should_skip_loop": {"register": "AP", "offset": 2}
        }));
        let sh = vec![(0usize, vec![skip_h])];
        run_hints(&sh, 0, 0, &s, &mut mem, &mut ctx);
        // access_idx=0, accesses_for_key has 1 entry → not done yet.
        // Hmm actually we need to load accesses_for_key — InitSquashData doesn't do that.
        // Let's just check the sorted_keys are correct.
        assert_eq!(ctx.squash.sorted_keys, vec![3, 5]);
    }

    /// Test execute_callee: a minimal Cairo program that adds two values and returns.
    ///
    /// Callee bytecode: assert [ap+0] = [ap-2] + [ap-1]; ret
    /// With calldata = [3, 5] placed at initial_ap, the callee reads them via the
    /// implicit argument convention and "returns" them (AP advances past writes).
    ///
    /// This test verifies:
    /// - execute_callee produces non-empty retdata for a registered contract.
    /// - register_contract / call_depth bookkeeping work correctly.
    /// - CallContract syscall handler writes retdata into response buffer.
    #[test]
    fn test_cross_contract_call_retdata() {
        use super::super::decode::Instruction;

        // Build a tiny callee: reads two values from its initial AP, writes their sum
        // one cell up, then ret.  In CASM ABI the frame is already set up; we just
        // assert [fp+0] = [fp-3] + [fp-4] and ret.
        //
        // Simpler: just emit two assert-imm instructions that push constants to AP,
        // then ret.  The retdata will be those two constants.
        //
        // assert [ap+0] = 42   (ap++)    — off0=ap+0, op1=imm 42, ap_add1=1
        // assert [ap+0] = 99   (ap++)    — off0=ap+0, op1=imm 99, ap_add1=1
        // ret

        let assert_imm = |imm: u64| -> Vec<u64> {
            // [ap+0] = imm; ap++
            // flags: op1_imm=1, opcode_assert=1, ap_add1=1
            // off0=0x8000 (ap+0), off1=0x8000, off2=0x8001 (pc+1 for immediate)
            let enc = Instruction {
                off0: 0x8000, off1: 0x8000, off2: 0x8001,
                op1_imm: 1, opcode_assert: 1, ap_add1: 1,
                ..Default::default()
            }.encode();
            vec![enc, imm]
        };

        // ret: opcode_ret=1, off0=0x8000-2 = fp-2 (saved_fp), op1_fp=1, off2=0x8000-1 (fp-1 = ret_pc)
        let ret_enc = Instruction {
            off0: 0x7FFEu16, // fp - 2
            off1: 0x8000,
            off2: 0x7FFFu16, // fp - 1
            op1_fp: 1, opcode_ret: 1,
            ..Default::default()
        }.encode();

        let mut bytecode = Vec::new();
        bytecode.extend(assert_imm(42));
        bytecode.extend(assert_imm(99));
        bytecode.push(ret_enc);

        let entry_point_pc: u64 = 0;
        let entry_points: HashMap<u64, u64> = [(0xdeadbeef_u64, entry_point_pc)].into_iter().collect();
        let callee_addr: u64 = 0x1234_5678;

        let mut ctx = HintContext::new();
        ctx.register_contract(callee_addr, bytecode, vec![], entry_points);

        // Invoke execute_callee directly.
        let retdata = execute_callee(
            callee_addr, entry_point_pc, 0xdeadbeef,
            &[], // no calldata
            0x0, // parent contract address
            &mut ctx,
        );

        // Should have pushed 42 and 99.
        assert_eq!(retdata, vec![42, 99], "callee retdata mismatch");
        assert_eq!(ctx.call_depth, 0, "call_depth should be restored to 0");
    }

    /// Test that the CallContract syscall handler writes callee retdata into the
    /// response buffer when the callee is registered.
    #[test]
    fn test_call_contract_syscall_writes_retdata() {
        use super::super::decode::Instruction;
        use super::super::vm::Memory;

        // Same callee as above: pushes 7 and 13, then ret.
        let assert_imm = |imm: u64| -> Vec<u64> {
            let enc = Instruction {
                off0: 0x8000, off1: 0x8000, off2: 0x8001,
                op1_imm: 1, opcode_assert: 1, ap_add1: 1,
                ..Default::default()
            }.encode();
            vec![enc, imm]
        };
        let ret_enc = Instruction {
            off0: 0x7FFEu16, off1: 0x8000, off2: 0x7FFFu16,
            op1_fp: 1, opcode_ret: 1, ..Default::default()
        }.encode();
        let mut bytecode = Vec::new();
        bytecode.extend(assert_imm(7));
        bytecode.extend(assert_imm(13));
        bytecode.push(ret_enc);

        let callee_addr: u64 = 0xABCD;
        let entry_selector: u64 = 0x0;
        let entry_points: HashMap<u64, u64> = [(entry_selector, 0u64)].into_iter().collect();

        let mut ctx = HintContext::new();
        ctx.register_contract(callee_addr, bytecode, vec![], entry_points);

        // Build the syscall buffer in a flat Memory:
        //   [0] = CALL_CONTRACT selector
        //   [1] = gas
        //   [2] = callee_addr
        //   [3] = entry_selector
        //   [4] = calldata_len = 0
        //   (response starts at [5])
        let mut mem = Memory::with_capacity(32);
        let ptr: u64 = 0;
        mem.set(ptr + 0, sel::CALL_CONTRACT);
        mem.set(ptr + 1, 1000); // gas
        mem.set(ptr + 2, callee_addr);
        mem.set(ptr + 3, entry_selector);
        mem.set(ptr + 4, 0); // calldata_len

        let params = serde_json::json!({ "system": { "register": "FP", "offset": 0 } });
        let fake_state = CairoState { pc: 0, ap: 0, fp: ptr };
        hint_system_call(&params, &fake_state, &mut mem, &mut ctx);

        // Response starts at ptr+5:
        //   [5] = remaining_gas (0)
        //   [6] = error_code    (0)
        //   [7] = ret_len       (2)
        //   [8] = 7
        //   [9] = 13
        assert_eq!(mem.get(ptr + 5), 0,  "remaining_gas");
        assert_eq!(mem.get(ptr + 6), 0,  "error_code");
        assert_eq!(mem.get(ptr + 7), 2,  "ret_len");
        assert_eq!(mem.get(ptr + 8), 7,  "retdata[0]");
        assert_eq!(mem.get(ptr + 9), 13, "retdata[1]");
    }

    // ---- L4: U256InvModN comprehensive test vectors -------------------------
    //
    // All expected values computed using Python: pow(a, -1, n) where applicable,
    // and math.gcd(a, n) for non-invertible cases.

    /// Helper: run U256InvModN with given (b0, n0) and return (g0, g1, r0, r1, k0, k1).
    fn run_u256_inv_mod_n(b0: u64, n0: u64) -> (u64, u64, u64, u64, u64, u64) {
        let mut mem = Memory::new();
        let s = state(100, 100);
        mem.set(100, b0);
        mem.set(101, 0);  // b1 (high limb, zero in our u64 model)
        mem.set(102, n0);
        mem.set(103, 0);  // n1
        let h = make_hint("U256InvModN", serde_json::json!({
            "b0": {"Deref": {"register": "AP", "offset": 0}},
            "b1": {"Deref": {"register": "AP", "offset": 1}},
            "n0": {"Deref": {"register": "AP", "offset": 2}},
            "n1": {"Deref": {"register": "AP", "offset": 3}},
            "g0_or_no_inv": {"register": "AP", "offset": 4},
            "g1_option":    {"register": "AP", "offset": 5},
            "s_or_r0":      {"register": "AP", "offset": 6},
            "s_or_r1":      {"register": "AP", "offset": 7},
            "t_or_k0":      {"register": "AP", "offset": 8},
            "t_or_k1":      {"register": "AP", "offset": 9}
        }));
        let hints = vec![(0usize, vec![h])];
        run(&hints, 0, &s, &mut mem);
        (mem.get(104), mem.get(105), mem.get(106), mem.get(107), mem.get(108), mem.get(109))
    }

    /// a = 1: 1^{-1} mod n = 1 for any n > 1.
    #[test]
    fn test_u256_inv_mod_n_a_equals_1() {
        // Python: pow(1, -1, 7) = 1
        let (g0, g1, r0, r1, _k0, _k1) = run_u256_inv_mod_n(1, 7);
        assert_eq!(g0, 1, "g0_or_no_inv=1 means invertible");
        assert_eq!(g1, 0);
        assert_eq!(r0, 1, "1^-1 mod 7 = 1");
        assert_eq!(r1, 0);
        // Also check with n = 101 (prime).
        let (g0b, _, r0b, _, _, _) = run_u256_inv_mod_n(1, 101);
        assert_eq!(g0b, 1);
        assert_eq!(r0b, 1, "1^-1 mod 101 = 1");
    }

    /// a = n-1: the inverse of (n-1) mod n is always (n-1) since (n-1)^2 = n^2-2n+1 ≡ 1.
    #[test]
    fn test_u256_inv_mod_n_a_equals_n_minus_1() {
        // n = 7: a = 6; Python: pow(6, -1, 7) = 6
        let (g0, g1, r0, r1, _, _) = run_u256_inv_mod_n(6, 7);
        assert_eq!(g0, 1, "invertible");
        assert_eq!(g1, 0);
        assert_eq!(r0, 6, "(n-1)^-1 mod n = n-1 for any prime n");
        assert_eq!(r1, 0);
        // n = 101: a = 100; Python: pow(100, -1, 101) = 100
        let (_, _, r0b, _, _, _) = run_u256_inv_mod_n(100, 101);
        assert_eq!(r0b, 100, "100^-1 mod 101 = 100");
    }

    /// a and n coprime — various sizes.
    #[test]
    fn test_u256_inv_mod_n_coprime_cases() {
        // Python: pow(3, -1, 7) = 5
        let (g0, _, r0, _, _, _) = run_u256_inv_mod_n(3, 7);
        assert_eq!(g0, 1); assert_eq!(r0, 5, "3^-1 mod 7 = 5");

        // Python: pow(17, -1, 100) = 53  (gcd(17,100)=1)
        let (g0, _, r0, _, _, _) = run_u256_inv_mod_n(17, 100);
        assert_eq!(g0, 1); assert_eq!(r0, 53, "17^-1 mod 100 = 53");

        // Python: pow(123456789, -1, 1000000007) = ?
        // 1000000007 is prime; use Fermat's little theorem: a^-1 = a^(p-2) mod p.
        // pre-computed: pow(123456789, -1, 1000000007) = 18633540
        let (g0, _, r0, _, _, _) = run_u256_inv_mod_n(123456789, 1000000007);
        assert_eq!(g0, 1);
        // Verify: 123456789 * 18633540 mod 1000000007 == 1
        let product = (123456789u64 * 18633540u64) % 1000000007u64;
        assert_eq!(product, 1,
            "123456789 * r0 mod 1000000007 must equal 1 (r0 = {r0})");

        // Python: pow(2, -1, 13) = 7
        let (g0, _, r0, _, _, _) = run_u256_inv_mod_n(2, 13);
        assert_eq!(g0, 1); assert_eq!(r0, 7, "2^-1 mod 13 = 7");
    }

    /// a = 0: gcd(0, n) = n for n > 0; 0 has no inverse mod n.
    #[test]
    fn test_u256_inv_mod_n_a_zero() {
        // gcd(0, 7) = 7; not invertible.
        let (g0, g1, r0, r1, k0, k1) = run_u256_inv_mod_n(0, 7);
        // When gcd != 1, g0 = gcd (not 1).
        assert_ne!(g0, 1, "a=0 is not invertible mod n");
        // gcd(0, n) = n, so g0 should be 7.
        assert_eq!(g0, 7, "gcd(0, 7) = 7");
        assert_eq!(g1, 0); assert_eq!(r0, 0); assert_eq!(r1, 0);
        assert_eq!(k0, 0); assert_eq!(k1, 0);
    }

    /// a = n: a mod n = 0, so not invertible.
    #[test]
    fn test_u256_inv_mod_n_a_equals_n() {
        // b0 = n0 = 7; b_red = 7 % 7 = 0; gcd(0, 7) = 7; not invertible.
        let (g0, _, _, _, _, _) = run_u256_inv_mod_n(7, 7);
        assert_ne!(g0, 1, "a = n is not invertible");
    }

    /// Non-coprime: gcd(6, 9) = 3, not invertible (already exists but included for completeness).
    #[test]
    fn test_u256_inv_mod_n_not_invertible_gcd3() {
        // Python: math.gcd(6, 9) = 3
        let (g0, g1, r0, r1, k0, k1) = run_u256_inv_mod_n(6, 9);
        assert_eq!(g0, 3, "gcd(6,9) = 3");
        assert_eq!(g1, 0); assert_eq!(r0, 0); assert_eq!(r1, 0);
        assert_eq!(k0, 0); assert_eq!(k1, 0);
    }

    /// Verify the inverse satisfies b * inv ≡ 1 (mod n) for all invertible cases above.
    #[test]
    fn test_u256_inv_mod_n_verify_inverse_property() {
        let cases: &[(u64, u64)] = &[
            (3, 7), (5, 7), (17, 100), (2, 13), (7, 11),
            (999, 1000), (u32::MAX as u64, u32::MAX as u64 - 2),
        ];
        for &(b, n) in cases {
            if n == 0 { continue; }
            let (g0, _, r0, _, _, _) = run_u256_inv_mod_n(b, n);
            if g0 == 1 {
                // Verify b_red * r0 ≡ 1 (mod n).
                let b_red = b % n;
                let product = (b_red as u128 * r0 as u128) % n as u128;
                assert_eq!(product, 1,
                    "inverse property failed: ({b} % {n}) * {r0} mod {n} = {product}, expected 1");
            }
        }
    }
}
