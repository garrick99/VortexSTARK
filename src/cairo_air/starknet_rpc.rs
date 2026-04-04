//! Starknet JSON-RPC client for fetching contract CASM and block data.
//!
//! Supports:
//! - `starknet_getCompiledCasm` — fetch CASM directly from a Sierra class hash
//! - `starknet_getClass` — fetch contract class (Sierra or Cairo 0)
//! - `starknet_getBlockWithTxs` — fetch block transactions
//!
//! Public RPC endpoints:
//!   Mainnet: https://starknet-mainnet.public.blastapi.io
//!   Sepolia: https://starknet-sepolia.public.blastapi.io

use serde::{Deserialize, Serialize};
use super::casm_loader::{CasmProgram, CasmFormat};

/// Default public RPC endpoints.
pub const MAINNET_RPC: &str = "https://rpc.starknet.lava.build";
pub const SEPOLIA_RPC: &str = "https://rpc.starknet-testnet.lava.build";

/// Starknet RPC client.
pub struct StarknetClient {
    client: reqwest::Client,
    rpc_url: String,
}

/// Block identifier for RPC calls.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum BlockId {
    Number { block_number: u64 },
    Hash { block_hash: String },
    Tag(String),
}

impl BlockId {
    pub fn latest() -> Self { BlockId::Tag("latest".to_string()) }
    pub fn number(n: u64) -> Self { BlockId::Number { block_number: n } }
    pub fn hash(h: &str) -> Self { BlockId::Hash { block_hash: h.to_string() } }
}

/// Transaction summary from a block.
#[derive(Clone, Debug)]
pub struct TransactionSummary {
    pub hash: String,
    pub class_hash: Option<String>,
    pub tx_type: String,
}

/// Block summary.
#[derive(Clone, Debug)]
pub struct BlockSummary {
    pub block_number: u64,
    pub block_hash: String,
    pub timestamp: u64,
    pub transactions: Vec<TransactionSummary>,
}

// --- JSON-RPC request/response types ---

#[derive(Serialize)]
struct RpcRequest<P: Serialize> {
    jsonrpc: &'static str,
    method: &'static str,
    params: P,
    id: u32,
}

#[derive(Deserialize)]
struct RpcResponse {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    result: Option<serde_json::Value>,
    error: Option<RpcError>,
}

#[derive(Deserialize, Debug)]
struct RpcError {
    code: i64,
    message: String,
}

impl StarknetClient {
    /// Create a new client for the given RPC URL.
    pub fn new(rpc_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            rpc_url: rpc_url.to_string(),
        }
    }

    /// Create a client for Starknet mainnet.
    pub fn mainnet() -> Self { Self::new(MAINNET_RPC) }

    /// Create a client for Starknet Sepolia testnet.
    pub fn sepolia() -> Self { Self::new(SEPOLIA_RPC) }

    /// Fetch the raw compiled CASM JSON for a Sierra class hash (unparsed).
    /// Returns the raw `serde_json::Value` from `starknet_getCompiledCasm`.
    /// Useful for passing directly to stwo-cairo's `run_and_prove` binary.
    pub async fn get_compiled_casm_raw(&self, class_hash: &str) -> Result<serde_json::Value, String> {
        let class_hash = normalize_hex(class_hash);
        #[derive(Serialize)]
        struct Params { class_hash: String }
        self.rpc_call("starknet_getCompiledCasm", Params { class_hash }).await
    }

    /// Fetch compiled CASM for a Sierra class hash.
    /// Uses `starknet_getCompiledCasm` (Starknet v0.13.2+).
    pub async fn get_compiled_casm(&self, class_hash: &str) -> Result<CasmProgram, String> {
        let class_hash = normalize_hex(class_hash);

        #[derive(Serialize)]
        struct Params { class_hash: String }

        let result = self.rpc_call(
            "starknet_getCompiledCasm",
            Params { class_hash: class_hash.clone() },
        ).await?;

        parse_casm_rpc_result(&result, &class_hash)
    }

    /// Fetch a contract class (Sierra or Cairo 0) by class hash.
    /// Falls back to extracting bytecode from whichever format is returned.
    pub async fn get_class(&self, class_hash: &str, block_id: &BlockId) -> Result<CasmProgram, String> {
        let class_hash = normalize_hex(class_hash);

        #[derive(Serialize)]
        struct Params { block_id: BlockId, class_hash: String }

        let result = self.rpc_call(
            "starknet_getClass",
            Params {
                block_id: block_id.clone(),
                class_hash: class_hash.clone(),
            },
        ).await?;

        // Check if it's a Cairo 0 class (has "program" field with "data")
        if let Some(program) = result.get("program") {
            if let Some(data) = program.get("data") {
                return parse_cairo0_rpc_result(data, program, &class_hash);
            }
        }

        // Sierra class — has "sierra_program" but not directly executable.
        // Try to get CASM via starknet_getCompiledCasm instead.
        if result.get("sierra_program").is_some() {
            return self.get_compiled_casm(&class_hash).await;
        }

        Err(format!("unexpected class format for {class_hash}"))
    }

    /// Fetch block transactions.
    pub async fn get_block_with_txs(&self, block_id: &BlockId) -> Result<BlockSummary, String> {
        #[derive(Serialize)]
        struct Params { block_id: BlockId }

        let result = self.rpc_call(
            "starknet_getBlockWithTxs",
            Params { block_id: block_id.clone() },
        ).await?;

        let block_number = result.get("block_number")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let block_hash = result.get("block_hash")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let timestamp = result.get("timestamp")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let mut transactions = Vec::new();
        if let Some(serde_json::Value::Array(txs)) = result.get("transactions") {
            for tx in txs {
                let hash = tx.get("transaction_hash")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let class_hash = tx.get("class_hash")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let tx_type = tx.get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("UNKNOWN")
                    .to_string();

                transactions.push(TransactionSummary {
                    hash,
                    class_hash,
                    tx_type,
                });
            }
        }

        Ok(BlockSummary {
            block_number,
            block_hash,
            timestamp,
            transactions,
        })
    }

    /// Make a JSON-RPC call and return the result field.
    async fn rpc_call<P: Serialize>(&self, method: &'static str, params: P) -> Result<serde_json::Value, String> {
        let request = RpcRequest {
            jsonrpc: "2.0",
            method,
            params,
            id: 1,
        };

        let response = self.client
            .post(&self.rpc_url)
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("RPC request failed: {e}"))?;

        if !response.status().is_success() {
            return Err(format!("RPC HTTP error: {}", response.status()));
        }

        let rpc_response: RpcResponse = response.json().await
            .map_err(|e| format!("RPC response parse error: {e}"))?;

        if let Some(error) = rpc_response.error {
            return Err(format!("RPC error {}: {}", error.code, error.message));
        }

        rpc_response.result.ok_or_else(|| "RPC response has no result".to_string())
    }
}

/// Parse CASM from starknet_getCompiledCasm result.
fn parse_casm_rpc_result(result: &serde_json::Value, class_hash: &str) -> Result<CasmProgram, String> {
    let bytecode_arr = result.get("bytecode")
        .and_then(|v| v.as_array())
        .ok_or("no bytecode in CASM response")?;

    let mut bytecode = Vec::with_capacity(bytecode_arr.len());
    let mut overflow_count = 0;

    for (i, val) in bytecode_arr.iter().enumerate() {
        let hex = val.as_str().ok_or(format!("bytecode[{i}] not a string"))?;
        let (v, overflow) = super::casm_loader::parse_hex_felt(hex)
            .map_err(|e| format!("bytecode[{i}]: {e}"))?;
        if overflow { overflow_count += 1; }
        bytecode.push(v);
    }

    // Extract entry points
    let (entry_point, builtins) = extract_entry_point(result);

    // Parse hints
    let hints = if let Some(hints_val) = result.get("hints") {
        super::casm_loader::parse_hints(hints_val)
    } else {
        vec![]
    };

    Ok(CasmProgram {
        bytecode,
        entry_point,
        name: format!("starknet_{}", &class_hash[..10.min(class_hash.len())]),
        builtins,
        format: CasmFormat::CasmJson,
        hints,
        overflow_count,
    })
}

/// Parse Cairo 0 class from starknet_getClass result.
fn parse_cairo0_rpc_result(
    data: &serde_json::Value,
    program: &serde_json::Value,
    class_hash: &str,
) -> Result<CasmProgram, String> {
    let data_arr = data.as_array()
        .ok_or("program.data not an array")?;

    let mut bytecode = Vec::with_capacity(data_arr.len());
    let mut overflow_count = 0;

    for (i, val) in data_arr.iter().enumerate() {
        let hex = val.as_str().ok_or(format!("data[{i}] not a string"))?;
        let (v, overflow) = super::casm_loader::parse_hex_felt(hex)
            .map_err(|e| format!("data[{i}]: {e}"))?;
        if overflow { overflow_count += 1; }
        bytecode.push(v);
    }

    let builtins = program.get("builtins")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    Ok(CasmProgram {
        bytecode,
        entry_point: 0,
        name: format!("cairo0_{}", &class_hash[..10.min(class_hash.len())]),
        builtins,
        format: CasmFormat::Cairo0Json,
        hints: vec![],
        overflow_count,
    })
}

/// Extract entry point and builtins from CASM entry_points_by_type.
fn extract_entry_point(result: &serde_json::Value) -> (u64, Vec<String>) {
    if let Some(eps) = result.get("entry_points_by_type") {
        for key in &["External", "EXTERNAL", "Constructor", "CONSTRUCTOR", "L1Handler", "L1_HANDLER"] {
            if let Some(serde_json::Value::Array(arr)) = eps.get(*key) {
                if let Some(ep) = arr.first() {
                    let offset = ep.get("offset").and_then(|v| v.as_u64()).unwrap_or(0);
                    let builtins = ep.get("builtins")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default();
                    return (offset, builtins);
                }
            }
        }
    }
    (0, vec![])
}

/// Normalize a hex string to have 0x prefix.
fn normalize_hex(s: &str) -> String {
    if s.starts_with("0x") || s.starts_with("0X") {
        s.to_string()
    } else {
        format!("0x{s}")
    }
}

/// Print a block summary.
pub fn print_block_summary(block: &BlockSummary) {
    eprintln!("Block #{} ({})", block.block_number, &block.block_hash[..18]);
    eprintln!("  Timestamp:    {}", block.timestamp);
    eprintln!("  Transactions: {}", block.transactions.len());

    let mut type_counts = std::collections::HashMap::new();
    let mut deploy_classes = Vec::new();

    for tx in &block.transactions {
        *type_counts.entry(tx.tx_type.clone()).or_insert(0) += 1;
        if let Some(ch) = &tx.class_hash {
            deploy_classes.push(ch.clone());
        }
    }

    for (t, count) in &type_counts {
        eprintln!("    {t}: {count}");
    }

    if !deploy_classes.is_empty() {
        eprintln!("  Unique classes deployed: {}", deploy_classes.len());
        for ch in deploy_classes.iter().take(5) {
            eprintln!("    {ch}");
        }
        if deploy_classes.len() > 5 {
            eprintln!("    ... and {} more", deploy_classes.len() - 5);
        }
    }
}
