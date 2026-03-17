//! GPU bytecode constraint evaluator.
//!
//! This module implements a "tracing" `EvalAtRow` that records constraint evaluation
//! as a flat bytecode program. The bytecode is the same for every row — only the
//! trace column values change. This makes it suitable for GPU execution: upload the
//! bytecode once, then interpret it in parallel across all rows.
//!
//! # Architecture
//!
//! 1. **`bytecode`** — Defines the `BytecodeOp` instruction set and `BytecodeProgram`.
//! 2. **`tracing`** — `TracingEvalAtRow` implements `EvalAtRow` by emitting bytecode ops.
//! 3. *(future)* **CUDA kernel** — GPU interpreter that replays the bytecode per-row.
//!
//! # Usage
//!
//! ```ignore
//! use vortex_cuda_backend::constraint_eval::record_bytecode;
//!
//! let program = record_bytecode(&my_framework_eval, claimed_sum);
//! println!("{}", program.dump());
//! ```

pub mod bytecode;
pub mod tracing;

pub use bytecode::{BytecodeOp, BytecodeProgram};
pub use tracing::record_bytecode;
