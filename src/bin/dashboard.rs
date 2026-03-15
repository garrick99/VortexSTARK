//! VortexSTARK web dashboard — live proof demo + GPU monitor.

use axum::{
    Router,
    extract::{State, ws::{WebSocket, WebSocketUpgrade, Message}},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, oneshot};

// ---- Types ----

#[derive(Clone, Serialize)]
struct GpuStats {
    vram_used_mb: u64,
    vram_total_mb: u64,
    temp_c: u64,
    util_pct: u64,
    power_w: f64,
}

#[derive(Deserialize)]
struct ProveRequest {
    proof_type: String,
    log_n: u32,
}

#[derive(Deserialize)]
struct PedersenRequest {
    n: usize,
}

#[derive(Clone, Serialize)]
struct WsMessage {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(flatten)]
    data: serde_json::Value,
}

struct GpuJob {
    request: GpuRequest,
    broadcast: broadcast::Sender<WsMessage>,
}

enum GpuRequest {
    Prove { proof_type: String, log_n: u32, resp: oneshot::Sender<()> },
    PedersenCompare { n: usize, resp: oneshot::Sender<()> },
}

#[derive(Clone)]
struct AppState {
    gpu_tx: mpsc::Sender<GpuJob>,
    broadcast_tx: broadcast::Sender<WsMessage>,
}

// ---- GPU Worker Thread ----

fn gpu_worker(mut rx: mpsc::Receiver<GpuJob>) {
    use vortexstark::cuda::ffi;

    ffi::init_memory_pool();
    vortexstark::cairo_air::pedersen::gpu_init();

    while let Some(job) = rx.blocking_recv() {
        let tx = &job.broadcast;
        match job.request {
            GpuRequest::Prove { proof_type, log_n, resp } => {
                gpu_prove(&proof_type, log_n, tx);
                let _ = resp.send(());
            }
            GpuRequest::PedersenCompare { n, resp } => {
                gpu_pedersen_compare(n, tx);
                let _ = resp.send(());
            }
        }
        // Release VRAM back to OS after each job (keep 256 MB for fast reuse)
        ffi::vram_release(256 * 1024 * 1024);
    }
}

fn broadcast(tx: &broadcast::Sender<WsMessage>, msg_type: &str, data: serde_json::Value) {
    let _ = tx.send(WsMessage {
        msg_type: msg_type.to_string(),
        data,
    });
}

fn gpu_prove(proof_type: &str, log_n: u32, tx: &broadcast::Sender<WsMessage>) {
    use vortexstark::field::M31;
    use std::time::Instant;

    broadcast(tx, "proof_started", serde_json::json!({
        "proof_type": proof_type, "log_n": log_n
    }));

    match proof_type {
        "fibonacci" => {
            let t0 = Instant::now();
            let proof = vortexstark::prover::prove_lean(M31(1), M31(1), log_n);
            let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

            broadcast(tx, "phase", serde_json::json!({"name": "prove (all phases)", "ms": prove_ms}));

            let t1 = Instant::now();
            let ok = vortexstark::verifier::verify(&proof).is_ok();
            let verify_ms = t1.elapsed().as_secs_f64() * 1000.0;

            broadcast(tx, "phase", serde_json::json!({"name": "verify", "ms": verify_ms}));

            broadcast(tx, "proof_done", serde_json::json!({
                "proof_type": "fibonacci", "log_n": log_n,
                "prove_ms": prove_ms, "verify_ms": verify_ms, "verified": ok
            }));
        }
        "cairo" => {
            use vortexstark::cairo_air::decode::Instruction;
            use vortexstark::cairo_air::prover::{cairo_prove_cached, cairo_verify, CairoProverCache};

            let n = 1usize << log_n;

            // Build fibonacci program for Cairo VM
            let assert_imm = Instruction {
                off0: 0x8000, off1: 0x8000, off2: 0x8001,
                op1_imm: 1, opcode_assert: 1, ap_add1: 1,
                ..Default::default()
            };
            let add_instr = Instruction {
                off0: 0x8000, off1: 0x8000u16.wrapping_sub(2), off2: 0x8000u16.wrapping_sub(1),
                op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
                ..Default::default()
            };
            let mut program = Vec::new();
            program.push(assert_imm.encode()); program.push(1);
            program.push(assert_imm.encode()); program.push(1);
            for _ in 0..n.saturating_sub(2) {
                program.push(add_instr.encode());
            }

            let t0 = Instant::now();
            let cache = CairoProverCache::new(log_n);
            let cache_ms = t0.elapsed().as_secs_f64() * 1000.0;
            broadcast(tx, "phase", serde_json::json!({"name": "cache init", "ms": cache_ms}));

            let t1 = Instant::now();
            let proof = cairo_prove_cached(&program, n, log_n, &cache, None);
            let prove_ms = t1.elapsed().as_secs_f64() * 1000.0;
            broadcast(tx, "phase", serde_json::json!({"name": "prove (all phases)", "ms": prove_ms}));

            let t2 = Instant::now();
            let ok = cairo_verify(&proof).is_ok();
            let verify_ms = t2.elapsed().as_secs_f64() * 1000.0;
            broadcast(tx, "phase", serde_json::json!({"name": "verify", "ms": verify_ms}));

            broadcast(tx, "proof_done", serde_json::json!({
                "proof_type": "cairo", "log_n": log_n,
                "prove_ms": prove_ms + cache_ms, "verify_ms": verify_ms, "verified": ok
            }));
        }
        _ => {
            broadcast(tx, "error", serde_json::json!({"message": format!("Unknown proof type: {proof_type}")}));
        }
    }
}

fn gpu_pedersen_compare(n: usize, tx: &broadcast::Sender<WsMessage>) {
    use vortexstark::cairo_air::pedersen;
    use vortexstark::cairo_air::stark252_field::{Fp, pedersen_hash};
    use std::time::Instant;

    let inputs_a: Vec<Fp> = (0..n).map(|i| Fp::from_u64(i as u64 + 1)).collect();
    let inputs_b: Vec<Fp> = (0..n).map(|i| Fp::from_u64(i as u64 + 100)).collect();

    // GPU benchmark
    let _ = pedersen::gpu_hash_batch(&inputs_a[..1.min(n)], &inputs_b[..1.min(n)]); // warmup
    let t0 = Instant::now();
    let _ = pedersen::gpu_hash_batch(&inputs_a, &inputs_b);
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let gpu_rate = n as f64 / (gpu_ms / 1000.0);

    // CPU benchmark (multi-threaded)
    let n_threads = std::thread::available_parallelism().map_or(1, |p| p.get());
    let t1 = Instant::now();
    std::thread::scope(|s| {
        let chunk = (n + n_threads - 1) / n_threads;
        for t in 0..n_threads {
            let start = t * chunk;
            let end = (start + chunk).min(n);
            if start >= end { continue; }
            let a = &inputs_a[start..end];
            let b = &inputs_b[start..end];
            s.spawn(move || {
                for (x, y) in a.iter().zip(b.iter()) {
                    let _ = pedersen_hash(*x, *y);
                }
            });
        }
    });
    let cpu_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let cpu_rate = n as f64 / (cpu_ms / 1000.0);

    let speedup = gpu_rate / cpu_rate;

    broadcast(tx, "pedersen_compare", serde_json::json!({
        "cpu_rate": cpu_rate, "gpu_rate": gpu_rate,
        "speedup": speedup, "n": n, "cpu_threads": n_threads
    }));
}

// ---- nvidia-smi ----

fn parse_gpu_stats() -> Option<GpuStats> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw",
               "--format=csv,noheader,nounits"])
        .output().ok()?;
    let s = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = s.trim().split(',').map(|p| p.trim()).collect();
    if parts.len() < 5 { return None; }
    Some(GpuStats {
        vram_used_mb: parts[0].parse().ok()?,
        vram_total_mb: parts[1].parse().ok()?,
        temp_c: parts[2].parse().ok()?,
        util_pct: parts[3].parse().ok()?,
        power_w: parts[4].parse().ok()?,
    })
}

// ---- Axum Handlers ----

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../../static/index.html"))
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(mut socket: WebSocket, state: Arc<AppState>) {
    let mut rx = state.broadcast_tx.subscribe();
    loop {
        tokio::select! {
            msg = rx.recv() => {
                match msg {
                    Ok(ws_msg) => {
                        let json = serde_json::to_string(&ws_msg).unwrap_or_default();
                        if socket.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(_)) => {} // ignore client messages
                    _ => break,
                }
            }
        }
    }
}

async fn prove_handler(
    State(state): State<Arc<AppState>>,
    axum::Json(req): axum::Json<ProveRequest>,
) -> StatusCode {
    // Validate log_n bounds
    let max_log_n = match req.proof_type.as_str() {
        "fibonacci" => 30,
        "cairo" => 26,
        _ => {
            broadcast(&state.broadcast_tx, "error",
                serde_json::json!({"message": format!("Unknown proof type: {}", req.proof_type)}));
            return StatusCode::BAD_REQUEST;
        }
    };
    if req.log_n > max_log_n || req.log_n < 4 {
        broadcast(&state.broadcast_tx, "error",
            serde_json::json!({"message": format!("log_n must be 4..{max_log_n}, got {}", req.log_n)}));
        return StatusCode::BAD_REQUEST;
    }

    let (resp_tx, _resp_rx) = oneshot::channel();
    let _ = state.gpu_tx.send(GpuJob {
        request: GpuRequest::Prove {
            proof_type: req.proof_type,
            log_n: req.log_n,
            resp: resp_tx,
        },
        broadcast: state.broadcast_tx.clone(),
    }).await;
    StatusCode::ACCEPTED
}

async fn pedersen_handler(
    State(state): State<Arc<AppState>>,
    axum::Json(req): axum::Json<PedersenRequest>,
) -> StatusCode {
    if req.n == 0 || req.n > 1_000_000 {
        broadcast(&state.broadcast_tx, "error",
            serde_json::json!({"message": "n must be 1..1000000"}));
        return StatusCode::BAD_REQUEST;
    }
    let (resp_tx, _resp_rx) = oneshot::channel();
    let _ = state.gpu_tx.send(GpuJob {
        request: GpuRequest::PedersenCompare { n: req.n, resp: resp_tx },
        broadcast: state.broadcast_tx.clone(),
    }).await;
    StatusCode::ACCEPTED
}

async fn gpu_stats_handler() -> impl IntoResponse {
    match parse_gpu_stats() {
        Some(stats) => axum::Json(serde_json::to_value(&stats).unwrap()).into_response(),
        None => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

// ---- Main ----

#[tokio::main]
async fn main() {
    let (gpu_tx, gpu_rx) = mpsc::channel::<GpuJob>(16);
    let (broadcast_tx, _) = broadcast::channel::<WsMessage>(256);

    // Spawn dedicated GPU worker thread
    std::thread::spawn(move || gpu_worker(gpu_rx));

    let state = Arc::new(AppState {
        gpu_tx,
        broadcast_tx: broadcast_tx.clone(),
    });

    // Background task: push GPU stats every 2 seconds
    let stats_tx = broadcast_tx.clone();
    tokio::spawn(async move {
        loop {
            if let Some(stats) = parse_gpu_stats() {
                let _ = stats_tx.send(WsMessage {
                    msg_type: "gpu_stats".to_string(),
                    data: serde_json::to_value(&stats).unwrap_or_default(),
                });
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        }
    });

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/ws", get(ws_handler))
        .route("/api/prove", post(prove_handler))
        .route("/api/pedersen-compare", post(pedersen_handler))
        .route("/api/gpu-stats", get(gpu_stats_handler))
        .with_state(state);

    let addr = "0.0.0.0:8080";
    println!("VortexSTARK dashboard: http://localhost:8080");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
