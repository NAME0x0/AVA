//! HTTP API Routes for AVA Engine
//!
//! Implements all HTTP endpoints that mirror the Python server API,
//! including WebSocket streaming support for real-time responses.

use crate::engine::models::*;
use crate::engine::state::AppState;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Json, State,
    },
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use sysinfo::System;
use tracing::{debug, error, info, warn};

/// Version constant
const VERSION: &str = "4.2.3";

/// Create the API router with all routes
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        // Health and status
        .route("/", get(root))
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/system", get(system_info))
        // Chat endpoints
        .route("/chat", post(chat))
        .route("/think", post(think))
        // State endpoints
        .route("/cognitive", get(cognitive_state))
        .route("/memory", get(memory_stats))
        .route("/belief", get(belief_state))
        .route("/stats", get(stats))
        // Control endpoints
        .route("/force_cortex", post(force_cortex))
        .route("/sleep", post(sleep_consolidate))
        .route("/clear_history", post(clear_history))
        // Tools
        .route("/tools", get(list_tools))
        // WebSocket streaming endpoint
        .route("/ws", get(websocket_handler))
        // Add state
        .with_state(state)
}

// =============================================================================
// Route Handlers
// =============================================================================

/// Root endpoint - simple welcome
async fn root() -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "AVA Neural Interface",
        "version": VERSION,
        "status": "running",
        "architecture": "Cortex-Medulla (Rust)"
    }))
}

/// Health check endpoint
/// Always returns 200 so the app can start - Ollama status is reported in response
async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let initialized = state.is_initialized().await;

    // Check Ollama connection (with timeout to prevent blocking)
    let (ollama_status, ollama_error) = match tokio::time::timeout(
        std::time::Duration::from_secs(2),
        state.engine.ollama().health_check(),
    )
    .await
    {
        Ok(Ok(true)) => ("connected".to_string(), None),
        Ok(Ok(false)) => (
            "disconnected".to_string(),
            Some("Ollama returned error".to_string()),
        ),
        Ok(Err(e)) => ("disconnected".to_string(), Some(e.to_string())),
        Err(_) => (
            "timeout".to_string(),
            Some("Ollama health check timed out".to_string()),
        ),
    };

    // Server is "healthy" as long as it's running - Ollama status is separate
    let status = if ollama_status == "connected" {
        if initialized {
            "healthy"
        } else {
            "initializing"
        }
    } else {
        "degraded" // Server works but AI features need Ollama
    };

    let response = HealthResponse {
        status: status.to_string(),
        service: "AVA Neural Interface".to_string(),
        version: VERSION.to_string(),
        ollama_status,
        ollama_error,
        uptime_seconds: state.uptime_seconds(),
    };

    // Always return 200 - the frontend can check ollama_status for details
    (StatusCode::OK, Json(response))
}

/// System status endpoint
async fn status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let (total_requests, cortex_requests) = state.engine.get_stats();

    // Get Ollama info
    let (ollama_connected, ollama_models) = match state.engine.ollama().list_models().await {
        Ok(models) => (true, models.iter().map(|m| m.name.clone()).collect()),
        Err(_) => (false, Vec::new()),
    };

    let active_model = state.engine.get_active_model().await;

    let response = SystemStatus {
        status: if state.is_initialized().await {
            "running"
        } else {
            "initializing"
        }
        .to_string(),
        version: VERSION.to_string(),
        python_version: "N/A (Rust native)".to_string(),
        platform: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        ollama_connected,
        ollama_models,
        active_model,
        uptime_seconds: state.uptime_seconds(),
        total_requests,
        cortex_requests,
    };

    Json(response)
}

/// Detailed system information
async fn system_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut sys = System::new_all();
    sys.refresh_all();

    let (total_requests, cortex_requests) = state.engine.get_stats();

    Json(serde_json::json!({
        "version": VERSION,
        "runtime": "Rust",
        "platform": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "uptime_seconds": state.uptime_seconds(),
        "memory": {
            "total_mb": sys.total_memory() / 1024 / 1024,
            "used_mb": sys.used_memory() / 1024 / 1024,
            "available_mb": sys.available_memory() / 1024 / 1024,
        },
        "cpu": {
            "cores": sys.cpus().len(),
            "name": sys.cpus().first().map(|c| c.brand()).unwrap_or("Unknown"),
        },
        "stats": {
            "total_requests": total_requests,
            "cortex_requests": cortex_requests,
        }
    }))
}

/// Chat endpoint - main interaction point
async fn chat(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> impl IntoResponse {
    debug!("Chat request: {:?}", request.message);

    if !state.is_initialized().await {
        // Try to initialize on first request
        if let Err(e) = state.initialize().await {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ChatResponse {
                    error: Some(format!("Engine not ready: {e}")),
                    ..Default::default()
                }),
            );
        }
    }

    let response = state.engine.process(&request).await;

    if response.error.is_some() {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
    } else {
        (StatusCode::OK, Json(response))
    }
}

/// Think endpoint - force deep reasoning
async fn think(
    State(state): State<Arc<AppState>>,
    Json(mut request): Json<ChatRequest>,
) -> impl IntoResponse {
    request.force_cortex = true;

    if !state.is_initialized().await {
        if let Err(e) = state.initialize().await {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ChatResponse {
                    error: Some(format!("Engine not ready: {e}")),
                    ..Default::default()
                }),
            );
        }
    }

    let response = state.engine.process(&request).await;

    if response.error.is_some() {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
    } else {
        (StatusCode::OK, Json(response))
    }
}

/// Get cognitive state
async fn cognitive_state(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cog_state = state.engine.get_cognitive_state().await;
    Json(cog_state)
}

/// Get memory statistics
async fn memory_stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let stats = state.engine.get_memory_stats().await;
    Json(stats)
}

/// Get belief state (Active Inference)
async fn belief_state(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let belief = state.engine.get_belief_state().await;
    Json(belief)
}

/// Get general statistics
async fn stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let (total, cortex) = state.engine.get_stats();

    Json(serde_json::json!({
        "total_requests": total,
        "cortex_requests": cortex,
        "medulla_requests": total - cortex,
        "cortex_ratio": if total > 0 { cortex as f64 / total as f64 } else { 0.0 },
        "uptime_seconds": state.uptime_seconds(),
    }))
}

/// Force Cortex for next response
async fn force_cortex() -> impl IntoResponse {
    // This is now handled per-request via ChatRequest.force_cortex
    Json(serde_json::json!({
        "status": "ok",
        "message": "Use force_cortex: true in chat request"
    }))
}

/// Trigger memory consolidation (sleep)
async fn sleep_consolidate(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Clear history as a simple "consolidation"
    state.engine.clear_history().await;

    Json(serde_json::json!({
        "status": "ok",
        "message": "Memory consolidation triggered (history cleared)"
    }))
}

/// Clear conversation history
async fn clear_history(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.engine.clear_history().await;

    Json(serde_json::json!({
        "status": "ok",
        "message": "Conversation history cleared"
    }))
}

/// List available tools
async fn list_tools() -> impl IntoResponse {
    // Basic built-in tools
    let tools = vec![
        serde_json::json!({
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "enabled": true
        }),
        serde_json::json!({
            "name": "current_time",
            "description": "Get the current date and time",
            "enabled": true
        }),
        serde_json::json!({
            "name": "web_search",
            "description": "Search the web for information",
            "enabled": false,
            "reason": "Requires API key configuration"
        }),
    ];

    Json(serde_json::json!({
        "tools": tools,
        "total": tools.len()
    }))
}

/// WebSocket streaming handler for real-time chat
/// Supports bidirectional communication with JSON messages
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_websocket(socket, state))
}

/// Handle WebSocket connection
async fn handle_websocket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();
    
    info!("WebSocket client connected");

    // Send connection acknowledgment
    let ack = serde_json::json!({
        "type": "connected",
        "version": VERSION,
        "message": "WebSocket connection established"
    });
    
    if let Err(e) = sender.send(Message::Text(ack.to_string())).await {
        error!("Failed to send WebSocket ack: {}", e);
        return;
    }

    // Process incoming messages
    while let Some(result) = receiver.next().await {
        match result {
            Ok(Message::Text(text)) => {
                debug!("WebSocket received: {}", text);
                
                // Parse the incoming message
                let request: Result<WebSocketRequest, _> = serde_json::from_str(&text);
                
                match request {
                    Ok(ws_request) => {
                        // Handle different request types
                        match ws_request.request_type.as_str() {
                            "chat" => {
                                handle_chat_ws(&mut sender, &state, ws_request).await;
                            }
                            "ping" => {
                                let pong = serde_json::json!({
                                    "type": "pong",
                                    "timestamp": chrono::Utc::now().to_rfc3339()
                                });
                                if let Err(e) = sender.send(Message::Text(pong.to_string())).await {
                                    warn!("Failed to send pong: {}", e);
                                    break;
                                }
                            }
                            "status" => {
                                let cog_state = state.engine.get_cognitive_state().await;
                                let status = serde_json::json!({
                                    "type": "status",
                                    "cognitive_state": cog_state,
                                    "timestamp": chrono::Utc::now().to_rfc3339()
                                });
                                if let Err(e) = sender.send(Message::Text(status.to_string())).await {
                                    warn!("Failed to send status: {}", e);
                                    break;
                                }
                            }
                            _ => {
                                let error = serde_json::json!({
                                    "type": "error",
                                    "message": format!("Unknown request type: {}", ws_request.request_type)
                                });
                                if let Err(e) = sender.send(Message::Text(error.to_string())).await {
                                    warn!("Failed to send error: {}", e);
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let error = serde_json::json!({
                            "type": "error",
                            "message": format!("Invalid JSON: {}", e)
                        });
                        if let Err(e) = sender.send(Message::Text(error.to_string())).await {
                            warn!("Failed to send parse error: {}", e);
                            break;
                        }
                    }
                }
            }
            Ok(Message::Close(_)) => {
                info!("WebSocket client disconnected");
                break;
            }
            Ok(Message::Ping(data)) => {
                if let Err(e) = sender.send(Message::Pong(data)).await {
                    warn!("Failed to send Pong: {}", e);
                    break;
                }
            }
            Ok(_) => {} // Ignore other message types
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
        }
    }
    
    info!("WebSocket connection closed");
}

/// Handle chat request over WebSocket with streaming
async fn handle_chat_ws(
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    state: &Arc<AppState>,
    request: WebSocketRequest,
) {
    let message = request.message.unwrap_or_default();
    
    // Send "thinking" status
    let thinking = serde_json::json!({
        "type": "thinking",
        "message": "Processing your request..."
    });
    if let Err(e) = sender.send(Message::Text(thinking.to_string())).await {
        error!("Failed to send thinking status: {}", e);
        return;
    }

    // Create chat request
    let chat_request = ChatRequest {
        message: message.clone(),
        conversation_id: request.conversation_id,
        force_cortex: request.force_cortex.unwrap_or(false),
        force_search: request.force_search.unwrap_or(false),
        stream: true,
        tools: request.tools.unwrap_or_default(),
    };

    // Process the request
    let response = state.engine.process(&chat_request).await;

    // For now, send the complete response (streaming can be enhanced later
    // by modifying the cognitive engine to yield chunks)
    let response_msg = serde_json::json!({
        "type": "response",
        "text": response.text,
        "used_cortex": response.used_cortex,
        "cognitive_state": response.cognitive_state,
        "confidence": response.confidence,
        "tools_used": response.tools_used,
        "response_time_ms": response.response_time_ms,
        "error": response.error,
        "done": true
    });

    if let Err(e) = sender.send(Message::Text(response_msg.to_string())).await {
        error!("Failed to send response: {}", e);
    }

    // Send completion message
    let complete = serde_json::json!({
        "type": "complete",
        "message": "Response complete"
    });
    if let Err(e) = sender.send(Message::Text(complete.to_string())).await {
        error!("Failed to send completion: {}", e);
    }
}
