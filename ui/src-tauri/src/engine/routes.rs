//! HTTP API Routes for AVA Engine
//!
//! Implements all HTTP endpoints that mirror the Python server API.

use crate::engine::models::*;
use crate::engine::state::AppState;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use sysinfo::System;
use tracing::debug;

/// Version constant
const VERSION: &str = "3.3.3";

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
        
        // WebSocket stub (returns JSON with info, not actual WS yet)
        .route("/ws", get(websocket_stub))
        
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
        state.engine.ollama().health_check()
    ).await {
        Ok(Ok(true)) => ("connected".to_string(), None),
        Ok(Ok(false)) => ("disconnected".to_string(), Some("Ollama returned error".to_string())),
        Ok(Err(e)) => ("disconnected".to_string(), Some(e.to_string())),
        Err(_) => ("timeout".to_string(), Some("Ollama health check timed out".to_string())),
    };
    
    // Server is "healthy" as long as it's running - Ollama status is separate
    let status = if ollama_status == "connected" {
        if initialized { "healthy" } else { "initializing" }
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
        status: if state.is_initialized().await { "running" } else { "initializing" }.to_string(),
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
                    error: Some(format!("Engine not ready: {}", e)),
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
                    error: Some(format!("Engine not ready: {}", e)),
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

/// WebSocket stub - returns info that WebSocket is not yet implemented in Rust backend
/// The frontend should gracefully handle this and use polling instead
async fn websocket_stub() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": "WebSocket not implemented",
            "message": "Real-time WebSocket streaming is not yet implemented in the Rust backend. Please use HTTP polling via /cognitive endpoint.",
            "alternative": "/cognitive"
        }))
    )
}
