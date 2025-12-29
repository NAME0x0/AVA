//! Tauri Command Handlers
//!
//! IPC bridge between the Next.js frontend and AVA Python backend

use crate::state::AppState;
use serde::{Deserialize, Serialize};
use tauri::State;

/// Response from the chat endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub response: String,
    pub cognitive_state: Option<CognitiveState>,
    pub surprise: Option<f32>,
    pub tokens_generated: Option<u32>,
    pub used_cortex: bool,
    pub policy_selected: String,
    pub response_time_ms: f64,
}

/// Cognitive state from the Medulla
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CognitiveState {
    pub label: String,
    pub entropy: f32,
    pub varentropy: f32,
    pub confidence: f32,
    pub surprise: f32,
    pub should_use_tools: bool,
    pub should_think: bool,
}

/// System state overview
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemState {
    pub connected: bool,
    pub system_state: String,
    pub active_component: String, // "medulla" | "cortex" | "idle"
    pub uptime_seconds: u64,
    pub total_interactions: u64,
    pub cortex_invocations: u64,
    pub avg_response_time_ms: f64,
}

/// Memory statistics from Titans
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_memories: u64,
    pub memory_updates: u64,
    pub avg_surprise: f32,
    pub backend: String,
    pub memory_utilization: f32,
}

/// Send a message to AVA
#[tauri::command]
pub async fn send_message(
    message: String,
    state: State<'_, AppState>,
) -> Result<ChatResponse, String> {
    let base_url = state.backend_url.lock().await.clone();

    let client = reqwest::Client::new();
    let start = std::time::Instant::now();

    let resp = client
        .post(format!("{base_url}/chat"))
        .json(&serde_json::json!({ "message": message }))
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    let elapsed = start.elapsed().as_millis() as f64;

    let mut data: ChatResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {e}"))?;

    data.response_time_ms = elapsed;

    Ok(data)
}

/// Get current system state
#[tauri::command]
pub async fn get_system_state(state: State<'_, AppState>) -> Result<SystemState, String> {
    let base_url = state.backend_url.lock().await.clone();

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base_url}/system/state"))
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    resp.json()
        .await
        .map_err(|e| format!("Failed to parse response: {e}"))
}

/// Get cognitive state from Medulla
#[tauri::command]
pub async fn get_cognitive_state(state: State<'_, AppState>) -> Result<CognitiveState, String> {
    let base_url = state.backend_url.lock().await.clone();

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base_url}/cognitive"))
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    resp.json()
        .await
        .map_err(|e| format!("Failed to parse response: {e}"))
}

/// Get Titans memory statistics
#[tauri::command]
pub async fn get_memory_stats(state: State<'_, AppState>) -> Result<MemoryStats, String> {
    let base_url = state.backend_url.lock().await.clone();

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base_url}/memory"))
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    resp.json()
        .await
        .map_err(|e| format!("Failed to parse response: {e}"))
}

/// Force Cortex invocation for next response
#[tauri::command]
pub async fn force_cortex(state: State<'_, AppState>) -> Result<(), String> {
    let base_url = state.backend_url.lock().await.clone();

    let client = reqwest::Client::new();
    client
        .post(format!("{base_url}/force_cortex"))
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    Ok(())
}

/// Force sleep cycle
#[tauri::command]
pub async fn force_sleep(state: State<'_, AppState>) -> Result<(), String> {
    let base_url = state.backend_url.lock().await.clone();

    let client = reqwest::Client::new();
    client
        .post(format!("{base_url}/sleep"))
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    Ok(())
}

/// Set the backend URL
#[tauri::command]
pub async fn set_backend_url(url: String, state: State<'_, AppState>) -> Result<(), String> {
    let mut backend_url = state.backend_url.lock().await;
    *backend_url = url;
    Ok(())
}

/// Simple greeting for testing
#[tauri::command]
pub fn greet(name: &str) -> String {
    format!("Hello, {name}! AVA Cortex-Medulla system is ready.")
}
