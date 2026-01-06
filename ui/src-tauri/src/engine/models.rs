//! Data models for AVA Engine
//!
//! These structs mirror the Python dataclasses but in idiomatic Rust.

// Allow dead code for API response structures that may not be used currently
// but are part of the complete Ollama API contract
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cognitive state of the engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CognitiveState {
    #[default]
    Flow,
    Hesitation,
    Confusion,
    Creative,
    Verifying,
}

impl std::fmt::Display for CognitiveState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CognitiveState::Flow => write!(f, "FLOW"),
            CognitiveState::Hesitation => write!(f, "HESITATION"),
            CognitiveState::Confusion => write!(f, "CONFUSION"),
            CognitiveState::Creative => write!(f, "CREATIVE"),
            CognitiveState::Verifying => write!(f, "VERIFYING"),
        }
    }
}

/// Processing mode for the engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ProcessingMode {
    #[default]
    Medulla, // Fast, reflexive
    Cortex, // Deep, thoughtful
    Hybrid, // Both (verify with Cortex)
}

/// Chat request from the client
#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub force_cortex: bool,
    #[serde(default)]
    pub force_search: bool,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Vec<String>,
}

/// WebSocket request message
#[derive(Debug, Clone, Deserialize)]
pub struct WebSocketRequest {
    /// Type of request: "chat", "ping", "status"
    #[serde(rename = "type")]
    pub request_type: String,
    /// Chat message (for "chat" type)
    pub message: Option<String>,
    /// Conversation ID for context
    #[serde(default)]
    pub conversation_id: Option<String>,
    /// Force Cortex processing
    #[serde(default)]
    pub force_cortex: Option<bool>,
    /// Force search-first routing
    #[serde(default)]
    pub force_search: Option<bool>,
    /// List of tools to enable
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

/// Chat response to the client
#[derive(Debug, Clone, Serialize)]
pub struct ChatResponse {
    /// Main response text - serialized as both "text" and "response" for compatibility
    #[serde(rename = "response")]
    pub text: String,
    pub used_cortex: bool,
    pub cognitive_state: String,
    pub confidence: f32,
    pub tools_used: Vec<String>,
    pub response_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Default for ChatResponse {
    fn default() -> Self {
        Self {
            text: String::new(),
            used_cortex: false,
            cognitive_state: "FLOW".to_string(),
            confidence: 0.8,
            tools_used: Vec::new(),
            response_time_ms: 0.0,
            error: None,
        }
    }
}

/// Health check response
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
    pub ollama_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ollama_error: Option<String>,
    pub uptime_seconds: u64,
}

/// System status response
#[derive(Debug, Clone, Serialize)]
pub struct SystemStatus {
    pub status: String,
    pub version: String,
    pub python_version: String, // Keep for compatibility, will be "N/A (Rust)"
    pub platform: String,
    pub ollama_connected: bool,
    pub ollama_models: Vec<String>,
    pub active_model: String,
    pub uptime_seconds: u64,
    pub total_requests: u64,
    pub cortex_requests: u64,
}

/// Cognitive state response
#[derive(Debug, Clone, Serialize)]
pub struct CognitiveStateResponse {
    pub label: String, // Changed from 'state' to match frontend
    pub entropy: f32,
    pub varentropy: f32,
    pub surprise: f32,
    pub confidence: f32,
    #[serde(rename = "shouldThink")]
    pub should_think: bool,
    #[serde(rename = "shouldUseTools")]
    pub should_use_tools: bool,
    pub complexity: f32,
    pub processing_mode: String,
    pub last_response_time_ms: f64,
}

/// Memory stats response
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    pub conversation_turns: usize,
    pub total_tokens_processed: u64,
    pub context_window_used: usize,
    pub context_window_max: usize,
    pub episodic_memories: usize,
    pub semantic_memories: usize,
    // Additional fields for frontend compatibility
    pub total_memories: usize,
    pub memory_updates: u64,
    pub avg_surprise: f32,
    pub backend: String,
    pub memory_utilization: f32,
}

/// Belief state response (Active Inference)
#[derive(Debug, Clone, Serialize)]
pub struct BeliefStateResponse {
    #[serde(rename = "currentState")]
    pub current_state: String,
    #[serde(rename = "stateDistribution")]
    pub state_distribution: std::collections::HashMap<String, f32>,
    #[serde(rename = "policyDistribution")]
    pub policy_distribution: std::collections::HashMap<String, f32>,
    #[serde(rename = "freeEnergy")]
    pub free_energy: f32,
}

/// Tool definition for the API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: HashMap<String, ToolParameter>,
    pub enabled: bool,
}

/// Tool parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    #[serde(rename = "type")]
    pub param_type: String,
    pub description: String,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
}

/// Streaming chunk for WebSocket/SSE
#[derive(Debug, Clone, Serialize)]
pub struct StreamChunk {
    #[serde(rename = "type")]
    pub chunk_type: String, // "token", "thinking", "complete", "error"
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<StreamMetadata>,
}

/// Metadata for stream chunks
#[derive(Debug, Clone, Serialize)]
pub struct StreamMetadata {
    pub tokens_generated: u32,
    pub time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cognitive_state: Option<String>,
}

/// Conversation message for history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String, // "user", "assistant", "system"
    pub content: String,
    pub timestamp: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MessageMetadata>,
}

/// Message metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    pub used_cortex: bool,
    pub cognitive_state: String,
    pub response_time_ms: f64,
    pub tools_used: Vec<String>,
}

/// Ollama model info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    pub size: u64,
    pub digest: String,
    pub modified_at: String,
}

/// Ollama chat message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
}

/// Ollama chat request
#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
}

/// Ollama generation options
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
}

/// Ollama chat response (non-streaming)
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub message: OllamaChatMessage,
    pub done: bool,
    #[serde(default)]
    pub total_duration: u64,
    #[serde(default)]
    pub load_duration: u64,
    #[serde(default)]
    pub prompt_eval_count: u32,
    #[serde(default)]
    pub eval_count: u32,
    #[serde(default)]
    pub eval_duration: u64,
}

/// Ollama streaming response chunk
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaStreamChunk {
    pub model: String,
    pub message: OllamaChatMessage,
    pub done: bool,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u32>,
}

/// Ollama tags response (list models)
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaModelInfo>,
}

/// Ollama model info from tags endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelInfo {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub modified_at: String,
}

/// Error response
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl ErrorResponse {
    pub fn new(error: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            code: code.into(),
            details: None,
        }
    }

    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}
