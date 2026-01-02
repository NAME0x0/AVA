//! Tests for data models and serialization
//!
//! These tests verify:
//! - Correct JSON serialization/deserialization
//! - Default values for optional fields
//! - Compatibility with API responses

use crate::commands::{ChatResponse, CognitiveState, MemoryStats, SystemState};

/// Test ChatResponse serialization roundtrip
#[test]
fn test_chat_response_serialization() {
    let response = ChatResponse {
        response: "Hello, world!".to_string(),
        cognitive_state: Some("focused".to_string()),
        surprise: Some(0.5),
        tokens_generated: Some(10),
        used_cortex: false,
        confidence: 0.95,
        response_time_ms: 150.0,
    };
    
    let json = serde_json::to_string(&response).expect("Serialization should succeed");
    let parsed: ChatResponse = serde_json::from_str(&json).expect("Deserialization should succeed");
    
    assert_eq!(parsed.response, "Hello, world!");
    assert_eq!(parsed.cognitive_state, Some("focused".to_string()));
    assert_eq!(parsed.surprise, Some(0.5));
}

/// Test ChatResponse with 'text' field alias (backwards compatibility)
#[test]
fn test_chat_response_text_alias() {
    let json = r#"{
        "text": "Hello from text field!",
        "used_cortex": true,
        "confidence": 0.8,
        "response_time_ms": 200.0
    }"#;
    
    let parsed: ChatResponse = serde_json::from_str(json).expect("Should parse with 'text' alias");
    
    assert_eq!(parsed.response, "Hello from text field!");
}

/// Test ChatResponse with minimal fields
#[test]
fn test_chat_response_minimal() {
    let json = r#"{
        "response": "Minimal response"
    }"#;
    
    let parsed: ChatResponse = serde_json::from_str(json).expect("Should parse minimal response");
    
    assert_eq!(parsed.response, "Minimal response");
    assert_eq!(parsed.cognitive_state, None);
    assert_eq!(parsed.surprise, None);
    assert_eq!(parsed.tokens_generated, None);
    assert!(!parsed.used_cortex);
    assert_eq!(parsed.confidence, 0.0);
    assert_eq!(parsed.response_time_ms, 0.0);
}

/// Test CognitiveState default values
#[test]
fn test_cognitive_state_default() {
    let state = CognitiveState::default();
    
    assert!(state.label.is_empty());
    assert_eq!(state.entropy, 0.0);
    assert_eq!(state.varentropy, 0.0);
    assert_eq!(state.confidence, 0.0);
    assert_eq!(state.surprise, 0.0);
    assert!(!state.should_use_tools);
    assert!(!state.should_think);
}

/// Test CognitiveState serialization
#[test]
fn test_cognitive_state_serialization() {
    let state = CognitiveState {
        label: "focused".to_string(),
        entropy: 1.5,
        varentropy: 0.3,
        confidence: 0.9,
        surprise: 0.2,
        should_use_tools: true,
        should_think: false,
    };
    
    let json = serde_json::to_string(&state).expect("Serialization should succeed");
    let parsed: CognitiveState = serde_json::from_str(&json).expect("Deserialization should succeed");
    
    assert_eq!(parsed.label, "focused");
    assert_eq!(parsed.entropy, 1.5);
    assert!(parsed.should_use_tools);
}

/// Test SystemState default values
#[test]
fn test_system_state_default() {
    let state = SystemState::default();
    
    assert!(!state.connected);
    assert!(state.system_state.is_empty());
    assert!(state.active_component.is_empty());
    assert_eq!(state.uptime_seconds, 0);
    assert_eq!(state.total_interactions, 0);
    assert_eq!(state.cortex_invocations, 0);
    assert_eq!(state.avg_response_time_ms, 0.0);
}

/// Test SystemState serialization
#[test]
fn test_system_state_serialization() {
    let state = SystemState {
        connected: true,
        system_state: "running".to_string(),
        active_component: "medulla".to_string(),
        uptime_seconds: 3600,
        total_interactions: 100,
        cortex_invocations: 10,
        avg_response_time_ms: 250.5,
    };
    
    let json = serde_json::to_string(&state).expect("Serialization should succeed");
    let parsed: SystemState = serde_json::from_str(&json).expect("Deserialization should succeed");
    
    assert!(parsed.connected);
    assert_eq!(parsed.system_state, "running");
    assert_eq!(parsed.uptime_seconds, 3600);
}

/// Test MemoryStats default values
#[test]
fn test_memory_stats_default() {
    let stats = MemoryStats::default();
    
    assert_eq!(stats.total_memories, 0);
    assert_eq!(stats.memory_updates, 0);
    assert_eq!(stats.avg_surprise, 0.0);
    assert!(stats.backend.is_empty());
    assert_eq!(stats.memory_utilization, 0.0);
}

/// Test MemoryStats serialization
#[test]
fn test_memory_stats_serialization() {
    let stats = MemoryStats {
        total_memories: 500,
        memory_updates: 50,
        avg_surprise: 0.45,
        backend: "titans".to_string(),
        memory_utilization: 0.75,
    };
    
    let json = serde_json::to_string(&stats).expect("Serialization should succeed");
    let parsed: MemoryStats = serde_json::from_str(&json).expect("Deserialization should succeed");
    
    assert_eq!(parsed.total_memories, 500);
    assert_eq!(parsed.backend, "titans");
    assert_eq!(parsed.memory_utilization, 0.75);
}

/// Test that structs can be serialized and re-parsed (roundtrip)
#[test]
fn test_default_serialization_roundtrip() {
    // Test that Default values can be serialized and re-parsed
    let cognitive = CognitiveState::default();
    let json = serde_json::to_string(&cognitive).expect("Should serialize");
    let parsed: CognitiveState = serde_json::from_str(&json).expect("Should deserialize");
    assert!(parsed.label.is_empty());
    
    let system = SystemState::default();
    let json = serde_json::to_string(&system).expect("Should serialize");
    let parsed: SystemState = serde_json::from_str(&json).expect("Should deserialize");
    assert!(!parsed.connected);
    
    let memory = MemoryStats::default();
    let json = serde_json::to_string(&memory).expect("Should serialize");
    let parsed: MemoryStats = serde_json::from_str(&json).expect("Should deserialize");
    assert_eq!(parsed.total_memories, 0);
}
