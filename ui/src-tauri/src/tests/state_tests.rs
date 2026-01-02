//! Tests for application state management
//!
//! These tests verify:
//! - State initialization
//! - Thread-safe state access
//! - Configuration loading and defaults

use crate::state::AppState;

/// Test that AppState::new() creates valid default state
#[test]
fn test_app_state_creation() {
    let state = AppState::new();
    // Should create without panicking
    assert!(true, "AppState creation succeeded");
}

/// Test that AppState::default() works
#[test]
fn test_app_state_default() {
    let state = AppState::default();
    // Should create without panicking
    assert!(true, "AppState default creation succeeded");
}

/// Test async access to backend_url
#[tokio::test]
async fn test_backend_url_access() {
    let state = AppState::new();

    let url = state.backend_url.lock().await.clone();

    assert!(url.starts_with("http://"), "Backend URL should be HTTP");
    assert!(
        url.contains("localhost") || url.contains("127.0.0.1"),
        "Backend URL should be localhost"
    );
}

/// Test async access to connected state
#[tokio::test]
async fn test_connected_state_access() {
    let state = AppState::new();

    let connected = *state.connected.lock().await;

    // Initial state should be disconnected
    assert!(!connected, "Initial state should be disconnected");
}

/// Test that state can be modified
#[tokio::test]
async fn test_state_modification() {
    let state = AppState::new();

    // Modify connected state
    {
        let mut connected = state.connected.lock().await;
        *connected = true;
    }

    // Verify modification persisted
    let connected = *state.connected.lock().await;
    assert!(
        connected,
        "Connected state should be true after modification"
    );
}

/// Test concurrent access to state
#[tokio::test]
async fn test_concurrent_state_access() {
    use std::sync::Arc;

    let state = Arc::new(AppState::new());

    // Spawn multiple tasks that access state concurrently
    let mut handles = vec![];

    for i in 0..10 {
        let state_clone = state.clone();
        handles.push(tokio::spawn(async move {
            // Read backend URL
            let _url = state_clone.backend_url.lock().await.clone();

            // Read connected state
            let _connected = *state_clone.connected.lock().await;

            i
        }));
    }

    // All tasks should complete without deadlock
    for handle in handles {
        let result = handle.await;
        assert!(result.is_ok(), "Task should complete successfully");
    }
}
