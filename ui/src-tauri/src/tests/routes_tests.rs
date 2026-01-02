//! Tests for HTTP API routes
//!
//! These tests verify the HTTP API endpoints function correctly.
//! Uses axum's test utilities for endpoint testing.

use crate::engine::config::AppConfig;
use crate::engine::routes::create_router;
use crate::engine::state::AppState;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use std::sync::Arc;
use tower::ServiceExt; // for `oneshot`

/// Helper to create a test app state
fn create_test_state() -> Arc<AppState> {
    let config = AppConfig::default();
    Arc::new(AppState::new(config))
}

/// Test root endpoint returns 200
#[tokio::test]
async fn test_root_endpoint() {
    let state = create_test_state();
    let app = create_router(state);
    
    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
}

/// Test health endpoint returns 200
#[tokio::test]
async fn test_health_endpoint() {
    let state = create_test_state();
    let app = create_router(state);
    
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    
    // Health should always return 200, even if degraded
    assert_eq!(response.status(), StatusCode::OK);
}

/// Test status endpoint returns 200
#[tokio::test]
async fn test_status_endpoint() {
    let state = create_test_state();
    let app = create_router(state);
    
    let response = app
        .oneshot(
            Request::builder()
                .uri("/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
}

/// Test cognitive state endpoint
#[tokio::test]
async fn test_cognitive_endpoint() {
    let state = create_test_state();
    let app = create_router(state);
    
    let response = app
        .oneshot(
            Request::builder()
                .uri("/cognitive")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
}

/// Test memory stats endpoint
#[tokio::test]
async fn test_memory_endpoint() {
    let state = create_test_state();
    let app = create_router(state);
    
    let response = app
        .oneshot(
            Request::builder()
                .uri("/memory")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
}

/// Test stats endpoint
#[tokio::test]
async fn test_stats_endpoint() {
    let state = create_test_state();
    let app = create_router(state);
    
    let response = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
}

/// Test tools endpoint
#[tokio::test]
async fn test_tools_endpoint() {
    let state = create_test_state();
    let app = create_router(state);
    
    let response = app
        .oneshot(
            Request::builder()
                .uri("/tools")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
}

/// Test system info endpoint
#[tokio::test]
async fn test_system_endpoint() {
    let state = create_test_state();
    let app = create_router(state);
    
    let response = app
        .oneshot(
            Request::builder()
                .uri("/system")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
}

/// Test belief state endpoint
#[tokio::test]
async fn test_belief_endpoint() {
    let state = create_test_state();
    let app = create_router(state);
    
    let response = app
        .oneshot(
            Request::builder()
                .uri("/belief")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
}

/// Test 404 for unknown routes
#[tokio::test]
async fn test_not_found() {
    let state = create_test_state();
    let app = create_router(state);
    
    let response = app
        .oneshot(
            Request::builder()
                .uri("/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

/// Test chat endpoint requires POST
#[tokio::test]
async fn test_chat_requires_post() {
    let state = create_test_state();
    let app = create_router(state);
    
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/chat")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    
    // GET on POST-only route should return 405 Method Not Allowed
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

/// Test chat endpoint with valid JSON body
#[tokio::test]
async fn test_chat_with_body() {
    let state = create_test_state();
    let app = create_router(state);
    
    let body = serde_json::json!({
        "message": "Hello, AVA!"
    });
    
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/chat")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    
    // Chat will likely return 200 even if Ollama isn't running (with error in body)
    // or 500 if something goes wrong
    assert!(
        response.status() == StatusCode::OK || response.status() == StatusCode::INTERNAL_SERVER_ERROR,
        "Chat should return either OK or error, got {:?}",
        response.status()
    );
}

/// Test clear_history endpoint
#[tokio::test]
async fn test_clear_history() {
    let state = create_test_state();
    let app = create_router(state);
    
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/clear_history")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
}
