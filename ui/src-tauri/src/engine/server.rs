//! Embedded HTTP Server for AVA
//!
//! Runs an HTTP server within the Tauri application, eliminating
//! the need for a separate Python backend process.

use crate::engine::config::AppConfig;
use crate::engine::routes::create_router;
use crate::engine::state::AppState;
use axum::http::{header, HeaderValue, Method};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::oneshot;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{error, info};

/// Server handle for managing the embedded server
pub struct ServerHandle {
    pub shutdown_tx: oneshot::Sender<()>,
    pub port: u16,
}

/// Start the embedded HTTP server
///
/// Returns a handle that can be used to shut down the server gracefully.
pub async fn start_embedded_server(config: AppConfig) -> Result<ServerHandle, String> {
    let port = config.server.port;
    let host = config.server.host.clone();
    
    // Create application state
    let state = Arc::new(AppState::new(config.clone()));
    
    // Initialize the engine (connect to Ollama)
    info!("Initializing AVA engine...");
    if let Err(e) = state.initialize().await {
        // Log but continue - we'll retry on first request
        error!("Initial engine initialization failed: {}. Will retry on first request.", e);
    }
    
    // Create router with CORS
    let cors = CorsLayer::new()
        .allow_origin([
            "http://localhost:3000".parse::<HeaderValue>().unwrap(),
            "http://127.0.0.1:3000".parse::<HeaderValue>().unwrap(),
            "tauri://localhost".parse::<HeaderValue>().unwrap(),
            "https://tauri.localhost".parse::<HeaderValue>().unwrap(),
        ])
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION, header::ACCEPT])
        .allow_credentials(true);
    
    let app = create_router(state)
        .layer(cors)
        .layer(TraceLayer::new_for_http());
    
    // Parse address
    let addr: SocketAddr = format!("{}:{}", host, port)
        .parse()
        .map_err(|e| format!("Invalid address: {}", e))?;
    
    // Create shutdown channel
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    
    // Start server
    info!("Starting AVA embedded server on http://{}", addr);
    
    tokio::spawn(async move {
        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) => {
                error!("Failed to bind to {}: {}", addr, e);
                return;
            }
        };
        
        info!("AVA server listening on http://{}", addr);
        
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
                info!("Server shutdown signal received");
            })
            .await
            .unwrap_or_else(|e| error!("Server error: {}", e));
        
        info!("AVA server stopped");
    });
    
    // Give the server a moment to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    Ok(ServerHandle { shutdown_tx, port })
}

/// Check if a port is available
pub fn is_port_available(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}

/// Find an available port starting from the given port
pub fn find_available_port(start_port: u16) -> u16 {
    for port in start_port..start_port + 100 {
        if is_port_available(port) {
            return port;
        }
    }
    start_port // Fall back to original if none found
}
