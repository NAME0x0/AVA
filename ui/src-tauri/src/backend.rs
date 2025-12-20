//! Backend Connection Management
//!
//! Monitors connection to Python backend and broadcasts state changes

use tauri::{AppHandle, Manager};
use std::time::Duration;

/// Start the background connection monitor
pub async fn start_connection_monitor(app: AppHandle) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    
    loop {
        let state = app.state::<crate::state::AppState>();
        let url = state.backend_url.lock().await.clone();
        
        let connected = match client.get(format!("{}/health", url)).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        };
        
        // Update state
        *state.connected.lock().await = connected;
        
        // Emit event to frontend
        let _ = app.emit_all("backend-status", serde_json::json!({
            "connected": connected,
            "url": url,
        }));
        
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}
