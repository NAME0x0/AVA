//! Application State Management
//!
//! Thread-safe state for the Tauri application

use tokio::sync::Mutex;

pub struct AppState {
    pub backend_url: Mutex<String>,
    pub connected: Mutex<bool>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            backend_url: Mutex::new("http://localhost:8080".to_string()),
            connected: Mutex::new(false),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
