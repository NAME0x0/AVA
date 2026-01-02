//! Application State for AVA Engine
//!
//! Manages shared state across the HTTP server and Tauri app.

use crate::engine::cognitive::CognitiveEngine;
use crate::engine::config::AppConfig;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::info;

/// Shared application state
#[allow(dead_code)]
pub struct AppState {
    /// The cognitive engine for processing
    pub engine: Arc<CognitiveEngine>,
    
    /// Application configuration
    pub config: AppConfig,
    
    /// Server start time
    pub start_time: Instant,
    
    /// Initialization status
    pub initialized: Arc<RwLock<bool>>,
    
    /// Last error message
    pub last_error: Arc<RwLock<Option<String>>>,
}

impl AppState {
    /// Create a new application state
    pub fn new(config: AppConfig) -> Self {
        let engine_config = config.to_engine_config();
        let engine = Arc::new(CognitiveEngine::new(engine_config));
        
        Self {
            engine,
            config,
            start_time: Instant::now(),
            initialized: Arc::new(RwLock::new(false)),
            last_error: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize the state (connect to Ollama, etc.)
    pub async fn initialize(&self) -> Result<(), String> {
        info!("Initializing AVA application state...");
        
        match self.engine.initialize().await {
            Ok(()) => {
                *self.initialized.write().await = true;
                *self.last_error.write().await = None;
                info!("AVA state initialized successfully");
                Ok(())
            }
            Err(e) => {
                let error_msg = e.to_string();
                *self.last_error.write().await = Some(error_msg.clone());
                Err(error_msg)
            }
        }
    }

    /// Check if initialized
    pub async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Get the last error
    #[allow(dead_code)]
    pub async fn get_last_error(&self) -> Option<String> {
        self.last_error.read().await.clone()
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new(AppConfig::default())
    }
}
