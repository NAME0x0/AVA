//! Configuration for AVA Engine
//!
//! Loads and manages configuration from files and environment.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::info;

pub use crate::engine::cognitive::EngineConfig;

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub engine: EngineConfigFile,
    #[serde(default)]
    pub ollama: OllamaConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            engine: EngineConfigFile::default(),
            ollama: OllamaConfig::default(),
        }
    }
}

/// HTTP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub cors_origins: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8085,
            cors_origins: vec![
                "http://localhost:3000".to_string(),
                "http://127.0.0.1:3000".to_string(),
                "tauri://localhost".to_string(),
            ],
        }
    }
}

/// Engine configuration from file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfigFile {
    pub fast_model: String,
    pub deep_model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub context_window: usize,
}

impl Default for EngineConfigFile {
    fn default() -> Self {
        Self {
            fast_model: "gemma3:4b".to_string(),
            deep_model: "qwen2.5:32b".to_string(),
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            context_window: 8192,
        }
    }
}

impl From<EngineConfigFile> for EngineConfig {
    fn from(file: EngineConfigFile) -> Self {
        Self {
            ollama_host: "http://localhost:11434".to_string(),
            fast_model: file.fast_model,
            deep_model: file.deep_model,
            max_tokens: file.max_tokens,
            temperature: file.temperature,
            top_p: file.top_p,
            context_window: file.context_window,
        }
    }
}

/// Ollama-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub host: String,
    pub timeout_secs: u64,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            host: "http://localhost:11434".to_string(),
            timeout_secs: 120,
        }
    }
}

impl AppConfig {
    /// Load configuration from file or use defaults
    pub fn load() -> Self {
        // Try to load from config file
        if let Some(config_path) = Self::find_config_file() {
            if let Ok(content) = std::fs::read_to_string(&config_path) {
                if let Ok(config) = toml::from_str(&content) {
                    info!("Loaded configuration from {:?}", config_path);
                    return config;
                }
            }
        }
        
        // Check environment variables
        let mut config = AppConfig::default();
        
        if let Ok(host) = std::env::var("OLLAMA_HOST") {
            config.ollama.host = host;
        }
        
        if let Ok(port) = std::env::var("AVA_PORT") {
            if let Ok(p) = port.parse() {
                config.server.port = p;
            }
        }
        
        if let Ok(model) = std::env::var("AVA_MODEL") {
            config.engine.fast_model = model;
        }
        
        config
    }

    /// Find configuration file in standard locations
    fn find_config_file() -> Option<PathBuf> {
        let locations = [
            // Portable: next to executable
            std::env::current_exe().ok()?.parent()?.join("config").join("ava.toml"),
            std::env::current_exe().ok()?.parent()?.join("ava.toml"),
            // Current directory
            PathBuf::from("config/ava.toml"),
            PathBuf::from("ava.toml"),
        ];
        
        for path in locations {
            if path.exists() {
                return Some(path);
            }
        }
        
        None
    }

    /// Get the data directory for portable mode
    pub fn data_dir() -> PathBuf {
        // Portable: next to executable
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(parent) = exe_path.parent() {
                let data_dir = parent.join("data");
                if data_dir.exists() || std::fs::create_dir_all(&data_dir).is_ok() {
                    return data_dir;
                }
            }
        }
        
        // Fallback to user data directory
        if let Some(dirs) = directories::ProjectDirs::from("com", "ava", "AVA") {
            return dirs.data_dir().to_path_buf();
        }
        
        // Last resort: current directory
        PathBuf::from("data")
    }

    /// Convert to engine config
    pub fn to_engine_config(&self) -> EngineConfig {
        EngineConfig {
            ollama_host: self.ollama.host.clone(),
            fast_model: self.engine.fast_model.clone(),
            deep_model: self.engine.deep_model.clone(),
            max_tokens: self.engine.max_tokens,
            temperature: self.engine.temperature,
            top_p: self.engine.top_p,
            context_window: self.engine.context_window,
        }
    }
}

/// Generate a default configuration file
pub fn generate_default_config() -> String {
    let config = AppConfig::default();
    toml::to_string_pretty(&config).unwrap_or_default()
}
