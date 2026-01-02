//! Ollama Client for AVA Engine
//!
//! Provides async communication with the Ollama API for LLM inference.
//! Supports both streaming and non-streaming chat completions.

// Allow dead code for API methods that are part of the complete Ollama API contract
// but may not be currently used
#![allow(dead_code)]

use crate::engine::models::*;
use futures_util::StreamExt;
use reqwest::Client;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, warn};

/// Default Ollama host
pub const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";

/// Default timeout for Ollama requests
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Ollama client errors
#[derive(Error, Debug)]
pub enum OllamaError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Request failed: {0}")]
    RequestFailed(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Timeout")]
    Timeout,

    #[error("Stream error: {0}")]
    StreamError(String),
}

/// Ollama client for LLM inference
#[derive(Clone)]
pub struct OllamaClient {
    client: Client,
    host: String,
    timeout: Duration,
}

impl OllamaClient {
    /// Create a new Ollama client with default settings
    pub fn new() -> Self {
        Self::with_config(DEFAULT_OLLAMA_HOST, DEFAULT_TIMEOUT_SECS)
    }

    /// Create a new Ollama client with custom host and timeout
    pub fn with_config(host: &str, timeout_secs: u64) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .pool_max_idle_per_host(5)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            host: host.trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(timeout_secs),
        }
    }

    /// Check if Ollama is running and accessible
    pub async fn health_check(&self) -> Result<bool, OllamaError> {
        let url = format!("{}/", self.host);
        
        match self.client.get(&url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(e) => {
                if e.is_timeout() {
                    Err(OllamaError::Timeout)
                } else if e.is_connect() {
                    Err(OllamaError::ConnectionFailed(format!(
                        "Cannot connect to Ollama at {}. Is it running?",
                        self.host
                    )))
                } else {
                    Err(OllamaError::ConnectionFailed(e.to_string()))
                }
            }
        }
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<OllamaModelInfo>, OllamaError> {
        let url = format!("{}/api/tags", self.host);
        
        let response = self.client.get(&url).send().await.map_err(|e| {
            if e.is_timeout() {
                OllamaError::Timeout
            } else if e.is_connect() {
                OllamaError::ConnectionFailed(format!("Cannot connect to {}", self.host))
            } else {
                OllamaError::RequestFailed(e.to_string())
            }
        })?;

        if !response.status().is_success() {
            return Err(OllamaError::RequestFailed(format!(
                "Status: {}",
                response.status()
            )));
        }

        let tags: OllamaTagsResponse = response.json().await.map_err(|e| {
            OllamaError::InvalidResponse(format!("Failed to parse response: {e}"))
        })?;

        Ok(tags.models)
    }

    /// Check if a specific model is available
    pub async fn has_model(&self, model_name: &str) -> Result<bool, OllamaError> {
        let models = self.list_models().await?;
        Ok(models.iter().any(|m| {
            m.name == model_name || m.name.starts_with(&format!("{model_name}:"))
        }))
    }

    /// Send a chat completion request (non-streaming)
    pub async fn chat(
        &self,
        model: &str,
        messages: Vec<OllamaChatMessage>,
        options: Option<OllamaOptions>,
    ) -> Result<OllamaChatResponse, OllamaError> {
        let url = format!("{}/api/chat", self.host);
        
        let request = OllamaChatRequest {
            model: model.to_string(),
            messages,
            stream: false,
            options,
        };

        debug!("Sending chat request to Ollama: model={}", model);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    OllamaError::Timeout
                } else {
                    OllamaError::RequestFailed(e.to_string())
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            
            if status.as_u16() == 404 {
                return Err(OllamaError::ModelNotFound(model.to_string()));
            }
            
            return Err(OllamaError::RequestFailed(format!(
                "Status: {status}, Body: {body}"
            )));
        }

        let chat_response: OllamaChatResponse = response.json().await.map_err(|e| {
            OllamaError::InvalidResponse(format!("Failed to parse response: {e}"))
        })?;

        debug!(
            "Chat response received: tokens={}, time={}ms",
            chat_response.eval_count,
            chat_response.total_duration / 1_000_000
        );

        Ok(chat_response)
    }

    /// Send a streaming chat completion request
    pub async fn chat_stream(
        &self,
        model: &str,
        messages: Vec<OllamaChatMessage>,
        options: Option<OllamaOptions>,
    ) -> Result<mpsc::Receiver<Result<OllamaStreamChunk, OllamaError>>, OllamaError> {
        let url = format!("{}/api/chat", self.host);
        
        let request = OllamaChatRequest {
            model: model.to_string(),
            messages,
            stream: true,
            options,
        };

        debug!("Sending streaming chat request to Ollama: model={}", model);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    OllamaError::Timeout
                } else {
                    OllamaError::RequestFailed(e.to_string())
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            if status.as_u16() == 404 {
                return Err(OllamaError::ModelNotFound(model.to_string()));
            }
            return Err(OllamaError::RequestFailed(format!("Status: {status}")));
        }

        // Create a channel for streaming chunks
        let (tx, rx) = mpsc::channel(100);

        // Spawn a task to process the stream
        let mut stream = response.bytes_stream();
        
        tokio::spawn(async move {
            let mut buffer = String::new();
            
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                        
                        // Process complete lines
                        while let Some(newline_pos) = buffer.find('\n') {
                            let line = buffer[..newline_pos].trim().to_string();
                            buffer = buffer[newline_pos + 1..].to_string();
                            
                            if line.is_empty() {
                                continue;
                            }
                            
                            match serde_json::from_str::<OllamaStreamChunk>(&line) {
                                Ok(chunk) => {
                                    let is_done = chunk.done;
                                    if tx.send(Ok(chunk)).await.is_err() {
                                        return; // Receiver dropped
                                    }
                                    if is_done {
                                        return;
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to parse stream chunk: {} - line: {}", e, line);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(OllamaError::StreamError(e.to_string()))).await;
                        return;
                    }
                }
            }
        });

        Ok(rx)
    }

    /// Generate embeddings for text
    pub async fn embeddings(&self, model: &str, text: &str) -> Result<Vec<f32>, OllamaError> {
        let url = format!("{}/api/embeddings", self.host);
        
        let request = serde_json::json!({
            "model": model,
            "prompt": text
        });

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| OllamaError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            return Err(OllamaError::RequestFailed(format!(
                "Status: {}",
                response.status()
            )));
        }

        #[derive(serde::Deserialize)]
        struct EmbeddingResponse {
            embedding: Vec<f32>,
        }

        let embed_response: EmbeddingResponse = response.json().await.map_err(|e| {
            OllamaError::InvalidResponse(format!("Failed to parse embeddings: {e}"))
        })?;

        Ok(embed_response.embedding)
    }

    /// Get the host URL
    pub fn host(&self) -> &str {
        &self.host
    }
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = OllamaClient::new();
        assert_eq!(client.host(), DEFAULT_OLLAMA_HOST);
    }

    #[tokio::test]
    async fn test_custom_host() {
        let client = OllamaClient::with_config("http://custom:11434", 60);
        assert_eq!(client.host(), "http://custom:11434");
    }
}
