//! Cognitive Engine for AVA
//!
//! Implements the Cortex-Medulla architecture in Rust:
//! - Medulla: Fast, reflexive responses
//! - Cortex: Deep, thoughtful reasoning (currently routes to larger model)
//! - Intelligent routing based on query complexity

use crate::engine::models::*;
use crate::engine::ollama::{OllamaClient, OllamaError};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

/// System prompt for AVA
const SYSTEM_PROMPT: &str = r#"You are AVA, an advanced AI assistant with a biomimetic dual-brain architecture.

Core Traits:
- Accurate: Prioritize correctness over speed
- Helpful: Provide actionable, relevant information
- Concise: Be clear and to the point
- Honest: Acknowledge limitations and uncertainty

When responding:
1. Think step-by-step for complex problems
2. Use tools when available and helpful
3. Ask clarifying questions if needed
4. Verify important information

You have access to the current date/time and can perform calculations."#;

/// Cognitive engine that handles all LLM interactions
pub struct CognitiveEngine {
    ollama: OllamaClient,
    config: EngineConfig,
    conversation_history: Arc<RwLock<Vec<ConversationMessage>>>,
    
    // Statistics
    total_requests: AtomicU64,
    cortex_requests: AtomicU64,
    
    // Current state
    current_state: Arc<RwLock<CognitiveStateInfo>>,
    
    // Active model
    active_model: Arc<RwLock<String>>,
}

/// Internal cognitive state info
#[derive(Clone)]
struct CognitiveStateInfo {
    state: CognitiveState,
    entropy: f32,
    varentropy: f32,
    surprise: f32,
    confidence: f32,
    complexity: f32,
    last_response_time_ms: f64,
}

impl Default for CognitiveStateInfo {
    fn default() -> Self {
        Self {
            state: CognitiveState::Flow,
            entropy: 0.5,
            varentropy: 0.3,
            surprise: 0.2,
            confidence: 0.8,
            complexity: 0.3,
            last_response_time_ms: 0.0,
        }
    }
}

/// Engine configuration
#[derive(Clone)]
pub struct EngineConfig {
    pub ollama_host: String,
    pub fast_model: String,
    pub deep_model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub context_window: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            ollama_host: "http://localhost:11434".to_string(),
            fast_model: "gemma3:4b".to_string(),
            deep_model: "qwen2.5:32b".to_string(),
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            context_window: 8192,
        }
    }
}

impl CognitiveEngine {
    /// Create a new cognitive engine
    pub fn new(config: EngineConfig) -> Self {
        let ollama = OllamaClient::with_config(&config.ollama_host, 120);
        
        Self {
            ollama,
            config: config.clone(),
            conversation_history: Arc::new(RwLock::new(Vec::new())),
            total_requests: AtomicU64::new(0),
            cortex_requests: AtomicU64::new(0),
            current_state: Arc::new(RwLock::new(CognitiveStateInfo::default())),
            active_model: Arc::new(RwLock::new(config.fast_model)),
        }
    }

    /// Initialize the engine and verify Ollama connection
    pub async fn initialize(&self) -> Result<(), OllamaError> {
        info!("Initializing AVA Cognitive Engine...");
        
        // Check Ollama health
        self.ollama.health_check().await?;
        info!("Ollama connection verified");
        
        // Verify model availability
        let models = self.ollama.list_models().await?;
        let model_names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        info!("Available models: {:?}", model_names);
        
        // Select best available model
        let selected_model = self.select_model(&model_names).await;
        *self.active_model.write().await = selected_model.clone();
        info!("Selected model: {}", selected_model);
        
        info!("AVA Cognitive Engine initialized successfully");
        Ok(())
    }

    /// Select the best available model
    async fn select_model(&self, available: &[&str]) -> String {
        // Preferred models in order
        let preferences = [
            &self.config.fast_model,
            "gemma3:4b",
            "gemma2:9b",
            "llama3.2:3b",
            "llama3.1:8b",
            "qwen2.5:7b",
            "mistral:7b",
        ];
        
        for pref in preferences {
            for model in available {
                if model.starts_with(pref) || *model == pref {
                    return model.to_string();
                }
            }
        }
        
        // Fall back to first available
        available.first().map(|s| s.to_string()).unwrap_or_else(|| "gemma3:4b".to_string())
    }

    /// Process a chat message
    pub async fn process(&self, request: &ChatRequest) -> ChatResponse {
        let start = Instant::now();
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        
        // Determine processing mode
        let use_cortex = request.force_cortex || self.should_use_cortex(&request.message).await;
        
        if use_cortex {
            self.cortex_requests.fetch_add(1, Ordering::Relaxed);
        }
        
        // Get model to use
        let model = if use_cortex {
            self.config.deep_model.clone()
        } else {
            self.active_model.read().await.clone()
        };
        
        debug!("Processing message with model: {} (cortex={})", model, use_cortex);
        
        // Build messages for Ollama
        let messages = self.build_messages(&request.message).await;
        
        // Configure options
        let options = Some(OllamaOptions {
            temperature: Some(self.config.temperature),
            top_p: Some(self.config.top_p),
            num_predict: Some(self.config.max_tokens as i32),
            repeat_penalty: Some(1.1),
            top_k: Some(40),
        });
        
        // Call Ollama
        match self.ollama.chat(&model, messages, options).await {
            Ok(response) => {
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                
                // Update state
                let mut state = self.current_state.write().await;
                state.last_response_time_ms = elapsed;
                state.confidence = self.calculate_confidence(&response);
                state.state = self.determine_cognitive_state(&response);
                
                // Store in history
                self.add_to_history(&request.message, &response.message.content).await;
                
                ChatResponse {
                    text: response.message.content,
                    used_cortex: use_cortex,
                    cognitive_state: state.state.to_string(),
                    confidence: state.confidence,
                    tools_used: Vec::new(),
                    response_time_ms: elapsed,
                    error: None,
                }
            }
            Err(e) => {
                error!("Ollama chat error: {}", e);
                ChatResponse {
                    text: String::new(),
                    used_cortex: use_cortex,
                    cognitive_state: "CONFUSION".to_string(),
                    confidence: 0.0,
                    tools_used: Vec::new(),
                    response_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                    error: Some(e.to_string()),
                }
            }
        }
    }

    /// Determine if we should use Cortex (deep thinking)
    async fn should_use_cortex(&self, message: &str) -> bool {
        // Simple heuristics for now
        let complexity_indicators = [
            "explain", "analyze", "compare", "contrast", "evaluate",
            "synthesize", "create", "design", "develop", "implement",
            "why", "how does", "what if", "prove", "derive",
            "code", "program", "algorithm", "debug",
        ];
        
        let lower = message.to_lowercase();
        
        // Check for complexity indicators
        let has_complexity = complexity_indicators.iter().any(|&ind| lower.contains(ind));
        
        // Check message length (longer messages often need more thought)
        let is_long = message.len() > 200;
        
        // Check for multi-part questions
        let has_multiple_questions = message.matches('?').count() > 1;
        
        has_complexity || is_long || has_multiple_questions
    }

    /// Build messages array for Ollama
    async fn build_messages(&self, user_message: &str) -> Vec<OllamaChatMessage> {
        let mut messages = vec![
            OllamaChatMessage {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
        ];
        
        // Add conversation history (last N turns)
        let history = self.conversation_history.read().await;
        let history_limit = 10; // Last 10 messages
        let start_idx = history.len().saturating_sub(history_limit);
        
        for msg in &history[start_idx..] {
            messages.push(OllamaChatMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
            });
        }
        
        // Add current message
        messages.push(OllamaChatMessage {
            role: "user".to_string(),
            content: user_message.to_string(),
        });
        
        messages
    }

    /// Add a conversation turn to history
    async fn add_to_history(&self, user_msg: &str, assistant_msg: &str) {
        let mut history = self.conversation_history.write().await;
        
        let now = chrono::Utc::now().timestamp();
        
        history.push(ConversationMessage {
            role: "user".to_string(),
            content: user_msg.to_string(),
            timestamp: now,
            metadata: None,
        });
        
        history.push(ConversationMessage {
            role: "assistant".to_string(),
            content: assistant_msg.to_string(),
            timestamp: now,
            metadata: None,
        });
        
        // Trim history if too long
        let max_history = 50;
        if history.len() > max_history {
            *history = history[history.len() - max_history..].to_vec();
        }
    }

    /// Calculate confidence based on response
    fn calculate_confidence(&self, _response: &OllamaChatResponse) -> f32 {
        // For now, return a reasonable default
        // In a full implementation, this would analyze the response
        0.85
    }

    /// Determine cognitive state based on response
    fn determine_cognitive_state(&self, response: &OllamaChatResponse) -> CognitiveState {
        let content = &response.message.content;
        
        // Check for uncertainty markers
        let uncertainty_markers = ["i'm not sure", "i think", "possibly", "maybe", "uncertain"];
        let has_uncertainty = uncertainty_markers.iter().any(|&m| content.to_lowercase().contains(m));
        
        if has_uncertainty {
            CognitiveState::Hesitation
        } else if content.contains("?") && content.len() < 100 {
            CognitiveState::Confusion
        } else {
            CognitiveState::Flow
        }
    }

    /// Get current cognitive state
    pub async fn get_cognitive_state(&self) -> CognitiveStateResponse {
        let state = self.current_state.read().await;
        let should_think = state.entropy > 3.5 || state.varentropy > 2.5;
        let should_use_tools = state.surprise > 0.7;
        
        CognitiveStateResponse {
            label: state.state.to_string(),
            entropy: state.entropy,
            varentropy: state.varentropy,
            surprise: state.surprise,
            confidence: state.confidence,
            should_think,
            should_use_tools,
            complexity: state.complexity,
            processing_mode: if self.cortex_requests.load(Ordering::Relaxed) > 0 {
                "cortex".to_string()
            } else {
                "medulla".to_string()
            },
            last_response_time_ms: state.last_response_time_ms,
        }
    }

    /// Get memory stats
    pub async fn get_memory_stats(&self) -> MemoryStats {
        let history = self.conversation_history.read().await;
        let state = self.current_state.read().await;
        
        // Estimate tokens (rough: 1 token ≈ 4 chars)
        let total_chars: usize = history.iter().map(|m| m.content.len()).sum();
        let estimated_tokens = total_chars / 4;
        let utilization = if self.config.context_window > 0 {
            (estimated_tokens as f32 / self.config.context_window as f32).min(1.0)
        } else {
            0.0
        };
        
        MemoryStats {
            conversation_turns: history.len() / 2,
            total_tokens_processed: estimated_tokens as u64,
            context_window_used: estimated_tokens.min(self.config.context_window),
            context_window_max: self.config.context_window,
            episodic_memories: history.len(),
            semantic_memories: 0,
            // Additional fields for frontend
            total_memories: history.len(),
            memory_updates: self.total_requests.load(Ordering::Relaxed),
            avg_surprise: state.surprise,
            backend: "rust-ollama".to_string(),
            memory_utilization: utilization,
        }
    }

    /// Get belief state (Active Inference metrics)
    pub async fn get_belief_state(&self) -> BeliefStateResponse {
        let state = self.current_state.read().await;
        let total = self.total_requests.load(Ordering::Relaxed);
        let cortex = self.cortex_requests.load(Ordering::Relaxed);
        
        // Calculate free energy as a measure of prediction error / surprise
        // Higher entropy + higher surprise = higher free energy
        let free_energy = state.entropy * 0.3 + state.surprise * 0.5 + state.varentropy * 0.2;
        
        // Build state distribution based on cognitive state
        let mut state_distribution = std::collections::HashMap::new();
        match state.state {
            CognitiveState::Flow => {
                state_distribution.insert("FLOW".to_string(), 0.7);
                state_distribution.insert("HESITATION".to_string(), 0.2);
                state_distribution.insert("CONFUSION".to_string(), 0.1);
            }
            CognitiveState::Hesitation => {
                state_distribution.insert("FLOW".to_string(), 0.3);
                state_distribution.insert("HESITATION".to_string(), 0.5);
                state_distribution.insert("CONFUSION".to_string(), 0.2);
            }
            CognitiveState::Confusion => {
                state_distribution.insert("FLOW".to_string(), 0.1);
                state_distribution.insert("HESITATION".to_string(), 0.3);
                state_distribution.insert("CONFUSION".to_string(), 0.6);
            }
            _ => {
                state_distribution.insert("FLOW".to_string(), 0.5);
                state_distribution.insert("HESITATION".to_string(), 0.3);
                state_distribution.insert("CONFUSION".to_string(), 0.2);
            }
        }
        
        // Build policy distribution based on usage patterns
        let mut policy_distribution = std::collections::HashMap::new();
        let medulla_ratio = if total > 0 { (total - cortex) as f32 / total as f32 } else { 0.8 };
        let cortex_ratio = if total > 0 { cortex as f32 / total as f32 } else { 0.2 };
        
        policy_distribution.insert("MEDULLA_REFLEX".to_string(), medulla_ratio * 0.6);
        policy_distribution.insert("MEDULLA_SEARCH".to_string(), medulla_ratio * 0.4);
        policy_distribution.insert("CORTEX_DEEP".to_string(), cortex_ratio * 0.7);
        policy_distribution.insert("CORTEX_VERIFY".to_string(), cortex_ratio * 0.3);
        
        BeliefStateResponse {
            current_state: state.state.to_string(),
            state_distribution,
            policy_distribution,
            free_energy,
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> (u64, u64) {
        (
            self.total_requests.load(Ordering::Relaxed),
            self.cortex_requests.load(Ordering::Relaxed),
        )
    }

    /// Get the Ollama client reference
    pub fn ollama(&self) -> &OllamaClient {
        &self.ollama
    }

    /// Get active model
    pub async fn get_active_model(&self) -> String {
        self.active_model.read().await.clone()
    }

    /// Clear conversation history
    pub async fn clear_history(&self) {
        self.conversation_history.write().await.clear();
    }

    /// Force Cortex for next response
    #[allow(dead_code)]
    pub fn force_cortex(&self) {
        // This is handled per-request via ChatRequest.force_cortex
    }
}
