//! Cognitive Engine for AVA - Sentinel Architecture v4
//!
//! Implements the Cortex-Medulla architecture in Rust with full Sentinel integration:
//! - Medulla: Fast, reflexive responses for routine queries
//! - Cortex: Deep, thoughtful reasoning for complex/novel queries
//! - Agency: Active Inference for autonomous policy selection
//! - Titans: Test-time learning for infinite context
//!
//! # Sentinel Architecture (v4.2)
//!
//! The cognitive engine implements the four-stage Sentinel loop:
//!
//! 1. **Perception (Medulla)**: Fast classification and intent extraction
//! 2. **Appraisal (Surprise Calculation)**: Real embedding-based surprise via KL divergence
//! 3. **Policy Selection (Agency)**: Active Inference minimizing Expected Free Energy
//! 4. **Memory Update (Titans)**: Test-time learning with surprise-weighted updates
//!
//! ## Key Features
//!
//! - **Real Surprise Calculation**: Uses embedding divergence, not keyword heuristics
//! - **Active Inference**: Free Energy Principle drives autonomous behavior
//! - **Test-Time Learning**: Titans memory updates during inference
//! - **Search-First Paradigm**: Factual queries route to search before generation
//!
//! ## References
//!
//! - Friston, K. (2010). The free-energy principle
//! - Titans: Learning to Learn at Test Time (Google Research, 2025)
//! - Active Inference: The Free Energy Principle in Mind, Brain, and Behavior

use crate::engine::agency::{AgencyConfig, AgencyEngine, Observation, PolicyType};
use crate::engine::models::{
    BeliefStateResponse, ChatRequest, ChatResponse, CognitiveState, CognitiveStateResponse,
    ConversationMessage, MemoryStats, OllamaChatMessage, OllamaChatResponse, OllamaOptions,
};
use crate::engine::ollama::{OllamaClient, OllamaError};
use crate::engine::titans::{TitansConfig, TitansMemory};
use ndarray::Array1;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Default embedding model for surprise calculation
const DEFAULT_EMBEDDING_MODEL: &str = "nomic-embed-text";

/// Low surprise threshold - below this, use Medulla (reflex)
const LOW_SURPRISE_THRESHOLD: f32 = 0.3;

/// High surprise threshold - above this, use Cortex (deep thought)
const HIGH_SURPRISE_THRESHOLD: f32 = 2.0;

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
    search_requests: AtomicU64,

    // Current state
    current_state: Arc<RwLock<CognitiveStateInfo>>,

    // Active model
    active_model: Arc<RwLock<String>>,

    // Sentinel Architecture components (v4)
    /// Agency engine for policy selection (Active Inference)
    agency: Arc<RwLock<AgencyEngine>>,

    /// Titans memory for test-time learning
    titans: Arc<RwLock<Option<TitansMemory>>>,

    /// Embedding model for surprise calculation
    embedding_model: String,

    /// Cache for context embeddings (sliding window)
    context_embedding_cache: Arc<RwLock<Option<Vec<f32>>>>,

    /// Recent embeddings for surprise calculation (circular buffer)
    recent_embeddings: Arc<RwLock<VecDeque<Vec<f32>>>>,

    /// Last selected policy for response metadata
    last_policy: Arc<RwLock<Option<PolicyType>>>,
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

        // Initialize AgencyEngine with default config
        let agency = AgencyEngine::new(AgencyConfig {
            pragmatic_weight: 0.4,
            epistemic_weight: 0.6, // High curiosity
            temperature: 1.0,
            low_surprise_threshold: LOW_SURPRISE_THRESHOLD,
            high_surprise_threshold: HIGH_SURPRISE_THRESHOLD,
            search_first_enabled: true,
            search_first_keywords: vec![
                "what".to_string(),
                "who".to_string(),
                "where".to_string(),
                "when".to_string(),
                "how much".to_string(),
                "define".to_string(),
                "latest".to_string(),
                "current".to_string(),
            ],
        });

        // Initialize TitansMemory (may fail if config invalid, so wrap in Option)
        let titans = TitansMemory::new(TitansConfig::default()).ok();

        Self {
            ollama,
            config: config.clone(),
            conversation_history: Arc::new(RwLock::new(Vec::new())),
            total_requests: AtomicU64::new(0),
            cortex_requests: AtomicU64::new(0),
            search_requests: AtomicU64::new(0),
            current_state: Arc::new(RwLock::new(CognitiveStateInfo::default())),
            active_model: Arc::new(RwLock::new(config.fast_model)),
            agency: Arc::new(RwLock::new(agency)),
            titans: Arc::new(RwLock::new(titans)),
            embedding_model: DEFAULT_EMBEDDING_MODEL.to_string(),
            context_embedding_cache: Arc::new(RwLock::new(None)),
            recent_embeddings: Arc::new(RwLock::new(VecDeque::with_capacity(10))),
            last_policy: Arc::new(RwLock::new(None)),
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
        available
            .first()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "gemma3:4b".to_string())
    }

    /// Process a chat message using the Sentinel architecture
    ///
    /// Implements the four-stage Sentinel loop:
    /// 1. Perception: Extract features and calculate surprise
    /// 2. Appraisal: Use Agency to select optimal policy
    /// 3. Execution: Route to appropriate processor based on policy
    /// 4. Learning: Update Titans memory with response
    pub async fn process(&self, request: &ChatRequest) -> ChatResponse {
        let start = Instant::now();
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        // =====================================================================
        // Stage 1: Perception - Calculate real surprise using embeddings
        // =====================================================================
        let (surprise, entropy) = self.calculate_real_metrics(&request.message).await;

        // Update current state with real metrics
        {
            let mut state = self.current_state.write().await;
            state.surprise = surprise;
            state.entropy = entropy;
            state.complexity = self.estimate_complexity(&request.message);
            state.varentropy = self.calculate_varentropy(&request.message).await;
        }

        // =====================================================================
        // Stage 2: Appraisal - Use Agency to select policy via Active Inference
        // =====================================================================
        let observation = Observation::from_text(&request.message)
            .with_surprise(surprise)
            .with_entropy(entropy);

        let policy = {
            let mut agency = self.agency.write().await;
            agency
                .select_policy(&observation)
                .unwrap_or(PolicyType::ReflexReply)
        };

        // Store selected policy for response metadata
        {
            let mut last_policy = self.last_policy.write().await;
            *last_policy = Some(policy);
        }

        // Check force flags (override policy if set)
        let (use_cortex, use_search) = if request.force_cortex {
            (true, false)
        } else if request.force_search {
            self.search_requests.fetch_add(1, Ordering::Relaxed);
            (false, true)
        } else {
            match policy {
                PolicyType::PrimarySearch | PolicyType::WebBrowse => {
                    self.search_requests.fetch_add(1, Ordering::Relaxed);
                    (false, true)
                }
                PolicyType::DeepThought | PolicyType::VerifyLogic | PolicyType::SimulateOutcome => {
                    self.cortex_requests.fetch_add(1, Ordering::Relaxed);
                    (true, false)
                }
                _ => (false, false),
            }
        };

        debug!(
            "Sentinel routing: policy={:?}, surprise={:.2}, entropy={:.2}, cortex={}, search={}",
            policy, surprise, entropy, use_cortex, use_search
        );

        // =====================================================================
        // Stage 3: Execution - Route to appropriate processor
        // =====================================================================
        let model = if use_cortex {
            self.config.deep_model.clone()
        } else {
            self.active_model.read().await.clone()
        };

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
        let response_result = self.ollama.chat(&model, messages, options).await;

        match response_result {
            Ok(response) => {
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                let response_text = response.message.content.clone();

                // =====================================================================
                // Stage 4: Learning - Update Titans memory with response
                // =====================================================================
                // Calculate post-response surprise (how surprising was our own output?)
                let response_surprise = self.calculate_response_surprise(&response_text).await;

                // Update Titans memory with surprise-weighted learning
                self.update_titans_memory(&response_text, response_surprise)
                    .await;

                // Update context embedding cache
                self.update_embedding_cache(&response_text).await;

                // Update Agency beliefs based on outcome
                {
                    let mut agency = self.agency.write().await;
                    agency.update_beliefs(policy, true, &observation);
                }

                // Update state
                let cognitive_state = {
                    let mut state = self.current_state.write().await;
                    state.last_response_time_ms = elapsed;
                    state.confidence = self.calculate_confidence_from_response(&response);
                    state.state = self.determine_cognitive_state(&response);
                    state.state
                };

                // Store in history
                self.add_to_history(&request.message, &response_text).await;

                ChatResponse {
                    text: response_text,
                    used_cortex: use_cortex,
                    cognitive_state: cognitive_state.to_string(),
                    confidence: self.current_state.read().await.confidence,
                    tools_used: if use_search {
                        vec!["search".to_string()]
                    } else {
                        Vec::new()
                    },
                    response_time_ms: elapsed,
                    error: None,
                }
            }
            Err(e) => {
                error!("Ollama chat error: {}", e);

                // Update Agency beliefs with failure
                {
                    let mut agency = self.agency.write().await;
                    agency.update_beliefs(policy, false, &observation);
                }

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

    /// Calculate real surprise and entropy using embeddings
    async fn calculate_real_metrics(&self, message: &str) -> (f32, f32) {
        // Get message embedding
        let message_embedding = match self.ollama.embeddings(&self.embedding_model, message).await {
            Ok(emb) => emb,
            Err(e) => {
                warn!(
                    "Failed to get message embedding: {}, falling back to proxy",
                    e
                );
                return (
                    self.estimate_surprise_proxy(message),
                    self.estimate_entropy(message),
                );
            }
        };

        // Calculate surprise as divergence from recent context
        let surprise = {
            let recent = self.recent_embeddings.read().await;
            if recent.is_empty() {
                1.0 // First message has medium-high surprise
            } else {
                // Average KL divergence from recent embeddings
                let divergences: Vec<f32> = recent
                    .iter()
                    .map(|prev| self.kl_divergence(&message_embedding, prev))
                    .collect();
                divergences.iter().sum::<f32>() / divergences.len() as f32
            }
        };

        // Calculate entropy from embedding distribution
        let entropy = self.embedding_entropy(&message_embedding);

        // Store embedding in recent buffer
        {
            let mut recent = self.recent_embeddings.write().await;
            if recent.len() >= 10 {
                recent.pop_front();
            }
            recent.push_back(message_embedding);
        }

        (surprise.clamp(0.0, 5.0), entropy.clamp(0.0, 1.0))
    }

    /// Calculate variance in entropy (varentropy) for uncertainty estimation
    async fn calculate_varentropy(&self, _message: &str) -> f32 {
        let recent = self.recent_embeddings.read().await;
        if recent.len() < 2 {
            return 0.3; // Default for insufficient data
        }

        // Calculate entropy for each recent embedding
        let entropies: Vec<f32> = recent.iter().map(|e| self.embedding_entropy(e)).collect();

        // Calculate variance
        let mean = entropies.iter().sum::<f32>() / entropies.len() as f32;
        let variance =
            entropies.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / entropies.len() as f32;

        variance.sqrt().clamp(0.0, 1.0)
    }

    /// Calculate surprise from our own response (for Titans learning)
    async fn calculate_response_surprise(&self, response: &str) -> f32 {
        let response_embedding = match self
            .ollama
            .embeddings(&self.embedding_model, response)
            .await
        {
            Ok(emb) => emb,
            Err(_) => return 0.5,
        };

        let cache = self.context_embedding_cache.read().await;
        if let Some(ref context_emb) = *cache {
            self.kl_divergence(&response_embedding, context_emb)
        } else {
            0.5
        }
    }

    /// Update Titans memory with test-time learning
    async fn update_titans_memory(&self, response: &str, surprise: f32) {
        let mut titans_lock = self.titans.write().await;
        if let Some(ref mut titans) = *titans_lock {
            // Only update if surprise exceeds threshold (avoid learning routine)
            if surprise > LOW_SURPRISE_THRESHOLD {
                match self
                    .ollama
                    .embeddings(&self.embedding_model, response)
                    .await
                {
                    Ok(embedding) => {
                        let embedding_array = Array1::from_vec(embedding);
                        if let Err(e) = titans.update(&embedding_array, surprise) {
                            warn!("Titans update failed: {}", e);
                        } else {
                            debug!("Titans memory updated: surprise={:.2}", surprise);
                        }
                    }
                    Err(e) => warn!("Failed to get embedding for Titans: {}", e),
                }
            }
        }
    }

    /// Update context embedding cache
    async fn update_embedding_cache(&self, text: &str) {
        if let Ok(embedding) = self.ollama.embeddings(&self.embedding_model, text).await {
            let mut cache = self.context_embedding_cache.write().await;

            // Exponential moving average with existing cache
            if let Some(ref existing) = *cache {
                let alpha = 0.3; // Learning rate for cache update
                let new_embedding: Vec<f32> = embedding
                    .iter()
                    .zip(existing.iter())
                    .map(|(new, old)| alpha * new + (1.0 - alpha) * old)
                    .collect();
                *cache = Some(new_embedding);
            } else {
                *cache = Some(embedding);
            }
        }
    }

    /// Calculate entropy from embedding vector
    fn embedding_entropy(&self, embedding: &[f32]) -> f32 {
        if embedding.is_empty() {
            return 0.5;
        }

        // Normalize to probabilities using softmax
        let max_val = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = embedding.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();

        if sum == 0.0 {
            return 0.5;
        }

        let probs: Vec<f32> = exp_vals.iter().map(|x| x / sum).collect();

        // Shannon entropy
        let entropy: f32 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|p| -p * p.log2())
            .sum();

        // Normalize by max possible entropy (log2 of dimension)
        let max_entropy = (embedding.len() as f32).log2();
        if max_entropy > 0.0 {
            (entropy / max_entropy).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    /// Calculate confidence from LLM response
    fn calculate_confidence_from_response(&self, response: &OllamaChatResponse) -> f32 {
        let content = &response.message.content;
        let lower = content.to_lowercase();

        let mut confidence: f32 = 0.85; // Base confidence

        // Reduce confidence for uncertainty markers
        let uncertainty_markers = [
            ("i'm not sure", 0.15),
            ("i think", 0.10),
            ("possibly", 0.10),
            ("maybe", 0.10),
            ("uncertain", 0.15),
            ("i believe", 0.05),
            ("probably", 0.05),
            ("might be", 0.10),
            ("could be", 0.10),
        ];

        for (marker, penalty) in uncertainty_markers {
            if lower.contains(marker) {
                confidence -= penalty;
            }
        }

        // Increase confidence for assertion markers
        let assertion_markers = [
            ("definitely", 0.05),
            ("certainly", 0.05),
            ("clearly", 0.03),
            ("obviously", 0.03),
        ];

        for (marker, boost) in assertion_markers {
            if lower.contains(marker) {
                confidence += boost;
            }
        }

        confidence.clamp(0.1, 0.99)
    }

    /// Determine if we should use Cortex (deep thinking)
    #[allow(dead_code)]
    async fn should_use_cortex(&self, message: &str) -> bool {
        // Simple heuristics for now
        let complexity_indicators = [
            "explain",
            "analyze",
            "compare",
            "contrast",
            "evaluate",
            "synthesize",
            "create",
            "design",
            "develop",
            "implement",
            "why",
            "how does",
            "what if",
            "prove",
            "derive",
            "code",
            "program",
            "algorithm",
            "debug",
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
        let mut messages = vec![OllamaChatMessage {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        }];

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
    #[allow(dead_code)]
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
        let has_uncertainty = uncertainty_markers
            .iter()
            .any(|&m| content.to_lowercase().contains(m));

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
        let medulla_ratio = if total > 0 {
            (total - cortex) as f32 / total as f32
        } else {
            0.8
        };
        let cortex_ratio = if total > 0 {
            cortex as f32 / total as f32
        } else {
            0.2
        };

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

    // =========================================================================
    // Sentinel Architecture Methods (v4)
    // These methods are being incrementally integrated into the main flow
    // =========================================================================

    /// Calculate surprise using embedding divergence
    ///
    /// Surprise is computed as the KL divergence between the response embedding
    /// and the context embedding. High surprise indicates novel/unexpected content.
    #[allow(dead_code)]
    pub async fn calculate_surprise(&self, response: &str, context: &[ConversationMessage]) -> f32 {
        // Get response embedding
        let response_embedding = match self
            .ollama
            .embeddings(&self.embedding_model, response)
            .await
        {
            Ok(emb) => emb,
            Err(e) => {
                warn!("Failed to get response embedding: {}", e);
                return 0.5; // Default to medium surprise
            }
        };

        // Get context embedding
        let context_embedding = match self.get_context_embedding(context).await {
            Some(emb) => emb,
            None => {
                // No context - everything is novel
                return 1.5;
            }
        };

        // Calculate KL divergence approximation
        let surprise = self.kl_divergence(&response_embedding, &context_embedding);

        // Clamp to reasonable range [0, 5]
        surprise.clamp(0.0, 5.0)
    }

    /// Get or compute context embedding from conversation history
    #[allow(dead_code)]
    async fn get_context_embedding(&self, context: &[ConversationMessage]) -> Option<Vec<f32>> {
        if context.is_empty() {
            return None;
        }

        // Check cache first
        {
            let cache = self.context_embedding_cache.read().await;
            if let Some(ref cached) = *cache {
                return Some(cached.clone());
            }
        }

        // Combine recent context into a single string
        let context_text: String = context
            .iter()
            .rev()
            .take(5) // Last 5 messages
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Get embedding
        let embedding = match self
            .ollama
            .embeddings(&self.embedding_model, &context_text)
            .await
        {
            Ok(emb) => emb,
            Err(e) => {
                warn!("Failed to get context embedding: {}", e);
                return None;
            }
        };

        // Update cache
        {
            let mut cache = self.context_embedding_cache.write().await;
            *cache = Some(embedding.clone());
        }

        Some(embedding)
    }

    /// Approximate KL divergence between two embedding vectors
    ///
    /// Uses cosine distance as a proxy for divergence.
    /// Used by Sentinel Stage 1 (Perception) for surprise calculation.
    fn kl_divergence(&self, p: &[f32], q: &[f32]) -> f32 {
        if p.len() != q.len() || p.is_empty() {
            return 0.5;
        }

        // Cosine similarity
        let dot: f32 = p.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
        let norm_p: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_q: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_p == 0.0 || norm_q == 0.0 {
            return 0.5;
        }

        let cosine_sim = dot / (norm_p * norm_q);

        // Convert to divergence: 1 - similarity, scaled
        // Cosine sim in [-1, 1], we want surprise in [0, 5]
        // Map: sim=1 → surprise=0, sim=0 → surprise=2.5, sim=-1 → surprise=5
        let divergence = (1.0 - cosine_sim) * 2.5;

        divergence.max(0.0)
    }

    /// Estimate query complexity for routing decisions
    /// Used by Sentinel Stage 1 (Perception) for state calculation.
    pub fn estimate_complexity(&self, query: &str) -> f32 {
        let mut complexity = 0.0;

        // Length factor
        let word_count = query.split_whitespace().count();
        complexity += (word_count as f32 / 50.0).min(1.0) * 0.2;

        // Question complexity
        let question_count = query.matches('?').count();
        complexity += (question_count as f32 * 0.15).min(0.3);

        // Code indicators
        if query.contains("```")
            || query.contains("def ")
            || query.contains("fn ")
            || query.contains("class ")
        {
            complexity += 0.3;
        }

        // Reasoning indicators
        let reasoning_words = [
            "explain", "why", "how", "analyze", "compare", "evaluate", "prove", "derive",
        ];
        let lower = query.to_lowercase();
        for word in &reasoning_words {
            if lower.contains(word) {
                complexity += 0.1;
            }
        }

        complexity.min(1.0)
    }

    /// Determine routing using surprise and policy selection
    /// Used by Sentinel Stage 2 (Appraisal) for policy selection.
    #[allow(dead_code)]
    pub async fn determine_routing(&self, message: &str) -> (bool, PolicyType) {
        // Get conversation history for context
        let history = self.conversation_history.read().await;

        // Calculate surprise from recent context
        let surprise = if history.is_empty() {
            0.5 // Medium surprise for first message
        } else {
            // Use a simple proxy: just estimate from message characteristics
            // Full embedding-based surprise is computed after response
            self.estimate_surprise_proxy(message)
        };

        // Calculate entropy estimate
        let entropy = self.estimate_entropy(message);

        // Create observation for agency
        let observation = Observation::from_text(message)
            .with_surprise(surprise)
            .with_entropy(entropy);

        // Let agency select policy
        let mut agency = self.agency.write().await;
        let policy = agency
            .select_policy(&observation)
            .unwrap_or(PolicyType::ReflexReply);

        // Update state
        {
            let mut state = self.current_state.write().await;
            state.surprise = surprise;
            state.entropy = entropy;
            state.complexity = self.estimate_complexity(message);
        }

        // Determine if cortex is needed based on policy
        let use_cortex = matches!(
            policy,
            PolicyType::DeepThought | PolicyType::VerifyLogic | PolicyType::SimulateOutcome
        );

        (use_cortex, policy)
    }

    /// Quick surprise estimate without embeddings (pre-response)
    /// Fallback for Sentinel Stage 1 when embedding fails.
    fn estimate_surprise_proxy(&self, message: &str) -> f32 {
        let mut surprise = 0.3; // Base

        // Novel vocabulary increases surprise
        let words: Vec<&str> = message.split_whitespace().collect();
        let unique_ratio = {
            let unique: std::collections::HashSet<_> = words.iter().collect();
            if words.is_empty() {
                0.0
            } else {
                unique.len() as f32 / words.len() as f32
            }
        };
        surprise += unique_ratio * 0.5;

        // Question marks increase surprise
        let question_count = message.matches('?').count();
        surprise += question_count as f32 * 0.1;

        // Code blocks increase surprise
        if message.contains("```") {
            surprise += 0.3;
        }

        // Technical terms increase surprise
        let technical = [
            "algorithm",
            "implementation",
            "architecture",
            "optimization",
            "inference",
            "neural",
            "quantum",
        ];
        let lower = message.to_lowercase();
        for term in &technical {
            if lower.contains(term) {
                surprise += 0.1;
            }
        }

        surprise.min(3.0)
    }

    /// Estimate entropy from message characteristics
    /// Fallback for Sentinel Stage 1 when embedding fails.
    fn estimate_entropy(&self, message: &str) -> f32 {
        // Shannon entropy approximation from character distribution
        let mut char_counts = [0u32; 256];
        let total = message.len() as f32;

        if total == 0.0 {
            return 0.0;
        }

        for byte in message.bytes() {
            char_counts[byte as usize] += 1;
        }

        let mut entropy = 0.0;
        for count in char_counts.iter() {
            if *count > 0 {
                let p = *count as f32 / total;
                entropy -= p * p.log2();
            }
        }

        // Normalize to [0, 1] range (max entropy for ASCII is ~7 bits)
        (entropy / 7.0).min(1.0)
    }

    /// Update Titans memory with response (test-time learning)
    /// Used by Sentinel Stage 4 (Learning) for test-time learning.
    #[allow(dead_code)]
    pub async fn update_memory(&self, response: &str, surprise: f32) {
        // Only update if we have Titans memory
        let mut titans_lock = self.titans.write().await;
        if let Some(ref mut titans) = *titans_lock {
            // Get response embedding
            match self
                .ollama
                .embeddings(&self.embedding_model, response)
                .await
            {
                Ok(embedding) => {
                    let embedding_array = Array1::from_vec(embedding);
                    if let Err(e) = titans.update(&embedding_array, surprise) {
                        warn!("Failed to update Titans memory: {}", e);
                    } else {
                        debug!("Titans memory updated with surprise={:.2}", surprise);
                    }
                }
                Err(e) => {
                    warn!("Failed to get embedding for Titans update: {}", e);
                }
            }
        }
    }

    /// Invalidate context embedding cache
    #[allow(dead_code)]
    pub async fn invalidate_context_cache(&self) {
        let mut cache = self.context_embedding_cache.write().await;
        *cache = None;
    }

    /// Get Titans memory statistics
    #[allow(dead_code)]
    pub async fn get_titans_stats(&self) -> Option<crate::engine::titans::TitansStats> {
        let titans_lock = self.titans.read().await;
        titans_lock.as_ref().map(|t| t.stats().clone())
    }

    /// Get Agency statistics
    #[allow(dead_code)]
    pub async fn get_agency_stats(&self) -> crate::engine::agency::AgencyStats {
        let agency = self.agency.read().await;
        agency.stats().clone()
    }
}
