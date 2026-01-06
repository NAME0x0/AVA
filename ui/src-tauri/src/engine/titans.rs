//! TITANS - Test-Time Learning Memory Buffer
//!
//! Implements the MAC (Memory as Context) variant of the Titans architecture
//! for test-time learning. This allows AVA to update its "neural" weights
//! during inference without retraining.
//!
//! # Architecture
//!
//! ```text
//! Input Embedding [768]
//!        ↓
//!   Linear Layer 1 (768 → 1024)
//!        ↓
//!      ReLU
//!        ↓
//!   Linear Layer 2 (1024 → 768)
//!        ↓
//! Output Embedding [768]
//! ```
//!
//! # Surprise-Weighted Updates
//!
//! The memory only updates when surprise exceeds a threshold:
//! - surprise < threshold: Skip update (routine input)
//! - surprise >= threshold: Update weights with lr * surprise
//!
//! # References
//!
//! - Titans: Learning to Learn at Test Time (Google Research, 2025)
//! - Memory as Context for Neural Networks (arXiv, 2024)

// Allow dead code as this module is part of the Sentinel architecture
// that is being incrementally integrated
#![allow(dead_code)]

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Default embedding dimension (matches common models)
const DEFAULT_EMBEDDING_DIM: usize = 768;

/// Default hidden layer dimension
const DEFAULT_HIDDEN_DIM: usize = 1024;

/// Default surprise threshold for updates
const DEFAULT_SURPRISE_THRESHOLD: f32 = 0.5;

/// Default learning rate
const DEFAULT_LEARNING_RATE: f32 = 0.01;

/// Default momentum for gradient accumulation
const DEFAULT_MOMENTUM: f32 = 0.9;

/// Default forgetting rate (weight decay)
const DEFAULT_FORGETTING_RATE: f32 = 0.01;

/// Maximum experiences to retain
const DEFAULT_MAX_EXPERIENCES: usize = 1000;

/// Error types for Titans memory operations
#[derive(Error, Debug)]
pub enum TitansError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Memory buffer full")]
    BufferFull,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Configuration for Titans memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TitansConfig {
    /// Embedding dimension (input/output size)
    pub embedding_dim: usize,

    /// Hidden layer dimension
    pub hidden_dim: usize,

    /// Minimum surprise for weight update
    pub surprise_threshold: f32,

    /// Base learning rate
    pub learning_rate: f32,

    /// Momentum for gradient accumulation
    pub momentum: f32,

    /// Weight decay (forgetting) rate
    pub forgetting_rate: f32,

    /// Maximum experiences to store
    pub max_experiences: usize,

    /// Enable surprise-weighted learning rate
    pub surprise_weighted_lr: bool,
}

impl Default for TitansConfig {
    fn default() -> Self {
        Self {
            embedding_dim: DEFAULT_EMBEDDING_DIM,
            hidden_dim: DEFAULT_HIDDEN_DIM,
            surprise_threshold: DEFAULT_SURPRISE_THRESHOLD,
            learning_rate: DEFAULT_LEARNING_RATE,
            momentum: DEFAULT_MOMENTUM,
            forgetting_rate: DEFAULT_FORGETTING_RATE,
            max_experiences: DEFAULT_MAX_EXPERIENCES,
            surprise_weighted_lr: true,
        }
    }
}

/// An experience stored in memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// The embedding vector
    pub embedding: Vec<f32>,

    /// Surprise value when stored
    pub surprise: f32,

    /// Timestamp when stored
    pub timestamp: u64,

    /// Optional content hash for deduplication
    pub content_hash: Option<u64>,

    /// Importance score (computed from surprise + recency)
    pub importance: f32,
}

impl Experience {
    /// Create a new experience
    pub fn new(embedding: Vec<f32>, surprise: f32) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            embedding,
            surprise,
            timestamp,
            content_hash: None,
            importance: surprise, // Initial importance = surprise
        }
    }

    /// Update importance based on recency
    pub fn update_importance(&mut self, current_time: u64, decay_rate: f32) {
        let age_ms = current_time.saturating_sub(self.timestamp) as f32;
        let age_factor = (-age_ms / 3600000.0 * decay_rate).exp(); // Decay over hours
        self.importance = self.surprise * age_factor;
    }
}

/// Statistics for Titans memory
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TitansStats {
    /// Total update calls
    pub total_updates: u64,

    /// Updates that actually modified weights (surprise above threshold)
    pub effective_updates: u64,

    /// Total retrievals
    pub total_retrievals: u64,

    /// Current experience count
    pub experience_count: usize,

    /// Average surprise of stored experiences
    pub avg_surprise: f32,

    /// Total forward passes
    pub forward_passes: u64,

    /// Cumulative processing time
    pub total_time_ms: f64,
}

/// Titans Memory Buffer
///
/// A 2-layer MLP that learns at test-time using surprise-weighted updates.
/// Implements the MAC (Memory as Context) variant.
pub struct TitansMemory {
    /// Configuration
    config: TitansConfig,

    /// First layer weights: [embedding_dim, hidden_dim]
    weights_1: Array2<f32>,

    /// First layer bias: [hidden_dim]
    bias_1: Array1<f32>,

    /// Second layer weights: [hidden_dim, embedding_dim]
    weights_2: Array2<f32>,

    /// Second layer bias: [embedding_dim]
    bias_2: Array1<f32>,

    /// Momentum accumulator for weights_1
    momentum_w1: Array2<f32>,

    /// Momentum accumulator for bias_1
    momentum_b1: Array1<f32>,

    /// Momentum accumulator for weights_2
    momentum_w2: Array2<f32>,

    /// Momentum accumulator for bias_2
    momentum_b2: Array1<f32>,

    /// Experience buffer
    experiences: VecDeque<Experience>,

    /// Statistics
    stats: TitansStats,

    /// Initialization timestamp
    created_at: Instant,
}

impl TitansMemory {
    /// Create a new Titans memory with the given configuration
    pub fn new(config: TitansConfig) -> Result<Self, TitansError> {
        // Validate configuration
        if config.embedding_dim == 0 || config.hidden_dim == 0 {
            return Err(TitansError::InvalidConfig(
                "Dimensions must be non-zero".to_string(),
            ));
        }

        // Initialize weights using Xavier/Glorot initialization
        let mut rng = rand::thread_rng();
        let scale_1 = (2.0 / (config.embedding_dim + config.hidden_dim) as f32).sqrt();
        let scale_2 = (2.0 / (config.hidden_dim + config.embedding_dim) as f32).sqrt();

        let weights_1 = Array2::from_shape_fn((config.embedding_dim, config.hidden_dim), |_| {
            rng.gen_range(-scale_1..scale_1)
        });

        let bias_1 = Array1::zeros(config.hidden_dim);

        let weights_2 = Array2::from_shape_fn((config.hidden_dim, config.embedding_dim), |_| {
            rng.gen_range(-scale_2..scale_2)
        });

        let bias_2 = Array1::zeros(config.embedding_dim);

        // Initialize momentum accumulators to zero
        let momentum_w1 = Array2::zeros((config.embedding_dim, config.hidden_dim));
        let momentum_b1 = Array1::zeros(config.hidden_dim);
        let momentum_w2 = Array2::zeros((config.hidden_dim, config.embedding_dim));
        let momentum_b2 = Array1::zeros(config.embedding_dim);

        Ok(Self {
            config,
            weights_1,
            bias_1,
            weights_2,
            bias_2,
            momentum_w1,
            momentum_b1,
            momentum_w2,
            momentum_b2,
            experiences: VecDeque::new(),
            stats: TitansStats::default(),
            created_at: Instant::now(),
        })
    }

    /// Create with default configuration
    pub fn with_defaults() -> Result<Self, TitansError> {
        Self::new(TitansConfig::default())
    }

    /// Forward pass through the MLP
    pub fn forward(&mut self, input: &Array1<f32>) -> Result<Array1<f32>, TitansError> {
        let start = Instant::now();

        // Validate input dimension
        if input.len() != self.config.embedding_dim {
            return Err(TitansError::DimensionMismatch {
                expected: self.config.embedding_dim,
                actual: input.len(),
            });
        }

        // Layer 1: Linear + ReLU
        let hidden = input.dot(&self.weights_1) + &self.bias_1;
        let hidden_relu = hidden.mapv(|x| x.max(0.0));

        // Layer 2: Linear
        let output = hidden_relu.dot(&self.weights_2) + &self.bias_2;

        // Update stats
        self.stats.forward_passes += 1;
        self.stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;

        Ok(output)
    }

    /// Update weights using surprise-weighted learning
    ///
    /// Only updates if surprise exceeds threshold.
    /// Returns true if weights were updated.
    pub fn update(&mut self, input: &Array1<f32>, surprise: f32) -> Result<bool, TitansError> {
        let start = Instant::now();
        self.stats.total_updates += 1;

        // Check surprise threshold
        if surprise < self.config.surprise_threshold {
            return Ok(false);
        }

        // Validate input dimension
        if input.len() != self.config.embedding_dim {
            return Err(TitansError::DimensionMismatch {
                expected: self.config.embedding_dim,
                actual: input.len(),
            });
        }

        // Calculate effective learning rate
        let lr = if self.config.surprise_weighted_lr {
            self.config.learning_rate * surprise.min(2.0) // Cap at 2x
        } else {
            self.config.learning_rate
        };

        // Forward pass to compute activations
        let hidden = input.dot(&self.weights_1) + &self.bias_1;
        let hidden_relu = hidden.mapv(|x| x.max(0.0));
        let output = hidden_relu.dot(&self.weights_2) + &self.bias_2;

        // Compute loss gradient (MSE with input as target - autoencoder behavior)
        let output_error = &output - input;

        // Backpropagation

        // Gradient for weights_2: hidden_relu.T @ output_error
        let grad_w2 = outer_product(&hidden_relu, &output_error);

        // Gradient for bias_2: output_error
        let grad_b2 = output_error.clone();

        // Gradient for hidden (before ReLU): output_error @ weights_2.T
        let grad_hidden = output_error.dot(&self.weights_2.t());

        // Apply ReLU derivative
        let grad_hidden_relu = &grad_hidden * &hidden.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        // Gradient for weights_1: input.T @ grad_hidden_relu
        let grad_w1 = outer_product(input, &grad_hidden_relu);

        // Gradient for bias_1: grad_hidden_relu
        let grad_b1 = grad_hidden_relu;

        // Apply momentum updates
        let m = self.config.momentum;

        self.momentum_w1 = &self.momentum_w1 * m + &grad_w1 * (1.0 - m);
        self.momentum_b1 = &self.momentum_b1 * m + &grad_b1 * (1.0 - m);
        self.momentum_w2 = &self.momentum_w2 * m + &grad_w2 * (1.0 - m);
        self.momentum_b2 = &self.momentum_b2 * m + &grad_b2 * (1.0 - m);

        // Update weights with gradient descent
        self.weights_1 = &self.weights_1 - &(&self.momentum_w1 * lr);
        self.bias_1 = &self.bias_1 - &(&self.momentum_b1 * lr);
        self.weights_2 = &self.weights_2 - &(&self.momentum_w2 * lr);
        self.bias_2 = &self.bias_2 - &(&self.momentum_b2 * lr);

        // Apply forgetting (weight decay)
        let decay = 1.0 - self.config.forgetting_rate;
        self.weights_1 *= decay;
        self.weights_2 *= decay;

        // Store experience
        self.store_experience(input.to_vec(), surprise)?;

        // Update stats
        self.stats.effective_updates += 1;
        self.stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;

        Ok(true)
    }

    /// Store an experience in the buffer
    fn store_experience(&mut self, embedding: Vec<f32>, surprise: f32) -> Result<(), TitansError> {
        let experience = Experience::new(embedding, surprise);

        // Remove oldest if at capacity
        while self.experiences.len() >= self.config.max_experiences {
            self.experiences.pop_front();
        }

        self.experiences.push_back(experience);
        self.stats.experience_count = self.experiences.len();

        // Update average surprise
        let total_surprise: f32 = self.experiences.iter().map(|e| e.surprise).sum();
        self.stats.avg_surprise = total_surprise / self.experiences.len() as f32;

        Ok(())
    }

    /// Retrieve relevant experiences for a query
    pub fn retrieve(&mut self, query: &Array1<f32>, top_k: usize) -> Vec<&Experience> {
        self.stats.total_retrievals += 1;

        if self.experiences.is_empty() {
            return vec![];
        }

        // Compute similarities
        let mut scored: Vec<(usize, f32)> = self
            .experiences
            .iter()
            .enumerate()
            .map(|(i, exp)| {
                let sim = cosine_similarity(query.as_slice().unwrap(), &exp.embedding);
                (i, sim)
            })
            .collect();

        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top_k
        scored
            .iter()
            .take(top_k)
            .map(|(i, _)| &self.experiences[*i])
            .collect()
    }

    /// Get compressed state representation
    pub fn get_state(&mut self, input: &Array1<f32>) -> Result<Array1<f32>, TitansError> {
        // Forward pass gives compressed representation
        self.forward(input)
    }

    /// Get statistics
    pub fn stats(&self) -> &TitansStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &TitansConfig {
        &self.config
    }

    /// Get experience count
    pub fn experience_count(&self) -> usize {
        self.experiences.len()
    }

    /// Get memory footprint in bytes (approximate)
    pub fn memory_footprint(&self) -> usize {
        let weights_bytes = (self.config.embedding_dim * self.config.hidden_dim * 2
            + self.config.hidden_dim
            + self.config.embedding_dim)
            * 4; // f32 = 4 bytes

        let momentum_bytes = weights_bytes;

        let experience_bytes = self.experiences.len() * (self.config.embedding_dim * 4 + 32); // embedding + metadata

        weights_bytes + momentum_bytes + experience_bytes
    }

    /// Clear all experiences (but keep weights)
    pub fn clear_experiences(&mut self) {
        self.experiences.clear();
        self.stats.experience_count = 0;
    }

    /// Reset weights to random initialization
    pub fn reset_weights(&mut self) {
        let mut rng = rand::thread_rng();
        let scale_1 = (2.0 / (self.config.embedding_dim + self.config.hidden_dim) as f32).sqrt();
        let scale_2 = (2.0 / (self.config.hidden_dim + self.config.embedding_dim) as f32).sqrt();

        self.weights_1 =
            Array2::from_shape_fn((self.config.embedding_dim, self.config.hidden_dim), |_| {
                rng.gen_range(-scale_1..scale_1)
            });
        self.bias_1 = Array1::zeros(self.config.hidden_dim);

        self.weights_2 =
            Array2::from_shape_fn((self.config.hidden_dim, self.config.embedding_dim), |_| {
                rng.gen_range(-scale_2..scale_2)
            });
        self.bias_2 = Array1::zeros(self.config.embedding_dim);

        // Reset momentum
        self.momentum_w1 = Array2::zeros((self.config.embedding_dim, self.config.hidden_dim));
        self.momentum_b1 = Array1::zeros(self.config.hidden_dim);
        self.momentum_w2 = Array2::zeros((self.config.hidden_dim, self.config.embedding_dim));
        self.momentum_b2 = Array1::zeros(self.config.embedding_dim);
    }

    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute outer product of two vectors
fn outer_product(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }

    result
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_titans_creation() {
        let titans = TitansMemory::with_defaults();
        assert!(titans.is_ok());

        let titans = titans.unwrap();
        assert_eq!(titans.config().embedding_dim, DEFAULT_EMBEDDING_DIM);
        assert_eq!(titans.experience_count(), 0);
    }

    #[test]
    fn test_forward_pass() {
        let mut titans = TitansMemory::with_defaults().unwrap();
        let input = Array1::from_vec(vec![0.1; DEFAULT_EMBEDDING_DIM]);

        let output = titans.forward(&input);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.len(), DEFAULT_EMBEDDING_DIM);
    }

    #[test]
    fn test_surprise_threshold() {
        let mut titans = TitansMemory::with_defaults().unwrap();
        let input = Array1::from_vec(vec![0.1; DEFAULT_EMBEDDING_DIM]);

        // Below threshold - should not update
        let updated = titans.update(&input, 0.1).unwrap();
        assert!(!updated);
        assert_eq!(titans.experience_count(), 0);

        // Above threshold - should update
        let updated = titans.update(&input, 0.8).unwrap();
        assert!(updated);
        assert_eq!(titans.experience_count(), 1);
    }

    #[test]
    fn test_retrieval() {
        let mut titans = TitansMemory::with_defaults().unwrap();

        // Add some experiences
        for i in 0..5 {
            let mut input = vec![0.0; DEFAULT_EMBEDDING_DIM];
            input[i] = 1.0;
            let input = Array1::from_vec(input);
            titans.update(&input, 0.8).unwrap();
        }

        assert_eq!(titans.experience_count(), 5);

        // Retrieve similar
        let mut query = vec![0.0; DEFAULT_EMBEDDING_DIM];
        query[0] = 1.0;
        let query = Array1::from_vec(query);

        let results = titans.retrieve(&query, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut titans = TitansMemory::with_defaults().unwrap();
        let wrong_input = Array1::from_vec(vec![0.1; 100]); // Wrong dimension

        let result = titans.forward(&wrong_input);
        assert!(matches!(result, Err(TitansError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_memory_footprint() {
        let titans = TitansMemory::with_defaults().unwrap();
        let footprint = titans.memory_footprint();

        // Should be reasonable for default config
        assert!(footprint > 0);
        assert!(footprint < 100_000_000); // Less than 100MB
    }
}
