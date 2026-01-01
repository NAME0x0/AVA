//! AVA Engine - Unified Rust Backend
//!
//! This module contains the complete backend implementation that replaces
//! the Python server. It provides:
//!
//! - HTTP API server (axum-based)
//! - Ollama integration for LLM inference
//! - Cognitive state management
//! - WebSocket streaming support
//! - Memory and conversation management

pub mod models;
pub mod ollama;
pub mod routes;
pub mod server;
pub mod state;
pub mod cognitive;
pub mod config;

pub use server::start_embedded_server;
