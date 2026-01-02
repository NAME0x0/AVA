//! AVA Test Suite
//!
//! Comprehensive tests for the AVA Neural Interface backend.
//!
//! # Test Modules
//!
//! - `ollama_tests` - Tests for Ollama client connectivity and API
//! - `state_tests` - Tests for application state management
//! - `routes_tests` - Tests for HTTP API endpoints
//! - `models_tests` - Tests for data model serialization

#[cfg(test)]
mod ollama_tests;

#[cfg(test)]
mod state_tests;

#[cfg(test)]
mod routes_tests;

#[cfg(test)]
mod models_tests;
