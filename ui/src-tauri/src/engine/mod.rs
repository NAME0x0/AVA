//! AVA Engine - Unified Rust Backend
//!
//! This module contains the complete backend implementation that replaces
//! the Python server. It provides a high-performance, single-binary solution
//! for running AVA on any platform.
//!
//! # Architecture
//!
//! The engine follows a Cortex-Medulla architecture inspired by the human brain:
//!
//! - **Medulla** (Fast Path): Quick, reflexive responses for simple queries
//! - **Cortex** (Deep Path): Thoughtful, analytical responses for complex queries
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    HTTP/WebSocket API                    │
//! │                     (routes.rs)                         │
//! └─────────────────────────────────────────────────────────┘
//!                            │
//!                            ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │                   Cognitive Engine                       │
//! │                   (cognitive.rs)                         │
//! │  ┌─────────────┐              ┌─────────────┐           │
//! │  │   Medulla   │◄──routing───►│   Cortex    │           │
//! │  │  (fast LLM) │              │ (deep LLM)  │           │
//! │  └─────────────┘              └─────────────┘           │
//! └─────────────────────────────────────────────────────────┘
//!                            │
//!                            ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │                  Ollama Client                           │
//! │                   (ollama.rs)                            │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Module Overview
//!
//! - [`cognitive`]: Core AI processing logic with model routing
//! - [`config`]: Configuration management and YAML loading
//! - [`models`]: Data structures for API requests/responses
//! - [`ollama`]: Ollama LLM client with streaming support
//! - [`routes`]: HTTP API endpoints (health, chat, status, etc.)
//! - [`server`]: HTTP server setup and middleware
//! - [`state`]: Shared application state management
//!
//! # Example Usage
//!
//! ```rust,ignore
//! use ava::engine::start_embedded_server;
//!
//! // Start the embedded server on port 8085
//! let handle = start_embedded_server(8085).await?;
//!
//! // Server runs in background, use handle to manage lifecycle
//! ```
//!
//! # API Endpoints
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/health` | GET | Health check |
//! | `/chat` | POST | Send a message |
//! | `/chat/stream` | POST | Stream response via SSE |
//! | `/status` | GET | System status |
//! | `/info` | GET | Detailed system info |
//! | `/settings` | GET/POST | Read/update settings |
//! | `/ws` | WebSocket | Real-time bidirectional chat |

pub mod cognitive;
pub mod config;
pub mod models;
pub mod ollama;
pub mod routes;
pub mod server;
pub mod state;

pub use server::start_embedded_server;
