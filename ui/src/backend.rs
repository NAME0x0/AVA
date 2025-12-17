//! Backend communication with AVA Python system

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Messages from UI to backend
#[derive(Debug, Clone, Serialize)]
pub enum UiCommand {
    SendMessage { content: String },
    RequestCognitiveState,
    RequestMemoryStats,
    RequestBufferStats,
    ForceSleep,
    Shutdown,
}

/// Cognitive state from Entropix
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CognitiveState {
    pub label: String,
    pub entropy: f32,
    pub varentropy: f32,
    pub confidence: f32,
    pub surprise: f32,
    pub should_use_tools: bool,
    pub should_think: bool,
}

impl CognitiveState {
    pub fn state_color(&self) -> [f32; 3] {
        match self.label.as_str() {
            "FLOW" => [0.31, 0.78, 0.47],        // Green
            "HESITATION" => [1.0, 0.78, 0.31],   // Amber
            "CONFUSION" => [1.0, 0.39, 0.39],    // Red
            "CREATIVE" => [0.71, 0.39, 1.0],     // Purple
            _ => [0.63, 0.63, 0.69],             // Gray
        }
    }
}

/// Memory statistics from Titans
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct MemoryStats {
    pub total_memories: u64,
    pub memory_updates: u64,
    pub avg_surprise: f32,
    pub backend: String,
}

/// Episodic buffer statistics
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct BufferStats {
    pub total_episodes: u64,
    pub avg_surprise: f32,
    pub avg_quality: f32,
    pub high_priority_count: u64,
}

/// Sleep state from Nightmare Engine
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SleepState {
    pub phase: String,
    pub is_sleeping: bool,
    pub progress: f32,
    pub episodes_processed: u64,
}

impl SleepState {
    pub fn phase_color(&self) -> [f32; 3] {
        match self.phase.as_str() {
            "DROWSY" => [0.39, 0.39, 0.63],
            "LIGHT_SLEEP" => [0.31, 0.31, 0.55],
            "DEEP_SLEEP" => [0.24, 0.24, 0.47],
            "REM" => [0.47, 0.31, 0.71],
            _ => [0.39, 0.39, 0.39],
        }
    }
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,  // "user" or "assistant"
    pub content: String,
    pub timestamp: String,
    pub cognitive_state: Option<CognitiveState>,
    pub surprise: Option<f32>,
}

/// Response from backend
#[derive(Debug, Clone, Deserialize)]
pub struct BackendResponse {
    pub response: String,
    pub cognitive_state: Option<CognitiveState>,
    pub surprise: Option<f32>,
    pub tokens_generated: Option<u32>,
}

/// Backend connection state
#[derive(Debug, Clone, Default)]
pub struct BackendState {
    pub connected: bool,
    pub model_name: String,
    pub cognitive: CognitiveState,
    pub memory: MemoryStats,
    pub buffer: BufferStats,
    pub sleep: SleepState,
    pub messages: Vec<ChatMessage>,
    pub is_generating: bool,
    pub error: Option<String>,
}

/// Async backend client
pub struct BackendClient {
    pub state: Arc<RwLock<BackendState>>,
    pub command_tx: mpsc::Sender<UiCommand>,
    command_rx: Option<mpsc::Receiver<UiCommand>>,
    base_url: String,
}

impl BackendClient {
    pub fn new(base_url: &str) -> Self {
        let (tx, rx) = mpsc::channel(100);
        Self {
            state: Arc::new(RwLock::new(BackendState::default())),
            command_tx: tx,
            command_rx: Some(rx),
            base_url: base_url.to_string(),
        }
    }
    
    /// Start the background communication loop
    pub fn start(&mut self, ctx: egui::Context) {
        let state = self.state.clone();
        let base_url = self.base_url.clone();
        let mut rx = self.command_rx.take().expect("Already started");
        
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            
            // Initial connection check
            match client.get(format!("{}/health", base_url)).send().await {
                Ok(_) => {
                    let mut s = state.write().await;
                    s.connected = true;
                    s.model_name = "llama3.2:latest".into();
                }
                Err(e) => {
                    let mut s = state.write().await;
                    s.error = Some(format!("Connection failed: {}", e));
                }
            }
            ctx.request_repaint();
            
            // Command processing loop
            while let Some(cmd) = rx.recv().await {
                match cmd {
                    UiCommand::SendMessage { content } => {
                        // Add user message
                        {
                            let mut s = state.write().await;
                            s.messages.push(ChatMessage {
                                role: "user".into(),
                                content: content.clone(),
                                timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
                                cognitive_state: None,
                                surprise: None,
                            });
                            s.is_generating = true;
                        }
                        ctx.request_repaint();
                        
                        // Send to backend
                        let resp = client.post(format!("{}/chat", base_url))
                            .json(&serde_json::json!({ "message": content }))
                            .send()
                            .await;
                        
                        match resp {
                            Ok(r) => {
                                if let Ok(data) = r.json::<BackendResponse>().await {
                                    let mut s = state.write().await;
                                    s.messages.push(ChatMessage {
                                        role: "assistant".into(),
                                        content: data.response,
                                        timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
                                        cognitive_state: data.cognitive_state.clone(),
                                        surprise: data.surprise,
                                    });
                                    if let Some(cog) = data.cognitive_state {
                                        s.cognitive = cog;
                                    }
                                    s.is_generating = false;
                                }
                            }
                            Err(e) => {
                                let mut s = state.write().await;
                                s.error = Some(format!("Request failed: {}", e));
                                s.is_generating = false;
                            }
                        }
                        ctx.request_repaint();
                    }
                    UiCommand::RequestCognitiveState => {
                        if let Ok(r) = client.get(format!("{}/cognitive", base_url)).send().await {
                            if let Ok(cog) = r.json::<CognitiveState>().await {
                                let mut s = state.write().await;
                                s.cognitive = cog;
                            }
                        }
                        ctx.request_repaint();
                    }
                    UiCommand::ForceSleep => {
                        let _ = client.post(format!("{}/sleep", base_url)).send().await;
                        ctx.request_repaint();
                    }
                    UiCommand::Shutdown => break,
                    _ => {}
                }
            }
        });
    }
}

impl Default for BackendClient {
    fn default() -> Self {
        Self::new("http://localhost:8080")
    }
}
