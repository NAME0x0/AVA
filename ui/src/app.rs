//! Main Application - AVA Neural Interface

use eframe::egui::{self, Color32, Stroke, Vec2};
use tokio::runtime::Runtime;

use crate::backend::{BackendClient, ChatMessage, CognitiveState, UiCommand};
use crate::components::{
    chat_bubble, cognitive_state_badge, sleep_phase_indicator,
    surprise_gauge, GlowingInput, MetricCard, NeuralActivityIndicator,
};
use crate::theme::NeuralColors;
use crate::animations::{AnimatedValue, PulseAnimation};

/// Main application state
pub struct AvaApp {
    // Runtime
    runtime: Runtime,
    
    // Backend communication
    backend: BackendClient,
    
    // UI state
    input: GlowingInput,
    neural_activity: NeuralActivityIndicator,
    
    // Animated metrics
    entropy_card: MetricCard,
    varentropy_card: MetricCard,
    surprise_card: MetricCard,
    memory_card: MetricCard,
    
    // Animation state
    logo_pulse: PulseAnimation,
    sidebar_width: AnimatedValue,
    show_sidebar: bool,
    
    // View state
    scroll_to_bottom: bool,
    
    // Demo mode (when backend not connected)
    demo_mode: bool,
    demo_messages: Vec<ChatMessage>,
    demo_cognitive: CognitiveState,
}

impl AvaApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let runtime = Runtime::new().expect("Failed to create Tokio runtime");
        
        let mut backend = BackendClient::new("http://localhost:8080");
        
        // Start backend communication
        let ctx = cc.egui_ctx.clone();
        runtime.block_on(async {
            backend.start(ctx);
        });
        
        Self {
            runtime,
            backend,
            input: GlowingInput::new(),
            neural_activity: NeuralActivityIndicator::new(),
            entropy_card: MetricCard::new("Entropy", 0.0, "H", NeuralColors::STATE_HESITATION),
            varentropy_card: MetricCard::new("Varentropy", 0.0, "V", NeuralColors::STATE_CREATIVE),
            surprise_card: MetricCard::new("Surprise", 0.0, "σ", NeuralColors::MEMORY_SURPRISE),
            memory_card: MetricCard::new("Memories", 0.0, "", NeuralColors::MEMORY_EPISODIC),
            logo_pulse: PulseAnimation::new(0.5, 0.15),
            sidebar_width: AnimatedValue::new(280.0),
            show_sidebar: true,
            scroll_to_bottom: false,
            demo_mode: true,
            demo_messages: vec![
                ChatMessage {
                    role: "assistant".into(),
                    content: "Hello! I'm AVA, your Adaptive Virtual Agent. I'm running in demo mode since the backend isn't connected. Try sending a message!".into(),
                    timestamp: "00:00:00".into(),
                    cognitive_state: Some(CognitiveState {
                        label: "FLOW".into(),
                        entropy: 1.2,
                        varentropy: 0.8,
                        confidence: 0.9,
                        surprise: 0.5,
                        should_use_tools: false,
                        should_think: false,
                    }),
                    surprise: Some(0.5),
                },
            ],
            demo_cognitive: CognitiveState {
                label: "FLOW".into(),
                entropy: 1.2,
                varentropy: 0.8,
                confidence: 0.9,
                surprise: 0.5,
                should_use_tools: false,
                should_think: false,
            },
        }
    }
    
    fn render_header(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Logo with pulse
            let pulse = self.logo_pulse.get();
            let logo_size = 32.0 + pulse * 4.0;
            
            let (logo_rect, _) = ui.allocate_exact_size(Vec2::splat(40.0), egui::Sense::hover());
            let painter = ui.painter();
            
            // Outer glow
            painter.circle_filled(
                logo_rect.center(),
                logo_size / 2.0 + 4.0,
                Color32::from_rgba_unmultiplied(0, 212, 200, (pulse * 40.0) as u8)
            );
            
            // Main circle
            painter.circle_filled(logo_rect.center(), logo_size / 2.0, NeuralColors::ACCENT_PRIMARY);
            
            // Inner ring
            painter.circle_stroke(
                logo_rect.center(),
                logo_size / 2.0 - 6.0,
                Stroke::new(2.0, NeuralColors::BG_VOID)
            );
            
            // "A" letter
            painter.text(
                logo_rect.center(),
                egui::Align2::CENTER_CENTER,
                "A",
                egui::FontId::proportional(18.0),
                NeuralColors::BG_VOID,
            );
            
            ui.add_space(12.0);
            
            ui.vertical(|ui| {
                ui.colored_label(
                    NeuralColors::TEXT_PRIMARY,
                    egui::RichText::new("AVA").size(20.0).strong()
                );
                ui.colored_label(
                    NeuralColors::TEXT_MUTED,
                    egui::RichText::new("Adaptive Virtual Agent").size(11.0)
                );
            });
            
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Connection status
                let (connected, model) = if self.demo_mode {
                    (false, "Demo Mode".to_string())
                } else {
                    let state = self.runtime.block_on(async { self.backend.state.read().await.clone() });
                    let model_str = if state.connected { state.model_name.clone() } else { "Disconnected".to_string() };
                    (state.connected, model_str)
                };
                
                let status_color = if connected { NeuralColors::SUCCESS } else { NeuralColors::WARNING };
                
                ui.horizontal(|ui| {
                    let (dot_rect, _) = ui.allocate_exact_size(Vec2::splat(8.0), egui::Sense::hover());
                    ui.painter().circle_filled(dot_rect.center(), 4.0, status_color);
                    ui.colored_label(NeuralColors::TEXT_SECONDARY, &model);
                });
                
                ui.add_space(16.0);
                
                // Toggle sidebar
                if ui.button(if self.show_sidebar { "◀" } else { "▶" }).clicked() {
                    self.show_sidebar = !self.show_sidebar;
                    self.sidebar_width.animate_to(if self.show_sidebar { 280.0 } else { 0.0 }, 300);
                }
            });
        });
    }
    
    fn render_sidebar(&mut self, ui: &mut egui::Ui, cognitive: &CognitiveState) {
        ui.vertical(|ui| {
            ui.add_space(8.0);
            
            // Cognitive State Section
            ui.colored_label(NeuralColors::TEXT_MUTED, "COGNITIVE STATE");
            ui.add_space(8.0);
            
            cognitive_state_badge(ui, &cognitive.label, cognitive.entropy, cognitive.varentropy);
            
            ui.add_space(16.0);
            
            // Neural Activity
            ui.colored_label(NeuralColors::TEXT_MUTED, "NEURAL ACTIVITY");
            ui.add_space(8.0);
            self.neural_activity.update(cognitive.entropy, cognitive.varentropy);
            self.neural_activity.show(ui, Vec2::new(ui.available_width(), 60.0));
            
            ui.add_space(16.0);
            
            // Metrics Grid
            ui.colored_label(NeuralColors::TEXT_MUTED, "METRICS");
            ui.add_space(8.0);
            
            self.entropy_card.set_value(cognitive.entropy);
            self.varentropy_card.set_value(cognitive.varentropy);
            self.surprise_card.set_value(cognitive.surprise);
            
            ui.columns(2, |cols| {
                self.entropy_card.show(&mut cols[0]);
                self.varentropy_card.show(&mut cols[1]);
            });
            
            ui.add_space(8.0);
            
            ui.columns(2, |cols| {
                self.surprise_card.show(&mut cols[0]);
                self.memory_card.show(&mut cols[1]);
            });
            
            ui.add_space(16.0);
            
            // Surprise Gauge
            ui.colored_label(NeuralColors::TEXT_MUTED, "SURPRISE LEVEL");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add_space((ui.available_width() - 80.0) / 2.0);
                surprise_gauge(ui, cognitive.surprise, 3.0);
            });
            
            ui.add_space(16.0);
            
            // Sleep Status
            ui.colored_label(NeuralColors::TEXT_MUTED, "SLEEP CYCLE");
            ui.add_space(8.0);
            sleep_phase_indicator(ui, "AWAKE", 0.0);
            
            // Actions
            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);
            
            if ui.add(egui::Button::new("🌙 Force Sleep").fill(NeuralColors::BG_ELEVATED)).clicked() {
                let _ = self.backend.command_tx.try_send(UiCommand::ForceSleep);
            }
        });
    }
    
    fn render_chat(&mut self, ui: &mut egui::Ui, messages: &[ChatMessage]) {
        // Chat messages area
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                
                ui.add_space(16.0);
                
                for msg in messages {
                    chat_bubble(
                        ui,
                        msg.role == "user",
                        &msg.content,
                        &msg.timestamp,
                        msg.surprise,
                    );
                    ui.add_space(12.0);
                }
                
                ui.add_space(16.0);
            });
    }
    
    fn render_input(&mut self, ui: &mut egui::Ui) {
        egui::Frame::none()
            .fill(NeuralColors::BG_SURFACE)
            .inner_margin(egui::Margin::symmetric(16.0, 12.0))
            .show(ui, |ui| {
                if let Some(message) = self.input.show(ui) {
                    if self.demo_mode {
                        // Demo mode response
                        self.demo_messages.push(ChatMessage {
                            role: "user".into(),
                            content: message.clone(),
                            timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
                            cognitive_state: None,
                            surprise: None,
                        });
                        
                        // Simulate cognitive state change
                        let entropy = (message.len() as f32 / 20.0).clamp(0.5, 4.0);
                        let varentropy = entropy * 0.7;
                        let surprise = if message.contains("?") { 1.8 } else { 0.8 };
                        
                        let state_label = if entropy < 1.5 {
                            "FLOW"
                        } else if entropy < 2.5 {
                            "HESITATION"
                        } else if varentropy > 2.0 {
                            "CREATIVE"
                        } else {
                            "CONFUSION"
                        };
                        
                        self.demo_cognitive = CognitiveState {
                            label: state_label.into(),
                            entropy,
                            varentropy,
                            confidence: 1.0 - entropy / 5.0,
                            surprise,
                            should_use_tools: entropy > 2.5,
                            should_think: entropy > 2.0,
                        };
                        
                        self.demo_messages.push(ChatMessage {
                            role: "assistant".into(),
                            content: format!(
                                "I processed your message in {} state. Entropy: {:.2}, Varentropy: {:.2}. \
                                This is demo mode - connect the backend with `python run_frankensystem.py --api` for full functionality!",
                                state_label, entropy, varentropy
                            ),
                            timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
                            cognitive_state: Some(self.demo_cognitive.clone()),
                            surprise: Some(surprise),
                        });
                    } else {
                        let _ = self.backend.command_tx.try_send(UiCommand::SendMessage { content: message });
                    }
                    self.scroll_to_bottom = true;
                }
            });
    }
}

impl eframe::App for AvaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request repaint for animations
        ctx.request_repaint();
        
        // Get current state
        let (messages, cognitive) = if self.demo_mode {
            (self.demo_messages.clone(), self.demo_cognitive.clone())
        } else {
            let state = self.runtime.block_on(async { self.backend.state.read().await.clone() });
            (state.messages, state.cognitive)
        };
        
        // Main layout
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(NeuralColors::BG_VOID))
            .show(ctx, |ui| {
                // Header
                egui::Frame::none()
                    .fill(NeuralColors::BG_SURFACE)
                    .inner_margin(egui::Margin::symmetric(20.0, 12.0))
                    .show(ui, |ui| {
                        self.render_header(ui);
                    });
                
                ui.separator();
                
                // Main content area
                ui.horizontal(|ui| {
                    // Sidebar
                    let sidebar_w = self.sidebar_width.get();
                    if sidebar_w > 1.0 {
                        egui::Frame::none()
                            .fill(NeuralColors::BG_SURFACE)
                            .inner_margin(egui::Margin::same(16.0))
                            .show(ui, |ui| {
                                ui.set_width(sidebar_w - 32.0);
                                ui.set_min_height(ui.available_height() - 80.0);
                                self.render_sidebar(ui, &cognitive);
                            });
                        
                        ui.separator();
                    }
                    
                    // Chat area
                    ui.vertical(|ui| {
                        // Messages
                        let available_height = ui.available_height() - 80.0;
                        ui.set_min_height(available_height);
                        self.render_chat(ui, &messages);
                    });
                });
                
                // Input area at bottom
                self.render_input(ui);
            });
    }
}
