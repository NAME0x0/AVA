//! UI Components - Reusable neural-styled widgets

use eframe::egui::{self, Color32, Pos2, Rect, Response, Rounding, Sense, Stroke, Ui, Vec2};
use crate::theme::NeuralColors;
use crate::animations::{AnimatedValue, PulseAnimation};

/// Neural activity indicator (brain wave visualization)
pub struct NeuralActivityIndicator {
    pulse: PulseAnimation,
    values: Vec<f32>,
    update_counter: u32,
}

impl NeuralActivityIndicator {
    pub fn new() -> Self {
        Self {
            pulse: PulseAnimation::new(1.5, 0.3),
            values: vec![0.5; 64],
            update_counter: 0,
        }
    }
    
    pub fn update(&mut self, entropy: f32, varentropy: f32) {
        self.update_counter += 1;
        if self.update_counter % 2 == 0 {
            // Shift values left
            self.values.remove(0);
            // Add new value based on entropy
            let noise = ((self.update_counter as f32 * 0.1).sin() * 0.2) + 
                       ((self.update_counter as f32 * 0.23).cos() * 0.15);
            let value = (entropy / 5.0).clamp(0.0, 1.0) * 0.6 + 
                       varentropy.sqrt() * 0.3 + noise;
            self.values.push(value.clamp(0.1, 0.9));
        }
    }
    
    pub fn show(&mut self, ui: &mut Ui, size: Vec2) -> Response {
        let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
        
        if ui.is_rect_visible(rect) {
            let painter = ui.painter();
            
            // Background
            painter.rect_filled(rect, Rounding::same(8.0), NeuralColors::BG_SURFACE);
            
            // Draw waveform
            let points: Vec<Pos2> = self.values.iter().enumerate().map(|(i, &v)| {
                let x = rect.left() + (i as f32 / self.values.len() as f32) * rect.width();
                let y = rect.center().y - (v - 0.5) * rect.height() * 0.8;
                Pos2::new(x, y)
            }).collect();
            
            // Glow effect
            let glow_alpha = (self.pulse.get() * 60.0) as u8;
            painter.add(egui::Shape::line(
                points.clone(),
                Stroke::new(4.0, Color32::from_rgba_unmultiplied(0, 212, 200, glow_alpha))
            ));
            
            // Main line
            painter.add(egui::Shape::line(
                points,
                Stroke::new(2.0, NeuralColors::ACCENT_PRIMARY)
            ));
            
            // Center line
            painter.line_segment(
                [Pos2::new(rect.left(), rect.center().y), Pos2::new(rect.right(), rect.center().y)],
                Stroke::new(1.0, NeuralColors::BG_HOVER)
            );
        }
        
        response
    }
}

/// Cognitive state badge
pub fn cognitive_state_badge(ui: &mut Ui, label: &str, entropy: f32, varentropy: f32) {
    let color = match label {
        "FLOW" => NeuralColors::STATE_FLOW,
        "HESITATION" => NeuralColors::STATE_HESITATION,
        "CONFUSION" => NeuralColors::STATE_CONFUSION,
        "CREATIVE" => NeuralColors::STATE_CREATIVE,
        _ => NeuralColors::TEXT_MUTED,
    };
    
    egui::Frame::none()
        .fill(color.gamma_multiply(0.2))
        .stroke(Stroke::new(1.0, color))
        .rounding(Rounding::same(16.0))
        .inner_margin(egui::Margin::symmetric(12.0, 6.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                // Pulsing dot
                let (dot_rect, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
                ui.painter().circle_filled(dot_rect.center(), 4.0, color);
                
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.colored_label(color, label);
                
                ui.separator();
                ui.colored_label(NeuralColors::TEXT_SECONDARY, format!("H:{:.2}", entropy));
                ui.colored_label(NeuralColors::TEXT_SECONDARY, format!("V:{:.2}", varentropy));
            });
        });
}

/// Metric card with animated value
pub struct MetricCard {
    label: String,
    animated_value: AnimatedValue,
    unit: String,
    color: Color32,
}

impl MetricCard {
    pub fn new(label: &str, initial: f32, unit: &str, color: Color32) -> Self {
        Self {
            label: label.to_string(),
            animated_value: AnimatedValue::new(initial),
            unit: unit.to_string(),
            color,
        }
    }
    
    pub fn set_value(&mut self, value: f32) {
        self.animated_value.animate_to(value, 400);
    }
    
    pub fn show(&mut self, ui: &mut Ui) {
        let value = self.animated_value.get();
        
        egui::Frame::none()
            .fill(NeuralColors::BG_SURFACE)
            .stroke(Stroke::new(1.0, NeuralColors::BG_HOVER))
            .rounding(Rounding::same(12.0))
            .inner_margin(egui::Margin::same(16.0))
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    ui.colored_label(NeuralColors::TEXT_MUTED, &self.label);
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.colored_label(self.color, 
                            egui::RichText::new(format!("{:.1}", value)).size(28.0).strong()
                        );
                        ui.colored_label(NeuralColors::TEXT_SECONDARY, &self.unit);
                    });
                });
            });
        
        if self.animated_value.is_animating() {
            ui.ctx().request_repaint();
        }
    }
}

/// Sleep phase indicator
pub fn sleep_phase_indicator(ui: &mut Ui, phase: &str, progress: f32) {
    let phases = ["AWAKE", "DROWSY", "LIGHT_SLEEP", "DEEP_SLEEP", "REM", "WAKING"];
    let current_idx = phases.iter().position(|&p| p == phase).unwrap_or(0);
    
    egui::Frame::none()
        .fill(NeuralColors::BG_SURFACE)
        .rounding(Rounding::same(12.0))
        .inner_margin(egui::Margin::same(16.0))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.colored_label(NeuralColors::TEXT_MUTED, "Sleep Cycle");
                ui.add_space(8.0);
                
                // Phase dots
                ui.horizontal(|ui| {
                    for (i, _phase_name) in phases.iter().enumerate() {
                        let is_current = i == current_idx;
                        let is_past = i < current_idx;
                        
                        let color = if is_current {
                            NeuralColors::SLEEP_REM
                        } else if is_past {
                            NeuralColors::ACCENT_DIM
                        } else {
                            NeuralColors::BG_HOVER
                        };
                        
                        let (rect, _) = ui.allocate_exact_size(Vec2::splat(12.0), Sense::hover());
                        ui.painter().circle_filled(rect.center(), if is_current { 6.0 } else { 4.0 }, color);
                        
                        if i < phases.len() - 1 {
                            let (line_rect, _) = ui.allocate_exact_size(Vec2::new(20.0, 12.0), Sense::hover());
                            ui.painter().line_segment(
                                [Pos2::new(line_rect.left(), line_rect.center().y),
                                 Pos2::new(line_rect.right(), line_rect.center().y)],
                                Stroke::new(2.0, if is_past { NeuralColors::ACCENT_DIM } else { NeuralColors::BG_HOVER })
                            );
                        }
                    }
                });
                
                ui.add_space(8.0);
                
                // Progress bar
                let (bar_rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 4.0), Sense::hover());
                ui.painter().rect_filled(bar_rect, Rounding::same(2.0), NeuralColors::BG_HOVER);
                let filled = Rect::from_min_size(
                    bar_rect.min,
                    Vec2::new(bar_rect.width() * progress, bar_rect.height())
                );
                ui.painter().rect_filled(filled, Rounding::same(2.0), NeuralColors::SLEEP_REM);
                
                ui.add_space(4.0);
                ui.colored_label(NeuralColors::TEXT_SECONDARY, phase.replace("_", " "));
            });
        });
}

/// Chat message bubble
pub fn chat_bubble(ui: &mut Ui, is_user: bool, content: &str, timestamp: &str, surprise: Option<f32>) {
    let (bg_color, text_color, align) = if is_user {
        (NeuralColors::ACCENT_DIM, NeuralColors::TEXT_PRIMARY, egui::Align::Max)
    } else {
        (NeuralColors::BG_ELEVATED, NeuralColors::TEXT_PRIMARY, egui::Align::Min)
    };
    
    ui.with_layout(egui::Layout::top_down(align), |ui| {
        egui::Frame::none()
            .fill(bg_color)
            .rounding(Rounding {
                nw: if is_user { 16.0 } else { 4.0 },
                ne: if is_user { 4.0 } else { 16.0 },
                sw: 16.0,
                se: 16.0,
            })
            .inner_margin(egui::Margin::symmetric(16.0, 12.0))
            .show(ui, |ui| {
                ui.set_max_width(500.0);
                ui.colored_label(text_color, content);
                
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.colored_label(NeuralColors::TEXT_MUTED, 
                        egui::RichText::new(timestamp).small());
                    
                    if let Some(s) = surprise {
                        if s > 1.5 {
                            ui.colored_label(NeuralColors::MEMORY_SURPRISE,
                                egui::RichText::new(format!("⚡ {:.1}", s)).small());
                        }
                    }
                });
            });
    });
}

/// Animated input field with glow effect
pub struct GlowingInput {
    text: String,
    is_focused: bool,
    glow_intensity: AnimatedValue,
}

impl GlowingInput {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            is_focused: false,
            glow_intensity: AnimatedValue::new(0.0),
        }
    }
    
    pub fn show(&mut self, ui: &mut Ui) -> Option<String> {
        let mut submitted = None;
        
        // Animate glow
        self.glow_intensity.animate_to(if self.is_focused { 1.0 } else { 0.0 }, 200);
        let glow = self.glow_intensity.get();
        
        let frame = egui::Frame::none()
            .fill(NeuralColors::BG_ELEVATED)
            .stroke(Stroke::new(
                1.0 + glow,
                Color32::from_rgba_unmultiplied(0, 212, 200, (100.0 + glow * 155.0) as u8)
            ))
            .rounding(Rounding::same(24.0))
            .inner_margin(egui::Margin::symmetric(20.0, 14.0));
        
        frame.show(ui, |ui| {
            ui.horizontal(|ui| {
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.text)
                        .hint_text("Ask AVA something...")
                        .frame(false)
                        .desired_width(ui.available_width() - 50.0)
                        .text_color(NeuralColors::TEXT_PRIMARY)
                );
                
                self.is_focused = response.has_focus();
                
                if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if !self.text.trim().is_empty() {
                        submitted = Some(std::mem::take(&mut self.text));
                    }
                }
                
                // Send button
                let send_btn = ui.add(
                    egui::Button::new("→")
                        .fill(if self.text.is_empty() { NeuralColors::BG_HOVER } else { NeuralColors::ACCENT_PRIMARY })
                        .rounding(Rounding::same(20.0))
                        .min_size(Vec2::splat(36.0))
                );
                
                if send_btn.clicked() && !self.text.trim().is_empty() {
                    submitted = Some(std::mem::take(&mut self.text));
                }
            });
        });
        
        if self.glow_intensity.is_animating() {
            ui.ctx().request_repaint();
        }
        
        submitted
    }
    
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// Surprise meter (circular gauge)
pub fn surprise_gauge(ui: &mut Ui, surprise: f32, max_surprise: f32) {
    let size = Vec2::splat(80.0);
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    
    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        let center = rect.center();
        let radius = size.x / 2.0 - 4.0;
        
        // Background arc
        let n_points = 32;
        for i in 0..n_points {
            let angle1 = std::f32::consts::PI * 0.75 + (i as f32 / n_points as f32) * std::f32::consts::PI * 1.5;
            let angle2 = std::f32::consts::PI * 0.75 + ((i + 1) as f32 / n_points as f32) * std::f32::consts::PI * 1.5;
            
            painter.line_segment(
                [
                    center + Vec2::new(angle1.cos(), angle1.sin()) * radius,
                    center + Vec2::new(angle2.cos(), angle2.sin()) * radius,
                ],
                Stroke::new(6.0, NeuralColors::BG_HOVER)
            );
        }
        
        // Filled arc based on surprise
        let fill_ratio = (surprise / max_surprise).clamp(0.0, 1.0);
        let fill_points = (n_points as f32 * fill_ratio) as usize;
        
        let color = if fill_ratio < 0.3 {
            NeuralColors::STATE_FLOW
        } else if fill_ratio < 0.6 {
            NeuralColors::STATE_HESITATION
        } else {
            NeuralColors::MEMORY_SURPRISE
        };
        
        for i in 0..fill_points {
            let angle1 = std::f32::consts::PI * 0.75 + (i as f32 / n_points as f32) * std::f32::consts::PI * 1.5;
            let angle2 = std::f32::consts::PI * 0.75 + ((i + 1) as f32 / n_points as f32) * std::f32::consts::PI * 1.5;
            
            painter.line_segment(
                [
                    center + Vec2::new(angle1.cos(), angle1.sin()) * radius,
                    center + Vec2::new(angle2.cos(), angle2.sin()) * radius,
                ],
                Stroke::new(6.0, color)
            );
        }
        
        // Center text
        painter.text(
            center,
            egui::Align2::CENTER_CENTER,
            format!("{:.1}", surprise),
            egui::FontId::proportional(18.0),
            NeuralColors::TEXT_PRIMARY,
        );
        
        painter.text(
            center + Vec2::new(0.0, 14.0),
            egui::Align2::CENTER_CENTER,
            "surprise",
            egui::FontId::proportional(10.0),
            NeuralColors::TEXT_MUTED,
        );
    }
}
