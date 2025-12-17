//! Theme configuration - Minimalist dark aesthetic with neural accents

use eframe::egui::{self, Color32, FontFamily, FontId, Rounding, Stroke, TextStyle, Visuals};

/// Neural accent colors
pub struct NeuralColors;

impl NeuralColors {
    // Primary palette - Deep space blacks
    pub const BG_VOID: Color32 = Color32::from_rgb(8, 8, 12);
    pub const BG_SURFACE: Color32 = Color32::from_rgb(14, 14, 20);
    pub const BG_ELEVATED: Color32 = Color32::from_rgb(22, 22, 30);
    pub const BG_HOVER: Color32 = Color32::from_rgb(30, 30, 42);
    
    // Text hierarchy
    pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(240, 240, 245);
    pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(160, 160, 175);
    pub const TEXT_MUTED: Color32 = Color32::from_rgb(100, 100, 120);
    
    // Neural accent - Electric cyan/teal
    pub const ACCENT_PRIMARY: Color32 = Color32::from_rgb(0, 212, 200);
    pub const ACCENT_DIM: Color32 = Color32::from_rgb(0, 150, 140);
    
    // Runtime-computed colors (not const)
    pub fn accent_glow() -> Color32 {
        Color32::from_rgba_unmultiplied(0, 212, 200, 80)
    }
    
    // Cognitive state colors
    pub const STATE_FLOW: Color32 = Color32::from_rgb(80, 200, 120);      // Green - confident
    pub const STATE_HESITATION: Color32 = Color32::from_rgb(255, 200, 80); // Amber - uncertain
    pub const STATE_CONFUSION: Color32 = Color32::from_rgb(255, 100, 100); // Red - lost
    pub const STATE_CREATIVE: Color32 = Color32::from_rgb(180, 100, 255);  // Purple - exploring
    
    // Sleep phases
    pub const SLEEP_DROWSY: Color32 = Color32::from_rgb(100, 100, 160);
    pub const SLEEP_LIGHT: Color32 = Color32::from_rgb(80, 80, 140);
    pub const SLEEP_DEEP: Color32 = Color32::from_rgb(60, 60, 120);
    pub const SLEEP_REM: Color32 = Color32::from_rgb(120, 80, 180);
    
    // Memory/Learning
    pub const MEMORY_EPISODIC: Color32 = Color32::from_rgb(100, 180, 255);
    pub const MEMORY_SURPRISE: Color32 = Color32::from_rgb(255, 150, 50);
    
    // Status
    pub const SUCCESS: Color32 = Color32::from_rgb(80, 200, 120);
    pub const WARNING: Color32 = Color32::from_rgb(255, 180, 50);
    pub const ERROR: Color32 = Color32::from_rgb(255, 80, 80);
}

/// Configure custom fonts
pub fn configure_fonts(ctx: &egui::Context) {
    let fonts = egui::FontDefinitions::default();
    
    // Use default fonts but adjust sizing
    let mut style = (*ctx.style()).clone();
    
    style.text_styles = [
        (TextStyle::Heading, FontId::new(28.0, FontFamily::Proportional)),
        (TextStyle::Name("H2".into()), FontId::new(22.0, FontFamily::Proportional)),
        (TextStyle::Name("H3".into()), FontId::new(18.0, FontFamily::Proportional)),
        (TextStyle::Body, FontId::new(14.0, FontFamily::Proportional)),
        (TextStyle::Monospace, FontId::new(13.0, FontFamily::Monospace)),
        (TextStyle::Button, FontId::new(14.0, FontFamily::Proportional)),
        (TextStyle::Small, FontId::new(11.0, FontFamily::Proportional)),
    ].into();
    
    ctx.set_style(style);
    ctx.set_fonts(fonts);
}

/// Configure dark neural visuals
pub fn configure_visuals(ctx: &egui::Context) {
    let mut visuals = Visuals::dark();
    
    // Window
    visuals.window_fill = NeuralColors::BG_SURFACE;
    visuals.window_stroke = Stroke::new(1.0, NeuralColors::BG_ELEVATED);
    visuals.window_rounding = Rounding::same(12.0);
    
    // Panel
    visuals.panel_fill = NeuralColors::BG_VOID;
    
    // Widgets
    visuals.widgets.noninteractive.bg_fill = NeuralColors::BG_SURFACE;
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, NeuralColors::TEXT_SECONDARY);
    visuals.widgets.noninteractive.rounding = Rounding::same(8.0);
    
    visuals.widgets.inactive.bg_fill = NeuralColors::BG_ELEVATED;
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, NeuralColors::TEXT_PRIMARY);
    visuals.widgets.inactive.rounding = Rounding::same(8.0);
    
    visuals.widgets.hovered.bg_fill = NeuralColors::BG_HOVER;
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.5, NeuralColors::ACCENT_PRIMARY);
    visuals.widgets.hovered.rounding = Rounding::same(8.0);
    
    visuals.widgets.active.bg_fill = NeuralColors::ACCENT_DIM;
    visuals.widgets.active.fg_stroke = Stroke::new(2.0, NeuralColors::ACCENT_PRIMARY);
    visuals.widgets.active.rounding = Rounding::same(8.0);
    
    // Selection
    visuals.selection.bg_fill = NeuralColors::accent_glow();
    visuals.selection.stroke = Stroke::new(1.0, NeuralColors::ACCENT_PRIMARY);
    
    // Hyperlinks
    visuals.hyperlink_color = NeuralColors::ACCENT_PRIMARY;
    
    // Extreme rounding for buttons
    visuals.widgets.inactive.rounding = Rounding::same(8.0);
    
    ctx.set_visuals(visuals);
}

/// Animation easing functions
pub mod easing {
    pub fn ease_out_expo(t: f32) -> f32 {
        if t >= 1.0 { 1.0 } else { 1.0 - 2.0_f32.powf(-10.0 * t) }
    }
    
    pub fn ease_out_cubic(t: f32) -> f32 {
        1.0 - (1.0 - t).powi(3)
    }
    
    pub fn ease_in_out_quad(t: f32) -> f32 {
        if t < 0.5 { 2.0 * t * t } else { 1.0 - (-2.0 * t + 2.0).powi(2) / 2.0 }
    }
    
    pub fn pulse(t: f32) -> f32 {
        (t * std::f32::consts::PI * 2.0).sin() * 0.5 + 0.5
    }
}
