//! AVA Neural UI - Lightweight Cognitive Interface
//! 
//! A minimalist yet maximalist UI built with egui for the AVA cognitive system.
//! Designed for extreme efficiency while providing rich visual feedback.

mod app;
mod components;
mod theme;
mod backend;
mod animations;

use eframe::egui;
use tracing_subscriber;

fn main() -> eframe::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_transparent(true)
            .with_decorations(true)
            .with_title("AVA • Adaptive Virtual Agent"),
        ..Default::default()
    };
    
    eframe::run_native(
        "AVA",
        options,
        Box::new(|cc| {
            // Configure fonts and visuals
            theme::configure_fonts(&cc.egui_ctx);
            theme::configure_visuals(&cc.egui_ctx);
            
            Ok(Box::new(app::AvaApp::new(cc)))
        }),
    )
}
