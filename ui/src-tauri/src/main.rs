//! AVA Neural Interface - Tauri Backend
//!
//! Rust backend for the Cortex-Medulla UI providing:
//! - Native performance for state management
//! - System tray integration
//! - IPC bridge to Python backend
//! - Hardware monitoring

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod backend;
mod state;

use tauri::Manager;
use tracing_subscriber;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    tauri::Builder::default()
        .setup(|app| {
            // Initialize app state
            app.manage(state::AppState::new());
            
            // Start backend connection monitor
            let app_handle = app.handle();
            tauri::async_runtime::spawn(async move {
                backend::start_connection_monitor(app_handle).await;
            });
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::send_message,
            commands::get_system_state,
            commands::get_cognitive_state,
            commands::get_memory_stats,
            commands::force_cortex,
            commands::force_sleep,
            commands::set_backend_url,
            commands::greet,
        ])
        .run(tauri::generate_context!())
        .expect("error while running AVA");
}
