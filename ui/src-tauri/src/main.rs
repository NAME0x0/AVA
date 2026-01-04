//! AVA Neural Interface - Unified Application
//!
//! A single, portable application that includes:
//! - Embedded HTTP server (Rust, replaces Python backend)
//! - Native desktop UI (Tauri)
//! - System tray integration
//! - No external dependencies except Ollama
//!
//! This is the production entry point for the distributed application.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;
mod bug_report;
mod commands;
mod engine;
mod state;
mod tray;

#[cfg(test)]
mod tests;

use engine::config::AppConfig;
use std::sync::Arc;
use tauri::Manager;
use tauri_plugin_autostart::MacosLauncher;
use tokio::sync::Mutex;
use tracing::{error, info};

/// Application version
const VERSION: &str = "4.1.0";

fn main() {
    // Initialize logging with environment filter
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("ava=info".parse().expect("valid directive"))
                .add_directive("tower_http=debug".parse().expect("valid directive")),
        )
        .init();

    info!("Starting AVA Neural Interface v{}", VERSION);
    info!("Architecture: Unified Rust (Cortex-Medulla)");

    tauri::Builder::default()
        // Single instance plugin - prevent multiple windows
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            // Focus existing window when trying to open another instance
            if let Some(window) = app.get_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        // Autostart plugin - optional start on boot
        .plugin(tauri_plugin_autostart::init(
            MacosLauncher::LaunchAgent,
            Some(vec!["--minimized"]),
        ))
        // System tray
        .system_tray(tray::create_tray_menu())
        .on_system_tray_event(tray::handle_tray_event)
        // Window close behavior - hide to tray instead of quit
        .on_window_event(|event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event.event() {
                // Hide window instead of closing
                let _ = event.window().hide();
                api.prevent_close();
            }
        })
        .setup(|app| {
            info!("Setting up AVA application...");

            // Initialize app state for Tauri
            app.manage(state::AppState::new());

            // Load configuration
            let config = AppConfig::load();
            let port = config.server.port;

            // Store server handle for cleanup
            let server_handle: Arc<Mutex<Option<engine::server::ServerHandle>>> =
                Arc::new(Mutex::new(None));
            let server_handle_clone = server_handle.clone();
            app.manage(server_handle);

            // Start the embedded HTTP server
            let app_handle = app.handle();
            tauri::async_runtime::spawn(async move {
                info!("Starting embedded AVA server...");

                // Check if port is available, find alternative if not
                let actual_port = if engine::server::is_port_available(port) {
                    port
                } else {
                    let new_port = engine::server::find_available_port(port);
                    info!("Port {} in use, using {} instead", port, new_port);
                    new_port
                };

                // Update config with actual port
                let mut config = config;
                config.server.port = actual_port;

                match engine::start_embedded_server(config).await {
                    Ok(handle) => {
                        info!("Embedded server started on port {}", handle.port);

                        // Store handle for cleanup
                        *server_handle_clone.lock().await = Some(handle);

                        // Emit success event to frontend
                        let _ = app_handle.emit_all(
                            "server-status",
                            serde_json::json!({
                                "running": true,
                                "port": actual_port,
                                "stage": "Ready",
                                "error": null
                            }),
                        );
                    }
                    Err(e) => {
                        error!("Failed to start embedded server: {}", e);

                        // Emit error event to frontend
                        let _ = app_handle.emit_all(
                            "server-status",
                            serde_json::json!({
                                "running": false,
                                "port": actual_port,
                                "stage": "Error",
                                "error": e
                            }),
                        );
                    }
                }
            });

            // Start backend connection monitor (for status updates)
            let app_handle = app.handle();
            tauri::async_runtime::spawn(async move {
                backend::start_connection_monitor(app_handle).await;
            });

            info!("AVA setup complete");
            Ok(())
        })
        // Handle app exit - stop server gracefully
        .on_menu_event(|event| {
            if event.menu_item_id() == "quit" {
                info!("Quit requested, shutting down...");
                std::process::exit(0);
            }
        })
        .invoke_handler(tauri::generate_handler![
            // Existing commands
            commands::send_message,
            commands::get_system_state,
            commands::get_cognitive_state,
            commands::get_memory_stats,
            commands::get_belief_state,
            commands::force_cortex,
            commands::force_sleep,
            commands::set_backend_url,
            commands::greet,
            // Bug report commands
            bug_report::create_bug_report,
            bug_report::open_bug_report_url,
            bug_report::is_error_reportable,
            // New commands for embedded server
            get_server_port,
            get_app_version,
        ])
        .run(tauri::generate_context!())
        .expect("error while running AVA");
}

/// Get the server port (useful for frontend)
#[tauri::command]
fn get_server_port() -> u16 {
    // Default port - frontend will also check via health endpoint
    8085
}

/// Get application version
#[tauri::command]
fn get_app_version() -> String {
    VERSION.to_string()
}
