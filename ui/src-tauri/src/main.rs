//! AVA Neural Interface - Tauri Backend
//!
//! Rust backend for the Cortex-Medulla UI providing:
//! - Native performance for state management
//! - System tray integration
//! - IPC bridge to Python backend
//! - Hardware monitoring
//! - Automated bug reporting

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;
mod bug_report;
mod commands;
mod state;
mod tray;

use tauri::Manager;
use tauri_plugin_autostart::MacosLauncher;
use tracing_subscriber;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

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
        .on_system_tray_event(|app, event| tray::handle_tray_event(app, event))
        // Window close behavior - hide to tray instead of quit
        .on_window_event(|event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event.event() {
                // Hide window instead of closing
                event.window().hide().unwrap();
                api.prevent_close();
            }
        })
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
            // Existing commands
            commands::send_message,
            commands::get_system_state,
            commands::get_cognitive_state,
            commands::get_memory_stats,
            commands::force_cortex,
            commands::force_sleep,
            commands::set_backend_url,
            commands::greet,
            // Bug report commands
            bug_report::create_bug_report,
            bug_report::open_bug_report_url,
            bug_report::is_error_reportable,
        ])
        .run(tauri::generate_context!())
        .expect("error while running AVA");
}
