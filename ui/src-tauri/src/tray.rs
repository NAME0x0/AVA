//! System Tray Module for AVA
//!
//! Provides system tray functionality including:
//! - Tray icon with status indicators
//! - Context menu (Show/Hide, Settings, Exit)
//! - Minimize to tray behavior
//! - Status-based icon changes (idle, connected, thinking, error)

use tauri::{
    AppHandle, CustomMenuItem, Manager, SystemTray, SystemTrayEvent, SystemTrayMenu,
    SystemTrayMenuItem, SystemTraySubmenu,
};

/// Connection status for tray icon
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(dead_code)]
pub enum TrayStatus {
    Idle,
    Connected,
    Thinking,
    Error,
}

/// Build the system tray menu
pub fn create_tray_menu() -> SystemTray {
    let show = CustomMenuItem::new("show".to_string(), "Show AVA");
    let hide = CustomMenuItem::new("hide".to_string(), "Hide to Tray");
    let settings = CustomMenuItem::new("settings".to_string(), "Settings");
    let report_bug = CustomMenuItem::new("report_bug".to_string(), "Report Bug...");
    let quit = CustomMenuItem::new("quit".to_string(), "Quit AVA");

    let status_menu = SystemTraySubmenu::new(
        "Status",
        SystemTrayMenu::new()
            .add_item(CustomMenuItem::new("status_backend".to_string(), "Backend: Checking...").disabled())
            .add_item(CustomMenuItem::new("status_memory".to_string(), "Memory: --").disabled()),
    );

    let tray_menu = SystemTrayMenu::new()
        .add_item(show)
        .add_item(hide)
        .add_native_item(SystemTrayMenuItem::Separator)
        .add_submenu(status_menu)
        .add_native_item(SystemTrayMenuItem::Separator)
        .add_item(settings)
        .add_item(report_bug)
        .add_native_item(SystemTrayMenuItem::Separator)
        .add_item(quit);

    SystemTray::new().with_menu(tray_menu)
}

/// Handle system tray events
pub fn handle_tray_event(app: &AppHandle, event: SystemTrayEvent) {
    match event {
        SystemTrayEvent::LeftClick { .. } => {
            // Toggle window visibility on left click
            if let Some(window) = app.get_window("main") {
                if window.is_visible().unwrap_or(false) {
                    let _ = window.hide();
                } else {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
        }
        SystemTrayEvent::MenuItemClick { id, .. } => match id.as_str() {
            "show" => {
                if let Some(window) = app.get_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
            "hide" => {
                if let Some(window) = app.get_window("main") {
                    let _ = window.hide();
                }
            }
            "settings" => {
                // Emit event to open settings in frontend
                let _ = app.emit_all("open-settings", ());
            }
            "report_bug" => {
                // Emit event to open bug report dialog
                let _ = app.emit_all("open-bug-report", ());
            }
            "quit" => {
                std::process::exit(0);
            }
            _ => {}
        },
        _ => {}
    }
}

/// Update tray icon based on status
#[allow(dead_code)]
pub fn update_tray_status(app: &AppHandle, status: TrayStatus) {
    let _icon_path = match status {
        TrayStatus::Idle => "icons/tray/idle.ico",
        TrayStatus::Connected => "icons/tray/connected.ico",
        TrayStatus::Thinking => "icons/tray/thinking.ico",
        TrayStatus::Error => "icons/tray/error.ico",
    };

    // TODO: Implement icon switching when tray icons are created
    // app.tray_handle().set_icon(tauri::Icon::File(icon_path.into())).ok();

    // Update status menu item
    if let Some(tray) = app.tray_handle().try_get_item("status_backend") {
        let status_text = match status {
            TrayStatus::Idle => "Backend: Idle",
            TrayStatus::Connected => "Backend: Connected",
            TrayStatus::Thinking => "Backend: Processing...",
            TrayStatus::Error => "Backend: Error",
        };
        let _ = tray.set_title(status_text);
    }
}

/// Update memory status in tray
#[allow(dead_code)]
pub fn update_memory_status(app: &AppHandle, used_mb: u64, total_mb: u64) {
    if let Some(tray) = app.tray_handle().try_get_item("status_memory") {
        let status_text = format!("Memory: {used_mb} / {total_mb} MB");
        let _ = tray.set_title(&status_text);
    }
}
