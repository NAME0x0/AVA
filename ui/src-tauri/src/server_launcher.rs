//! Server Launcher Module for AVA
//!
//! Manages the lifecycle of the Python backend server:
//! - Production: Uses PyInstaller sidecar (ava-server.exe)
//! - Development: Falls back to system Python with server.py
//! - Monitors health and emits status events
//! - Clean shutdown on app exit

use std::path::{Path, PathBuf};
use std::process::{Child, Command as StdCommand, Stdio};
use std::sync::Arc;
use std::time::Duration;
use tauri::api::process::{Command, CommandChild, CommandEvent};
use tauri::{AppHandle, Manager};
use tokio::sync::Mutex;

use crate::tray::{update_tray_status, TrayStatus};

/// Default port for the AVA backend server
pub const DEFAULT_SERVER_PORT: u16 = 8085;

/// Server process state - can be either sidecar or Python subprocess
enum ServerChild {
    Sidecar(CommandChild),
    Python(Child),
}

/// Server process state
struct ServerProcess {
    child: Option<ServerChild>,
    port: u16,
    is_sidecar: bool,
}

/// Server launcher - manages Python backend as sidecar or subprocess
pub struct ServerLauncher {
    process: Arc<Mutex<ServerProcess>>,
    app_handle: AppHandle,
}

/// Status event sent to frontend
#[derive(Clone, Debug, serde::Serialize)]
pub struct ServerStatus {
    pub running: bool,
    pub port: u16,
    pub stage: String,
    pub error: Option<String>,
}

impl ServerLauncher {
    /// Create a new server launcher
    pub fn new(app_handle: AppHandle) -> Self {
        Self {
            process: Arc::new(Mutex::new(ServerProcess {
                child: None,
                port: DEFAULT_SERVER_PORT,
                is_sidecar: false,
            })),
            app_handle,
        }
    }

    /// Start the backend server - tries sidecar first, falls back to Python
    pub async fn start(&self) -> Result<(), String> {
        // Try sidecar first (production mode)
        match self.start_sidecar().await {
            Ok(()) => {
                tracing::info!("Server started via sidecar");
                return Ok(());
            }
            Err(e) => {
                tracing::warn!("Sidecar failed: {}, trying Python fallback", e);
            }
        }

        // Fall back to Python (development mode)
        self.start_python_fallback().await
    }

    /// Start server using Tauri sidecar (PyInstaller executable)
    async fn start_sidecar(&self) -> Result<(), String> {
        let mut process = self.process.lock().await;

        if process.child.is_some() {
            tracing::info!("Server already running");
            return Ok(());
        }

        self.emit_status(ServerStatus {
            running: false,
            port: process.port,
            stage: "Starting sidecar...".to_string(),
            error: None,
        });

        // Spawn sidecar
        let (mut rx, child) = Command::new_sidecar("ava-server")
            .map_err(|e| format!("Sidecar not found: {e}. Is the app built with sidecar?"))?
            .args(["--host", "127.0.0.1", "--port", &process.port.to_string()])
            .spawn()
            .map_err(|e| format!("Failed to spawn sidecar: {e}"))?;

        tracing::info!("Sidecar spawned, monitoring output...");

        // Monitor sidecar output in background
        let app_handle = self.app_handle.clone();
        tauri::async_runtime::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    CommandEvent::Stdout(line) => {
                        tracing::info!("[ava-server] {}", line);
                    }
                    CommandEvent::Stderr(line) => {
                        tracing::warn!("[ava-server] {}", line);
                    }
                    CommandEvent::Error(e) => {
                        tracing::error!("[ava-server] Error: {}", e);
                    }
                    CommandEvent::Terminated(payload) => {
                        tracing::info!("[ava-server] Terminated: {:?}", payload);
                        update_tray_status(&app_handle, TrayStatus::Error);
                    }
                    _ => {}
                }
            }
        });

        process.child = Some(ServerChild::Sidecar(child));
        process.is_sidecar = true;
        let port = process.port;
        drop(process);

        // Wait for server to be ready
        self.wait_for_ready(port).await?;

        update_tray_status(&self.app_handle, TrayStatus::Connected);

        self.emit_status(ServerStatus {
            running: true,
            port,
            stage: "Ready".to_string(),
            error: None,
        });

        Ok(())
    }

    /// Start server using Python (development fallback)
    async fn start_python_fallback(&self) -> Result<(), String> {
        let mut process = self.process.lock().await;

        if process.child.is_some() {
            tracing::info!("Server already running");
            return Ok(());
        }

        self.emit_status(ServerStatus {
            running: false,
            port: process.port,
            stage: "Finding Python...".to_string(),
            error: None,
        });

        let python = self.find_python()?;

        self.emit_status(ServerStatus {
            running: false,
            port: process.port,
            stage: "Finding server script...".to_string(),
            error: None,
        });

        let server_script = self.find_server_script()?;
        let working_dir = self.get_working_dir(&server_script);

        self.emit_status(ServerStatus {
            running: false,
            port: process.port,
            stage: "Starting server...".to_string(),
            error: None,
        });

        tracing::info!(
            "Starting Python server: {:?} {:?} --port {}",
            python,
            server_script,
            process.port
        );

        // Spawn the server process
        let child = StdCommand::new(&python)
            .arg(&server_script)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(process.port.to_string())
            .current_dir(&working_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start server: {e}"))?;

        let pid = child.id();
        process.child = Some(ServerChild::Python(child));
        process.is_sidecar = false;
        tracing::info!("Python server started with PID: {}", pid);

        let port = process.port;
        drop(process);

        // Wait for server to be ready
        self.wait_for_ready(port).await?;

        update_tray_status(&self.app_handle, TrayStatus::Connected);

        self.emit_status(ServerStatus {
            running: true,
            port,
            stage: "Ready".to_string(),
            error: None,
        });

        tracing::info!("Server is ready and accepting connections");
        Ok(())
    }

    /// Find Python executable - checks venv first, then system PATH
    fn find_python(&self) -> Result<PathBuf, String> {
        let resource_dir = self
            .app_handle
            .path_resolver()
            .resource_dir()
            .ok_or_else(|| "Could not find resource directory".to_string())?;

        // Check for bundled venv (production install)
        let venv_python = resource_dir.join("venv").join("Scripts").join("python.exe");
        if venv_python.exists() {
            tracing::info!("Found bundled Python: {:?}", venv_python);
            return Ok(venv_python);
        }

        // Check for development venv relative to project root
        if let Some(project_root) = resource_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
        {
            let dev_venv = project_root.join("venv").join("Scripts").join("python.exe");
            if dev_venv.exists() {
                tracing::info!("Found dev venv Python: {:?}", dev_venv);
                return Ok(dev_venv);
            }
        }

        // Fall back to system Python
        let output = StdCommand::new("python").arg("--version").output();

        match output {
            Ok(out) if out.status.success() => {
                tracing::info!("Using system Python");
                Ok(PathBuf::from("python"))
            }
            _ => {
                let output3 = StdCommand::new("python3").arg("--version").output();
                match output3 {
                    Ok(out) if out.status.success() => {
                        tracing::info!("Using system Python3");
                        Ok(PathBuf::from("python3"))
                    }
                    _ => Err(
                        "Python not found. Please install Python 3.10+ or ensure venv exists."
                            .to_string(),
                    ),
                }
            }
        }
    }

    /// Find server.py location
    fn find_server_script(&self) -> Result<PathBuf, String> {
        let resource_dir = self
            .app_handle
            .path_resolver()
            .resource_dir()
            .ok_or_else(|| "Could not find resource directory".to_string())?;

        // Production: bundled with app resources
        let bundled = resource_dir.join("server.py");
        if bundled.exists() {
            tracing::info!("Found bundled server.py: {:?}", bundled);
            return Ok(bundled);
        }

        // Development: relative to project root
        if let Some(project_root) = resource_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
        {
            let dev_server = project_root.join("server.py");
            if dev_server.exists() {
                tracing::info!("Found dev server.py: {:?}", dev_server);
                return Ok(dev_server);
            }
        }

        Err("server.py not found. Please ensure it exists in the project root.".to_string())
    }

    /// Get the working directory for the server
    fn get_working_dir(&self, server_path: &Path) -> PathBuf {
        server_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    /// Wait for server health endpoint to respond
    async fn wait_for_ready(&self, port: u16) -> Result<(), String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| e.to_string())?;

        let url = format!("http://127.0.0.1:{port}/health");

        for attempt in 1..=30 {
            self.emit_status(ServerStatus {
                running: false,
                port,
                stage: format!("Waiting for server... ({attempt}s)"),
                error: None,
            });

            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!("Server health check passed on attempt {}", attempt);
                    return Ok(());
                }
                Ok(resp) => {
                    tracing::debug!(
                        "Health check attempt {} returned status: {}",
                        attempt,
                        resp.status()
                    );
                }
                Err(e) => {
                    tracing::debug!("Health check attempt {} failed: {}", attempt, e);
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        update_tray_status(&self.app_handle, TrayStatus::Error);

        Err("Server failed to start within 30 seconds. Check if Python dependencies are installed and Ollama is running.".to_string())
    }

    /// Stop the backend server gracefully
    #[allow(dead_code)]
    pub async fn stop(&self) -> Result<(), String> {
        let mut process = self.process.lock().await;

        match process.child.take() {
            Some(ServerChild::Sidecar(child)) => {
                tracing::info!("Stopping sidecar server");
                child.kill().map_err(|e| format!("Failed to kill sidecar: {e}"))?;
            }
            Some(ServerChild::Python(mut child)) => {
                let pid = child.id();
                tracing::info!("Stopping Python server with PID: {}", pid);

                #[cfg(windows)]
                {
                    let _ = StdCommand::new("taskkill")
                        .args(["/PID", &pid.to_string(), "/T", "/F"])
                        .output();
                }

                #[cfg(not(windows))]
                {
                    let _ = child.kill();
                }

                match child.wait() {
                    Ok(status) => {
                        tracing::info!("Server process exited with status: {}", status);
                    }
                    Err(e) => {
                        tracing::warn!("Error waiting for server process: {}", e);
                    }
                }
            }
            None => {
                tracing::info!("No server process to stop");
            }
        }

        update_tray_status(&self.app_handle, TrayStatus::Idle);

        self.emit_status(ServerStatus {
            running: false,
            port: DEFAULT_SERVER_PORT,
            stage: "Stopped".to_string(),
            error: None,
        });

        Ok(())
    }

    /// Check if server is running
    #[allow(dead_code)]
    pub async fn is_running(&self) -> bool {
        let process = self.process.lock().await;
        process.child.is_some()
    }

    /// Emit status event to frontend
    fn emit_status(&self, status: ServerStatus) {
        if let Err(e) = self.app_handle.emit_all("server-status", &status) {
            tracing::warn!("Failed to emit server status: {}", e);
        }
    }
}

impl Clone for ServerLauncher {
    fn clone(&self) -> Self {
        Self {
            process: Arc::clone(&self.process),
            app_handle: self.app_handle.clone(),
        }
    }
}

impl Drop for ServerLauncher {
    fn drop(&mut self) {
        tracing::debug!("ServerLauncher dropped");
    }
}
