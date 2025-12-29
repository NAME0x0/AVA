//! Server Launcher Module for AVA
//!
//! Manages the lifecycle of the Python backend server as a helper function:
//! - Detects Python/venv installation
//! - Spawns server.py process
//! - Monitors health and emits status events
//! - Clean shutdown on app exit

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;
use tauri::{AppHandle, Manager};
use tokio::sync::Mutex;

use crate::tray::{update_tray_status, TrayStatus};

/// Server process state
struct ServerProcess {
    child: Option<Child>,
    port: u16,
}

/// Server launcher - manages Python backend as a helper process
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
                port: 8085,
            })),
            app_handle,
        }
    }

    /// Find Python executable - checks venv first, then system PATH
    fn find_python(&self) -> Result<PathBuf, String> {
        // Get the app's resource directory
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
        // Resource dir is usually ui/src-tauri/target/debug or release
        // Project root is 3 levels up from src-tauri
        if let Some(project_root) = resource_dir
            .parent() // target
            .and_then(|p| p.parent()) // src-tauri
            .and_then(|p| p.parent()) // ui
            .and_then(|p| p.parent())
        // project root
        {
            let dev_venv = project_root.join("venv").join("Scripts").join("python.exe");
            if dev_venv.exists() {
                tracing::info!("Found dev venv Python: {:?}", dev_venv);
                return Ok(dev_venv);
            }
        }

        // Fall back to system Python
        let output = Command::new("python").arg("--version").output();

        match output {
            Ok(out) if out.status.success() => {
                tracing::info!("Using system Python");
                Ok(PathBuf::from("python"))
            }
            _ => {
                // Try python3
                let output3 = Command::new("python3").arg("--version").output();
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

    /// Start the backend server
    pub async fn start(&self) -> Result<(), String> {
        let mut process = self.process.lock().await;

        // Don't start if already running
        if process.child.is_some() {
            tracing::info!("Server already running");
            return Ok(());
        }

        // Emit starting status
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
            "Starting server: {:?} {:?} --port {}",
            python,
            server_script,
            process.port
        );

        // Spawn the server process
        let child = Command::new(&python)
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
        process.child = Some(child);
        tracing::info!("Server process started with PID: {}", pid);

        // Release lock during health check
        let port = process.port;
        drop(process);

        // Wait for server to be ready
        self.wait_for_ready(port).await?;

        // Update tray status
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

            // Check if process is still running
            let process = self.process.lock().await;
            if let Some(ref child) = process.child {
                // Try to check if process exited
                // Note: We can't easily check exit status without consuming the child
                tracing::debug!("Process PID {} still registered", child.id());
            }
        }

        // Server failed to start
        update_tray_status(&self.app_handle, TrayStatus::Error);

        Err("Server failed to start within 30 seconds. Check if Python dependencies are installed and Ollama is running.".to_string())
    }

    /// Stop the backend server gracefully
    #[allow(dead_code)]
    pub async fn stop(&self) -> Result<(), String> {
        let mut process = self.process.lock().await;

        if let Some(mut child) = process.child.take() {
            let pid = child.id();
            tracing::info!("Stopping server process with PID: {}", pid);

            // On Windows, use taskkill for clean shutdown of process tree
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .args(["/PID", &pid.to_string(), "/T", "/F"])
                    .output();
            }

            // On Unix, send SIGTERM
            #[cfg(not(windows))]
            {
                let _ = child.kill();
            }

            // Wait for process to exit
            match child.wait() {
                Ok(status) => {
                    tracing::info!("Server process exited with status: {}", status);
                }
                Err(e) => {
                    tracing::warn!("Error waiting for server process: {}", e);
                }
            }
        }

        update_tray_status(&self.app_handle, TrayStatus::Idle);

        self.emit_status(ServerStatus {
            running: false,
            port: 8085,
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

// Ensure server is stopped when launcher is dropped
impl Drop for ServerLauncher {
    fn drop(&mut self) {
        // We can't do async cleanup in drop, but the process will be killed
        // when the Child handle is dropped anyway
        tracing::debug!("ServerLauncher dropped");
    }
}
