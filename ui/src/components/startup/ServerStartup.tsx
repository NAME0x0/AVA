"use client";

import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Loader2, AlertCircle, RefreshCw, ExternalLink } from "lucide-react";

// Check if we're in Tauri environment
const isTauri = typeof window !== "undefined" && "__TAURI__" in window;

interface ServerStatus {
  running: boolean;
  port: number;
  stage: string;
  error: string | null;
}

interface ServerStartupProps {
  onReady: () => void;
}

export function ServerStartup({ onReady }: ServerStartupProps) {
  const [status, setStatus] = useState<ServerStatus>({
    running: false,
    port: 8085,
    stage: "Initializing...",
    error: null,
  });
  const [showTroubleshooting, setShowTroubleshooting] = useState(false);

  // In browser mode, skip startup screen
  useEffect(() => {
    if (!isTauri) {
      onReady();
    }
  }, [onReady]);

  // Listen for server status events from Tauri
  useEffect(() => {
    if (!isTauri) return;

    let unlisten: (() => void) | undefined;

    const setupListener = async () => {
      try {
        const { listen } = await import("@tauri-apps/api/event");

        unlisten = await listen<ServerStatus>("server-status", (event) => {
          setStatus(event.payload);

          // If server is running, notify parent
          if (event.payload.running) {
            // Small delay to show "Ready" state before transitioning
            setTimeout(() => {
              onReady();
            }, 500);
          }

          // Show troubleshooting if error occurs
          if (event.payload.error) {
            setShowTroubleshooting(true);
          }
        });
      } catch (err) {
        console.error("Failed to setup server status listener:", err);
        // In dev mode, proceed anyway after a timeout
        setTimeout(onReady, 3000);
      }
    };

    setupListener();

    return () => {
      if (unlisten) unlisten();
    };
  }, [onReady]);

  // Retry starting the server
  const handleRetry = useCallback(async () => {
    if (!isTauri) return;

    setStatus((s) => ({ ...s, error: null, stage: "Retrying..." }));
    setShowTroubleshooting(false);

    try {
      const { invoke } = await import("@tauri-apps/api/tauri");
      // This would need a corresponding command in the Rust backend
      // For now, we just reload the window which restarts everything
      window.location.reload();
    } catch (err) {
      console.error("Failed to retry:", err);
    }
  }, []);

  // Open external links
  const openLink = useCallback(async (url: string) => {
    if (isTauri) {
      try {
        const { open } = await import("@tauri-apps/api/shell");
        await open(url);
      } catch {
        window.open(url, "_blank");
      }
    } else {
      window.open(url, "_blank");
    }
  }, []);

  // Don't render in browser mode
  if (!isTauri) {
    return null;
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-neural-void"
      >
        <div className="flex flex-col items-center gap-8 p-8 max-w-md">
          {/* Logo */}
          <motion.div
            animate={{
              scale: [1, 1.05, 1],
              opacity: [0.8, 1, 0.8],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut",
            }}
            className="relative"
          >
            <Brain className="w-20 h-20 text-cortex-active" />
            <motion.div
              className="absolute inset-0 rounded-full bg-cortex-active/20 blur-xl"
              animate={{
                scale: [1, 1.3, 1],
                opacity: [0.3, 0.5, 0.3],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
          </motion.div>

          {/* Title */}
          <div className="text-center">
            <h1 className="text-2xl font-semibold text-text-primary mb-2">
              AVA Neural Interface
            </h1>
            <p className="text-sm text-text-secondary">
              Starting Cortex-Medulla Backend
            </p>
          </div>

          {/* Status */}
          <div className="flex flex-col items-center gap-4 w-full">
            {status.error ? (
              <>
                <AlertCircle className="w-8 h-8 text-status-error" />
                <div className="text-center">
                  <p className="text-status-error font-medium mb-2">
                    Failed to start server
                  </p>
                  <p className="text-sm text-text-muted max-w-xs">
                    {status.error}
                  </p>
                </div>
              </>
            ) : (
              <>
                <Loader2 className="w-8 h-8 text-cortex-active animate-spin" />
                <p className="text-text-secondary">{status.stage}</p>
              </>
            )}
          </div>

          {/* Progress Bar */}
          {!status.error && (
            <div className="w-full h-1 bg-neural-surface rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-medulla-active to-cortex-active"
                initial={{ width: "0%" }}
                animate={{
                  width: status.running
                    ? "100%"
                    : status.stage.includes("Waiting")
                    ? "70%"
                    : status.stage.includes("Starting")
                    ? "50%"
                    : status.stage.includes("Finding")
                    ? "30%"
                    : "10%",
                }}
                transition={{ duration: 0.5 }}
              />
            </div>
          )}

          {/* Troubleshooting */}
          {showTroubleshooting && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="w-full p-4 bg-neural-surface rounded-lg border border-neural-hover"
            >
              <h3 className="text-sm font-medium text-text-primary mb-3">
                Troubleshooting
              </h3>
              <ul className="space-y-2 text-xs text-text-muted">
                <li className="flex items-start gap-2">
                  <span className="text-cortex-active">1.</span>
                  <span>
                    Ensure Python 3.10+ is installed and in PATH
                    <button
                      onClick={() => openLink("https://www.python.org/downloads/")}
                      className="ml-1 text-cortex-active hover:underline inline-flex items-center gap-0.5"
                    >
                      Download <ExternalLink className="w-3 h-3" />
                    </button>
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-cortex-active">2.</span>
                  <span>
                    Ensure Ollama is running with gemma3:4b
                    <button
                      onClick={() => openLink("https://ollama.ai/")}
                      className="ml-1 text-cortex-active hover:underline inline-flex items-center gap-0.5"
                    >
                      Download <ExternalLink className="w-3 h-3" />
                    </button>
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-cortex-active">3.</span>
                  <span>Check if port {status.port} is available</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-cortex-active">4.</span>
                  <span>
                    Run{" "}
                    <code className="px-1 py-0.5 bg-neural-hover rounded text-text-secondary">
                      pip install -r requirements.txt
                    </code>{" "}
                    in the AVA directory
                  </span>
                </li>
              </ul>

              {/* Retry Button */}
              <button
                onClick={handleRetry}
                className="mt-4 w-full flex items-center justify-center gap-2 px-4 py-2 bg-cortex-active/20 hover:bg-cortex-active/30 text-cortex-active rounded-lg transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Retry
              </button>
            </motion.div>
          )}

          {/* Skip Button (dev mode) */}
          {process.env.NODE_ENV === "development" && !status.running && (
            <button
              onClick={onReady}
              className="text-xs text-text-muted hover:text-text-secondary transition-colors"
            >
              Skip (dev mode)
            </button>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
