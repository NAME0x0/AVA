"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Loader2, AlertCircle, RefreshCw, ExternalLink, Settings, CheckCircle } from "lucide-react";
import { useCoreStore } from "@/stores/core";

// Check if we're in Tauri environment
const isTauri = typeof window !== "undefined" && "__TAURI__" in window;

// Default backend URL - embedded server runs on localhost
const DEFAULT_BACKEND_URL = "http://localhost:8085";

interface HealthResponse {
  status: string;
  service: string;
  version: string;
  ollama_status?: string;
  ollama_error?: string;
}

interface ServerStartupProps {
  onReady: () => void;
}

export function ServerStartup({ onReady }: ServerStartupProps) {
  const { backendUrl, setSettingsPanelOpen } = useCoreStore();
  const [stage, setStage] = useState<"checking" | "connecting" | "ready" | "error">("checking");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [healthInfo, setHealthInfo] = useState<HealthResponse | null>(null);
  const [attempts, setAttempts] = useState(0);
  const [ollamaError, setOllamaError] = useState<string | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const maxAttempts = 30; // 30 attempts = 1 minute at 2s intervals

  const effectiveUrl = backendUrl || DEFAULT_BACKEND_URL;

  // Check backend health
  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      const response = await fetch(`${effectiveUrl}/health`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });

      if (response.ok) {
        const data: HealthResponse = await response.json();
        setHealthInfo(data);

        // Check if Ollama is connected
        if (data.ollama_status === "connected") {
          setStage("ready");
          setOllamaError(null);
          return true;
        } else {
          // Server running but Ollama not connected - show warning but proceed
          console.warn("[ServerStartup] Ollama not connected:", data.ollama_error);
          setOllamaError(data.ollama_error || "Ollama not connected");
          setStage("ready");
          return true;
        }
      }
    } catch (err) {
      console.debug("[ServerStartup] Health check failed:", err);
    }
    return false;
  }, [effectiveUrl]);

  // Start polling for server
  const startPolling = useCallback(() => {
    if (pollIntervalRef.current) return;

    setStage("connecting");
    setAttempts(0);

    // Check immediately
    checkHealth().then((success) => {
      if (success) {
        setTimeout(onReady, 500); // Brief delay to show success state
        return;
      }
    });

    // Poll every 2 seconds
    pollIntervalRef.current = setInterval(async () => {
      setAttempts((prev) => {
        const next = prev + 1;
        if (next >= maxAttempts) {
          // Stop polling after max attempts
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
          setStage("error");
          setErrorMessage("Server not responding. Please check that AVA is running properly.");
        }
        return next;
      });

      const success = await checkHealth();
      if (success) {
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
        setTimeout(onReady, 500);
      }
    }, 2000);
  }, [checkHealth, onReady]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, []);

  // Start checking on mount
  useEffect(() => {
    // In Tauri mode, the embedded server starts automatically
    // In browser mode, check once and proceed
    checkHealth().then((success) => {
      if (success) {
        onReady();
      } else {
        // Start polling for server
        startPolling();
      }
    });
  }, [checkHealth, onReady, startPolling]);

  // Retry handler
  const handleRetry = useCallback(() => {
    setErrorMessage(null);
    setOllamaError(null);
    setAttempts(0);
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    startPolling();
  }, [startPolling]);

  // Open settings panel
  const handleOpenSettings = useCallback(() => {
    setSettingsPanelOpen(true);
  }, [setSettingsPanelOpen]);

  // Open external link
  const openLink = useCallback((url: string) => {
    window.open(url, "_blank");
  }, []);

  // Skip to app (dev mode or if user wants to configure manually)
  const handleSkip = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    onReady();
  }, [onReady]);

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
              scale: stage === "ready" ? [1, 1.1, 1] : [1, 1.05, 1],
              opacity: [0.8, 1, 0.8],
            }}
            transition={{
              duration: stage === "ready" ? 0.5 : 2,
              repeat: stage === "ready" ? 0 : Infinity,
              ease: "easeInOut",
            }}
            className="relative"
          >
            <Brain
              className={`w-20 h-20 ${
                stage === "ready" ? "text-state-flow" : "text-cortex-active"
              }`}
            />
            <motion.div
              className={`absolute inset-0 rounded-full blur-xl ${
                stage === "ready" ? "bg-state-flow/20" : "bg-cortex-active/20"
              }`}
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
              {stage === "checking" && "Checking connection..."}
              {stage === "connecting" && `Connecting to backend (${attempts}/${maxAttempts})...`}
              {stage === "ready" && "Connected!"}
              {stage === "error" && "Connection failed"}
            </p>
          </div>

          {/* Status */}
          <div className="flex flex-col items-center gap-4 w-full">
            {stage === "error" ? (
              <>
                <AlertCircle className="w-8 h-8 text-status-error" />
                <div className="text-center">
                  <p className="text-status-error font-medium mb-2">Cannot reach server</p>
                  <p className="text-sm text-text-muted max-w-xs">{errorMessage}</p>
                </div>
              </>
            ) : stage === "ready" ? (
              <>
                <CheckCircle className="w-8 h-8 text-state-flow" />
                <div className="text-center">
                  <p className="text-state-flow font-medium">Connected</p>
                  {healthInfo && (
                    <p className="text-xs text-text-muted">
                      AVA v{healthInfo.version} | Ollama: {healthInfo.ollama_status || "unknown"}
                    </p>
                  )}
                </div>
              </>
            ) : (
              <>
                <Loader2 className="w-8 h-8 text-cortex-active animate-spin" />
                <p className="text-text-secondary">Looking for AVA server at {effectiveUrl}</p>
              </>
            )}
          </div>

          {/* Progress Bar */}
          {(stage === "checking" || stage === "connecting") && (
            <div className="w-full h-1 bg-neural-surface rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-medulla-active to-cortex-active"
                initial={{ width: "0%" }}
                animate={{
                  width: `${Math.min((attempts / maxAttempts) * 100, 95)}%`,
                }}
                transition={{ duration: 0.5 }}
              />
            </div>
          )}

          {/* Error Actions */}
          {stage === "error" && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="w-full space-y-4"
            >
              {/* Troubleshooting for unified app */}
              <div className="p-4 bg-cortex-active/10 rounded-lg border border-cortex-active/30">
                <p className="text-sm text-cortex-active font-medium mb-2">The embedded server did not start</p>
                <p className="text-xs text-text-muted">
                  This usually means Ollama is not running. AVA requires Ollama to be installed and running.
                </p>
              </div>

              {/* Troubleshooting Steps */}
              <div className="p-4 bg-neural-surface rounded-lg border border-neural-hover">
                <h3 className="text-sm font-medium text-text-primary mb-3">Quick Setup</h3>
                <ul className="space-y-2 text-xs text-text-muted">
                  <li className="flex items-start gap-2">
                    <span className="text-cortex-active">1.</span>
                    <span>
                      Install Ollama if you haven&apos;t already
                      <button
                        onClick={() => openLink("https://ollama.ai/")}
                        className="ml-1 text-cortex-active hover:underline inline-flex items-center gap-0.5"
                      >
                        Download <ExternalLink className="w-3 h-3" />
                      </button>
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-cortex-active">2.</span>
                    <span>
                      Run{" "}
                      <code className="px-1.5 py-0.5 bg-neural-void rounded text-text-secondary">
                        ollama serve
                      </code>{" "}
                      in a terminal
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-cortex-active">3.</span>
                    <span>
                      Pull a model:{" "}
                      <code className="px-1.5 py-0.5 bg-neural-void rounded text-text-secondary">
                        ollama pull gemma3:4b
                      </code>
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-cortex-active">4.</span>
                    <span>Click Retry below</span>
                  </li>
                </ul>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-2">
                <button
                  onClick={handleRetry}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-cortex-active/20 hover:bg-cortex-active/30 text-cortex-active rounded-lg transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  Retry
                </button>
                <button
                  onClick={handleOpenSettings}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-neural-hover hover:bg-neural-surface text-text-secondary rounded-lg transition-colors"
                >
                  <Settings className="w-4 h-4" />
                  Settings
                </button>
              </div>
            </motion.div>
          )}

          {/* Ollama Warning (when server connected but Ollama has issues) */}
          {stage === "ready" && ollamaError && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="w-full p-3 bg-amber-500/10 rounded-lg border border-amber-500/30"
            >
              <p className="text-xs text-amber-400">
                <span className="font-medium">Warning:</span> {ollamaError}
              </p>
            </motion.div>
          )}

          {/* Skip Button (always available for manual configuration) */}
          {stage !== "ready" && (
            <button
              onClick={handleSkip}
              className="text-xs text-text-muted hover:text-text-secondary transition-colors"
            >
              {process.env.NODE_ENV === "development"
                ? "Skip (dev mode)"
                : "Configure manually"}
            </button>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
