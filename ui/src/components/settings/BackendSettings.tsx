"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Wifi,
  WifiOff,
  Loader2,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Server,
  Clock,
  Cpu,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Search,
} from "lucide-react";
import { useCoreStore } from "@/stores/core";
import { cn } from "@/lib/utils";

interface ServerInfo {
  version: string;
  python?: {
    version: string;
    executable: string;
  };
  platform?: {
    system: string;
    node: string;
  };
  server?: {
    uptime_seconds: number;
    start_time: string;
  };
  ollama?: {
    status: string;
    models?: string[];
    version?: string;
    error?: string;
  };
  ava_initialized?: boolean;
}

interface ConnectionTestResult {
  status: "idle" | "testing" | "success" | "error" | "partial";
  backend_reachable: boolean;
  ollama_status: string;
  server_info: ServerInfo | null;
  latency_ms: number | null;
  error: string | null;
}

export function BackendSettings() {
  const { backendUrl, setBackendUrl, wsConnected, connectWebSocket, disconnectWebSocket } =
    useCoreStore();

  const [urlInput, setUrlInput] = useState(backendUrl);
  const [testResult, setTestResult] = useState<ConnectionTestResult>({
    status: "idle",
    backend_reachable: false,
    ollama_status: "unknown",
    server_info: null,
    latency_ms: null,
    error: null,
  });
  const [showDetails, setShowDetails] = useState(false);
  const [showTroubleshooting, setShowTroubleshooting] = useState(false);
  const [scanning, setScanning] = useState(false);

  // Test connection to backend
  const testConnection = useCallback(async (url: string) => {
    setTestResult((prev) => ({ ...prev, status: "testing", error: null }));

    const startTime = performance.now();

    try {
      // Test health endpoint
      const healthResponse = await fetch(`${url}/health`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });

      if (!healthResponse.ok) {
        throw new Error(`Server returned ${healthResponse.status}`);
      }

      const healthData = await healthResponse.json();
      const latency = Math.round(performance.now() - startTime);

      // Try to get system info for more details
      let serverInfo: ServerInfo | null = null;
      try {
        const systemResponse = await fetch(`${url}/system`, {
          method: "GET",
          signal: AbortSignal.timeout(5000),
        });
        if (systemResponse.ok) {
          serverInfo = await systemResponse.json();
        }
      } catch {
        // System endpoint may not exist in older versions
        serverInfo = {
          version: healthData.version || "unknown",
        };
      }

      // Determine overall status
      const ollamaStatus = healthData.ollama_status || serverInfo?.ollama?.status || "unknown";
      const isPartial = ollamaStatus !== "connected";

      setTestResult({
        status: isPartial ? "partial" : "success",
        backend_reachable: true,
        ollama_status: ollamaStatus,
        server_info: serverInfo,
        latency_ms: latency,
        error: isPartial ? "Ollama not connected" : null,
      });

      // Update the backend URL if test succeeded
      setBackendUrl(url);
    } catch (err) {
      setTestResult({
        status: "error",
        backend_reachable: false,
        ollama_status: "unknown",
        server_info: null,
        latency_ms: null,
        error: err instanceof Error ? err.message : "Connection failed",
      });
      setShowTroubleshooting(true);
    }
  }, [setBackendUrl]);

  // Auto-detect server on common ports
  const autoDetectServer = useCallback(async () => {
    setScanning(true);
    const portsToTry = [8085, 8080, 8000, 5000, 3000];
    const hosts = ["localhost", "127.0.0.1"];

    for (const host of hosts) {
      for (const port of portsToTry) {
        const url = `http://${host}:${port}`;
        try {
          const response = await fetch(`${url}/health`, {
            method: "GET",
            signal: AbortSignal.timeout(1000),
          });
          if (response.ok) {
            const data = await response.json();
            if (data.service === "AVA") {
              setUrlInput(url);
              await testConnection(url);
              setScanning(false);
              return;
            }
          }
        } catch {
          // Continue to next port
        }
      }
    }

    setScanning(false);
    setTestResult((prev) => ({
      ...prev,
      status: "error",
      error: "No AVA server found on common ports",
    }));
    setShowTroubleshooting(true);
  }, [testConnection]);

  // Test connection when URL changes
  useEffect(() => {
    // Debounce URL changes
    const timer = setTimeout(() => {
      if (urlInput && urlInput !== backendUrl) {
        testConnection(urlInput);
      }
    }, 500);
    return () => clearTimeout(timer);
  }, [urlInput, backendUrl, testConnection]);

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  return (
    <div className="space-y-4">
      {/* URL Input */}
      <div className="p-4 rounded-lg bg-neural-elevated/50 space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-text-primary">Backend URL</label>
          <StatusBadge result={testResult} wsConnected={wsConnected} />
        </div>

        <div className="flex gap-2">
          <input
            type="url"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            className="flex-1 px-3 py-2 rounded-lg bg-neural-surface border border-neural-hover text-text-primary text-sm focus:outline-none focus:border-accent-primary"
            placeholder="http://localhost:8085"
          />
          <motion.button
            onClick={() => testConnection(urlInput)}
            disabled={testResult.status === "testing"}
            className="px-3 py-2 rounded-lg bg-accent-primary/20 text-accent-primary hover:bg-accent-primary/30 transition-colors disabled:opacity-50"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {testResult.status === "testing" ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4" />
            )}
          </motion.button>
        </div>

        {/* Auto-detect button */}
        <motion.button
          onClick={autoDetectServer}
          disabled={scanning}
          className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-neural-surface text-text-secondary hover:text-text-primary hover:bg-neural-hover transition-colors text-sm"
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
        >
          {scanning ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Scanning...
            </>
          ) : (
            <>
              <Search className="w-4 h-4" />
              Auto-detect Server
            </>
          )}
        </motion.button>
      </div>

      {/* Server Info (when connected) */}
      <AnimatePresence>
        {testResult.server_info && testResult.status !== "error" && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="p-4 rounded-lg bg-neural-elevated/50 space-y-3">
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="w-full flex items-center justify-between text-sm font-medium text-text-primary"
              >
                <div className="flex items-center gap-2">
                  <Server className="w-4 h-4 text-accent-primary" />
                  Server Info
                </div>
                {showDetails ? (
                  <ChevronUp className="w-4 h-4 text-text-muted" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-text-muted" />
                )}
              </button>

              <AnimatePresence>
                {showDetails && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-2 text-sm"
                  >
                    <InfoRow
                      icon={<Server className="w-3 h-3" />}
                      label="Version"
                      value={testResult.server_info.version}
                    />
                    {testResult.latency_ms && (
                      <InfoRow
                        icon={<Clock className="w-3 h-3" />}
                        label="Latency"
                        value={`${testResult.latency_ms}ms`}
                      />
                    )}
                    {testResult.server_info.server?.uptime_seconds && (
                      <InfoRow
                        icon={<Clock className="w-3 h-3" />}
                        label="Uptime"
                        value={formatUptime(testResult.server_info.server.uptime_seconds)}
                      />
                    )}
                    {testResult.server_info.platform?.system && (
                      <InfoRow
                        icon={<Cpu className="w-3 h-3" />}
                        label="Platform"
                        value={testResult.server_info.platform.system}
                      />
                    )}
                    <InfoRow
                      icon={
                        testResult.ollama_status === "connected" ? (
                          <CheckCircle className="w-3 h-3 text-state-flow" />
                        ) : (
                          <AlertTriangle className="w-3 h-3 text-state-confusion" />
                        )
                      }
                      label="Ollama"
                      value={testResult.ollama_status}
                      valueClass={
                        testResult.ollama_status === "connected"
                          ? "text-state-flow"
                          : "text-state-confusion"
                      }
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Connection Actions */}
      <div className="flex gap-2">
        <motion.button
          onClick={connectWebSocket}
          disabled={testResult.status !== "success" && testResult.status !== "partial"}
          className={cn(
            "flex-1 py-2 rounded-lg text-sm font-medium transition-colors",
            wsConnected
              ? "bg-state-flow/20 text-state-flow"
              : "bg-accent-primary/20 text-accent-primary hover:bg-accent-primary/30"
          )}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {wsConnected ? (
            <span className="flex items-center justify-center gap-2">
              <CheckCircle className="w-4 h-4" />
              Connected
            </span>
          ) : (
            "Connect WebSocket"
          )}
        </motion.button>
        {wsConnected && (
          <motion.button
            onClick={disconnectWebSocket}
            className="px-4 py-2 rounded-lg bg-state-confusion/20 text-state-confusion hover:bg-state-confusion/30 transition-colors text-sm"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Disconnect
          </motion.button>
        )}
      </div>

      {/* Troubleshooting (when error) */}
      <AnimatePresence>
        {showTroubleshooting && testResult.status === "error" && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <TroubleshootingGuide error={testResult.error} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function StatusBadge({
  result,
  wsConnected,
}: {
  result: ConnectionTestResult;
  wsConnected: boolean;
}) {
  const getStatusConfig = () => {
    if (result.status === "testing") {
      return { icon: Loader2, text: "Testing...", class: "bg-accent-primary/20 text-accent-primary" };
    }
    if (wsConnected) {
      return { icon: Wifi, text: "Live", class: "bg-state-flow/20 text-state-flow" };
    }
    if (result.status === "success") {
      return { icon: CheckCircle, text: "Ready", class: "bg-state-flow/20 text-state-flow" };
    }
    if (result.status === "partial") {
      return { icon: AlertTriangle, text: "Partial", class: "bg-state-curiosity/20 text-state-curiosity" };
    }
    if (result.status === "error") {
      return { icon: XCircle, text: "Error", class: "bg-state-confusion/20 text-state-confusion" };
    }
    return { icon: WifiOff, text: "Offline", class: "bg-neural-hover text-text-muted" };
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  return (
    <span className={cn("flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium", config.class)}>
      <Icon className={cn("w-3 h-3", result.status === "testing" && "animate-spin")} />
      {config.text}
    </span>
  );
}

function InfoRow({
  icon,
  label,
  value,
  valueClass,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="flex items-center gap-2 text-text-muted">
        {icon}
        {label}
      </span>
      <span className={cn("text-text-secondary", valueClass)}>{value}</span>
    </div>
  );
}

function TroubleshootingGuide({ error }: { error: string | null }) {
  const openLink = (url: string) => {
    window.open(url, "_blank");
  };

  return (
    <div className="p-4 rounded-lg bg-state-confusion/10 border border-state-confusion/30 space-y-3">
      <h4 className="text-sm font-medium text-state-confusion flex items-center gap-2">
        <AlertTriangle className="w-4 h-4" />
        Connection Failed
      </h4>

      {error && (
        <p className="text-xs text-text-muted bg-neural-surface px-2 py-1 rounded font-mono">
          {error}
        </p>
      )}

      <div className="space-y-2 text-xs text-text-muted">
        <div className="p-2 rounded bg-accent-primary/10 border border-accent-primary/30">
          <p className="text-accent-primary font-medium mb-1">Quick fix:</p>
          <p>
            Run{" "}
            <code className="px-1.5 py-0.5 bg-neural-surface rounded text-text-secondary">
              python ava_server.py
            </code>{" "}
            in the AVA directory
          </p>
        </div>

        <p className="font-medium text-text-secondary mt-2">Checklist:</p>
        <ul className="space-y-1.5 ml-3">
          <li className="flex items-start gap-2">
            <span className="text-accent-primary">1.</span>
            <span>
              Ensure Python 3.10+ is installed
              <button
                onClick={() => openLink("https://www.python.org/downloads/")}
                className="ml-1 text-accent-primary hover:underline inline-flex items-center gap-0.5"
              >
                Download <ExternalLink className="w-3 h-3" />
              </button>
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-accent-primary">2.</span>
            <span>
              Ensure Ollama is running
              <button
                onClick={() => openLink("https://ollama.ai/")}
                className="ml-1 text-accent-primary hover:underline inline-flex items-center gap-0.5"
              >
                Download <ExternalLink className="w-3 h-3" />
              </button>
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-accent-primary">3.</span>
            <span>Check if port 8085 is available</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-accent-primary">4.</span>
            <span>
              Install dependencies:{" "}
              <code className="px-1 py-0.5 bg-neural-surface rounded text-text-secondary">
                pip install -r requirements.txt
              </code>
            </span>
          </li>
        </ul>
      </div>
    </div>
  );
}
