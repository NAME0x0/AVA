'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Wifi,
  WifiOff,
  Server,
  Cpu,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Settings,
  ChevronDown,
  ChevronUp,
  ExternalLink,
} from 'lucide-react';
import { useConnectionHealth, ConnectionState } from '@/hooks/useConnectionHealth';
import { useCoreStore } from '@/stores/core';
import { cn } from '@/lib/utils';

interface DiagnosticInfo {
  backend_reachable: boolean;
  backend_latency_ms: number | null;
  ollama_status: string;
  websocket_connected: boolean;
  server_version: string | null;
  server_uptime: number | null;
  error_details: string | null;
}

interface ASCIIStatusProps {
  status: 'online' | 'offline' | 'thinking';
}

function ASCIIStatus({ status }: ASCIIStatusProps) {
  const symbols = {
    online: '◉',
    offline: '○',
    thinking: '◐',
  };

  const labels = {
    online: 'ONLINE',
    offline: 'OFFLINE',
    thinking: 'CHECKING',
  };

  const colors = {
    online: 'text-status-success',
    offline: 'text-status-error',
    thinking: 'text-accent-primary',
  };

  return (
    <div className="font-mono text-center select-none">
      <div className="text-text-muted">╭──────────╮</div>
      <div className="text-text-muted">
        │{'  '}
        <span className={colors[status]}>{symbols[status]}</span>
        {' AVA  '}│
      </div>
      <div className="text-text-muted">
        │{'  '}
        <span className={colors[status]}>{labels[status].padEnd(6)}</span>
        {'  '}│
      </div>
      <div className="text-text-muted">╰──────────╯</div>
    </div>
  );
}

function DiagnosticRow({
  icon,
  label,
  status,
  detail,
}: {
  icon: React.ReactNode;
  label: string;
  status: 'ok' | 'warning' | 'error' | 'unknown';
  detail?: string;
}) {
  const statusConfig = {
    ok: { icon: CheckCircle, class: 'text-state-flow' },
    warning: { icon: AlertTriangle, class: 'text-state-curiosity' },
    error: { icon: XCircle, class: 'text-state-confusion' },
    unknown: { icon: AlertTriangle, class: 'text-text-muted' },
  };

  const config = statusConfig[status];
  const StatusIcon = config.icon;

  return (
    <div className="flex items-center justify-between py-2">
      <div className="flex items-center gap-2 text-sm text-text-secondary">
        {icon}
        {label}
      </div>
      <div className="flex items-center gap-2">
        {detail && <span className="text-xs text-text-muted">{detail}</span>}
        <StatusIcon className={cn('w-4 h-4', config.class)} />
      </div>
    </div>
  );
}

interface ConnectionStatusProps {
  onConnected?: () => void;
}

export function ConnectionStatus({ onConnected }: ConnectionStatusProps) {
  const { state, error, errorDetails, retryIn, retry, cancel } = useConnectionHealth();
  const { backendUrl, wsConnected, setSettingsPanelOpen } = useCoreStore();
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [diagnostics, setDiagnostics] = useState<DiagnosticInfo>({
    backend_reachable: false,
    backend_latency_ms: null,
    ollama_status: 'unknown',
    websocket_connected: false,
    server_version: null,
    server_uptime: null,
    error_details: null,
  });

  // Fetch diagnostics
  const fetchDiagnostics = useCallback(async () => {
    const startTime = performance.now();

    try {
      const response = await fetch(`${backendUrl}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });

      if (response.ok) {
        const data = await response.json();
        const latency = Math.round(performance.now() - startTime);

        setDiagnostics({
          backend_reachable: true,
          backend_latency_ms: latency,
          ollama_status: data.ollama_status || 'unknown',
          websocket_connected: wsConnected,
          server_version: data.version || null,
          server_uptime: data.uptime_seconds || null,
          error_details: data.ollama_error || null,
        });
      } else {
        setDiagnostics((prev) => ({
          ...prev,
          backend_reachable: false,
          error_details: `Server returned ${response.status}`,
        }));
      }
    } catch (err) {
      setDiagnostics((prev) => ({
        ...prev,
        backend_reachable: false,
        error_details: err instanceof Error ? err.message : 'Connection failed',
      }));
    }
  }, [backendUrl, wsConnected]);

  // Fetch diagnostics when showing or on state change
  useEffect(() => {
    if (showDiagnostics || state !== 'connected') {
      fetchDiagnostics();
    }
  }, [showDiagnostics, state, fetchDiagnostics]);

  // Update websocket status
  useEffect(() => {
    setDiagnostics((prev) => ({
      ...prev,
      websocket_connected: wsConnected,
    }));
  }, [wsConnected]);

  // Notify parent when connected
  if (state === 'connected' && onConnected) {
    onConnected();
  }

  const getASCIIStatus = (state: ConnectionState): 'online' | 'offline' | 'thinking' => {
    switch (state) {
      case 'connected':
        return 'online';
      case 'checking':
      case 'initializing':
      case 'retrying':
        return 'thinking';
      default:
        return 'offline';
    }
  };

  const isError = state === 'ollama_unavailable' || state === 'server_unavailable';
  const isChecking = state === 'checking' || state === 'initializing';
  const isRetrying = state === 'retrying';

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const openSettings = () => {
    setSettingsPanelOpen(true);
  };

  if (state === 'connected') {
    return null; // Don't show anything when connected
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-neural-void/95 backdrop-blur-sm">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="max-w-md w-full mx-4"
      >
        <div className="neural-card p-8 text-center">
          {/* ASCII Status */}
          <motion.div
            key={state}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mb-6"
          >
            <ASCIIStatus status={getASCIIStatus(state)} />
          </motion.div>

          {/* Status Message */}
          <AnimatePresence mode="wait">
            {isChecking && (
              <motion.div
                key="checking"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                <p className="text-text-secondary mb-2">Connecting to AVA...</p>
                <p className="text-xs text-text-muted mb-4">{backendUrl}</p>
                <div className="flex justify-center gap-1">
                  {[0, 1, 2].map((i) => (
                    <motion.div
                      key={i}
                      className="w-2 h-2 bg-accent-primary rounded-full"
                      animate={{
                        opacity: [0.3, 1, 0.3],
                      }}
                      transition={{
                        duration: 1,
                        repeat: Infinity,
                        delay: i * 0.2,
                      }}
                    />
                  ))}
                </div>
              </motion.div>
            )}

            {isError && (
              <motion.div
                key="error"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                <h3 className="text-lg font-medium text-text-primary mb-2">
                  {error}
                </h3>
                {errorDetails && (
                  <p className="text-text-secondary text-sm mb-4">
                    {errorDetails}
                  </p>
                )}

                {/* Quick Fix */}
                <div className="p-3 mb-4 bg-cortex-active/10 rounded-lg border border-cortex-active/30 text-left">
                  <p className="text-sm text-cortex-active font-medium mb-1">Quick fix:</p>
                  <p className="text-xs text-text-muted">
                    Run{' '}
                    <code className="px-1 py-0.5 bg-neural-surface rounded text-text-secondary">
                      python ava_server.py
                    </code>{' '}
                    in the AVA directory
                  </p>
                </div>

                {/* Diagnostics Toggle */}
                <button
                  onClick={() => setShowDiagnostics(!showDiagnostics)}
                  className="w-full flex items-center justify-between py-2 px-3 mb-4 rounded-lg bg-neural-surface text-sm text-text-secondary hover:bg-neural-hover transition-colors"
                >
                  <span>Diagnostics</span>
                  {showDiagnostics ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </button>

                {/* Diagnostics Panel */}
                <AnimatePresence>
                  {showDiagnostics && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="p-3 mb-4 rounded-lg bg-neural-surface border border-neural-hover text-left">
                        <DiagnosticRow
                          icon={<Server className="w-4 h-4" />}
                          label="Backend"
                          status={diagnostics.backend_reachable ? 'ok' : 'error'}
                          detail={
                            diagnostics.backend_latency_ms
                              ? `${diagnostics.backend_latency_ms}ms`
                              : undefined
                          }
                        />
                        <DiagnosticRow
                          icon={<Cpu className="w-4 h-4" />}
                          label="Ollama"
                          status={
                            diagnostics.ollama_status === 'connected'
                              ? 'ok'
                              : diagnostics.ollama_status === 'unknown'
                              ? 'unknown'
                              : 'error'
                          }
                          detail={diagnostics.ollama_status}
                        />
                        <DiagnosticRow
                          icon={<Wifi className="w-4 h-4" />}
                          label="WebSocket"
                          status={diagnostics.websocket_connected ? 'ok' : 'warning'}
                          detail={diagnostics.websocket_connected ? 'connected' : 'not connected'}
                        />
                        {diagnostics.server_version && (
                          <div className="flex items-center justify-between py-2 text-xs text-text-muted">
                            <span>Version</span>
                            <span>v{diagnostics.server_version}</span>
                          </div>
                        )}
                        {diagnostics.server_uptime && (
                          <div className="flex items-center justify-between py-2 text-xs text-text-muted">
                            <span>Uptime</span>
                            <span>{formatUptime(diagnostics.server_uptime)}</span>
                          </div>
                        )}
                        {diagnostics.error_details && (
                          <div className="mt-2 p-2 rounded bg-state-confusion/10 text-xs text-state-confusion">
                            {diagnostics.error_details}
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Action Buttons */}
                <div className="flex gap-3 justify-center">
                  <button
                    onClick={retry}
                    className="neural-button-primary px-6 py-2 rounded-lg btn-lift flex items-center gap-2"
                  >
                    <RefreshCw className="w-4 h-4" />
                    Retry
                  </button>
                  <button
                    onClick={openSettings}
                    className="neural-button px-6 py-2 rounded-lg btn-lift flex items-center gap-2"
                  >
                    <Settings className="w-4 h-4" />
                    Settings
                  </button>
                </div>

                {/* Help Links */}
                <div className="mt-4 flex gap-4 justify-center text-xs">
                  {state === 'ollama_unavailable' && (
                    <button
                      onClick={() => window.open('https://ollama.ai', '_blank')}
                      className="text-cortex-active hover:underline inline-flex items-center gap-1"
                    >
                      Get Ollama <ExternalLink className="w-3 h-3" />
                    </button>
                  )}
                  <button
                    onClick={() => window.open('https://www.python.org/downloads/', '_blank')}
                    className="text-text-muted hover:text-text-secondary inline-flex items-center gap-1"
                  >
                    Get Python <ExternalLink className="w-3 h-3" />
                  </button>
                </div>
              </motion.div>
            )}

            {isRetrying && (
              <motion.div
                key="retrying"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                <h3 className="text-lg font-medium text-text-primary mb-2">
                  {error}
                </h3>
                {errorDetails && (
                  <p className="text-text-secondary text-sm mb-4">
                    {errorDetails}
                  </p>
                )}

                {/* Retry countdown */}
                <div className="flex items-center justify-center gap-4 mb-4">
                  <div className="flex items-center gap-2 text-text-muted">
                    <motion.div
                      className="w-2 h-2 bg-accent-primary rounded-full"
                      animate={{
                        opacity: [0.3, 1, 0.3],
                      }}
                      transition={{
                        duration: 1,
                        repeat: Infinity,
                      }}
                    />
                    <span className="text-sm">
                      Retrying in {retryIn}s...
                    </span>
                  </div>
                  <button
                    onClick={cancel}
                    className="text-sm text-text-muted hover:text-text-secondary transition-colors"
                  >
                    Cancel
                  </button>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3 justify-center">
                  <button
                    onClick={retry}
                    className="neural-button-primary px-6 py-2 rounded-lg btn-lift"
                  >
                    Retry Now
                  </button>
                  <button
                    onClick={openSettings}
                    className="neural-button px-6 py-2 rounded-lg btn-lift"
                  >
                    Settings
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* ASCII Border */}
          <div className="mt-6 font-mono text-xs text-text-muted/50">
            ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default ConnectionStatus;
