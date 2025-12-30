'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { useConnectionHealth, ConnectionState } from '@/hooks/useConnectionHealth';

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

interface ConnectionStatusProps {
  onConnected?: () => void;
}

export function ConnectionStatus({ onConnected }: ConnectionStatusProps) {
  const { state, error, errorDetails, retryIn, retry, cancel } = useConnectionHealth();

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
                  <p className="text-text-secondary text-sm mb-6">
                    {errorDetails}
                  </p>
                )}

                {/* Action Buttons */}
                <div className="flex gap-3 justify-center">
                  <button
                    onClick={retry}
                    className="neural-button-primary px-6 py-2 rounded-lg btn-lift"
                  >
                    Retry
                  </button>
                  {state === 'ollama_unavailable' && (
                    <button
                      onClick={() => window.open('https://ollama.ai', '_blank')}
                      className="neural-button px-6 py-2 rounded-lg btn-lift"
                    >
                      How to Start
                    </button>
                  )}
                  {state === 'server_unavailable' && (
                    <button
                      onClick={() => {
                        // Could open logs or show debug info
                        console.log('View logs requested');
                      }}
                      className="neural-button px-6 py-2 rounded-lg btn-lift"
                    >
                      View Logs
                    </button>
                  )}
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
