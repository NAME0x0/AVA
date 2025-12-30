'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { useCoreStore } from '@/stores/core';

export type ConnectionState =
  | 'initializing'
  | 'checking'
  | 'connected'
  | 'ollama_unavailable'
  | 'server_unavailable'
  | 'retrying';

export interface ConnectionHealth {
  state: ConnectionState;
  error: string | null;
  errorDetails: string | null;
  retryIn: number | null;
  retryCount: number;
  retry: () => void;
  cancel: () => void;
}

// Exponential backoff delays: 1s, 2s, 4s, 8s, 16s, 30s (max)
const BACKOFF_DELAYS = [1000, 2000, 4000, 8000, 16000, 30000];

function getBackoffDelay(retryCount: number): number {
  return BACKOFF_DELAYS[Math.min(retryCount, BACKOFF_DELAYS.length - 1)];
}

export function useConnectionHealth(): ConnectionHealth {
  const [state, setState] = useState<ConnectionState>('initializing');
  const [error, setError] = useState<string | null>(null);
  const [errorDetails, setErrorDetails] = useState<string | null>(null);
  const [retryIn, setRetryIn] = useState<number | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  const backendUrl = useCoreStore((s) => s.backendUrl);
  const setConnected = useCoreStore((s) => s.setConnected);

  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const countdownRef = useRef<NodeJS.Timeout | null>(null);
  const isCancelledRef = useRef(false);

  const clearTimers = useCallback(() => {
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }
    if (countdownRef.current) {
      clearInterval(countdownRef.current);
      countdownRef.current = null;
    }
  }, []);

  const checkConnection = useCallback(async () => {
    if (isCancelledRef.current) return;

    setState('checking');
    setError(null);
    setErrorDetails(null);
    setRetryIn(null);

    try {
      // Check server health
      const healthResponse = await fetch(`${backendUrl}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });

      if (!healthResponse.ok) {
        throw new Error(`Server returned ${healthResponse.status}`);
      }

      const healthData = await healthResponse.json();

      // Check if Ollama is available (server might report this)
      if (healthData.ollama_status === 'unavailable' || healthData.ollama === false) {
        setState('ollama_unavailable');
        setError('Neural engine not detected');
        setErrorDetails("Ollama isn't running. AVA needs it to think and respond.");
        setConnected(false);
        scheduleRetry();
        return;
      }

      // All good!
      setState('connected');
      setError(null);
      setErrorDetails(null);
      setRetryCount(0);
      setConnected(true);
    } catch (err) {
      if (isCancelledRef.current) return;

      const errorMessage = err instanceof Error ? err.message : 'Unknown error';

      // Determine error type
      if (errorMessage.includes('fetch') || errorMessage.includes('network') || errorMessage.includes('timeout')) {
        setState('server_unavailable');
        setError('Backend server is not responding');
        setErrorDetails('The AVA server may have crashed or is still starting up.');
      } else if (errorMessage.includes('Ollama') || errorMessage.includes('ollama')) {
        setState('ollama_unavailable');
        setError('Neural engine not detected');
        setErrorDetails("Ollama isn't running. AVA needs it to think and respond.");
      } else {
        setState('server_unavailable');
        setError('Connection failed');
        setErrorDetails(errorMessage);
      }

      setConnected(false);
      scheduleRetry();
    }
  }, [backendUrl, setConnected]);

  const scheduleRetry = useCallback(() => {
    if (isCancelledRef.current) return;

    const delay = getBackoffDelay(retryCount);
    const delaySeconds = Math.ceil(delay / 1000);

    setState('retrying');
    setRetryIn(delaySeconds);
    setRetryCount((c) => c + 1);

    // Countdown timer
    let remaining = delaySeconds;
    countdownRef.current = setInterval(() => {
      remaining -= 1;
      setRetryIn(remaining);
      if (remaining <= 0) {
        clearInterval(countdownRef.current!);
        countdownRef.current = null;
      }
    }, 1000);

    // Schedule actual retry
    retryTimeoutRef.current = setTimeout(() => {
      if (!isCancelledRef.current) {
        checkConnection();
      }
    }, delay);
  }, [retryCount, checkConnection]);

  const retry = useCallback(() => {
    clearTimers();
    isCancelledRef.current = false;
    setRetryCount(0);
    checkConnection();
  }, [clearTimers, checkConnection]);

  const cancel = useCallback(() => {
    isCancelledRef.current = true;
    clearTimers();
    setRetryIn(null);
    // Keep current error state but stop retrying
    if (state === 'retrying') {
      setState(error?.includes('Ollama') ? 'ollama_unavailable' : 'server_unavailable');
    }
  }, [clearTimers, state, error]);

  // Initial check on mount
  useEffect(() => {
    isCancelledRef.current = false;
    checkConnection();

    return () => {
      isCancelledRef.current = true;
      clearTimers();
    };
  }, [checkConnection, clearTimers]);

  // Re-check when backend URL changes
  useEffect(() => {
    retry();
  }, [backendUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    state,
    error,
    errorDetails,
    retryIn,
    retryCount,
    retry,
    cancel,
  };
}
