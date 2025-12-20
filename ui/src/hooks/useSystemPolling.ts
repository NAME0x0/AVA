/**
 * System Polling Hook
 * 
 * Polls the backend for system state, cognitive state, and memory stats.
 */

import { useEffect, useCallback, useRef } from "react";
import { useCoreStore } from "@/stores/core";

const POLL_INTERVAL = 2000; // 2 seconds

export function useSystemPolling() {
  const {
    backendUrl,
    connected,
    setConnected,
    setCognitiveState,
    setMemoryStats,
    setSystemState,
  } = useCoreStore();

  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const pollBackend = useCallback(async () => {
    const isTauri = typeof window !== "undefined" && "__TAURI__" in window;

    try {
      if (isTauri) {
        const { invoke } = await import("@tauri-apps/api/tauri");
        
        // Poll cognitive state
        try {
          const cognitive = await invoke("get_cognitive_state");
          if (cognitive) {
            setCognitiveState({
              label: (cognitive as any).label,
              entropy: (cognitive as any).entropy,
              varentropy: (cognitive as any).varentropy,
              confidence: (cognitive as any).confidence,
              surprise: (cognitive as any).surprise,
              shouldUseTools: (cognitive as any).should_use_tools,
              shouldThink: (cognitive as any).should_think,
            });
          }
        } catch {}

        // Poll memory stats
        try {
          const memory = await invoke("get_memory_stats");
          if (memory) {
            setMemoryStats({
              totalMemories: (memory as any).total_memories,
              memoryUpdates: (memory as any).memory_updates,
              avgSurprise: (memory as any).avg_surprise,
              backend: (memory as any).backend,
              memoryUtilization: (memory as any).memory_utilization || 0,
            });
          }
        } catch {}

        // Poll system state
        try {
          const system = await invoke("get_system_state");
          if (system) {
            setSystemState({
              systemState: (system as any).system_state,
              activeComponent: (system as any).active_component,
              uptimeSeconds: (system as any).uptime_seconds,
              totalInteractions: (system as any).total_interactions,
              cortexInvocations: (system as any).cortex_invocations,
              avgResponseTimeMs: (system as any).avg_response_time_ms,
            });
          }
          setConnected(true);
        } catch {
          setConnected(false);
        }
      } else {
        // Browser mode - use fetch
        try {
          // Health check
          const healthRes = await fetch(`${backendUrl}/health`, {
            signal: AbortSignal.timeout(3000),
          });
          setConnected(healthRes.ok);

          if (healthRes.ok) {
            // Poll cognitive state
            try {
              const cogRes = await fetch(`${backendUrl}/cognitive`);
              if (cogRes.ok) {
                const cognitive = await cogRes.json();
                setCognitiveState({
                  label: cognitive.label,
                  entropy: cognitive.entropy,
                  varentropy: cognitive.varentropy,
                  confidence: cognitive.confidence,
                  surprise: cognitive.surprise,
                  shouldUseTools: cognitive.should_use_tools,
                  shouldThink: cognitive.should_think,
                });
              }
            } catch {}

            // Poll memory stats
            try {
              const memRes = await fetch(`${backendUrl}/memory`);
              if (memRes.ok) {
                const memory = await memRes.json();
                setMemoryStats({
                  totalMemories: memory.total_memories,
                  memoryUpdates: memory.memory_updates,
                  avgSurprise: memory.avg_surprise,
                  backend: memory.backend,
                  memoryUtilization: memory.memory_utilization || 0,
                });
              }
            } catch {}

            // Poll system state
            try {
              const sysRes = await fetch(`${backendUrl}/stats`);
              if (sysRes.ok) {
                const stats = await sysRes.json();
                setSystemState({
                  systemState: stats.system?.is_running ? "running" : "paused",
                  totalInteractions: stats.system?.interaction_count || 0,
                });
              }
            } catch {}
          }
        } catch {
          setConnected(false);
        }
      }
    } catch (error) {
      setConnected(false);
    }
  }, [backendUrl, setConnected, setCognitiveState, setMemoryStats, setSystemState]);

  useEffect(() => {
    // Initial poll
    pollBackend();

    // Set up interval
    intervalRef.current = setInterval(pollBackend, POLL_INTERVAL);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [pollBackend]);

  // Listen for Tauri events if available
  useEffect(() => {
    const setupTauriListener = async () => {
      if (typeof window !== "undefined" && "__TAURI__" in window) {
        const { listen } = await import("@tauri-apps/api/event");
        
        const unlisten = await listen("backend-status", (event: any) => {
          setConnected(event.payload.connected);
        });

        return unlisten;
      }
    };

    const cleanup = setupTauriListener();
    return () => {
      cleanup.then((unlisten) => unlisten?.());
    };
  }, [setConnected]);

  return { connected, pollBackend };
}
