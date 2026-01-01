/**
 * System Polling Hook
 * 
 * Polls the backend for system state, cognitive state, memory stats, and belief state.
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
    setBeliefState,
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
              shouldUseTools: (cognitive as any).should_use_tools || (cognitive as any).shouldUseTools,
              shouldThink: (cognitive as any).should_think || (cognitive as any).shouldThink,
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

        // Poll belief state
        try {
          const belief = await invoke("get_belief_state");
          if (belief) {
            setBeliefState({
              currentState: (belief as any).currentState || (belief as any).current_state,
              stateDistribution: (belief as any).stateDistribution || (belief as any).state_distribution || {},
              policyDistribution: (belief as any).policyDistribution || (belief as any).policy_distribution || {},
              freeEnergy: (belief as any).freeEnergy || (belief as any).free_energy || 0,
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
                  shouldUseTools: cognitive.should_use_tools || cognitive.shouldUseTools,
                  shouldThink: cognitive.should_think || cognitive.shouldThink,
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

            // Poll belief state
            try {
              const beliefRes = await fetch(`${backendUrl}/belief`);
              if (beliefRes.ok) {
                const belief = await beliefRes.json();
                setBeliefState({
                  currentState: belief.currentState || belief.current_state,
                  stateDistribution: belief.stateDistribution || belief.state_distribution || {},
                  policyDistribution: belief.policyDistribution || belief.policy_distribution || {},
                  freeEnergy: belief.freeEnergy || belief.free_energy || 0,
                });
              }
            } catch {}

            // Poll system state / stats
            try {
              const sysRes = await fetch(`${backendUrl}/stats`);
              if (sysRes.ok) {
                const stats = await sysRes.json();
                setSystemState({
                  systemState: "running",
                  totalInteractions: stats.total_requests || 0,
                  cortexInvocations: stats.cortex_requests || 0,
                  uptimeSeconds: stats.uptime_seconds || 0,
                  avgResponseTimeMs: stats.avg_response_time_ms || 0,
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
  }, [backendUrl, setConnected, setCognitiveState, setMemoryStats, setSystemState, setBeliefState]);

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
