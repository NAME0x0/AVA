/**
 * AVA Core State Store
 *
 * Centralized state management using Zustand for the Cortex-Medulla UI.
 * Handles all system state, chat messages, and UI state.
 *
 * Features:
 * - State persistence for UI preferences (sidebar, split pane)
 * - Environment variable support for backend URL
 * - WebSocket streaming with HTTP fallback
 */

import { create } from "zustand";
import { persist, subscribeWithSelector } from "zustand/middleware";

// Backend URL from environment or default
const DEFAULT_BACKEND_URL =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_BACKEND_URL) ||
  "http://localhost:8085";

// Types
export interface CognitiveState {
  label: string;
  entropy: number;
  varentropy: number;
  confidence: number;
  surprise: number;
  shouldUseTools: boolean;
  shouldThink: boolean;
}

export interface MemoryStats {
  totalMemories: number;
  memoryUpdates: number;
  avgSurprise: number;
  backend: string;
  memoryUtilization: number;
}

export interface SystemState {
  connected: boolean;
  systemState: "initializing" | "running" | "paused" | "sleeping" | "error";
  activeComponent: "medulla" | "cortex" | "bridge" | "idle";
  uptimeSeconds: number;
  totalInteractions: number;
  cortexInvocations: number;
  avgResponseTimeMs: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  cognitiveState?: CognitiveState;
  surprise?: number;
  usedCortex?: boolean;
  policySelected?: string;
  responseTimeMs?: number;
  isStreaming?: boolean;
  toolsUsed?: string[];
}

export interface BeliefState {
  currentState: string;
  stateDistribution: Record<string, number>;
  policyDistribution: Record<string, number>;
  freeEnergy: number;
}

interface CoreStore {
  // Connection
  connected: boolean;
  backendUrl: string;
  setConnected: (connected: boolean) => void;
  setBackendUrl: (url: string) => void;

  // Server Launch State (Tauri)
  serverLaunchState: 'not_started' | 'starting' | 'ready' | 'failed';
  serverError: string | null;
  setServerLaunchState: (state: CoreStore['serverLaunchState'], error?: string) => void;

  // System State
  systemState: SystemState;
  setSystemState: (state: Partial<SystemState>) => void;

  // Cognitive State (from Medulla)
  cognitiveState: CognitiveState;
  setCognitiveState: (state: Partial<CognitiveState>) => void;

  // Memory Stats (from Titans)
  memoryStats: MemoryStats;
  setMemoryStats: (stats: Partial<MemoryStats>) => void;

  // Belief State (from Agency/Active Inference)
  beliefState: BeliefState;
  setBeliefState: (state: Partial<BeliefState>) => void;

  // Chat
  messages: Message[];
  addMessage: (message: Omit<Message, "id" | "timestamp">) => void;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  clearMessages: () => void;

  // UI State
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;

  // Command Palette
  commandPaletteOpen: boolean;
  setCommandPaletteOpen: (open: boolean) => void;
  toggleCommandPalette: () => void;

  // Split Pane
  splitPaneMode: "chat-only" | "chat-metrics" | "chat-code";
  splitPaneRatio: number;
  setSplitPaneMode: (mode: "chat-only" | "chat-metrics" | "chat-code") => void;
  setSplitPaneRatio: (ratio: number) => void;

  // Settings Panel
  settingsPanelOpen: boolean;
  setSettingsPanelOpen: (open: boolean) => void;
  toggleSettingsPanel: () => void;

  // Tools Panel
  toolsPanelOpen: boolean;
  setToolsPanelOpen: (open: boolean) => void;
  toggleToolsPanel: () => void;

  // Preferences
  preferences: {
    streamingMode: 'http' | 'websocket';
    animations: 'full' | 'reduced' | 'none';
    showToolsUsed: boolean;
    theme: {
      mode: 'light' | 'dark' | 'system';
      fontSize: 'small' | 'medium' | 'large';
      reduceMotion: boolean;
    };
  };
  updatePreference: <K extends keyof CoreStore['preferences']>(
    key: K,
    value: CoreStore['preferences'][K]
  ) => void;

  // Setup/Wizard State
  setupState: {
    hasCompletedSetup: boolean;
    userName: string | null;
    setupTimestamp: number | null;
  };
  setSetupComplete: (userName?: string) => void;
  resetSetup: () => void;

  // WebSocket State
  wsConnected: boolean;
  wsDisabled: boolean;
  wsRef: WebSocket | null;
  setWsConnected: (connected: boolean) => void;
  connectWebSocket: () => void;
  disconnectWebSocket: () => void;

  // Thinking State
  thinkingStage: "idle" | "perceiving" | "routing" | "searching" | "generating" | "verifying";
  setThinkingStage: (stage: "idle" | "perceiving" | "routing" | "searching" | "generating" | "verifying") => void;

  // Input
  inputValue: string;
  setInputValue: (value: string) => void;
  isGenerating: boolean;
  setIsGenerating: (generating: boolean) => void;

  // Actions
  sendMessage: (content: string, forceSearch?: boolean, forceCortex?: boolean) => Promise<void>;
  forceCortex: () => Promise<void>;
  forceSleep: () => Promise<void>;
}

const generateId = () => Math.random().toString(36).substring(2, 15);

export const useCoreStore = create<CoreStore>()(
  persist(
    subscribeWithSelector((set, get) => ({
      // Connection
      connected: false,
      backendUrl: DEFAULT_BACKEND_URL,
      setConnected: (connected) => set({ connected }),
      setBackendUrl: (backendUrl) => set({ backendUrl }),

      // Server Launch State (Tauri)
      serverLaunchState: 'not_started',
      serverError: null,
      setServerLaunchState: (state, error) => set({
        serverLaunchState: state,
        serverError: error || null,
        // Also update connected state based on server state
        connected: state === 'ready',
      }),

    // System State
    systemState: {
      connected: false,
      systemState: "initializing",
      activeComponent: "idle",
      uptimeSeconds: 0,
      totalInteractions: 0,
      cortexInvocations: 0,
      avgResponseTimeMs: 0,
    },
    setSystemState: (state) =>
      set((s) => ({ systemState: { ...s.systemState, ...state } })),

    // Cognitive State
    cognitiveState: {
      label: "UNKNOWN",
      entropy: 0,
      varentropy: 0,
      confidence: 0,
      surprise: 0,
      shouldUseTools: false,
      shouldThink: false,
    },
    setCognitiveState: (state) =>
      set((s) => ({ cognitiveState: { ...s.cognitiveState, ...state } })),

    // Memory Stats
    memoryStats: {
      totalMemories: 0,
      memoryUpdates: 0,
      avgSurprise: 0,
      backend: "none",
      memoryUtilization: 0,
    },
    setMemoryStats: (stats) =>
      set((s) => ({ memoryStats: { ...s.memoryStats, ...stats } })),

    // Belief State
    beliefState: {
      currentState: "IDLE",
      stateDistribution: {},
      policyDistribution: {},
      freeEnergy: 0,
    },
    setBeliefState: (state) =>
      set((s) => ({ beliefState: { ...s.beliefState, ...state } })),

    // Chat
    messages: [
      {
        id: generateId(),
        role: "assistant",
        content:
          "Hello! I'm AVA, running on the Cortex-Medulla architecture. My Medulla provides instant reflexive responses while my Cortex handles deep reasoning. Ask me anything!",
        timestamp: new Date(),
        cognitiveState: {
          label: "FLOW",
          entropy: 1.2,
          varentropy: 0.8,
          confidence: 0.9,
          surprise: 0.3,
          shouldUseTools: false,
          shouldThink: false,
        },
        usedCortex: false,
        policySelected: "REFLEX_REPLY",
      },
    ],
    addMessage: (message) =>
      set((s) => ({
        messages: [
          ...s.messages,
          { ...message, id: generateId(), timestamp: new Date() },
        ],
      })),
    updateMessage: (id, updates) =>
      set((s) => ({
        messages: s.messages.map((m) =>
          m.id === id ? { ...m, ...updates } : m
        ),
      })),
    clearMessages: () => set({ messages: [] }),

    // UI State
    sidebarOpen: true,
    toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
    setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),

    // Command Palette
    commandPaletteOpen: false,
    setCommandPaletteOpen: (commandPaletteOpen) => set({ commandPaletteOpen }),
    toggleCommandPalette: () => set((s) => ({ commandPaletteOpen: !s.commandPaletteOpen })),

    // Split Pane
    splitPaneMode: "chat-metrics",
    splitPaneRatio: 0.65,
    setSplitPaneMode: (splitPaneMode) => set({ splitPaneMode }),
    setSplitPaneRatio: (splitPaneRatio) => set({ splitPaneRatio }),

    // Settings Panel
    settingsPanelOpen: false,
    setSettingsPanelOpen: (settingsPanelOpen) => set({ settingsPanelOpen }),
    toggleSettingsPanel: () => set((s) => ({ settingsPanelOpen: !s.settingsPanelOpen })),

    // Tools Panel
    toolsPanelOpen: false,
    setToolsPanelOpen: (toolsPanelOpen) => set({ toolsPanelOpen }),
    toggleToolsPanel: () => set((s) => ({ toolsPanelOpen: !s.toolsPanelOpen })),

    // Preferences
    preferences: {
      streamingMode: 'websocket',
      animations: 'full',
      showToolsUsed: true,
      theme: {
        mode: 'system',
        fontSize: 'medium',
        reduceMotion: false,
      },
    },
    updatePreference: (key, value) =>
      set((s) => ({
        preferences: { ...s.preferences, [key]: value },
      })),

    // Setup/Wizard State
    setupState: {
      hasCompletedSetup: false,
      userName: null,
      setupTimestamp: null,
    },
    setSetupComplete: (userName) =>
      set({
        setupState: {
          hasCompletedSetup: true,
          userName: userName || null,
          setupTimestamp: Date.now(),
        },
      }),
    resetSetup: () =>
      set({
        setupState: {
          hasCompletedSetup: false,
          userName: null,
          setupTimestamp: null,
        },
      }),

    // WebSocket State
    wsConnected: false,
    wsRef: null,
    wsDisabled: false, // Set to true if WS endpoint doesn't exist
    setWsConnected: (wsConnected) => set({ wsConnected }),
    connectWebSocket: () => {
      const state = get();
      // Don't retry if WS is disabled (server doesn't support it)
      if (state.wsDisabled) return;
      if (state.wsRef?.readyState === WebSocket.OPEN) return;

      if (typeof window === "undefined") return;

      try {
        const url = new URL(state.backendUrl);
        const protocol = url.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${url.host}/ws`;

        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          set({ wsConnected: true, wsRef: ws });
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            const state = get();

            if (data.type === "status") {
              // Update thinking stage
              state.setThinkingStage(data.stage || "generating");
            } else if (data.type === "chunk" || data.type === "token") {
              // Append streaming content
              const messages = state.messages;
              const lastMessage = messages[messages.length - 1];
              if (lastMessage?.isStreaming && lastMessage.role === "assistant") {
                set({
                  messages: [
                    ...messages.slice(0, -1),
                    {
                      ...lastMessage,
                      content: lastMessage.content + (data.text || data.token || data.content || ""),
                    },
                  ],
                });
              }
            } else if (data.type === "response" || data.type === "complete" || data.type === "end") {
              // Finalize message with metadata
              const messages = state.messages;
              const lastMessage = messages[messages.length - 1];
              if (lastMessage?.isStreaming) {
                set({
                  messages: [
                    ...messages.slice(0, -1),
                    {
                      ...lastMessage,
                      content: data.response || lastMessage.content,
                      isStreaming: false,
                      cognitiveState: data.cognitive_state ? {
                        label: data.cognitive_state.label || data.cognitive_state,
                        entropy: data.cognitive_state.entropy || 0,
                        varentropy: data.cognitive_state.varentropy || 0,
                        confidence: data.cognitive_state.confidence || 0,
                        surprise: data.cognitive_state.surprise || 0,
                        shouldUseTools: data.cognitive_state.should_use_tools || false,
                        shouldThink: data.cognitive_state.should_think || false,
                      } : undefined,
                      usedCortex: data.used_cortex,
                      policySelected: data.policy_selected,
                      responseTimeMs: data.response_time_ms,
                      toolsUsed: data.tools_used,
                    },
                  ],
                  isGenerating: false,
                  thinkingStage: "idle",
                });

                // Update cognitive state
                if (data.cognitive_state) {
                  state.setCognitiveState({
                    label: data.cognitive_state.label || data.cognitive_state,
                    entropy: data.cognitive_state.entropy || 0,
                    varentropy: data.cognitive_state.varentropy || 0,
                    confidence: data.cognitive_state.confidence || 0,
                    surprise: data.cognitive_state.surprise || 0,
                  });
                }
              }
            } else if (data.type === "error") {
              const messages = state.messages;
              const lastMessage = messages[messages.length - 1];
              if (lastMessage?.isStreaming) {
                set({
                  messages: [
                    ...messages.slice(0, -1),
                    {
                      ...lastMessage,
                      content: `Error: ${data.message || "Unknown error"}`,
                      isStreaming: false,
                    },
                  ],
                  isGenerating: false,
                  thinkingStage: "idle",
                });
              }
            }
          } catch (e) {
            console.error("Failed to parse WebSocket message:", e);
          }
        };

        ws.onclose = () => {
          set({ wsConnected: false, wsRef: null });
        };

        ws.onerror = (error) => {
          // Disable WS retry if server doesn't support it (404 or connection refused)
          console.log("[AVA] WebSocket error - disabling future reconnection attempts. Use HTTP polling instead.");
          set({ wsConnected: false, wsDisabled: true });
        };

        set({ wsRef: ws });
      } catch (e) {
        console.error("Failed to connect WebSocket:", e);
      }
    },
    disconnectWebSocket: () => {
      const { wsRef } = get();
      if (wsRef) {
        wsRef.close(1000, "User disconnected");
        set({ wsConnected: false, wsRef: null });
      }
    },

    // Thinking State
    thinkingStage: "idle",
    setThinkingStage: (thinkingStage) => set({ thinkingStage }),

    // Input
    inputValue: "",
    setInputValue: (inputValue) => set({ inputValue }),
    isGenerating: false,
    setIsGenerating: (isGenerating) => set({ isGenerating }),

    // Actions
    sendMessage: async (content: string, forceSearch: boolean = false, forceCortex: boolean = false) => {
      const {
        backendUrl,
        addMessage,
        updateMessage,
        setIsGenerating,
        setCognitiveState,
        setSystemState,
        setThinkingStage,
        wsRef,
        wsConnected,
        preferences,
      } = get();

      // Add user message
      addMessage({ role: "user", content });

      // Add placeholder for assistant
      const assistantId = generateId();
      set((s) => ({
        messages: [
          ...s.messages,
          {
            id: assistantId,
            role: "assistant",
            content: "",
            timestamp: new Date(),
            isStreaming: true,
          },
        ],
        inputValue: "",
        isGenerating: true,
        thinkingStage: "perceiving",
      }));

      try {
        // Try WebSocket first if connected and preferred
        if (preferences.streamingMode === 'websocket' && wsConnected && wsRef?.readyState === WebSocket.OPEN) {
          // Send via WebSocket - response handled by onmessage handler
          wsRef.send(JSON.stringify({
            message: content,
            force_search: forceSearch,
            force_cortex: forceCortex,
          }));
          // Don't set isGenerating to false here - WebSocket handler will do it
          return;
        }

        // HTTP fallback
        const isTauri = typeof window !== "undefined" && "__TAURI__" in window;

        let response;

        if (isTauri) {
          // Use Tauri command
          const { invoke } = await import("@tauri-apps/api/tauri");
          response = await invoke("send_message", { message: content });
        } else {
          // Use fetch for browser
          const res = await fetch(`${backendUrl}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              message: content,
              force_search: forceSearch,
              force_cortex: forceCortex,
            }),
          });
          response = await res.json();
        }

        // Update the placeholder message
        updateMessage(assistantId, {
          content: response.response || response.error || "No response",
          isStreaming: false,
          cognitiveState: response.cognitive_state
            ? {
                label: response.cognitive_state.label,
                entropy: response.cognitive_state.entropy,
                varentropy: response.cognitive_state.varentropy,
                confidence: response.cognitive_state.confidence,
                surprise: response.cognitive_state.surprise,
                shouldUseTools: response.cognitive_state.should_use_tools,
                shouldThink: response.cognitive_state.should_think,
              }
            : undefined,
          surprise: response.surprise,
          usedCortex: response.used_cortex,
          policySelected: response.policy_selected,
          responseTimeMs: response.response_time_ms,
          toolsUsed: response.tools_used,
        });

        // Update cognitive state if available
        if (response.cognitive_state) {
          setCognitiveState({
            label: response.cognitive_state.label,
            entropy: response.cognitive_state.entropy,
            varentropy: response.cognitive_state.varentropy,
            confidence: response.cognitive_state.confidence,
            surprise: response.cognitive_state.surprise,
            shouldUseTools: response.cognitive_state.should_use_tools,
            shouldThink: response.cognitive_state.should_think,
          });
        }

        // Update system stats
        setSystemState({
          totalInteractions: get().systemState.totalInteractions + 1,
          cortexInvocations: response.used_cortex
            ? get().systemState.cortexInvocations + 1
            : get().systemState.cortexInvocations,
        });

        setIsGenerating(false);
        setThinkingStage("idle");
      } catch (error) {
        updateMessage(assistantId, {
          content: `Error: ${error instanceof Error ? error.message : "Connection failed"}`,
          isStreaming: false,
        });
        setIsGenerating(false);
        setThinkingStage("idle");
      }
    },

    forceCortex: async () => {
      const { backendUrl } = get();
      try {
        const isTauri = typeof window !== "undefined" && "__TAURI__" in window;
        if (isTauri) {
          const { invoke } = await import("@tauri-apps/api/tauri");
          await invoke("force_cortex");
        } else {
          await fetch(`${backendUrl}/force_cortex`, { method: "POST" });
        }
      } catch (error) {
        console.error("Failed to force cortex:", error);
      }
    },

    forceSleep: async () => {
      const { backendUrl } = get();
      try {
        const isTauri = typeof window !== "undefined" && "__TAURI__" in window;
        if (isTauri) {
          const { invoke } = await import("@tauri-apps/api/tauri");
          await invoke("force_sleep");
        } else {
          await fetch(`${backendUrl}/sleep`, { method: "POST" });
        }
      } catch (error) {
        console.error("Failed to force sleep:", error);
      }
    },
  })),
    {
      name: "ava-ui-preferences",
      // Only persist UI preferences, not transient state
      partialize: (state) => ({
        sidebarOpen: state.sidebarOpen,
        splitPaneMode: state.splitPaneMode,
        splitPaneRatio: state.splitPaneRatio,
        backendUrl: state.backendUrl,
        preferences: state.preferences,
        setupState: state.setupState,
      }),
    }
  )
);
