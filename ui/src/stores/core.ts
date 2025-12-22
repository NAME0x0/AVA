/**
 * AVA Core State Store
 * 
 * Centralized state management using Zustand for the Cortex-Medulla UI.
 * Handles all system state, chat messages, and UI state.
 */

import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

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

  // Thinking State
  thinkingStage: "idle" | "perceiving" | "routing" | "searching" | "generating" | "verifying";
  setThinkingStage: (stage: "idle" | "perceiving" | "routing" | "searching" | "generating" | "verifying") => void;

  // Input
  inputValue: string;
  setInputValue: (value: string) => void;
  isGenerating: boolean;
  setIsGenerating: (generating: boolean) => void;

  // Actions
  sendMessage: (content: string) => Promise<void>;
  forceCortex: () => Promise<void>;
  forceSleep: () => Promise<void>;
}

const generateId = () => Math.random().toString(36).substring(2, 15);

export const useCoreStore = create<CoreStore>()(
  subscribeWithSelector((set, get) => ({
    // Connection
    connected: false,
    backendUrl: "http://localhost:8085",
    setConnected: (connected) => set({ connected }),
    setBackendUrl: (backendUrl) => set({ backendUrl }),

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

    // Thinking State
    thinkingStage: "idle",
    setThinkingStage: (thinkingStage) => set({ thinkingStage }),

    // Input
    inputValue: "",
    setInputValue: (inputValue) => set({ inputValue }),
    isGenerating: false,
    setIsGenerating: (isGenerating) => set({ isGenerating }),

    // Actions
    sendMessage: async (content: string) => {
      const { backendUrl, addMessage, updateMessage, setIsGenerating, setCognitiveState, setSystemState } = get();

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
      }));

      try {
        // Check if we're in Tauri
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
            body: JSON.stringify({ message: content }),
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
      } catch (error) {
        updateMessage(assistantId, {
          content: `Error: ${error instanceof Error ? error.message : "Connection failed"}`,
          isStreaming: false,
        });
      } finally {
        setIsGenerating(false);
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
  }))
);
