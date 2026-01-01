"use client";

import { useEffect, useCallback, useState } from "react";
import { AnimatePresence } from "framer-motion";
import { TitleBar } from "@/components/layout/TitleBar";
import { Sidebar } from "@/components/layout/Sidebar";
import { ChatArea } from "@/components/chat/ChatArea";
import { SystemStatus } from "@/components/system/SystemStatus";
import { CommandPalette } from "@/components/CommandPalette";
import { SettingsPanel } from "@/components/settings/SettingsPanel";
import { ToolsPanel } from "@/components/tools/ToolsPanel";
import { BugReportDialog } from "@/components/feedback/BugReportDialog";
import { ServerStartup } from "@/components/startup/ServerStartup";
import { WizardOverlay } from "@/components/wizard";
import { useCoreStore } from "@/stores/core";
import { useSystemPolling } from "@/hooks/useSystemPolling";

// Check if we're in Tauri environment
const isTauri = typeof window !== "undefined" && "__TAURI__" in window;

export default function Home() {
  const {
    sidebarOpen,
    commandPaletteOpen,
    setCommandPaletteOpen,
    toggleSidebar,
    clearMessages,
    forceCortex,
    toggleSettingsPanel,
    setSettingsPanelOpen,
    connectWebSocket,
    setupState,
  } = useCoreStore();

  // Server startup state (for Tauri)
  const [serverReady, setServerReady] = useState(!isTauri);

  // Wizard state
  const [showWizard, setShowWizard] = useState(false);

  // Bug report dialog state
  const [bugReportOpen, setBugReportOpen] = useState(false);
  const [bugReportError, setBugReportError] = useState("");

  // Show wizard on first run
  useEffect(() => {
    console.log('[AVA] Setup check:', { hasCompletedSetup: setupState.hasCompletedSetup, serverReady, showWizard });
    if (!setupState.hasCompletedSetup && serverReady) {
      console.log('[AVA] Showing wizard overlay');
      setShowWizard(true);
    }
  }, [setupState.hasCompletedSetup, serverReady]);

  // Start polling for system state
  useSystemPolling();

  // Connect WebSocket on mount
  useEffect(() => {
    connectWebSocket();
  }, [connectWebSocket]);

  // Listen for Tauri tray menu events
  useEffect(() => {
    if (!isTauri) return;

    let unlisten: (() => void) | undefined;

    const setupTauriListeners = async () => {
      try {
        const { listen } = await import("@tauri-apps/api/event");

        // Listen for settings event from tray
        const unlistenSettings = await listen("open-settings", () => {
          setSettingsPanelOpen(true);
        });

        // Listen for bug report event from tray
        const unlistenBugReport = await listen("open-bug-report", () => {
          setBugReportOpen(true);
        });

        unlisten = () => {
          unlistenSettings();
          unlistenBugReport();
        };
      } catch (err) {
        console.error("Failed to setup Tauri listeners:", err);
      }
    };

    setupTauriListeners();

    return () => {
      if (unlisten) unlisten();
    };
  }, [setSettingsPanelOpen]);

  // Global error handler for bug reporting
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      setBugReportError(event.message);
      setBugReportOpen(true);
    };

    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      const message = event.reason?.message || String(event.reason);
      setBugReportError(message);
      setBugReportOpen(true);
    };

    window.addEventListener("error", handleError);
    window.addEventListener("unhandledrejection", handleUnhandledRejection);

    return () => {
      window.removeEventListener("error", handleError);
      window.removeEventListener("unhandledrejection", handleUnhandledRejection);
    };
  }, []);

  // Global keyboard shortcuts
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Check for modifier key
    const isCtrlOrCmd = e.ctrlKey || e.metaKey;

    if (isCtrlOrCmd) {
      switch (e.key.toLowerCase()) {
        case 'k':
          e.preventDefault();
          setCommandPaletteOpen(true);
          break;
        case 'b':
          e.preventDefault();
          toggleSidebar();
          break;
        case 'l':
          e.preventDefault();
          clearMessages();
          break;
        case 'd':
          e.preventDefault();
          forceCortex();
          break;
        case ',':
          e.preventDefault();
          toggleSettingsPanel();
          break;
      }
    }
  }, [setCommandPaletteOpen, toggleSidebar, clearMessages, forceCortex, toggleSettingsPanel]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Debug logging
  useEffect(() => {
    console.log('[AVA] serverReady:', serverReady, 'isTauri:', isTauri);
  }, [serverReady]);

  return (
    <>
      {/* Server Startup Screen (Tauri only) */}
      {!serverReady && isTauri && (
        <ServerStartup onReady={() => {
          console.log('[AVA] Server ready, transitioning to main app');
          setServerReady(true);
        }} />
      )}

      <main className={`flex flex-col h-screen bg-neural-void overflow-hidden rounded-xl border border-neural-hover/50 ${!serverReady && isTauri ? 'invisible' : ''}`}>
        {/* Custom Title Bar for Tauri */}
        <TitleBar />

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar - System Metrics */}
        <Sidebar isOpen={sidebarOpen} />

        {/* Chat Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <ChatArea />
        </div>
      </div>

      {/* Bottom Status Bar */}
      <SystemStatus />

      {/* Command Palette Overlay */}
      <CommandPalette
        isOpen={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
      />

      {/* Settings Panel */}
      <SettingsPanel />

      {/* Tools Panel */}
      <ToolsPanel />

      {/* Bug Report Dialog */}
      <BugReportDialog
        isOpen={bugReportOpen}
        onClose={() => {
          setBugReportOpen(false);
          setBugReportError("");
        }}
        errorMessage={bugReportError}
      />
      </main>

      {/* First-Run Wizard */}
      <AnimatePresence>
        {showWizard && (
          <WizardOverlay onComplete={() => setShowWizard(false)} />
        )}
      </AnimatePresence>
    </>
  );
}
