"use client";

import { useEffect, useCallback } from "react";
import { TitleBar } from "@/components/layout/TitleBar";
import { Sidebar } from "@/components/layout/Sidebar";
import { ChatArea } from "@/components/chat/ChatArea";
import { SystemStatus } from "@/components/system/SystemStatus";
import { CommandPalette } from "@/components/CommandPalette";
import { useCoreStore } from "@/stores/core";
import { useSystemPolling } from "@/hooks/useSystemPolling";

export default function Home() {
  const {
    sidebarOpen,
    commandPaletteOpen,
    setCommandPaletteOpen,
    toggleSidebar,
    clearMessages,
    forceCortex,
  } = useCoreStore();

  // Start polling for system state
  useSystemPolling();

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
      }
    }
  }, [setCommandPaletteOpen, toggleSidebar, clearMessages, forceCortex]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return (
    <main className="flex flex-col h-screen bg-neural-void overflow-hidden rounded-xl border border-neural-hover/50">
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
    </main>
  );
}
