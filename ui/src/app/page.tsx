"use client";

import { useEffect } from "react";
import { TitleBar } from "@/components/layout/TitleBar";
import { Sidebar } from "@/components/layout/Sidebar";
import { ChatArea } from "@/components/chat/ChatArea";
import { SystemStatus } from "@/components/system/SystemStatus";
import { useCoreStore } from "@/stores/core";
import { useSystemPolling } from "@/hooks/useSystemPolling";

export default function Home() {
  const { sidebarOpen } = useCoreStore();
  
  // Start polling for system state
  useSystemPolling();

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
    </main>
  );
}
