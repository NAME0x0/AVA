"use client";

import { motion } from "framer-motion";
import { Minus, Square, X, PanelLeftClose, PanelLeft } from "lucide-react";
import { useCoreStore } from "@/stores/core";

export function TitleBar() {
  const { sidebarOpen, toggleSidebar, connected, systemState } = useCoreStore();

  const handleMinimize = async () => {
    if (typeof window !== "undefined" && "__TAURI__" in window) {
      const { appWindow } = await import("@tauri-apps/api/window");
      await appWindow.minimize();
    }
  };

  const handleMaximize = async () => {
    if (typeof window !== "undefined" && "__TAURI__" in window) {
      const { appWindow } = await import("@tauri-apps/api/window");
      const isMaximized = await appWindow.isMaximized();
      if (isMaximized) {
        await appWindow.unmaximize();
      } else {
        await appWindow.maximize();
      }
    }
  };

  const handleClose = async () => {
    if (typeof window !== "undefined" && "__TAURI__" in window) {
      const { appWindow } = await import("@tauri-apps/api/window");
      await appWindow.close();
    }
  };

  const handleDrag = async () => {
    if (typeof window !== "undefined" && "__TAURI__" in window) {
      const { appWindow } = await import("@tauri-apps/api/window");
      await appWindow.startDragging();
    }
  };

  return (
    <header className="h-12 bg-neural-surface/80 backdrop-blur-md border-b border-neural-hover flex items-center justify-between px-4 shrink-0">
      {/* Left Section */}
      <div className="flex items-center gap-3">
        {/* Sidebar Toggle */}
        <button
          onClick={toggleSidebar}
          className="p-1.5 rounded-lg hover:bg-neural-hover transition-colors text-text-muted hover:text-text-primary"
        >
          {sidebarOpen ? (
            <PanelLeftClose className="w-4 h-4" />
          ) : (
            <PanelLeft className="w-4 h-4" />
          )}
        </button>

        {/* Logo & Title */}
        <div
          className="flex items-center gap-3 cursor-move select-none"
          onMouseDown={handleDrag}
        >
          {/* Animated Logo */}
          <motion.div
            className="relative w-8 h-8"
            animate={{
              scale: systemState.activeComponent !== "idle" ? [1, 1.05, 1] : 1,
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          >
            {/* Outer glow */}
            <motion.div
              className="absolute inset-0 rounded-full bg-accent-primary/30 blur-md"
              animate={{
                opacity: [0.3, 0.6, 0.3],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
            
            {/* Main circle */}
            <div className="absolute inset-0 rounded-full bg-accent-primary flex items-center justify-center">
              <span className="text-neural-void font-bold text-sm">A</span>
            </div>

            {/* Active indicator ring */}
            {systemState.activeComponent !== "idle" && (
              <motion.div
                className={`absolute inset-[-2px] rounded-full border-2 ${
                  systemState.activeComponent === "cortex"
                    ? "border-cortex-active"
                    : systemState.activeComponent === "medulla"
                    ? "border-medulla-active"
                    : "border-accent-primary"
                }`}
                animate={{
                  opacity: [0.5, 1, 0.5],
                }}
                transition={{
                  duration: 1,
                  repeat: Infinity,
                }}
              />
            )}
          </motion.div>

          {/* Title */}
          <div className="flex flex-col">
            <span className="text-text-primary font-semibold text-sm leading-tight">
              AVA
            </span>
            <span className="text-text-muted text-[10px] leading-tight">
              Cortex-Medulla
            </span>
          </div>
        </div>
      </div>

      {/* Center - Draggable Area */}
      <div
        className="flex-1 h-full cursor-move"
        onMouseDown={handleDrag}
      />

      {/* Right Section */}
      <div className="flex items-center gap-2">
        {/* Connection Status */}
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-neural-elevated">
          <motion.div
            className={`w-2 h-2 rounded-full ${
              connected ? "bg-status-success" : "bg-status-error"
            }`}
            animate={{
              opacity: connected ? 1 : [1, 0.5, 1],
            }}
            transition={{
              duration: 1,
              repeat: connected ? 0 : Infinity,
            }}
          />
          <span className="text-xs text-text-secondary">
            {connected ? "Connected" : "Disconnected"}
          </span>
        </div>

        {/* Window Controls */}
        <div className="flex items-center ml-2">
          <button
            onClick={handleMinimize}
            className="p-2 hover:bg-neural-hover rounded-lg transition-colors text-text-muted hover:text-text-primary"
          >
            <Minus className="w-4 h-4" />
          </button>
          <button
            onClick={handleMaximize}
            className="p-2 hover:bg-neural-hover rounded-lg transition-colors text-text-muted hover:text-text-primary"
          >
            <Square className="w-3 h-3" />
          </button>
          <button
            onClick={handleClose}
            className="p-2 hover:bg-status-error/20 rounded-lg transition-colors text-text-muted hover:text-status-error"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
}
