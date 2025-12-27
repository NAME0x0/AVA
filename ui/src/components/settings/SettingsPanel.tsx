"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X, Wifi, WifiOff, Sparkles, Eye, Download, Trash2, Settings } from "lucide-react";
import { useCoreStore } from "@/stores/core";
import { cn } from "@/lib/utils";

interface SettingsSectionProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}

function SettingsSection({ title, icon, children }: SettingsSectionProps) {
  return (
    <div className="mb-6">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-accent-primary">{icon}</span>
        <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wide">
          {title}
        </h3>
      </div>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

interface ToggleProps {
  label: string;
  description?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

function Toggle({ label, description, checked, onChange }: ToggleProps) {
  return (
    <label className="flex items-center justify-between p-3 rounded-lg bg-neural-elevated/50 hover:bg-neural-elevated transition-colors cursor-pointer">
      <div>
        <div className="text-sm font-medium text-text-primary">{label}</div>
        {description && (
          <div className="text-xs text-text-muted">{description}</div>
        )}
      </div>
      <motion.button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative w-11 h-6 rounded-full transition-colors",
          checked ? "bg-accent-primary" : "bg-neural-hover"
        )}
        whileTap={{ scale: 0.95 }}
      >
        <motion.span
          className="absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow"
          animate={{ x: checked ? 20 : 0 }}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
        />
      </motion.button>
    </label>
  );
}

interface SelectOptionProps {
  label: string;
  description?: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}

function SelectOption({ label, description, value, options, onChange }: SelectOptionProps) {
  return (
    <div className="p-3 rounded-lg bg-neural-elevated/50">
      <div className="flex items-center justify-between mb-2">
        <div>
          <div className="text-sm font-medium text-text-primary">{label}</div>
          {description && (
            <div className="text-xs text-text-muted">{description}</div>
          )}
        </div>
      </div>
      <div className="flex gap-2">
        {options.map((opt) => (
          <motion.button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className={cn(
              "flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors",
              value === opt.value
                ? "bg-accent-primary text-white"
                : "bg-neural-surface text-text-muted hover:text-text-primary"
            )}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {opt.label}
          </motion.button>
        ))}
      </div>
    </div>
  );
}

export function SettingsPanel() {
  const {
    settingsPanelOpen,
    setSettingsPanelOpen,
    preferences,
    updatePreference,
    backendUrl,
    setBackendUrl,
    wsConnected,
    connectWebSocket,
    disconnectWebSocket,
    clearMessages,
    messages,
  } = useCoreStore();

  const handleExportChat = () => {
    const data = {
      exported: new Date().toISOString(),
      messages: messages.map((m) => ({
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
        cognitiveState: m.cognitiveState?.label,
        usedCortex: m.usedCortex,
      })),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ava-chat-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <AnimatePresence>
      {settingsPanelOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/30 backdrop-blur-sm z-40"
            onClick={() => setSettingsPanelOpen(false)}
          />

          {/* Panel */}
          <motion.aside
            initial={{ width: 0, opacity: 0, x: 100 }}
            animate={{ width: 320, opacity: 1, x: 0 }}
            exit={{ width: 0, opacity: 0, x: 100 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="fixed right-0 top-0 h-full z-50 bg-neural-surface/95 backdrop-blur-md border-l border-neural-hover overflow-hidden"
          >
            <div className="h-full flex flex-col">
              {/* Header */}
              <div className="flex items-center justify-between p-4 border-b border-neural-hover">
                <div className="flex items-center gap-2">
                  <Settings className="w-5 h-5 text-accent-primary" />
                  <h2 className="text-lg font-semibold text-text-primary">Settings</h2>
                </div>
                <motion.button
                  onClick={() => setSettingsPanelOpen(false)}
                  className="p-2 rounded-lg hover:bg-neural-hover transition-colors"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <X className="w-5 h-5 text-text-muted" />
                </motion.button>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto p-4">
                {/* Connection Section */}
                <SettingsSection
                  title="Connection"
                  icon={wsConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
                >
                  <div className="p-3 rounded-lg bg-neural-elevated/50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-text-primary">Backend URL</span>
                      <span
                        className={cn(
                          "px-2 py-0.5 rounded text-xs",
                          wsConnected
                            ? "bg-state-flow/20 text-state-flow"
                            : "bg-state-confusion/20 text-state-confusion"
                        )}
                      >
                        {wsConnected ? "Connected" : "Disconnected"}
                      </span>
                    </div>
                    <input
                      type="text"
                      value={backendUrl}
                      onChange={(e) => setBackendUrl(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-neural-surface border border-neural-hover text-text-primary text-sm focus:outline-none focus:border-accent-primary"
                      placeholder="http://localhost:8085"
                    />
                    <div className="flex gap-2 mt-2">
                      <motion.button
                        onClick={connectWebSocket}
                        className="flex-1 py-2 rounded-lg bg-accent-primary/20 text-accent-primary text-sm font-medium hover:bg-accent-primary/30 transition-colors"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        Connect
                      </motion.button>
                      <motion.button
                        onClick={disconnectWebSocket}
                        className="flex-1 py-2 rounded-lg bg-state-confusion/20 text-state-confusion text-sm font-medium hover:bg-state-confusion/30 transition-colors"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        Disconnect
                      </motion.button>
                    </div>
                  </div>

                  <SelectOption
                    label="Streaming Mode"
                    description="How to receive responses"
                    value={preferences.streamingMode}
                    options={[
                      { value: "websocket", label: "WebSocket" },
                      { value: "http", label: "HTTP" },
                    ]}
                    onChange={(v) => updatePreference("streamingMode", v as "http" | "websocket")}
                  />
                </SettingsSection>

                {/* Display Section */}
                <SettingsSection title="Display" icon={<Eye className="w-4 h-4" />}>
                  <SelectOption
                    label="Animation Level"
                    description="Motion and transitions"
                    value={preferences.animations}
                    options={[
                      { value: "full", label: "Full" },
                      { value: "reduced", label: "Reduced" },
                      { value: "none", label: "None" },
                    ]}
                    onChange={(v) => updatePreference("animations", v as "full" | "reduced" | "none")}
                  />

                  <Toggle
                    label="Show Tools Used"
                    description="Display tool badges on messages"
                    checked={preferences.showToolsUsed}
                    onChange={(v) => updatePreference("showToolsUsed", v)}
                  />
                </SettingsSection>

                {/* Data Section */}
                <SettingsSection title="Data" icon={<Sparkles className="w-4 h-4" />}>
                  <motion.button
                    onClick={handleExportChat}
                    className="w-full flex items-center gap-3 p-3 rounded-lg bg-neural-elevated/50 hover:bg-neural-elevated transition-colors text-left"
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                  >
                    <Download className="w-4 h-4 text-accent-primary" />
                    <div>
                      <div className="text-sm font-medium text-text-primary">Export Chat</div>
                      <div className="text-xs text-text-muted">
                        Download as JSON ({messages.length} messages)
                      </div>
                    </div>
                  </motion.button>

                  <motion.button
                    onClick={() => {
                      if (confirm("Are you sure you want to clear all messages?")) {
                        clearMessages();
                      }
                    }}
                    className="w-full flex items-center gap-3 p-3 rounded-lg bg-state-confusion/10 hover:bg-state-confusion/20 transition-colors text-left"
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                  >
                    <Trash2 className="w-4 h-4 text-state-confusion" />
                    <div>
                      <div className="text-sm font-medium text-state-confusion">Clear Chat</div>
                      <div className="text-xs text-text-muted">Remove all messages</div>
                    </div>
                  </motion.button>
                </SettingsSection>
              </div>

              {/* Footer */}
              <div className="p-4 border-t border-neural-hover text-center">
                <p className="text-xs text-text-muted">
                  AVA v3 - Cortex-Medulla Architecture
                </p>
              </div>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
