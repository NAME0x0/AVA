"use client";

import { motion } from "framer-motion";
import {
  Activity,
  Clock,
  Brain,
  Zap,
  HardDrive,
  MessageSquare,
} from "lucide-react";
import { useCoreStore } from "@/stores/core";
import { cn, formatDuration } from "@/lib/utils";

export function SystemStatus() {
  const { connected, systemState, cognitiveState, memoryStats } = useCoreStore();

  // Safe defaults for potentially undefined values
  const label = cognitiveState?.label ?? "UNKNOWN";
  const entropy = cognitiveState?.entropy ?? 0;
  const varentropy = cognitiveState?.varentropy ?? 0;
  const totalMemories = memoryStats?.totalMemories ?? 0;
  const uptimeSeconds = systemState?.uptimeSeconds ?? 0;
  const totalInteractions = systemState?.totalInteractions ?? 0;
  const cortexInvocations = systemState?.cortexInvocations ?? 0;
  const currentSystemState = systemState?.systemState ?? "initializing";
  const activeComponent = systemState?.activeComponent ?? "idle";
  const avgResponseTimeMs = systemState?.avgResponseTimeMs ?? 0;

  return (
    <footer className="h-8 bg-neural-surface/80 backdrop-blur-sm border-t border-neural-hover flex items-center justify-between px-4 text-[11px] shrink-0">
      {/* Left - System State */}
      <div className="flex items-center gap-4">
        {/* System Status */}
        <div className="flex items-center gap-1.5">
          <motion.div
            className={cn(
              "w-1.5 h-1.5 rounded-full",
              currentSystemState === "running"
                ? "bg-status-success"
                : currentSystemState === "sleeping"
                ? "bg-sleep-rem"
                : currentSystemState === "error"
                ? "bg-status-error"
                : "bg-status-warning"
            )}
            animate={{
              opacity: currentSystemState === "running" ? 1 : [1, 0.5, 1],
            }}
            transition={{
              duration: 1,
              repeat: currentSystemState !== "running" ? Infinity : 0,
            }}
          />
          <span className="text-text-secondary capitalize">
            {currentSystemState}
          </span>
        </div>

        {/* Uptime */}
        <div className="flex items-center gap-1 text-text-muted">
          <Clock className="w-3 h-3" />
          <span>{formatDuration(uptimeSeconds)}</span>
        </div>

        {/* Active Component */}
        <div
          className={cn(
            "flex items-center gap-1 px-1.5 py-0.5 rounded",
            activeComponent === "cortex"
              ? "bg-cortex-active/20 text-cortex-active"
              : activeComponent === "medulla"
              ? "bg-medulla-active/20 text-medulla-active"
              : "bg-neural-hover text-text-muted"
          )}
        >
          {activeComponent === "cortex" ? (
            <Brain className="w-3 h-3" />
          ) : (
            <Zap className="w-3 h-3" />
          )}
          <span className="capitalize">{activeComponent}</span>
        </div>
      </div>

      {/* Center - Cognitive State */}
      <div className="flex items-center gap-3">
        <span className="text-text-muted">Cognitive:</span>
        <span
          className={cn(
            "font-medium",
            label === "FLOW"
              ? "text-state-flow"
              : label === "HESITATION"
              ? "text-state-hesitation"
              : label === "CONFUSION"
              ? "text-state-confusion"
              : label === "CREATIVE"
              ? "text-state-creative"
              : "text-text-secondary"
          )}
        >
          {label}
        </span>
        <span className="text-text-muted">
          H:{entropy.toFixed(1)} V:{varentropy.toFixed(1)}
        </span>
      </div>

      {/* Right - Statistics */}
      <div className="flex items-center gap-4">
        {/* Interactions */}
        <div className="flex items-center gap-1 text-text-muted">
          <MessageSquare className="w-3 h-3" />
          <span>{totalInteractions}</span>
        </div>

        {/* Cortex invocations */}
        <div className="flex items-center gap-1 text-cortex-active/70">
          <Brain className="w-3 h-3" />
          <span>{cortexInvocations}</span>
        </div>

        {/* Memory */}
        <div className="flex items-center gap-1 text-memory-episodic/70">
          <HardDrive className="w-3 h-3" />
          <span>{totalMemories}</span>
        </div>

        {/* Avg Response Time */}
        <div className="flex items-center gap-1 text-text-muted">
          <Activity className="w-3 h-3" />
          <span>{avgResponseTimeMs.toFixed(0)}ms avg</span>
        </div>
      </div>
    </footer>
  );
}
