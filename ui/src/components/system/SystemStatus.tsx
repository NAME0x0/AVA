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

  return (
    <footer className="h-8 bg-neural-surface/80 backdrop-blur-sm border-t border-neural-hover flex items-center justify-between px-4 text-[11px] shrink-0">
      {/* Left - System State */}
      <div className="flex items-center gap-4">
        {/* System Status */}
        <div className="flex items-center gap-1.5">
          <motion.div
            className={cn(
              "w-1.5 h-1.5 rounded-full",
              systemState.systemState === "running"
                ? "bg-status-success"
                : systemState.systemState === "sleeping"
                ? "bg-sleep-rem"
                : systemState.systemState === "error"
                ? "bg-status-error"
                : "bg-status-warning"
            )}
            animate={{
              opacity: systemState.systemState === "running" ? 1 : [1, 0.5, 1],
            }}
            transition={{
              duration: 1,
              repeat: systemState.systemState !== "running" ? Infinity : 0,
            }}
          />
          <span className="text-text-secondary capitalize">
            {systemState.systemState}
          </span>
        </div>

        {/* Uptime */}
        <div className="flex items-center gap-1 text-text-muted">
          <Clock className="w-3 h-3" />
          <span>{formatDuration(systemState.uptimeSeconds)}</span>
        </div>

        {/* Active Component */}
        <div
          className={cn(
            "flex items-center gap-1 px-1.5 py-0.5 rounded",
            systemState.activeComponent === "cortex"
              ? "bg-cortex-active/20 text-cortex-active"
              : systemState.activeComponent === "medulla"
              ? "bg-medulla-active/20 text-medulla-active"
              : "bg-neural-hover text-text-muted"
          )}
        >
          {systemState.activeComponent === "cortex" ? (
            <Brain className="w-3 h-3" />
          ) : (
            <Zap className="w-3 h-3" />
          )}
          <span className="capitalize">{systemState.activeComponent}</span>
        </div>
      </div>

      {/* Center - Cognitive State */}
      <div className="flex items-center gap-3">
        <span className="text-text-muted">Cognitive:</span>
        <span
          className={cn(
            "font-medium",
            cognitiveState.label === "FLOW"
              ? "text-state-flow"
              : cognitiveState.label === "HESITATION"
              ? "text-state-hesitation"
              : cognitiveState.label === "CONFUSION"
              ? "text-state-confusion"
              : cognitiveState.label === "CREATIVE"
              ? "text-state-creative"
              : "text-text-secondary"
          )}
        >
          {cognitiveState.label}
        </span>
        <span className="text-text-muted">
          H:{cognitiveState.entropy.toFixed(1)} V:{cognitiveState.varentropy.toFixed(1)}
        </span>
      </div>

      {/* Right - Statistics */}
      <div className="flex items-center gap-4">
        {/* Interactions */}
        <div className="flex items-center gap-1 text-text-muted">
          <MessageSquare className="w-3 h-3" />
          <span>{systemState.totalInteractions}</span>
        </div>

        {/* Cortex invocations */}
        <div className="flex items-center gap-1 text-cortex-active/70">
          <Brain className="w-3 h-3" />
          <span>{systemState.cortexInvocations}</span>
        </div>

        {/* Memory */}
        <div className="flex items-center gap-1 text-memory-episodic/70">
          <HardDrive className="w-3 h-3" />
          <span>{memoryStats.totalMemories}</span>
        </div>

        {/* Avg Response Time */}
        <div className="flex items-center gap-1 text-text-muted">
          <Activity className="w-3 h-3" />
          <span>{systemState.avgResponseTimeMs.toFixed(0)}ms avg</span>
        </div>
      </div>
    </footer>
  );
}
