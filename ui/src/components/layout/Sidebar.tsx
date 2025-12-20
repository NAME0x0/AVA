"use client";

import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  Zap,
  Database,
  Activity,
  Moon,
  TrendingUp,
  Sparkles,
  CircuitBoard,
} from "lucide-react";
import { useCoreStore } from "@/stores/core";
import { cn, getCognitiveStateBgColor, getCognitiveStateColor, formatNumber, getComponentColor } from "@/lib/utils";
import { CognitiveStateCard } from "@/components/metrics/CognitiveStateCard";
import { MetricCard } from "@/components/metrics/MetricCard";
import { NeuralActivity } from "@/components/metrics/NeuralActivity";
import { BeliefStateCard } from "@/components/metrics/BeliefStateCard";

interface SidebarProps {
  isOpen: boolean;
}

export function Sidebar({ isOpen }: SidebarProps) {
  const { cognitiveState, memoryStats, systemState, beliefState, forceCortex, forceSleep } = useCoreStore();

  const componentColors = getComponentColor(systemState.activeComponent);

  return (
    <AnimatePresence mode="wait">
      {isOpen && (
        <motion.aside
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: 300, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
          className="h-full bg-neural-surface/50 backdrop-blur-sm border-r border-neural-hover overflow-hidden shrink-0"
        >
          <div className="w-[300px] h-full overflow-y-auto p-4 space-y-4">
            {/* Active Component Indicator */}
            <motion.div
              className={cn(
                "p-3 rounded-xl border",
                componentColors.bg,
                componentColors.glow ? componentColors.glow : "",
                "border-current/20"
              )}
              animate={{
                boxShadow:
                  systemState.activeComponent !== "idle"
                    ? [
                        "0 0 10px currentColor",
                        "0 0 20px currentColor",
                        "0 0 10px currentColor",
                      ]
                    : "none",
              }}
              transition={{ duration: 1.5, repeat: Infinity }}
            >
              <div className="flex items-center gap-2">
                <CircuitBoard className={cn("w-4 h-4", componentColors.text)} />
                <span className="text-xs text-text-muted uppercase tracking-wider">
                  Active Component
                </span>
              </div>
              <div className={cn("mt-1 font-semibold text-lg capitalize", componentColors.text)}>
                {systemState.activeComponent === "idle"
                  ? "Idle"
                  : systemState.activeComponent}
              </div>
            </motion.div>

            {/* Cognitive State */}
            <section>
              <h3 className="text-xs text-text-muted uppercase tracking-wider mb-2 flex items-center gap-2">
                <Brain className="w-3 h-3" />
                Cognitive State
              </h3>
              <CognitiveStateCard state={cognitiveState} />
            </section>

            {/* Neural Activity Visualization */}
            <section>
              <h3 className="text-xs text-text-muted uppercase tracking-wider mb-2 flex items-center gap-2">
                <Activity className="w-3 h-3" />
                Neural Activity
              </h3>
              <NeuralActivity
                entropy={cognitiveState.entropy}
                varentropy={cognitiveState.varentropy}
              />
            </section>

            {/* Metrics Grid */}
            <section>
              <h3 className="text-xs text-text-muted uppercase tracking-wider mb-2 flex items-center gap-2">
                <TrendingUp className="w-3 h-3" />
                Metrics
              </h3>
              <div className="grid grid-cols-2 gap-2">
                <MetricCard
                  label="Entropy"
                  value={cognitiveState.entropy}
                  unit="H"
                  color="text-state-hesitation"
                  icon={<Zap className="w-3 h-3" />}
                />
                <MetricCard
                  label="Varentropy"
                  value={cognitiveState.varentropy}
                  unit="V"
                  color="text-state-creative"
                  icon={<Sparkles className="w-3 h-3" />}
                />
                <MetricCard
                  label="Surprise"
                  value={cognitiveState.surprise}
                  unit="σ"
                  color="text-memory-surprise"
                  icon={<Activity className="w-3 h-3" />}
                />
                <MetricCard
                  label="Memories"
                  value={memoryStats.totalMemories}
                  unit=""
                  color="text-memory-episodic"
                  icon={<Database className="w-3 h-3" />}
                />
              </div>
            </section>

            {/* Belief State (Active Inference) */}
            <section>
              <h3 className="text-xs text-text-muted uppercase tracking-wider mb-2 flex items-center gap-2">
                <Brain className="w-3 h-3" />
                Belief State
              </h3>
              <BeliefStateCard state={beliefState} />
            </section>

            {/* Actions */}
            <section className="pt-2 border-t border-neural-hover">
              <div className="space-y-2">
                <button
                  onClick={forceCortex}
                  className="w-full neural-button flex items-center justify-center gap-2"
                >
                  <Brain className="w-4 h-4 text-cortex-active" />
                  <span>Force Cortex</span>
                </button>
                <button
                  onClick={forceSleep}
                  className="w-full neural-button flex items-center justify-center gap-2"
                >
                  <Moon className="w-4 h-4 text-sleep-rem" />
                  <span>Force Sleep</span>
                </button>
              </div>
            </section>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
