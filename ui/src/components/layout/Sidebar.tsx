"use client";

import { Suspense, lazy, useEffect, useState } from "react";
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
  Loader2,
  X,
} from "lucide-react";
import { useCoreStore } from "@/stores/core";
import { cn, getCognitiveStateBgColor, getCognitiveStateColor, formatNumber, getComponentColor } from "@/lib/utils";
import { CognitiveStateCard } from "@/components/metrics/CognitiveStateCard";
import { MetricCard } from "@/components/metrics/MetricCard";
import { NeuralActivity } from "@/components/metrics/NeuralActivity";
import { BeliefStateCard } from "@/components/metrics/BeliefStateCard";

// Lazy load Three.js component for performance
const NeuralBrain = lazy(() =>
  import("@/components/three/NeuralBrain").then((mod) => ({ default: mod.NeuralBrain }))
);

function BrainLoader() {
  return (
    <div className="w-full h-[150px] sm:h-[200px] flex items-center justify-center bg-neural-surface/30 rounded-xl">
      <Loader2 className="w-6 h-6 text-accent-primary animate-spin" />
    </div>
  );
}

interface SidebarProps {
  isOpen: boolean;
}

export function Sidebar({ isOpen }: SidebarProps) {
  const { cognitiveState, memoryStats, systemState, beliefState, forceCortex, forceSleep, toggleSidebar } = useCoreStore();
  const [isMobile, setIsMobile] = useState(false);

  // Detect mobile viewport
  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  const componentColors = getComponentColor(systemState.activeComponent);

  // Mobile: Full-screen overlay | Desktop: Side panel
  const sidebarWidth = isMobile ? "100%" : 300;

  return (
    <AnimatePresence mode="wait">
      {isOpen && (
        <motion.div key="sidebar-container" initial={{ opacity: 1 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
          {/* Mobile backdrop */}
          {isMobile && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 md:hidden"
              onClick={toggleSidebar}
            />
          )}

          <motion.aside
            initial={{ width: 0, opacity: 0, x: isMobile ? -100 : 0 }}
            animate={{ width: sidebarWidth, opacity: 1, x: 0 }}
            exit={{ width: 0, opacity: 0, x: isMobile ? -100 : 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className={cn(
              "h-full bg-neural-surface/50 backdrop-blur-sm border-r border-neural-hover overflow-hidden shrink-0",
              isMobile && "fixed left-0 top-0 z-50 bg-neural-surface"
            )}
            role="complementary"
            aria-label="System metrics sidebar"
          >
            <div className={cn(
              "h-full overflow-y-auto p-3 sm:p-4 space-y-3 sm:space-y-4",
              isMobile ? "w-full" : "w-[300px]"
            )}>
              {/* Mobile close button */}
              {isMobile && (
                <button
                  onClick={toggleSidebar}
                  className="absolute top-3 right-3 p-2 rounded-lg bg-neural-elevated hover:bg-neural-hover transition-colors z-10"
                  aria-label="Close sidebar"
                >
                  <X className="w-5 h-5 text-text-muted" />
                </button>
              )}
            {/* 3D Neural Brain Visualization */}
            <section className="relative">
              <div className="absolute top-2 left-2 z-10 flex items-center gap-1.5 px-2 py-1 bg-neural-surface/80 backdrop-blur-sm rounded-lg">
                <div className={cn(
                  "w-2 h-2 rounded-full animate-pulse",
                  systemState.activeComponent === "cortex" ? "bg-cortex-active" :
                  systemState.activeComponent === "medulla" ? "bg-primary" : "bg-text-muted"
                )} />
                <span className="text-[10px] text-text-muted uppercase tracking-wider">
                  Live Neural Activity
                </span>
              </div>
              <div className="h-[200px] rounded-xl overflow-hidden border border-neural-hover bg-neural-surface/30">
                <Suspense fallback={<BrainLoader />}>
                  <NeuralBrain
                    activeComponent={systemState.activeComponent}
                    cognitiveState={cognitiveState.label}
                    entropy={cognitiveState.entropy}
                    activity={cognitiveState.confidence}
                  />
                </Suspense>
              </div>
            </section>

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
        </motion.div>
      )}
    </AnimatePresence>
  );
}
