"use client";

import { motion } from "framer-motion";
import { useCoreStore, CognitiveState } from "@/stores/core";
import { cn, getCognitiveStateBgColor, getCognitiveStateColor } from "@/lib/utils";

interface CognitiveStateCardProps {
  state: CognitiveState;
}

export function CognitiveStateCard({ state }: CognitiveStateCardProps) {
  const stateColor = getCognitiveStateColor(state.label);
  const stateBgColor = getCognitiveStateBgColor(state.label);

  return (
    <motion.div
      className={cn(
        "p-3 rounded-xl border transition-all duration-300",
        stateBgColor
      )}
      layout
    >
      <div className="flex items-center justify-between">
        {/* State Label with pulse */}
        <div className="flex items-center gap-2">
          <motion.div
            className={cn("w-2 h-2 rounded-full", stateColor.replace("text-", "bg-"))}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.7, 1, 0.7],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
          <span className={cn("font-semibold", stateColor)}>{state.label}</span>
        </div>

        {/* Confidence Badge */}
        <div className="text-xs text-text-muted bg-neural-void/50 px-2 py-0.5 rounded-full">
          {(state.confidence * 100).toFixed(0)}% conf
        </div>
      </div>

      {/* Metrics Row */}
      <div className="mt-2 flex gap-4 text-xs">
        <div>
          <span className="text-text-muted">H:</span>
          <span className="text-text-secondary ml-1">{state.entropy.toFixed(2)}</span>
        </div>
        <div>
          <span className="text-text-muted">V:</span>
          <span className="text-text-secondary ml-1">{state.varentropy.toFixed(2)}</span>
        </div>
        <div>
          <span className="text-text-muted">σ:</span>
          <span className="text-text-secondary ml-1">{state.surprise.toFixed(2)}</span>
        </div>
      </div>

      {/* Flags */}
      <div className="mt-2 flex gap-2">
        {state.shouldThink && (
          <span className="text-[10px] px-2 py-0.5 rounded bg-cortex-active/20 text-cortex-active border border-cortex-active/30">
            THINKING
          </span>
        )}
        {state.shouldUseTools && (
          <span className="text-[10px] px-2 py-0.5 rounded bg-accent-primary/20 text-accent-primary border border-accent-primary/30">
            TOOLS
          </span>
        )}
      </div>
    </motion.div>
  );
}
