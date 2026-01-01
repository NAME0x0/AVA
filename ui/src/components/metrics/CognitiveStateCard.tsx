"use client";

import { motion } from "framer-motion";
import { useCoreStore, CognitiveState } from "@/stores/core";
import { cn, getCognitiveStateBgColor, getCognitiveStateColor } from "@/lib/utils";

interface CognitiveStateCardProps {
  state: CognitiveState;
}

export function CognitiveStateCard({ state }: CognitiveStateCardProps) {
  // Safe defaults for potentially undefined values
  const label = state?.label ?? "UNKNOWN";
  const confidence = state?.confidence ?? 0;
  const entropy = state?.entropy ?? 0;
  const varentropy = state?.varentropy ?? 0;
  const surprise = state?.surprise ?? 0;
  const shouldThink = state?.shouldThink ?? false;
  const shouldUseTools = state?.shouldUseTools ?? false;

  const stateColor = getCognitiveStateColor(label);
  const stateBgColor = getCognitiveStateBgColor(label);

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
          <span className={cn("font-semibold", stateColor)}>{label}</span>
        </div>

        {/* Confidence Badge */}
        <div className="text-xs text-text-muted bg-neural-void/50 px-2 py-0.5 rounded-full">
          {(confidence * 100).toFixed(0)}% conf
        </div>
      </div>

      {/* Metrics Row */}
      <div className="mt-2 flex gap-4 text-xs">
        <div>
          <span className="text-text-muted">H:</span>
          <span className="text-text-secondary ml-1">{entropy.toFixed(2)}</span>
        </div>
        <div>
          <span className="text-text-muted">V:</span>
          <span className="text-text-secondary ml-1">{varentropy.toFixed(2)}</span>
        </div>
        <div>
          <span className="text-text-muted">σ:</span>
          <span className="text-text-secondary ml-1">{surprise.toFixed(2)}</span>
        </div>
      </div>

      {/* Flags */}
      <div className="mt-2 flex gap-2">
        {shouldThink && (
          <span className="text-[10px] px-2 py-0.5 rounded bg-cortex-active/20 text-cortex-active border border-cortex-active/30">
            THINKING
          </span>
        )}
        {shouldUseTools && (
          <span className="text-[10px] px-2 py-0.5 rounded bg-accent-primary/20 text-accent-primary border border-accent-primary/30">
            TOOLS
          </span>
        )}
      </div>
    </motion.div>
  );
}
