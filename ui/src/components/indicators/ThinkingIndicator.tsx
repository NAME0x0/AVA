"use client";

import { motion } from "framer-motion";

interface ThinkingIndicatorProps {
  stage?: "idle" | "perceiving" | "routing" | "searching" | "generating" | "verifying";
  className?: string;
}

const STAGE_LABELS: Record<string, string> = {
  idle: "",
  perceiving: "Perceiving input...",
  routing: "Routing to processor...",
  searching: "Searching for information...",
  generating: "Generating response...",
  verifying: "Verifying accuracy...",
};

/**
 * ThinkingIndicator - Neural activity visualization
 *
 * A brain-inspired animation showing the current processing stage.
 * Uses animated neural pathways to represent cognitive activity.
 */
export function ThinkingIndicator({
  stage = "generating",
  className = "",
}: ThinkingIndicatorProps) {
  const stageLabel = STAGE_LABELS[stage] || "Thinking...";

  return (
    <div className={`flex flex-col items-center gap-3 p-4 ${className}`}>
      {/* Neural Brain Animation */}
      <div className="relative w-16 h-16">
        {/* Outer ring - slow pulse */}
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-accent-primary/30"
          animate={{
            scale: [1, 1.1, 1],
            opacity: [0.3, 0.6, 0.3],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Middle ring */}
        <motion.div
          className="absolute inset-2 rounded-full border-2 border-accent-dim/40"
          animate={{
            scale: [1, 1.15, 1],
            opacity: [0.4, 0.7, 0.4],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 0.2,
          }}
        />

        {/* Inner core */}
        <motion.div
          className="absolute inset-4 rounded-full bg-gradient-to-br from-accent-primary to-accent-dim"
          animate={{
            scale: [0.9, 1.1, 0.9],
            opacity: [0.7, 1, 0.7],
          }}
          transition={{
            duration: 1,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Neural sparks */}
        {[0, 60, 120, 180, 240, 300].map((angle, i) => (
          <motion.div
            key={angle}
            className="absolute w-1 h-1 rounded-full bg-accent-primary"
            style={{
              top: "50%",
              left: "50%",
            }}
            animate={{
              x: [0, Math.cos((angle * Math.PI) / 180) * 30],
              y: [0, Math.sin((angle * Math.PI) / 180) * 30],
              opacity: [0, 1, 0],
              scale: [0.5, 1.5, 0.5],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              delay: i * 0.2,
              ease: "easeOut",
            }}
          />
        ))}
      </div>

      {/* Stage label */}
      <motion.span
        className="text-sm text-text-muted font-medium"
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 1.5, repeat: Infinity }}
      >
        {stageLabel}
      </motion.span>

      {/* Progress dots */}
      <div className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className="w-1.5 h-1.5 rounded-full bg-accent-primary"
            animate={{
              scale: [1, 1.5, 1],
              opacity: [0.3, 1, 0.3],
            }}
            transition={{
              duration: 0.8,
              repeat: Infinity,
              delay: i * 0.2,
            }}
          />
        ))}
      </div>
    </div>
  );
}
