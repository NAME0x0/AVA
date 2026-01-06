"use client";

import { motion } from "framer-motion";
import { useCoreStore } from "@/stores/core";
import { cn } from "@/lib/utils";

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
 * Cortex (gold) vs Medulla (cyan) color scheme based on active component.
 */
export function ThinkingIndicator({
  stage = "generating",
  className = "",
}: ThinkingIndicatorProps) {
  const stageLabel = STAGE_LABELS[stage] || "Thinking...";
  const systemState = useCoreStore((state) => state.systemState);
  const cognitiveState = useCoreStore((state) => state.cognitiveState);
  const preferences = useCoreStore((state) => state.preferences);

  // Respect reduced motion preference
  const reduceMotion = preferences.theme?.reduceMotion || false;

  // Use cortex colors (gold) when cortex is active, medulla colors (cyan) otherwise
  const isCortexActive = systemState.activeComponent === "cortex" || cognitiveState.shouldThink;
  const primaryColor = isCortexActive ? "var(--accent-gold)" : "var(--accent-cyan)";
  const glowColor = isCortexActive ? "var(--glow-gold)" : "var(--glow-cyan)";

  // Simplified animations for reduced motion
  const animationConfig = reduceMotion
    ? { duration: 0.01, repeat: 0 }
    : undefined;

  // Reduced motion version - simple static indicator
  if (reduceMotion) {
    return (
      <div
        className={cn(
          "flex flex-col items-center gap-3 p-6 rounded-2xl",
          "bg-neural-surface/40 backdrop-blur-md border border-neural-border/30",
          isCortexActive ? "shadow-[0_0_30px_rgba(255,179,71,0.2)]" : "shadow-[0_0_30px_rgba(0,212,200,0.15)]",
          className
        )}
      >
        <div
          className="w-12 h-12 rounded-full"
          style={{
            background: isCortexActive
              ? "linear-gradient(135deg, #FFB347 0%, #F5A623 100%)"
              : "linear-gradient(135deg, #00FFE5 0%, #00D4C8 100%)",
            boxShadow: `0 0 20px ${glowColor}`,
          }}
        />
        <div className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: primaryColor }}
          />
          <span className={cn(
            "text-xs font-medium uppercase tracking-wider",
            isCortexActive ? "text-cortex-active" : "text-medulla-active"
          )}>
            {isCortexActive ? "Cortex" : "Medulla"}
          </span>
        </div>
        <span className="text-sm text-text-muted font-medium">{stageLabel}</span>
      </div>
    );
  }

  return (
    <motion.div
      className={cn(
        "flex flex-col items-center gap-3 p-6 rounded-2xl",
        "bg-neural-surface/40 backdrop-blur-md border border-neural-border/30",
        isCortexActive ? "shadow-[0_0_30px_rgba(255,179,71,0.2)]" : "shadow-[0_0_30px_rgba(0,212,200,0.15)]",
        className
      )}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
    >
      {/* Neural Brain Animation */}
      <div className="relative w-20 h-20">
        {/* Ambient glow */}
        <motion.div
          className="absolute inset-[-20px] rounded-full"
          style={{ background: `radial-gradient(circle, ${glowColor} 0%, transparent 70%)` }}
          animate={{
            opacity: [0.3, 0.6, 0.3],
            scale: [0.9, 1.1, 0.9],
          }}
          transition={{
            duration: 2.5,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Outer ring - slow pulse */}
        <motion.div
          className="absolute inset-0 rounded-full border-2"
          style={{ borderColor: `color-mix(in srgb, ${primaryColor} 30%, transparent)` }}
          animate={{
            scale: [1, 1.15, 1],
            opacity: [0.3, 0.7, 0.3],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Middle ring */}
        <motion.div
          className="absolute inset-2 rounded-full border-2"
          style={{ borderColor: `color-mix(in srgb, ${primaryColor} 50%, transparent)` }}
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.4, 0.8, 0.4],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 0.2,
          }}
        />

        {/* Inner core - gradient */}
        <motion.div
          className="absolute inset-4 rounded-full"
          style={{
            background: isCortexActive
              ? "linear-gradient(135deg, #FFB347 0%, #F5A623 50%, #FF9500 100%)"
              : "linear-gradient(135deg, #00FFE5 0%, #00D4C8 50%, #00A89D 100%)",
            boxShadow: `0 0 20px ${glowColor}`,
          }}
          animate={{
            scale: [0.85, 1.15, 0.85],
            opacity: [0.8, 1, 0.8],
          }}
          transition={{
            duration: 1.2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Neural sparks - more dynamic */}
        {[0, 45, 90, 135, 180, 225, 270, 315].map((angle, i) => (
          <motion.div
            key={angle}
            className="absolute w-1.5 h-1.5 rounded-full"
            style={{
              top: "50%",
              left: "50%",
              backgroundColor: primaryColor,
              boxShadow: `0 0 6px ${primaryColor}`,
            }}
            animate={{
              x: [0, Math.cos((angle * Math.PI) / 180) * 40],
              y: [0, Math.sin((angle * Math.PI) / 180) * 40],
              opacity: [0, 1, 0],
              scale: [0.3, 1.2, 0.3],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: i * 0.15,
              ease: "easeOut",
            }}
          />
        ))}

        {/* Orbiting particle */}
        <motion.div
          className="absolute w-2 h-2 rounded-full"
          style={{
            top: "50%",
            left: "50%",
            backgroundColor: primaryColor,
            boxShadow: `0 0 10px ${primaryColor}`,
          }}
          animate={{
            x: [0, 35, 0, -35, 0],
            y: [-35, 0, 35, 0, -35],
            scale: [0.8, 1, 0.8, 1, 0.8],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      </div>

      {/* Component label */}
      <div className="flex items-center gap-2">
        <motion.div
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: primaryColor }}
          animate={{ scale: [1, 1.3, 1] }}
          transition={{ duration: 0.8, repeat: Infinity }}
        />
        <span className={cn(
          "text-xs font-medium uppercase tracking-wider",
          isCortexActive ? "text-cortex-active" : "text-medulla-active"
        )}>
          {isCortexActive ? "Cortex" : "Medulla"}
        </span>
      </div>

      {/* Stage label */}
      <motion.span
        className="text-sm text-text-muted font-medium"
        animate={{ opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 1.5, repeat: Infinity }}
      >
        {stageLabel}
      </motion.span>

      {/* Progress bar */}
      <div className="w-full max-w-[120px] h-1 bg-neural-hover rounded-full overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: primaryColor }}
          animate={{
            width: ["0%", "100%"],
            opacity: [0.5, 1, 0.5],
          }}
          transition={{
            width: { duration: 2, repeat: Infinity, ease: "easeInOut" },
            opacity: { duration: 1, repeat: Infinity },
          }}
        />
      </div>
    </motion.div>
  );
}
