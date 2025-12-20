"use client";

import { motion } from "framer-motion";
import { BeliefState } from "@/stores/core";
import { cn } from "@/lib/utils";

interface BeliefStateCardProps {
  state: BeliefState;
}

export function BeliefStateCard({ state }: BeliefStateCardProps) {
  // Get top 3 policies by probability
  const topPolicies = Object.entries(state.policyDistribution)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 3);

  return (
    <div className="neural-card space-y-3">
      {/* Current State */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-text-muted">Current State</span>
        <span className="text-sm font-medium text-text-primary">
          {state.currentState || "IDLE"}
        </span>
      </div>

      {/* Free Energy */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-text-muted">Free Energy</span>
        <div className="flex items-center gap-2">
          <motion.div
            className="h-1.5 bg-neural-elevated rounded-full overflow-hidden w-24"
          >
            <motion.div
              className="h-full bg-gradient-to-r from-state-flow to-state-confusion rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(100, state.freeEnergy * 20)}%` }}
              transition={{ duration: 0.5 }}
            />
          </motion.div>
          <span className="text-xs text-text-secondary tabular-nums w-10 text-right">
            {state.freeEnergy.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Policy Distribution */}
      {topPolicies.length > 0 && (
        <div className="space-y-1.5">
          <span className="text-xs text-text-muted">Policy Distribution</span>
          {topPolicies.map(([policy, prob]) => (
            <div key={policy} className="flex items-center gap-2">
              <span className="text-[10px] text-text-secondary w-24 truncate">
                {policy.replace(/_/g, " ")}
              </span>
              <div className="flex-1 h-1 bg-neural-elevated rounded-full overflow-hidden">
                <motion.div
                  className={cn(
                    "h-full rounded-full",
                    policy.includes("CORTEX") || policy.includes("DEEP")
                      ? "bg-cortex-active"
                      : policy.includes("REFLEX")
                      ? "bg-medulla-active"
                      : "bg-accent-primary"
                  )}
                  initial={{ width: 0 }}
                  animate={{ width: `${prob * 100}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
              <span className="text-[10px] text-text-muted w-8 text-right tabular-nums">
                {(prob * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
