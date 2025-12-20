"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface MetricCardProps {
  label: string;
  value: number;
  unit: string;
  color: string;
  icon?: ReactNode;
}

export function MetricCard({ label, value, unit, color, icon }: MetricCardProps) {
  return (
    <motion.div
      className="neural-card"
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
    >
      <div className="flex items-center gap-1.5 text-text-muted mb-1">
        {icon}
        <span className="text-[10px] uppercase tracking-wider">{label}</span>
      </div>
      <div className="flex items-baseline gap-1">
        <motion.span
          className={cn("text-2xl font-bold tabular-nums", color)}
          key={value}
          initial={{ opacity: 0.5, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          {typeof value === "number" ? value.toFixed(1) : value}
        </motion.span>
        {unit && <span className="text-xs text-text-muted">{unit}</span>}
      </div>
    </motion.div>
  );
}
