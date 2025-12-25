import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format time duration in human-readable format
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}

/**
 * Format number with appropriate suffix
 */
export function formatNumber(num: number): string {
  if (num < 1000) return num.toFixed(1);
  if (num < 1000000) return `${(num / 1000).toFixed(1)}K`;
  return `${(num / 1000000).toFixed(1)}M`;
}

/**
 * Get color class for cognitive state
 */
export function getCognitiveStateColor(label: string): string {
  switch (label) {
    case "FLOW":
      return "text-state-flow";
    case "HESITATION":
      return "text-state-hesitation";
    case "CONFUSION":
      return "text-state-confusion";
    case "CREATIVE":
      return "text-state-creative";
    case "THINKING":
      return "text-state-thinking";
    default:
      return "text-text-muted";
  }
}

/**
 * Get background color class for cognitive state
 */
export function getCognitiveStateBgColor(label: string): string {
  switch (label) {
    case "FLOW":
      return "bg-state-flow/20 border-state-flow/50";
    case "HESITATION":
      return "bg-state-hesitation/20 border-state-hesitation/50";
    case "CONFUSION":
      return "bg-state-confusion/20 border-state-confusion/50";
    case "CREATIVE":
      return "bg-state-creative/20 border-state-creative/50";
    case "THINKING":
      return "bg-state-thinking/20 border-state-thinking/50";
    default:
      return "bg-neural-hover border-neural-hover";
  }
}

/**
 * Get active component color
 */
export function getComponentColor(component: string): {
  text: string;
  bg: string;
  glow: string;
} {
  switch (component) {
    case "cortex":
      return {
        text: "text-cortex-active",
        bg: "bg-cortex-active/20",
        glow: "shadow-cortex",
      };
    case "medulla":
      return {
        text: "text-medulla-active",
        bg: "bg-medulla-active/20",
        glow: "shadow-medulla",
      };
    case "bridge":
      return {
        text: "text-accent-primary",
        bg: "bg-accent-primary/20",
        glow: "shadow-neural",
      };
    default:
      return {
        text: "text-text-muted",
        bg: "bg-neural-hover",
        glow: "",
      };
  }
}
