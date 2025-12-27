/**
 * Animation Library for AVA UI
 *
 * Centralized animation configurations for consistent motion design.
 * Uses Framer Motion's spring and tween animations.
 */

import { Transition, Variants } from "framer-motion";

// ============================================================================
// Spring Configurations
// ============================================================================

export const springConfig = {
  /** Gentle spring for subtle movements */
  gentle: { type: "spring" as const, stiffness: 200, damping: 20 },
  /** Snappy spring for quick interactions */
  snappy: { type: "spring" as const, stiffness: 400, damping: 25 },
  /** Bouncy spring for playful effects */
  bouncy: { type: "spring" as const, stiffness: 500, damping: 15 },
  /** Stiff spring for immediate feedback */
  stiff: { type: "spring" as const, stiffness: 600, damping: 30 },
};

// ============================================================================
// Transition Presets
// ============================================================================

export const transitions = {
  /** Fast transition for micro-interactions */
  fast: { duration: 0.15, ease: "easeOut" } as Transition,
  /** Normal transition for most animations */
  normal: { duration: 0.3, ease: "easeInOut" } as Transition,
  /** Slow transition for emphasis */
  slow: { duration: 0.5, ease: "easeInOut" } as Transition,
  /** Entrance transition */
  enter: { duration: 0.3, ease: [0.4, 0, 0.2, 1] } as Transition,
  /** Exit transition */
  exit: { duration: 0.2, ease: [0.4, 0, 1, 1] } as Transition,
};

// ============================================================================
// Animation Variants
// ============================================================================

/** Fade in from below */
export const fadeInUp: Variants = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -5 },
};

/** Fade in from above */
export const fadeInDown: Variants = {
  initial: { opacity: 0, y: -10 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: 10 },
};

/** Fade in from left */
export const fadeInLeft: Variants = {
  initial: { opacity: 0, x: -20 },
  animate: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: -10 },
};

/** Fade in from right */
export const fadeInRight: Variants = {
  initial: { opacity: 0, x: 20 },
  animate: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: 10 },
};

/** Scale in from center */
export const scaleIn: Variants = {
  initial: { opacity: 0, scale: 0.95 },
  animate: { opacity: 1, scale: 1 },
  exit: { opacity: 0, scale: 0.95 },
};

/** Pop in effect */
export const popIn: Variants = {
  initial: { opacity: 0, scale: 0.8 },
  animate: {
    opacity: 1,
    scale: 1,
    transition: springConfig.bouncy,
  },
  exit: { opacity: 0, scale: 0.8 },
};

// ============================================================================
// Interaction Helpers
// ============================================================================

/** Scale on hover and tap */
export const scaleOnHover = {
  whileHover: { scale: 1.02 },
  whileTap: { scale: 0.98 },
};

/** Scale on hover and tap (larger) */
export const scaleOnHoverLarge = {
  whileHover: { scale: 1.05 },
  whileTap: { scale: 0.95 },
};

/** Lift on hover */
export const liftOnHover = {
  whileHover: { y: -2, boxShadow: "0 4px 20px rgba(0, 0, 0, 0.15)" },
  whileTap: { y: 0 },
};

/** Slide on hover */
export const slideOnHover = {
  whileHover: { x: 4 },
  whileTap: { x: 2 },
};

// ============================================================================
// Glow Effects
// ============================================================================

/** Pulsing glow effect */
export const glowPulse = (color: string) => ({
  animate: {
    boxShadow: [
      `0 0 10px ${color}33`,
      `0 0 20px ${color}66`,
      `0 0 10px ${color}33`,
    ],
  },
  transition: { duration: 2, repeat: Infinity },
});

/** Static glow on hover */
export const glowOnHover = (color: string) => ({
  whileHover: {
    boxShadow: `0 0 20px ${color}40`,
  },
});

// ============================================================================
// Stagger Configurations
// ============================================================================

/** Container variants for staggered children */
export const staggerContainer: Variants = {
  initial: { opacity: 0 },
  animate: {
    opacity: 1,
    transition: {
      staggerChildren: 0.05,
      delayChildren: 0.1,
    },
  },
  exit: {
    opacity: 0,
    transition: {
      staggerChildren: 0.03,
      staggerDirection: -1,
    },
  },
};

/** Fast stagger for lists */
export const staggerFast = {
  staggerChildren: 0.02,
  delayChildren: 0.05,
};

/** Normal stagger for lists */
export const staggerNormal = {
  staggerChildren: 0.05,
  delayChildren: 0.1,
};

/** Slow stagger for emphasis */
export const staggerSlow = {
  staggerChildren: 0.1,
  delayChildren: 0.2,
};

// ============================================================================
// Special Effects
// ============================================================================

/** Typing cursor blink */
export const cursorBlink = {
  animate: { opacity: [1, 0] },
  transition: { duration: 0.5, repeat: Infinity, repeatType: "reverse" as const },
};

/** Loading spinner */
export const spinnerRotate = {
  animate: { rotate: 360 },
  transition: { duration: 1, repeat: Infinity, ease: "linear" },
};

/** Pulse scale effect */
export const pulseScale = {
  animate: { scale: [1, 1.05, 1] },
  transition: { duration: 1, repeat: Infinity, ease: "easeInOut" },
};

/** Shake effect for errors */
export const shake = {
  animate: { x: [0, -10, 10, -10, 10, 0] },
  transition: { duration: 0.5 },
};

// ============================================================================
// Component-Specific Animations
// ============================================================================

/** Message bubble animation */
export const messageBubble: Variants = {
  initial: { opacity: 0, y: 20, scale: 0.95 },
  animate: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      ...springConfig.snappy,
      duration: 0.3,
    },
  },
  exit: {
    opacity: 0,
    scale: 0.95,
    transition: transitions.fast,
  },
};

/** Sidebar animation */
export const sidebar: Variants = {
  initial: { x: -280, opacity: 0 },
  animate: {
    x: 0,
    opacity: 1,
    transition: transitions.normal,
  },
  exit: {
    x: -280,
    opacity: 0,
    transition: transitions.exit,
  },
};

/** Modal animation */
export const modal: Variants = {
  initial: { opacity: 0, scale: 0.95, y: -20 },
  animate: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: springConfig.snappy,
  },
  exit: {
    opacity: 0,
    scale: 0.95,
    y: -10,
    transition: transitions.fast,
  },
};

/** Tooltip animation */
export const tooltip: Variants = {
  initial: { opacity: 0, scale: 0.8, y: 5 },
  animate: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: springConfig.stiff,
  },
  exit: {
    opacity: 0,
    scale: 0.8,
    transition: { duration: 0.1 },
  },
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Create a delay-based stagger animation
 */
export const createStaggerDelay = (index: number, baseDelay = 0.05) => ({
  initial: { opacity: 0, y: 10 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { delay: index * baseDelay },
  },
});

/**
 * Create a custom spring animation
 */
export const createSpring = (
  stiffness: number,
  damping: number,
  mass: number = 1
) => ({
  type: "spring" as const,
  stiffness,
  damping,
  mass,
});

/**
 * Create a custom tween animation
 */
export const createTween = (
  duration: number,
  ease: string | number[] = "easeInOut"
) => ({
  type: "tween" as const,
  duration,
  ease,
});
