'use client';

import { motion } from 'framer-motion';

interface WelcomeStepProps {
  onNext: () => void;
}

export function WelcomeStep({ onNext }: WelcomeStepProps) {
  return (
    <div className="neural-card p-8 text-center">
      {/* ASCII Art Logo */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="font-mono text-accent-primary mb-6"
      >
        <pre className="text-xs leading-tight inline-block text-left">
{`
    ╭─────────────────╮
    │   ◉  A V A     │
    │  Neural Brain   │
    ╰─────────────────╯
`}
        </pre>
      </motion.div>

      {/* Title */}
      <motion.h1
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-2xl font-semibold text-text-primary mb-2"
      >
        Welcome to AVA
      </motion.h1>

      {/* Subtitle */}
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="text-text-secondary mb-8"
      >
        Your personal neural assistant
      </motion.p>

      {/* Description */}
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="text-text-muted text-sm mb-8 max-w-sm mx-auto"
      >
        AVA uses a dual-brain architecture for fast reflexive responses
        and deep reasoning. Let&apos;s get you set up.
      </motion.p>

      {/* Get Started Button */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <button
          onClick={onNext}
          className="neural-button-primary px-8 py-3 rounded-lg font-medium btn-lift"
        >
          Get Started
        </button>
      </motion.div>

      {/* ASCII Decoration */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="mt-8 font-mono text-xs text-text-muted/30"
      >
        ─ ─ ─ ─ ─ ─ ─ ─ ─
      </motion.div>
    </div>
  );
}

export default WelcomeStep;
