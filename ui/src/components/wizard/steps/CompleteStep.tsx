'use client';

import { motion } from 'framer-motion';

interface CompleteStepProps {
  userName?: string;
  onComplete: () => void;
}

export function CompleteStep({ userName, onComplete }: CompleteStepProps) {
  return (
    <div className="neural-card p-8 text-center">
      {/* Success Animation */}
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: 'spring', stiffness: 200, damping: 15 }}
        className="w-20 h-20 mx-auto mb-6 rounded-full bg-accent-subtle flex items-center justify-center"
      >
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="text-4xl text-accent-primary"
        >
          ✓
        </motion.span>
      </motion.div>

      {/* ASCII Art */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="font-mono text-accent-primary mb-6"
      >
        <pre className="text-xs leading-tight inline-block text-left">
{`
    ╭─────────────────╮
    │   ◉  A V A     │
    │     READY      │
    ╰─────────────────╯
`}
        </pre>
      </motion.div>

      {/* Title */}
      <motion.h2
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="text-2xl font-semibold text-text-primary mb-2"
      >
        AVA is Ready
      </motion.h2>

      {/* Personalized Message */}
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="text-text-secondary mb-8"
      >
        {userName
          ? `Nice to meet you, ${userName}! Your second brain awaits.`
          : 'Your second brain awaits.'}
      </motion.p>

      {/* Start Button */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <button
          onClick={onComplete}
          className="neural-button-primary px-8 py-3 rounded-lg font-medium btn-lift"
        >
          Start Chatting
        </button>
      </motion.div>

      {/* Keyboard Shortcuts Hint */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7 }}
        className="mt-8 text-xs text-text-muted"
      >
        <p>Quick tip: Press <kbd className="px-1.5 py-0.5 bg-neural-elevated rounded text-text-secondary">Ctrl+K</kbd> to open command palette</p>
      </motion.div>

      {/* ASCII Decoration */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="mt-6 font-mono text-xs text-text-muted/30"
      >
        ─ ─ ─ ─ ─ ─ ─ ─ ─
      </motion.div>
    </div>
  );
}

export default CompleteStep;
