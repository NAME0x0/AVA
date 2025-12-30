'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

type ThemeMode = 'light' | 'dark' | 'system';
type FontSize = 'small' | 'medium' | 'large';

interface PersonalizationStepProps {
  onNext: () => void;
  onBack: () => void;
  userName: string;
  onUserNameChange: (name: string) => void;
  themeMode: ThemeMode;
  onThemeModeChange: (mode: ThemeMode) => void;
  fontSize: FontSize;
  onFontSizeChange: (size: FontSize) => void;
}

const THEME_OPTIONS: { value: ThemeMode; label: string; icon: string }[] = [
  { value: 'light', label: 'Light', icon: '☀' },
  { value: 'dark', label: 'Dark', icon: '☾' },
  { value: 'system', label: 'System', icon: '◐' },
];

const FONT_SIZE_OPTIONS: { value: FontSize; label: string }[] = [
  { value: 'small', label: 'S' },
  { value: 'medium', label: 'M' },
  { value: 'large', label: 'L' },
];

export function PersonalizationStep({
  onNext,
  onBack,
  userName,
  onUserNameChange,
  themeMode,
  onThemeModeChange,
  fontSize,
  onFontSizeChange,
}: PersonalizationStepProps) {
  return (
    <div className="neural-card p-8">
      {/* Title */}
      <motion.h2
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-xl font-semibold text-text-primary mb-2 text-center"
      >
        Personalization
      </motion.h2>

      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="text-text-secondary text-sm mb-8 text-center"
      >
        Make AVA yours
      </motion.p>

      <div className="space-y-6">
        {/* Theme Selection */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <label className="block text-sm text-text-secondary mb-3">
            Theme
          </label>
          <div className="grid grid-cols-3 gap-3">
            {THEME_OPTIONS.map((option) => (
              <button
                key={option.value}
                onClick={() => onThemeModeChange(option.value)}
                className={cn(
                  'p-4 rounded-lg border transition-all duration-200 btn-lift',
                  themeMode === option.value
                    ? 'border-accent-primary bg-accent-subtle'
                    : 'border-neural-border bg-neural-elevated hover:border-accent-dim'
                )}
              >
                <span className="text-2xl mb-2 block">{option.icon}</span>
                <span className="text-sm text-text-primary">{option.label}</span>
                {option.value === 'system' && (
                  <span className="text-xs text-accent-primary block mt-1">
                    Recommended
                  </span>
                )}
              </button>
            ))}
          </div>
        </motion.div>

        {/* Font Size Selection */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <label className="block text-sm text-text-secondary mb-3">
            Font Size
          </label>
          <div className="flex gap-3">
            {FONT_SIZE_OPTIONS.map((option) => (
              <button
                key={option.value}
                onClick={() => onFontSizeChange(option.value)}
                className={cn(
                  'w-12 h-12 rounded-lg border transition-all duration-200 btn-lift font-mono',
                  fontSize === option.value
                    ? 'border-accent-primary bg-accent-subtle text-accent-primary'
                    : 'border-neural-border bg-neural-elevated hover:border-accent-dim text-text-secondary'
                )}
              >
                {option.label}
              </button>
            ))}
          </div>
        </motion.div>

        {/* Name Input */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <label className="block text-sm text-text-secondary mb-3">
            What should I call you? <span className="text-text-muted">(optional)</span>
          </label>
          <input
            type="text"
            value={userName}
            onChange={(e) => onUserNameChange(e.target.value)}
            placeholder="Enter your name"
            className="neural-input w-full"
            maxLength={50}
          />
        </motion.div>
      </div>

      {/* Navigation */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="flex justify-between mt-8"
      >
        <button
          onClick={onBack}
          className="neural-button px-6 py-2 rounded-lg btn-lift"
        >
          Back
        </button>
        <button
          onClick={onNext}
          className="neural-button-primary px-6 py-2 rounded-lg btn-lift"
        >
          Continue
        </button>
      </motion.div>
    </div>
  );
}

export default PersonalizationStep;
