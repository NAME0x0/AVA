'use client';

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useCoreStore } from '@/stores/core';
import { useTheme } from '@/providers/ThemeProvider';
import { WelcomeStep } from './steps/WelcomeStep';
import { SystemCheckStep } from './steps/SystemCheckStep';
import { PersonalizationStep } from './steps/PersonalizationStep';
import { CompleteStep } from './steps/CompleteStep';
import { StepIndicator } from './shared/StepIndicator';

export type WizardStep = 'welcome' | 'system-check' | 'personalization' | 'complete';

const STEPS: WizardStep[] = ['welcome', 'system-check', 'personalization', 'complete'];

interface WizardOverlayProps {
  onComplete: () => void;
}

export function WizardOverlay({ onComplete }: WizardOverlayProps) {
  const [currentStep, setCurrentStep] = useState<WizardStep>('welcome');
  const [userName, setUserName] = useState<string>('');

  const setSetupComplete = useCoreStore((s) => s.setSetupComplete);
  const updatePreference = useCoreStore((s) => s.updatePreference);
  const preferences = useCoreStore((s) => s.preferences);
  const { setMode } = useTheme();

  const currentIndex = STEPS.indexOf(currentStep);

  const goToNext = useCallback(() => {
    const nextIndex = currentIndex + 1;
    if (nextIndex < STEPS.length) {
      setCurrentStep(STEPS[nextIndex]);
    }
  }, [currentIndex]);

  const goToPrevious = useCallback(() => {
    const prevIndex = currentIndex - 1;
    if (prevIndex >= 0) {
      setCurrentStep(STEPS[prevIndex]);
    }
  }, [currentIndex]);

  const handleComplete = useCallback(() => {
    setSetupComplete(userName || undefined);
    onComplete();
  }, [userName, setSetupComplete, onComplete]);

  const handleThemeChange = useCallback((mode: 'light' | 'dark' | 'system') => {
    setMode(mode);
    updatePreference('theme', { ...preferences.theme, mode });
  }, [setMode, updatePreference, preferences.theme]);

  const handleFontSizeChange = useCallback((fontSize: 'small' | 'medium' | 'large') => {
    updatePreference('theme', { ...preferences.theme, fontSize });
  }, [updatePreference, preferences.theme]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-neural-void"
    >
      <div className="w-full max-w-lg mx-4">
        {/* Step Indicator */}
        <StepIndicator
          steps={STEPS.length}
          currentStep={currentIndex}
          className="mb-8"
        />

        {/* Step Content */}
        <AnimatePresence mode="wait">
          {currentStep === 'welcome' && (
            <motion.div
              key="welcome"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
            >
              <WelcomeStep onNext={goToNext} />
            </motion.div>
          )}

          {currentStep === 'system-check' && (
            <motion.div
              key="system-check"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
            >
              <SystemCheckStep
                onNext={goToNext}
                onBack={goToPrevious}
              />
            </motion.div>
          )}

          {currentStep === 'personalization' && (
            <motion.div
              key="personalization"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
            >
              <PersonalizationStep
                onNext={goToNext}
                onBack={goToPrevious}
                userName={userName}
                onUserNameChange={setUserName}
                themeMode={preferences.theme.mode}
                onThemeModeChange={handleThemeChange}
                fontSize={preferences.theme.fontSize}
                onFontSizeChange={handleFontSizeChange}
              />
            </motion.div>
          )}

          {currentStep === 'complete' && (
            <motion.div
              key="complete"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
            >
              <CompleteStep
                userName={userName}
                onComplete={handleComplete}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

export default WizardOverlay;
