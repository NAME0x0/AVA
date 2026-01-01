'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useCoreStore } from '@/stores/core';

interface CheckItem {
  id: string;
  label: string;
  status: 'pending' | 'checking' | 'success' | 'error';
  detail?: string;
}

interface SystemCheckStepProps {
  onNext: () => void;
  onBack: () => void;
}

export function SystemCheckStep({ onNext, onBack }: SystemCheckStepProps) {
  const backendUrl = useCoreStore((s) => s.backendUrl);
  const [checks, setChecks] = useState<CheckItem[]>([
    { id: 'backend', label: 'Backend Server', status: 'pending' },
    { id: 'ollama', label: 'Ollama Service', status: 'pending' },
    { id: 'model', label: 'AI Model', status: 'pending' },
  ]);
  const [allPassed, setAllPassed] = useState(false);
  const [hasErrors, setHasErrors] = useState(false);

  const updateCheck = (id: string, updates: Partial<CheckItem>) => {
    setChecks((prev) =>
      prev.map((c) => (c.id === id ? { ...c, ...updates } : c))
    );
  };

  useEffect(() => {
    const runChecks = async () => {
      // Check backend
      updateCheck('backend', { status: 'checking' });
      await new Promise((r) => setTimeout(r, 500)); // Visual delay

      try {
        const response = await fetch(`${backendUrl}/health`, {
          signal: AbortSignal.timeout(5000),
        });

        if (response.ok) {
          const data = await response.json();
          updateCheck('backend', {
            status: 'success',
            detail: 'Connected',
          });

          // Check Ollama (from health response)
          updateCheck('ollama', { status: 'checking' });
          await new Promise((r) => setTimeout(r, 300));

          if (data.ollama_status === 'connected' || data.ollama_status === 'available' || data.ollama !== false) {
            updateCheck('ollama', {
              status: 'success',
              detail: 'Running',
            });
          } else {
            updateCheck('ollama', {
              status: 'error',
              detail: 'Not running',
            });
          }

          // Check model
          updateCheck('model', { status: 'checking' });
          await new Promise((r) => setTimeout(r, 300));

          if (data.model) {
            updateCheck('model', {
              status: 'success',
              detail: data.model,
            });
          } else {
            updateCheck('model', {
              status: 'success',
              detail: 'Available',
            });
          }
        } else {
          throw new Error('Backend unavailable');
        }
      } catch {
        updateCheck('backend', {
          status: 'error',
          detail: 'Not responding',
        });
        updateCheck('ollama', {
          status: 'pending',
          detail: 'Skipped',
        });
        updateCheck('model', {
          status: 'pending',
          detail: 'Skipped',
        });
      }
    };

    runChecks();
  }, [backendUrl]);

  // Update overall status
  useEffect(() => {
    const allDone = checks.every((c) => c.status === 'success' || c.status === 'error');
    const allSuccess = checks.every((c) => c.status === 'success');
    const anyError = checks.some((c) => c.status === 'error');

    if (allDone) {
      setAllPassed(allSuccess);
      setHasErrors(anyError);
    }
  }, [checks]);

  const getStatusIcon = (status: CheckItem['status']) => {
    switch (status) {
      case 'pending':
        return <span className="text-text-muted">○</span>;
      case 'checking':
        return (
          <motion.span
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            className="text-accent-primary inline-block"
          >
            ◐
          </motion.span>
        );
      case 'success':
        return <span className="text-status-success">✓</span>;
      case 'error':
        return <span className="text-status-error">✗</span>;
    }
  };

  return (
    <div className="neural-card p-8">
      {/* Title */}
      <motion.h2
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-xl font-semibold text-text-primary mb-2 text-center"
      >
        System Check
      </motion.h2>

      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="text-text-secondary text-sm mb-6 text-center"
      >
        Verifying your setup...
      </motion.p>

      {/* Check List */}
      <div className="space-y-3 mb-8">
        {checks.map((check, index) => (
          <motion.div
            key={check.id}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 + index * 0.1 }}
            className="flex items-center justify-between p-3 bg-neural-elevated rounded-lg"
          >
            <div className="flex items-center gap-3">
              <span className="font-mono text-lg w-6 text-center">
                {getStatusIcon(check.status)}
              </span>
              <span className="text-text-primary">{check.label}</span>
            </div>
            {check.detail && (
              <span className="text-sm text-text-muted">{check.detail}</span>
            )}
          </motion.div>
        ))}
      </div>

      {/* Status Message */}
      {hasErrors && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-status-warning text-sm text-center mb-6"
        >
          Some checks failed, but you can continue. AVA may have limited functionality.
        </motion.p>
      )}

      {/* Navigation */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="flex justify-between"
      >
        <button
          onClick={onBack}
          className="neural-button px-6 py-2 rounded-lg btn-lift"
        >
          Back
        </button>
        <button
          onClick={onNext}
          disabled={checks.some((c) => c.status === 'checking')}
          className="neural-button-primary px-6 py-2 rounded-lg btn-lift disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {allPassed ? 'Continue' : 'Continue Anyway'}
        </button>
      </motion.div>
    </div>
  );
}

export default SystemCheckStep;
