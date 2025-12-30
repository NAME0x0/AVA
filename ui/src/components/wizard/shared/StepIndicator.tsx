'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface StepIndicatorProps {
  steps: number;
  currentStep: number;
  className?: string;
}

export function StepIndicator({ steps, currentStep, className }: StepIndicatorProps) {
  return (
    <div className={cn('flex items-center justify-center gap-2', className)}>
      {Array.from({ length: steps }).map((_, index) => (
        <motion.div
          key={index}
          initial={false}
          animate={{
            scale: index === currentStep ? 1 : 0.8,
            opacity: index <= currentStep ? 1 : 0.3,
          }}
          className={cn(
            'transition-colors duration-200',
            index === currentStep
              ? 'w-8 h-2 bg-accent-primary rounded-full'
              : index < currentStep
              ? 'w-2 h-2 bg-accent-primary rounded-full'
              : 'w-2 h-2 bg-neural-hover rounded-full'
          )}
        />
      ))}
    </div>
  );
}

export default StepIndicator;
