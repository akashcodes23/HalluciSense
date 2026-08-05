'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Search, Cpu, Database, CheckCircle2, ShieldAlert, Sparkles } from 'lucide-react';

const PROGRESS_STEPS = [
  { label: 'Extracting Claims...', icon: Sparkles },
  { label: 'Retrieving Evidence...', icon: Search },
  { label: 'Ranking Evidence...', icon: Database },
  { label: 'Running NLI & Consensus...', icon: Cpu },
  { label: 'Calculating H-Score...', icon: CheckCircle2 },
];

export function VerificationProgress() {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev < PROGRESS_STEPS.length - 1 ? prev + 1 : prev));
    }, 1200);
    return () => clearInterval(interval);
  }, []);

  const StepIcon = PROGRESS_STEPS[currentStep].icon;
  const progressPercent = ((currentStep + 1) / PROGRESS_STEPS.length) * 100;

  return (
    <div className="bg-amber-500/[0.06] border border-amber-500/20 rounded-2xl p-4 my-3 space-y-3">
      {/* Step Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <motion.div
            key={currentStep}
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="p-1.5 rounded-lg bg-amber-500/20 text-amber-400"
          >
            <StepIcon className="w-4 h-4 animate-spin-slow" />
          </motion.div>
          <div>
            <span className="text-xs font-semibold text-amber-300">
              {PROGRESS_STEPS[currentStep].label}
            </span>
            <span className="text-[10px] text-amber-400/60 block">
              Step {currentStep + 1} of {PROGRESS_STEPS.length}
            </span>
          </div>
        </div>
        <span className="text-xs font-mono font-bold text-amber-400">
          {Math.round(progressPercent)}%
        </span>
      </div>

      {/* Progress Bar */}
      <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
        <motion.div
          className="h-full bg-gradient-to-r from-amber-500 to-yellow-400 rounded-full"
          initial={{ width: '0%' }}
          animate={{ width: `${progressPercent}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
}
