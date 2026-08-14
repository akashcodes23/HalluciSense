"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Search,
  Scale,
  Activity,
  GitBranch,
  Layers,
  ShieldCheck,
  CheckCircle2,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";

const STAGES = [
  { id: "retrieval", label: "Retrieving Evidence", sublabel: "BM25 + Dense Hybrid Search", icon: Search, duration: 1200 },
  { id: "nli", label: "NLI Verification", sublabel: "Cross-encoder entailment analysis", icon: Scale, duration: 800 },
  { id: "confidence", label: "Confidence Analysis", sublabel: "Entropy & uncertainty estimation", icon: Activity, duration: 600 },
  { id: "consistency", label: "Consistency Check", sublabel: "Paraphrase self-consistency", icon: GitBranch, duration: 500 },
  { id: "fusion", label: "Adaptive Fusion", sublabel: "Weighted pillar aggregation", icon: Layers, duration: 400 },
  { id: "decision", label: "Risk Assessment", sublabel: "Platt-calibrated classification", icon: ShieldCheck, duration: 300 },
];

interface PipelineAnimationProps {
  isActive: boolean;
}

export function PipelineAnimation({ isActive }: PipelineAnimationProps) {
  const [activeStage, setActiveStage] = useState(0);
  const [completedStages, setCompletedStages] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (!isActive) {
      setActiveStage(0);
      setCompletedStages(new Set());
      return;
    }

    let currentStage = 0;
    const timers: NodeJS.Timeout[] = [];

    const advanceStage = () => {
      if (currentStage < STAGES.length) {
        setActiveStage(currentStage);
        const timer = setTimeout(() => {
          setCompletedStages((prev) => new Set([...prev, currentStage]));
          currentStage++;
          advanceStage();
        }, STAGES[currentStage].duration);
        timers.push(timer);
      }
    };

    advanceStage();

    return () => timers.forEach(clearTimeout);
  }, [isActive]);

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-6 space-y-3">
      <div className="flex items-center gap-2 mb-4">
        <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
        <span className="text-sm font-medium text-slate-300">Processing Pipeline</span>
      </div>

      <div className="space-y-2">
        {STAGES.map((stage, index) => {
          const Icon = stage.icon;
          const isCompleted = completedStages.has(index);
          const isActive = activeStage === index && !isCompleted;
          const isPending = index > activeStage && !isCompleted;

          return (
            <motion.div
              key={stage.id}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.08, duration: 0.3 }}
              className={cn(
                "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300",
                isCompleted && "bg-emerald-500/[0.06] border border-emerald-500/[0.1]",
                isActive && "bg-blue-500/[0.08] border border-blue-500/[0.15]",
                isPending && "opacity-40"
              )}
            >
              {/* Status Icon */}
              <div className={cn(
                "flex items-center justify-center w-8 h-8 rounded-lg shrink-0",
                isCompleted && "bg-emerald-500/20 text-emerald-400",
                isActive && "bg-blue-500/20 text-blue-400",
                isPending && "bg-white/[0.04] text-slate-400"
              )}>
                {isCompleted ? (
                  <CheckCircle2 className="w-4 h-4" />
                ) : isActive ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Icon className="w-4 h-4" />
                )}
              </div>

              {/* Label */}
              <div className="flex-1 min-w-0">
                <p className={cn(
                  "text-sm font-medium",
                  isCompleted && "text-emerald-400",
                  isActive && "text-blue-400",
                  isPending && "text-slate-300"
                )}>
                  {stage.label}
                </p>
                <p className={cn("text-xs truncate", isPending ? "text-slate-400/80" : "text-slate-500")}>
                  {stage.sublabel}
                </p>
              </div>

              {/* Progress bar for active stage */}
              {isActive && (
                <div className="w-16 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-blue-500"
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ duration: stage.duration / 1000, ease: "linear" }}
                  />
                </div>
              )}

              {isCompleted && (
                <span className="text-[10px] text-emerald-500 font-mono">Done</span>
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
