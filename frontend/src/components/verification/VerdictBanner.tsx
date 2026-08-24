"use client";

import React from "react";
import { motion } from "framer-motion";
import { ShieldCheck, ShieldX, AlertTriangle, MinusCircle, GitBranch, Clock } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatLatency } from "@/lib/format";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Link from "next/link";

interface VerdictBannerProps {
  riskLevel: string;
  hScore: number | null | undefined;
  rootCause?: string | null;
  traceId?: string | null;
  latencyMs?: number | null;
  totalClaims?: number;
  flaggedClaims?: number;
  onCorrect?: () => void;
  correctionAvailable?: boolean;
}

const verdictConfig: Record<string, {
  icon: React.ReactNode;
  label: string;
  sublabel: string;
  className: string;
  badgeVariant: "verified" | "hallucination" | "warning" | "outline";
}> = {
  VERIFIED: {
    icon: <ShieldCheck className="w-6 h-6" />,
    label: "✓ VERIFIED",
    sublabel: "All claims verified against evidence",
    className: "verdict-verified",
    badgeVariant: "verified",
  },
  LIKELY_HALLUCINATED: {
    icon: <ShieldX className="w-6 h-6" />,
    label: "✕ HALLUCINATION DETECTED",
    sublabel: "One or more claims contradict available evidence",
    className: "verdict-hallucination",
    badgeVariant: "hallucination",
  },
  NEEDS_VERIFICATION: {
    icon: <AlertTriangle className="w-6 h-6" />,
    label: "⚠ NEEDS REVIEW",
    sublabel: "Insufficient evidence to confirm or refute",
    className: "verdict-warning",
    badgeVariant: "warning",
  },
  MODERATE_RISK: {
    icon: <AlertTriangle className="w-6 h-6" />,
    label: "⚠ MODERATE RISK",
    sublabel: "Some claims show inconsistencies",
    className: "verdict-warning",
    badgeVariant: "warning",
  },
  FAILED: {
    icon: <MinusCircle className="w-6 h-6" />,
    label: "— VERIFICATION FAILED",
    sublabel: "The verification engine could not complete this request",
    className: "verdict-failed",
    badgeVariant: "outline",
  },
};

export function VerdictBanner({
  riskLevel,
  hScore,
  rootCause,
  traceId,
  latencyMs,
  totalClaims,
  flaggedClaims,
  onCorrect,
  correctionAvailable = false,
}: VerdictBannerProps) {
  const config = verdictConfig[riskLevel] || verdictConfig.FAILED;
  const scoreAvailable = hScore !== null && hScore !== undefined && !isNaN(hScore);
  const scorePercent = scoreAvailable ? Math.round(hScore * 100) : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn("rounded-[var(--radius-lg)] p-5", config.className)}
    >
      <div className="flex flex-col md:flex-row md:items-center gap-4">
        {/* Left: Verdict */}
        <div className="flex items-center gap-3 flex-1 min-w-0">
          {config.icon}
          <div className="min-w-0">
            <h3 className="text-lg font-bold tracking-tight">{config.label}</h3>
            <p className="text-sm opacity-80 mt-0.5">{config.sublabel}</p>
          </div>
        </div>

        {/* Right: Score & Meta */}
        <div className="flex items-center gap-4 shrink-0">
          {/* H-Score */}
          <div className="text-center">
            <p className="text-[10px] font-semibold uppercase tracking-wider opacity-60 mb-1">H-Score</p>
            <p className="text-3xl font-bold font-mono tracking-tight">
              {scoreAvailable ? `${scorePercent}%` : "—"}
            </p>
          </div>

          {/* Claims Summary */}
          {totalClaims !== undefined && (
            <div className="text-center border-l border-current/20 pl-4">
              <p className="text-[10px] font-semibold uppercase tracking-wider opacity-60 mb-1">Claims</p>
              <p className="text-lg font-bold font-mono">
                {flaggedClaims ?? 0}/{totalClaims}
                <span className="text-xs font-normal opacity-60 ml-1">flagged</span>
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Bottom: Meta & Actions */}
      <div className="flex flex-wrap items-center gap-2 mt-4 pt-3 border-t border-current/10">
        {rootCause && rootCause !== "VERIFIED" && rootCause !== "NONE" && (
          <Badge variant={config.badgeVariant} size="sm">
            {rootCause.replace(/_/g, " ")}
          </Badge>
        )}
        {latencyMs != null && (
          <Badge variant="outline" size="sm">
            <Clock className="w-3 h-3" /> {formatLatency(latencyMs)}
          </Badge>
        )}
        {traceId && (
          <Link href={`/traces?id=${traceId}`}>
            <Badge variant="outline" size="sm" className="cursor-pointer hover:border-[var(--border-hover)]">
              <GitBranch className="w-3 h-3" /> View Trace
            </Badge>
          </Link>
        )}
        {correctionAvailable && onCorrect && riskLevel === "LIKELY_HALLUCINATED" && (
          <Button
            variant="default"
            size="sm"
            onClick={onCorrect}
            className="ml-auto"
          >
            Correct with HalluciSense
          </Button>
        )}
      </div>
    </motion.div>
  );
}
