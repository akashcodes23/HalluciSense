import React from "react";
import { cn } from "@/lib/utils";
import { CheckCircle2, AlertTriangle, XCircle, MinusCircle, Clock, Loader2 } from "lucide-react";

type StatusType = "verified" | "hallucination" | "warning" | "failed" | "pending" | "processing" | "default";

const statusConfig: Record<StatusType, {
  icon: React.ReactNode;
  label: string;
  className: string;
}> = {
  verified: {
    icon: <CheckCircle2 className="w-3.5 h-3.5" />,
    label: "✓ Verified",
    className: "bg-[var(--verified-soft)] text-[var(--verified)] border-[var(--verified-border)]",
  },
  hallucination: {
    icon: <XCircle className="w-3.5 h-3.5" />,
    label: "✕ Hallucination",
    className: "bg-[var(--hallucination-soft)] text-[var(--hallucination)] border-[var(--hallucination-border)]",
  },
  warning: {
    icon: <AlertTriangle className="w-3.5 h-3.5" />,
    label: "⚠ Warning",
    className: "bg-[var(--warning-soft)] text-[var(--warning)] border-[var(--warning-border)]",
  },
  failed: {
    icon: <MinusCircle className="w-3.5 h-3.5" />,
    label: "— Failed",
    className: "bg-[var(--surface)] text-[var(--text-muted)] border-[var(--border)]",
  },
  pending: {
    icon: <Clock className="w-3.5 h-3.5" />,
    label: "Pending",
    className: "bg-[var(--surface)] text-[var(--text-muted)] border-[var(--border)]",
  },
  processing: {
    icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
    label: "Processing",
    className: "bg-[var(--ai-soft)] text-[var(--ai)] border-[var(--ai-border)]",
  },
  default: {
    icon: null,
    label: "Unknown",
    className: "bg-[var(--surface)] text-[var(--text-muted)] border-[var(--border)]",
  },
};

interface StatusBadgeProps {
  status: string;
  size?: "sm" | "default" | "lg";
  showIcon?: boolean;
  className?: string;
}

export function StatusBadge({ status, size = "default", showIcon = true, className }: StatusBadgeProps) {
  const normalized = normalizeStatus(status);
  const config = statusConfig[normalized];

  const sizeClasses = {
    sm: "px-1.5 py-px text-[10px] gap-1",
    default: "px-2 py-0.5 text-[11px] gap-1",
    lg: "px-2.5 py-1 text-xs gap-1.5",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center font-semibold rounded-[var(--radius-sm)] border whitespace-nowrap",
        config.className,
        sizeClasses[size],
        className
      )}
      role="status"
      aria-label={config.label}
    >
      {showIcon && config.icon}
      <span>{config.label}</span>
    </span>
  );
}

function normalizeStatus(status: string): StatusType {
  const s = status.toUpperCase().replace(/[_-]/g, "");
  if (s === "VERIFIED" || s === "LOWRISK" || s === "PASSED" || s === "SUCCESS") return "verified";
  if (s === "LIKELYHALLUCINATED" || s === "HALLUCINATED" || s === "CRITICAL") return "hallucination";
  if (s === "NEEDSVERIFICATION" || s === "MODERATERISK" || s === "WARNING" || s === "UNCERTAIN") return "warning";
  if (s === "FAILED" || s === "ERROR") return "failed";
  if (s === "PENDING") return "pending";
  if (s === "PROCESSING" || s === "RUNNING") return "processing";
  return "default";
}
