import React from "react";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface StatCardProps {
  label: string;
  value: string | number;
  caption?: string;
  icon?: LucideIcon;
  trend?: { value: number; label?: string };
  status?: "default" | "verified" | "hallucination" | "warning" | "evidence";
  className?: string;
}

const statusColors = {
  default: {
    icon: "text-[var(--text-muted)]",
    accent: "var(--text-muted)",
  },
  verified: {
    icon: "text-[var(--verified)]",
    accent: "var(--verified)",
  },
  hallucination: {
    icon: "text-[var(--hallucination)]",
    accent: "var(--hallucination)",
  },
  warning: {
    icon: "text-[var(--warning)]",
    accent: "var(--warning)",
  },
  evidence: {
    icon: "text-[var(--evidence)]",
    accent: "var(--evidence)",
  },
};

export function StatCard({
  label,
  value,
  caption,
  icon: Icon,
  trend,
  status = "default",
  className,
}: StatCardProps) {
  const colors = statusColors[status];

  return (
    <div
      className={cn(
        "rounded-[var(--radius-lg)] bg-[var(--bg-surface)] border border-[var(--border)]",
        "p-4 flex flex-col gap-2 min-w-0",
        "transition-all duration-150 hover:border-[var(--border-hover)]",
        className
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          {label}
        </span>
        {Icon && (
          <Icon className={cn("w-4 h-4 shrink-0", colors.icon)} />
        )}
      </div>

      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold text-[var(--text-primary)] tracking-tight font-[var(--font-display)]">
          {value}
        </span>
        {trend && (
          <span
            className={cn(
              "text-xs font-medium",
              trend.value > 0 ? "text-[var(--verified)]" : trend.value < 0 ? "text-[var(--hallucination)]" : "text-[var(--text-muted)]"
            )}
          >
            {trend.value > 0 ? "↑" : trend.value < 0 ? "↓" : "→"} {Math.abs(trend.value).toFixed(1)}%
            {trend.label && <span className="ml-0.5">{trend.label}</span>}
          </span>
        )}
      </div>

      {caption && (
        <p className="text-[11px] text-[var(--text-muted)] leading-relaxed truncate">
          {caption}
        </p>
      )}
    </div>
  );
}
