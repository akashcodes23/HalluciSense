import React from "react";
import { cn } from "@/lib/utils";
import { LucideIcon, Inbox, GitBranch, BarChart3, ShieldCheck, AlertTriangle } from "lucide-react";
import { Button } from "./button";

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  secondaryAction?: {
    label: string;
    onClick: () => void;
  };
  variant?: "default" | "compact";
  className?: string;
}

export function EmptyState({
  icon: Icon = Inbox,
  title,
  description,
  action,
  secondaryAction,
  variant = "default",
  className,
}: EmptyStateProps) {
  if (variant === "compact") {
    return (
      <div className={cn("flex flex-col items-center justify-center py-8 px-4 text-center", className)}>
        <Icon className="w-8 h-8 text-[var(--text-dim)] mb-3" />
        <p className="text-sm font-medium text-[var(--text-muted)] mb-1">{title}</p>
        {description && (
          <p className="text-xs text-[var(--text-dim)] max-w-xs">{description}</p>
        )}
        {action && (
          <Button variant="outline" size="sm" onClick={action.onClick} className="mt-3">
            {action.label}
          </Button>
        )}
      </div>
    );
  }

  return (
    <div className={cn("flex flex-col items-center justify-center py-16 px-6 text-center", className)}>
      <div className="w-14 h-14 rounded-[var(--radius-lg)] bg-[var(--surface)] border border-[var(--border)] flex items-center justify-center mb-4">
        <Icon className="w-6 h-6 text-[var(--text-dim)]" />
      </div>
      <h3 className="text-base font-semibold text-[var(--text-secondary)] mb-1">{title}</h3>
      {description && (
        <p className="text-sm text-[var(--text-muted)] max-w-md leading-relaxed mb-4">{description}</p>
      )}
      {(action || secondaryAction) && (
        <div className="flex items-center gap-2">
          {action && (
            <Button variant="default" size="default" onClick={action.onClick}>
              {action.label}
            </Button>
          )}
          {secondaryAction && (
            <Button variant="outline" size="default" onClick={secondaryAction.onClick}>
              {secondaryAction.label}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}

// Pre-built empty states for common scenarios
export function AwaitingTelemetry({ className }: { className?: string }) {
  return (
    <EmptyState
      icon={BarChart3}
      title="Awaiting production telemetry"
      description="No verification data recorded yet. Run your first verification to see live metrics."
      variant="compact"
      className={className}
    />
  );
}

export function NoTraces({ className, onNavigate, variant = "compact" }: { className?: string; onNavigate?: () => void; variant?: "default" | "compact" }) {
  return (
    <EmptyState
      icon={GitBranch}
      title="No traces recorded"
      description="Verification traces will appear here after running analyses."
      action={onNavigate ? { label: "Go to Verify", onClick: onNavigate } : undefined}
      variant={variant}
      className={className}
    />
  );
}

export function NoErrors({ className, variant = "compact" }: { className?: string; variant?: "default" | "compact" }) {
  return (
    <EmptyState
      icon={ShieldCheck}
      title="No errors detected"
      description="When hallucinations or verification failures occur, they'll appear in this feed."
      variant={variant}
      className={className}
    />
  );
}

export function VerificationUnavailable({ onRetry }: { onRetry?: () => void }) {
  return (
    <EmptyState
      icon={AlertTriangle}
      title="Verification unavailable"
      description="The verification engine could not complete this request. This may be due to a temporary infrastructure issue."
      action={onRetry ? { label: "Retry", onClick: onRetry } : undefined}
    />
  );
}
