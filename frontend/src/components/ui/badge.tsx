import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-[var(--radius-sm)] font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "bg-[var(--surface)] text-[var(--text-secondary)] border border-[var(--border)]",
        verified: "bg-[var(--verified-soft)] text-[var(--verified)] border border-[var(--verified-border)]",
        hallucination: "bg-[var(--hallucination-soft)] text-[var(--hallucination)] border border-[var(--hallucination-border)]",
        warning: "bg-[var(--warning-soft)] text-[var(--warning)] border border-[var(--warning-border)]",
        evidence: "bg-[var(--evidence-soft)] text-[var(--evidence)] border border-[var(--evidence-border)]",
        ai: "bg-[var(--ai-soft)] text-[var(--ai)] border border-[var(--ai-border)]",
        primary: "bg-[var(--primary-soft)] text-[var(--primary)] border border-[var(--ai-border)]",
        success: "bg-[var(--verified-soft)] text-[var(--verified)] border border-[var(--verified-border)]",
        destructive: "bg-[var(--hallucination-soft)] text-[var(--hallucination)] border border-[var(--hallucination-border)]",
        danger: "bg-[var(--hallucination-soft)] text-[var(--hallucination)] border border-[var(--hallucination-border)]",
        info: "bg-[var(--evidence-soft)] text-[var(--evidence)] border border-[var(--evidence-border)]",
        outline: "text-[var(--text-secondary)] border border-[var(--border)]",
      },
      size: {
        default: "px-2 py-0.5 text-[11px]",
        sm: "px-1.5 py-px text-[10px]",
        lg: "px-2.5 py-1 text-xs",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, size, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant, size }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
