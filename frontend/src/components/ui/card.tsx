"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'active';
  status?: 'success' | 'warning' | 'error' | null;
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'default', status = null, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          "rounded-2xl border border-white/[0.04] bg-bg-surface text-slate-100 transition-all duration-300 relative overflow-hidden",
          variant === 'active' && "border-accent-primary/40 shadow-[0_0_20px_rgba(168,85,247,0.1)]",
          status === 'success' && "border-l-4 border-l-status-success",
          status === 'warning' && "border-l-4 border-l-status-warning",
          status === 'error' && "border-l-4 border-l-status-error",
          className
        )}
        {...props}
      />
    );
  }
);
Card.displayName = "Card";

const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex flex-col gap-1.5 p-6", className)} {...props} />
  )
);
CardHeader.displayName = "CardHeader";

const CardTitle = React.forwardRef<HTMLHeadingElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn("text-heading-sm font-semibold tracking-tight text-slate-100", className)}
      {...props}
    />
  )
);
CardTitle.displayName = "CardTitle";

const CardDescription = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => (
    <p ref={ref} className={cn("text-label-md text-slate-400", className)} {...props} />
  )
);
CardDescription.displayName = "CardDescription";

const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
  )
);
CardContent.displayName = "CardContent";

const CardFooter = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex items-center p-6 pt-0", className)} {...props} />
  )
);
CardFooter.displayName = "CardFooter";

// Keep GlassCard matching base Card style but with backdrop-blur
const GlassCard = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'default', status = null, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "rounded-2xl border border-white/[0.04] bg-bg-surface/85 backdrop-blur-xl transition-all duration-300 relative overflow-hidden",
        variant === 'active' && "border-accent-primary/40 shadow-[0_0_20px_rgba(168,85,247,0.1)]",
        status === 'success' && "border-l-4 border-l-status-success",
        status === 'warning' && "border-l-4 border-l-status-warning",
        status === 'error' && "border-l-4 border-l-status-error",
        className
      )}
      {...props}
    />
  )
);
GlassCard.displayName = "GlassCard";

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent, GlassCard };
