"use client";

import React from "react";
import Link from "next/link";
import { Button } from "./button";
import { Card } from "./card";
import { cn } from "@/lib/utils";

interface EmptyStateProps {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  actionLabel?: string;
  actionHref?: string;
  className?: string;
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  actionLabel,
  actionHref,
  className,
}: EmptyStateProps) {
  return (
    <Card
      className={cn(
        "flex flex-col items-center justify-center text-center p-8 md:p-12 border border-white/[0.04] bg-bg-surface/50 rounded-2xl space-y-4 max-w-lg mx-auto",
        className
      )}
    >
      <Icon className="w-10 h-10 text-slate-500/80 mb-1 shrink-0" />

      <div className="space-y-1">
        <h3 className="text-heading-sm font-semibold text-white tracking-tight">{title}</h3>
        <p className="text-label-md text-slate-500 leading-normal max-w-sm">{description}</p>
      </div>

      {actionLabel && actionHref && (
        <Link href={actionHref}>
          <Button
            variant="default"
            size="sm"
            className="mt-2 bg-accent-primary hover:bg-accent-primary/90 text-white font-medium px-4 py-2 rounded-xl transition-all cursor-pointer shadow-lg shadow-accent-primary/10"
          >
            {actionLabel}
          </Button>
        </Link>
      )}
    </Card>
  );
}
