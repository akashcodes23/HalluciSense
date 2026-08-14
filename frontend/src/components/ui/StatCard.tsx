"use client";

import React from "react";
import { Card } from "./card";
import { cn } from "@/lib/utils";

interface StatCardProps {
  label: string;
  value: string | number;
  caption: string;
  icon: React.ComponentType<{ className?: string }>;
  status?: "success" | "warning" | "error" | "default" | null;
  className?: string;
}

export function StatCard({
  label,
  value,
  caption,
  icon: Icon,
  status = "default",
  className,
}: StatCardProps) {
  const getStatusColor = () => {
    switch (status) {
      case "success":
        return "text-status-success";
      case "warning":
        return "text-status-warning";
      case "error":
        return "text-status-error";
      default:
        return "text-white";
    }
  };

  const getIconColor = () => {
    switch (status) {
      case "success":
        return "text-status-success/80";
      case "warning":
        return "text-status-warning/80";
      case "error":
        return "text-status-error/80";
      default:
        return "text-slate-400";
    }
  };

  return (
    <Card status={status === "default" ? null : status} className={cn("p-5 space-y-3", className)}>
      <div className="flex items-center justify-between text-label-sm text-slate-500 font-mono">
        <span>{label}</span>
        <Icon className={cn("w-4 h-4", getIconColor())} />
      </div>
      <div className={cn("text-heading-md font-bold font-mono tracking-tight", getStatusColor())}>
        {value}
      </div>
      <p className="text-label-md text-slate-500 font-sans font-normal leading-tight">{caption}</p>
    </Card>
  );
}
