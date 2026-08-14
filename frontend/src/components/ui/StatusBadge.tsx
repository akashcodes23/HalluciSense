"use client";

import React from "react";
import { Badge } from "./badge";
import { cn } from "@/lib/utils";

interface StatusBadgeProps {
  label: string;
  status: "success" | "warning" | "error" | "default" | "info";
  className?: string;
}

export function StatusBadge({ label, status, className }: StatusBadgeProps) {
  const getVariant = () => {
    switch (status) {
      case "success":
        return "verified";
      case "warning":
        return "warning";
      case "error":
        return "danger";
      case "info":
        return "info";
      default:
        return "default";
    }
  };

  return (
    <Badge
      variant={getVariant()}
      className={cn(
        "text-[10px] font-semibold font-mono tracking-wider uppercase px-2 py-0.5 rounded-lg border",
        status === "success" && "bg-status-success/10 border-status-success/20 text-status-success",
        status === "warning" && "bg-status-warning/10 border-status-warning/20 text-status-warning",
        status === "error" && "bg-status-error/10 border-status-error/20 text-status-error",
        status === "info" && "bg-blue-500/10 border-blue-500/20 text-blue-400",
        status === "default" && "bg-white/5 border-white/10 text-slate-400",
        className
      )}
    >
      {label}
    </Badge>
  );
}
