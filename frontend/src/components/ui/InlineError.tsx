"use client";

import React, { useState } from "react";
import { AlertTriangle, ChevronDown, X } from "lucide-react";
import { Card } from "./card";
import { cn } from "@/lib/utils";

interface InlineErrorProps {
  message: string;
  details?: unknown;
  onClear?: () => void;
  className?: string;
}

export function InlineError({
  message,
  details,
  onClear,
  className,
}: InlineErrorProps) {
  const [showDetails, setShowDetails] = useState(false);

  const getDescriptiveMessage = () => {
    if (message.includes("422") || message.includes("schema") || message.includes("validation")) {
      return "The request payload did not match the expected server schema. Please check if response contains non-empty text.";
    }
    return message;
  };

  return (
    <div
      className={cn(
        "p-4 bg-[var(--hallucination-soft)] border border-[var(--hallucination-border)] rounded-[var(--radius-lg)] space-y-2 relative",
        className
      )}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-status-error font-semibold font-sans text-sm">
          <AlertTriangle className="w-4.5 h-4.5 shrink-0" />
          <span>Verification Failed</span>
        </div>
        {onClear && (
          <button
            onClick={onClear}
            className="p-1 text-slate-500 hover:text-white rounded-md transition-colors cursor-pointer"
            aria-label="Dismiss error"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      <p className="text-slate-300 text-sm leading-relaxed">{getDescriptiveMessage()}</p>

      {!!details && (
        <div className="pt-2 border-t border-white/5">
          <button
            type="button"
            onClick={() => setShowDetails(!showDetails)}
            className="text-xs text-slate-400 hover:text-white underline cursor-pointer inline-flex items-center gap-1 font-mono"
          >
            {showDetails ? "Hide technical details" : "Show technical details"}
            <ChevronDown className={cn("w-3 h-3 transition-transform", showDetails && "rotate-180")} />
          </button>
          {showDetails && (
            <pre className="mt-2 p-3 rounded-lg bg-black/40 text-xs font-mono text-status-error/80 overflow-x-auto leading-relaxed select-text">
              {JSON.stringify(details, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
