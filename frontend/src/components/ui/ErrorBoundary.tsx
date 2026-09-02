"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";
import { Button } from "./button";

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
  fallbackMessage?: string;
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("[TraceErrorBoundary] Caught render error:", error, errorInfo);
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: null });
    this.props.onReset?.();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="p-6 rounded-[var(--radius-lg)] bg-[var(--surface)] border border-[var(--border)] space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-[var(--radius)] bg-amber-500/10 border border-amber-500/20 flex items-center justify-center shrink-0">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                {this.props.fallbackTitle || "Trace unavailable"}
              </h3>
              <p className="text-xs text-[var(--text-muted)] mt-0.5">
                {this.props.fallbackMessage || "Unable to render trace data. The verification result remains available."}
              </p>
            </div>
          </div>
          {this.state.error && (
            <div className="p-3 rounded-[var(--radius-sm)] bg-black/40 border border-white/[0.04] text-[11px] font-mono text-[var(--text-dim)] overflow-x-auto">
              {this.state.error.message}
            </div>
          )}
          <div className="pt-1">
            <Button variant="outline" size="sm" onClick={this.handleReset} className="gap-1.5 text-xs">
              <RotateCcw className="w-3.5 h-3.5" />
              Retry Trace Rendering
            </Button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
