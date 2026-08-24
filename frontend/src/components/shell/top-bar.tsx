"use client";

import React from "react";
import { usePathname } from "next/navigation";
import {
  Search,
  ChevronDown,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useHealth } from "@/hooks/use-analysis";

interface TopBarProps {
  onCommandPaletteOpen: () => void;
}

export function TopBar({ onCommandPaletteOpen }: TopBarProps) {
  const { data: health } = useHealth();
  const pathname = usePathname();
  const isHealthy = health?.status === "ok" || health?.status === "healthy";

  const pageTitle = getPageTitle(pathname);

  return (
    <header
      className="h-[var(--topbar-height)] shrink-0 flex items-center justify-between px-4 border-b border-[var(--border)] bg-[var(--bg-surface)]"
      role="banner"
    >
      {/* Left: Page context */}
      <div className="flex items-center gap-3 min-w-0">
        <span className="text-label-md text-[var(--text-secondary)] truncate hidden sm:block">
          {pageTitle}
        </span>
      </div>

      {/* Center: Command palette trigger */}
      <button
        onClick={onCommandPaletteOpen}
        className={cn(
          "flex items-center gap-2 px-3 py-1.5 rounded-[var(--radius)]",
          "bg-[var(--surface)] border border-[var(--border)]",
          "text-[var(--text-muted)] text-sm",
          "hover:border-[var(--border-hover)] hover:text-[var(--text-secondary)]",
          "transition-all duration-150 cursor-pointer",
          "max-w-[320px] w-full sm:w-auto"
        )}
        aria-label="Open command palette (Ctrl+K or Cmd+K)"
      >
        <Search className="w-3.5 h-3.5 shrink-0" />
        <span className="hidden sm:inline truncate">Search or run a command…</span>
        <kbd className="hidden md:inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-[var(--radius-xs)] bg-[var(--surface-hover)] text-[10px] font-mono text-[var(--text-dim)] ml-auto">
          ⌘K
        </kbd>
      </button>

      {/* Right: Status & Profile */}
      <div className="flex items-center gap-3">
        {/* System Status */}
        <div
          className={cn(
            "hidden md:flex items-center gap-2 px-2.5 py-1 rounded-[var(--radius-sm)]",
            "text-[11px] font-medium font-mono",
            isHealthy
              ? "text-[var(--verified)] bg-[var(--verified-soft)]"
              : "text-[var(--hallucination)] bg-[var(--hallucination-soft)]"
          )}
          role="status"
          aria-label={isHealthy ? "All systems operational" : "System offline"}
        >
          <span className={cn(
            "w-1.5 h-1.5 rounded-full",
            isHealthy ? "bg-[var(--verified)] animate-pulse-dot" : "bg-[var(--hallucination)]"
          )} />
          {isHealthy ? "Operational" : "Offline"}
        </div>

        {/* Profile placeholder */}
        <button
          className={cn(
            "flex items-center gap-1.5 px-2 py-1 rounded-[var(--radius)]",
            "hover:bg-[var(--surface-hover)] transition-colors cursor-pointer"
          )}
          aria-label="User menu"
        >
          <div className="w-6 h-6 rounded-full bg-[var(--primary-soft)] border border-[var(--primary)] flex items-center justify-center">
            <span className="text-[10px] font-bold text-[var(--primary)]">H</span>
          </div>
          <ChevronDown className="w-3 h-3 text-[var(--text-muted)] hidden sm:block" />
        </button>
      </div>
    </header>
  );
}

function getPageTitle(pathname: string): string {
  if (pathname.startsWith("/overview")) return "Overview";
  if (pathname.startsWith("/verify")) return "Verify";
  if (pathname.startsWith("/evaluate")) return "Evaluate";
  if (pathname.startsWith("/errors")) return "Error Feed";
  if (pathname.startsWith("/traces")) return "Traces";
  if (pathname.startsWith("/scientific")) return "Scientific Lab";
  if (pathname.startsWith("/settings")) return "Settings";
  if (pathname.startsWith("/chat")) return "Chat";
  if (pathname.startsWith("/dashboard")) return "Dashboard";
  if (pathname.startsWith("/admin")) return "Admin";
  return "HalluciSense";
}
