"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Command } from "cmdk";
import {
  LayoutDashboard,
  ShieldCheck,
  MessageSquare,
  BarChart3,
  AlertTriangle,
  GitBranch,
  FlaskConical,
  Settings,
  Search,
  Globe,
  Sparkles,
} from "lucide-react";

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((open) => !open);
      }
    };

    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  const runCommand = (command: () => void) => {
    setOpen(false);
    command();
  };

  if (!open) return null;

  const itemClass =
    "flex items-center gap-3 px-3 py-2 rounded-[var(--radius)] text-[13px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-hover)] cursor-pointer transition-colors";

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh] p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div
        className="fixed inset-0"
        onClick={() => setOpen(false)}
        aria-hidden="true"
      />
      <div className="relative w-full max-w-lg overflow-hidden rounded-[var(--radius-xl)] border border-[var(--border-strong)] bg-[var(--bg-surface)] shadow-[var(--shadow-lg)] z-10 animate-scale-in">
        <Command className="w-full">
          <div className="flex items-center px-4 border-b border-[var(--border)]">
            <Search className="w-4 h-4 text-[var(--text-dim)] shrink-0 mr-3" />
            <Command.Input
              placeholder="Search or run a command…"
              className="w-full py-3.5 text-sm text-[var(--text-primary)] bg-transparent outline-none placeholder:text-[var(--text-dim)]"
              autoFocus
            />
            <kbd className="shrink-0 px-1.5 py-0.5 rounded-[var(--radius-xs)] bg-[var(--surface)] text-[10px] font-mono text-[var(--text-dim)]">
              ESC
            </kbd>
          </div>

          <Command.List className="max-h-[340px] overflow-y-auto p-1.5">
            <Command.Empty className="py-6 text-center text-xs text-[var(--text-muted)]">
              No results found.
            </Command.Empty>

            <Command.Group heading="Navigation" className="text-[10px] uppercase text-[var(--text-dim)] font-semibold px-2 py-1.5 tracking-wider">
              <Command.Item onSelect={() => runCommand(() => router.push("/overview"))} className={itemClass}>
                <LayoutDashboard className="w-4 h-4 text-[var(--text-muted)]" />
                <span>Overview — Command Center</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => router.push("/verify"))} className={itemClass}>
                <ShieldCheck className="w-4 h-4 text-[var(--primary)]" />
                <span>Verify — Analyze LLM response</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => router.push("/chat"))} className={itemClass}>
                <MessageSquare className="w-4 h-4 text-[var(--evidence)]" />
                <span>Chat — Conversation + Verification</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => router.push("/evaluate"))} className={itemClass}>
                <BarChart3 className="w-4 h-4 text-[var(--verified)]" />
                <span>Evaluate — Benchmarks & Metrics</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => router.push("/errors"))} className={itemClass}>
                <AlertTriangle className="w-4 h-4 text-[var(--warning)]" />
                <span>Error Feed — Hallucination log</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => router.push("/traces"))} className={itemClass}>
                <GitBranch className="w-4 h-4 text-[var(--ai)]" />
                <span>Traces — Pipeline execution</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => router.push("/scientific"))} className={itemClass}>
                <FlaskConical className="w-4 h-4 text-[var(--verified)]" />
                <span>Scientific Lab — Research results</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => router.push("/settings"))} className={itemClass}>
                <Settings className="w-4 h-4 text-[var(--text-muted)]" />
                <span>Settings</span>
              </Command.Item>
            </Command.Group>

            <Command.Group heading="Quick Actions" className="text-[10px] uppercase text-[var(--text-dim)] font-semibold px-2 py-1.5 tracking-wider pt-2">
              <Command.Item onSelect={() => runCommand(() => router.push("/verify"))} className={itemClass}>
                <Sparkles className="w-4 h-4 text-[var(--warning)]" />
                <span>Verify a claim</span>
              </Command.Item>
            </Command.Group>

            <Command.Group heading="External" className="text-[10px] uppercase text-[var(--text-dim)] font-semibold px-2 py-1.5 tracking-wider pt-2">
              <Command.Item
                onSelect={() => runCommand(() => window.open("https://github.com/akashcodes23/HalluciSense", "_blank"))}
                className={itemClass}
              >
                <Globe className="w-4 h-4 text-[var(--text-muted)]" />
                <span>GitHub Repository</span>
              </Command.Item>
            </Command.Group>
          </Command.List>

          <div className="px-4 py-2 border-t border-[var(--border)] flex items-center justify-between text-[10px] text-[var(--text-dim)]">
            <span>↑↓ navigate · ↵ select · esc close</span>
            <span className="font-mono">HalluciSense v2.0</span>
          </div>
        </Command>
      </div>
    </div>
  );
}
