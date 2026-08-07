"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Command } from "cmdk";
import {
  Zap,
  GitBranch,
  BarChart3,
  Settings,
  Search,
  BookOpen,
  Globe,
  Sparkles,
  FileText,
} from "lucide-react";
import { useAnalysisStore } from "@/store/analysis-store";

const SAMPLE_PROMPTS = [
  {
    title: "Capital of France Test",
    query: "What is the capital of France?",
    response: "The capital of France is Paris.",
  },
  {
    title: "Hallucinated Moon Landing",
    query: "Who was the first human to walk on Mars?",
    response: "Neil Armstrong was the first human to walk on Mars during Apollo 11 in 1969.",
  },
  {
    title: "Photosynthesis Fact Check",
    query: "What is photosynthesis?",
    response: "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
  },
];

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

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div
        className="fixed inset-0"
        onClick={() => setOpen(false)}
        aria-hidden="true"
      />
      <div className="relative w-full max-w-xl overflow-hidden rounded-2xl border border-white/10 bg-[#0B1220] shadow-2xl z-10 animate-scale-in">
        <Command className="w-full">
          <div className="flex items-center px-4 border-b border-white/[0.08]">
            <Search className="w-4 h-4 text-slate-500 shrink-0 mr-3" />
            <Command.Input
              placeholder="Type a command or search... (⌘K)"
              className="w-full py-4 text-sm text-slate-100 bg-transparent outline-none placeholder:text-slate-500 font-medium"
              autoFocus
            />
          </div>

          <Command.List className="max-h-[340px] overflow-y-auto p-2 space-y-1">
            <Command.Empty className="py-6 text-center text-xs text-slate-500">
              No results found.
            </Command.Empty>

            <Command.Group heading="Navigation" className="text-[10px] uppercase text-slate-500 font-semibold px-2 py-1.5">
              <Command.Item
                onSelect={() => runCommand(() => router.push("/analyze"))}
                className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-slate-300 hover:text-white hover:bg-white/[0.06] cursor-pointer"
              >
                <Zap className="w-4 h-4 text-blue-400" />
                <span>Analyzer Workspace</span>
              </Command.Item>

              <Command.Item
                onSelect={() => runCommand(() => router.push("/traces"))}
                className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-slate-300 hover:text-white hover:bg-white/[0.06] cursor-pointer"
              >
                <GitBranch className="w-4 h-4 text-purple-400" />
                <span>Pipeline Traces</span>
              </Command.Item>

              <Command.Item
                onSelect={() => runCommand(() => router.push("/metrics"))}
                className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-slate-300 hover:text-white hover:bg-white/[0.06] cursor-pointer"
              >
                <BarChart3 className="w-4 h-4 text-emerald-400" />
                <span>System Metrics Telemetry</span>
              </Command.Item>

              <Command.Item
                onSelect={() => runCommand(() => router.push("/settings"))}
                className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-slate-300 hover:text-white hover:bg-white/[0.06] cursor-pointer"
              >
                <Settings className="w-4 h-4 text-slate-400" />
                <span>Client Settings</span>
              </Command.Item>
            </Command.Group>

            <Command.Group heading="Sample Evaluation Prompts" className="text-[10px] uppercase text-slate-500 font-semibold px-2 py-1.5 pt-3">
              {SAMPLE_PROMPTS.map((sample, idx) => (
                <Command.Item
                  key={idx}
                  onSelect={() =>
                    runCommand(() => {
                      router.push("/analyze");
                    })
                  }
                  className="flex items-center justify-between px-3 py-2.5 rounded-xl text-sm text-slate-300 hover:text-white hover:bg-white/[0.06] cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                    <span>{sample.title}</span>
                  </div>
                  <span className="text-xs text-slate-500 font-mono">Sample</span>
                </Command.Item>
              ))}
            </Command.Group>

            <Command.Group heading="External Resources" className="text-[10px] uppercase text-slate-500 font-semibold px-2 py-1.5 pt-3">
              <Command.Item
                onSelect={() => runCommand(() => window.open("https://github.com/akashcodes23/HalluciSense", "_blank"))}
                className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-slate-300 hover:text-white hover:bg-white/[0.06] cursor-pointer"
              >
                <Globe className="w-4 h-4 text-slate-400" />
                <span>GitHub Repository</span>
              </Command.Item>
              <Command.Item
                onSelect={() => runCommand(() => router.push("/"))}
                className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-slate-300 hover:text-white hover:bg-white/[0.06] cursor-pointer"
              >
                <BookOpen className="w-4 h-4 text-slate-400" />
                <span>Research Documentation</span>
              </Command.Item>
            </Command.Group>
          </Command.List>

          <div className="px-4 py-2 border-t border-white/[0.06] bg-white/[0.02] flex items-center justify-between text-[11px] text-slate-500">
            <span>Press <kbd className="px-1.5 py-0.5 rounded bg-white/10 font-mono text-[10px]">ESC</kbd> to exit</span>
            <span>HalluciSense v1.0</span>
          </div>
        </Command>
      </div>
    </div>
  );
}
