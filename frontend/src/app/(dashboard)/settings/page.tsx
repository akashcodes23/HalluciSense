"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Settings, Server, Moon, Sun, Monitor, Cpu, Check, RotateCcw } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { GlassCard } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MODEL_OPTIONS } from "@/lib/constants";
import { toast } from "sonner";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const [apiUrl, setApiUrl] = useState(
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  );
  const [defaultModel, setDefaultModel] = useState("GPT-4");
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    toast.success("Settings saved successfully");
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    setApiUrl("http://localhost:8000");
    setDefaultModel("GPT-4");
    setTheme("dark");
    toast.info("Settings reset to default");
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-slate-700 to-slate-900 border border-white/10 shadow-lg">
              <Settings className="w-5 h-5 text-slate-200" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">Settings</h1>
              <p className="text-sm text-slate-500">Configure client environment & preferences</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={handleReset}>
              <RotateCcw className="w-4 h-4" />
              Reset
            </Button>
            <Button onClick={handleSave} size="sm">
              {saved ? <Check className="w-4 h-4" /> : null}
              {saved ? "Saved" : "Save Changes"}
            </Button>
          </div>
        </motion.div>

        <div className="space-y-6">
          {/* API Configuration */}
          <GlassCard className="p-6 space-y-4">
            <div className="flex items-center gap-2 text-slate-200 font-semibold">
              <Server className="w-4 h-4 text-blue-400" />
              <span>Backend API Connection</span>
            </div>
            <div className="space-y-2">
              <label className="text-xs text-slate-400 font-medium">REST API Base Endpoint</label>
              <Input
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                placeholder="http://localhost:8000"
              />
              <p className="text-xs text-slate-500">
                Target production endpoint for `/api/v1/analyze`, `/api/v1/explain`, and `/api/v1/metrics`.
              </p>
            </div>
          </GlassCard>

          {/* Model Preferences */}
          <GlassCard className="p-6 space-y-4">
            <div className="flex items-center gap-2 text-slate-200 font-semibold">
              <Cpu className="w-4 h-4 text-purple-400" />
              <span>Model & Pipeline Defaults</span>
            </div>
            <div className="space-y-2">
              <label className="text-xs text-slate-400 font-medium">Default Target Architecture</label>
              <select
                value={defaultModel}
                onChange={(e) => setDefaultModel(e.target.value)}
                className="w-full h-10 px-3 rounded-xl border border-white/[0.08] bg-white/[0.03] text-sm text-slate-300 focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 cursor-pointer"
              >
                {MODEL_OPTIONS.map((m) => (
                  <option key={m.value} value={m.value} className="bg-[#111827]">
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
          </GlassCard>

          {/* Theme Support */}
          <GlassCard className="p-6 space-y-4">
            <div className="flex items-center gap-2 text-slate-200 font-semibold">
              <Sun className="w-4 h-4 text-amber-400" />
              <span>Appearance & Theme</span>
            </div>
            <div className="grid grid-cols-3 gap-3">
              <button
                onClick={() => setTheme("dark")}
                className={`flex flex-col items-center justify-center gap-2 p-4 rounded-xl border transition-all ${
                  theme === "dark"
                    ? "border-blue-500 bg-blue-500/10 text-white"
                    : "border-white/10 bg-white/[0.02] text-slate-400 hover:text-slate-200"
                }`}
              >
                <Moon className="w-5 h-5" />
                <span className="text-xs font-medium">Dark Mode</span>
              </button>
              <button
                onClick={() => setTheme("light")}
                className={`flex flex-col items-center justify-center gap-2 p-4 rounded-xl border transition-all ${
                  theme === "light"
                    ? "border-blue-500 bg-blue-500/10 text-white"
                    : "border-white/10 bg-white/[0.02] text-slate-400 hover:text-slate-200"
                }`}
              >
                <Sun className="w-5 h-5" />
                <span className="text-xs font-medium">Light Mode</span>
              </button>
              <button
                onClick={() => setTheme("system")}
                className={`flex flex-col items-center justify-center gap-2 p-4 rounded-xl border transition-all ${
                  theme === "system"
                    ? "border-blue-500 bg-blue-500/10 text-white"
                    : "border-white/10 bg-white/[0.02] text-slate-400 hover:text-slate-200"
                }`}
              >
                <Monitor className="w-5 h-5" />
                <span className="text-xs font-medium">System Default</span>
              </button>
            </div>
          </GlassCard>

          {/* System Information */}
          <GlassCard className="p-6 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-slate-300">Frontend Environment</span>
              <Badge variant="verified">v1.0.0 Release Candidate</Badge>
            </div>
            <div className="text-xs text-slate-500 space-y-1 font-mono">
              <p>Framework: Next.js 16 (App Router) + React 19</p>
              <p>Styling: TailwindCSS + Glassmorphic Design System</p>
              <p>State: React Query (TanStack) + Zustand</p>
            </div>
          </GlassCard>
        </div>
      </div>
    </div>
  );
}
