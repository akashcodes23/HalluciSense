"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Settings, Server, Moon, Sun, Monitor, Cpu, Check, RotateCcw, Activity, RefreshCw } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { GlassCard } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MODEL_OPTIONS } from "@/lib/constants";
import { getHealth } from "@/services/hallucisense-api";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const [apiUrl, setApiUrl] = useState(
    process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "https://hallucisense-production.up.railway.app"
  );
  const [defaultModel, setDefaultModel] = useState("GPT-4o");
  const [saved, setSaved] = useState(false);

  // Connection Test State
  const [testing, setTesting] = useState(false);
  const [connResult, setConnResult] = useState<{ success: boolean; latencyMs?: number; msg?: string } | null>(null);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    setApiUrl("https://hallucisense-production.up.railway.app");
    setDefaultModel("GPT-4o");
    setTheme("dark");
    setConnResult(null);
  };

  const handleTestConnection = async () => {
    setTesting(true);
    setConnResult(null);
    const start = performance.now();
    try {
      const res = await getHealth();
      const latency = Math.round(performance.now() - start);
      if (res.status === "ok" || res.status === "healthy") {
        setConnResult({ success: true, latencyMs: latency });
      } else {
        setConnResult({ success: false, msg: `Health status: ${res.status}` });
      }
    } catch (err: unknown) {
      setConnResult({
        success: false,
        msg: err instanceof Error ? err.message : "Could not reach backend API endpoint",
      });
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="p-6 md:p-8 space-y-8 max-w-4xl mx-auto">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between border-b border-white/[0.06] pb-6"
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-slate-700 to-slate-900 border border-white/10 shadow-lg">
            <Settings className="w-5 h-5 text-slate-200" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">System Settings</h1>
            <p className="text-xs text-slate-400 mt-0.5">Configure API connectivity, model defaults, and runtime preferences</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={handleReset} className="text-slate-400 hover:text-white">
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
        {/* ── API Connection Settings ────────────────────────────────────── */}
        <GlassCard className="p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-slate-200 font-semibold text-sm">
              <Server className="w-4 h-4 text-blue-400" />
              <span>Backend REST API Base Endpoint</span>
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={handleTestConnection}
              disabled={testing}
              className="text-xs border-white/10"
            >
              {testing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Activity className="w-3.5 h-3.5 text-emerald-400" />}
              Test Connection
            </Button>
          </div>

          <div className="space-y-2">
            <Input
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="https://hallucisense-production.up.railway.app"
              className="font-mono text-xs"
            />
            <p className="text-xs text-slate-500">
              Target production endpoint for `/api/v1/analyze`, `/api/v1/explain`, and `/api/v1/metrics`.
            </p>
          </div>

          {/* Connection Test Outcome */}
          {connResult && (
            <div
              className={`p-3 rounded-xl border text-xs font-mono flex items-center justify-between ${
                connResult.success
                  ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                  : "bg-rose-500/10 border-rose-500/30 text-rose-400"
              }`}
            >
              <span>
                {connResult.success
                  ? `✓ Backend Connected Successfully • Latency: ${connResult.latencyMs}ms`
                  : `✕ Backend Connection Failed: ${connResult.msg}`}
              </span>
            </div>
          )}
        </GlassCard>

        {/* ── Model Preferences ─────────────────────────────────────────── */}
        <GlassCard className="p-6 space-y-4">
          <div className="flex items-center gap-2 text-slate-200 font-semibold text-sm">
            <Cpu className="w-4 h-4 text-purple-400" />
            <span>Default Target Architecture</span>
          </div>
          <div className="space-y-2">
            <select
              value={defaultModel}
              onChange={(e) => setDefaultModel(e.target.value)}
              className="w-full h-10 px-3 rounded-xl border border-white/[0.08] bg-[#0b1220] text-xs font-mono text-slate-200 focus:outline-none focus:border-blue-500/50 cursor-pointer"
            >
              {MODEL_OPTIONS.map((m) => (
                <option key={m.value} value={m.value} className="bg-[#0b1220]">
                  {m.label}
                </option>
              ))}
            </select>
          </div>
        </GlassCard>

        {/* ── Appearance & Theme ────────────────────────────────────────── */}
        <GlassCard className="p-6 space-y-4">
          <div className="flex items-center gap-2 text-slate-200 font-semibold text-sm">
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

        {/* ── System Environment Metadata ───────────────────────────────── */}
        <GlassCard className="p-6 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-slate-300">Frontend Environment</span>
            <Badge variant="verified">v1.0 Production Release</Badge>
          </div>
          <div className="text-xs text-slate-500 space-y-1 font-mono">
            <p>Framework: Next.js 16 (App Router) + React 19</p>
            <p>Styling: Vanilla TailwindCSS v4 Design System</p>
            <p>State: TanStack Query v5 + Zustand</p>
          </div>
        </GlassCard>
      </div>
    </div>
  );
}
