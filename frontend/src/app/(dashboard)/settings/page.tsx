"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Settings, Server, Moon, Sun, Monitor, Cpu, Check, RotateCcw, Activity, RefreshCw } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { MODEL_OPTIONS } from "@/lib/constants";
import { getHealth } from "@/services/hallucisense-api";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const [apiUrl, setApiUrl] = useState("https://hallucisense-production.up.railway.app");
  const [defaultModel, setDefaultModel] = useState("GPT-4o");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedUrl = localStorage.getItem("hallucisense_api_url");
      if (savedUrl) {
        setApiUrl(savedUrl);
      } else {
        setApiUrl(
          process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "https://hallucisense-production.up.railway.app"
        );
      }

      const savedModel = localStorage.getItem("hallucisense_default_model");
      if (savedModel) {
        setDefaultModel(savedModel);
      }
    }
  }, []);

  // Connection Test State
  const [testing, setTesting] = useState(false);
  const [connResult, setConnResult] = useState<{ success: boolean; latencyMs?: number; msg?: string } | null>(null);

  const handleSave = () => {
    if (typeof window !== "undefined") {
      localStorage.setItem("hallucisense_api_url", apiUrl);
      localStorage.setItem("hallucisense_default_model", defaultModel);
    }
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    const defaultUrl = "https://hallucisense-production.up.railway.app";
    const defaultModelVal = "GPT-4o";
    setApiUrl(defaultUrl);
    setDefaultModel(defaultModelVal);
    setTheme("dark");
    setConnResult(null);
    if (typeof window !== "undefined") {
      localStorage.setItem("hallucisense_api_url", defaultUrl);
      localStorage.setItem("hallucisense_default_model", defaultModelVal);
    }
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
        className="flex items-center justify-between border-b border-white/[0.04] pb-6"
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-white/[0.01] border border-white/[0.04] shadow-lg">
            <Settings className="w-5 h-5 text-slate-400" />
          </div>
          <div>
            <h1 className="text-heading-md font-bold text-white tracking-tight leading-none">System Settings</h1>
            <p className="text-label-md text-slate-400 mt-0.5">Configure API connectivity, model defaults, and runtime preferences</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={handleReset} className="text-slate-500 hover:text-white font-mono text-xs cursor-pointer">
            <RotateCcw className="w-4 h-4" />
            Reset
          </Button>
          <Button onClick={handleSave} size="sm" className="bg-accent-primary hover:bg-accent-primary/90 text-white font-mono text-xs cursor-pointer shadow-[0_0_24px_rgba(168,85,247,0.2)] rounded-xl">
            {saved ? <Check className="w-4 h-4" /> : null}
            {saved ? "Saved" : "Save Changes"}
          </Button>
        </div>
      </motion.div>

      <div className="space-y-6">
        {/* ── API Connection Settings ────────────────────────────────────── */}
        <Card className="p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-slate-200 font-semibold text-sm">
              <Server className="w-4 h-4 text-slate-400" />
              <span>Backend REST API Base Endpoint</span>
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={handleTestConnection}
              disabled={testing}
              className="text-xs border-white/5 bg-white/[0.01] hover:bg-white/[0.04] text-slate-300 font-mono cursor-pointer"
            >
              {testing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Activity className="w-3.5 h-3.5 text-status-success animate-pulse" />}
              Test Connection
            </Button>
          </div>

          <div className="space-y-2">
            <Input
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="https://hallucisense-production.up.railway.app"
              className="font-mono text-xs bg-bg-surface border-white/[0.04] focus:border-accent-primary/40 focus:ring-accent-primary/10 text-slate-300"
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
                  ? "bg-status-success/10 border-status-success/30 text-status-success"
                  : "bg-status-error/10 border-status-error/30 text-status-error"
              }`}
            >
              <span>
                {connResult.success
                  ? `✓ Backend Connected Successfully • Latency: ${connResult.latencyMs}ms`
                  : `✕ Backend Connection Failed: ${connResult.msg}`}
              </span>
            </div>
          )}
        </Card>

        {/* ── Model Preferences ─────────────────────────────────────────── */}
        <Card className="p-6 space-y-4">
          <div className="flex items-center gap-2 text-slate-200 font-semibold text-sm">
            <Cpu className="w-4 h-4 text-slate-400" />
            <span>Default Target Architecture</span>
          </div>
          <div className="space-y-2">
            <select
              value={defaultModel}
              onChange={(e) => setDefaultModel(e.target.value)}
              className="w-full h-10 px-3 rounded-xl border border-white/[0.04] bg-bg-surface text-xs font-mono text-slate-300 focus:outline-none focus:border-accent-primary/40 cursor-pointer"
            >
              {MODEL_OPTIONS.map((m) => (
                <option key={m.value} value={m.value} className="bg-bg-surface">
                  {m.label}
                </option>
              ))}
            </select>
          </div>
        </Card>

        {/* ── Appearance & Theme ────────────────────────────────────────── */}
        <Card className="p-6 space-y-4">
          <div className="flex items-center gap-2 text-slate-200 font-semibold text-sm">
            <Sun className="w-4 h-4 text-slate-400" />
            <span>Appearance & Theme</span>
          </div>
          <div className="flex p-1 rounded-xl bg-white/[0.02] border border-white/[0.04] gap-1">
            <button
              onClick={() => setTheme("dark")}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg transition-all cursor-pointer text-xs font-medium ${
                theme === "dark"
                  ? "bg-accent-primary text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.02]"
              }`}
            >
              <Moon className="w-4 h-4" />
              <span>Dark Mode</span>
            </button>
            <button
              onClick={() => setTheme("light")}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg transition-all cursor-pointer text-xs font-medium ${
                theme === "light"
                  ? "bg-accent-primary text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.02]"
              }`}
            >
              <Sun className="w-4 h-4" />
              <span>Light Mode</span>
            </button>
            <button
              onClick={() => setTheme("system")}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg transition-all cursor-pointer text-xs font-medium ${
                theme === "system"
                  ? "bg-accent-primary text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.02]"
              }`}
            >
              <Monitor className="w-4 h-4" />
              <span>System Default</span>
            </button>
          </div>
        </Card>

        {/* ── System Environment Metadata ───────────────────────────────── */}
        <Card className="p-6 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-slate-300 font-sans">Frontend Environment</span>
            <StatusBadge label="v1.0 Production Release" status="success" />
          </div>
          <dl className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 border-t border-white/[0.04] text-[11px]">
            <div>
              <dt className="text-slate-500 font-sans">Framework</dt>
              <dd className="text-slate-200 font-mono font-medium mt-0.5">Next.js 16 (App Router) + React 19</dd>
            </div>
            <div>
              <dt className="text-slate-500 font-sans">Styling</dt>
              <dd className="text-slate-200 font-mono font-medium mt-0.5">Vanilla TailwindCSS v4 Design System</dd>
            </div>
            <div>
              <dt className="text-slate-500 font-sans">State</dt>
              <dd className="text-slate-200 font-mono font-medium mt-0.5">TanStack Query v5 + Zustand</dd>
            </div>
          </dl>
        </Card>
      </div>
    </div>
  );
}
