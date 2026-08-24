"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Settings, Server, Moon, Sun, Monitor, Cpu, Check, RotateCcw, Activity, RefreshCw, Sparkles, Shield } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { MODEL_OPTIONS } from "@/lib/constants";
import { getHealth } from "@/services/hallucisense-api";
import { cn } from "@/lib/utils";

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
    <div className="p-5 md:p-8 space-y-6 max-w-4xl mx-auto pb-20 md:pb-8">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between border-b border-[var(--border)] pb-5"
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-[var(--radius-lg)] bg-[var(--surface)] border border-[var(--border)]">
            <Settings className="w-5 h-5 text-[var(--text-secondary)]" />
          </div>
          <div>
            <h1 className="text-heading-md font-bold text-[var(--text-primary)] tracking-tight">System Settings</h1>
            <p className="text-label-md text-[var(--text-muted)] mt-0.5">Configure API connectivity, model defaults, and runtime preferences</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={handleReset} className="text-xs">
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Defaults
          </Button>
          <Button variant="default" size="sm" onClick={handleSave} className="text-xs min-w-[80px]">
            {saved ? (
              <>
                <Check className="w-3.5 h-3.5 text-[var(--verified)]" />
                Saved
              </>
            ) : (
              "Save Changes"
            )}
          </Button>
        </div>
      </motion.div>

      {/* ── API Configuration ─────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Server className="w-4 h-4 text-[var(--evidence)]" />
            Backend Connection
          </CardTitle>
          <CardDescription>
            Specify the backend URL running the HalluciSense verification pipeline
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">API Base URL</label>
            <div className="flex gap-2">
              <Input
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                placeholder="https://your-backend.railway.app"
                className="font-mono text-xs bg-[var(--surface)] border-[var(--border)]"
              />
              <Button
                variant="secondary"
                size="sm"
                onClick={handleTestConnection}
                disabled={testing}
                className="shrink-0 text-xs"
              >
                {testing ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Testing…
                  </>
                ) : (
                  <>
                    <Activity className="w-3.5 h-3.5 text-[var(--primary)]" />
                    Test Ping
                  </>
                )}
              </Button>
            </div>
          </div>

          {connResult && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              className={cn(
                "p-3 rounded-[var(--radius)] text-xs flex items-center justify-between border",
                connResult.success
                  ? "bg-[var(--verified-soft)] border-[var(--verified-border)] text-[var(--verified)]"
                  : "bg-[var(--hallucination-soft)] border-[var(--hallucination-border)] text-[var(--hallucination)]"
              )}
            >
              <div className="flex items-center gap-2">
                <span className={cn("w-2 h-2 rounded-full", connResult.success ? "bg-[var(--verified)]" : "bg-[var(--hallucination)]")} />
                <span>{connResult.success ? "Connection successful" : connResult.msg}</span>
              </div>
              {connResult.latencyMs !== undefined && (
                <span className="font-mono font-bold">{connResult.latencyMs}ms</span>
              )}
            </motion.div>
          )}
        </CardContent>
      </Card>

      {/* ── Default Target Model ──────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Cpu className="w-4 h-4 text-[var(--ai)]" />
            Default Model Configuration
          </CardTitle>
          <CardDescription>
            Target architecture assumed when evaluating hallucination scores
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Active Model Profile</label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {MODEL_OPTIONS.map((m) => {
                const isSelected = defaultModel === m.value;
                return (
                  <button
                    key={m.value}
                    type="button"
                    onClick={() => setDefaultModel(m.value)}
                    className={cn(
                      "p-3 rounded-[var(--radius)] border text-left transition-all cursor-pointer",
                      isSelected
                        ? "bg-[var(--primary-soft)] border-[var(--ai-border)] text-[var(--primary)]"
                        : "bg-[var(--surface)] border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--border-hover)] hover:bg-[var(--surface-hover)]"
                    )}
                  >
                    <div className="text-xs font-semibold">{m.label}</div>
                    <div className="text-[10px] text-[var(--text-muted)] mt-0.5 font-mono">
                      {isSelected ? "Active default" : "Select profile"}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ── Appearance ────────────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Moon className="w-4 h-4 text-[var(--text-secondary)]" />
            Interface Appearance
          </CardTitle>
          <CardDescription>
            Select your preferred visual environment (Enterprise Dark optimized)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-2 max-w-sm">
            {[
              { id: "dark", label: "Dark", icon: Moon },
              { id: "light", label: "Light", icon: Sun },
              { id: "system", label: "System", icon: Monitor },
            ].map((opt) => {
              const Icon = opt.icon;
              const isSelected = theme === opt.id;
              return (
                <button
                  key={opt.id}
                  onClick={() => setTheme(opt.id)}
                  className={cn(
                    "flex items-center justify-center gap-2 py-2.5 px-3 rounded-[var(--radius)] border text-xs font-medium transition-all cursor-pointer",
                    isSelected
                      ? "bg-[var(--primary-soft)] border-[var(--ai-border)] text-[var(--primary)]"
                      : "bg-[var(--surface)] border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--border-hover)]"
                  )}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {opt.label}
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* ── Platform Architecture & Metadata ──────────────────────────── */}
      <Card variant="inset">
        <CardContent className="p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Platform Architecture</span>
            <Badge variant="outline" size="sm">v2.0 Enterprise</Badge>
          </div>
          <p className="text-xs text-[var(--text-dim)] leading-relaxed">
            HalluciSense Multi-Pillar Hybrid Observability Platform. Powered by dense & BM25 grounding (Pillar 1), token-level uncertainty estimation (Pillar 2), and semantic self-consistency reasoning (Pillar 3) with calibrated isotonic fusion.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
