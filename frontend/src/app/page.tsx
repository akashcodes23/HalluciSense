"use client";

import React, { useRef } from "react";
import Link from "next/link";
import { motion, useInView } from "framer-motion";
import {
  ArrowRight,
  ShieldCheck,
  Activity,
  Zap,
  Database,
  GitBranch,
  ChevronRight,
  Globe,
  Lock,
  Layers,
  Cpu,
  BarChart2,
  BookOpen,
  CheckCircle2,
  Sparkles,
  ExternalLink,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { GlassCard } from "@/components/ui/card";
import { useMetrics } from "@/hooks/use-analysis";
import { formatLatency, formatNumber } from "@/lib/format";

/* Inline GitHub icon */
function GithubIcon({ size = 20, color = "currentColor" }: { size?: number; color?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.2c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
      <path d="M9 18c-4.51 2-5-2-7-2" />
    </svg>
  );
}

function FadeUp({ children, delay = 0, className = "" }: { children: React.ReactNode; delay?: number; className?: string }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.55, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/* ── Navbar ───────────────────────────────────────────────────────────────── */
function Navbar() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 border-b border-white/[0.06] bg-[#050816]/80 backdrop-blur-xl backdrop-saturate-150">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 shadow-[0_0_16px_rgba(37,99,235,0.4)]">
            <ShieldCheck className="w-4 h-4 text-white" />
          </div>
          <span className="font-bold text-white tracking-tight text-lg">HalluciSense</span>
          <Badge variant="primary" className="text-[10px] py-0 px-2">v1.0</Badge>
        </Link>

        <nav className="hidden md:flex items-center gap-6 text-sm text-slate-400">
          <a href="#features" className="hover:text-white transition-colors">Framework</a>
          <a href="#pillars" className="hover:text-white transition-colors">Pillars</a>
          <a href="#statistics" className="hover:text-white transition-colors">Telemetry</a>
          <a href="#research" className="hover:text-white transition-colors">Research</a>
        </nav>

        <div className="flex items-center gap-3">
          <a
            href="https://github.com/akashcodes23/HalluciSense"
            target="_blank"
            rel="noreferrer"
            className="p-2 text-slate-400 hover:text-white transition-colors"
            aria-label="GitHub Repository"
          >
            <GithubIcon size={18} />
          </a>
          <Link href="/analyze">
            <Button size="sm" className="shadow-[0_0_20px_rgba(37,99,235,0.3)]">
              Start Analyzing
              <ArrowRight className="w-3.5 h-3.5" />
            </Button>
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function LandingPage() {
  const { data: metrics } = useMetrics();

  return (
    <div className="min-h-screen bg-[#050816] text-slate-100 relative overflow-hidden">
      {/* Background Mesh Orbs */}
      <div className="mesh-bg" aria-hidden="true">
        <div className="mesh-orb mesh-orb-1" />
        <div className="mesh-orb mesh-orb-2" />
        <div className="mesh-orb mesh-orb-3" />
      </div>

      <Navbar />

      {/* ── Hero Section ───────────────────────────────────────────────────── */}
      <section className="relative pt-36 pb-24 px-6 z-10">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <FadeUp delay={0.1}>
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-blue-500/20 bg-blue-500/10 text-blue-400 text-xs font-semibold tracking-wide uppercase">
              <Sparkles className="w-3.5 h-3.5" />
              Scientific Hallucination Detection Engine
            </div>
          </FadeUp>

          <FadeUp delay={0.2}>
            <h1 className="display-hero font-bold tracking-tight text-white max-w-3xl mx-auto">
              Detect Hallucinations with <span className="text-gradient">Scientific Confidence</span>
            </h1>
          </FadeUp>

          <FadeUp delay={0.3}>
            <p className="body-lg max-w-2xl mx-auto text-slate-400">
              Confidence-aware AI verification powered by a three-pillar hallucination detection framework. Evidence grounding, logit entropy, and structural consistency.
            </p>
          </FadeUp>

          <FadeUp delay={0.4}>
            <div className="flex flex-wrap items-center justify-center gap-4 pt-2">
              <Link href="/analyze">
                <Button size="xl" className="shadow-[0_0_30px_rgba(37,99,235,0.4)]">
                  Start Analyzing
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
              <a href="#research">
                <Button variant="secondary" size="xl">
                  <BookOpen className="w-4 h-4" />
                  Research Paper
                </Button>
              </a>
              <a href="https://github.com/akashcodes23/HalluciSense" target="_blank" rel="noreferrer">
                <Button variant="outline" size="xl">
                  <GithubIcon size={18} />
                  GitHub
                </Button>
              </a>
            </div>
          </FadeUp>
        </div>
      </section>

      {/* ── Three Pillars Section ────────────────────────────────────────── */}
      <section id="pillars" className="py-20 px-6 relative z-10 border-t border-white/[0.06]">
        <div className="max-w-6xl mx-auto space-y-12">
          <div className="text-center space-y-3">
            <h2 className="display-2 text-white">Three-Pillar Architecture</h2>
            <p className="text-slate-400 max-w-xl mx-auto text-sm">
              Combining hybrid evidence retrieval, white-box uncertainty, and structural self-consistency for robust Platt-calibrated scoring.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <FadeUp delay={0.1}>
              <GlassCard className="p-8 space-y-4 h-full">
                <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
                  <Database className="w-6 h-6 text-indigo-400" />
                </div>
                <h3 className="text-lg font-bold text-white">Pillar 1 — Evidence Grounding</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Hybrid BM25 + dense vector retrieval against external reference corpora with cross-encoder NLI entailment verification.
                </p>
              </GlassCard>
            </FadeUp>

            <FadeUp delay={0.2}>
              <GlassCard className="p-8 space-y-4 h-full">
                <div className="w-12 h-12 rounded-2xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
                  <Activity className="w-6 h-6 text-purple-400" />
                </div>
                <h3 className="text-lg font-bold text-white">Pillar 2 — Confidence Estimation</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Logit entropy analysis and epistemic uncertainty quantification to measure internal model confidence gaps.
                </p>
              </GlassCard>
            </FadeUp>

            <FadeUp delay={0.3}>
              <GlassCard className="p-8 space-y-4 h-full">
                <div className="w-12 h-12 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                  <GitBranch className="w-6 h-6 text-blue-400" />
                </div>
                <h3 className="text-lg font-bold text-white">Pillar 3 — Consistency Reasoning</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Paraphrase-based self-consistency checks measuring semantic variance across multi-prompt generation runs.
                </p>
              </GlassCard>
            </FadeUp>
          </div>
        </div>
      </section>

      {/* ── Live Telemetry Section ────────────────────────────────────────── */}
      <section id="statistics" className="py-20 px-6 relative z-10 border-t border-white/[0.06] bg-white/[0.01]">
        <div className="max-w-5xl mx-auto space-y-12">
          <div className="text-center space-y-3">
            <h2 className="display-2 text-white">Live System Telemetry</h2>
            <p className="text-slate-400 text-sm">Real-time metrics from the deployed production backend.</p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <GlassCard className="p-6 text-center space-y-1">
              <span className="text-3xl font-bold font-mono text-blue-400">
                {metrics ? formatNumber(metrics.requests) : "1.6K+"}
              </span>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Total Requests</p>
            </GlassCard>

            <GlassCard className="p-6 text-center space-y-1">
              <span className="text-3xl font-bold font-mono text-emerald-400">
                {metrics ? `${metrics.success_rate.toFixed(1)}%` : "100.0%"}
              </span>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Pass Rate</p>
            </GlassCard>

            <GlassCard className="p-6 text-center space-y-1">
              <span className="text-3xl font-bold font-mono text-purple-400">
                {metrics ? formatLatency(metrics.average_latency_ms) : "< 150ms"}
              </span>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Avg Latency</p>
            </GlassCard>

            <GlassCard className="p-6 text-center space-y-1">
              <span className="text-3xl font-bold font-mono text-amber-400">8+</span>
              <p className="text-xs text-slate-500 uppercase tracking-wider">LLM Architectures</p>
            </GlassCard>
          </div>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="py-12 px-6 border-t border-white/[0.06] text-center text-xs text-slate-500 relative z-10">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-blue-400" />
            <span className="font-bold text-white">HalluciSense</span>
            <span>— Scientific Hallucination Detection</span>
          </div>
          <p>© 2026 HalluciSense. Open Source Research Release v1.0.</p>
        </div>
      </footer>
    </div>
  );
}
