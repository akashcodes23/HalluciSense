"use client";

import React, { useRef } from "react";
import Link from "next/link";
import { motion, useInView } from "framer-motion";
import {
  ArrowRight,
  ShieldCheck,
  Activity,
  Database,
  GitBranch,
  BookOpen,
  Sparkles,
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
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: [0.25, 1, 0.5, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

function ScrollReveal({ children, delay = 0, className = "" }: { children: React.ReactNode; delay?: number; className?: string }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 16 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay, ease: [0.25, 1, 0.5, 1] }}
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
          <Link href="/verify">
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
    <div className="min-h-screen bg-[#050816] text-slate-100 relative overflow-x-hidden font-sans">
      <Navbar />

      {/* ── Hero Section ───────────────────────────────────────────────────── */}
      <section className="relative pt-32 pb-24 px-6 z-10">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <FadeUp delay={0.1}>
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-accent-primary/20 bg-accent-primary/5 text-accent-primary text-[10px] font-semibold tracking-wide uppercase font-mono">
              <Sparkles className="w-3.5 h-3.5" />
              Confidence-Aware AI Verification
            </div>
          </FadeUp>

          <FadeUp delay={0.2}>
            <h1 className="text-4xl sm:text-5xl lg:text-6.5xl font-[family-name:var(--font-space-grotesk)] font-extrabold tracking-tight text-white max-w-4xl mx-auto leading-[1.15]">
              Detect Hallucinations.<br className="hidden sm:inline" />
              Measure Confidence. <span className="text-accent-primary">Verify Evidence.</span>
            </h1>
          </FadeUp>

          <FadeUp delay={0.3}>
            <p className="text-body max-w-2xl mx-auto text-slate-400 leading-relaxed">
              Evidence-grounded verification with adaptive confidence and consistency analysis. Combines invariant external retrieval grounding with real-time model uncertainty and multi-sample consensus.
            </p>
          </FadeUp>

          <FadeUp delay={0.4}>
            <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
              <Link href="/verify">
                <Button size="xl" className="bg-accent-primary hover:bg-accent-primary/90 text-white shadow-[0_0_24px_rgba(168,85,247,0.2)] rounded-xl cursor-pointer">
                  Start Analyzing
                  <ArrowRight className="w-4 h-4 ml-1" />
                </Button>
              </Link>
              <a href="#pillars">
                <Button variant="outline" size="xl" className="border-white/5 bg-white/[0.01] hover:bg-white/[0.04] text-slate-400 hover:text-white rounded-xl flex items-center gap-2">
                  <BookOpen className="w-4 h-4" />
                  Explore Framework
                </Button>
              </a>
            </div>
          </FadeUp>
        </div>
      </section>

      {/* ── Three Pillars Section ────────────────────────────────────────── */}
      <section id="pillars" className="py-24 px-6 relative z-10 border-t border-white/[0.04] bg-bg-surface/10">
        <div className="max-w-6xl mx-auto space-y-12">
          <div className="text-center space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/20 bg-blue-500/5 text-blue-400 text-[10px] font-semibold tracking-wide uppercase font-mono mb-2">
              Availability-Aware Fusion Engine
            </div>
            <h2 className="text-heading-lg text-white tracking-tight">Adaptive Multi-Signal Architecture</h2>
            <p className="text-slate-400 max-w-2xl mx-auto text-label-md leading-relaxed">
              Grounded evidence retrieval serves as the invariant foundation for every claim, with token uncertainty and stochastic consensus activating dynamically for live model generation streams.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Pillar 1 */}
            <ScrollReveal delay={0.1}>
              <GlassCard className="p-8 space-y-6 h-full flex flex-col justify-between border-white/[0.04] bg-bg-surface/40">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Database className="w-8 h-8 text-blue-400" />
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-semibold uppercase">
                      Invariant Base
                    </span>
                  </div>
                  <h3 className="text-heading-sm font-bold text-white">Pillar 1 — Evidence Grounding</h3>
                  <p className="text-label-md text-slate-400 leading-relaxed">
                    Hybrid BM25 sparse + FAISS dense vector retrieval against external reference corpora with DeBERTa-v3 cross-encoder NLI entailment scoring. Evaluates all claims, including offline text.
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-white/[0.01] border border-white/5 font-mono text-[10px] tracking-wider uppercase text-blue-400 flex items-center justify-center gap-2">
                  Always Active · BM25 + FAISS + NLI
                </div>
              </GlassCard>
            </ScrollReveal>

            {/* Pillar 2 */}
            <ScrollReveal delay={0.2}>
              <GlassCard className="p-8 space-y-6 h-full flex flex-col justify-between border-white/[0.04] bg-bg-surface/40">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Activity className="w-8 h-8 text-status-warning" />
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 font-semibold uppercase">
                      Adaptive Signal
                    </span>
                  </div>
                  <h3 className="text-heading-sm font-bold text-white">Pillar 2 — Predictive Confidence</h3>
                  <p className="text-label-md text-slate-400 leading-relaxed">
                    Token log-probability entropy analysis quantifying internal model uncertainty. Dynamically active during live streaming generation when provider logprobs are exposed.
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-white/[0.01] border border-white/5 font-mono text-[10px] tracking-wider uppercase text-status-warning flex items-center justify-center gap-2">
                  Live Streams · Shannon Entropy H(p)
                </div>
              </GlassCard>
            </ScrollReveal>

            {/* Pillar 3 */}
            <ScrollReveal delay={0.3}>
              <GlassCard className="p-8 space-y-6 h-full flex flex-col justify-between border-white/[0.04] bg-bg-surface/40">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <GitBranch className="w-8 h-8 text-status-success" />
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold uppercase">
                      Adaptive Signal
                    </span>
                  </div>
                  <h3 className="text-heading-sm font-bold text-white">Pillar 3 — Semantic Consistency</h3>
                  <p className="text-label-md text-slate-400 leading-relaxed">
                    Evaluates semantic embedding cosine variance and claim-aligned contradiction across exactly 3 stochastic alternate generations during live multi-candidate runs.
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-white/[0.01] border border-white/5 font-mono text-[10px] tracking-wider uppercase text-status-success flex items-center justify-center gap-2">
                  Multi-Sample · 3 Stochastic Candidates
                </div>
              </GlassCard>
            </ScrollReveal>
          </div>

          {/* Availability-Aware Explanation Banner */}
          <div className="p-5 rounded-2xl bg-black/30 border border-white/[0.06] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-xs font-mono">
            <div className="space-y-1">
              <span className="text-slate-300 font-sans font-semibold text-sm">Execution Model Provenance</span>
              <p className="text-slate-400 font-sans text-xs">
                Static Text Verification is evaluated primarily via Pillar 1 Evidence Grounding ($H = P_1$). Live model streams unlock full three-pillar fusion ($H = 0.45 P_1 + 0.30 P_2 + 0.25 P_3$).
              </p>
            </div>
            <div className="shrink-0 flex items-center gap-2 text-[11px]">
              <span className="px-2.5 py-1 rounded-lg bg-white/[0.04] text-slate-300 border border-white/[0.08]">
                Transparent Renormalization
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Live Telemetry Section ────────────────────────────────────────── */}
      <section id="statistics" className="py-24 px-6 relative z-10 border-t border-white/[0.04]">
        <div className="max-w-5xl mx-auto space-y-12">
          <div className="text-center space-y-3">
            <h2 className="text-heading-lg text-white tracking-tight">Live System Telemetry</h2>
            <p className="text-slate-400 text-label-md">Real-time performance metrics from the deployed production backend.</p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <ScrollReveal delay={0.1}>
              <GlassCard className="p-6 text-center space-y-2 border-white/[0.04] bg-bg-surface/40 h-full">
                <span className="text-heading-md font-bold font-mono text-white">
                  {metrics && metrics.requests > 0 ? formatNumber(metrics.requests) : "—"}
                </span>
                <p className="text-label-sm text-slate-500 font-mono">Total Requests</p>
              </GlassCard>
            </ScrollReveal>

            <ScrollReveal delay={0.2}>
              <GlassCard className="p-6 text-center space-y-2 border-white/[0.04] bg-bg-surface/40 h-full">
                <span className="text-heading-md font-bold font-mono text-status-success">
                  {metrics && metrics.requests > 0 && metrics.success_rate !== null ? `${metrics.success_rate.toFixed(1)}%` : "—"}
                </span>
                <p className="text-label-sm text-slate-500 font-mono">Pass Rate</p>
              </GlassCard>
            </ScrollReveal>

            <ScrollReveal delay={0.3}>
              <GlassCard className="p-6 text-center space-y-2 border-white/[0.04] bg-bg-surface/40 h-full">
                <span className="text-heading-md font-bold font-mono text-blue-400">
                  {metrics && metrics.requests > 0 && metrics.average_latency_ms !== null ? formatLatency(metrics.average_latency_ms) : "—"}
                </span>
                <p className="text-label-sm text-slate-500 font-mono">Avg Latency</p>
              </GlassCard>
            </ScrollReveal>

            <ScrollReveal delay={0.4}>
              <GlassCard className="p-6 text-center space-y-2 border-white/[0.04] bg-bg-surface/40 h-full">
                <span className="text-heading-md font-bold font-mono text-status-warning">8+</span>
                <p className="text-label-sm text-slate-500 font-mono">Active Models</p>
              </GlassCard>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="py-6 px-6 border-t border-white/[0.04] text-center text-label-md text-slate-500 relative z-10 bg-[#050816]">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-accent-primary" />
            <span className="font-bold text-white">HalluciSense</span>
            <span>— Scientific Hallucination Detection</span>
          </div>
          <p>© 2026 HalluciSense. Open Source Research Release v1.0.</p>
        </div>
      </footer>
    </div>
  );
}
