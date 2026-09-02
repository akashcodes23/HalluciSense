"use client";

import React, { useRef, useState, useEffect } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { motion, AnimatePresence, useInView } from "framer-motion";
import {
  ArrowRight,
  ShieldCheck,
  Activity,
  Database,
  GitBranch,
  BookOpen,
  Sparkles,
  CheckCircle2,
  Lock,
  Layers,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { GlassCard } from "@/components/ui/card";
import { useMetrics } from "@/hooks/use-analysis";
import { formatLatency, formatNumber } from "@/lib/format";
import { useAnalysisStore } from "@/store/analysis-store";

/* ── Dynamic SSR-Safe 3D Canvas ─────────────────────────────────────────────── */
const Hero3DCanvas = dynamic(() => import("@/components/Hero3DCanvas"), {
  ssr: false,
  loading: () => (
    <div className="relative w-full h-[380px] sm:h-[440px] lg:h-[500px] flex items-center justify-center pointer-events-none select-none">
      <div className="w-52 h-52 rounded-full border border-emerald-500/20 bg-emerald-500/5 animate-pulse" />
    </div>
  ),
});

/* ── Inline GitHub Icon ─────────────────────────────────────────────────────── */
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

/* ── Navbar ────────────────────────────────────────────────────────────────── */
function Navbar() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 border-b border-white/[0.06] bg-[#07090e]/80 backdrop-blur-xl backdrop-saturate-150">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-8 h-8 rounded-xl bg-gradient-to-br from-emerald-600 to-teal-700 shadow-[0_0_16px_rgba(16,185,129,0.35)]">
            <ShieldCheck className="w-4 h-4 text-white" />
          </div>
          <span className="font-bold text-white tracking-tight text-lg">HalluciSense</span>
          <Badge variant="primary" className="text-[10px] py-0 px-2 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            v1.0
          </Badge>
        </Link>

        <nav className="hidden md:flex items-center gap-6 text-sm text-slate-400">
          <a href="#pillars" className="hover:text-white transition-colors">Architecture</a>
          <a href="#statistics" className="hover:text-white transition-colors">Telemetry</a>
          <Link href="/overview" className="hover:text-white transition-colors">Dashboard</Link>
          <Link href="/verify" className="hover:text-white transition-colors">Workbench</Link>
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
            <Button size="sm" className="bg-emerald-600 hover:bg-emerald-500 text-white shadow-[0_0_20px_rgba(16,185,129,0.3)] cursor-pointer">
              Open Verifier
              <ArrowRight className="w-3.5 h-3.5 ml-1" />
            </Button>
          </Link>
        </div>
      </div>
    </header>
  );
}

/* ── Hero State Data Dictionary ────────────────────────────────────────────── */
const HERO_STATES = {
  detect: {
    tag: "Pillar 1 · Evidence Grounding (FE)",
    headline: "Detect Hallucinations.",
    color: "text-emerald-400",
    glowColor: "rgba(16,185,129,0.3)",
    description: "Hybrid BM25 sparse + FAISS dense neural retrieval with cross-encoder NLI isolating factual claims from ungrounded parametric hallucinations.",
  },
  confidence: {
    tag: "Pillar 2 · Confidence Gap (CG)",
    headline: "Measure Confidence.",
    color: "text-teal-400",
    glowColor: "rgba(20,184,166,0.3)",
    description: "Token-level probability and Shannon entropy H(p) quantification estimating internal model uncertainty during generation.",
  },
  verify: {
    tag: "Pillar 3 · Consistency Failure (CF)",
    headline: "Verify Evidence.",
    color: "text-cyan-400",
    glowColor: "rgba(6,182,212,0.3)",
    description: "Multi-sample semantic consistency checks and cross-generation contradiction analysis measuring response stability.",
  },
} as const;

export default function LandingPage() {
  const { data: metrics } = useMetrics();
  const activeHeroState = useAnalysisStore((s) => s.activeHeroState);
  const setActiveHeroState = useAnalysisStore((s) => s.setActiveHeroState);
  const [reducedMotion, setReducedMotion] = useState(false);

  // Check reduced-motion preferences
  useEffect(() => {
    if (typeof window !== "undefined") {
      const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
      setReducedMotion(mediaQuery.matches);
    }
  }, []);

  // Hero State Automated Transition Loop (3.5s per state)
  useEffect(() => {
    if (reducedMotion) return;

    const interval = setInterval(() => {
      setActiveHeroState(
        activeHeroState === "detect"
          ? "confidence"
          : activeHeroState === "confidence"
          ? "verify"
          : "detect"
      );
    }, 3500);

    return () => clearInterval(interval);
  }, [activeHeroState, reducedMotion, setActiveHeroState]);

  const currentHero = HERO_STATES[activeHeroState];

  return (
    <div className="min-h-screen bg-[#07090e] text-slate-100 relative overflow-x-hidden font-sans">
      {/* Background Volumetric Gradients */}
      <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-radial from-emerald-500/8 via-teal-500/4 to-transparent blur-3xl pointer-events-none" />
      <div className="absolute top-48 right-1/4 w-[500px] h-[500px] bg-radial from-cyan-500/6 via-emerald-500/3 to-transparent blur-3xl pointer-events-none" />

      <Navbar />

      {/* ── 3D Hero Section ─────────────────────────────────────────────────── */}
      <section className="relative pt-32 pb-20 px-6 z-10">
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left Column: Editorial Headline & Synchronized State Machine */}
          <div className="lg:col-span-7 space-y-8 text-left">
            <FadeUp delay={0.1}>
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-emerald-500/20 bg-emerald-500/5 text-emerald-400 text-[11px] font-semibold tracking-wider uppercase font-mono">
                <Sparkles className="w-3.5 h-3.5" />
                EXPLAINABLE AI · DETECTION · CORRECTION · RE-VERIFICATION
              </div>
            </FadeUp>

            {/* Main Headline */}
            <div className="space-y-2">
              <FadeUp delay={0.2}>
                <h1 className="text-4xl sm:text-5xl lg:text-6xl font-[family-name:var(--font-space-grotesk)] font-extrabold tracking-tight text-white leading-[1.12]">
                  AI answers. <br />
                  We verify them.
                </h1>
              </FadeUp>

              {/* Dynamic Synchronized Animated Line */}
              <div className="h-[44px] sm:h-[52px] flex items-center overflow-hidden">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeHeroState}
                    initial={{ opacity: 0, y: 16, filter: "blur(4px)" }}
                    animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                    exit={{ opacity: 0, y: -16, filter: "blur(4px)" }}
                    transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
                    className={`text-2xl sm:text-3xl lg:text-3.5xl font-bold font-[family-name:var(--font-space-grotesk)] tracking-tight ${currentHero.color}`}
                  >
                    {currentHero.headline}
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>

            {/* Dynamic Description Line */}
            <div className="min-h-[64px] sm:min-h-[56px]">
              <AnimatePresence mode="wait">
                <motion.p
                  key={activeHeroState}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.35 }}
                  className="text-slate-400 text-sm sm:text-base max-w-xl leading-relaxed font-sans"
                >
                  HalluciSense decomposes an LLM response into atomic claims and evaluates each claim through evidence grounding, confidence gap, and consistency failure signals before producing an explainable, calibrated decision.
                </motion.p>
              </AnimatePresence>
            </div>

            {/* Hero Interactive State Indicator Pills */}
            <div className="flex items-center gap-2 pt-1">
              {(["detect", "confidence", "verify"] as const).map((st) => (
                <button
                  key={st}
                  onClick={() => setActiveHeroState(st)}
                  className={`px-3 py-1 rounded-lg text-xs font-mono transition-all cursor-pointer border ${
                    activeHeroState === st
                      ? "bg-white/[0.08] text-white border-emerald-500/40 shadow-[0_0_12px_rgba(16,185,129,0.2)]"
                      : "bg-white/[0.02] text-slate-500 border-white/[0.04] hover:text-slate-300 hover:border-white/[0.1]"
                  }`}
                >
                  {st.toUpperCase()}
                </button>
              ))}
            </div>

            {/* CTA Buttons */}
            <FadeUp delay={0.35}>
              <div className="flex flex-wrap items-center gap-4 pt-2">
                <Link href="/verify">
                  <Button size="xl" className="bg-emerald-600 hover:bg-emerald-500 text-white shadow-[0_0_24px_rgba(16,185,129,0.3)] rounded-xl cursor-pointer">
                    Open Verifier
                    <ArrowRight className="w-4 h-4 ml-1.5" />
                  </Button>
                </Link>
                <a href="#pillars">
                  <Button variant="outline" size="xl" className="border-white/10 bg-white/[0.02] hover:bg-white/[0.06] text-slate-300 hover:text-white rounded-xl flex items-center gap-2 cursor-pointer">
                    <BookOpen className="w-4 h-4" />
                    Explore the System
                  </Button>
                </a>
              </div>
            </FadeUp>
          </div>

          {/* Right Column: High-Performance 3D Scientific Gyroscope & Neural Core */}
          <div className="lg:col-span-5 relative flex items-center justify-center">
            <Hero3DCanvas heroState={activeHeroState} reducedMotion={reducedMotion} />
          </div>
        </div>
      </section>

      {/* ── Three Pillars Architecture Section ─────────────────────────────── */}
      <section id="pillars" className="py-24 px-6 relative z-10 border-t border-white/[0.04] bg-[#090c14]/40">
        <div className="max-w-6xl mx-auto space-y-12">
          <div className="text-center space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-emerald-500/20 bg-emerald-500/5 text-emerald-400 text-[10px] font-semibold tracking-wide uppercase font-mono mb-2">
              Tri-Modal Research Framework · Detect · Explain · Correct · Re-Verify
            </div>
            <h2 className="text-3xl sm:text-4xl font-bold font-[family-name:var(--font-space-grotesk)] text-white tracking-tight">
              Adaptive Three-Pillar Architecture
            </h2>
            <p className="text-slate-400 max-w-2xl mx-auto text-sm leading-relaxed">
              Grounded external evidence retrieval acts as the foundation for every claim, with predictive model uncertainty and stochastic generation consistency activating dynamically for live inference.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Pillar 1 */}
            <ScrollReveal delay={0.1}>
              <GlassCard className="p-8 space-y-6 h-full flex flex-col justify-between border-white/[0.06] bg-[#0c101c]/60 hover:border-emerald-500/30 transition-colors">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Database className="w-8 h-8 text-emerald-400" />
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold uppercase">
                      P1 · FE
                    </span>
                  </div>
                  <h3 className="text-xl font-bold text-white font-[family-name:var(--font-space-grotesk)]">
                    Pillar 1 — Evidence Grounding (FE)
                  </h3>
                  <p className="text-sm text-slate-400 leading-relaxed">
                    Hybrid BM25 sparse + FAISS dense retrieval against external reference corpora with DeBERTa-v3 cross-encoder NLI entailment scoring. Evaluates factual error and evidence contradiction across claims.
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-white/[0.02] border border-white/5 font-mono text-[10px] tracking-wider uppercase text-emerald-400 flex items-center justify-center gap-2">
                  Always Active · BM25 + FAISS + NLI
                </div>
              </GlassCard>
            </ScrollReveal>

            {/* Pillar 2 */}
            <ScrollReveal delay={0.2}>
              <GlassCard className="p-8 space-y-6 h-full flex flex-col justify-between border-white/[0.06] bg-[#0c101c]/60 hover:border-teal-500/30 transition-colors">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Activity className="w-8 h-8 text-teal-400" />
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-teal-500/10 text-teal-400 border border-teal-500/20 font-semibold uppercase">
                      P2 · CG
                    </span>
                  </div>
                  <h3 className="text-xl font-bold text-white font-[family-name:var(--font-space-grotesk)]">
                    Pillar 2 — Confidence Gap (CG)
                  </h3>
                  <p className="text-sm text-slate-400 leading-relaxed">
                    Token log-probability distribution and Shannon entropy H(p) analysis quantifying internal generation uncertainty during streaming LLM generation.
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-white/[0.02] border border-white/5 font-mono text-[10px] tracking-wider uppercase text-teal-400 flex items-center justify-center gap-2">
                  Live Streams · Shannon Entropy H(p)
                </div>
              </GlassCard>
            </ScrollReveal>

            {/* Pillar 3 */}
            <ScrollReveal delay={0.3}>
              <GlassCard className="p-8 space-y-6 h-full flex flex-col justify-between border-white/[0.06] bg-[#0c101c]/60 hover:border-cyan-500/30 transition-colors">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <GitBranch className="w-8 h-8 text-cyan-400" />
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-semibold uppercase">
                      P3 · CF
                    </span>
                  </div>
                  <h3 className="text-xl font-bold text-white font-[family-name:var(--font-space-grotesk)]">
                    Pillar 3 — Consistency Failure (CF)
                  </h3>
                  <p className="text-sm text-slate-400 leading-relaxed">
                    Semantic consistency and cross-sample contradiction analysis across independent stochastic generations or intra-response claim pairs.
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-white/[0.02] border border-white/5 font-mono text-[10px] tracking-wider uppercase text-cyan-400 flex items-center justify-center gap-2">
                  Multi-Sample · Cross-Generation Stability
                </div>
              </GlassCard>
            </ScrollReveal>
          </div>

          {/* Availability-Aware Explanation Banner */}
          <div className="p-5 rounded-2xl bg-black/40 border border-white/[0.08] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-xs font-mono">
            <div className="space-y-1">
              <span className="text-slate-200 font-sans font-semibold text-sm">Deterministic Symbolic Verification</span>
              <p className="text-slate-400 font-sans text-xs">
                Arithmetic expressions, temporal logic, and unit conversions are verified deterministically by the Evidence Intelligence Gateway, eliminating numerical false negatives.
              </p>
            </div>
            <div className="shrink-0 flex items-center gap-2 text-[11px]">
              <span className="px-3 py-1 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Deterministic S1 Override
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Live Telemetry Section ─────────────────────────────────────────── */}
      <section id="statistics" className="py-24 px-6 relative z-10 border-t border-white/[0.04]">
        <div className="max-w-5xl mx-auto space-y-12">
          <div className="text-center space-y-3">
            <h2 className="text-3xl sm:text-4xl font-bold font-[family-name:var(--font-space-grotesk)] text-white tracking-tight">
              Live System Telemetry
            </h2>
            <p className="text-slate-400 text-sm">Real-time performance metrics from the deployed production backend.</p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <ScrollReveal delay={0.1}>
              <GlassCard className="p-6 text-center space-y-2 border-white/[0.06] bg-[#0c101c]/50 h-full">
                <span className="text-2xl sm:text-3xl font-bold font-mono text-white">
                  {metrics && metrics.requests > 0 ? formatNumber(metrics.requests) : "—"}
                </span>
                <p className="text-xs text-slate-500 font-mono">Total Requests</p>
              </GlassCard>
            </ScrollReveal>

            <ScrollReveal delay={0.2}>
              <GlassCard className="p-6 text-center space-y-2 border-white/[0.06] bg-[#0c101c]/50 h-full">
                <span className="text-2xl sm:text-3xl font-bold font-mono text-emerald-400">
                  {metrics && metrics.requests > 0 && metrics.success_rate !== null ? `${metrics.success_rate.toFixed(1)}%` : "—"}
                </span>
                <p className="text-xs text-slate-500 font-mono">Pass Rate</p>
              </GlassCard>
            </ScrollReveal>

            <ScrollReveal delay={0.3}>
              <GlassCard className="p-6 text-center space-y-2 border-white/[0.06] bg-[#0c101c]/50 h-full">
                <span className="text-2xl sm:text-3xl font-bold font-mono text-teal-400">
                  {metrics && metrics.requests > 0 && metrics.average_latency_ms !== null ? formatLatency(metrics.average_latency_ms) : "—"}
                </span>
                <p className="text-xs text-slate-500 font-mono">Avg Latency</p>
              </GlassCard>
            </ScrollReveal>

            <ScrollReveal delay={0.4}>
              <GlassCard className="p-6 text-center space-y-2 border-white/[0.06] bg-[#0c101c]/50 h-full">
                <span className="text-2xl sm:text-3xl font-bold font-mono text-cyan-400">8+</span>
                <p className="text-xs text-slate-500 font-mono">Active Modalities</p>
              </GlassCard>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="py-8 px-6 border-t border-white/[0.06] text-center text-xs text-slate-500 relative z-10 bg-[#07090e]">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span className="font-bold text-white">HalluciSense</span>
            <span>— Scientific Hallucination Detection & Verification Engine</span>
          </div>
          <p>© 2026 HalluciSense. Open Source Research Release v1.0.</p>
        </div>
      </footer>
    </div>
  );
}
