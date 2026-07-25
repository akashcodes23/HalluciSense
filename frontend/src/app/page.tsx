'use client';

import React, { useRef } from 'react';
import Link from 'next/link';
import { motion, useInView } from 'framer-motion';
import {
  ArrowRight, ShieldCheck, Activity, Zap,
  Database, GitBranch, ChevronRight,
  Globe, Lock, Layers, Cpu, BarChart2, BookOpen,
} from 'lucide-react';

/* Inline GitHub icon — lucide-react v1.x removed this icon */
function GithubIcon({ size = 24, color = 'currentColor' }: { size?: number; color?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24"
      fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.2c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
      <path d="M9 18c-4.51 2-5-2-7-2" />
    </svg>
  );
}

/* ═══════════════════════════════════ UTILITIES ═══════════════════════════ */

function FadeUp({ children, delay = 0, className = '' }: any) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });
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

function FadeIn({ children, delay = 0, className = '' }: any) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0 }}
      animate={inView ? { opacity: 1 } : {}}
      transition={{ duration: 0.6, delay }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/* ═══════════════════════════════════ NAVBAR ══════════════════════════════ */

function Navbar() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 100,
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        background: 'rgba(5,7,15,0.8)',
      }}
    >
      <div style={{ maxWidth: 'var(--max-w)', margin: '0 auto', padding: '0 24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: '64px' }}>
          
          {/* Logo */}
          <Link href="/" style={{ display: 'flex', alignItems: 'center', gap: '10px', textDecoration: 'none' }}>
            <div style={{
              width: 32, height: 32, borderRadius: 8,
              background: 'rgba(99,102,241,0.15)',
              border: '1px solid rgba(99,102,241,0.3)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Database size={16} color="#818cf8" />
            </div>
            <span style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
              HalluciSense
            </span>
          </Link>

          {/* Nav links - desktop */}
          <nav style={{ display: 'flex', alignItems: 'center', gap: 32 }}>
            {['Features', 'Technology', 'Documentation'].map(link => (
              <a key={link} href={`#${link.toLowerCase()}`} className="nav-link">{link}</a>
            ))}
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="nav-link"
               style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <GithubIcon size={15} /> GitHub
            </a>
          </nav>

          {/* CTA */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Link href="/login" className="btn btn-ghost btn-sm">Sign In</Link>
            <Link href="/login" className="btn btn-primary btn-sm" style={{ gap: 6 }}>
              Get Started <ArrowRight size={14} />
            </Link>
          </div>
        </div>
      </div>
    </motion.header>
  );
}

/* ═══════════════════════════════════ HERO ════════════════════════════════ */

function HeroSection() {
  return (
    <section className="section" style={{ paddingTop: 120, paddingBottom: 120, textAlign: 'center' }}>
      <div className="container" style={{ maxWidth: 780, margin: '0 auto' }}>
        
        {/* Eyebrow pill */}
        <FadeIn delay={0.05}>
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 28 }}>
            <span className="pill pill-accent">
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#818cf8', display: 'inline-block' }} />
              Tri-Pillar Hallucination Detection Engine
            </span>
          </div>
        </FadeIn>

        {/* Headline */}
        <FadeUp delay={0.1}>
          <h1 className="display-1" style={{ marginBottom: 24 }}>
            Know when to{' '}
            <span className="text-gradient">trust your AI.</span>
          </h1>
        </FadeUp>

        {/* Description */}
        <FadeUp delay={0.18}>
          <p className="body-lg" style={{ maxWidth: 580, margin: '0 auto 44px', textAlign: 'center' }}>
            HalluciSense intercepts every LLM response and analyzes it across three independent 
            dimensions — factual grounding, generation confidence, and semantic consistency — 
            before a single word reaches you.
          </p>
        </FadeUp>

        {/* CTAs */}
        <FadeUp delay={0.25}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12, flexWrap: 'wrap' }}>
            <Link href="/login" className="btn btn-primary btn-lg">
              Start Verifying <ArrowRight size={16} />
            </Link>
            <Link href="/login" className="btn btn-ghost btn-lg">
              Sign In
            </Link>
          </div>
        </FadeUp>

        {/* Trust line */}
        <FadeIn delay={0.45}>
          <p style={{ marginTop: 32, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Free to start · Research-grade accuracy · No vendor lock-in
          </p>
        </FadeIn>

        {/* Hero visual — pipeline preview card */}
        <FadeUp delay={0.35}>
          <div style={{ marginTop: 64, position: 'relative' }}>
            {/* Glow behind the card */}
            <div style={{
              position: 'absolute', inset: '-40px -60px',
              background: 'radial-gradient(ellipse 70% 50% at 50% 40%, rgba(99,102,241,0.18) 0%, transparent 70%)',
              pointerEvents: 'none',
            }} />
            <div className="glass-strong" style={{ borderRadius: 20, padding: '4px', overflow: 'hidden', position: 'relative' }}>
              <div style={{
                borderRadius: 16, padding: '20px 24px',
                background: 'rgba(0,0,0,0.2)',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12
              }}>
                {[
                  { label: 'H-Score', value: '0.23', color: '#22c55e', bg: 'rgba(34,197,94,0.1)' },
                  { label: 'Factual Error', value: '4%', color: '#22c55e', bg: 'rgba(34,197,94,0.1)' },
                  { label: 'Confidence Gap', value: '18%', color: '#f59e0b', bg: 'rgba(245,158,11,0.1)' },
                  { label: 'Consistency', value: '97%', color: '#22c55e', bg: 'rgba(34,197,94,0.1)' },
                  { label: 'Risk Level', value: 'Low', color: '#22c55e', bg: 'rgba(34,197,94,0.1)' },
                ].map(({ label, value, color, bg }) => (
                  <div key={label} style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4, letterSpacing: '0.06em', textTransform: 'uppercase' }}>{label}</div>
                    <div style={{
                      fontSize: '1.1rem', fontWeight: 700, color,
                      background: bg, padding: '4px 12px', borderRadius: 8,
                      border: `1px solid ${color}30`
                    }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </FadeUp>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════ FEATURES ════════════════════════════ */

function FeatureCards() {
  const features = [
    {
      icon: ShieldCheck, color: '#22c55e',
      colorSoft: 'rgba(34,197,94,0.1)',
      title: 'Factual Grounding',
      tag: 'Pillar I',
      description: 'Extracts claims from every AI response and cross-references them against Wikipedia, internal knowledge bases, and custom sources using a hybrid BM25 + FAISS + CrossEncoder retrieval pipeline.',
      metrics: ['BM25 Sparse Search', 'FAISS Dense Vectors', 'CrossEncoder Reranking'],
    },
    {
      icon: Activity, color: '#6366f1',
      colorSoft: 'rgba(99,102,241,0.1)',
      title: 'Confidence Analysis',
      tag: 'Pillar II',
      description: 'Reads raw per-token logprobs from the language model and computes entropy-based confidence scores to detect uncertainty, hedging, and hallucination-prone regions at the token level.',
      metrics: ['Token Logprob Analysis', 'Entropy Scoring', 'Confidence Gap (CG)'],
    },
    {
      icon: Zap, color: '#a855f7',
      colorSoft: 'rgba(168,85,247,0.1)',
      title: 'Semantic Consistency',
      tag: 'Pillar III',
      description: 'Generates N paraphrased responses to the same prompt, embeds them using sentence-transformers, and computes a consistency failure score via cosine similarity across the response matrix.',
      metrics: ['Multi-sample Generation', 'Sentence Embeddings', 'Consistency Matrix'],
    },
  ];

  return (
    <section id="features" className="section" style={{ paddingTop: 80, paddingBottom: 80 }}>
      <div className="container">
        <FadeUp>
          <div style={{ textAlign: 'center', marginBottom: 60 }}>
            <p className="label" style={{ marginBottom: 12 }}>Detection Engine</p>
            <h2 className="display-2" style={{ marginBottom: 16 }}>Three pillars. One H-Score.</h2>
            <p className="body-lg" style={{ maxWidth: 520, margin: '0 auto' }}>
              Every pillar runs in parallel. The fusion layer combines them into a single, explainable Hallucination Score per sentence.
            </p>
          </div>
        </FadeUp>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 20 }}>
          {features.map(({ icon: Icon, color, colorSoft, title, tag, description, metrics }, i) => (
            <FadeUp key={title} delay={i * 0.1}>
              <div className="card" style={{ padding: 28, height: '100%', display: 'flex', flexDirection: 'column' }}>
                {/* Icon + tag */}
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 20 }}>
                  <div style={{
                    width: 44, height: 44, borderRadius: 12,
                    background: colorSoft, border: `1px solid ${color}30`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <Icon size={22} color={color} />
                  </div>
                  <span style={{
                    fontSize: '0.7rem', fontWeight: 700, letterSpacing: '0.08em',
                    textTransform: 'uppercase', color, background: colorSoft,
                    border: `1px solid ${color}25`, padding: '3px 10px', borderRadius: 99
                  }}>{tag}</span>
                </div>

                <h3 className="headline" style={{ marginBottom: 12 }}>{title}</h3>
                <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: 24, flex: 1 }}>
                  {description}
                </p>

                {/* Metrics */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {metrics.map(m => (
                    <div key={m} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <ChevronRight size={13} color={color} />
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 500 }}>{m}</span>
                    </div>
                  ))}
                </div>
              </div>
            </FadeUp>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════ HOW IT WORKS ════════════════════════ */

function HowItWorks() {
  const steps = [
    { icon: '💬', label: 'User Prompt',      desc: 'You send a message' },
    { icon: '🤖', label: 'LLM Generation',   desc: 'Streams tokens + logprobs' },
    { icon: '⚙️', label: 'HalluciSense',     desc: 'Pipeline triggered async' },
    { icon: '🔍', label: 'Evidence Retrieval',desc: 'BM25 + FAISS + Wikipedia' },
    { icon: '📊', label: 'Score Fusion',      desc: 'H = αFE + βCG + γCF' },
    { icon: '🎨', label: 'Annotated Response', desc: 'Color-coded, clickable' },
  ];

  return (
    <section id="technology" className="section" style={{ background: 'rgba(255,255,255,0.01)' }}>
      <div className="container">
        <FadeUp>
          <div style={{ textAlign: 'center', marginBottom: 64 }}>
            <p className="label" style={{ marginBottom: 12 }}>Architecture</p>
            <h2 className="display-2" style={{ marginBottom: 16 }}>How it works</h2>
            <p className="body-lg" style={{ maxWidth: 480, margin: '0 auto' }}>
              From prompt to annotated response in under 3 seconds.
            </p>
          </div>
        </FadeUp>

        {/* Horizontal pipeline */}
        <FadeIn delay={0.1}>
          <div style={{ overflowX: 'auto', paddingBottom: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 0, minWidth: 800, justifyContent: 'center' }}>
              {steps.map((step, i) => (
                <React.Fragment key={step.label}>
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.08, duration: 0.4 }}
                    style={{ textAlign: 'center', width: 130 }}
                  >
                    <div className="glass" style={{
                      width: 56, height: 56, borderRadius: 16, margin: '0 auto 12px',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: '1.5rem',
                      boxShadow: '0 4px 16px rgba(0,0,0,0.2)',
                    }}>
                      {step.icon}
                    </div>
                    <p style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>{step.label}</p>
                    <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>{step.desc}</p>
                  </motion.div>

                  {i < steps.length - 1 && (
                    <motion.div
                      initial={{ scaleX: 0 }}
                      whileInView={{ scaleX: 1 }}
                      viewport={{ once: true }}
                      transition={{ delay: i * 0.08 + 0.2, duration: 0.4 }}
                      style={{
                        flex: 1, height: 1, minWidth: 24,
                        background: 'linear-gradient(90deg, rgba(99,102,241,0.4), rgba(168,85,247,0.4))',
                        transformOrigin: 'left',
                        position: 'relative',
                      }}
                    >
                      <div style={{
                        position: 'absolute', right: -4, top: -3,
                        width: 7, height: 7, borderRadius: '50%',
                        background: 'var(--purple)',
                        boxShadow: '0 0 8px var(--purple)',
                      }} />
                    </motion.div>
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>
        </FadeIn>

        {/* Formula callout */}
        <FadeUp delay={0.2}>
          <div style={{
            marginTop: 56, textAlign: 'center',
            padding: '24px 32px', borderRadius: 16,
            background: 'rgba(99,102,241,0.06)',
            border: '1px solid rgba(99,102,241,0.15)',
            display: 'inline-block',
          }}>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 8, letterSpacing: '0.06em', textTransform: 'uppercase' }}>H-Score Formula</p>
            <p style={{
              fontFamily: 'var(--font-jetbrains-mono), monospace',
              fontSize: 'clamp(1rem, 2.5vw, 1.4rem)',
              fontWeight: 500, letterSpacing: '0.02em',
              color: '#a5b4fc',
            }}>
              H = α·FE + β·CG + γ·CF
            </p>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 8 }}>
              Weights α, β, γ optimized via gradient descent on benchmark datasets
            </p>
          </div>
        </FadeUp>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════ BENEFITS ════════════════════════════ */

function Benefits() {
  const items = [
    { icon: Zap,      label: 'Real-Time',        desc: 'Stream first, annotate async. No waiting.',    color: '#f59e0b' },
    { icon: ShieldCheck, label: 'Evidence-Backed', desc: 'Every claim linked to a real source snippet.', color: '#22c55e' },
    { icon: Globe,    label: 'Model Agnostic',    desc: 'Works with Gemini, OpenAI, Llama, and more.',  color: '#6366f1' },
    { icon: Lock,     label: 'Secure by Design',  desc: 'JWT auth, RLS isolation, no plain-text secrets.', color: '#a855f7' },
    { icon: Layers,   label: 'Clean Architecture',desc: 'Domain-driven, modular, independently testable.',color: '#3b82f6' },
    { icon: BookOpen, label: 'Research-Grade',    desc: 'Evaluated on TruthfulQA, HaluEval, and FEVER.', color: '#ec4899' },
  ];

  return (
    <section className="section">
      <div className="container">
        <FadeUp>
          <div style={{ textAlign: 'center', marginBottom: 60 }}>
            <p className="label" style={{ marginBottom: 12 }}>Why HalluciSense</p>
            <h2 className="display-2" style={{ marginBottom: 0 }}>Built for production.</h2>
          </div>
        </FadeUp>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          {items.map(({ icon: Icon, label, desc, color }, i) => (
            <FadeUp key={label} delay={i * 0.07}>
              <div style={{
                display: 'flex', alignItems: 'flex-start', gap: 16,
                padding: '20px 24px', borderRadius: 14,
                border: '1px solid var(--border)',
                background: 'var(--surface)',
                transition: 'border-color 0.2s, background 0.2s',
              }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-hover)';
                  (e.currentTarget as HTMLElement).style.background = 'var(--surface-hover)';
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = 'var(--border)';
                  (e.currentTarget as HTMLElement).style.background = 'var(--surface)';
                }}
              >
                <div style={{
                  width: 38, height: 38, borderRadius: 10,
                  background: `${color}18`, border: `1px solid ${color}30`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0
                }}>
                  <Icon size={18} color={color} />
                </div>
                <div>
                  <p style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: 4 }}>{label}</p>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>{desc}</p>
                </div>
              </div>
            </FadeUp>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════ CTA BANNER ══════════════════════════ */

function CTABanner() {
  return (
    <section className="section-sm" style={{ padding: '60px 24px' }}>
      <div className="container">
        <FadeUp>
          <div style={{
            padding: '56px 48px', borderRadius: 24,
            background: 'linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(168,85,247,0.08) 100%)',
            border: '1px solid rgba(99,102,241,0.2)',
            textAlign: 'center',
            position: 'relative', overflow: 'hidden',
          }}>
            <div style={{
              position: 'absolute', top: -80, left: '50%', transform: 'translateX(-50%)',
              width: 400, height: 200,
              background: 'radial-gradient(ellipse, rgba(99,102,241,0.2) 0%, transparent 70%)',
              pointerEvents: 'none',
            }} />
            <h2 className="display-2" style={{ marginBottom: 16, position: 'relative' }}>
              Start verifying today.
            </h2>
            <p className="body-lg" style={{ maxWidth: 420, margin: '0 auto 36px', position: 'relative' }}>
              Join researchers and engineers building with AI you can actually trust.
            </p>
            <div style={{ display: 'flex', justifyContent: 'center', gap: 12, flexWrap: 'wrap', position: 'relative' }}>
              <Link href="/login" className="btn btn-primary btn-lg">
                Get Started Free <ArrowRight size={16} />
              </Link>
              <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="btn btn-ghost btn-lg"
                style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                <GithubIcon size={16} /> View on GitHub
              </a>
            </div>
          </div>
        </FadeUp>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════ FOOTER ══════════════════════════════ */

function Footer() {
  return (
    <footer style={{
      borderTop: '1px solid var(--border)',
      padding: '32px 24px',
      position: 'relative', zIndex: 1,
    }}>
      <div className="container" style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexWrap: 'wrap', gap: 16,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 7,
            background: 'rgba(99,102,241,0.15)',
            border: '1px solid rgba(99,102,241,0.25)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Database size={14} color="#818cf8" />
          </div>
          <span style={{ fontWeight: 600, fontSize: '0.875rem', color: 'var(--text-secondary)' }}>HalluciSense</span>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>© 2025</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          {['GitHub', 'Documentation', 'Privacy', 'Terms', 'Contact'].map(link => (
            <a key={link} href="#" className="nav-link" style={{ fontSize: '0.8rem' }}>{link}</a>
          ))}
        </div>
      </div>
    </footer>
  );
}

/* ═══════════════════════════════════ PAGE ════════════════════════════════ */

export default function LandingPage() {
  return (
    <>
      {/* Mesh background */}
      <div className="mesh-bg" aria-hidden>
        <div className="mesh-orb mesh-orb-1" />
        <div className="mesh-orb mesh-orb-2" />
        <div className="mesh-orb mesh-orb-3" />
      </div>

      <Navbar />
      <main style={{ position: 'relative', zIndex: 1 }}>
        <HeroSection />
        <div className="divider" style={{ margin: '0 24px' }} />
        <FeatureCards />
        <div className="divider" style={{ margin: '0 24px' }} />
        <HowItWorks />
        <div className="divider" style={{ margin: '0 24px' }} />
        <Benefits />
        <CTABanner />
      </main>
      <Footer />
    </>
  );
}
