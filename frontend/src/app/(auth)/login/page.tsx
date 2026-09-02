'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import {
  Database, Mail, Lock, User,
  ArrowLeft, Eye, EyeOff, ShieldCheck, Activity, Zap
} from 'lucide-react';
import { authService } from '../../../services/authService';
import { useAuthStore } from '../../../stores/authStore';

/* ─── Inline SVGs ────────────────────────────────────────────────────────── */
function GithubIcon({ size = 20, color = 'currentColor' }: { size?: number; color?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24"
      fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.2c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
      <path d="M9 18c-4.51 2-5-2-7-2" />
    </svg>
  );
}

function GoogleIcon({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
      <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
      <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
      <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
    </svg>
  );
}

/* ─── Shared Components ──────────────────────────────────────────────────── */

const fieldVariants = {
  hidden: (dir: number) => ({ opacity: 0, x: dir * 10 }),
  visible: { opacity: 1, x: 0, transition: { type: 'spring' as const, damping: 25, stiffness: 300 } },
  exit: (dir: number) => ({ opacity: 0, x: dir * -10, transition: { duration: 0.15 } })
};

function Field({
  id, icon: Icon, label, type = 'text', placeholder, value, onChange, extra, customDir
}: {
  id: string; icon: React.ComponentType<{ size?: number; style?: React.CSSProperties }>; label: string; type?: string; placeholder: string;
  value: string; onChange: (e: React.ChangeEvent<HTMLInputElement>) => void; extra?: React.ReactNode;
  customDir: number;
}) {
  const [showPw, setShowPw] = useState(false);
  const isPassword = type === 'password';
  const inputType = isPassword ? (showPw ? 'text' : 'password') : type;

  return (
    <motion.div variants={fieldVariants} custom={customDir} style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <label htmlFor={id} style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-secondary)' }}>
          {label}
        </label>
        {extra}
      </div>
      <div style={{ position: 'relative' }}>
        <Icon size={16} style={{
          position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)',
          color: 'var(--text-muted)', pointerEvents: 'none',
        }} />
        <input
          id={id} type={inputType} required placeholder={placeholder} value={value} onChange={onChange}
          style={{
            width: '100%',
            padding: '12px 44px 12px 42px',
            background: 'rgba(255, 255, 255, 0.02)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: 12,
            color: 'var(--text-primary)',
            fontSize: '0.9rem',
            outline: 'none',
            transition: 'all 0.2s ease',
            boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.1)',
          }}
          onFocus={e => {
            e.target.style.borderColor = 'rgba(99, 102, 241, 0.5)';
            e.target.style.background = 'rgba(99, 102, 241, 0.03)';
            e.target.style.boxShadow = '0 0 0 3px rgba(99, 102, 241, 0.15)';
          }}
          onBlur={e => {
            e.target.style.borderColor = 'rgba(255, 255, 255, 0.08)';
            e.target.style.background = 'rgba(255, 255, 255, 0.02)';
            e.target.style.boxShadow = 'inset 0 1px 2px rgba(0,0,0,0.1)';
          }}
        />
        {isPassword && (
          <button
            type="button" onClick={() => setShowPw(!showPw)}
            style={{
              position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)',
              background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)',
              display: 'flex', padding: 4
            }}
          >
            {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
          </button>
        )}
      </div>
    </motion.div>
  );
}

function SocialButton({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <motion.button
      type="button"
      whileHover={{ backgroundColor: 'rgba(255,255,255,0.06)' }}
      whileTap={{ scale: 0.98 }}
      style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
        padding: '12px', background: 'rgba(255, 255, 255, 0.025)',
        border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: 12,
        color: 'var(--text-primary)', fontSize: '0.85rem', fontWeight: 600,
        cursor: 'pointer', transition: 'background-color 0.2s',
      }}
    >
      {icon}
      {label}
    </motion.button>
  );
}

function SubmitButton({ loading, label }: { loading: boolean; label: string }) {
  return (
    <motion.button
      type="submit" disabled={loading}
      whileHover={loading ? {} : { scale: 1.015, boxShadow: '0 8px 24px rgba(45,212,191,0.3)' }}
      whileTap={{ scale: 0.985 }}
      style={{
        width: '100%', padding: '14px', borderRadius: 12, border: 'none',
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
        background: 'linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%)',
        color: '#09090b', fontSize: '0.95rem', fontWeight: 600, cursor: loading ? 'default' : 'pointer',
        marginTop: 16, marginBottom: 24, boxShadow: '0 4px 14px rgba(45,212,191,0.2)',
        transition: 'box-shadow 0.2s',
      }}
    >
      {loading ? (
        <span style={{
          width: 16, height: 16, borderRadius: '50%', border: '2px solid rgba(0,0,0,0.3)',
          borderTopColor: '#09090b', animation: 'spin 0.7s linear infinite'
        }} />
      ) : label}
    </motion.button>
  );
}

/* ─── Main Auth Page ─────────────────────────────────────────────────────── */
export default function AuthPage() {
  const router = useRouter();
  const { login, fetchUser } = useAuthStore();
  
  const [mode, setMode] = useState<'login' | 'signup'>('login');
  const [dir, setDir] = useState(1);
  const [loading, setLoading] = useState(false);
  
  const [loginData, setL] = useState({ email: '', password: '' });
  const [signupData, setS] = useState({ full_name: '', email: '', password: '' });

  const switchTo = (next: 'login' | 'signup') => {
    if (next === mode) return;
    setDir(next === 'signup' ? 1 : -1);
    setMode(next);
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault(); setLoading(true);
    try {
      const res = await authService.login(loginData);
      login(res.tokens.access_token, res.tokens.refresh_token, res.user);
      await fetchUser();
      toast.success('Welcome back!');
      router.push('/dashboard');
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      const errMsg = typeof detail === 'string' ? detail : (Array.isArray(detail) ? detail[0]?.msg : 'Invalid credentials.');
      toast.error(errMsg);
    } finally { setLoading(false); }
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault(); setLoading(true);
    try {
      await authService.register(signupData);
      toast.success('Account created! Sign in to continue.');
      switchTo('login');
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      const errMsg = typeof detail === 'string' ? detail : (Array.isArray(detail) ? detail[0]?.msg : 'Failed to create account.');
      toast.error(errMsg);
    } finally { setLoading(false); }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', background: '#05070f', color: 'var(--text-primary)', overflow: 'hidden' }}>
      
      {/* ── Animated Background ── */}
      <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0, overflow: 'hidden' }}>
        <div className="ambient-glow glow-1" />
        <div className="ambient-glow glow-2" />
        <div className="ambient-glow glow-3" />
      </div>

      {/* ── Left Branding Area (Desktop Only) ── */}
      <div className="auth-brand" style={{
        flex: '0 0 50%', display: 'flex', flexDirection: 'column', padding: '64px 80px',
        position: 'relative', zIndex: 1
      }}>
        {/* Top Logo */}
        <Link href="/" style={{
          display: 'inline-flex', alignItems: 'center', gap: 12, textDecoration: 'none'
        }}>
          <div style={{
            width: 40, height: 40, borderRadius: 12, background: 'rgba(45,212,191,0.1)',
            border: '1px solid rgba(45,212,191,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <Database size={20} color="#2dd4bf" />
          </div>
          <span style={{ fontWeight: 700, fontSize: '1.2rem', color: 'var(--text-primary)' }}>
            HalluciSense
          </span>
        </Link>
        
        {/* Centered Message */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <h1 style={{
            fontSize: 'clamp(2.8rem, 5.5vw, 4.2rem)', fontWeight: 700, letterSpacing: '-0.04em',
            lineHeight: 1.05, color: 'var(--text-primary)', maxWidth: 600, marginBottom: 24
          }}>
            Know when to <br />
            <span style={{ color: '#2dd4bf', fontWeight: 600 }}>trust</span> your AI.
          </h1>
          <p style={{ fontSize: '1.1rem', color: 'var(--text-secondary)', maxWidth: 480, lineHeight: 1.6, marginBottom: 56 }}>
            Intercept every LLM response and analyze it across three independent dimensions before a single word reaches you.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>
            {[
              { icon: ShieldCheck, title: 'Factual Grounding', desc: 'Cross-references claims against verified sources.' },
              { icon: Activity, title: 'Confidence Analysis', desc: 'Token-level entropy scoring detects uncertainty.' },
              { icon: Zap, title: 'Semantic Consistency', desc: 'Multi-sample generation validates coherence.' },
            ].map((f, i) => (
              <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 + i * 0.1 }} style={{ display: 'flex', alignItems: 'flex-start', gap: 24 }}>
                <div style={{ width: 48, height: 48, borderRadius: 12, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <f.icon size={24} color="var(--text-primary)" />
                </div>
                <div>
                  <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>{f.title}</h3>
                  <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>{f.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
      
      {/* ── Right Auth Area ── */}
      <div className="auth-right-panel" style={{
        flex: '0 0 50%', display: 'flex', flexDirection: 'column', position: 'relative', zIndex: 1,
        alignItems: 'flex-start', justifyContent: 'center', padding: '24px 64px'
      }}>
        
        {/* Mobile Logo & Back Link */}
        <div style={{ position: 'absolute', top: 32, left: 32, right: 32, display: 'flex', justifyContent: 'space-between' }}>
          <Link href="/" className="auth-back-link" style={{
            display: 'inline-flex', alignItems: 'center', gap: 8, textDecoration: 'none',
            color: 'var(--text-muted)', fontSize: '0.9rem', fontWeight: 500
          }}
            onMouseEnter={e => e.currentTarget.style.color = 'var(--text-primary)'}
            onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
          >
            <ArrowLeft size={18} /> Back
          </Link>
          <Link href="/" className="auth-mobile-logo" style={{
            display: 'none', alignItems: 'center', gap: 8, textDecoration: 'none'
          }}>
            <Database size={24} color="#2dd4bf" />
            <span style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--text-primary)' }}>HalluciSense</span>
          </Link>
        </div>

        {/* Auth Floating Card */}
        <motion.div
          layout
          style={{
            width: '100%', maxWidth: 440,
            background: 'rgba(255,255,255,0.015)',
            backdropFilter: 'blur(40px)',
            WebkitBackdropFilter: 'blur(40px)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 24,
            padding: '40px',
            boxShadow: '0 32px 100px -16px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1)',
            overflow: 'hidden'
          }}
        >
          {/* Segmented Control */}
          <div style={{
            display: 'flex', background: 'rgba(0,0,0,0.2)', padding: 4, borderRadius: 14, marginBottom: 32,
            border: '1px solid rgba(255,255,255,0.05)'
          }}>
            {(['login', 'signup'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => switchTo(tab)}
                style={{
                  flex: 1, position: 'relative', padding: '10px',
                  background: 'none', border: 'none', cursor: 'pointer',
                  color: mode === tab ? 'var(--text-primary)' : 'var(--text-muted)',
                  fontSize: '0.85rem', fontWeight: 600,
                  transition: 'color 0.2s ease', zIndex: 1
                }}
              >
                {mode === tab && (
                  <motion.div
                    layoutId="auth-tab-active"
                    style={{
                      position: 'absolute', inset: 0,
                      background: 'rgba(255,255,255,0.1)',
                      borderRadius: 10,
                      boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                      border: '1px solid rgba(255,255,255,0.08)'
                    }}
                    transition={{ type: 'spring', damping: 20, stiffness: 300 }}
                  />
                )}
                <span style={{ position: 'relative', zIndex: 2 }}>
                  {tab === 'login' ? 'Sign In' : 'Create Account'}
                </span>
              </button>
            ))}
          </div>

          <div style={{ position: 'relative' }}>
            <AnimatePresence mode="wait" custom={dir} initial={false}>
              {mode === 'login' ? (
                <motion.div
                  key="login" custom={dir}
                  initial="hidden"
                  animate="visible"
                  exit="exit"
                  variants={{
                    hidden: (d: number) => ({ opacity: 0, x: d * 20 }),
                    visible: { opacity: 1, x: 0, transition: { type: 'spring' as const, damping: 25, stiffness: 300 } },
                    exit: (d: number) => ({ opacity: 0, x: d * -20, transition: { duration: 0.15 } })
                  }}
                >
                  <form onSubmit={handleLogin}>
                    <Field
                      id="login-email" icon={Mail} label="Email address" type="email" placeholder="name@company.com"
                      value={loginData.email} onChange={e => setL({ ...loginData, email: e.target.value })}
                      customDir={dir}
                    />
                    <Field
                      id="login-password" icon={Lock} label="Password" type="password" placeholder="••••••••"
                      value={loginData.password} onChange={e => setL({ ...loginData, password: e.target.value })}
                      customDir={dir}
                      extra={
                        <a href="#" style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', textDecoration: 'none' }}>
                          Forgot password?
                        </a>
                      }
                    />
                    <motion.div variants={fieldVariants} custom={dir}>
                      <SubmitButton loading={loading} label="Sign In" />
                    </motion.div>
                  </form>

                  <motion.div variants={fieldVariants} custom={dir}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
                      <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.06)' }} />
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>or</span>
                      <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.06)' }} />
                    </div>

                    <div style={{ display: 'flex', gap: 12 }}>
                      <SocialButton icon={<GoogleIcon size={18} />} label="Google" />
                      <SocialButton icon={<GithubIcon size={18} color="var(--text-secondary)" />} label="GitHub" />
                    </div>
                  </motion.div>
                </motion.div>
              ) : (
                <motion.div
                  key="signup" custom={dir}
                  initial="hidden"
                  animate="visible"
                  exit="exit"
                  variants={{
                    hidden: (d: number) => ({ opacity: 0, x: d * 20 }),
                    visible: { opacity: 1, x: 0, transition: { type: 'spring' as const, damping: 25, stiffness: 300 } },
                    exit: (d: number) => ({ opacity: 0, x: d * -20, transition: { duration: 0.15 } })
                  }}
                >
                  <form onSubmit={handleSignup}>
                    <Field
                      id="signup-name" icon={User} label="Full name" placeholder="Jane Doe"
                      value={signupData.full_name} onChange={e => setS({ ...signupData, full_name: e.target.value })}
                      customDir={dir}
                    />
                    <Field
                      id="signup-email" icon={Mail} label="Email address" type="email" placeholder="name@company.com"
                      value={signupData.email} onChange={e => setS({ ...signupData, email: e.target.value })}
                      customDir={dir}
                    />
                    <Field
                      id="signup-password" icon={Lock} label="Password" type="password" placeholder="Min. 8 characters"
                      value={signupData.password} onChange={e => setS({ ...signupData, password: e.target.value })}
                      customDir={dir}
                    />
                    <motion.div variants={fieldVariants} custom={dir}>
                      <SubmitButton loading={loading} label="Create Account" />
                    </motion.div>
                  </form>

                  <motion.div variants={fieldVariants} custom={dir}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
                      <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.06)' }} />
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>or</span>
                      <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.06)' }} />
                    </div>

                    <div style={{ display: 'flex', gap: 12 }}>
                      <SocialButton icon={<GoogleIcon size={18} />} label="Google" />
                      <SocialButton icon={<GithubIcon size={18} color="var(--text-secondary)" />} label="GitHub" />
                    </div>
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>
      
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .auth-brand { display: flex; }
        .auth-mobile-logo { display: none; }
        
        .ambient-glow {
          position: absolute;
          border-radius: 50%;
          filter: blur(100px);
          opacity: 0.6;
          animation: pulse-glow 10s infinite alternate ease-in-out;
        }
        .glow-1 {
          top: -10%; left: -10%; width: 60%; height: 60%;
          background: radial-gradient(circle, rgba(45,212,191,0.08) 0%, transparent 60%);
        }
        .glow-2 {
          bottom: -10%; right: -10%; width: 50%; height: 50%;
          background: radial-gradient(circle, rgba(96,165,250,0.06) 0%, transparent 60%);
          animation-delay: -5s;
        }
        .glow-3 {
          top: 40%; left: 40%; width: 40%; height: 40%;
          background: radial-gradient(circle, rgba(34,197,94,0.04) 0%, transparent 60%);
          animation-delay: -2s;
        }
        
        @keyframes pulse-glow {
          0% { transform: scale(1) translate(0, 0); opacity: 0.5; }
          100% { transform: scale(1.1) translate(20px, -20px); opacity: 0.8; }
        }
        
        @media (max-width: 1000px) {
          .auth-right-panel {
            flex: 1 !important;
            align-items: center !important;
            padding: 24px !important;
          }
          .auth-back-link {
            display: none !important;
          }
        }
        
        @media (max-width: 900px) {
          .auth-brand { display: none !important; }
          .auth-mobile-logo { display: inline-flex !important; }
        }
      `}</style>
    </div>
  );
}
