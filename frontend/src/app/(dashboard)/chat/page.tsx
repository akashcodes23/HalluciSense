"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Sparkles,
  ShieldCheck,
  AlertTriangle,
  Info,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FileText,
  Clock,
  Layers,
  ArrowRight,
  RefreshCw,
  HelpCircle,
  ExternalLink,
} from "lucide-react";

interface VerificationSummary {
  status: "VERIFIED" | "CORRECTED" | "REVIEW" | "FAILED" | "UNVERIFIED";
  h_score?: number;
  risk_level?: string;
  claims_total?: number;
  claims_flagged?: number;
  error_message?: string;
}

interface ClaimCorrection {
  claim_id: string;
  original_claim: string;
  corrected_claim: string;
  error_type: string;
  evidence_basis: string;
}

interface CorrectionSummary {
  performed: boolean;
  reason: string;
  claims_corrected: ClaimCorrection[];
  original_to_corrected: Array<{ original: string; corrected: string }>;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  original_content?: string;
  verification?: VerificationSummary;
  correction?: CorrectionSummary;
  evidence?: Array<{ source_name: string; snippet: string; claim?: string }>;
  sources?: string[];
  latency_ms?: number;
  trace_id?: string;
  timestamp: string;
}

const SAMPLE_QUESTIONS = [
  "What is the speed of light in vacuum?",
  "What is the standard atmospheric pressure at sea level?",
  "What is the molar mass of water?",
  "In which direction does eukaryotic DNA replication proceed?",
  "What causes Type 1 diabetes mellitus?",
];

const PIPELINE_STAGES = [
  "Generating draft response...",
  "Retrieving authoritative evidence (BM25 + Dense FAISS)...",
  "Decomposing response into atomic propositions...",
  "Evaluating NLI entailment & symbolic scientific checks...",
  "Synthesizing evidence-grounded repair...",
  "Running independent re-verification gate...",
];

export default function ClosedLoopChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputQuery, setInputQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentStageIdx, setCurrentStageIdx] = useState(0);
  const [expandedDetails, setExpandedDetails] = useState<Record<string, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSendMessage = async (queryText?: string) => {
    const text = queryText || inputQuery;
    if (!text.trim() || isLoading) return;

    const userMsgId = `user_${Date.now()}`;
    const newMsg: ChatMessage = {
      id: userMsgId,
      role: "user",
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, newMsg]);
    setInputQuery("");
    setIsLoading(true);
    setCurrentStageIdx(0);

    // Multi-stage visual progression
    const stageInterval = setInterval(() => {
      setCurrentStageIdx((prev) => {
        if (prev < PIPELINE_STAGES.length - 1) return prev + 1;
        return prev;
      });
    }, 450);

    try {
      const response = await fetch("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          enable_verification: true,
          auto_correct: true,
        }),
      });

      clearInterval(stageInterval);

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      const assistantMsg: ChatMessage = {
        id: data.message_id || `asst_${Date.now()}`,
        role: "assistant",
        content: data.final_response,
        original_content: data.original_response,
        verification: data.verification,
        correction: data.correction,
        evidence: data.evidence,
        sources: data.sources,
        latency_ms: data.latency_ms,
        trace_id: data.trace_id,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      clearInterval(stageInterval);
      const errorMsg: ChatMessage = {
        id: `asst_err_${Date.now()}`,
        role: "assistant",
        content: "Verification could not be completed because the verification service encountered an internal error.",
        verification: {
          status: "FAILED",
          error_message: "Verification could not be completed because the verification service encountered an internal error.",
        },
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleDetails = (msgId: string) => {
    setExpandedDetails((prev) => ({ ...prev, [msgId]: !prev[msgId] }));
  };

  return (
    <div className="flex flex-col h-screen bg-[#050816] text-slate-100 font-sans">
      {/* ── Top Header ─────────────────────────────────────────────────── */}
      <header className="px-6 py-4 border-b border-white/[0.06] bg-[#060a14]/80 backdrop-blur-md flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-accent-primary/15 border border-accent-primary/30 text-accent-primary shadow-[0_0_15px_rgba(99,102,241,0.2)]">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-white tracking-tight">HalluciSense Chat</h1>
              <span className="px-2 py-0.5 text-[10px] font-semibold bg-accent-primary/20 text-accent-primary border border-accent-primary/40 rounded-full">
                Closed-Loop v1.0
              </span>
            </div>
            <p className="text-xs text-slate-400">Ask anything. HalluciSense verifies before it trusts.</p>
          </div>
        </div>

        <div className="hidden sm:flex items-center gap-4 text-xs text-slate-400 font-mono">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>P1 Hybrid Active</span>
          </div>
          <div className="flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-accent-primary" />
            <span>Re-Verification Gating</span>
          </div>
        </div>
      </header>

      {/* ── Main Chat Scroll Area ───────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto text-center space-y-6 py-12">
            <div className="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/[0.08] flex items-center justify-center text-accent-primary shadow-2xl">
              <ShieldCheck className="w-8 h-8" />
            </div>
            <div className="space-y-2">
              <h2 className="text-xl font-bold text-white tracking-tight">Evidence-Grounded AI Answer System</h2>
              <p className="text-sm text-slate-400 max-w-md">
                Every response is decomposed into atomic claims, checked against authoritative scientific evidence, and automatically repaired if errors are detected.
              </p>
            </div>

            <div className="w-full space-y-2 text-left pt-4">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-1">
                Suggested Scientific Inquiries
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {SAMPLE_QUESTIONS.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSendMessage(q)}
                    className="p-3 text-xs text-slate-300 bg-white/[0.02] hover:bg-white/[0.06] border border-white/[0.06] hover:border-accent-primary/40 rounded-xl text-left transition-all duration-150 flex items-center justify-between group cursor-pointer"
                  >
                    <span className="truncate pr-2">{q}</span>
                    <ArrowRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-accent-primary shrink-0 transition-colors" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"} max-w-3xl ${
                msg.role === "user" ? "ml-auto" : "mr-auto"
              } w-full space-y-2`}
            >
              {/* Message Header */}
              <div className="flex items-center gap-2 px-1 text-[11px] font-mono text-slate-400">
                <span>{msg.role === "user" ? "USER" : "HALLUCISENSE"}</span>
                <span>•</span>
                <span>{msg.timestamp}</span>
                {msg.latency_ms && (
                  <>
                    <span>•</span>
                    <span>{msg.latency_ms.toFixed(1)}ms</span>
                  </>
                )}
              </div>

              {/* Message Bubble */}
              <div
                className={`p-4 rounded-2xl text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-accent-primary text-white font-medium rounded-tr-sm shadow-lg shadow-accent-primary/20"
                    : "bg-[#0b1222] border border-white/[0.08] text-slate-100 rounded-tl-sm shadow-xl w-full"
                }`}
              >
                {/* Assistant Verification Status Badge */}
                {msg.role === "assistant" && msg.verification && (
                  <div className="mb-3 flex items-center justify-between pb-3 border-b border-white/[0.06]">
                    <div className="flex items-center gap-2">
                      {msg.verification.status === "VERIFIED" && (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold bg-emerald-500/15 border border-emerald-500/30 text-emerald-400">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          VERIFIED
                        </span>
                      )}
                      {msg.verification.status === "CORRECTED" && (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold bg-amber-500/15 border border-amber-500/30 text-amber-400">
                          <AlertTriangle className="w-3.5 h-3.5" />
                          CORRECTED & RE-VERIFIED
                        </span>
                      )}
                      {msg.verification.status === "REVIEW" && (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold bg-indigo-500/15 border border-indigo-500/30 text-indigo-400">
                          <Info className="w-3.5 h-3.5" />
                          REQUIRES REVIEW
                        </span>
                      )}
                      {msg.verification.status === "FAILED" && (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold bg-rose-500/15 border border-rose-500/30 text-rose-400">
                          <AlertTriangle className="w-3.5 h-3.5" />
                          FAILED
                        </span>
                      )}
                      {msg.verification.h_score !== undefined && msg.verification.h_score !== null ? (
                        <span className="text-[11px] font-mono text-slate-400">
                          H-Score: {(msg.verification.h_score * 100).toFixed(1)}%
                        </span>
                      ) : (
                        <span className="text-[11px] font-mono text-slate-400">
                          Verification unavailable
                        </span>
                      )}
                    </div>

                    <button
                      onClick={() => toggleDetails(msg.id)}
                      className="text-xs text-accent-primary hover:text-accent-primary/80 font-medium flex items-center gap-1 cursor-pointer transition-colors"
                    >
                      <span>Verification Details</span>
                      {expandedDetails[msg.id] ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    </button>
                  </div>
                )}

                {/* Main Content */}
                <div className="space-y-3">
                  <p className="whitespace-pre-wrap">{msg.content}</p>

                  {/* Original vs Corrected Diff Panel (if correction occurred) */}
                  {msg.correction && msg.correction.performed && (
                    <div className="mt-3 p-3 rounded-xl bg-amber-500/[0.04] border border-amber-500/20 text-xs space-y-2">
                      <div className="font-semibold text-amber-400 flex items-center gap-1.5">
                        <AlertTriangle className="w-3.5 h-3.5" />
                        <span>Closed-Loop Repair Summary</span>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px]">
                        <div className="p-2 rounded-lg bg-black/40 border border-white/[0.04]">
                          <span className="text-slate-400 font-semibold block mb-1">ORIGINAL DRAFT:</span>
                          <span className="text-rose-300 line-through opacity-80">{msg.original_content}</span>
                        </div>
                        <div className="p-2 rounded-lg bg-emerald-950/20 border border-emerald-500/20">
                          <span className="text-emerald-400 font-semibold block mb-1">CORRECTED CLAIM:</span>
                          <span className="text-emerald-300 font-medium">{msg.content}</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Expandable Verification Details */}
                  <AnimatePresence>
                    {expandedDetails[msg.id] && msg.verification && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 pt-4 border-t border-white/[0.06] space-y-3 text-xs"
                      >
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px]">
                          <div className="p-2 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                            <span className="text-slate-500 block text-[9px] uppercase">Risk Level</span>
                            <span className="font-bold text-slate-200">{msg.verification.risk_level}</span>
                          </div>
                          <div className="p-2 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                            <span className="text-slate-500 block text-[9px] uppercase">Claims Total</span>
                            <span className="font-bold text-slate-200">{msg.verification.claims_total}</span>
                          </div>
                          <div className="p-2 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                            <span className="text-slate-500 block text-[9px] uppercase">Claims Flagged</span>
                            <span className="font-bold text-amber-400">{msg.verification.claims_flagged}</span>
                          </div>
                          <div className="p-2 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                            <span className="text-slate-500 block text-[9px] uppercase">Trace ID</span>
                            <span className="font-bold text-indigo-400 truncate block">{msg.trace_id || "N/A"}</span>
                          </div>
                        </div>

                        {/* Evidence Sources */}
                        {msg.evidence && msg.evidence.length > 0 && (
                          <div className="space-y-1.5 pt-2">
                            <span className="text-[11px] font-semibold text-slate-400 flex items-center gap-1.5">
                              <FileText className="w-3.5 h-3.5" />
                              <span>Retrieved Scientific Evidence</span>
                            </span>
                            <div className="space-y-1">
                              {msg.evidence.map((ev, eIdx) => (
                                <div
                                  key={eIdx}
                                  className="p-2 rounded-lg bg-white/[0.02] border border-white/[0.04] text-[11px] space-y-0.5"
                                >
                                  <span className="font-semibold text-accent-primary">{ev.source_name}</span>
                                  <p className="text-slate-300 italic">{ev.snippet}</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </div>
          ))
        )}

        {/* Real-time Multi-Stage Loading Indicator */}
        {isLoading && (
          <div className="flex flex-col items-start max-w-2xl mr-auto w-full space-y-2">
            <div className="flex items-center gap-2 px-1 text-[11px] font-mono text-slate-400">
              <span>HALLUCISENSE PIPELINE</span>
              <span>•</span>
              <span className="text-accent-primary font-bold animate-pulse">PROCESSING</span>
            </div>

            <div className="p-4 rounded-2xl bg-[#0b1222] border border-accent-primary/30 text-slate-100 rounded-tl-sm shadow-xl w-full space-y-3">
              <div className="flex items-center gap-3">
                <RefreshCw className="w-4 h-4 text-accent-primary animate-spin" />
                <span className="text-xs font-semibold text-white">
                  {PIPELINE_STAGES[currentStageIdx]}
                </span>
              </div>

              {/* Progress bar */}
              <div className="w-full h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-indigo-500 to-emerald-400 rounded-full"
                  initial={{ width: "10%" }}
                  animate={{ width: `${((currentStageIdx + 1) / PIPELINE_STAGES.length) * 100}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>

              <div className="flex items-center justify-between text-[10px] font-mono text-slate-500">
                <span>Stage {currentStageIdx + 1} of {PIPELINE_STAGES.length}</span>
                <span>P1 Hybrid + Symbolic Engine</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* ── Bottom Input Bar ────────────────────────────────────────────── */}
      <div className="p-4 md:p-6 border-t border-white/[0.06] bg-[#060a14]/90 backdrop-blur-lg shrink-0">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage();
          }}
          className="max-w-3xl mx-auto relative flex items-center"
        >
          <input
            type="text"
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            placeholder="Ask a scientific question..."
            disabled={isLoading}
            className="w-full bg-[#0b1222] border border-white/[0.1] focus:border-accent-primary rounded-xl pl-4 pr-12 py-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-accent-primary transition-all duration-150 shadow-inner"
          />
          <button
            type="submit"
            disabled={!inputQuery.trim() || isLoading}
            className="absolute right-2 p-2 rounded-lg bg-accent-primary hover:bg-accent-primary/90 text-white disabled:opacity-30 disabled:hover:bg-accent-primary transition-all cursor-pointer shadow-md"
            aria-label="Send query"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
        <p className="text-center text-[10px] text-slate-500 mt-2 font-mono">
          HalluciSense validates answers against peer-reviewed literature. Evidence-backed verification does not replace professional medical advice.
        </p>
      </div>
    </div>
  );
}
