'use client';

import React, { useState } from 'react';
import { Bot, User, ShieldCheck, ShieldAlert, ShieldX, Copy, Check, Sparkles, Clock, Cpu } from 'lucide-react';
import { motion } from 'framer-motion';
import { AnnotatedResponse } from './AnnotatedResponse';
import { VerificationProgress } from '../verification/VerificationProgress';
import { MarkdownRenderer } from '../markdown/MarkdownRenderer';
import { useUIStore } from '../../stores/uiStore';
import toast from 'react-hot-toast';

interface MessageProps {
  id: string;
  role: 'USER' | 'ASSISTANT' | 'SYSTEM';
  content: string;
  verification_status?: 'PENDING' | 'PROCESSING' | 'COMPLETE' | 'FAILED';
  verification_report?: any;
  model_used?: string;
  processing_time_ms?: number;
}

const statusConfig = {
  COMPLETE: { label: 'Verified', bg: 'bg-green-500/10', text: 'text-green-400', border: 'border-green-500/30', icon: ShieldCheck },
  PROCESSING: { label: 'Analyzing', bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/30', icon: Sparkles },
  PENDING: { label: 'Queued', bg: 'bg-slate-500/10', text: 'text-slate-400', border: 'border-slate-500/30', icon: Clock },
  FAILED: { label: 'Unverified', bg: 'bg-red-500/10', text: 'text-red-400', border: 'border-red-500/30', icon: ShieldX },
};

export function MessageBubble({
  id,
  role,
  content,
  verification_status,
  verification_report,
  model_used = 'gemini-3.1-pro',
  processing_time_ms,
}: MessageProps) {
  const [copied, setCopied] = useState(false);
  const { openPanel } = useUIStore();
  const isUser = role === 'USER';
  const isAnnotated = !isUser && verification_status === 'COMPLETE' && verification_report;

  const status = statusConfig[verification_status || 'PENDING'] || statusConfig.PENDING;
  const StatusIcon = status.icon;

  const handleCopyMessage = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    toast.success('Message copied to clipboard');
    setTimeout(() => setCopied(false), 2000);
  };

  const handleInspect = () => {
    if (verification_report) {
      openPanel(id, verification_report, 0);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-5 p-6 rounded-3xl mb-6 transition-all ${
        isUser
          ? 'bg-white/[0.02] border border-white/5'
          : 'bg-white/[0.01] border border-white/5 hover:border-white/10'
      }`}
    >
      {/* Avatar */}
      <div
        className={`w-9 h-9 rounded-2xl flex items-center justify-center shrink-0 ${
          isUser
            ? 'bg-slate-500/15 text-slate-300 border border-slate-500/20'
            : 'bg-teal-500/15 text-teal-400 border border-teal-500/25'
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      {/* Main Body */}
      <div className="flex-1 min-w-0 space-y-3">
        {/* Header Bar */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-bold text-xs text-slate-200 tracking-wider uppercase">
              {isUser ? 'You' : 'HalluciSense Engine'}
            </span>

            {!isUser && (
              <span className="flex items-center gap-1 text-[10px] font-mono font-medium text-slate-400 bg-white/5 px-2 py-0.5 rounded-full border border-white/10">
                <Cpu className="w-3 h-3 text-teal-400" /> {model_used}
              </span>
            )}

            {!isUser && verification_status && (
              <span
                className={`flex items-center gap-1 text-[10px] uppercase font-bold px-2 py-0.5 rounded-full border ${status.bg} ${status.text} ${status.border}`}
              >
                <StatusIcon className="w-3 h-3" /> {status.label}
              </span>
            )}

            {!isUser && processing_time_ms && (
              <span className="text-[10px] text-slate-400 font-mono">
                {(processing_time_ms / 1000).toFixed(1)}s
              </span>
            )}
          </div>

          {/* Quick Actions */}
          <div className="flex items-center gap-2">
            {!isUser && isAnnotated && (
              <button
                onClick={handleInspect}
                className="flex items-center gap-1 text-xs font-semibold text-teal-400 hover:text-teal-300 bg-teal-500/10 hover:bg-teal-500/20 px-2.5 py-1 rounded-xl border border-teal-500/30 transition-all cursor-pointer"
              >
                <ShieldCheck className="w-3.5 h-3.5" /> Inspect Verification
              </button>
            )}
            <button
              onClick={handleCopyMessage}
              title="Copy response"
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-colors"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>

        {/* Live Stepper Progress during PROCESSING */}
        {!isUser && verification_status === 'PROCESSING' && (
          <VerificationProgress />
        )}

        {/* Content Body */}
        {isAnnotated ? (
          <AnnotatedResponse messageId={id} report={verification_report} />
        ) : (
          <MarkdownRenderer content={content || (verification_status === 'PROCESSING' ? '…' : '')} />
        )}
      </div>
    </motion.div>
  );
}
