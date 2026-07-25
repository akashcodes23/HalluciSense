import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, User } from 'lucide-react';
import { motion } from 'framer-motion';
import { AnnotatedResponse } from './AnnotatedResponse';
import { VerificationReport } from '../../stores/uiStore';

interface MessageProps {
  id: string;
  role: 'USER' | 'ASSISTANT' | 'SYSTEM';
  content: string;
  verification_status?: 'PENDING' | 'PROCESSING' | 'COMPLETE' | 'FAILED';
  verification_report?: any;
}

const statusStyles: Record<string, string> = {
  COMPLETE:   'bg-green-500/20 text-green-400',
  PROCESSING: 'bg-yellow-500/20 text-yellow-400 animate-pulse',
  PENDING:    'bg-slate-500/20 text-slate-400',
  FAILED:     'bg-red-500/20 text-red-400',
};

export function MessageBubble({ id, role, content, verification_status, verification_report }: MessageProps) {
  const isUser = role === 'USER';
  const isAnnotated = !isUser && verification_status === 'COMPLETE' && verification_report;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-6 p-8 rounded-[24px] mb-8 ${
        isUser ? 'bg-white/[0.02] border border-white/5' : ''
      }`}
    >
      {/* Avatar */}
      <div className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 ${
        isUser
          ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30'
          : 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
      }`}>
        {isUser ? <User className="w-4.5 h-4.5" /> : <Bot className="w-4.5 h-4.5" />}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3 mb-3">
          <span className="font-semibold text-xs text-slate-300 tracking-wider uppercase">
            {isUser ? 'You' : 'HalluciSense Engine'}
          </span>
          {!isUser && verification_status && (
            <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full badge-in ${statusStyles[verification_status]}`}>
              {verification_status === 'COMPLETE' ? '✓ Verified' : verification_status}
            </span>
          )}
        </div>

        {/* Render annotated or plain markdown */}
        {isAnnotated ? (
          <AnnotatedResponse messageId={id} report={verification_report} />
        ) : (
          <div className="markdown-body prose prose-invert max-w-none text-slate-300">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {content || (verification_status === 'PROCESSING' ? '…' : '')}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </motion.div>
  );
}
