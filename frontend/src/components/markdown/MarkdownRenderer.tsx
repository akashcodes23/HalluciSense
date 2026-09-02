'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { Copy, Check, Code2 } from 'lucide-react';
import toast from 'react-hot-toast';

interface MarkdownRendererProps {
  content: string;
}

function CodeBlock({ children, className }: { children: React.ReactNode; className?: string }) {
  const [copied, setCopied] = useState(false);
  const match = /language-(\w+)/.exec(className || '');
  const lang = match ? match[1] : 'code';
  const codeText = String(children).replace(/\n$/, '');

  const handleCopy = () => {
    navigator.clipboard.writeText(codeText);
    setCopied(true);
    toast.success('Code copied to clipboard');
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group my-4 rounded-xl overflow-hidden bg-slate-950 border border-white/10 shadow-xl">
      {/* Code Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-white/[0.04] border-b border-white/5 font-mono text-xs text-slate-400">
        <span className="flex items-center gap-1.5 font-semibold text-slate-300">
          <Code2 className="w-3.5 h-3.5 text-teal-400" /> {lang}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-white px-2 py-1 rounded bg-white/5 hover:bg-white/10 transition-all cursor-pointer"
        >
          {copied ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>

      {/* Code Body */}
      <pre className="p-4 overflow-x-auto text-xs font-mono text-slate-200 leading-relaxed">
        <code className={className}>{children}</code>
      </pre>
    </div>
  );
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="markdown-body prose prose-invert max-w-none text-slate-200 leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          code({ inline, className, children, ...props }: React.HTMLAttributes<HTMLElement> & { inline?: boolean }) {
            if (inline) {
              return (
                <code className="bg-white/10 text-teal-300 px-1.5 py-0.5 rounded font-mono text-[13px]" {...props}>
                  {children}
                </code>
              );
            }
            return <CodeBlock className={className}>{children}</CodeBlock>;
          },
          table({ children }) {
            return (
              <div className="overflow-x-auto my-4 rounded-xl border border-white/10">
                <table className="min-w-full divide-y divide-white/10 text-xs text-left">{children}</table>
              </div>
            );
          },
          thead({ children }) {
            return <thead className="bg-white/[0.05] font-semibold text-slate-300 uppercase tracking-wider">{children}</thead>;
          },
          th({ children }) {
            return <th className="px-4 py-3">{children}</th>;
          },
          td({ children }) {
            return <td className="px-4 py-3 border-t border-white/5">{children}</td>;
          },
          blockquote({ children }) {
            return (
              <blockquote className="border-l-4 border-teal-500/50 bg-teal-500/[0.03] pl-4 py-2 italic text-slate-300 my-3 rounded-r-xl">
                {children}
              </blockquote>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
