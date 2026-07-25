import React, { useRef, useEffect, useState } from 'react';
import { MessageBubble } from './MessageBubble';
import { InputBar } from './InputBar';
import { Download } from 'lucide-react';
import { useParams } from 'next/navigation';

import { Message } from '@/types/api';
import { useChatStore } from '../../stores/chatStore';

interface ChatWindowProps {
  messages: Message[];
  onSendMessage: (msg: string) => void;
  isLoading: boolean;
}

export function ChatWindow({ messages, onSendMessage, isLoading }: ChatWindowProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { isStreaming, streamingContent } = useChatStore();
  const [exportOpen, setExportOpen] = useState(false);
  const params = useParams();
  const chatId = params.id as string;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading, streamingContent]);

  const handleExport = (format: 'json' | 'md') => {
    setExportOpen(false);
    const token = localStorage.getItem('halucisense_token'); // Or via store if accessible
    window.open(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'}/export/chats/${chatId}?format=${format}&token=${token || ''}`, '_blank');
  };

  return (
    <div className="flex flex-col h-full relative z-10 w-full max-w-5xl mx-auto pt-6">
      
      {/* Export Header */}
      {messages.length > 0 && (
        <div className="absolute top-4 right-8 z-50">
          <div className="relative">
            <button 
              onClick={() => setExportOpen(!exportOpen)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-slate-300 hover:text-white hover:bg-white/10 transition-colors text-sm font-medium"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
            {exportOpen && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setExportOpen(false)}></div>
                <div className="absolute right-0 top-10 w-32 bg-[#1e1e24] border border-white/10 rounded-lg shadow-xl z-50 overflow-hidden py-1">
                  <button onClick={() => handleExport('md')} className="w-full text-left px-4 py-2 text-sm text-slate-300 hover:bg-white/5 hover:text-white">Markdown</button>
                  <button onClick={() => handleExport('json')} className="w-full text-left px-4 py-2 text-sm text-slate-300 hover:bg-white/5 hover:text-white">JSON</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-4 py-8 custom-scrollbar">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center opacity-60 text-center px-8">
            <h2 className="text-2xl font-semibold mb-2">How can I help you today?</h2>
            <p className="text-slate-400 text-sm max-w-md">
              HalluciSense verifies answers across three dimensions: factual grounding, confidence, and semantic consistency.
              <br />Click any highlighted sentence to inspect it.
            </p>
          </div>
        ) : (
          messages.map((msg, idx) => {
            const isLastMessage = idx === messages.length - 1;
            const displayContent = (isLastMessage && isStreaming && msg.role === 'ASSISTANT') 
              ? streamingContent 
              : msg.content;
            
            return (
              <MessageBubble
                key={msg.id || idx}
                id={msg.id}
                role={msg.role}
                content={displayContent}
                verification_status={msg.verification_status}
                verification_report={msg.verification_report}
              />
            );
          })
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="pb-8 pt-4 bg-gradient-to-t from-[var(--bg)] to-transparent">
        <InputBar onSend={onSendMessage} isLoading={isLoading} />
      </div>
    </div>
  );
}
