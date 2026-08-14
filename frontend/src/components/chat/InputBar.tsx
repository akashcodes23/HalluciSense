import React, { useState, useRef, useEffect } from 'react';
import { Send, StopCircle, CheckCircle, MessageSquare, ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useChatStore } from '../../stores/chatStore';

interface InputBarProps {
  onSend: (message: string) => void;
  isLoading: boolean;
  onStop?: () => void;
}

export function InputBar({ onSend, isLoading, onStop }: InputBarProps) {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  
  const { inputMode, setInputMode, selectedModel, setSelectedModel } = useChatStore();

  const models = [
    { id: 'auto', name: 'Auto' },
    { id: 'openai', name: 'OpenAI (GPT-4o)' },
    { id: 'gemini-3.1-pro', name: 'Gemini (3.1 Pro)' },
    { id: 'ollama-llama3', name: 'Ollama (Llama 3)' },
    { id: 'ollama-mistral', name: 'Ollama (Mistral)' },
  ];

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (input.trim() && !isLoading) {
      onSend(input);
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = '56px';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="p-4 mx-auto max-w-3xl w-full relative z-20">
      
      {/* Sprint 7: Mode Toggle & Model Selector */}
      <div className="flex flex-col sm:flex-row justify-between items-center mb-3 px-2 gap-3">
        {/* Mode Toggle */}
        <div className="flex bg-[#131722] rounded-full p-1 border border-white/5">
          <button
            onClick={() => setInputMode('chat')}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-full text-sm transition-all cursor-pointer ${
              inputMode === 'chat'
                ? 'bg-indigo-600 text-white font-semibold shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-100 hover:bg-white/[0.04]'
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            <span>Chat with AI</span>
          </button>
          <button
            onClick={() => setInputMode('verify')}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-full text-sm transition-all cursor-pointer ${
              inputMode === 'verify'
                ? 'bg-indigo-600 text-white font-semibold shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-100 hover:bg-white/[0.04]'
            }`}
          >
            <CheckCircle className="w-4 h-4" />
            <span>Verify Existing Response</span>
          </button>
        </div>

        {/* Model Selector */}
        {inputMode === 'chat' && (
          <div className="relative">
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#131722] border border-white/5 text-sm font-medium text-slate-200 hover:text-white transition-colors cursor-pointer"
            >
              <span>{models.find(m => m.id === selectedModel)?.name || 'Select Model'}</span>
              <ChevronDown className="w-4 h-4 text-slate-400" />
            </button>
            <AnimatePresence>
              {isDropdownOpen && (
                <motion.div
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 5 }}
                  className="absolute right-0 bottom-full mb-2 w-48 bg-[#131722] border border-white/10 rounded-xl shadow-xl overflow-hidden z-50 py-1 backdrop-blur-xl"
                >
                  {models.map(model => {
                    const isSelected = selectedModel === model.id;
                    return (
                      <button
                        key={model.id}
                        onClick={() => {
                          setSelectedModel(model.id);
                          setIsDropdownOpen(false);
                        }}
                        className={`w-full text-left px-4 py-2 text-sm transition-colors cursor-pointer ${
                          isSelected
                            ? 'bg-white/[0.08] text-indigo-300 font-semibold'
                            : 'text-slate-300 hover:bg-white/5'
                        }`}
                      >
                        {model.name}
                      </button>
                    );
                  })}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>

      <div className="relative flex items-end bg-[#151923] border border-white/10 rounded-[28px] p-2 shadow-2xl shadow-black/50 focus-within:ring-2 focus-within:ring-indigo-500/30 transition-all duration-200 ease-out">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={inputMode === 'chat' ? "Message HalluciSense..." : "Paste AI response or article here to verify..."}
          className="w-full max-h-[200px] min-h-[56px] py-4 pl-6 pr-14 bg-transparent text-[15px] text-slate-200 placeholder:text-slate-500 resize-none outline-none overflow-y-auto font-medium leading-relaxed"
          rows={1}
        />
        
        <div className="absolute right-3 bottom-3">
          {isLoading ? (
            <button 
              onClick={onStop}
              className="p-2 rounded-full bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
            >
              <StopCircle className="w-5 h-5" />
            </button>
          ) : (
            <button 
              onClick={handleSubmit}
              disabled={!input.trim()}
              className="p-2.5 rounded-full bg-indigo-500 text-white hover:bg-indigo-600 disabled:opacity-30 disabled:hover:bg-indigo-500 transition-all shadow-lg shadow-indigo-500/20"
            >
              {inputMode === 'verify' ? <CheckCircle className="w-4 h-4" /> : <Send className="w-4 h-4" />}
            </button>
          )}
        </div>
      </div>
      <div className="text-center mt-3 text-xs text-slate-500">
        {inputMode === 'verify' 
          ? "Verification mode does not generate new AI responses. It only analyzes the text provided." 
          : "HalluciSense engine verifies statements in real-time. Responses may take longer than standard LLMs."}
      </div>
    </div>
  );
}
