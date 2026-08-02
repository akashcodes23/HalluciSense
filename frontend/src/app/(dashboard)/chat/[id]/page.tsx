'use client';

import React, { useEffect, useRef, useCallback } from 'react';
import { useParams } from 'next/navigation';
import { ChatWindow } from '../../../../components/chat/ChatWindow';
import { VerificationPanel } from '../../../../components/verification/VerificationPanel';
import { useChatStore } from '../../../../stores/chatStore';
import { useAuthStore } from '../../../../stores/authStore';
import { messageService, createStreamingConnection } from '../../../../services/messageService';
import { Message } from '@/types/api';

export default function ChatPage() {
  const params = useParams();
  const chatId = params.id as string;

  const { accessToken } = useAuthStore();
  const {
    messages,
    setMessages,
    addMessage,
    setStreaming,
    appendStreamToken,
    clearStream,
    isLoadingMessages,
    setActiveChatId,
    inputMode,
    selectedModel,
  } = useChatStore();

  // Track whether we have loaded history for this specific chatId
  const loadedChatId = useRef<string | null>(null);
  // Track whether we've fired the initial message for this session
  const initialMsgFired = useRef(false);

  // ─── Load history whenever chatId changes ────────────────────────────────
  useEffect(() => {
    // Clear messages immediately when switching chats to prevent stale display
    setMessages([]);
    setActiveChatId(chatId);
    loadedChatId.current = chatId;
    initialMsgFired.current = false;

    let cancelled = false;

    const loadHistory = async () => {
      try {
        const history = await messageService.getHistory(chatId);
        if (!cancelled && loadedChatId.current === chatId) {
          setMessages(history);
        }
      } catch (err) {
        console.error('[ChatPage] Failed to load history:', err);
      }
    };

    loadHistory();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId]);

  // ─── Fire initial message from sessionStorage (set by dashboard) ─────────
  useEffect(() => {
    if (!accessToken || initialMsgFired.current) return;

    const pendingMsg = sessionStorage.getItem(`chat_init_${chatId}`);
    if (pendingMsg) {
      sessionStorage.removeItem(`chat_init_${chatId}`);
      initialMsgFired.current = true;
      // Small delay to let history load settle first
      setTimeout(() => handleSendMessage(pendingMsg), 200);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId, accessToken]);

  // ─── Send Message handler ────────────────────────────────────────────────
  const handleSendMessage = useCallback(async (msg: string) => {
    if (!accessToken) return;

    if (inputMode === 'verify') {
      // VERIFY EXISTING RESPONSE MODE — no AI generation, pure pipeline
      const tempId = `verify-${Date.now()}`;
      addMessage({
        id: tempId,
        chat_id: chatId,
        role: 'ASSISTANT',
        content: msg,
        verification_status: 'PROCESSING',
        processing_time_ms: null,
        created_at: new Date().toISOString(),
      });

      setStreaming(true);
      try {
        await messageService.verifyExternal(chatId, msg);
        // Poll once after 5s for verification result then sync
        setTimeout(async () => {
          try {
            const history = await messageService.getHistory(chatId);
            setMessages(history);
          } finally {
            setStreaming(false);
          }
        }, 5000);
      } catch (error: any) {
        console.error('[ChatPage] Verification failed:', error);
        setStreaming(false);
      }
      return;
    }

    // ── CHAT WITH AI MODE ──────────────────────────────────────────────────
    // Add user message immediately for instant feedback
    const userMsg: Message = {
      id: `temp-user-${Date.now()}`,
      chat_id: chatId,
      role: 'USER',
      content: msg,
      verification_status: 'COMPLETE',
      processing_time_ms: null,
      created_at: new Date().toISOString(),
    };
    addMessage(userMsg);

    // Placeholder for streaming AI response
    const aiPlaceholderId = `temp-ai-${Date.now()}`;
    addMessage({
      id: aiPlaceholderId,
      chat_id: chatId,
      role: 'ASSISTANT',
      content: '',
      verification_status: 'PROCESSING',
      processing_time_ms: null,
      created_at: new Date().toISOString(),
    });

    setStreaming(true);
    clearStream();

    createStreamingConnection(
      chatId,
      msg,
      selectedModel,
      accessToken,
      (tokenText) => {
        appendStreamToken(tokenText);
      },
      (_finalMessageId) => {
        // Streaming + verification dispatched — refresh history from backend
        // This replaces the temp messages with the final persisted ones (no duplication)
        setStreaming(false);
        messageService.getHistory(chatId).then((history) => {
          setMessages(history);
        }).catch((err) => {
          console.error('[ChatPage] Failed to refresh history:', err);
        });
      },
      (error) => {
        setStreaming(false);
        console.error('[ChatPage] Streaming error:', error);
        
        // Find and remove the stuck PROCESSING placeholder message
        setMessages(useChatStore.getState().messages.filter(m => m.id !== aiPlaceholderId));
        
        import('react-hot-toast').then(({ toast }) => {
            toast.error(error);
        });
      }
    );
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken, chatId, inputMode, selectedModel, setMessages, setStreaming, clearStream, addMessage, appendStreamToken]);

  return (
    <div className="flex flex-1 h-full overflow-hidden relative">
      {/* Ambient glow */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-purple-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Chat Area */}
      <div className="flex-1 h-full overflow-hidden min-w-0">
        <ChatWindow
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoadingMessages}
        />
      </div>

      {/* Verification Inspector Drawer */}
      <VerificationPanel />
    </div>
  );
}
