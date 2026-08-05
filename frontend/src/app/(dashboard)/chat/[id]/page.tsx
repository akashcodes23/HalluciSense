'use client';

import React, { useEffect, useRef, useCallback } from 'react';
import { useParams } from 'next/navigation';
import { ChatWindow } from '../../../../components/chat/ChatWindow';
import { VerificationPanel } from '../../../../components/verification/VerificationPanel';
import { useChatStore } from '../../../../stores/chatStore';
import { useAuthStore } from '../../../../stores/authStore';
import { messageService, createStreamingConnection } from '../../../../services/messageService';
import { chatService } from '../../../../services/chatService';
import { Message } from '@/types/api';

export default function ChatPage() {
  const params = useParams();
  const chatId = params.id as string;

  const { accessToken } = useAuthStore();
  const {
    messages,
    setMessages,
    setChats,
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
    setMessages([]);
    setActiveChatId(chatId);
    loadedChatId.current = chatId;
    initialMsgFired.current = false;

    let cancelled = false;

    const loadHistory = async () => {
      try {
        console.log('[ChatPage] Loading message history for chatId:', chatId);
        const history = await messageService.getHistory(chatId);
        if (!cancelled && loadedChatId.current === chatId) {
          setMessages(history);
          console.log('[ChatPage] History loaded. Total messages:', history.length);
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

  // ─── Auto-poll processing messages until verification completes ──────────
  useEffect(() => {
    const hasProcessing = messages.some((m) => m.verification_status === 'PROCESSING');
    if (!hasProcessing) return;

    const timer = setTimeout(async () => {
      try {
        const freshHistory = await messageService.getHistory(chatId);
        setMessages(freshHistory);
      } catch (err) {
        console.error('[ChatPage] Auto-refresh verification error:', err);
      }
    }, 2500);

    return () => clearTimeout(timer);
  }, [messages, chatId, setMessages]);

  // ─── Fire initial message from sessionStorage (set by dashboard) ─────────
  useEffect(() => {
    if (!accessToken || initialMsgFired.current) return;

    const pendingMsg = sessionStorage.getItem(`chat_init_${chatId}`);
    if (pendingMsg) {
      sessionStorage.removeItem(`chat_init_${chatId}`);
      initialMsgFired.current = true;
      setTimeout(() => handleSendMessage(pendingMsg), 200);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId, accessToken]);

  // ─── Send Message handler ────────────────────────────────────────────────
  const handleSendMessage = useCallback(async (msg: string) => {
    if (!accessToken) {
      console.error('[ChatPage] Cannot send message: No access token available');
      return;
    }

    console.log('[ChatPage] handleSendMessage triggered:', {
      chatId,
      selectedModel,
      inputMode,
      tokenPresence: Boolean(accessToken),
      promptLength: msg.length,
    });

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
        setTimeout(async () => {
          try {
            const history = await messageService.getHistory(chatId);
            setMessages(history);
            const updatedChats = await chatService.list();
            setChats(updatedChats.items);
          } finally {
            setStreaming(false);
          }
        }, 3000);
      } catch (error: any) {
        console.error('[ChatPage] Verification failed:', error);
        setStreaming(false);
      }
      return;
    }

    // ── CHAT WITH AI MODE ──────────────────────────────────────────────────
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
        console.log('[ChatPage] Generation & Verification completed for message:', _finalMessageId);
        setStreaming(false);
        
        // Refresh active message history AND sidebar chats list
        Promise.all([
          messageService.getHistory(chatId),
          chatService.list()
        ]).then(([history, chatsRes]) => {
          setMessages(history);
          setChats(chatsRes.items);
          console.log('[ChatPage] History and chat list refreshed successfully.');
        }).catch((err) => {
          console.error('[ChatPage] Failed to refresh history or chat list:', err);
        });
      },
      (error) => {
        setStreaming(false);
        console.error('[ChatPage] Streaming error:', error);
        
        setMessages(useChatStore.getState().messages.filter(m => m.id !== aiPlaceholderId));
        
        import('react-hot-toast').then(({ toast }) => {
            toast.error(error);
        });
      }
    );
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken, chatId, inputMode, selectedModel, setMessages, setChats, setStreaming, clearStream, addMessage, appendStreamToken]);

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
