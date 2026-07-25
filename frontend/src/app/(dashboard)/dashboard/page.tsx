'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { ChatWindow } from '../../../components/chat/ChatWindow';
import { chatService } from '../../../services/chatService';
import { useChatStore } from '../../../stores/chatStore';

export default function DashboardPage() {
  const router = useRouter();
  const [messages] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const { selectedModel } = useChatStore();

  const handleSendMessage = async (msg: string) => {
    setIsLoading(true);
    try {
      // Create a new chat on the backend with the selected model
      const newChat = await chatService.create(
        msg.length > 40 ? msg.substring(0, 40).trimEnd() + '…' : msg,
        selectedModel
      );
      // Store the initial message in sessionStorage so the chat page can pick it up
      // This avoids a fragile query param and works across refreshes gracefully
      sessionStorage.setItem(`chat_init_${newChat.id}`, msg);
      // Navigate to the correct route: /chat/[id] (not /dashboard/chat/[id])
      router.push(`/chat/${newChat.id}`);
    } catch (err: any) {
      console.error('Failed to create chat:', err);
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 h-full relative">
      {/* Background ambient glow */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-purple-500/5 rounded-full blur-3xl pointer-events-none" />
      <ChatWindow
        messages={messages}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
      />
    </div>
  );
}
