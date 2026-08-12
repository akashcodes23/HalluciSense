import { api } from './api';
import { Message } from '@/types/api';

export type { Message };

export const messageService = {
  getMessages: async (chatId: string): Promise<Message[]> => {
    const response = await api.get(`/chats/${chatId}/messages`);
    return response.data;
  },

  getHistory: async (chatId: string): Promise<Message[]> => {
    const response = await api.get(`/chats/${chatId}/messages`);
    return response.data;
  },

  sendMessage: async (chatId: string, content: string, model: string): Promise<Message> => {
    const response = await api.post(`/chats/${chatId}/messages`, {
      content,
      model,
    });
    return response.data;
  },

  verifyExternal: async (chatId: string, text: string): Promise<unknown> => {
    const response = await api.post(`/chats/${chatId}/verify-external`, { text });
    return response.data;
  },
};

const getWsBaseUrl = (): string => {
  if (process.env.NEXT_PUBLIC_WS_URL) {
    return process.env.NEXT_PUBLIC_WS_URL;
  }
  const httpUrl = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API_BASE_URL;
  if (httpUrl) {
    return httpUrl.replace(/^http/, "ws") + "/api/v1";
  }
  if (typeof window !== "undefined" && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
    return "wss://hallucisense-production.up.railway.app/api/v1";
  }
  return "ws://localhost:8000/api/v1";
};

export function createStreamingConnection(
  chatId: string,
  userMessage: string,
  model: string,
  accessToken: string,
  onToken: (token: string) => void,
  onDone: (messageId: string) => void,
  onError: (err: string) => void
): WebSocket {
  const wsBase = getWsBaseUrl();
  const url = `${wsBase}/chats/${chatId}/messages/stream?token=${encodeURIComponent(accessToken)}`;

  console.log('[WebSocket] Connecting...', {
    chatId,
    selectedModel: model,
    wsUrl: `${wsBase}/chats/${chatId}/messages/stream?token=***`,
    hasToken: Boolean(accessToken),
    tokenLength: accessToken?.length || 0,
  });

  const ws = new WebSocket(url);

  ws.onopen = () => {
    console.log('[WebSocket] Connection OPEN. Sending prompt payload...', {
      chatId,
      model,
      contentLength: userMessage.length,
    });

    const payload = JSON.stringify({
      chat_id: chatId,
      content: userMessage,
      model_name: model,
      temperature: 0.7,
      top_p: 0.9,
    });

    ws.send(payload);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'token' && data.content) {
        onToken(data.content);
      } else if (data.type === 'done') {
        onDone(data.message_id || 'done');
      } else if (data.type === 'error') {
        onError(data.detail || data.message || 'Streaming verification failed');
      }
    } catch {
      onToken(event.data);
    }
  };

  ws.onerror = (evt) => {
    console.error('[WebSocket] Error encountered:', evt);
    onError('WebSocket connection error encountered.');
  };

  ws.onclose = (evt) => {
    console.log('[WebSocket] Connection closed:', { code: evt.code, reason: evt.reason });
    if (!evt.wasClean && evt.code !== 1000) {
      onError(`WebSocket closed unexpectedly (Code ${evt.code})`);
    }
  };

  return ws;
}
