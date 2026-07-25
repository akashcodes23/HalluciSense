import { api } from './api';
import { Message } from '@/types/api';

export const messageService = {
  async getHistory(chatId: string, limit = 50, offset = 0): Promise<Message[]> {
    const response = await api.get<{items: Message[], total: number}>(`/chats/${chatId}/messages`, {
      params: { limit, offset },
    });
    return response.data.items;
  },
  
  async verifyExternal(chatId: string, content: string): Promise<{ message_id: string }> {
    const response = await api.post<{ message_id: string }>(`/chats/${chatId}/messages/verify-external`, {
      content,
    });
    return response.data;
  },
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
  const wsBase = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/api/v1';
  const ws = new WebSocket(
    `${wsBase}/chats/${chatId}/messages/stream?token=${accessToken}`
  );

  ws.onopen = () => {
    ws.send(JSON.stringify({ chat_id: chatId, content: userMessage, model }));
  };

  ws.onmessage = (event) => {
    try {
      const chunk = JSON.parse(event.data);
      if (chunk.type === 'token' && chunk.content) {
        onToken(chunk.content);
      } else if (chunk.type === 'verification_dispatched' && chunk.message_id) {
        onDone(chunk.message_id);
        ws.close();
      } else if (chunk.type === 'error') {
        onError(chunk.error || 'Unknown error');
        ws.close();
      }
    } catch {
      // Non-JSON frames ignored
    }
  };

  ws.onerror = () => {
    onError('WebSocket connection failed. Check the backend.');
  };

  return ws;
}
