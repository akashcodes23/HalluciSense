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
      model: model,
    });

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(payload);
      console.log('[WebSocket] Payload sent successfully');
    } else {
      console.error('[WebSocket] Socket opened but readyState is not OPEN:', ws.readyState);
    }
  };

  ws.onmessage = (event) => {
    try {
      const chunk = JSON.parse(event.data);
      console.log('[WebSocket] Frame received:', chunk.type, chunk.text ? `(${chunk.text.length} chars)` : '');

      if (chunk.type === 'token' && chunk.text) {
        onToken(chunk.text);
      } else if (chunk.type === 'verification_dispatched' && chunk.message_id) {
        console.log('[WebSocket] Verification dispatched:', chunk.message_id);
        onDone(chunk.message_id);
        ws.close(1000);
      } else if (chunk.type === 'error') {
        console.error('[WebSocket] Backend error frame:', chunk.error);
        onError(chunk.error || 'Unknown server error');
        ws.close(1000);
      }
    } catch (parseErr) {
      console.warn('[WebSocket] Non-JSON or parse error:', parseErr);
    }
  };

  ws.onerror = (event) => {
    console.error('[WebSocket] Socket error event:', event);
    onError('WebSocket connection failed.');
  };
  
  ws.onclose = (event) => {
    console.log('[WebSocket] Connection CLOSED:', {
      code: event.code,
      reason: event.reason,
      wasClean: event.wasClean,
    });

    // 1000 is normal closure
    if (event.code !== 1000) {
      onError(`WebSocket closed (Code ${event.code}): ${event.reason || 'Unexpected closure'}`);
    }
  };

  return ws;
}
