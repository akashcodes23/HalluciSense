import { api } from './api';
import { Chat, ChatListResponse } from '@/types/api';

export const chatService = {
  async list(limit = 50, offset = 0): Promise<ChatListResponse> {
    const response = await api.get<ChatListResponse>('/chats', { params: { limit, offset } });
    return response.data;
  },

  async create(title?: string, model_used?: string): Promise<Chat> {
    const response = await api.post<Chat>('/chats', { title, model_used });
    return response.data;
  },

  async get(chatId: string): Promise<Chat> {
    const response = await api.get<Chat>(`/chats/${chatId}`);
    return response.data;
  },

  async update(chatId: string, data: { title?: string; is_archived?: boolean }): Promise<Chat> {
    const response = await api.patch<Chat>(`/chats/${chatId}`, data);
    return response.data;
  },

  // Convenience alias for rename from sidebar
  async updateChat(chatId: string, title?: string, isArchived?: boolean): Promise<Chat> {
    return this.update(chatId, { title, is_archived: isArchived });
  },

  async delete(chatId: string): Promise<void> {
    await api.delete(`/chats/${chatId}`);
  },

  // Convenience alias for delete from sidebar
  async deleteChat(chatId: string): Promise<void> {
    return this.delete(chatId);
  },
};
