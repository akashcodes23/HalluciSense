import { create } from 'zustand';
import { Chat, Message } from '@/types/api';
import { chatService } from '../services/chatService';

interface ChatState {
  // Sidebar
  chats: Chat[];
  isLoadingChats: boolean;
  setChats: (chats: Chat[]) => void;
  addChat: (chat: Chat) => void;
  updateChat: (chatId: string, updates: Partial<Chat>) => void;
  removeChat: (chatId: string) => void;
  renameChat: (chatId: string, newTitle: string) => Promise<void>;
  deleteChat: (chatId: string) => Promise<void>;

  // Active chat
  activeChatId: string | null;
  messages: Message[];
  isLoadingMessages: boolean;
  isStreaming: boolean;
  streamingContent: string;
  setActiveChatId: (id: string | null) => void;
  setMessages: (messages: Message[]) => void;
  addMessage: (message: Message) => void;
  updateMessage: (messageId: string, updates: Partial<Message>) => void;
  appendStreamToken: (token: string) => void;
  clearStream: () => void;
  setStreaming: (streaming: boolean) => void;
  
  // Sprint 7 additions
  inputMode: 'chat' | 'verify';
  setInputMode: (mode: 'chat' | 'verify') => void;
  selectedModel: string;
  setSelectedModel: (model: string) => void;
}

export const useChatStore = create<ChatState>()((set) => ({
  chats: [],
  isLoadingChats: false,
  setChats: (chats) => set({ chats }),
  addChat: (chat) => set((state) => ({ chats: [chat, ...state.chats] })),
  updateChat: (chatId, updates) =>
    set((state) => ({
      chats: state.chats.map((c) => (c.id === chatId ? { ...c, ...updates } : c)),
    })),
  removeChat: (chatId) =>
    set((state) => ({ chats: state.chats.filter((c) => c.id !== chatId) })),
    
  renameChat: async (chatId, newTitle) => {
    try {
      await chatService.updateChat(chatId, newTitle);
      set((state) => ({
        chats: state.chats.map((c) => (c.id === chatId ? { ...c, title: newTitle } : c)),
      }));
    } catch (error) {
      console.error("Failed to rename chat:", error);
    }
  },

  deleteChat: async (chatId) => {
    try {
      await chatService.deleteChat(chatId);
      set((state) => ({
        chats: state.chats.filter((c) => c.id !== chatId),
        activeChatId: state.activeChatId === chatId ? null : state.activeChatId,
        messages: state.activeChatId === chatId ? [] : state.messages
      }));
    } catch (error) {
      console.error("Failed to delete chat:", error);
    }
  },

  activeChatId: null,
  messages: [],
  isLoadingMessages: false,
  isStreaming: false,
  streamingContent: '',
  setActiveChatId: (activeChatId) => set({ activeChatId }),
  setMessages: (messages) => set({ messages }),
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  updateMessage: (messageId, updates) =>
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === messageId ? { ...m, ...updates } : m
      ),
    })),
  appendStreamToken: (token) =>
    set((state) => ({ streamingContent: state.streamingContent + token })),
  clearStream: () => set({ streamingContent: '' }),
  setStreaming: (isStreaming) => set({ isStreaming }),

  // Sprint 7 additions
  inputMode: 'chat',
  setInputMode: (inputMode) => set({ inputMode }),
  selectedModel: 'gemini-3.1-pro',
  setSelectedModel: (selectedModel) => {
    // Optionally persist to localStorage
    if (typeof window !== 'undefined') {
      localStorage.setItem('halusicense_selected_model', selectedModel);
    }
    set({ selectedModel });
  },
}));

// Initialize selected model from localStorage if available
if (typeof window !== 'undefined') {
  const savedModel = localStorage.getItem('halusicense_selected_model');
  if (savedModel) {
    useChatStore.getState().setSelectedModel(savedModel);
  }
}
