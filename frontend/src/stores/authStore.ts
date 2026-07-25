import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authService } from '../services/authService';

interface User {
  id: string;
  email: string;
  full_name: string;
  role: string;
  preferred_model?: string;
  avatar_url?: string | null;
  is_active?: boolean;
  is_verified?: boolean;
}

interface AuthState {
  user: User | null;
  accessToken: string | null;
  token: string | null; // alias for accessToken - backward compat
  refreshToken: string | null;
  isAuthenticated: boolean;
  setToken: (token: string, refreshToken?: string) => void;
  setUser: (user: User) => void;
  login: (token: string, refreshToken: string, user: User) => void;
  logout: () => void;
  fetchUser: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      accessToken: null,
      token: null,
      refreshToken: null,
      isAuthenticated: false,
      setToken: (token, refreshToken) => set((state) => ({
        accessToken: token,
        token: token, // keep alias in sync
        refreshToken: refreshToken !== undefined ? refreshToken : state.refreshToken,
        isAuthenticated: true
      })),
      setUser: (user) => set({ user }),
      login: (token, refreshToken, user) => set({
        accessToken: token,
        token: token, // keep alias in sync
        refreshToken,
        user,
        isAuthenticated: true
      }),
      logout: () => {
        set({ accessToken: null, token: null, refreshToken: null, user: null, isAuthenticated: false });
      },
      fetchUser: async () => {
        try {
          if (!get().accessToken) return;
          const user = await authService.getCurrentUser();
          set({ user });
        } catch (error) {
          console.error('Failed to fetch user', error);
          set({ accessToken: null, token: null, user: null, isAuthenticated: false });
        }
      }
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ accessToken: state.accessToken, token: state.accessToken, refreshToken: state.refreshToken }),
    }
  )
);
