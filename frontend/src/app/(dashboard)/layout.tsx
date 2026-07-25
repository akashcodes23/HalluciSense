'use client';

import React, { useState, useEffect } from 'react';
import { Sidebar } from '../../components/layout/Sidebar';
import { useAuthStore } from '../../stores/authStore';
import { useRouter } from 'next/navigation';
import { Menu } from 'lucide-react';
import { useChatStore } from '../../stores/chatStore';
import { verificationService } from '../../services/verificationService';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { isAuthenticated, fetchUser } = useAuthStore();
  const router = useRouter();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login');
    } else {
      fetchUser();
    }
  }, [isAuthenticated, router, fetchUser]);

  // Global Notification WebSocket
  useEffect(() => {
    const token = useAuthStore.getState().accessToken;
    if (!token || !isAuthenticated) return;

    let ws: WebSocket;
    let active = true;

    const connectWS = () => {
      const wsBase = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/api/v1';
      ws = new WebSocket(`${wsBase}/notifications/ws?token=${token}`);

      ws.onmessage = async (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'verification_complete' && data.message_id) {
            // Fetch the completed report directly
            const report = await verificationService.getReport(data.message_id);
            
            // Update the message in the chat store instantly!
            const updateMessage = useChatStore.getState().updateMessage;
            updateMessage(data.message_id, {
              verification_status: 'COMPLETE',
              verification_report: report,
            });
          }
        } catch (err) {
          console.error('Failed to parse notification:', err);
        }
      };

      ws.onclose = () => {
        if (active) setTimeout(connectWS, 3000); // Simple reconnect
      };
    };

    connectWS();

    return () => {
      active = false;
      if (ws) ws.close();
    };
  }, [isAuthenticated]);

  if (!isAuthenticated) {
    return null;
  }

  return (
    <div
      className="h-screen overflow-hidden bg-[var(--hs-bg)]"
      style={{
        display: 'grid',
        gridTemplateColumns: sidebarOpen ? '280px 1fr' : '0px 1fr',
        transition: 'grid-template-columns 300ms ease-in-out',
      }}
    >
      {/* Sidebar column — in document flow, width animated via grid */}
      <div className="overflow-hidden h-screen">
        <Sidebar isOpen={sidebarOpen} toggleSidebar={() => setSidebarOpen(false)} />
      </div>

      {/* Main Content column — always takes the remaining space */}
      <main className="flex flex-col h-screen overflow-hidden min-w-0 relative">
        {/* Floating hamburger — only visible when sidebar is collapsed */}
        {!sidebarOpen && (
          <button
            onClick={() => setSidebarOpen(true)}
            className="absolute top-4 left-4 z-50 p-2 rounded-lg bg-white/5 border border-white/5 text-slate-400 hover:text-white hover:bg-white/10 transition-all duration-200 ease-out"
            aria-label="Open sidebar"
          >
            <Menu className="w-5 h-5" />
          </button>
        )}

        {children}
      </main>
    </div>
  );
}
