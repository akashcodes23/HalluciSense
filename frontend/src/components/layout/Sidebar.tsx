'use client';

import React, { useEffect, useState, useRef } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { 
  MessageSquarePlus, Settings, LogOut, PanelLeftClose, 
  Database, History, MessageSquare, MoreVertical, Edit2, Trash2, ShieldCheck
} from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import { useChatStore } from '../../stores/chatStore';
import { chatService } from '../../services/chatService';

export function Sidebar({ isOpen, toggleSidebar }: { isOpen: boolean, toggleSidebar: () => void }) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, logout } = useAuthStore();
  const { chats, setChats, activeChatId, renameChat, deleteChat } = useChatStore();

  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');

  useEffect(() => {
    if (user) {
      chatService.list().then((res) => {
        setChats(res.items);
      }).catch(err => console.error(err));
    }
  }, [user, setChats]);

  const handleLogout = () => {
    logout();
    window.location.href = '/login';
  };

  const handleRenameSubmit = async (e: React.FormEvent, chatId: string) => {
    e.preventDefault();
    if (editTitle.trim()) {
      await renameChat(chatId, editTitle.trim());
    }
    setEditingId(null);
  };

  const handleDelete = async (chatId: string) => {
    if (confirm('Are you sure you want to delete this chat?')) {
      await deleteChat(chatId);
      if (activeChatId === chatId) {
        router.push('/dashboard');
      }
    }
    setMenuOpenId(null);
  };

  return (
    <aside className="w-[280px] h-screen bg-[var(--bg-2)] border-r border-white/5 flex flex-col shrink-0 relative z-50">
      <div className="flex flex-col h-full p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Link href="/dashboard" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center">
              <Database className="w-5 h-5 text-indigo-400" />
            </div>
            <span className="font-semibold text-lg tracking-tight">HalluciSense</span>
          </Link>
          <button onClick={toggleSidebar} className="p-2 hover:bg-white/5 rounded-md transition-colors">
            <PanelLeftClose className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        {/* New Chat Button */}
        <Link 
          href="/dashboard"
          className="flex items-center justify-center gap-2 w-full py-2.5 px-4 rounded-lg font-medium text-slate-200 bg-white/5 hover:bg-white/10 transition-colors mb-6 border border-white/5"
        >
          <MessageSquarePlus className="w-4 h-4" />
          New Chat
        </Link>

        {/* Navigation Links */}
        <div className="flex-1 overflow-y-auto -mx-2 px-2 custom-scrollbar">
          <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-4 pl-3 mt-2">Chats</div>
          <nav className="space-y-1.5 mb-6">
            {chats.map(chat => (
              <div 
                key={chat.id} 
                className={`group flex items-center justify-between px-3 py-2 rounded-lg transition-all duration-200 ease-out relative ${
                  pathname === `/chat/${chat.id}` ? 'bg-white/10 text-white' : 'hover:bg-white/5 text-slate-400'
                }`}
              >
                {editingId === chat.id ? (
                  <form onSubmit={(e) => handleRenameSubmit(e, chat.id)} className="flex-1 mr-2">
                    <input 
                      autoFocus
                      type="text" 
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      onBlur={(e) => handleRenameSubmit(e, chat.id)}
                      className="w-full bg-black/40 border border-white/10 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-indigo-500"
                    />
                  </form>
                ) : (
                  <Link href={`/chat/${chat.id}`} className="flex items-center gap-3 flex-1 min-w-0">
                    <MessageSquare className="w-4 h-4 shrink-0" />
                    <span className="text-sm truncate">{chat.title}</span>
                  </Link>
                )}

                {!editingId && (
                  <div className="relative shrink-0">
                    <button 
                      onClick={(e) => {
                        e.preventDefault();
                        setMenuOpenId(menuOpenId === chat.id ? null : chat.id);
                      }}
                      className={`p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity ${
                        menuOpenId === chat.id ? 'opacity-100 bg-white/10' : 'hover:bg-white/10'
                      }`}
                    >
                      <MoreVertical className="w-4 h-4 text-slate-400 hover:text-white" />
                    </button>
                    
                    {menuOpenId === chat.id && (
                      <>
                        <div className="fixed inset-0 z-40" onClick={() => setMenuOpenId(null)}></div>
                        <div className="absolute right-0 top-6 w-36 bg-[#1e1e24] border border-white/10 rounded-lg shadow-xl z-50 py-1 overflow-hidden">
                          <button 
                            onClick={() => {
                              setEditTitle(chat.title);
                              setEditingId(chat.id);
                              setMenuOpenId(null);
                            }}
                            className="w-full text-left px-3 py-2 text-sm text-slate-300 hover:bg-white/5 hover:text-white flex items-center gap-2 transition-colors"
                          >
                            <Edit2 className="w-3.5 h-3.5" /> Rename
                          </button>
                          <button 
                            onClick={() => handleDelete(chat.id)}
                            className="w-full text-left px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 hover:text-red-300 flex items-center gap-2 transition-colors"
                          >
                            <Trash2 className="w-3.5 h-3.5" /> Delete
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            ))}
          </nav>
          
          <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-4 pl-3 border-t border-white/5 pt-4">Workspace</div>
          <nav className="space-y-1.5">
            <Link href="/dashboard" className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200 ease-out ${pathname === '/dashboard' && !activeChatId ? 'bg-white/10 text-white' : 'hover:bg-white/5 text-slate-400'}`}>
              <MessageSquarePlus className="w-4 h-4" />
              <span className="text-sm">New Session</span>
            </Link>
            <Link href="/verify" className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200 ease-out ${pathname === '/verify' ? 'bg-white/10 text-white' : 'hover:bg-white/5 text-slate-400'}`}>
              <ShieldCheck className="w-4 h-4" />
              <span className="text-sm">Verification Engine</span>
            </Link>
            <Link href="/benchmark" className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200 ease-out ${pathname === '/benchmark' ? 'bg-white/10 text-white' : 'hover:bg-white/5 text-slate-400'}`}>
              <Database className="w-4 h-4 text-purple-400" />
              <span className="text-sm">Benchmark Leaderboard</span>
            </Link>
          </nav>
        </div>

        {/* Footer */}
        <div className="pt-4 mt-auto border-t border-white/5">
          <div className="space-y-1.5 -mx-2 px-2 mb-4">
            <Link href="/settings" className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-all duration-200 ease-out">
              <Settings className="w-4 h-4" />
              <span className="text-sm">Settings</span>
            </Link>
            <div className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-all duration-200 ease-out cursor-pointer" onClick={handleLogout}>
               <LogOut className="w-4 h-4" />
               <span className="text-sm">Logout</span>
            </div>
          </div>
          <div className="pt-4 border-t border-white/5 flex items-center gap-3">
             <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center text-xs font-bold text-slate-300">
                {user?.full_name?.charAt(0).toUpperCase() || 'U'}
             </div>
             <div className="flex flex-col min-w-0">
                <span className="text-sm font-medium text-slate-200 truncate">{user?.full_name || 'User'}</span>
                <span className="text-[11px] text-slate-500 truncate">{user?.email}</span>
             </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
