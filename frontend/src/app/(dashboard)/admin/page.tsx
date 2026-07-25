'use client';

import React, { useEffect, useState } from 'react';
import { api } from '../../../services/api';
import { useAuthStore } from '../../../stores/authStore';
import { ShieldAlert } from 'lucide-react';

export default function AdminPage() {
  const { user } = useAuthStore();
  const [users, setUsers] = useState<any[]>([]);
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    if (user?.role === 'ADMIN') {
      api.get('/admin/users').then(res => setUsers(res.data)).catch(console.error);
      api.get('/admin/system-health').then(res => setHealth(res.data)).catch(console.error);
    }
  }, [user]);

  if (user?.role !== 'ADMIN') {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-400">
        <ShieldAlert className="w-16 h-16 mb-4 text-red-400/50" />
        <h2 className="text-xl">Access Denied</h2>
        <p>You must be an administrator to view this page.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 h-full overflow-y-auto p-8 relative z-10 w-full max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold mb-8 text-white">Admin Dashboard</h1>
      
      {health && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="bg-white/5 border border-white/10 rounded-xl p-6">
            <h3 className="text-slate-400 text-sm mb-2">System Status</h3>
            <p className="text-2xl font-bold text-green-400 uppercase">{health.status}</p>
          </div>
          <div className="bg-white/5 border border-white/10 rounded-xl p-6">
            <h3 className="text-slate-400 text-sm mb-2">Total Users</h3>
            <p className="text-2xl font-bold text-white">{health.total_users}</p>
          </div>
          <div className="bg-white/5 border border-white/10 rounded-xl p-6">
            <h3 className="text-slate-400 text-sm mb-2">Total Messages</h3>
            <p className="text-2xl font-bold text-white">{health.total_messages}</p>
          </div>
        </div>
      )}

      <h2 className="text-xl font-bold mb-4 text-white">User Management</h2>
      <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
        <table className="w-full text-left text-sm text-slate-300">
          <thead className="bg-white/5 border-b border-white/10">
            <tr>
              <th className="px-6 py-4 font-semibold">Name</th>
              <th className="px-6 py-4 font-semibold">Email</th>
              <th className="px-6 py-4 font-semibold">Role</th>
              <th className="px-6 py-4 font-semibold">Chats</th>
              <th className="px-6 py-4 font-semibold">Joined</th>
            </tr>
          </thead>
          <tbody>
            {users.map(u => (
              <tr key={u.id} className="border-b border-white/5 hover:bg-white/5">
                <td className="px-6 py-4 font-medium text-white">{u.full_name}</td>
                <td className="px-6 py-4">{u.email}</td>
                <td className="px-6 py-4">
                  <span className={`px-2 py-1 rounded-full text-xs font-bold ${u.role === 'ADMIN' ? 'bg-purple-500/20 text-purple-400' : 'bg-slate-500/20 text-slate-400'}`}>
                    {u.role}
                  </span>
                </td>
                <td className="px-6 py-4">{u.chat_count}</td>
                <td className="px-6 py-4">{new Date(u.created_at).toLocaleDateString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
