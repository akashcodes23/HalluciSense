'use client';

import React, { useState } from 'react';
import { useAuthStore } from '../../../stores/authStore';
import { api } from '../../../services/api';
import { Save } from 'lucide-react';

export default function SettingsPage() {
  const { user, fetchUser } = useAuthStore();
  const [name, setName] = useState(user?.full_name || '');
  const [model, setModel] = useState(user?.preferred_model || 'gemini-3.1-pro');
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.patch('/users/me', {
        full_name: name,
        preferred_model: model,
      });
      await fetchUser();
      alert('Settings saved successfully!');
    } catch (err) {
      console.error(err);
      alert('Failed to save settings.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex-1 h-full overflow-y-auto p-8 relative z-10 w-full max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold mb-8 text-white">Profile & Settings</h1>
      
      <div className="bg-white/5 border border-white/10 rounded-xl p-8 mb-8">
        <h2 className="text-xl font-semibold text-white mb-6">Account Information</h2>
        
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Email Address</label>
            <input 
              type="text" 
              value={user?.email || ''} 
              disabled 
              className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-2.5 text-slate-500 cursor-not-allowed"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Full Name</label>
            <input 
              type="text" 
              value={name} 
              onChange={e => setName(e.target.value)}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-indigo-500 transition-colors"
            />
          </div>
        </div>
      </div>

      <div className="bg-white/5 border border-white/10 rounded-xl p-8 mb-8">
        <h2 className="text-xl font-semibold text-white mb-6">Preferences</h2>
        
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Preferred AI Model</label>
            <select 
              value={model}
              onChange={e => setModel(e.target.value)}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-indigo-500 transition-colors appearance-none"
            >
              <option value="gemini-3.1-pro">Gemini 1.5 Pro</option>
              <option value="gemini-3.1-flash">Gemini 1.5 Flash</option>
              <option value="gpt-4o">GPT-4o (Planned)</option>
              <option value="claude-3.5-sonnet">Claude 3.5 Sonnet (Planned)</option>
            </select>
          </div>
        </div>
      </div>

      <div className="flex justify-end">
        <button 
          onClick={handleSave}
          disabled={saving}
          className="bg-indigo-500 hover:bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium transition-colors flex items-center gap-2 disabled:opacity-50"
        >
          <Save className="w-4 h-4" />
          {saving ? 'Saving...' : 'Save Changes'}
        </button>
      </div>
    </div>
  );
}
