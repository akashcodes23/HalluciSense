'use client';

import React, { useEffect, useState } from 'react';
import { api } from '../../../services/api';
import { useAuthStore } from '../../../stores/authStore';

export default function AnalyticsPage() {
  const { user } = useAuthStore();
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const res = await api.get('/analytics/summary');
        setData(res.data);
      } catch (err) {
        console.error(err);
      }
    };
    fetchAnalytics();
  }, []);

  if (!data) return <div className="p-8 text-white">Loading analytics...</div>;

  return (
    <div className="flex-1 h-full overflow-y-auto p-8 relative z-10 w-full max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold mb-8 text-white">Your Analytics</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white/5 border border-white/10 rounded-xl p-6">
          <h3 className="text-slate-400 text-sm mb-2">Total Verified Messages</h3>
          <p className="text-4xl font-bold text-white">{data.total_verified}</p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-xl p-6">
          <h3 className="text-slate-400 text-sm mb-2">Average H-Score</h3>
          <p className="text-4xl font-bold text-indigo-400">{data.avg_h_score.toFixed(2)}</p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-xl p-6">
          <h3 className="text-slate-400 text-sm mb-2">Risk Breakdown</h3>
          <ul className="text-sm space-y-1">
            {Object.entries(data.risk_distribution).map(([key, val]) => (
              <li key={key} className="flex justify-between">
                <span className="text-slate-300">{key.replace('_', ' ')}</span>
                <span className="font-bold text-white">{val as number}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
