import React, { useState, useEffect, useCallback } from 'react';
import { getLatestDebug, getDebugTrace } from '@/services/hallucisense-api';

interface StageDetails {
  duration_ms: number;
  memory_mb: number;
  confidence: number | null;
  details: Record<string, unknown>;
}

interface TraceSummary {
  total_duration_ms: number;
  total_memory_mb: number;
  final_h_score: number;
  risk_level: string;
  root_cause_classification: string;
  stage_count: number;
}

interface PipelineTrace {
  trace_id: string;
  timestamp: string;
  stages: Record<string, StageDetails>;
  summary: TraceSummary;
}

export const DebugDashboard: React.FC = () => {
  const [trace, setTrace] = useState<PipelineTrace | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [searchId, setSearchId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const fetchLatestTrace = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getLatestDebug();
      setTrace(data as unknown as PipelineTrace);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to fetch trace');
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchTraceById = async (id: string) => {
    if (!id.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getDebugTrace(id.trim());
      setTrace(data as unknown as PipelineTrace);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Trace fetch failed');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLatestTrace();
  }, [fetchLatestTrace]);

  const downloadJson = () => {
    if (!trace) return;
    const blob = new Blob([JSON.stringify(trace, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${trace.trace_id}.json`;
    a.click();
  };

  return (
    <div className="p-6 bg-[#0b1220] border border-white/[0.08] rounded-xl space-y-6 text-white font-sans">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <h2 className="text-xl font-bold tracking-tight">Pipeline Diagnostics & Trace Explorer</h2>
          <p className="text-xs text-slate-400 mt-1">
            Real-time stage latency, memory footprint, and telemetry
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            type="text"
            placeholder="Search Trace ID..."
            value={searchId}
            onChange={(e) => setSearchId(e.target.value)}
            className="px-3 py-1.5 bg-black/40 border border-white/10 rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 font-mono"
          />
          <button
            onClick={() => fetchTraceById(searchId)}
            className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold rounded-lg transition-colors"
          >
            Search
          </button>
          <button
            onClick={fetchLatestTrace}
            className="px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-semibold rounded-lg transition-colors"
          >
            Latest
          </button>
        </div>
      </div>

      {loading && (
        <div className="py-12 text-center text-slate-400 text-sm">
          Loading trace telemetry...
        </div>
      )}

      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-xs font-mono">
          {error}
        </div>
      )}

      {!loading && trace && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 bg-white/[0.02] border border-white/5 rounded-lg">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Trace ID</span>
              <span className="text-xs font-mono text-indigo-400 font-semibold">{trace.trace_id}</span>
            </div>
            <div className="p-3 bg-white/[0.02] border border-white/5 rounded-lg">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Total Latency</span>
              <span className="text-xs font-mono text-emerald-400 font-semibold">{trace.summary?.total_duration_ms?.toFixed(2)} ms</span>
            </div>
            <div className="p-3 bg-white/[0.02] border border-white/5 rounded-lg">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block">H-Score</span>
              <span className="text-xs font-mono text-amber-400 font-semibold">{(trace.summary?.final_h_score * 100)?.toFixed(1)}%</span>
            </div>
            <div className="p-3 bg-white/[0.02] border border-white/5 rounded-lg flex items-center justify-between">
              <div>
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Risk Level</span>
                <span className="text-xs font-semibold text-white">{trace.summary?.risk_level}</span>
              </div>
              <button
                onClick={downloadJson}
                className="px-2 py-1 bg-white/10 hover:bg-white/20 text-[10px] rounded font-semibold transition-colors"
              >
                Export JSON
              </button>
            </div>
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-slate-300">Execution Stage Breakdown</h3>
            <div className="space-y-2">
              {Object.entries(trace.stages || {}).map(([stageName, stageData]) => (
                <div key={stageName} className="p-3 bg-white/[0.02] border border-white/5 rounded-lg space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-200 capitalize">{stageName.replace(/_/g, ' ')}</span>
                    <span className="font-mono text-slate-400">{stageData.duration_ms?.toFixed(2)} ms</span>
                  </div>
                  {stageData.details && Object.keys(stageData.details).length > 0 && (
                    <pre className="p-2 bg-black/40 rounded text-[10px] text-slate-400 font-mono overflow-x-auto">
                      {JSON.stringify(stageData.details, null, 2)}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
