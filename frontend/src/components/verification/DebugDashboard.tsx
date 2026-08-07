import React, { useState, useEffect } from 'react';

interface StageDetails {
  duration_ms: number;
  memory_mb: number;
  confidence: number | null;
  details: Record<string, any>;
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

  const fetchLatestTrace = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/v1/debug/latest');
      if (!res.ok) throw new Error('No trace data found');
      const data = await res.json();
      setTrace(data);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch trace');
    } finally {
      setLoading(false);
    }
  };

  const fetchTraceById = async (id: string) => {
    if (!id.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/v1/debug/trace/${id.trim()}`);
      if (!res.ok) throw new Error(`Trace ID ${id} not found`);
      const data = await res.json();
      setTrace(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLatestTrace();
  }, []);

  const downloadJson = () => {
    if (!trace) return;
    const blob = new Blob([JSON.stringify(trace, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${trace.trace_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-6 bg-slate-950 text-slate-100 rounded-xl border border-slate-800 shadow-2xl space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-2xl font-bold text-sky-400 flex items-center gap-2">
            <span>🔬</span> HalluciSense Pipeline Debug Inspector
          </h2>
          <p className="text-slate-400 text-sm">
            Phase 25 Trace Diagnostics, Step-by-Step Claim Execution & Single-Label Root Cause Classifier
          </p>
        </div>
        <div className="flex items-center gap-3">
          <input
            type="text"
            placeholder="Search TRACE_ID..."
            value={searchId}
            onChange={(e) => setSearchId(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-sky-500"
          />
          <button
            onClick={() => fetchTraceById(searchId)}
            className="bg-sky-600 hover:bg-sky-500 text-white font-medium text-xs px-3 py-2 rounded-lg transition"
          >
            Search
          </button>
          <button
            onClick={fetchLatestTrace}
            className="bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs px-3 py-2 rounded-lg transition"
          >
            Latest Trace
          </button>
          {trace && (
            <button
              onClick={downloadJson}
              className="bg-emerald-700 hover:bg-emerald-600 text-white font-medium text-xs px-3 py-2 rounded-lg transition"
            >
              Export JSON
            </button>
          )}
        </div>
      </div>

      {loading && (
        <div className="text-center py-12 text-slate-400 animate-pulse">
          Loading diagnostic trace payload...
        </div>
      )}

      {error && (
        <div className="bg-rose-950/40 border border-rose-800 text-rose-300 p-4 rounded-lg text-sm">
          ⚠️ {error}
        </div>
      )}

      {trace && !loading && (
        <div className="space-y-6">
          {/* Summary Banner */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 text-center">
              <span className="text-slate-400 text-xs block">Trace ID</span>
              <span className="text-sky-400 font-mono font-bold text-sm">{trace.trace_id}</span>
            </div>
            <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 text-center">
              <span className="text-slate-400 text-xs block">Final H-Score</span>
              <span className="text-emerald-400 font-bold text-lg">{trace.summary.final_h_score.toFixed(4)}</span>
            </div>
            <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 text-center">
              <span className="text-slate-400 text-xs block">Risk Level</span>
              <span className="text-amber-400 font-semibold text-sm">{trace.summary.risk_level}</span>
            </div>
            <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 text-center">
              <span className="text-slate-400 text-xs block">Root Cause Category</span>
              <span className="text-purple-400 font-bold text-sm">{trace.summary.root_cause_classification}</span>
            </div>
            <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 text-center">
              <span className="text-slate-400 text-xs block">Execution Latency</span>
              <span className="text-indigo-400 font-bold text-sm">{trace.summary.total_duration_ms} ms</span>
            </div>
          </div>

          {/* Execution Stages Timeline */}
          <div className="bg-slate-900 p-5 rounded-lg border border-slate-800 space-y-4">
            <h3 className="text-md font-semibold text-slate-200 border-b border-slate-800 pb-2">
              Pipeline Stage Diagnostics Timeline
            </h3>
            <div className="space-y-3">
              {Object.entries(trace.stages).map(([stageName, details]) => (
                <div key={stageName} className="bg-slate-950 p-3.5 rounded border border-slate-800 flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
                  <div>
                    <span className="font-mono text-sky-300 font-medium text-sm">{stageName}</span>
                    <p className="text-slate-400 text-xs mt-0.5">
                      Details: {JSON.stringify(details.details)}
                    </p>
                  </div>
                  <div className="flex items-center gap-4 text-xs font-mono">
                    <span className="text-slate-400">⏱️ {details.duration_ms} ms</span>
                    <span className="text-slate-400">💾 {details.memory_mb} MB</span>
                    {details.confidence !== null && (
                      <span className="text-emerald-400">Confidence: {(details.confidence * 100).toFixed(1)}%</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
