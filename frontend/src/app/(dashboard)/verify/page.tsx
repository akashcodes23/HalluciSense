'use client';

import React, { useState } from 'react';
import { ShieldCheck, ShieldAlert, ShieldX, Play, Info, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { verificationService } from '../../../services/verificationService';
import { StandaloneVerificationResponse } from '../../../types/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import toast from 'react-hot-toast';

export default function VerifyPage() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<StandaloneVerificationResponse | null>(null);

  const handleVerify = async () => {
    if (!text.trim()) {
      toast.error('Please enter text to verify.');
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const data = await verificationService.verifyText(text);
      setResult(data);
      toast.success('Verification complete');
    } catch (err: any) {
      console.error(err);
      toast.error(err.response?.data?.detail || 'Verification failed');
    } finally {
      setLoading(false);
    }
  };

  const renderSkeleton = () => (
    <div className="animate-pulse space-y-6 mt-8">
      <div className="h-8 bg-white/5 rounded w-1/4"></div>
      <div className="grid grid-cols-4 gap-4">
        {[1, 2, 3, 4, 5, 6, 7, 8].map(i => (
          <div key={i} className="h-24 bg-white/5 rounded-xl"></div>
        ))}
      </div>
      <div className="h-40 bg-white/5 rounded-xl"></div>
    </div>
  );

  const getRiskColor = (risk: string) => {
    if (risk === 'VERIFIED') return '#22c55e';
    if (risk === 'NEEDS_VERIFICATION') return '#f59e0b';
    return '#ef4444';
  };

  const getRiskIcon = (risk: string) => {
    if (risk === 'VERIFIED') return <ShieldCheck className="w-6 h-6 text-green-400" />;
    if (risk === 'NEEDS_VERIFICATION') return <ShieldAlert className="w-6 h-6 text-yellow-400" />;
    return <ShieldX className="w-6 h-6 text-red-400" />;
  };

  return (
    <div className="h-full overflow-y-auto p-8 custom-scrollbar">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Verification Engine</h1>
          <p className="text-slate-400">
            Paste any AI-generated text below. The HalluciSense engine will analyze every claim, 
            retrieve supporting evidence, and highlight hallucinations.
          </p>
        </div>

        {/* Input */}
        <div className="bg-[#1e1e24] border border-white/5 rounded-2xl p-4 shadow-sm">
          <textarea
            className="w-full h-40 bg-transparent text-slate-200 placeholder-slate-500 focus:outline-none resize-none"
            placeholder="E.g., India has 35 states. The Sun revolves around Earth."
            value={text}
            onChange={e => setText(e.target.value)}
          />
          <div className="flex justify-end mt-2 pt-2 border-t border-white/5">
            <button
              onClick={handleVerify}
              disabled={loading}
              className="flex items-center gap-2 px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg font-medium transition-colors"
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Play className="w-4 h-4 fill-current" />
              )}
              Run Analysis
            </button>
          </div>
        </div>

        {loading && renderSkeleton()}

        {/* Results */}
        {result && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
            
            {/* Metrics Dashboard */}
            <div>
              <h2 className="text-lg font-semibold text-white mb-4">Metrics Dashboard</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard title="H-Score" value={`${(result.overall_h_score * 100).toFixed(1)}%`} color={getRiskColor(result.risk_level)} />
                <MetricCard title="Risk Level" value={result.risk_level.replace('_', ' ')} color={getRiskColor(result.risk_level)} icon={getRiskIcon(result.risk_level)} />
                <MetricCard title="Trust Score" value={`${(result.trust_score * 100).toFixed(1)}%`} color="#10b981" />
                <MetricCard title="Confidence" value={`${(result.confidence_score * 100).toFixed(1)}%`} color="#8b5cf6" />
                <MetricCard title="Evidence Coverage" value={`${(result.evidence_coverage * 100).toFixed(1)}%`} color="#3b82f6" />
                <MetricCard title="Verified Claims" value={result.verified_claims.toString()} color="#22c55e" />
                <MetricCard title="Hallucinated Claims" value={result.hallucinated_claims.toString()} color="#ef4444" />
                <MetricCard title="Processing Time" value={`${result.processing_time.toFixed(0)} ms`} color="#94a3b8" />
              </div>
            </div>

            {/* Charts Row */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-[#1e1e24] border border-white/5 rounded-2xl p-6">
                <h3 className="text-sm font-semibold text-slate-400 mb-4 uppercase tracking-wider">Risk Breakdown</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'Verified', value: result.verified_claims, color: '#22c55e' },
                          { name: 'Hallucinated', value: result.hallucinated_claims, color: '#ef4444' }
                        ]}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        paddingAngle={5}
                      >
                        {[
                          { name: 'Verified', value: result.verified_claims, color: '#22c55e' },
                          { name: 'Hallucinated', value: result.hallucinated_claims, color: '#ef4444' }
                        ].map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1e1e24', borderColor: 'rgba(255,255,255,0.1)' }}
                        itemStyle={{ color: '#fff' }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-[#1e1e24] border border-white/5 rounded-2xl p-6">
                <h3 className="text-sm font-semibold text-slate-400 mb-4 uppercase tracking-wider">Sentence Hallucination %</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={result.sentence_analysis.map((s, i) => ({ name: `S${i+1}`, value: s.h_score * 100 }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
                      <YAxis stroke="#64748b" fontSize={12} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1e1e24', borderColor: 'rgba(255,255,255,0.1)' }}
                        itemStyle={{ color: '#fff' }}
                        cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                      />
                      <Bar dataKey="value" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Corrected Response Block */}
            <div>
              <h2 className="text-lg font-semibold text-white mb-4">Corrected Response</h2>
              {result.risk_level === 'VERIFIED' ? (
                <div className="bg-green-500/10 border border-green-500/30 rounded-2xl p-6 flex items-start gap-4">
                  <CheckCircle2 className="w-6 h-6 text-green-400 shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-medium text-green-300 mb-1">No Correction Required</h3>
                    <p className="text-green-200/70">{result.corrected_response}</p>
                  </div>
                </div>
              ) : (
                <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-2xl p-6">
                  <p className="text-indigo-200 whitespace-pre-wrap">{result.corrected_response}</p>
                </div>
              )}
            </div>

            {/* Sentence Breakdown */}
            <div>
              <h2 className="text-lg font-semibold text-white mb-4">Sentence Breakdown & Explainability</h2>
              <div className="space-y-4">
                {result.sentence_analysis.map((sentence, index) => (
                  <div key={sentence.id || index} className="bg-[#1e1e24] border border-white/5 rounded-2xl overflow-hidden">
                    <div className="p-4 border-b border-white/5 flex gap-4">
                      <div className="shrink-0 w-8 h-8 rounded-full bg-white/5 flex items-center justify-center font-bold text-xs text-slate-400">
                        {index + 1}
                      </div>
                      <div className="flex-1">
                        <p className="text-slate-200 font-medium mb-2">{sentence.sentence_text}</p>
                        <div className="flex flex-wrap gap-2 text-xs">
                          <span className="px-2 py-1 rounded-full bg-black/40 text-slate-300">
                            H-Score: {(sentence.h_score * 100).toFixed(1)}%
                          </span>
                          <span className="px-2 py-1 rounded-full bg-black/40" style={{ color: getRiskColor(sentence.risk_level) }}>
                            {sentence.risk_level.replace('_', ' ')}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    {sentence.reasoning && (
                      <div className="p-4 bg-white/[0.02] border-b border-white/5 flex gap-3">
                        <Info className="w-5 h-5 text-indigo-400 shrink-0" />
                        <div>
                          <span className="text-xs font-bold uppercase tracking-wider text-slate-500 block mb-1">Reasoning</span>
                          <p className="text-sm text-slate-300 whitespace-pre-wrap">{sentence.reasoning}</p>
                        </div>
                      </div>
                    )}

                    {sentence.corrected_response && sentence.risk_level !== 'VERIFIED' && (
                      <div className="p-4 bg-green-500/5 border-b border-white/5 flex gap-3">
                        <CheckCircle2 className="w-5 h-5 text-green-400 shrink-0" />
                        <div>
                          <span className="text-xs font-bold uppercase tracking-wider text-green-600/70 block mb-1">Correction</span>
                          <p className="text-sm text-green-200/80">{sentence.corrected_response}</p>
                        </div>
                      </div>
                    )}

                    {sentence.evidence.length > 0 && (
                      <div className="p-4 space-y-3">
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-500 block">Retrieved Evidence</span>
                        {sentence.evidence.map((ev, i) => (
                          <div key={i} className="bg-black/20 rounded-lg p-3 text-sm">
                            <p className="text-slate-300 mb-1">"{ev.snippet}"</p>
                            <div className="flex justify-between items-center text-xs">
                              <a href={ev.source_url || '#'} target="_blank" rel="noreferrer" className="text-indigo-400 hover:underline">
                                {ev.source_name}
                              </a>
                              <span className={ev.is_supporting ? "text-green-400" : "text-red-400"}>
                                {ev.is_supporting ? 'Supporting' : 'Contradicting'} ({(ev.similarity_score * 100).toFixed(1)}%)
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  );
}

function MetricCard({ title, value, color, icon }: { title: string, value: string, color: string, icon?: React.ReactNode }) {
  return (
    <div className="bg-[#1e1e24] border border-white/5 rounded-2xl p-4 flex flex-col justify-between h-full hover:bg-white/[0.03] transition-colors">
      <h3 className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-2">{title}</h3>
      <div className="flex items-center justify-between">
        <span className="text-2xl font-bold" style={{ color }}>{value}</span>
        {icon && <span>{icon}</span>}
      </div>
    </div>
  );
}
