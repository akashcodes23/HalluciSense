'use client';

import React from 'react';
import { ResponseVerificationSummary, ClaimVerificationResult } from '@/types/verification-types';
import { CheckCircle2, XCircle, AlertCircle, HelpCircle, Shield, Scale, Terminal } from 'lucide-react';

interface Props {
  summary?: ResponseVerificationSummary;
}

export const VerificationTracePanel: React.FC<Props> = ({ summary }) => {
  if (!summary || !summary.claims || summary.claims.length === 0) {
    return null;
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'VERIFIED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <CheckCircle2 className="w-3.5 h-3.5" /> VERIFIED
          </span>
        );
      case 'CONTRADICTED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <XCircle className="w-3.5 h-3.5" /> CONTRADICTED
          </span>
        );
      case 'INSUFFICIENT_EVIDENCE':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <HelpCircle className="w-3.5 h-3.5" /> INSUFFICIENT EVIDENCE
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-500/10 text-slate-400 border border-slate-500/20">
            <AlertCircle className="w-3.5 h-3.5" /> {status}
          </span>
        );
    }
  };

  return (
    <div className="bg-slate-900/60 backdrop-blur border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-teal-400" />
          <h3 className="text-sm font-semibold text-slate-200">Evidence Intelligence & Audit Trace</h3>
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-400">
          <span>Req ID: <code className="text-slate-300 font-mono">{summary.request_id.slice(0, 8)}</code></span>
          <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 font-mono">
            {summary.primary_status}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-slate-950/40 p-3 rounded-lg border border-slate-800/60">
          <p className="text-xs text-slate-400">Total Claims</p>
          <p className="text-lg font-bold text-slate-200">{summary.total_claims}</p>
        </div>
        <div className="bg-slate-950/40 p-3 rounded-lg border border-slate-800/60">
          <p className="text-xs text-emerald-400">Verified Claims</p>
          <p className="text-lg font-bold text-emerald-400">{summary.verified_claims}</p>
        </div>
        <div className="bg-slate-950/40 p-3 rounded-lg border border-slate-800/60">
          <p className="text-xs text-rose-400">Contradicted Claims</p>
          <p className="text-lg font-bold text-rose-400">{summary.contradicted_claims}</p>
        </div>
        <div className="bg-slate-950/40 p-3 rounded-lg border border-slate-800/60">
          <p className="text-xs text-amber-400">Insufficient Evidence</p>
          <p className="text-lg font-bold text-amber-400">{summary.unsupported_claims}</p>
        </div>
      </div>

      <div className="space-y-3 pt-2">
        {summary.claims.map((claim, idx) => (
          <div key={idx} className="bg-slate-950/60 border border-slate-800 rounded-lg p-3.5 space-y-2">
            <div className="flex items-start justify-between gap-2">
              <div className="space-y-1">
                <span className="text-xs font-mono text-slate-500 uppercase">Claim #{claim.claim_id + 1} • {claim.claim_type}</span>
                <p className="text-sm font-medium text-slate-200">{claim.claim_text}</p>
              </div>
              <div>{getStatusBadge(claim.status)}</div>
            </div>

            {claim.symbolic_result && (
              <div className="bg-slate-900/80 border border-slate-800 rounded p-2.5 text-xs font-mono text-slate-300 space-y-1">
                <div className="flex items-center gap-1.5 text-teal-400 font-semibold">
                  <Terminal className="w-3.5 h-3.5" /> Deterministic Symbolic Computation
                </div>
                <p>{claim.reason}</p>
              </div>
            )}

            {claim.reason && !claim.symbolic_result && (
              <p className="text-xs text-slate-400 italic">Audit note: {claim.reason}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
