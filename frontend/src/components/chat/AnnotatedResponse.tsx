'use client';

import React from 'react';
import { SentenceChip } from './SentenceChip';
import { VerificationReport } from '../../stores/uiStore';

interface AnnotatedResponseProps {
  messageId: string;
  report: VerificationReport;
}

export function AnnotatedResponse({ messageId, report }: AnnotatedResponseProps) {
  // Build a map from sentence text → analysis for fast lookup
  const analysisMap = new Map(
    report.sentence_analyses.map(s => [s.sentence_text.trim(), s])
  );

  // The report contains the sentences; render them as chips
  return (
    <div className="leading-[1.85] text-slate-300">
      {report.sentence_analyses.map((analysis, idx) => (
        <React.Fragment key={idx}>
          <SentenceChip
            text={analysis.sentence_text}
            sentenceIndex={analysis.sentence_index}
            riskLevel={analysis.risk_level}
            hScore={analysis.h_score}
            messageId={messageId}
            report={report}
          />
          {' '}
        </React.Fragment>
      ))}
    </div>
  );
}
