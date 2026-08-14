---
target: frontend/src/app/(dashboard)/benchmark/page.tsx
total_score: 35
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 2
timestamp: 2026-08-14T05-32-14Z
slug: frontend-src-app-dashboard-benchmark-page-tsx
---
#### Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Static dataset representation; lacks active dataset indicator or refresh state |
| 2 | Match System / Real World | 4 | Excellent scientific precision (AUROC, MCC, ECE, Brier Score, Bootstrap CIs B=10,000) |
| 3 | User Control and Freedom | 3 | Static table rows; users cannot sort columns (e.g. sort by Latency or ECE) |
| 4 | Consistency and Standards | 3 | Uses ad-hoc `bg-[var(--hs-bg)]` rather than canonical design tokens (`bg-bg-surface`, `border-white/[0.04]`) |
| 5 | Error Prevention | 4 | Safe fixed-precision decimal formatting across all metric cells |
| 6 | Recognition Rather Than Recall | 3 | Column acronyms (AUROC, MCC, ECE) lack hover tooltips for non-specialist reviewers |
| 7 | Flexibility and Efficiency | 3 | Missing 1-click LaTeX / CSV export and dataset selector tabs (Multi-Domain, HaluEval, TruthfulQA) |
| 8 | Aesthetic and Minimalist Design | 3 | Contrast warnings on highlighted row rank badges; lacks inline relative bar indicators |
| 9 | Error Recovery | 4 | Robust fallback state for metric calculations |
| 10 | Help and Documentation | 4 | Three structured highlight cards explaining McNemar's test, Cohen's d, and ECE calibration |
| **Total** | | **35/40** | **Good (87.5%)** |

#### Design Specificity Verdict

**LLM Assessment**: High academic domain specificity. The page avoids generic dashboard charts and instead presents a formal peer-reviewed benchmark leaderboard with rigorous statistical metrics (ECE, MCC, bootstrap intervals, and McNemar's significance tests).

**Deterministic Scan**: 2 contrast warnings:
- `text-slate-300 on bg-indigo-500` (line 94)
- `text-slate-400 on bg-amber-500` (line 99)

#### Overall Impression
A clean, authoritative empirical evaluation leaderboard that effectively demonstrates HalluciSense's state-of-the-art performance against 8 baseline algorithms.

#### What's Working
1. **Peer-Reviewed Metric Density**: Tracks 13 rigorous classification and calibration metrics per model.
2. **Statistical Significance Cards**: Explicitly details McNemar's test ($p < 0.001$), Cohen's $d = 0.84$, and ECE calibration in dedicated panels.
3. **Highlighted Top Baseline**: Prominently features HalluciSense at the top with rank badges.

#### Priority Issues
- **[P1] Interactive Column Sorting**: Allow researchers to click any column header (AUROC, F1, Latency, ECE) to sort ascending/descending.  
  *Suggested command: `/impeccable delight frontend/src/app/(dashboard)/benchmark/page.tsx`*
- **[P1] One-Click LaTeX & CSV Table Export**: Researchers need to cite and copy the leaderboard table into LaTeX papers directly.  
  *Suggested command: `/impeccable delight frontend/src/app/(dashboard)/benchmark/page.tsx`*
- **[P2] Design System Token Alignment**: Update canvas background and borders to use `DESIGN.md` tokens (`bg-bg`, `border-white/[0.06]`, `bg-bg-surface`).  
  *Suggested command: `/impeccable polish frontend/src/app/(dashboard)/benchmark/page.tsx`*
- **[P3] Metric Definition Tooltips**: Add contextual hover explanations to metric column headers ($ECE$, $MCC$, $AUROC$, $Brier$).  
  *Suggested command: `/impeccable clarify frontend/src/app/(dashboard)/benchmark/page.tsx`*

#### Persona Red Flags
- **Dr. Evelyn (Peer Reviewer)**: Cannot sort the table to see which model has the lowest latency or lowest ECE calibration error.
- **Alex (ML Engineer)**: Wants to export the comparative leaderboard as raw CSV or LaTeX without manual transcription.

#### Questions to Consider
- Should we add dataset toggle tabs (e.g. Multi-Domain N=750 vs HaluEval vs TruthfulQA)?
- Would inline visual bar meters in the AUROC column enhance scannability?
