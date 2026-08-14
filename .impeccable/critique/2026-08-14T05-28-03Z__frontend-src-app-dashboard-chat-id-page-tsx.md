---
target: frontend/src/app/(dashboard)/chat/[id]/page.tsx
total_score: 34
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 1
timestamp: 2026-08-14T05-28-03Z
slug: frontend-src-app-dashboard-chat-id-page-tsx
---
#### Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 4 | Real-time token streaming with live auto-scroll and status pills (Queued, Analyzing, Verified) |
| 2 | Match System / Real World | 4 | Familiar chat interface with instant mode toggling (Chat with AI vs Verify Response) |
| 3 | User Control and Freedom | 4 | Side drawer toggle, full history export (Markdown & JSON), and multi-model selector |
| 4 | Consistency and Standards | 3 | Verification drawer uses a 3-tier risk map rather than the canonical 4-tier taxonomy (#10b981, #f59e0b, #f97316, #ef4444) |
| 5 | Error Prevention | 4 | Token presence preflight, empty-message guard, and automatic cleanup of failed stream placeholders |
| 6 | Recognition Rather Than Recall | 4 | Clickable sentence highlights inside MessageBubble open the exact sentence in VerificationPanel |
| 7 | Flexibility and Efficiency | 3 | Smooth Enter/Shift-Enter typing; lacks keyboard escape (Esc) to close the right inspection drawer |
| 8 | Aesthetic and Minimalist Design | 3 | InputBar mode toggles flag contrast warnings (text-slate-400 on active indigo-500); EvidenceCard uses CSS width transition |
| 9 | Error Recovery | 4 | Real-time toast notifications on streaming drops with clean message state recovery |
| 10 | Help and Documentation | 4 | Helpful empty state: "Click any highlighted sentence to inspect it across three dimensions" |
| **Total** | | **34/40** | **Good (85%)** |

#### Design Specificity Verdict

**LLM Assessment**: Very strong domain craft. Unlike standard AI chatbots, HalluciSense uniquely transitions from raw streaming Markdown (Phase 1) to sentence-annotated risk inspection (Phase 2). The side-by-side drawer architecture allows instant drilldown into hybrid retrieval evidence without losing chat context.

**Deterministic Scan**: 7 detector warnings:
1. `frontend/src/components/chat/InputBar.tsx`: 5 contrast warnings for `text-slate-400`/`text-slate-200` on `bg-indigo-500` active mode pill.
2. `frontend/src/components/verification/EvidenceCard.tsx`: 1 AI-color-palette warning (`from-indigo-500 gradient`) and 1 layout-transition warning (`transition: width 0.6s ease`).

#### Overall Impression
A sophisticated two-phase live verification workspace. Asynchronous verification swapping provides rapid LLM streaming followed by deep factual and epistemic inspection.

#### What's Working
1. **Two-Phase Verification Flow**: Instant token streaming followed by non-blocking asynchronous verification annotation swap.
2. **Context-Preserving Side Inspector**: Clicking any sentence annotation immediately opens the `VerificationPanel` docked alongside the chat thread.
3. **Multi-Format Export**: 1-click Markdown and JSON export dropdown for saving verified transcripts.

#### Priority Issues
- **[P1] Mode Toggle Text Contrast in InputBar**: The active pill uses `bg-indigo-500` with `text-slate-400` / `text-slate-200`, causing low text contrast. It should use `text-white font-semibold` and active glow.  
  *Suggested command: `/impeccable colorize frontend/src/components/chat/InputBar.tsx`*
- **[P2] Risk Taxonomy Alignment in VerificationPanel**: Standardize the drawer's risk badges and circular gauge thresholds to the project's 4-tier taxonomy (Verified, Needs Verification, Moderate Risk, Likely Hallucinated).  
  *Suggested command: `/impeccable polish frontend/src/components/verification/VerificationPanel.tsx`*
- **[P2] Drawer Keyboard Escape (Esc)**: Pressing `Esc` when `VerificationPanel` is open should smoothly dismiss the side drawer.  
  *Suggested command: `/impeccable delight frontend/src/components/verification/VerificationPanel.tsx`*
- **[P3] EvidenceCard Progress Bar Optimization**: Replace layout-thrashing `transition: width` with transform-based GPU acceleration or solid tokenized meter styling.  
  *Suggested command: `/impeccable optimize frontend/src/components/verification/EvidenceCard.tsx`*

#### Persona Red Flags
- **Alex (Power User)**: Expects `Esc` to close the side verification drawer without having to reach for the top-right `X` button.
- **Jordan (First-Timer)**: Might wonder why the response text shifts slightly when verification annotations arrive (Phase 2 swap animation needs a smooth cross-fade).
- **Sam (Low-Vision / A11y)**: Mode toggle buttons in `InputBar` fail WCAG AA contrast when inactive text styles bleed into active states.

#### Minor Observations
- Chat auto-scrolls smoothly on every incoming token chunk.
- Responsive handling wraps mode buttons nicely on mobile viewports.

#### Questions to Consider
- Should we add a subtle pulsing indicator on the "Inspect Verification" button when new verification analysis completes?
- Would a 1-click "Re-verify with Custom Context" button inside the message bubble streamline iterative prompt tuning?
