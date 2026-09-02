# Phase 54A — Implementation Plan: HalluciSense Premium 3D Frontend Transformation

## 1. Current Architecture
- **Framework**: Next.js 16.2.11 (App Router) with React 19.2.4 and TypeScript 5.
- **Styling**: Tailwind CSS v4 with custom design tokens in `globals.css` and CSS variables (`--bg`, `--primary`, `--verified`, etc.).
- **Component System**: Radix UI primitives, Lucide icons, Framer Motion for micro-interactions, Recharts for charts.
- **State Management**: Zustand stores (`analysis-store.ts`, `uiStore.ts`, `authStore.ts`, `chatStore.ts`).
- **Telemetry & API**: TanStack React Query + Axios for API communication with FastAPI backend.

---

## 2. Landing Page Entry Point
- **Path**: `frontend/src/app/page.tsx`
- **Current Layout**: Fixed Navbar, hero header with static text, 3 feature pillars cards, metrics ticker, pipeline architecture flowchart, CTA banner, and footer.

---

## 3. Dashboard Route Structure
- **Route Group**: `frontend/src/app/(dashboard)/`
- **Layout**: `frontend/src/app/(dashboard)/layout.tsx` providing desktop `AppSidebar`, `TopBar`, and mobile bottom navigation.
- **Key Dashboard Routes**:
  * `/overview`: Telemetry dashboard, metric cards, verification status feed.
  * `/verify`: Interactive verification workbench, claim breakdown, evidence view.
  * `/chat`: Real-time streaming conversational interface with live sentence-level hallucination scoring.
  * `/traces`: Deep execution traces, latency timelines, and pillar score breakdowns.
  * `/errors`: Anomaly and failure taxonomy feed.
  * `/benchmark`, `/analytics`, `/scientific`, `/inspector`, `/admin`, `/settings`.

---

## 4. Existing Zustand Stores
- `frontend/src/store/analysis-store.ts`: Authoritative store for current analysis result, explain responses, history, error feed, sidebar state, active tab.
- `frontend/src/stores/uiStore.ts`: Slide-over inspector panel state and active sentence selection.
- `frontend/src/stores/chatStore.ts`: Chat session history and active conversation streams.
- `frontend/src/stores/authStore.ts`: Session tokens and user authentication.

---

## 5. Existing Reusable UI Components
- `frontend/src/components/ui/button.tsx`: Multi-variant button.
- `frontend/src/components/ui/card.tsx`: `GlassCard`, `MetricCard`, `SurfaceCard`.
- `frontend/src/components/ui/badge.tsx`: Status and variant badges.
- `frontend/src/components/shell/top-bar.tsx`: Global top bar with status pills and search trigger.
- `frontend/src/components/layout/app-sidebar.tsx`: Collapsible navigation sidebar.

---

## 6. Existing Styling System
- Design language: High-density dark mode (`#09090b` foundation, `#111113` surface, `#18181b` raised).
- Color accents: Emerald (`#10b981`), Teal (`#14b8a6`), restrained Cyan highlights, subtle white border opacities (`rgba(255,255,255,0.06)`).
- Fonts: `Inter` (sans), `Space Grotesk` (display headlines), `JetBrains Mono` (code/numbers).

---

## 7. Planned 3D Architecture
- **Component**: `frontend/src/components/Hero3DCanvas.tsx`
- **Renderer**: React Three Fiber (`@react-three/fiber`) + Drei (`@react-three/drei`) + Three.js (`three`).
- **Scene Elements**:
  1. *Wireframe Gyroscope*: Nested geometric rings (torus/ring geometries) with independent axis rotation and restrained emissive wireframe styling.
  2. *Holographic Neural Cognition Core*: Procedural point cloud / particle network (500–800 nodes) with subtle breathing pulsation and interconnected line segments.
  3. *Lighting*: Restrained emerald point light (`#10b981`), teal point light (`#14b8a6`), subtle ambient light.
- **Hero State Dynamic Coupling**:
  * `"detect"`: Gyroscope dynamic rotation, neural node excitement, subtle outward signal propagation.
  * `"confidence"`: Stabilized rotation, ordered particle convergence, contracted focal glow.
  * `"verify"`: Orbital alignment, sequential line illumination, focused scientific verification pulse.

---

## 8. Performance Strategy & Unmount Safeguards
- **SSR Safety**: `Hero3DCanvas` imported dynamically in `page.tsx` with `next/dynamic` and `{ ssr: false }`.
- **Dashboard Separation**: `Hero3DCanvas` is rendered strictly on `page.tsx`. Navigating to `/(dashboard)/*` completely unmounts the canvas and halts WebGL RAF loops.
- **Resource Disposal**: React Three Fiber automatically cleans up geometry and material GPU buffers upon unmount.
- **Framerate & DPR**: Capped at `dpr={[1, 1.5]}` with fixed particle counts to ensure 60 FPS on standard modern laptops without thermal throttle.
- **Reduced Motion**: Respects `prefers-reduced-motion`, stopping automatic state cycling and continuous rotation.

---

## 9. Files to Modify / Create
1. `frontend/implementation_plan.md` [NEW]
2. `frontend/src/components/Hero3DCanvas.tsx` [NEW]
3. `frontend/src/store/analysis-store.ts` [MODIFY: Add `activeHeroState`, `setActiveHeroState`]
4. `frontend/src/app/page.tsx` [MODIFY: Integrate Hero3DCanvas with Framer Motion typography, hero state machine, and cinematic composition]
5. `frontend/PHASE54A_3D_FRONTEND_REPORT.md` [NEW]

---

## 10. Files Intentionally Left Untouched
- All backend ML engines, classifiers, scalers, and verification pipelines in `backend/`.
- All dashboard operational views: `(dashboard)/overview`, `(dashboard)/verify`, `(dashboard)/chat`, `(dashboard)/traces`, `(dashboard)/errors`, etc.
- Backend API routes and schemas.
