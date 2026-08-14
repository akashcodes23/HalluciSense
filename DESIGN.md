---
name: HalluciSense
description: Scientific hallucination detection & confidence calibration framework
colors:
  primary: "#a855f7"
  primary-hover: "#b86df9"
  secondary: "#3b82f6"
  accent-indigo: "#6366f1"
  neutral-bg: "#0a0a0c"
  neutral-surface: "#131316"
  neutral-tertiary: "#1b1b22"
  status-success: "#10b981"
  status-warning: "#f59e0b"
  status-danger: "#ef4444"
  text-primary: "#f8fafc"
  text-secondary: "#94a3b8"
  text-muted: "#475569"
  border-default: "rgba(255, 255, 255, 0.04)"
  border-hover: "rgba(255, 255, 255, 0.08)"
typography:
  display:
    fontFamily: "var(--font-space-grotesk), system-ui, sans-serif"
    fontSize: "56px"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.03em"
  headline:
    fontFamily: "var(--font-space-grotesk), system-ui, sans-serif"
    fontSize: "40px"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.02em"
  title:
    fontFamily: "var(--font-space-grotesk), system-ui, sans-serif"
    fontSize: "20px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.01em"
  body:
    fontFamily: "var(--font-sans), system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "normal"
  label:
    fontFamily: "var(--font-mono), monospace"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: 1.5
    letterSpacing: "0.05em"
rounded:
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  full: "9999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.md}"
    padding: "10px 20px"
  button-secondary:
    backgroundColor: "{colors.border-default}"
    textColor: "{colors.text-secondary}"
    rounded: "{rounded.md}"
    padding: "10px 20px"
  badge-default:
    backgroundColor: "{colors.border-default}"
    textColor: "{colors.text-secondary}"
    rounded: "{rounded.full}"
    padding: "4px 12px"
  card-glass:
    backgroundColor: "{colors.neutral-surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.lg}"
    padding: "24px 24px"
---

# Design System: HalluciSense

## Overview

**Creative North Star: "The Precision Observatory"**

HalluciSense visualizes multi-pillar AI hallucination analysis through the lens of a deep-space astronomical observatory. The canvas is a pitch-dark, vacuum-like obsidian void (`#0a0a0c`) upon which translucent frosted-glass instrumentation panels float with quiet authority. Telemetry readouts, confidence gauges, and attention heatmaps emerge with spectral clarity, using curated violet, cobalt, and four-tier risk hues to indicate statistical certainty without visual noise.

Every interface element feels engineered, calibrated, and reproducible. Monospace quantitative readouts ground abstract model behaviors in hard empirical data, while soft ambient gradients simulate the subtle glow of high-precision diagnostic monitors.

**Key Characteristics:**
- **Deep Space Atmosphere:** Infinite dark backdrop with layered translucent glass surfaces (`backdrop-blur-xl`).
- **Tactile Scientific Instrumentation:** Rounded-xl panels, crisp 1px borders with subtle hover highlights, and dedicated status edge accents.
- **Spectral Signal Precision:** Vibrant primary accents (`#a855f7`, `#3b82f6`) and strict 4-tier risk tokens for instantaneous cognitive scanning.
- **Mathematical Typography:** Space Grotesk for architectural titles paired with high-contrast Monospace metrics.

## Colors

The color palette balances deep space darkness with luminous spectral indicators designed for rigorous data visualization and maximum perceptual contrast.

### Primary
- **Spectral Violet** (`#a855f7`): Primary brand accent used for top-level call-to-actions, active navigation states, and primary confidence highlights.
- **Electric Cobalt** (`#3b82f6`): Secondary action accent used for retrieval indicators, interactive hyperlinks, and secondary buttons.

### Secondary
- **Accent Indigo** (`#6366f1`): Transition gradient anchor and tertiary telemetry highlights.

### Neutral
- **Obsidian Canvas** (`#0a0a0c`): Deep root background representing the observational void.
- **Nocturne Surface** (`#131316`): Base card and sidebar container surface at 85% opacity.
- **Starlit Text** (`#f8fafc`): Primary high-contrast typography for headings and quantitative values.
- **Slate Dim** (`#94a3b8`): Secondary descriptive typography and icon fills.
- **Muted Void** (`#475569`): Inactive labels, disabled controls, and subtle chart gridlines.

### Status (4-Tier Risk Taxonomy)
- **Calibrated Emerald / Verified** (`#10b981`): High confidence, verified evidence entailment.
- **Amber Warning / Needs Verification** (`#f59e0b`): Marginal confidence, incomplete evidence support.
- **Tangerine Orange / Moderate Risk** (`#f97316`): High entropy, elevated uncertainty detected.
- **Rose Alert / Likely Hallucinated** (`#ef4444`): Contradiction, severe hallucination risk detected.

### Named Rules
**The Signal Rarity Rule.** Vibrant primary accents (`#a855f7` / `#3b82f6`) are reserved for active actions, key confidence metrics, and focus states, occupying $\le 10\%$ of any given viewport. Its rarity preserves focal contrast.

**The Semantic Invariance Rule.** Status hues (`#10b981`, `#f59e0b`, `#f97316`, `#ef4444`) are strictly reserved for verification risk classifications and must never be repurposed for generic decorative styling.

## Typography

**Display Font:** Space Grotesk (with system-ui, sans-serif fallback)  
**Body Font:** Geist Sans / Inter (with system-ui, sans-serif fallback)  
**Label/Mono Font:** JetBrains Mono / SF Mono (with monospace fallback)  

**Character:** Technical, crisp, and authoritative. Space Grotesk gives structural identity to scientific section titles, while monospace elements deliver unambiguous numerical precision.

### Hierarchy
- **Display** (Bold 700, 56px / `clamp(2.5rem, 5vw, 3.5rem)`, line-height 1.1, tracking `-0.03em`): Hero headlines on marketing and landing surfaces.
- **Headline** (Bold 700, 40px, line-height 1.15, tracking `-0.02em`): Main view headers and major architectural section titles.
- **Title** (SemiBold 600, 20px–28px, line-height 1.2–1.3, tracking `-0.01em`): Card headers, modal titles, and widget banners.
- **Body** (Regular 400, 16px, line-height 1.6, max line length 70ch): Paragraphs, descriptions, analysis narrative.
- **Label** (SemiBold 600, 12px, tracking `0.05em`, uppercase font-mono): Metric captions, table column headers, and telemetry indicators.

### Named Rules
**The Tabular Metric Rule.** All floating-point scores, execution latencies, and probability values must be rendered in `font-mono` with tabular numbers to eliminate layout jitter during live streaming.

## Layout

The spatial model relies on a 12-column responsive grid with a standard 8px spatial rhythm (`spacing: 8px, 16px, 24px, 32px`). 
- **Application Shell:** Fixed 260px left sidebar navigation with a 64px sticky glass header.
- **Content Max-Width:** Max 1200px container for dashboard workspaces (`/verify`, `/benchmark`, `/metrics`), with flexible fluid side-by-side split panels on desktop (`md:` breakpoint $\ge 768\text{px}$).
- **Spacing Density:** High information density with compact padding (`p-4` to `p-6`) inside analytical cards.

## Elevation & Depth

HalluciSense employs layered glassmorphism with subtle tonal depth rather than opaque drop shadows. Surfaces rest flat on the obsidian background and lift via ambient glow and border luminescence upon interaction.

### Shadow & Glow Vocabulary
- **Card Rest:** `border: 1px solid rgba(255, 255, 255, 0.04)`, `backdrop-filter: blur(20px)`.
- **Active Card Glow:** `border: 1px solid rgba(168, 85, 247, 0.4)`, `box-shadow: 0 0 20px rgba(168, 85, 247, 0.1)`.
- **Button Primary Glow:** `box-shadow: 0 0 0 1px rgba(37,99,235,0.4), 0 4px 16px rgba(37,99,235,0.3)`.
- **Button Hover Glow:** `box-shadow: 0 0 0 1px rgba(37,99,235,0.6), 0 6px 24px rgba(37,99,235,0.45)`.

### Named Rules
**The Luminescent Border Rule.** Depth is primarily established by micro-contrast border luminance (`4%` resting $\rightarrow$ `15%` hover $\rightarrow$ `40%` active accent) rather than heavy drop shadows.

## Shapes

- **Base Radius:** 16px (`rounded-2xl` / `rounded-xl`) for cards, modal dialogs, and workspace containers.
- **Control Radius:** 12px (`rounded-xl`) for buttons, text inputs, and select dropdowns.
- **Chip & Badge Radius:** Full pill radius (`rounded-full` / `9999px`) for metadata tags and status chips.
- **Border Treatment:** Ultra-thin 1px solid translucent borders across all container perimeters.

## Components

### Buttons
- **Shape:** Rounded-xl (12px radius).
- **Primary:** Background `#a855f7` or `#2563eb`, white text, 10px 20px padding (`h-10 px-5`), subtle box-shadow glow.
- **Hover / Focus:** `-translate-y-0.5`, enhanced glow shadow, 2px focus ring with `#050816` ring offset.
- **Secondary / Ghost:** `bg-white/[0.04]` border `white/[0.08]` text `slate-300`, hover `bg-white/[0.08] text-white`.

### Cards & Glass Panels (`GlassCard`)
- **Corner Style:** 16px–24px radius (`rounded-2xl`).
- **Background:** `bg-[#131316]/85` with `backdrop-blur-xl`.
- **Border:** `1px solid rgba(255, 255, 255, 0.04)`.
- **Status Indicator:** 4px solid left border colored by risk tier (`border-l-4 border-l-status-*`).

### Status Badges (`Badge` / `StatusBadge`)
- **Shape:** Full pill shape (`rounded-full`).
- **Style:** 10% opacity background tint, 20% opacity matching border, high-contrast text color (`text-xs font-semibold`).
- **Variants:** Verified (emerald), Warning (amber), Danger (red), Info (sky), Primary (blue), Purple (violet).

### Inputs & Fields
- **Style:** `bg-[#131316]`, `border: 1px solid rgba(255, 255, 255, 0.08)`, text `#f8fafc`, `rounded-xl`.
- **Focus State:** `border-accent-primary/50`, `ring-2 ring-accent-primary/20`, outline none.

### Token Heatmaps
- **Style:** Inline token span highlights with 15% opacity background and 80% lightness text (`.token-green`, `.token-yellow`, `.token-orange`, `.token-red`).

## Do's and Don'ts

### Do:
- **Do** wrap tabular floating-point readouts and latencies in `font-mono`.
- **Do** maintain the 4-tier risk color taxonomy consistently across graphs, badges, cards, and token heatmaps.
- **Do** use `GlassCard` with `backdrop-blur-xl` and 1px translucent borders for all content containers.
- **Do** provide smooth transitions (`duration-200` to `duration-300`) on hover and active states.

### Don't:
- **Don't** use opaque solid white or light-gray backgrounds for cards on dark views.
- **Don't** create aggressive multi-colored drop shadows that detract from data readability.
- **Don't** use status colors (emerald, amber, red) for non-status decorative accents.
- **Don't** remove monospace alignment from quantitative evaluation scores.
