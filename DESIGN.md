---
name: Edge Monitor
description: Kalshi trading bot dashboard inspired by the WHOOP health app
version: alpha

colors:
  # — Surface stack (darker → lighter tonal layers)
  background: "#0A0F14"          # canvas; deep slate-black with cool tint
  surface: "#151A21"             # primary card / panel background
  surface-raised: "#1E242D"      # nested element inside a card (dial track,
                                 # input field, inner divider)
  surface-overlay: "#252B35"     # hovered card / pressed state
  border: "#2A313A"              # 1px hairlines between cards / table rows

  # — Text
  text: "#FFFFFF"                # primary number / headline
  text-secondary: "#9CA3AF"      # uppercase labels, secondary copy
  text-muted: "#5E6470"          # captions, comparison baselines, dim help
  text-disabled: "#3A3F47"

  # — Semantic palette (state-driven, NOT decorative)
  primary: "#3DA5F5"             # WHOOP-blue; current-state info, dial fill
                                 # when the value isn't a win/loss judgement
  primary-soft: "#1E5485"        # dial track / chart-fill background
  accent: "#7C9EFF"              # secondary highlights, floating action

  success: "#3DD68C"             # winning trades, "within range" badges
  success-soft: "#143D27"        # success pill background
  danger: "#FF4D4D"              # losing trades, alerts, "out of range"
  danger-soft: "#4B1414"         # danger pill background
  warning: "#FBB454"             # caution, deltas vs 30-day baseline
  info: "#5BC0EB"                # informational neutral (e.g. CLV samples)

  # — Sport / surface accents (low-saturation, used in tags only)
  tennis: "#9D7CF7"              # purple — winning surface
  cricket: "#F59E3D"             # amber
  mlb: "#5BC0EB"                 # light blue
  hockey: "#7C9EFF"              # ice blue
  basketball: "#FF8A3D"          # orange

typography:
  # Inter is the free font closest to WHOOP's display face. Tight
  # tracking + heavy weight on numbers reproduces the look.
  hero:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: 3.25rem            # 52px — the iconic "73%" dial number
    fontWeight: 700
    letterSpacing: "-0.02em"
    lineHeight: 1.0
  display:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: 2.25rem            # 36px — secondary big numbers (17.6 rpm)
    fontWeight: 700
    letterSpacing: "-0.02em"
    lineHeight: 1.1
  unit:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: 1rem               # 16px — "%" / "rpm" / "bpm" suffix
    fontWeight: 500
    letterSpacing: "-0.01em"
  h1:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: 1.75rem            # 28px — page titles ("My Day")
    fontWeight: 600
    letterSpacing: "-0.01em"
  label:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: 0.6875rem          # 11px — uppercase tags ("HEART RATE")
    fontWeight: 600
    letterSpacing: "0.12em"
    textTransform: uppercase
  body:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: 0.9375rem          # 15px
    fontWeight: 400
    lineHeight: 1.5
  caption:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: 0.75rem            # 12px — baselines / comparisons
    fontWeight: 400
  mono:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: 0.8125rem          # 13px — tickers, IDs

rounded:
  sm: 6px                        # pill badges, small chips
  md: 10px                       # buttons, inputs
  lg: 16px                       # cards, panels (WHOOP's default)
  xl: 24px                       # modal / sheet
  full: 9999px                   # circular dials, avatar, FAB

spacing:
  xs: 4px                        # half-step
  sm: 8px                        # base unit (WHOOP uses 8px scale)
  md: 16px                       # card padding interior
  lg: 24px                       # gap between sections
  xl: 32px                       # outer page padding
  xxl: 48px                      # large hero spacing

components:
  card:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.lg}"
    padding: "{spacing.md}"
    borderColor: "transparent"   # depth via tonal layer, NOT border
  card-bordered:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.lg}"
    padding: "{spacing.md}"
    borderColor: "{colors.border}"
  dial:                          # the iconic Whoop circular ring
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.full}"
    trackColor: "{colors.surface-raised}"
    activeColor: "{colors.primary}"
    strokeWidth: 6px
  metric-card:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.lg}"
    padding: "{spacing.md}"
  pill-success:
    backgroundColor: "{colors.success-soft}"
    textColor: "{colors.success}"
    rounded: "{rounded.sm}"
    padding: "4px 8px"
  pill-danger:
    backgroundColor: "{colors.danger-soft}"
    textColor: "{colors.danger}"
    rounded: "{rounded.sm}"
    padding: "4px 8px"
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "#0A0F14"
    rounded: "{rounded.md}"
    padding: "10px 16px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.text}"
    rounded: "{rounded.md}"
    padding: "10px 16px"
  nav-tab:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-secondary}"
    rounded: "{rounded.xl}"
    padding: "12px 16px"
---

## Overview

**Edge Monitor** — premium dark-mode dashboard for a Kalshi paper-trading
bot. The UI evokes a high-end performance app (WHOOP, Linear, Stripe
Atlas): authoritative, restrained, data-first. Every screen is built
around one or two **hero numbers** in massive type, supported by smaller
context that explains *why this number matters right now*.

The bot is a tool for decision-making, not entertainment. The UI should
never compete with the data — it should disappear around it. No
gradients on text, no decorative illustrations, no celebratory
animation on wins. Information density is high but reads calm because
of generous spacing and a strict 3-tier text hierarchy (white →
gray-400 → gray-500).

Emotional target: *"this bot is run by a quantitative fund."*

## Colors

The palette is built around four roles. Every color in the system
belongs to exactly one of them.

- **Surface stack** (`background`, `surface`, `surface-raised`,
  `surface-overlay`) — four steps of dark slate. Depth is conveyed by
  stacking these, never with a drop shadow. The home canvas is
  `background`; cards sit on it in `surface`; dial tracks and table
  rows live inside cards in `surface-raised`.

- **Text** (`text`, `text-secondary`, `text-muted`, `text-disabled`) —
  four steps of luminance. A typical card has the value in `text`,
  the label in `text-secondary`, and the baseline/comparison in
  `text-muted`. Never use more than three text colors per card.

- **State** (`success`, `danger`, `warning`, `info`, `primary`) —
  color is a signal, not decoration. A `+$70.56` cell is green
  because the P&L is positive, not because P&L cells are green.
  Use `primary` (WHOOP blue) for "this is the current value" when
  the value isn't a win/loss judgement (e.g. open-position fill price).

- **Surface accents** (`tennis`, `cricket`, `mlb`, etc.) — only used
  as small chips on the trade row to tag the sport. Never used as
  fill on a chart or as background of a card.

## Typography

One typeface across the entire app: **Inter** (or Inter Display for
the very largest weights). Inter at heavy weight with tight tracking
is the closest free analog to WHOOP's custom face. JetBrains Mono is
used **only** for tickers, hashes, and IDs.

The visual hierarchy is brutal on purpose:

- **Hero numbers** dominate. `52px / 700-weight / -0.02em tracking`.
  Used for the current-state value at the top of any view: P&L,
  win rate, open count.
- **Display numbers** (`36px`) for secondary big numbers — the inner
  cells of the metric grid.
- **Labels** are always uppercase, `11px`, `0.12em letter-spacing`,
  `text-secondary`. They sit *above* the value they describe.
- **Body** is `15px` regular. Used sparingly.
- **Captions** (`12px`) for the always-present comparison line:
  *"vs last 30 days"*, *"within range"*, *"baseline 0.65"*.

A hero number's unit (`%`, `bpm`, `$`) is rendered at `unit` style
*next* to the number, not below it. The number is the subject.

## Layout

- **Canvas:** 16px outer padding on mobile, 32px on desktop. Max
  content width 1200px, centered.
- **Grid:** 8px base unit with 4px half-step. All spacing values
  multiples of 8 except for micro-adjustments.
- **Hero row:** the home screen leads with a row of **three circular
  dials** at the top — the canonical WHOOP pattern. Each dial shows
  one hero number with a partial-arc fill indicating progress vs. a
  reference. For Edge Monitor these are:
    1. **Today's P&L** (dial fill = -100% to +100% scaled)
    2. **Win Rate** (dial fill = 0-100%)
    3. **Open Positions** (dial fill = 0-15, the position cap)
- **Card grid:** below the dials, a 2-column grid of metric cards
  on mobile, 3-4 columns on desktop. Each card is one self-contained
  number + label + status pill.
- **Deep-dive page** layout: hero dial at top → context table below
  → weekly trends → drill-down sections. (Mirrors WHOOP's Health
  Monitor → strain detail flow.)
- **Bottom nav** (mobile): pill-shaped floating bar with 4 tabs:
  Home · Trades · Whales · More. Replaces the FastAPI dashboard's
  current flat tabbed layout.

## Elevation & Depth

- **No drop shadows anywhere.** Depth is conveyed entirely by
  tonal contrast between layers of the surface stack.
- The dial uses the same trick — track is `surface-raised`,
  active arc is `primary` (or `success`/`danger` if semantic),
  the only visual cue is luminance contrast.
- Hover/press states *brighten* the background to `surface-overlay`
  rather than introducing a shadow or a border.
- The single exception: the floating-action button at the bottom-right
  (e.g. "force digest") gets a soft outer glow in `primary` at 30%
  opacity — pure WHOOP move.

## Shapes

- **Card radius is 16px** everywhere. This is the dashboard's
  signature corner. Never mix radii on the same screen.
- **Pills** (status badges, sport tags) are 6px — small enough to
  read as a different element class than cards.
- **Dials** are perfectly circular (rounded-full), stroke 6px, with
  a 270° active range starting at the bottom-left (WHOOP starts at
  the top — we deliberately match this).
- **Inputs and buttons** use 10px (between pill and card) so they
  feel tappable without being playful.
- **Bottom nav** uses 24px (larger than cards) so it reads as a
  floating element rather than another panel.

## Components

Defined as token-references in the front matter (see `components` block).
A few component-specific notes:

- **Hero dial** — circular ring with the value at center, label
  below. Ring fill color is *semantic by metric*: P&L dial is
  green when positive / red when negative / muted when flat;
  WinRate dial is the same; Open Positions dial uses `primary`
  (informational).

- **Metric card** — uppercase label at top-left, optional icon
  beside it. Hero or display number in the middle. Status pill
  at the bottom (`within range` / `over baseline` / `+12% vs 7d`).
  The trio (label / number / status) is the heart of the language.

- **Trade row** (Recent Trades table) — uses `mono` for ticker,
  `body` for matchup, `display` number for P&L, sport accent chip
  on the right. Background `surface`, dividers `surface-raised`
  (never visible border lines on rows).

- **Status pill** — single-purpose component: `text-success-soft
  background + text-success foreground` for in-range/positive,
  `danger-soft + danger` for out-of-range/negative. Always paired
  with an icon (check ✓ or alert !).

- **Sparkline** — used inside the heart-rate-style hero on the
  detail pages. Single 2px-stroke line in `primary`, with a single
  end-point dot. No grid, no axes, no labels — pure signal.

- **Bar chart** — column charts (steps/calories analog) show values
  labeled *above each bar*. Bars use `primary`. Today's bar is
  visually demoted with a darker fill to indicate "still in progress."

- **Weekly trend chart** — same shape as WHOOP's STRAIN weekly bars.
  X-axis = day labels in `caption`. Today is highlighted by a
  faint vertical band behind the bar.

## Do's and Don'ts

**Do:**
- Make the hero number 4-6x bigger than its label.
- Use ONE color of meaning per card. If P&L is green, don't also
  put green on the win-rate inside the same card.
- Always show a comparison or baseline ("vs 30d", "vs fill",
  "within 72-75 bpm"). A naked number is a missed opportunity.
- Right-align numbers in tables; left-align text labels.
- Reserve `success` and `danger` for outcomes, not for buttons
  or links.
- Tag every trade row with a sport-accent chip so the eye can
  scan by sport without reading.
- Use the `?` chevron pattern (e.g. `HEALTH MONITOR ›`) for any
  card that drills into a detail page.

**Don't:**
- Don't use drop shadows. Use the surface stack.
- Don't gradient text or numbers. Solid colors only.
- Don't put more than three text colors in one card.
- Don't put a border around a card unless the card is interactive
  on hover (use `card-bordered` only for clickable previews).
- Don't animate wins (no green flash, no confetti). Numbers
  update silently.
- Don't use sport-accent colors as backgrounds. They live only
  on chips/tags.
- Don't write a card without an uppercase label above the number.
  The label is required — the eye finds the label first.
- Don't show charts with axis labels and grid lines unless
  precision matters. WHOOP charts show values *on* bars and no
  grid — copy that.
- Don't use rounded-full on anything except dials, avatars, and
  the FAB. Pills are rounded-sm, not full.
