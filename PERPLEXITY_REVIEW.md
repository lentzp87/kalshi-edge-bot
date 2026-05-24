# Kalshi Edge Bot — Technical Specification for Independent Review

**Purpose of this document:** A complete description of an automated
prediction-market trading bot, written so an independent analyst can
critique it. A companion `stats.json` file (current performance data)
should be reviewed alongside this spec.

**What we want from the review** (see "Questions for the Reviewer" at
the end):
1. **Strategy & edge critique** — is the core betting thesis sound, or
   are we fooling ourselves?
2. **Logic & assumption audit** — flawed assumptions, survivorship
   bias, structural mistakes.
3. **New ideas** — markets, edges, or techniques we haven't tried.

---

## 1. Overview

The bot trades **Kalshi**, a CFTC-regulated prediction-market exchange,
in **paper-trading mode** (simulated fills, no real money) with a
nominal **$2,000 bankroll**. It is hosted as a single Python process on
Render.

**Core thesis.** Kalshi sports markets ("Will TEAM X win?") are priced
by retail flow. Professional sportsbooks price the same games far more
sharply. If we take a sharp book's probability, strip its vig, and
compare to Kalshi's price, a persistent gap *might* represent an
exploitable edge. The bot:

1. Scans every open Kalshi sports market every 30 seconds.
2. For each, fetches a "fair" probability from a sharp source.
3. Computes edge = fair_probability − Kalshi_executable_price.
4. Applies a stack of filters (price, confidence, per-market fee gate).
5. Sizes a position with fractional Kelly.
6. Optionally boosts size when "whale" order-flow agrees.
7. Exits via take-profit or a time-based rule.

**Honest framing.** This is a research project. Lifetime paper P&L is
**negative** (currently ≈ −$210 over ~209 trades). Recent multi-day
windows have been positive then negative. A central open question for
the reviewer: *is there actually an edge here once Kalshi's fee is
paid, or is the negative P&L telling us the answer?*

---

## 2. Architecture

Single-process Python (asyncio). One `asyncio.gather` runs these
coroutines concurrently:

| Coroutine | Job |
|---|---|
| `trading_loop` | Scan → model → decide → execute, every 30s |
| `dashboard_server` | Embedded FastAPI dashboard (Whoop-styled UI) |
| `settlement_loop` | Backfill "held-to-settlement" P&L on closed trades |
| `kalshi_ws` | Read-only Kalshi WebSocket; feeds the whale fast-path |
| `cross_exchange_loop` | Kalshi-vs-Polymarket spread monitor (5 min) |
| `slack_digest_loop` | 6-hourly P&L digest to Slack |
| `golf_3ball_loop` | Golf matchup advisor (read-only, Slack) |
| `golf_leader_loop` | Live golf round-leader advisor (read-only) |

**Persistence.** SQLite (`trades_sports_v4.db`). Two tables: `signals`
(every model opinion, even untraded) and `trades` (opened/closed
positions with realized P&L, CLV, settlement counterfactuals,
intra-hold mid range).

**Orphan recovery.** On startup the bot reconciles any open positions
left behind by a previous deploy (Render redeploys kill in-process
watchers).

---

## 3. Data Sources (priority tiers)

| Tier | Source | Used for | Notes |
|---|---|---|---|
| 1 | **Pinnacle** (guest API) | NBA/NFL/MLB/NHL, tennis, cricket | Sharpest public book. Free. NBA/NFL/MLB/NHL via fixed league IDs; tennis (sport 33) + cricket (sport 8) via dynamic per-round league enumeration. |
| 2 | **The Odds API** | Multi-book median devig | Current subscription tier returns 401 for MLB, soccer, tennis — effectively dead for most sports. |
| 3 | **ESPN** | Pickcenter odds + in-game win probability | Single-book; also powers the in-game lag model. |
| — | **DataGolf** | Golf model + golf advisors | `preds/in-play`, `preds/pre-tournament`, and `betting-tools` (matchups, outrights). |
| — | **Polymarket Gamma** | Cross-exchange spread monitor | Read-only comparison, no trading. |

**Known constraint:** because The Odds API tier is mostly blocked, the
bot effectively runs on Pinnacle alone for team sports + tennis +
cricket, ESPN as fallback, DataGolf for golf. Soccer and UFC models
exist but are data-starved and effectively dormant.

---

## 4. Models

Models are dispatched by Kalshi's category. `SportsModel` is the entry
point for everything tagged "Sports" and sub-dispatches by series
prefix:

- **SportsModel (pregame)** — team sports (NBA/NFL/MLB/NHL). Fetches a
  book's home/away probabilities, de-vigs them, returns p_yes for our
  side. Only fires inside a pregame window (60 min – 12 h before tip).
  *WNBA was removed* (see §9).
- **InGameSportsModel** — for team sports already in progress. Uses
  ESPN's live win-probability. Thesis: ESPN's win-prob updates lag the
  true state slightly during late-game swings; Kalshi sometimes lags
  even more. Fires only in a "late game" window.
- **TennisModel** — Pinnacle moneyline for ATP/WTA, de-vigged.
- **CricketModel** — Pinnacle moneyline for IPL/T20I/ODI/Test, treated
  as effectively 2-way.
- **GolfModel** — DataGolf outright win probability (in-play first,
  pre-tournament fallback). Note: golf outright markets are an N-way
  market where almost every player is priced 1–15%, so the decision
  filters (below) reject nearly all of them — golf rarely trades.
- **UFCModel, SoccerModel** — exist, data-starved, effectively dormant.

**De-vigging.** Two-way markets: convert both sides' odds to implied
probabilities, normalize so they sum to 1. The normalization removes
the book's hold.

**Confidence.** Each model attaches a `confidence` scalar (team sports
~0.65–0.85, tennis ~0.65, etc.). It scales Kelly size.

---

## 5. Decision Layer

For a market with model probability `p_yes`:

1. **Executable price.** Edge is computed against the price we'd
   actually pay — `yes_ask` for a YES buy, `(1 − yes_bid)` for a NO buy
   — NOT the midpoint. (Using midpoint historically overstated edge by
   half the spread.)
2. **Pick the side** with positive gross edge.
3. **Filters, in order** (each logs a skip reason):
   - `min_entry_price` — side-specific floor (`min_entry_price_yes`,
     `min_entry_price_no`, both currently 0.35). Skip cheap entries
     where fee drag is brutal.
   - `min_p_yes` (0.52) — skip if our side's modeled probability is a
     coin flip or worse.
   - `max_p_yes` (0.70) — skip if our side's modeled probability is
     *too* high (see §9 — high-confidence picks empirically
     underperform).
   - **Per-market required-edge gate.** `gross_edge` must exceed
     `entry_fee + exit_fee + half_spread + slippage_buffer +
     safety_pp`. Kalshi's fee is `ceil(0.07 · p · (1−p) · 100)/100` per
     contract — it peaks near p=0.50 and is severe at low prices. This
     gate is per-market because the fee curve varies by price.
   - `min_edge` (−0.05) — a loose paper-mode floor on net edge.
4. **Sizing.** Fractional Kelly: `f* = edge / (1 − price)`, scaled by
   `kelly_fraction` (0.25) and by model confidence. Capped at
   `max_position_size_usd` ($20).

---

## 6. Whale Detector & Size Boost

Kalshi doesn't stream individual trades, so "whale" activity is
inferred from deltas between order-book snapshots.

**Three signal types:**

| Signal | Trigger | Direction |
|---|---|---|
| `price_jump` | `last_price` moves ≥ 5¢ vs a ~10s-old baseline | sign of move |
| `volume_burst` | 24h volume jumps ≥ 5,000 contracts | unknown ("?") |
| `resting` | top-of-book size ≥ 2,000 contracts (any of the 4 book sides) | implied by side |

**Baseline.** A per-ticker snapshot history; each new snapshot is
diffed against the one closest to (now − 10s). This makes the detector
work identically for 30s scanner polls and high-rate WebSocket ticks.

**Fast path.** A WebSocket tick → whale tracker → if a signal fires,
an out-of-band `model → decide → submit` runs immediately
(~1–3s latency) instead of waiting for the next 30s scan.

**Size boost.** When a whale signal's direction matches the side the
model already chose, position size is multiplied. The multiplier is
**continuous within each class**, scaled by signal magnitude:

| Class | Magnitude band | Multiplier | Status |
|---|---|---|---|
| aggressive (price_jump) | 5–7¢ | ~1.5–2.5× | **under review — losing** |
| aggressive | 7–10¢ | ~2.5–5× | strong performer |
| aggressive | 10¢+ | ~5× | tiny sample, mixed |
| burst (volume) | any | **1.0× (disabled)** | was losing |
| resting | 2–5k contracts | **1.0× (disabled)** | was losing |
| resting | 5–10k | ~1.2× | performer |
| resting | 10k+ | ~1.5× | strongest single signal |

Boosted size is capped at `whale_max_position_size_usd` ($50).

**Whale thesis.** Snapshot polling sees a whale *after* they've moved
the price, so following them blindly is "exit-liquidity cosplay." The
intended use is **confirmation**: if our model already says buy, and a
whale just bought the same side, the convergence of two independent
signals is what we size up on.

---

## 7. Exit Logic

| Exit | Rule | Status |
|---|---|---|
| `take_profit` | Close when position is +50% from fill | active |
| `time_exit` | Close at (tipoff − 5 min) | active |
| `stop_loss` | — | **DISABLED** (see §9) |

A background watcher marks each open position to market every 15s
(also rolling its max/min mid). A separate task samples the Kalshi mid
~5 min before tipoff to record **CLV** (closing-line value).

**Settlement backtester.** A background task pulls each closed trade's
eventual Kalshi resolution and computes a counterfactual "what if we
held to settlement" P&L. An A/B exit-cohort simulator then compares the
*actual* exit policy against `tp_only`, `sl_only`, `time_only`, and
`tp+sl` policies using the recorded mid range.

---

## 8. Current Configuration

```
bankroll_usd: 2000              mode: paper

decision:
  min_edge: -0.05               kelly_fraction: 0.25
  min_entry_price_yes: 0.35     min_entry_price_no: 0.35
  min_p_yes: 0.52               max_p_yes: 0.70
  required_edge_safety_pp: 0.005

risk:
  max_position_size_usd: 20     whale_max_position_size_usd: 50
  max_concurrent_positions: 15  max_open_exposure_pct: 0.40
  max_daily_loss_usd: 100       max_consecutive_losses: 3
  cooldown_minutes_after_kill: 60

execution:
  take_profit_pct: 0.50         stop_loss_pct: 1.00 (disabled)
  time_exit_minutes: 240        order_type: limit
  scale_in_chunks: 2

scanner:
  loop_interval_seconds: 30
  pregame window: 60 min – 12 h before tip
  price band: 0.10 – 0.90
```

---

## 9. Decision History (what we tried, what the data said, what we changed)

This is the evolution. Each change was driven by dashboard data.

1. **P&L sign bug on NO-side positions.** Early on, NO-side P&L was
   computed with an inverted sign — take-profit fired on losses. Fixed;
   journal DB rotated to discard corrupted data.

2. **Odds source replaced.** The original Odds API key died. Wired
   **Pinnacle** (guest API) as the Tier-1 source. Later discovered the
   Odds API subscription tier blocks most sports anyway.

3. **Stop-loss disabled.** The settlement backtester showed that across
   9 stop-loss exits, actual P&L was −$115 vs +$7 if held to
   settlement — the SL was firing on bid-ask whipsaw at fill (one trade
   stopped out at hold=0 min). The A/B panel later confirmed: the
   bot's actual exit policy beats every single-policy alternative
   (including any SL variant) by a wide margin. SL stays off.

4. **Filters loosened, then partially re-tightened.** To generate
   paper-mode volume we loosened `min_entry_price` (0.50 → 0.35) and
   `min_p_yes` (0.60 → 0.52), and widened the pregame window. The
   per-market required-edge gate was kept as the real cost floor.

5. **Whale detector built, then refined.** Added the three-signal
   detector; later replaced flat per-class multipliers with the
   continuous magnitude-scaled ladder; later still added NO-side book
   detection (we were only watching the YES book).

6. **New sports added.** Tennis (Pinnacle, dynamic per-round leagues),
   cricket (Pinnacle sport 8), golf (DataGolf). Tennis became the best
   surface; cricket the worst.

7. **WNBA removed.** 18 trades, 17% win rate, −$131. The model was
   structurally wrong about WNBA. Series dropped from discovery.

8. **Confidence ceiling added.** The `by_confidence` panel showed an
   *inverted* calibration: the model's highest-confidence picks won
   *least*. The 80%+ bucket: 36 trades, 33% win rate, −$285. We added
   `max_p_yes`, first 0.78, then 0.70, to stop trading the bands where
   the model is anti-predictive.

9. **Per-event dedup fix.** Tennis markets list both players as
   separate tickers; the bot traded both sides of the same match
   (a +$15 win then a −$33 loss on the mirror). Dedup now derives the
   event key from the ticker prefix.

10. **Single-trade risk capped.** One MLB trade (whale-boosted to $100)
    lost the full $100 — 35% of total P&L on one position.
    `whale_max_position_size_usd` cut 200 → 100 → 50.

11. **Whale classes pruned.** `burst` (volume) class disabled — 18
    trades, 44% wr, −$104. `rest_2-5k` (small resting) disabled — 26
    trades, 42% wr, −$49.

12. **CLV / mid-range tracking bug fixed.** A timestamp mismatch
    (`log_open` stamped a fresh timestamp; the CLV sampler and the
    mid-range watcher queried by the in-memory object's timestamp)
    meant every CLV write and every max/min-mid update silently
    affected zero rows. After 188 trades, `n_clv` was 0 and the A/B
    exit panel was degenerate (all policies identical). Fixed; both now
    populate (forward-only — old trades keep NULL columns).

---

## 10. Current Performance Snapshot (~209 trades, paper)

(The companion `stats.json` has the full breakdown — summary here.)

- **Lifetime:** ~209 trades, ~53.6% win rate, **≈ −$210 P&L**, avg
  ≈ −$1.00/trade.
- **A/B exit panel:** actual ≈ −$210 vs every single-policy
  alternative −$475 to −$586. The current exit logic is, by this
  measure, the best available.
- **CLV:** 10 samples (sampler recently fixed), **avg −4.05 bp**, 60%
  positive — early and negative.

**Profitable cohorts:**
- WTA tennis — 51 trades, 60% wr, +$197.
- ATP tennis — 41 trades, 58% wr, +$20.
- `take_profit` exits — 71 trades, 100% wr, +$1,108.
- Resting whales 10k+ — 56 trades, 67% wr, +$87.
- 15–60 min holds — 72 trades, 63% wr, +$244.
- Model-confidence 60–65% bucket — 24 trades, 66% wr, +$220.

**Losing cohorts:**
- 80%+ confidence — 36 trades, 33% wr, −$285 (frozen by the cap).
- Favorites (entry ≥ 50¢) — 106 trades, 43% wr, −$239.
- MLB — 76 trades, 56% wr, −$109.
- Cricket — 12 trades, 41% wr, −$135.
- `time_exit` exits — 123 trades, 30% wr, −$1,114.
- 1–4 h holds — 102 trades, 46% wr, −$532.
- `aggressive` whale, 5–7¢ band — 22 trades, 54% wr, −$85.

---

## 11. Open Questions We Have Not Resolved

These are the things we are genuinely unsure about — prime targets for
the review:

1. **Is there a real edge at all?** Lifetime paper P&L is negative. The
   thesis (devig a sharp book, bet Kalshi's disagreement) is plausible,
   but Kalshi's fee — `ceil(0.07·p·(1−p)·100)/100` per contract, both
   legs — is a heavy, price-dependent drag. Does the edge survive it?

2. **Inverted confidence calibration.** The model's *most confident*
   picks lose the most (80%+ bucket: 33% wr). A well-calibrated model
   should win ~80% of its 80% picks. Why is it inverted? Candidates:
   over-fitting heavy favorites; the favorite-longshot bias; selection
   effect (we only bet when our number diverges from the market, and
   on heavy favorites a divergence may mean *we're* wrong, not the
   market); something else.

3. **Negative CLV.** Early CLV samples average −4 bp — the closing line
   moves *against* us after we enter. If real, our entries are
   systematically mistimed or on the wrong side. How should we
   diagnose and fix this?

4. **The time_exit problem.** `time_exit` trades are −$1,114 over 123
   trades — but the A/B simulator says every alternative exit policy is
   *worse*. Interpretation: the trades that never hit take-profit are
   simply bad trades, and no exit rule saves them — the fix must be
   upstream (don't enter them). Is that interpretation correct?

5. **Favorites lose, underdogs win.** Favorites (entry ≥ 50¢): 43% wr,
   −$239. Underdogs: 64% wr, +$30. Is this the favorite-longshot bias,
   a model bug, or a fee artifact (favorites cost more, so fee is a
   larger % of a smaller potential gain)?

6. **Whale boost — confirmation or noise?** Resting 10k+ whales look
   genuinely predictive (67% wr). But is that causal, or are large
   resting orders just correlated with markets that are easier to model
   anyway? Is the whole whale apparatus adding value or adding variance?

7. **Paper vs live.** All results are paper fills at the posted price.
   Real Kalshi fills would face slippage and partial fills. How much of
   the (already negative) paper edge would survive live execution?

---

## 12. Questions for the Reviewer

**Strategy & edge critique**
- Is "de-vig a sharp book, bet the exchange's disagreement" a viable
  edge on Kalshi specifically, given its fee structure? Or is the
  fee fatal for anything but very large edges?
- The bot's one consistently profitable surface is tennis (Pinnacle
  devig). Is there a structural reason tennis would be more beatable
  than team sports, or is that likely noise?
- Should a bot like this concentrate on far fewer, higher-conviction
  trades rather than casting a wide net?

**Logic & assumption audit**
- The inverted confidence calibration (Open Question #2) — what is the
  most likely cause, and how would you test for it?
- Negative CLV (#3) — what does it imply, and what's the cleanest
  diagnostic?
- Is disabling the stop-loss defensible, or is the settlement-backtest
  reasoning flawed (e.g. survivorship in which trades reached
  settlement)?
- The whale detector infers order flow from snapshot deltas. Is that
  methodologically sound, or is it picking up noise that happens to
  correlate with outcomes in a small sample?
- Any survivorship or look-ahead bias in how edges/CLV/settlement are
  measured?

**New ideas**
- What markets or edges on Kalshi (or adjacent exchanges) would you
  prioritize that this bot isn't touching?
- Is the in-game / live-lag angle (ESPN win-prob vs Kalshi price)
  more promising than the pregame devig angle?
- Given negative lifetime P&L, what would you change *first*?

---

*End of spec. Companion file: `stats.json` (current performance data).*
