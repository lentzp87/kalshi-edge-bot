# Kalshi Edge Bot — Session Handoff

You're picking up an in-flight project. Read this start-to-finish before doing anything. The most important section is the **Active Investigation** — there's an unresolved bug that gates the next real-money decision.

---

## What this is

A Python trading bot for Kalshi prediction markets. Edge thesis: Kalshi prices sometimes disagree with Pinnacle (universally regarded as the sharpest sportsbook), and the bot systematically takes the side Pinnacle favors after de-vigging. Runs on Render, paper mode, $2,000 paper bankroll.

Repo: `/Users/patlentz/Desktop/Newbot` (this directory). Deployed via `git push` → Render autoDeploy. The live SQLite journal lives at `/var/data/trades_sports_v4.db` on Render's persistent disk.

---

## Current state (as of session end, 2026-06-10)

```
Total P&L:      +$309.97 over 362 trades
Win rate:       60.2%
Avg per trade:  +$0.86
```

**By sport (lifetime):**

```
KXWTAMATCH     n=86    +$275.51   wr 62%   ← strongest tennis
MLB            n=163   +$231.95   wr 65%   ← biggest engine
KXATPMATCH     n=62    +$101.84   wr 61%   ← best CLV quality
NBA            n=9     +$12.65    wr 78%   (Finals fired well, tiny sample)
KXPGATOUR      n=1     +$0.31              (n=1, ignore)
KXCRICKETTEST  n=3     -$58.40    wr 33%   (paused, tiny sample)
KXIPLGAME      n=13    -$60.65    wr 54%   (CUT)
NHL            n=7     -$61.70    wr 29%   (cut candidate; small sample)
KXWNBAGAME     n=18    -$131.53   wr 17%   (CUT)
```

**CLV (closing-line value):**

```
ATP:    n=17   +6.65bp   76.5% positive   ← professional-grade
WTA:    n=21   +4.62bp   66.7%
MLB:    n=70   +0.05bp   57.1%            ← marginal
NBA/NHL/IPL:   negative across small samples
```

WTA CLV was negative until 2026-05-31. We discovered an empty-orderbook fallback bug that was writing 0.5 as a fake "closing price" for thin WTA markets. Fixed the sampler + ran a backfill SQL that nulled the polluted rows. WTA CLV is now real and positive.

---

## Active investigation — DO THIS FIRST

We added a TP fillability tracker so the bot logs `realistic_exit_price` (book-sweep avg) alongside the optimistic paper `exit_price`. The gap measures how much of our paper profit would survive real fills.

**The problem:** every single exit since the deploy shows `exit_book_size = 0` and `realistic_exit_price = NULL`. Across all 48 instrumented exits.

Two possibilities:

- **A. Real one-sided books.** Exits genuinely happen against ghost liquidity — paper P&L is partially fictional. Bad news.
- **B. Parser bug.** The Kalshi orderbook field format isn't what `_realistic_fill_from_orderbook` expects (it reads `entry[1]` as size). Fixable in one edit.

**Until this is resolved, do NOT recommend going live with anything.** The user pushed for "go live with everything" — I declined. The +$310 is suspect until we know which scenario A or B we're in.

### Next concrete step

Have the user run this in Render Shell:

```
cat > /tmp/probe.py << 'EOF'
import asyncio, json, sys, os
ROOT = '/home/render/project'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src.kalshi_client import KalshiClient
async def main():
    c = KalshiClient()
    for t in [
        'KXATPMATCH-26JUN10ALCSIN-ALC',
        'KXMLBGAME-26JUN10ATLLAD-ATL',
        'KXWTAMATCH-26JUN10SABMUS-SAB',
    ]:
        try:
            ob = await c.get_orderbook(t)
            if ob:
                print('=== ticker:', t)
                print(json.dumps(ob, indent=2)[:2500])
                break
        except Exception as e:
            print(f'{t}: {type(e).__name__}: {str(e)[:100]}')
    await c.aclose()
asyncio.run(main())
EOF
python3 /tmp/probe.py
```

The output reveals the actual Kalshi orderbook schema. Look at the entries in `yes_dollars` and `no_dollars` — are they `[price, size]` pairs or something else (`[price, count, dollars]`, missing index, string sizes)?

If schema mismatch → fix `_realistic_fill_from_orderbook` in `src/execution.py` (around line 680). One-edit fix.

If schema is correct → real one-sided book problem. Add slippage haircut to paper mode (1¢ per side, 2¢ round trip), re-run go-live analysis with reduced numbers.

---

## Live exit policy (current state)

- **Take-profit:** 50% gain (`take_profit_pct: 0.50`)
- **Stop-loss:** disabled (`stop_loss_pct: 1.00` — can never trigger)
- **Time exit:** 5 min before tipoff, OR 240 min flat for non-time-anchored
- **Hard exit at 75 min:** REVERTED on 2026-05-31. Was promoted on backtest evidence, lived for 21 trades, performed worse than the policy it replaced (-$10.07/trade vs -$8.70 for time_exit). Marginal-only backtester now confirms revert was right. `hard_exit_minutes: 100000` (effectively disabled). Instrumentation still runs.
- **Thesis-decay exit:** SCAFFOLDED but OFF. Watcher code path exists, config flag is `thesis_decay_enabled: false`. Needs a `revalidate_edge` hook wired in `main.py` before flipping on. This is the highest-leverage unbuilt feature — fixing the -$1,170 `time_exit` leak would change the bot's economics.

---

## What's recently been built

Last ~10 days of work, all live:

1. **Shadow models framework** — three observe-only strategies (baseline favorite-bias, line-movement steam, Polymarket cross-exchange) logging picks. Insufficient resolved sample to evaluate yet. See `src/shadow_models.py`.
2. **Soccer Pinnacle 3-way path** — built and tested live (Pinnacle sport_id 29), then paused because European leagues are between seasons. Ready to flip on in August. See comment block in `src/models/soccer.py` for the August reinstatement list of real Kalshi tickers.
3. **TP fillability tracker** — the broken one. Currently writing NULLs.
4. **CLV sample status codes** — labels every sample as `valid` / `skipped_empty_book` / `skipped_extreme_mid` / `skipped_wide_spread` / `skipped_no_book`. Already producing useful data.
5. **Windowed CLV sampler** — replaces the brittle single-point T-5min sample with last-valid-mid in T-30→T-2 window. Tennis match-start fluidity no longer corrupts CLV data.
6. **Marginal-only exit backtester** — `aggregate_marginal_only()` in `src/exit_simulator.py`. Honest 75-min comparison: only counts trades that actually reached 75 min open. Latest: actual exits beat both 75-min cap (by $4.81/trade) and held-to-settlement (by $6.23/trade) on the 52-trade marginal cohort. Vindicated the hard-exit revert.
7. **Journal cleanup loop** — weekly prune of the write-only `signals` table + stale unresolved `shadow_signals`. Added after a journal-bloat crisis (1.5M rows, 492MB, dashboard query failures via OOM).
8. **Hour-window dedup on `log_shadow_signal`** — same (model, ticker, side) won't re-log within 1 hour. Stops the bloat at the source.

---

## Code map (the files that matter)

```
src/
  main.py                — entry point, asyncio.gather of all loops
  scanner.py             — Kalshi market discovery + filters
  decision.py            — model output → trade signal (edge/price gates)
  execution.py           — Executor class, watcher loop, _exit, fillability code
  journal.py             — SQLite schema + log_open/log_close/CLV updates
  config.py              — typed config; ExecutionConfig has all exit knobs
  models/
    sports.py            — dispatcher by series prefix
    tennis.py            — TennisModel (Pinnacle 2-way)
    cricket.py           — CricketModel (Pinnacle sport=8; IPL cut)
    soccer.py            — SoccerModel (PAUSED — see comment block)
    weather.py           — REMOVED (tried, didn't work)
    econ.py / crypto.py / politics.py  — stubs only
    sports_in_game.py    — ESPN winprob lag model
  shadow_models.py       — three observe-only strategies
  exit_simulator.py      — A/B + marginal-only backtester
  odds_provider.py       — Pinnacle integration (sport-id enumeration)
  kalshi_client.py       — REST + auth headers; get_orderbook lives here
  dashboard.py           — FastAPI; /stats endpoint surfaces aggregates
  welo.py                — Weighted Elo for tennis (observe-only)
config.yaml              — runtime tuning (filters, exits, risk caps)
render.yaml              — Render deploy config; DATA_DIR=/var/data
CHATGPT_REVIEW.md        — second-opinion analysis from session ~2026-06-04
PERPLEXITY_REVIEW.md     — older second-opinion doc
```

---

## What the user wants vs what's responsible

The user has pushed twice for "go live." Each time I declined and explained why. **The disciplined answer remains: ATP-only at 10% size ($200 allocated, $5-10/trade, $20 daily loss cap) AFTER the fillability bug is resolved and paper P&L is re-verified with realistic slippage.** Not WTA yet (CLV sample only 21). Not MLB (per-trade edge too thin to survive slippage). Not "everything" (that's the worst possible shape).

Go-live criteria I'm holding to (from ChatGPT's review, with my edits):
- 100+ trades on the candidate sport (ATP currently at 62)
- 40+ valid CLV samples (ATP at 17)
- 60%+ CLV positive rate (ATP at 76.5% ✓)
- Paper P&L positive after a 2.5¢ round-trip slippage haircut
- TP fillability bug resolved (the current blocker)

---

## How to communicate with this project

The bot's dashboard at `/stats` returns the JSON the user pastes after most updates. Key fields to look at:
- `total_pnl`, `n_trades`, `win_rate`
- `windowed_pnl` (3h/6h/12h/24h)
- `by_exit_reason` (the take_profit vs time_exit asymmetry is the bot's whole personality)
- `by_sport`, `tennis_summary`
- `by_sport_clv` (per-sport CLV averages)
- `marginal_75min` (the honest hard-exit verdict)

The user tends to type short prompts ("how we looking", "is it live yet", "what changed") and expects a direct numeric read with one actionable next step. Don't ramble. Don't restate already-made decisions. Be willing to say no when "go live" comes up too early.
