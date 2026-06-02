# Kalshi Edge Bot — Second-Opinion Review Brief

Paste this into ChatGPT. Self-contained — no prior session context needed.

---

## What this is

A Python trading bot that takes positions on Kalshi prediction markets (mostly sports moneylines). Running in **paper mode** on a $2,000 bankroll. Hosted on Render. The core edge thesis: Kalshi's prices sometimes disagree with Pinnacle (universally regarded as the sharpest sportsbook), and we systematically take the side Pinnacle favors after de-vigging.

Active sports right now: **Tennis (WTA + ATP), MLB, NBA, NHL.** Cut: WNBA (-$132 / 17% wr), IPL cricket (-$69 / 50% wr). Paused: Soccer (most European leagues between seasons).

## Current state — aggregates over ~30 days of paper trading

```
Total P&L:       -$32.24 over 298 trades
Win rate:        57.7%  (best trade +$70, worst -$100)
Avg per trade:   -$0.11
```

### By sport (lifetime)

```
KXWTAMATCH     n=74    +$219.39   wr 61%   ← profitable
MLB            n=121   +$29.45    wr 62%   ← profitable
KXATPMATCH     n=55    +$18.82    wr 60%   ← profitable
KXPGATOUR      n=1     +$0.31     wr 100%  (n=1, ignore)
NBA            n=7     -$4.76     wr 71%   (n=7, basically noise)
NHL            n=6     -$44.87    wr 33%   ← negative, small sample
KXCRICKETTEST  n=3     -$58.40    wr 33%   ← negative, tiny sample
KXIPLGAME      n=13    -$60.65    wr 54%   ← cut
KXWNBAGAME    n=18    -$131.53   wr 17%   ← cut
```

### By exit reason

```
take_profit       n=120   +$1,696   avg +$14.13   wr 100%   ← engine
time_exit         n=134   -$1,170   avg  -$8.73   wr  32%   ← drag
hard_exit_75m     n=20    -$207     avg -$10.33   wr  20%   ← see below
stop_loss         n=9     -$115     avg -$12.81   wr   0%   ← disabled
orphan_settlement n=15    -$236     avg -$15.73   wr  33%
```

The TP+time_exit asymmetry is the bot's whole personality: when we're right, we hit +20% TP fast; when we're wrong, we ride to a tip-aligned time exit (5 min before kickoff) and lose.

### CLV (closing-line value, in bp)

```
MLB:        n=44   avg  +0.8bp   %positive 59.1%
ATP:        n=13   avg  +9.4bp   %positive 84.6%   ← sharp signal
WTA:        n=13   avg  -2.4bp   %positive 46.2%   ← see "WTA mystery" below
IPL:        n=5    avg -19.5bp   %positive 20.0%   (cut)
NBA:        n=2    avg -30.5bp   %positive  0.0%   (n=2, ignore)
```

## Recent decisions (already made — don't revisit unless wrong)

1. **WNBA / IPL cut** because of negative P&L on small-but-meaningful samples.
2. **Cricket (non-IPL) kept enabled** but no fresh markets currently.
3. **Stop-loss disabled** because settlement backtest showed it cost +$122 over 9 SL exits.
4. **Soccer model paused** (Pinnacle 3-way path built and ready, just no major-league markets active until August).
5. **Shadow models framework** built — three observe-only alternative strategies (baseline favorite-bias, line-movement steam, Polymarket cross-exchange) logging picks but not trading. Insufficient resolved sample to evaluate yet.
6. **Hour-window dedup + weekly cleanup loop** added after a journal bloat incident (1.5M signals + 887k shadow_signals rows on a 1GB Render disk caused dashboard query failures via OOM).

## Two unresolved items the bot is in the middle of

### 1. Hard exit (75-min cap) — just reverted

**Background.** Backtest simulator said an "exit at 75 min if still open" policy would have made +$502 / 65% wr over 164 instrumented trades vs the actual -$110 / 57% on the same set. Promoted the policy to live on 2026-05-24.

**Live result over 20 hard exits:**
```
hard_exit_75m   n=20   -$207   avg -$10.33   wr 20%
```

Compared to the `time_exit` policy it replaced (avg -$8.73), the live hard exit is **$1.60/trade worse** — opposite of what the simulator predicted.

**Hypothesis on the gap:** the simulator's +$3.16/trade average mostly came from take-profit winners that closed *before* 75 minutes (the policy doesn't change those trades). The *marginal* effect — only on trades that would have held past 75 min — may have been close to zero in the sim. The live cohort isolates only those marginal trades and shows them losing.

**Action taken:** set `hard_exit_minutes` to 100000 (effectively disabled) on 2026-05-31. Watcher code path and `mid_at_75min` instrumentation still run, so we keep collecting data for re-evaluation.

### 2. WTA CLV mystery — fixed but data still polluted

**The paradox.** WTA wins (61%, +$219 over 74 trades) but raw CLV says -2.4bp / 46% positive (n=13). ATP with the same 60% win rate shows +9.4bp / 84.6% positive CLV. Two sports with identical win rates and opposite CLV signals shouldn't happen if the edge is real.

**Investigation.** Dumped raw WTA CLV samples from the journal. Found that **8 of the 12 samples were exactly `clv_price = 0.5`** and one was `0.01`. The 0.5 reads are an empty-orderbook fallback in `_mid_from_orderbook` — fine for the watcher loop's mark-to-market default, but corrupting the CLV measurement when sampled at the 5-min-pre-tipoff mark on thinly-traded early-round WTA matches. ATP doesn't have this problem because ATP markets have real books pre-tipoff.

**Fix shipped:** `_sample_clv` now detects empty book BEFORE calling `_mid_from_orderbook` and skips the write. Also guards against extreme mids (<0.05 or >0.95) which are usually post-settlement reads.

**Backfill pending:** the existing polluted rows need to be nulled out via SQL. Until that runs, the dashboard is still computing WTA CLV against the bad data.

## Specific questions where a second opinion would help

1. **Hard exit decision.** Was reverting to "effectively disabled" the right call given 20 trades / -$207, or would you have tried an intermediate (e.g. 90 min, 120 min) first? Is there a smarter way to distinguish "trade should be cut early because it's lost its inflection point" from "trade should ride to time_exit for CLV alignment"?

2. **WTA CLV diagnosis.** Given the raw sample below, does the empty-book-fallback hypothesis explain everything, or could there be a separate WTA-specific issue (e.g. name matching, tipoff time accuracy)?

   ```
   ticker                     side  fill  clv    pnl     result
   KXWTAMATCH-...-SVIBEN-BEN  yes   0.47  0.505  -2.21   L
   KXWTAMATCH-...-KOSSWI-KOS  no    0.56  0.5    -3.10   L
   KXWTAMATCH-...-SHNOLI-SHN  no    0.38  0.01   -22.05  L
   KXWTAMATCH-...-JOVOSA-JOV  no    0.49  0.645  +5.95   W
   KXWTAMATCH-...-SAKCHW-SAK  yes   0.49  0.5    +10.87  W
   KXWTAMATCH-...-WANSTA-WAN  no    0.54  0.5    -10.92  L
   KXWTAMATCH-...-OLIBIR-OLI  yes   0.59  0.5    +12.19  W
   KXWTAMATCH-...-SNISTE-STE  no    0.44  0.185  -17.94  L
   KXWTAMATCH-...-KORWAN-WAN  no    0.53  0.5    +9.77   W
   KXWTAMATCH-...-PAOYAS-YAS  no    0.48  0.5    +9.46   W
   KXWTAMATCH-...-SAMTEI-TEI  no    0.59  0.5    -10.53  L
   KXWTAMATCH-...-MBOCRI-MBO  no    0.47  0.605  +5.99   W
   ```

3. **Go-live criteria for ATP.** ATP at 55 trades / 60% wr / +9.4bp CLV / 84.6% positive looks structurally healthy. What sample size, CLV consistency, and slippage assumptions would you require before going live with real money at ~10% of paper size? What's the right kill switch?

4. **The time_exit problem.** 134 trades, -$1,170, 32% win rate, avg -$8.73. This is the bot's single biggest leak. We hold to "tip-5min" for CLV alignment, but the closing line clearly isn't moving our way most of the time. Three plausible interpretations:

   - (a) Our entry prices are bad — we're buying favorites at prices Pinnacle later disagrees with.
   - (b) The 5-minutes-before-tip exit is too early — late lineup news / weather / scratches haven't fully hit Kalshi yet.
   - (c) We're picking up too many small edges that don't survive vig.

   Which would you prioritize investigating, and how would you tell them apart from the data we have?

5. **Anything else.** If you were operating this bot, what would you look at next that we haven't mentioned? Where's the dumbest thing we're doing that's not obvious from the aggregates?

---

## How to give the most useful answer

Don't restate decisions already made (WNBA cut, IPL cut, soccer paused, stop-loss disabled). Skip generic prediction-market lectures — assume familiarity with Kalshi, de-vigging, CLV, and Kelly sizing. Give concrete recommendations with numeric thresholds where relevant. If you disagree with my diagnoses, say so directly with the reasoning.
