"""Execution engine.

In `paper` mode, fills are simulated instantly at the limit price; exits use
mid-price snapshots to compute realized P&L.

In `live` mode, the engine submits real limit orders to Kalshi. Scale-in is
respected (the entry size is split across `scale_in_chunks` orders posted at
the touch and one tick deeper).

Every position spawns a watcher that polls the orderbook and triggers exit on:
  - mid drift to TP
  - mid drift to SL
  - elapsed time >= time_exit_minutes
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import structlog

from .config import env_config, file_config
from .decision import TradeSignal
from .fee_model import fee_per_contract_dollars
from .journal import Journal
from .kalshi_client import KalshiClient, Market
from .market_fields import event_start_utc
from .risk import RiskEngine

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Whale boost math
# ---------------------------------------------------------------------------
# Continuous magnitude-based multiplier. Same per-class ceilings as the prior
# binary tiers so worst-case sizing doesn't grow:
#   price_jump    1.5x (5¢)   -> 5.0x (10¢+)
#   volume_burst  1.2x (5k)   -> 2.5x (50k+)
#   resting       1.1x (2k)   -> 1.5x (20k+)
# The reason string contains the raw magnitude (e.g. "price_jump_+7c",
# "volume_burst_12345", "large_yes_bid_3500c"). We extract the number,
# normalize to a 0..1 ratio against the class's saturation point, and
# interpolate between the class min and max multipliers.
#
# Returns (multiplier, class_label, magnitude_str) where magnitude_str is the
# canonical bucket the dashboard's `By Whale Magnitude` panel groups by.

import re as _re

_WHALE_PRICE_RE = _re.compile(r"price_jump_([+\-]?\d+)c")
_WHALE_VOL_RE   = _re.compile(r"volume_burst_(\d+)")
_WHALE_REST_RE  = _re.compile(r"large_(?:yes|no)_(?:bid|ask)_(\d+)")


def _lerp(t: float, lo: float, hi: float) -> float:
    t = max(0.0, min(1.0, t))
    return lo + (hi - lo) * t


def _whale_multiplier(wreason: str) -> tuple[float, str, str]:
    """Map a whale reason string to (multiplier, class_label, magnitude_str).
    Falls back to (2.0x, "other", "unknown") for anything we don't parse.
    """
    if wreason.startswith("price_jump"):
        m = _WHALE_PRICE_RE.search(wreason)
        cents = abs(int(m.group(1))) if m else 5
        # 5c floor → 1.5x, 10c saturation → 5.0x
        t = (cents - 5) / 5.0
        mult = _lerp(t, 1.5, 5.0)
        # Bucket label for dashboard analysis
        if cents < 7:        bucket = "price_5-7c"
        elif cents < 10:     bucket = "price_7-10c"
        else:                bucket = "price_10c+"
        return mult, "aggressive", bucket
    if wreason.startswith("volume_burst"):
        m = _WHALE_VOL_RE.search(wreason)
        contracts = int(m.group(1)) if m else 5000
        # 2026-05-20: burst class disabled. 18 trades, 44% wr, -$104.
        # Direction-ambiguous volume bursts aren't predictive. We keep
        # the class label + magnitude bucket for logging continuity so
        # the dashboard's `By Whale Class` panel still tracks the
        # cohort, but the size multiplier is 1.0 (no boost).
        if contracts < 10_000:    bucket = "burst_5-10k"
        elif contracts < 25_000:  bucket = "burst_10-25k"
        elif contracts < 50_000:  bucket = "burst_25-50k"
        else:                     bucket = "burst_50k+"
        return 1.0, "burst", bucket
    if wreason.startswith("large_"):
        m = _WHALE_REST_RE.search(wreason)
        contracts = int(m.group(1)) if m else 2000
        if contracts < 5_000:
            # 2026-05-21: rest_2-5k disabled. 26 trades, 42% wr, -$49.
            # Small resting orders are spoof-prone noise; only 5k+ pays
            # (rest_5-10k 73% wr, rest_10k+ 65% wr). Label kept for the
            # dashboard's By Whale Magnitude panel; multiplier is 1.0.
            return 1.0, "resting", "rest_2-5k"
        # 5k → ~1.2x, 20k saturation → 1.5x
        import math as _math
        lo, hi = _math.log(5000), _math.log(20000)
        x = max(lo, min(hi, _math.log(max(contracts, 5000))))
        t = (x - lo) / (hi - lo)
        mult = _lerp(t, 1.2, 1.5)
        bucket = "rest_5-10k" if contracts < 10_000 else "rest_10k+"
        return mult, "resting", bucket
    return 2.0, "other", "unknown"


@dataclass
class OpenPosition:
    signal: TradeSignal
    fill_price: float
    contracts: int
    opened_at: datetime
    children: list[asyncio.Task] = field(default_factory=list)


class Executor:
    def __init__(
        self, client: KalshiClient, risk: RiskEngine, journal: Journal,
        *, whale_tracker=None,
        revalidate_edge=None,
    ) -> None:
        self.client = client
        self.risk = risk
        self.journal = journal
        self.cfg = file_config().execution
        self.mode = env_config().mode
        self.open: dict[str, OpenPosition] = {}
        # Optional whale tracker — when an aligned whale signal exists for
        # this ticker, size gets boosted to whale_max_position_size_usd.
        self.whale_tracker = whale_tracker
        # Optional thesis-decay revalidator. Async callable:
        #   await revalidate_edge(ticker, side, current_mid) -> float | None
        # Returns the current net edge (model_p - exit_price - spread
        # cushion) if it can recompute, or None if the model can't
        # produce a probability right now (e.g. Pinnacle returned no
        # match). When it returns a value below cfg.thesis_decay_min_
        # negative_edge, the watcher exits with reason 'thesis_decay'.
        # Wiring lives in main.py — None by default = decay disabled.
        self.revalidate_edge = revalidate_edge
        # Event-level dedup: BOS-yes and TB-no on the same Kalshi event
        # are the same bet (both win if Boston wins). Locking by
        # event_ticker prevents double-exposure across mirror tickers.
        self.open_events: set[str] = set()
        # Post-close cooldown: after an exit, lock the event for 60 min
        # so we don't immediately re-enter when the model still sees an
        # edge. Without this we churn through fees firing the same trade
        # every loop after each take_profit / stop_loss.
        self.cooldown_until: dict[str, datetime] = {}

    _COOLDOWN_MINUTES = 90  # was 60 — Khachanov/Prizmic re-entered at 59m

    @staticmethod
    def _event_ticker(market: Market) -> str:
        """Return the canonical event key shared by every side of a Kalshi
        market.

        We DERIVE this from the ticker string rather than trust
        market.raw.event_ticker because Kalshi tennis markets ship a
        DIFFERENT event_ticker for each player side — which broke our
        dedup on 2026-05-12 (Khachanov-YES TP'd at 05:52, Prizmic-NO
        re-entered at 06:51 = same Khachanov-to-win bet at a worse
        price, lost $33).

        Convention: Kalshi sports tickers look like
        `SERIES-MATCHID-SIDE`, e.g. `KXATPMATCH-26MAY11KHAPRI-KHA` and
        `KXATPMATCH-26MAY11KHAPRI-PRI`. Stripping the final `-SIDE`
        chunk gives `KXATPMATCH-26MAY11KHAPRI`, identical for every
        side of the same event (works for 2-way moneylines AND 3-way
        soccer where there's an extra `-TIE` ticker).

        Falls back to raw.event_ticker / ticker if the split fails.
        """
        t = (market.ticker or "").strip()
        if t and t.count("-") >= 2:
            return t.rsplit("-", 1)[0]
        # Defensive fallback: trust Kalshi's field if our split heuristic
        # can't find at least two hyphens (shouldn't happen on real
        # sports markets but might on non-sport ones).
        return (market.raw.get("event_ticker") or t).strip()

    async def submit(self, signal: TradeSignal, market: Market) -> None:
        # 1) One position per ticker (existing dedup)
        if signal.ticker in self.open:
            log.debug("exec.skip.already_open", ticker=signal.ticker)
            return

        # 2) One position per EVENT — blocks the mirror-side ticker
        ev = self._event_ticker(market)
        if ev and ev in self.open_events:
            log.info("exec.skip.event_already_open",
                     ticker=signal.ticker, event_ticker=ev)
            return

        # 3) Cooldown after a recent close on this event
        cd_until = self.cooldown_until.get(ev) if ev else None
        if cd_until and datetime.now(timezone.utc) < cd_until:
            mins_left = (cd_until - datetime.now(timezone.utc)).total_seconds() / 60
            log.info("exec.skip.event_cooldown",
                     ticker=signal.ticker, event_ticker=ev,
                     mins_left=round(mins_left, 1))
            return

        if not self.risk.approve(signal):
            return

        # Whale boost — magnitude-scaled within each class.
        # 2026-05-13: replaced binary tier (5x / 2.5x / 1.5x) with a
        # continuous multiplier driven by the raw whale magnitude. Same
        # per-class ceilings as before so worst case doesn't grow, but
        # smaller whales get smaller boosts. The reason string now
        # captures the raw magnitude (cents / contracts) so dashboard
        # bucketing can read calibration per magnitude band.
        risk_cfg = file_config().risk
        whale_signal = None
        if self.whale_tracker is not None:
            whale_signal = self.whale_tracker.has_aligned_signal(
                signal.ticker, signal.side
            )
        if whale_signal and risk_cfg.whale_max_position_size_usd > signal.size_usd:
            wreason = whale_signal.reason
            multiplier, whale_class, magnitude = _whale_multiplier(wreason)

            old_size = signal.size_usd
            signal.size_usd = min(
                risk_cfg.whale_max_position_size_usd,
                signal.size_usd * multiplier,
            )
            # magnitude= field is what `By Whale Magnitude` panel reads.
            signal.reason = (
                f"{signal.reason} | WHALE_ALIGNED class={whale_class} "
                f"magnitude={magnitude} "
                f"dir={whale_signal.direction} conf={whale_signal.confidence:.2f} "
                f"({whale_signal.reason}) size_boost "
                f"${old_size:.0f}->${signal.size_usd:.0f} ({multiplier:.2f}x)"
            )
            log.info("exec.whale_boost",
                     ticker=signal.ticker, side=signal.side,
                     whale_class=whale_class, multiplier=round(multiplier, 2),
                     magnitude=magnitude,
                     old_size=round(old_size, 2),
                     new_size=round(signal.size_usd, 2),
                     whale=whale_signal.reason)

        contracts = max(1, int(signal.size_usd / max(signal.price_cents / 100, 0.01)))
        if self.mode == "paper":
            fill_price = signal.price_cents / 100
            self._record_open(signal, market, fill_price, contracts)
        else:
            await self._submit_live(signal, market, contracts)

    async def _submit_live(self, signal: TradeSignal, market: Market, contracts: int) -> None:
        """Place real buy orders, then journal ONLY what actually filled.

        The old stub recorded an optimistic fill at the limit price
        regardless of execution. Now we poll fills until
        live_entry_fill_timeout_s, cancel any unfilled remainder, and
        record the position at the actual average fill price for the
        actual filled count. Zero fills -> no position, no journal row.
        """
        chunks = max(1, self.cfg.scale_in_chunks)
        per_chunk = max(1, contracts // chunks)
        order_ids: list[str] = []
        for i in range(chunks):
            client_order_id = f"{signal.ticker}-{uuid.uuid4().hex[:8]}"
            try:
                resp = await self.client.place_order(
                    ticker=signal.ticker,
                    side=signal.side,
                    action="buy",
                    count=per_chunk,
                    price_cents=signal.price_cents - i,  # post one tick deeper each chunk
                    client_order_id=client_order_id,
                )
                log.info("exec.live.placed", order=resp)
                oid = (resp.get("order") or {}).get("order_id")
                if oid:
                    order_ids.append(oid)
            except Exception as e:
                log.exception("exec.live.error", ticker=signal.ticker, err=str(e))

        if not order_ids:
            log.warning("exec.live.no_orders_placed", ticker=signal.ticker)
            return

        filled, avg_price_dollars = await self._reconcile_entry(
            signal.ticker, signal.side, order_ids,
        )
        if filled <= 0 or avg_price_dollars is None:
            log.info("exec.live.entry_unfilled", ticker=signal.ticker,
                     orders=len(order_ids))
            return
        log.info("exec.live.entry_filled", ticker=signal.ticker,
                 filled=filled, requested=contracts,
                 avg_price=round(avg_price_dollars, 4))
        self._record_open(signal, market, avg_price_dollars, filled)

    async def _reconcile_entry(
        self, ticker: str, side: str, order_ids: list[str],
    ) -> tuple[int, float | None]:
        """Poll order statuses until all are terminal or timeout; cancel
        leftovers; return (filled_contracts, avg_fill_price_dollars)."""
        deadline = (datetime.now(timezone.utc)
                    + timedelta(seconds=self.cfg.live_entry_fill_timeout_s))
        while datetime.now(timezone.utc) < deadline:
            await asyncio.sleep(self.cfg.live_fill_poll_s)
            try:
                statuses = []
                for oid in order_ids:
                    o = (await self.client.get_order(oid)).get("order") or {}
                    statuses.append(o.get("status", ""))
                if all(s in ("executed", "canceled") for s in statuses):
                    break
            except Exception as e:  # noqa: BLE001
                log.warning("exec.live.order_poll_error",
                            ticker=ticker, err=str(e)[:200])
        # Cancel whatever is still resting. Safe to call on terminal
        # orders too — Kalshi errors, we swallow and move on.
        for oid in order_ids:
            try:
                await self.client.cancel_order(oid)
                log.info("exec.live.entry_canceled_remainder",
                         ticker=ticker, order_id=oid)
            except Exception:  # noqa: BLE001
                pass
        return await self._fills_for_orders(side, order_ids)

    async def _fills_for_orders(
        self, side: str, order_ids: list[str],
    ) -> tuple[int, float | None]:
        """Sum executed contracts + average price across orders' fills."""
        total = 0
        value = 0.0
        for oid in order_ids:
            try:
                fills = (await self.client.get_fills(order_id=oid)
                         ).get("fills") or []
            except Exception as e:  # noqa: BLE001
                log.warning("exec.live.fills_error", order_id=oid,
                            err=str(e)[:200])
                continue
            for f in fills:
                price = self._fill_price_dollars(f, side)
                try:
                    count = int(round(float(f.get("count", 0))))
                except (ValueError, TypeError):
                    continue
                if price is None or count <= 0:
                    continue
                total += count
                value += count * price
        if total <= 0:
            return 0, None
        return total, value / total

    @staticmethod
    def _fill_price_dollars(fill: dict, side: str) -> float | None:
        """Extract our side's fill price in dollars. The v2 API reports
        cents in `yes_price` / `no_price`; some payloads carry
        `*_price_dollars` / `*_price_fp` (already dollars) instead."""
        v = fill.get(f"{side}_price")
        if v is not None:
            try:
                return float(v) / 100.0
            except (ValueError, TypeError):
                pass
        for key in (f"{side}_price_dollars", f"{side}_price_fp"):
            v = fill.get(key)
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return None

    def _record_open(self, signal: TradeSignal, market: Market, fill_price: float, contracts: int) -> None:
        pos = OpenPosition(
            signal=signal,
            fill_price=fill_price,
            contracts=contracts,
            opened_at=datetime.now(timezone.utc),
        )
        self.open[signal.ticker] = pos
        ev = self._event_ticker(market)
        if ev:
            self.open_events.add(ev)
            # Stash event_ticker on the position so _exit can release the lock
            pos.event_ticker = ev  # type: ignore[attr-defined]
        # Stash the game's tip time so the watcher can use it for the
        # CLV-aligned exit (5 min before tipoff). Without this we'd
        # exit on a flat 4h timer that fires hours before the close.
        tip_utc = event_start_utc(market.raw)
        if tip_utc:
            pos.tip_utc = tip_utc  # type: ignore[attr-defined]
        self.risk.record_open(signal)
        # Pass pos.opened_at so the journal row's opened_ts matches what
        # _sample_clv / _watch use in their UPDATE WHERE clauses. Without
        # this they target a non-existent (ticker, opened_ts) pair and
        # silently update zero rows.
        self.journal.log_open(
            signal=signal, market=market, fill_price=fill_price,
            contracts=contracts, opened_ts=pos.opened_at.isoformat(),
        )
        # Slack ping. Fire-and-forget — failure never blocks the trade flow.
        try:
            from .slack_notifier import notify_open
            notify_open(
                ticker=signal.ticker, side=signal.side,
                size_usd=signal.size_usd, fill_price=fill_price,
                edge=signal.edge, reason=signal.reason,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("slack.notify_open_failed", err=str(e)[:200])
        pos.children.append(asyncio.create_task(self._watch(signal.ticker)))
        # Schedule a CLV sampler — captures Kalshi's mid price ~5 min
        # before tipoff so we can measure closing-line value even when
        # the position has already exited via TP/SL.
        if tip_utc:
            opened_ts = pos.opened_at.isoformat()
            pos.children.append(asyncio.create_task(
                self._sample_clv(signal.ticker, signal.side, tip_utc, opened_ts)
            ))

    async def _sample_clv(
        self, ticker: str, side: str, tip_utc: datetime, opened_ts: str,
    ) -> None:
        """Windowed CLV sampler: capture the LAST VALID Kalshi mid
        between T-30min and T-2min before tipoff.

        Why windowed (and not the old single-point T-5min sample):
        tennis match start times are notoriously fluid. Matches can
        start early when prior ones end fast, get delayed by rain, get
        suspended mid-match, or end in walkovers — any of which means
        a single T-5min poll hits an empty book, a settled market, or
        the wrong moment. We poll every 2-3 min through the closing
        window and keep whichever sample was the LATEST one that
        passed validity (non-empty book, mid in [0.05, 0.95], spread
        <= 12¢). The status code we write to the trade row tells the
        dashboard WHY we accepted or skipped — silent skipping was
        what hid the 0.5 fallback bug for weeks (see audit 2026-05-31).
        """
        # Sleep until the window opens (T-30 min before tip).
        window_open = tip_utc - timedelta(minutes=30)
        window_close = tip_utc - timedelta(minutes=2)
        wait_seconds = (
            window_open - datetime.now(timezone.utc)
        ).total_seconds()
        # Cap at 36 hours so a far-future tip doesn't pin the task forever
        wait_seconds = max(0.0, min(wait_seconds, 36 * 3600))
        try:
            await asyncio.sleep(wait_seconds)
        except asyncio.CancelledError:
            return
        last_valid_mid: float | None = None
        last_status = "skipped_no_book"
        poll_interval = 180  # 3 minutes — gives us ~10 samples per window
        try:
            while datetime.now(timezone.utc) < window_close:
                status, mid = await self._clv_sample_once(ticker, side)
                if status == "valid" and mid is not None:
                    last_valid_mid = mid
                    last_status = "valid"
                elif last_status != "valid":
                    # Track the most recent failure reason so the
                    # dashboard can show why a window produced nothing.
                    last_status = status
                try:
                    await asyncio.sleep(poll_interval)
                except asyncio.CancelledError:
                    break
            # Stamp the outcome — either the last valid mid or the
            # last failure reason. update_clv_status writes the status
            # code regardless; update_clv_price only fires on success.
            if last_valid_mid is not None:
                self.journal.update_clv_price(
                    ticker=ticker, opened_ts=opened_ts,
                    clv_price=last_valid_mid,
                )
            self.journal.update_clv_status(
                ticker=ticker, opened_ts=opened_ts, status=last_status,
            )
            log.info("clv.window_done",
                     ticker=ticker, side=side,
                     status=last_status,
                     clv_price=round(last_valid_mid, 4)
                              if last_valid_mid is not None else None)
        except Exception:
            log.exception("clv.sample_failed", ticker=ticker)

    async def _clv_sample_once(
        self, ticker: str, side: str,
    ) -> tuple[str, float | None]:
        """One CLV poll. Returns (status, mid_or_None).

        Status codes:
          'valid'                 — book non-empty, spread tight, mid in band
          'skipped_empty_book'    — yes and no books both empty
          'skipped_extreme_mid'   — outside [0.05, 0.95] (settled / broken)
          'skipped_wide_spread'   — spread > 12¢ (untradable book)
          'skipped_no_book'       — fetch returned nothing
        """
        try:
            ob = await self.client.get_orderbook(ticker)
        except Exception:
            log.debug("clv.poll.fetch_error", ticker=ticker)
            return "skipped_no_book", None
        if not ob:
            return "skipped_no_book", None
        book = ob.get("orderbook_fp") or ob.get("orderbook") or {}
        yes_book = book.get("yes_dollars") or book.get("yes") or []
        no_book = book.get("no_dollars") or book.get("no") or []
        if not yes_book and not no_book:
            return "skipped_empty_book", None
        # Compute yes_bid and yes_ask explicitly so we can also gate on
        # spread. (mid_from_orderbook already does this internally but
        # we need the bid/ask separately for the spread check.)
        try:
            # Best bid = MAX price, not "last element". Docs describe the
            # fp arrays as best-to-worst; old code assumed ascending.
            # max() is correct under either ordering.
            yes_bid = max((float(e[0]) for e in yes_book), default=0.0)
            no_bid = max((float(e[0]) for e in no_book), default=0.0)
            yes_ask = (1.0 - no_bid) if no_bid > 0 else 0.0
        except (ValueError, IndexError, TypeError):
            return "skipped_no_book", None
        if yes_bid > 0 and yes_ask > 0:
            spread_cents = abs(yes_ask - yes_bid) * 100.0
            if spread_cents > 12.0:
                return "skipped_wide_spread", None
        mid = self._mid_from_orderbook(ob, side)
        if not (0.05 < mid < 0.95):
            return "skipped_extreme_mid", None
        return "valid", mid

    async def _watch(self, ticker: str) -> None:
        pos = self.open.get(ticker)
        if not pos:
            return
        # CLV-aligned deadline: prefer "tip_utc - 5min" so we hold through
        # the closing-line move (sharps + late lineup news + weather hit
        # most prices in the final hour). Fall back to the flat
        # time_exit_minutes config if we don't have a tip time. Cap at
        # opened + time_exit_minutes only as a sanity bound when tip is
        # absurdly far out (e.g. multi-day outright markets we shouldn't
        # be holding anyway).
        tip_utc = getattr(pos, "tip_utc", None)
        flat_deadline = pos.opened_at + timedelta(minutes=self.cfg.time_exit_minutes)
        if tip_utc:
            tip_deadline = tip_utc - timedelta(minutes=5)
            # Use the LATER of (tip-5min, flat) — for short-fuse trades
            # we want at least a few minutes; for long-fuse trades we
            # want to ride to close.
            deadline = max(tip_deadline, pos.opened_at + timedelta(minutes=15))
        else:
            deadline = flat_deadline
        # Hard 75-min exit cap (promoted from the A/B exit simulator,
        # 2026-05-24). Whatever the tip/flat deadline above works out to,
        # never hold a position past `hard_exit_minutes` — the backtest
        # showed trades open this long have usually missed their
        # inflection point and bleed into a losing time_exit. When this
        # cap is the binding deadline we journal the exit under a distinct
        # reason so the dashboard tracks the cohort separately.
        hard_deadline = pos.opened_at + timedelta(
            minutes=self.cfg.hard_exit_minutes
        )
        hard_capped = hard_deadline < deadline
        if hard_capped:
            deadline = hard_deadline
        opened_ts = pos.opened_at.isoformat()
        while ticker in self.open:
            await asyncio.sleep(15)
            try:
                ob = await self.client.get_orderbook(ticker)
                mid = self._mid_from_orderbook(ob, pos.signal.side)
                # Persist the latest mid so the dashboard can show
                # mark-to-market P&L without making its own API calls.
                if 0 < mid < 1:
                    self.journal.update_current_mid(
                        ticker=ticker, opened_ts=opened_ts, mid=mid,
                    )
                    # Instrument-only: snapshot the mid once when the
                    # position crosses ~75 min old, so the A/B exit
                    # simulator can backtest a "hard exit at 75 min"
                    # policy. The journal write is itself write-once
                    # (mid_at_75min IS NULL guard); the flag here just
                    # avoids a redundant UPDATE every tick afterward.
                    if not getattr(pos, "mid75_recorded", False):
                        age_min = (datetime.now(timezone.utc)
                                   - pos.opened_at).total_seconds() / 60
                        if age_min >= 75:
                            self.journal.update_mid_at_75min(
                                ticker=ticker, opened_ts=opened_ts, mid=mid,
                            )
                            pos.mid75_recorded = True  # type: ignore[attr-defined]
                # `mid` is already side-adjusted (NO mid for NO bets). So
                # profit is mid - fill regardless of side. The previous
                # branching here flipped the sign on every NO position,
                # making take_profit trigger on losses and vice versa.
                pnl_pct = (mid - pos.fill_price) / pos.fill_price
                if pnl_pct >= self.cfg.take_profit_pct:
                    # Live mode: _exit returns False if the sell didn't
                    # fill — keep watching and retry next tick.
                    if await self._exit(ticker, mid, reason="take_profit",
                                        orderbook=ob):
                        return
                    continue
                if pnl_pct <= -self.cfg.stop_loss_pct:
                    if await self._exit(ticker, mid, reason="stop_loss",
                                        orderbook=ob):
                        return
                    continue
                # Thesis-decay check (gated by config + age + cadence).
                # Calls back into the model dispatcher to re-ask: given
                # the CURRENT Kalshi exit price and the CURRENT Pinnacle
                # fair, is the edge still there? If the answer is "no,
                # by more than `thesis_decay_min_negative_edge`", exit.
                # This replaces blunt clock-based exits with a state-
                # based one — fixes the "time_exit eats -$8.73/trade"
                # leak ChatGPT review surfaced 2026-05-31.
                if (self.cfg.thesis_decay_enabled
                        and self.revalidate_edge is not None):
                    age_min = (
                        datetime.now(timezone.utc) - pos.opened_at
                    ).total_seconds() / 60
                    last_check = getattr(pos, "last_decay_check", None)
                    cadence_ok = (
                        last_check is None
                        or (datetime.now(timezone.utc) - last_check)
                            .total_seconds() / 60
                            >= self.cfg.thesis_decay_revalidate_minutes
                    )
                    if (age_min >= self.cfg.thesis_decay_min_age_minutes
                            and cadence_ok):
                        pos.last_decay_check = (  # type: ignore[attr-defined]
                            datetime.now(timezone.utc)
                        )
                        try:
                            current_edge = await self.revalidate_edge(
                                ticker, pos.signal.side, mid,
                            )
                        except Exception:
                            log.exception("exec.decay.revalidate_error",
                                          ticker=ticker)
                            current_edge = None
                        if (current_edge is not None
                                and current_edge
                                    < self.cfg.thesis_decay_min_negative_edge):
                            log.info("exec.exit.thesis_decay",
                                     ticker=ticker, side=pos.signal.side,
                                     age_min=round(age_min, 1),
                                     current_edge=round(current_edge, 4))
                            if await self._exit(
                                ticker, mid, reason="thesis_decay",
                                orderbook=ob, exit_edge=current_edge,
                            ):
                                return
                            continue
                if datetime.now(timezone.utc) >= deadline:
                    exit_reason = (
                        f"hard_exit_{self.cfg.hard_exit_minutes}m"
                        if hard_capped else "time_exit"
                    )
                    if await self._exit(ticker, mid, reason=exit_reason,
                                        orderbook=ob):
                        return
                    continue
            except Exception as e:
                log.exception("exec.watch.error", ticker=ticker, err=str(e))

    @staticmethod
    def _realistic_fill_from_orderbook(
        ob: dict, side: str, contracts: int,
    ) -> tuple[float | None, int]:
        """Compute the realistic avg fill price for a SELL of `contracts`
        contracts on `side` by sweeping the relevant bid stack.

        Selling YES = lifting the YES bid stack (we want to hit buyers).
        Selling NO  = lifting the NO  bid stack (same idea, other side).
        Each Kalshi book entry is `[price_dollars, size_contracts]` and
        the array is sorted ASCENDING by price — so the LAST entries
        are the highest bids, which is what we hit first.

        Returns (avg_fill_price, total_book_size). avg_fill_price is None
        if the book on our side is empty (no realistic exit liquidity at
        all — paper P&L is fictional in that case).

        The point of this is the dashboard's "Paper TP vs Realistic TP"
        panel: paper mode credits the full exit at `mid`, but a real-
        money sweep would average down through the bid stack and give
        us strictly worse fills. The gap is our paper-overstatement.
        """
        try:
            book = ob.get("orderbook_fp") or ob.get("orderbook") or {}
            if side == "yes":
                stack = book.get("yes_dollars") or book.get("yes") or []
            else:
                stack = book.get("no_dollars") or book.get("no") or []
            if not stack:
                return None, 0
            # Kalshi's orderbook_fp sizes are fixed-point decimal STRINGS
            # ("100.00") per the official docs — int("100.00") raises
            # ValueError. The old `sum(int(e[1]) ...)` sat outside the
            # per-entry try, so one bad entry nuked the whole computation
            # to (None, 0). That was the bug writing NULL/0 on every exit.
            # Parse via float() and tolerate malformed entries.
            parsed: list[tuple[float, int]] = []
            for e in stack:
                try:
                    parsed.append((float(e[0]), int(round(float(e[1])))))
                except (ValueError, IndexError, TypeError, KeyError):
                    continue
            if not parsed:
                return None, 0
            total_size = sum(sz for _, sz in parsed)
            needed = max(int(contracts), 1)
            filled = 0
            total_value = 0.0
            # Walk best -> worst explicitly (sort desc by price) instead
            # of trusting Kalshi's array ordering, which the docs describe
            # as "best to worst" while our old code assumed ascending.
            for price, size in sorted(parsed, key=lambda p: p[0],
                                      reverse=True):
                take = min(size, needed)
                total_value += take * price
                filled += take
                needed -= take
                if needed <= 0:
                    break
            if filled == 0:
                return None, total_size
            return total_value / filled, total_size
        except Exception:
            log.exception("exec.realistic_fill_error")
            return None, 0

    @staticmethod
    def _mid_from_orderbook(ob: dict, side: str) -> float:
        """Compute the YES-equivalent mid price from Kalshi's orderbook payload.

        Kalshi returns the book under `orderbook_fp` with `yes_dollars` and
        `no_dollars` arrays. Each entry is `[price_str, size_str]` and the
        prices are ALREADY IN DOLLARS (0.01-0.99). The arrays are sorted
        ascending by price, so the LAST entry of each side is the best bid
        on that side. To get the YES ask, look at the best NO bid and
        invert (buying NO at p == selling YES at 1-p).
        """
        try:
            book = ob.get("orderbook_fp") or ob.get("orderbook") or {}
            yes_book = book.get("yes_dollars") or book.get("yes") or []
            no_book = book.get("no_dollars") or book.get("no") or []

            # Best bid = MAX price, not "last element". Docs describe the
            # fp arrays as best-to-worst; old code assumed ascending.
            # max() is correct under either ordering.
            yes_bid = max((float(e[0]) for e in yes_book), default=0.0)
            no_bid = max((float(e[0]) for e in no_book), default=0.0)
            # Best YES ask is implied from best NO bid: 1 - no_bid
            yes_ask = (1.0 - no_bid) if no_bid > 0 else 0.0

            if yes_bid > 0 and yes_ask > 0:
                yes_mid = (yes_bid + yes_ask) / 2
            elif yes_bid > 0:
                yes_mid = yes_bid
            elif yes_ask > 0:
                yes_mid = yes_ask
            else:
                yes_mid = 0.5  # empty book — neutral fallback

            return yes_mid if side == "yes" else (1.0 - yes_mid)
        except Exception:
            log.exception("exec.orderbook_parse_error")
            return 0.5

    async def _exit(
        self, ticker: str, exit_price: float, *, reason: str,
        orderbook: dict | None = None, exit_edge: float | None = None,
    ) -> bool:
        """Close a position. Returns True if the position is closed
        (journal written), False if it remains open — live mode only,
        when the sell couldn't fill. Callers in the watcher must keep
        watching on False instead of abandoning the position.

        Paper mode: instant close at `exit_price` (the optimistic mid),
        unchanged behavior. Live mode: a real sell order must fill
        first, and the journal records the ACTUAL average fill price,
        not the mid that triggered the exit.
        """
        pos = self.open.get(ticker)
        if not pos:
            return True
        if self.mode != "paper":
            # Throttle retry storms: a failed exit leaves the position
            # open and the watcher will call us again next tick.
            last = getattr(pos, "last_exit_attempt", None)
            now = datetime.now(timezone.utc)
            if last and (now - last).total_seconds() < \
                    self.cfg.live_exit_retry_cooldown_s:
                return False
            pos.last_exit_attempt = now  # type: ignore[attr-defined]
            avg = await self._sell_live(pos, ticker)
            if avg is None:
                log.warning("exec.live.exit_unfilled", ticker=ticker,
                            reason=reason,
                            sold=getattr(pos, "sold_contracts", 0),
                            total=pos.contracts)
                return False
            exit_price = avg  # the real average fill, in dollars
        self.open.pop(ticker, None)
        self._finalize_close(ticker, pos, exit_price, reason=reason,
                             orderbook=orderbook, exit_edge=exit_edge)
        return True

    async def _sell_live(self, pos, ticker: str) -> float | None:
        """Sell the position's remaining contracts with limit orders at
        the live best bid, repricing on a cadence. Accumulates partial
        fills across calls on the position object. Returns the average
        fill price in dollars across ALL sold contracts once the
        position is fully closed, else None (remainder stays open).
        """
        side = pos.signal.side
        sold = getattr(pos, "sold_contracts", 0)
        sold_value = getattr(pos, "sold_value", 0.0)  # dollars
        remaining = pos.contracts - sold
        for _attempt in range(self.cfg.live_exit_max_reprices + 1):
            if remaining <= 0:
                break
            try:
                ob = await self.client.get_orderbook(ticker)
            except Exception as e:  # noqa: BLE001
                log.warning("exec.live.exit_ob_error", ticker=ticker,
                            err=str(e)[:200])
                break
            bid = self._best_bid(ob, side)
            if bid is None or bid <= 0:
                # Empty bid stack on our side — the ghost-liquidity
                # scenario, observed for real this time. Nothing to hit.
                log.warning("exec.live.exit_no_bids", ticker=ticker,
                            side=side)
                break
            bid_cents = max(1, min(99, int(round(bid * 100))))
            client_order_id = f"X{ticker[:24]}-{uuid.uuid4().hex[:8]}"
            try:
                resp = await self.client.place_order(
                    ticker=ticker, side=side, action="sell",
                    count=remaining, price_cents=bid_cents,
                    client_order_id=client_order_id,
                )
            except Exception as e:  # noqa: BLE001
                log.exception("exec.live.exit_order_error",
                              ticker=ticker, err=str(e))
                break
            oid = (resp.get("order") or {}).get("order_id")
            if not oid:
                break
            # Give the order live_exit_reprice_s to fill, polling.
            window_end = (datetime.now(timezone.utc)
                          + timedelta(seconds=self.cfg.live_exit_reprice_s))
            while datetime.now(timezone.utc) < window_end:
                await asyncio.sleep(self.cfg.live_fill_poll_s)
                try:
                    o = (await self.client.get_order(oid)).get("order") or {}
                    if o.get("status") in ("executed", "canceled"):
                        break
                except Exception:  # noqa: BLE001
                    pass
            try:
                await self.client.cancel_order(oid)
            except Exception:  # noqa: BLE001
                pass
            n, avg = await self._fills_for_orders(side, [oid])
            if n > 0 and avg is not None:
                sold += n
                sold_value += n * avg
                remaining = pos.contracts - sold
                log.info("exec.live.exit_partial_fill", ticker=ticker,
                         n=n, avg=round(avg, 4), remaining=remaining)
        pos.sold_contracts = sold  # type: ignore[attr-defined]
        pos.sold_value = sold_value  # type: ignore[attr-defined]
        if sold >= pos.contracts and sold > 0:
            return sold_value / sold
        return None

    @staticmethod
    def _best_bid(ob: dict, side: str) -> float | None:
        """Best bid in dollars on `side` from a Kalshi orderbook payload."""
        try:
            book = ob.get("orderbook_fp") or ob.get("orderbook") or {}
            if side == "yes":
                stack = book.get("yes_dollars") or book.get("yes") or []
            else:
                stack = book.get("no_dollars") or book.get("no") or []
            best = max((float(e[0]) for e in stack), default=None)
            if best is None:
                return None
            # Legacy integer-cents payloads: values > 1 are cents.
            return best / 100.0 if best > 1.0 else best
        except Exception:  # noqa: BLE001
            return None

    def _finalize_close(
        self, ticker: str, pos, exit_price: float, *, reason: str,
        orderbook: dict | None = None, exit_edge: float | None = None,
    ) -> None:
        # P&L per contract is the price move in dollars. `exit_price` is
        # already side-adjusted (NO mid for NO bets, YES mid for YES bets),
        # and `fill_price` is in the same units (the side we bought).
        # So profit = exit - fill regardless of side. Contracts settle at
        # $1, so dollar P&L = (price_move) * contracts. NO extra * 100.
        pnl_per_contract = exit_price - pos.fill_price
        pnl_usd = pnl_per_contract * pos.contracts
        # Kalshi fee = 0.07 * p * (1-p) per contract, ceiled to next cent,
        # charged on BOTH entry and exit. The old 0.07 * contracts flat was
        # 4-7x too aggressive and made every paper trade look like a loss.
        entry_fee = fee_per_contract_dollars(pos.fill_price) * pos.contracts
        exit_fee = fee_per_contract_dollars(exit_price) * pos.contracts
        fees_usd = entry_fee + exit_fee
        pnl_usd -= fees_usd
        # Sanity clamp: a long position can never lose more than its size.
        # Catches any residual unit-conversion bug before it trips the
        # daily-loss kill switch on imaginary losses.
        if pnl_usd < -pos.signal.size_usd:
            log.warning("exec.pnl_clamped_to_size",
                        ticker=ticker, raw_pnl=pnl_usd, size=pos.signal.size_usd)
            pnl_usd = -pos.signal.size_usd

        # TP fillability: compute the realistic avg fill from the live
        # orderbook (book-sweep), not the optimistic mid we credited.
        # Paper mode pretends the full size fills at `exit_price`; in
        # real life selling 200 contracts walks the bid stack and lands
        # at a worse average. Recording both lets the dashboard show how
        # much paper TP profit would survive real fills.
        realistic_exit_price = None
        exit_book_size = None
        if orderbook is not None:
            realistic_exit_price, exit_book_size = (
                self._realistic_fill_from_orderbook(
                    orderbook, pos.signal.side, pos.contracts,
                )
            )

        self.risk.record_close(size_usd=pos.signal.size_usd, realized_pnl_usd=pnl_usd)
        self.journal.log_close(
            ticker=ticker,
            exit_price=exit_price,
            pnl_usd=pnl_usd,
            fees_usd=fees_usd,
            reason=reason,
            realistic_exit_price=realistic_exit_price,
            exit_book_size=exit_book_size,
            exit_edge=exit_edge,
        )
        # Slack ping on close. Includes hold duration via pos.opened_at.
        try:
            from .slack_notifier import notify_close
            notify_close(
                ticker=ticker, side=pos.signal.side,
                fill_price=pos.fill_price, exit_price=exit_price,
                pnl_usd=pnl_usd, fees_usd=fees_usd,
                reason=pos.signal.reason, exit_reason=reason,
                opened_at=pos.opened_at,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("slack.notify_close_failed", err=str(e)[:200])
        # Release event lock and start cooldown so we don't immediately
        # re-enter the same trade after a take_profit / stop_loss.
        ev = getattr(pos, "event_ticker", None)
        if ev:
            self.open_events.discard(ev)
            self.cooldown_until[ev] = (
                datetime.now(timezone.utc) + timedelta(minutes=self._COOLDOWN_MINUTES)
            )
        # NOTE: kwarg renamed from `event=` — structlog >=25 reserves
        # `event` for the log message and raises TypeError, which would
        # crash every exit after an unpinned structlog upgrade.
        log.info("exec.exit", ticker=ticker, reason=reason, pnl=round(pnl_usd, 2),
                 event_ticker=ev)
