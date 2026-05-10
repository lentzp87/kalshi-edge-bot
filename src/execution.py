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
        # Event-level dedup: BOS-yes and TB-no on the same Kalshi event
        # are the same bet (both win if Boston wins). Locking by
        # event_ticker prevents double-exposure across mirror tickers.
        self.open_events: set[str] = set()
        # Post-close cooldown: after an exit, lock the event for 60 min
        # so we don't immediately re-enter when the model still sees an
        # edge. Without this we churn through fees firing the same trade
        # every loop after each take_profit / stop_loss.
        self.cooldown_until: dict[str, datetime] = {}

    _COOLDOWN_MINUTES = 60

    @staticmethod
    def _event_ticker(market: Market) -> str:
        """Kalshi events group mirror tickers (TEAM-A-yes ≡ TEAM-B-no).
        Falls back to the market ticker if event_ticker isn't present.
        """
        return (market.raw.get("event_ticker") or market.ticker or "").strip()

    async def submit(self, signal: TradeSignal, market: Market) -> None:
        # 1) One position per ticker (existing dedup)
        if signal.ticker in self.open:
            log.debug("exec.skip.already_open", ticker=signal.ticker)
            return

        # 2) One position per EVENT — blocks the mirror-side ticker
        ev = self._event_ticker(market)
        if ev and ev in self.open_events:
            log.info("exec.skip.event_already_open",
                     ticker=signal.ticker, event=ev)
            return

        # 3) Cooldown after a recent close on this event
        cd_until = self.cooldown_until.get(ev) if ev else None
        if cd_until and datetime.now(timezone.utc) < cd_until:
            mins_left = (cd_until - datetime.now(timezone.utc)).total_seconds() / 60
            log.info("exec.skip.event_cooldown",
                     ticker=signal.ticker, event=ev,
                     mins_left=round(mins_left, 1))
            return

        if not self.risk.approve(signal):
            return

        # Whale boost: if an aligned whale signal exists for this market,
        # scale up to whale_max_position_size_usd (default same as normal
        # cap = no-op). The signal is logged and stamped onto the
        # journal reason so we can analyze whale-aligned trades on the
        # dashboard.
        risk_cfg = file_config().risk
        whale_signal = None
        if self.whale_tracker is not None:
            whale_signal = self.whale_tracker.has_aligned_signal(
                signal.ticker, signal.side
            )
        if whale_signal and risk_cfg.whale_max_position_size_usd > signal.size_usd:
            old_size = signal.size_usd
            signal.size_usd = min(
                risk_cfg.whale_max_position_size_usd,
                # Cap at 5x the original Kelly so we don't over-bet badly
                # calibrated signals just because a whale showed up.
                signal.size_usd * 5,
            )
            signal.reason = (
                f"{signal.reason} | WHALE_ALIGNED dir={whale_signal.direction} "
                f"conf={whale_signal.confidence:.2f} ({whale_signal.reason}) "
                f"size_boost ${old_size:.0f}->${signal.size_usd:.0f}"
            )
            log.info("exec.whale_boost",
                     ticker=signal.ticker, side=signal.side,
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
        chunks = max(1, self.cfg.scale_in_chunks)
        per_chunk = max(1, contracts // chunks)
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
            except Exception as e:
                log.exception("exec.live.error", ticker=signal.ticker, err=str(e))

        # In live mode the fill watcher should reconcile actual fills before recording.
        # Stub: record optimistically so the journal sees the intent.
        self._record_open(signal, market, signal.price_cents / 100, contracts)

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
        self.journal.log_open(signal=signal, market=market, fill_price=fill_price, contracts=contracts)
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
        """Wait until ~5 min before tipoff, then record Kalshi mid as CLV.

        Runs independently of the position lifecycle — even if we exit
        via TP/SL hours before tip, we still capture the closing-line
        proxy for that ticker. The journal row is matched by
        (ticker, opened_ts) so concurrent positions on the same ticker
        across the session don't collide.
        """
        target = tip_utc - timedelta(minutes=5)
        wait_seconds = (target - datetime.now(timezone.utc)).total_seconds()
        # Cap at 36 hours so a far-future tip doesn't pin the task forever
        wait_seconds = max(0.0, min(wait_seconds, 36 * 3600))
        try:
            await asyncio.sleep(wait_seconds)
        except asyncio.CancelledError:
            return
        try:
            ob = await self.client.get_orderbook(ticker)
            mid = self._mid_from_orderbook(ob, side)
            if 0 < mid < 1:
                self.journal.update_clv_price(
                    ticker=ticker, opened_ts=opened_ts, clv_price=mid,
                )
                log.info("clv.recorded", ticker=ticker, side=side,
                         clv_price=round(mid, 4))
        except Exception:
            log.exception("clv.sample_failed", ticker=ticker)

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
                # `mid` is already side-adjusted (NO mid for NO bets). So
                # profit is mid - fill regardless of side. The previous
                # branching here flipped the sign on every NO position,
                # making take_profit trigger on losses and vice versa.
                pnl_pct = (mid - pos.fill_price) / pos.fill_price
                if pnl_pct >= self.cfg.take_profit_pct:
                    await self._exit(ticker, mid, reason="take_profit")
                    return
                if pnl_pct <= -self.cfg.stop_loss_pct:
                    await self._exit(ticker, mid, reason="stop_loss")
                    return
                if datetime.now(timezone.utc) >= deadline:
                    await self._exit(ticker, mid, reason="time_exit")
                    return
            except Exception as e:
                log.exception("exec.watch.error", ticker=ticker, err=str(e))

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

            yes_bid = float(yes_book[-1][0]) if yes_book else 0.0
            no_bid = float(no_book[-1][0]) if no_book else 0.0
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

    async def _exit(self, ticker: str, exit_price: float, *, reason: str) -> None:
        pos = self.open.pop(ticker, None)
        if not pos:
            return
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

        self.risk.record_close(size_usd=pos.signal.size_usd, realized_pnl_usd=pnl_usd)
        self.journal.log_close(
            ticker=ticker,
            exit_price=exit_price,
            pnl_usd=pnl_usd,
            fees_usd=fees_usd,
            reason=reason,
        )
        # Release event lock and start cooldown so we don't immediately
        # re-enter the same trade after a take_profit / stop_loss.
        ev = getattr(pos, "event_ticker", None)
        if ev:
            self.open_events.discard(ev)
            self.cooldown_until[ev] = (
                datetime.now(timezone.utc) + timedelta(minutes=self._COOLDOWN_MINUTES)
            )
        log.info("exec.exit", ticker=ticker, reason=reason, pnl=round(pnl_usd, 2),
                 event=ev)
